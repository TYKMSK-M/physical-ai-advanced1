"""C1: VLM 特徴の事前計算キャッシュ (Basic 学習の高速化).

VLM (SmolVLM2-500M) は完全凍結なので、学習データ全体の vlm_hidden は
学習中に不変である。にもかかわらず notebook の学習ループは毎 step
500M forward を実行していた (学習時間の大半)。

本モジュールはそれを 1 回だけ precompute して disk に保存し、
学習時はそれを参照する (学習可能な projector / state_proj / flow_head
だけを毎回実行する)。

コストモデル (2026-10-28 実測: L40S, 3.1 sample/s):
- precompute コスト = N(データ件数) x t_vlm
- cache 無し学習コスト = S(学習 step 数) x t_vlm
=> **cache が得をするのは S > N、つまり学習が 1 エポックを超えるときだけ**。
    既定の 12000 step x batch1 > 6159 samples では cache が有利なため、
    should_use_cache() が自動で有効化する。

設計:
- 1 サンプル = 可変長 (L, 960) fp16 tensor。DataLoader の collate を
  壊さないよう 1 ファイル 1 サンプルで保存し、ランダムアクセスに備える
  (50 ep x ~300 frame = ~1.5 万ファイル、1 ファイル ~0.5 MB)。
- cache key は (dataset revision, episode 一覧, VLM repo) のハッシュで
  無効化。manifest.json の件数一致でも妥当性チェック。
- batch_size=1 前提 (notebook の既定と同じ。可変長 hidden のため)。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config


VLM_CACHE_ESTIMATED_BYTES_PER_SAMPLE = 2_500_000


def ensure_vlm_cache_capacity(sample_count: int) -> dict[str, int | float]:
    """VLM cacheの保守的な必要量と空き容量を検査する."""
    config.PY_VLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.PY_VLM_CACHE_DIR)
    estimated_bytes = sample_count * VLM_CACHE_ESTIMATED_BYTES_PER_SAMPLE
    required_bytes = int(estimated_bytes * 1.2)
    report: dict[str, int | float] = {
        "sample_count": sample_count,
        "estimated_bytes_per_sample": VLM_CACHE_ESTIMATED_BYTES_PER_SAMPLE,
        "estimated_bytes": estimated_bytes,
        "required_bytes_with_margin": required_bytes,
        "filesystem_total_bytes": usage.total,
        "filesystem_free_bytes": usage.free,
        "margin_ratio": 1.2,
    }
    if usage.free < required_bytes:
        raise RuntimeError(
            "VLM cacheの空き容量が不足しています: "
            f"required={required_bytes / 2**30:.1f} GiB, "
            f"free={usage.free / 2**30:.1f} GiB"
        )
    print(
        "[C1] cache capacity: "
        f"estimated={estimated_bytes / 2**30:.1f} GiB, "
        f"required={required_bytes / 2**30:.1f} GiB, "
        f"free={usage.free / 2**30:.1f} GiB",
        flush=True,
    )
    return report


def _cache_key(episode_indices: list[int]) -> str:
    """cache の妥当性キー (revision / episode / VLM / solver 設定)."""
    raw = json.dumps(
        {
            "dataset": config.DATASET_REPO,
            "revision": config.DATASET_REVISION,
            "episodes": sorted(episode_indices),
            "vlm": config.VLM_REPO,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_dir_for(episode_indices: list[int]) -> Path:
    return config.PY_VLM_CACHE_DIR / _cache_key(episode_indices)


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def is_cache_valid(cache_dir: Path, expected_count: int) -> bool:
    manifest = _manifest_path(cache_dir)
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        int(data.get("count", -1)) == expected_count
        and (cache_dir / f"sample_0.pt").is_file()
    )


def should_use_cache(
    dataset_count: int,
    train_steps: int,
    batch_size: int,
    cache_dir: Path | None = None,
) -> tuple[bool, str]:
    """cache を使うべきかを (判定, 理由) で返す.

    precompute はデータ全件 (N 件) の VLM forward、cache 無し学習は
    step 数 (S) ぶんの VLM forward なので、cache が得をするのは
    S > N のとき。加えて**有効な cache が既に存在するなら S >= 1 でも得**
    (precompute 済み = 償却済みだから) なので、その場合は常に True。
    """
    vlm_forwards_no_cache = train_steps * batch_size
    if cache_dir is not None and is_cache_valid(cache_dir, dataset_count):
        return True, (
            f"有効な cache あり ({dataset_count} samples, 償却済み)。"
            f"学習 {vlm_forwards_no_cache} 回の VLM forward をスキップ"
        )
    if vlm_forwards_no_cache > dataset_count:
        return True, (
            f"学習 {vlm_forwards_no_cache} 回 > データ {dataset_count} 件 "
            f"(>{vlm_forwards_no_cache / dataset_count:.1f} エポック) なので "
            "precompute が償却される"
        )
    return False, (
        f"学習 {vlm_forwards_no_cache} 回 <= データ {dataset_count} 件 "
        f"({vlm_forwards_no_cache / dataset_count:.1%} エポック)。"
        "precompute の方が遅いため cache をスキップ"
    )


@torch.no_grad()
def precompute_vlm_features(
    train_dataset,
    vlm_encoder,
    episode_indices: list[int],
    *,
    force: bool = False,
) -> Path:
    """学習データ全体の vlm_hidden を計算して cache_dir に保存する.

    notebook の学習時と同一の前処理 (batch=1, 同一 processor 経路) を
    通すため、数値は学習中に online で計算した場合と一致する。
    """
    from tqdm.auto import tqdm

    cache_dir = cache_dir_for(episode_indices)
    total = len(train_dataset)

    if not force and is_cache_valid(cache_dir, total):
        print(f"VLM cache 再利用: {cache_dir} ({total} samples)")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    reusable_samples = 0
    if not force:
        reusable_samples = sum(
            (cache_dir / f"sample_{index}.pt").is_file()
            for index in range(total)
        )
    print(f"VLM 特徴を precompute 中: {total} samples -> {cache_dir}", flush=True)
    if reusable_samples:
        print(
            f"VLM cache partial resume: {reusable_samples}/{total} samples reuse",
            flush=True,
        )

    dtype = vlm_encoder._dtype
    for i in tqdm(range(total), desc="vlm precompute", unit="sample"):
        sample_path = cache_dir / f"sample_{i}.pt"
        if not force and sample_path.is_file():
            continue
        sample = train_dataset[i]

        img = sample["observation.images.front"]
        if img.dtype.is_floating_point:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        task = sample["task"]
        if isinstance(task, (list, tuple)):
            task = str(task[0]) if len(task) > 0 else str(task)

        # 学習時 (RealVLMEncoder, B=1) と同一経路
        hidden = vlm_encoder([img_np], [str(task)])  # (1, L, 960)
        tensor = hidden[0].detach().to(dtype).cpu()

        torch.save(tensor, sample_path)

    _manifest_path(cache_dir).write_text(
        json.dumps(
            {
                "count": total,
                "key": _cache_key(episode_indices),
                "hidden_dim": int(vlm_encoder.hidden_dim),
                "dtype": str(dtype),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"VLM cache 完了: {cache_dir}")
    return cache_dir


class CachedVlmDataset(Dataset):
    """base dataset のサンプルに 'vlm_hidden' (凍結 VLM 生出力) を追加する.

    学習ループ側は 'vlm_hidden' を SmolVLA.compute_loss の
    vlm_hidden_raw として渡す (凍結 VLM の forward は走らない)。
    可変長 (L, 960) なので batch_size=1 専用。
    """

    def __init__(self, base_dataset, cache_dir: Path):
        self.base = base_dataset
        self.cache_dir = Path(cache_dir)
        self._loaded_index: int | None = None
        self._loaded_tensor: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        # 同一 index の連続参照 (shuffle=False 等) は再読込みをスキップ
        if self._loaded_index != index:
            self._loaded_tensor = torch.load(
                self.cache_dir / f"sample_{index}.pt", map_location="cpu"
            )
            self._loaded_index = index
        sample["vlm_hidden"] = self._loaded_tensor
        return sample
