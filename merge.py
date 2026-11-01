"""LoRA マージと対照ベースライン作成 (Section 8.3-8.4 相当).

- merge_lora   : 学習後 adapter を PeftModel でロード -> merge_and_unload で
                 LoRA 差分をベース weight に足し込み、通常の LeRobot モデル形式で保存
- build_baseline: 公開の初期重みを、マージ済みモデルと同じ processor 設定・
                 入出力 features に揃えた「対照モデル」として保存し直す
                 (評価時の前処理/後処理を完全に一致させ比較を公正にする)
"""

from __future__ import annotations

import contextlib
import gc
import io
import shutil
from pathlib import Path

from . import config


def _final_checkpoint_dir() -> Path:
    # lerobot-train が保存した最終 checkpoint (adapter が入っている)
    return (
        config.OUTPUT_DIR
        / "checkpoints"
        / f"{config.STEPS:06d}"
        / "pretrained_model"
    )


def _copy_processor_stats(src_dir: Path, dst_dir: Path) -> None:
    # 学習中に更新された前処理/後処理 stats も同じディレクトリへコピーしないと
    # 評価時に入出力スケーリングが崩れる
    for pattern in [
        "policy_preprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor*.safetensors",
    ]:
        for source_path in src_dir.glob(pattern):
            shutil.copy2(source_path, dst_dir / source_path.name)


def _fix_saved_config(policy) -> None:
    # 保存 config は「単体モデル・再 push 無し・VLM は再ロードしない」に固定
    policy.config.use_peft = False
    policy.config.pretrained_path = None
    policy.config.push_to_hub = False
    policy.config.repo_id = None
    policy.config.device = None
    policy.config.load_vlm_weights = False
    policy.config.vlm_model_name = config.VLM_REPO


def merge_lora(base_model_local: Path) -> Path:
    checkpoint_dir = _final_checkpoint_dir()
    if not (checkpoint_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Final adapter not found.")

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # GPU メモリを空けてから CPU 上でマージ (VRAM を節約)
    gc.collect()
    torch.cuda.empty_cache()

    # --- ベース + LoRA のロード設定 -------------------------------
    merge_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    merge_config.device = "cpu"  # マージは CPU で実行
    merge_config.pretrained_path = base_model_local  # ベース weight の場所を明示
    merge_config.use_peft = False  # マージ後は peft 経路を使わない

    # ロード時の noisy な info ログを黙らせる
    quiet_output = io.StringIO()
    with (
        contextlib.redirect_stdout(quiet_output),
        contextlib.redirect_stderr(quiet_output),
    ):
        # 1) ベース SmolVLA を CPU にロード
        base_policy = SmolVLAPolicy.from_pretrained(
            base_model_local, config=merge_config, strict=False
        )
        # 2) 学習済み adapter を被せる
        peft_policy = PeftModel.from_pretrained(
            base_policy, checkpoint_dir, is_trainable=False, torch_device="cpu"
        )
        # 3) LoRA を base に足し込んで単体モデル化
        merged_policy = peft_policy.merge_and_unload(safe_merge=True)

    # --- マージ済みモデルを保存 -----------------------------------
    shutil.rmtree(config.MERGED_MODEL_DIR, ignore_errors=True)
    config.MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _fix_saved_config(merged_policy)
    merged_policy.save_pretrained(config.MERGED_MODEL_DIR)
    _copy_processor_stats(checkpoint_dir, config.MERGED_MODEL_DIR)

    # --- マージ結果に LoRA パラメータが残っていないか検証 ---------
    merged_weights_path = config.MERGED_MODEL_DIR / "model.safetensors"
    with safe_open(merged_weights_path, framework="pt", device="cpu") as weights:
        if any("lora_" in key.lower() for key in weights.keys()):
            raise RuntimeError("LoRA parameters remain after merge.")

    # CPU 上に膨らんだモデルを片付ける (次の baseline ロードが重い)
    del peft_policy
    del base_policy
    del merged_policy
    gc.collect()
    torch.cuda.empty_cache()

    print("Merged model ready.")
    return config.MERGED_MODEL_DIR


def build_baseline(base_model_local: Path) -> Path:
    # マージ済みモデルの config を土台にする (入出力 schema が同一になる)
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    baseline_config = PreTrainedConfig.from_pretrained(config.MERGED_MODEL_DIR)
    baseline_config.device = "cpu"
    baseline_config.pretrained_path = base_model_local
    baseline_config.use_peft = False
    baseline_config.load_vlm_weights = False

    quiet_output = io.StringIO()
    with (
        contextlib.redirect_stdout(quiet_output),
        contextlib.redirect_stderr(quiet_output),
    ):
        # 公開の初期 weight を読みつつ、config だけマージ済み側と揃える
        baseline_policy = SmolVLAPolicy.from_pretrained(
            base_model_local, config=baseline_config, strict=False
        )

    shutil.rmtree(config.BASELINE_MODEL_DIR, ignore_errors=True)
    config.BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 保存 config を単体モデル運用向けに固定 (マージ済みモデルと同じ扱い)
    _fix_saved_config(baseline_policy)
    baseline_policy.save_pretrained(config.BASELINE_MODEL_DIR)

    # processor stats はマージ済み側からコピー (両者を完全一致させる)
    _copy_processor_stats(config.MERGED_MODEL_DIR, config.BASELINE_MODEL_DIR)

    del baseline_policy
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    print("Baseline ready.")
    return config.BASELINE_MODEL_DIR
