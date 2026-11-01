"""LeRobot v0.6.0 の editable install と互換性パッチ (Section 3 相当).

手順 (notebook と同じ):
  1. 既存 install の除去 (torchao も同時に。SmolVLA 非対応のため)
  2. 固定 tag の LeRobot を shallow clone (進捗を stream)
  3. bf16 非対応 GPU なら SmolVLM を fp16 に書換
  4. lerobot_train.py の冗長ログと tqdm を軽微パッチで抑制
  5. editable install (training + smolvla + peft の extras)
  6. torchao を再度アンインストール (依存で復活するケースの保険)
  7. sys.modules と sys.path を掃除して clone した src を優先させる
"""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

from . import config
from .proc import run_quiet, run_streaming_rolling


def _strip_local_version(v: str) -> str:
    return v.split("+", 1)[0]


def _align_cuda_builds(*packages: str) -> None:
    """torchvision / torchcodec を torch と同じ CUDA ビルドに揃える.

    lerobot の依存解決は PyPI 既定版 (torch とは別の CUDA major でビルド) を
    優先して入れるため、放置すると import 時に
    "PyTorch and torchvision were compiled with different CUDA major versions"
    で落ちる。--no-deps で torch を巻き込まずに張り直す。
    """
    import torch

    expected = (
        f"cu{torch.version.cuda.replace('.', '')}"
        if torch.version.cuda
        else "cpu"
    )

    mismatched = []
    for name in packages:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue  # 未導入のパッケージは対象外
        local = version.split("+", 1)[1] if "+" in version else ""
        if local != expected:
            mismatched.append(name)
    if not mismatched:
        return

    print(
        f"CUDA build mismatch: {mismatched} を +{expected} に揃えます",
        flush=True,
    )
    run_streaming_rolling(
        [
            sys.executable, "-m", "pip", "install",
            "--progress-bar", "off",
            "--no-deps", "--force-reinstall",
            "--index-url", f"https://download.pytorch.org/whl/{expected}",
            *mismatched,
        ],
        prefix="[pip] ",
        tail=10,
    )


def install_lerobot() -> None:
    import torch

    # --- 0. 安全ガード: torch のバージョン整合性チェック ------------
    # 前回のセッションで torch を import 済みで、その後 pip install が
    # torch を更新すると、C 拡張(.so)は旧版・Python 側は新版という
    # 不整合状態になる。事前にディスク上と in-memory の版を照合する。
    torch_version_in_memory = _strip_local_version(torch.__version__)
    try:
        torch_dist_version = _strip_local_version(
            importlib.metadata.version("torch")
        )
    except importlib.metadata.PackageNotFoundError:
        torch_dist_version = None

    if torch_dist_version and torch_dist_version != torch_version_in_memory:
        raise RuntimeError(
            f"torch mismatch: in-memory={torch_version_in_memory} "
            f"on-disk={torch_dist_version}. "
            "プロセスを再起動してこのコマンドを最初から実行し直してください。"
        )

    # --- 1. 既存 install を除去 (再実行安全性のため) ----------------
    run_quiet(
        [sys.executable, "-m", "pip", "uninstall", "-y", "lerobot", "torchao"],
        check=False,
    )
    shutil.rmtree(config.LEROBOT_DIR, ignore_errors=True)

    # --- 2. LeRobot を shallow clone (進捗を stream 表示) -----------
    print("Cloning LeRobot...", flush=True)
    run_streaming_rolling(
        [
            "git", "clone", "--progress", "--depth", "1",
            "--branch", config.LEROBOT_TAG,
            "https://github.com/huggingface/lerobot.git",
            str(config.LEROBOT_DIR),
        ],
        prefix="[git] ",
        tail=10,
    )

    # --- 3. dtype 互換性パッチ -------------------------------------
    # T4/V100 系など bf16 未対応 GPU では bf16 で load すると壊れるので fp16 に置換
    smolvlm_source = (
        config.LEROBOT_SRC
        / "lerobot"
        / "policies"
        / "smolvla"
        / "smolvlm_with_expert.py"
    )
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        source = smolvlm_source.read_text(encoding="utf-8")
        source = source.replace('torch_dtype="bfloat16",', 'torch_dtype="float16",', 1)
        smolvlm_source.write_text(source, encoding="utf-8")

    # --- 4. lerobot_train.py の軽微パッチ ---------------------------
    # 学習 subprocess のログを絞る:
    #   - 設定 pformat の INFO を DEBUG に落とす
    #   - inside_slurm() 判定に依らず tqdm を強制 disable
    train_script = config.LEROBOT_SRC / "lerobot" / "scripts" / "lerobot_train.py"
    source = train_script.read_text(encoding="utf-8")
    source = source.replace(
        "logging.info(pformat(cfg.to_dict()))",
        "logging.debug(pformat(cfg.to_dict()))",
        1,
    )
    source = source.replace("disable=inside_slurm(),", "disable=True,", 1)
    train_script.write_text(source, encoding="utf-8")

    # --- 5. editable install (extras 込み) --------------------------
    # --upgrade を付けない: torch を含む依存が既に満たされていれば
    # 何も触らせない (在庫の C 拡張との不整合を避けるため)。
    print("Installing LeRobot (this may take a while)...", flush=True)
    run_streaming_rolling(
        [
            sys.executable, "-m", "pip", "install",
            "--progress-bar", "off",
            "-e", f"{config.LEROBOT_DIR}[training,smolvla,peft]",
        ],
        prefix="[pip] ",
        tail=10,
    )

    # --- 5.5. インストール後 torch が動いていないことを再確認 --------
    torch_dist_version_after = _strip_local_version(
        importlib.metadata.version("torch")
    )
    if torch_dist_version_after != torch_version_in_memory:
        raise RuntimeError(
            f"pip install が torch を {torch_version_in_memory} -> "
            f"{torch_dist_version_after} に変更しました。"
            "プロセスを再起動してこのコマンドを最初から実行し直してください。"
        )

    # --- 5.6. torchvision / torchcodec の CUDA ビルド整合 -------------
    # lerobot の依存解決が入れる PyPI 版は torch と CUDA major が食い、
    # import 時に RuntimeError になるため torch と同じ build に張り直す
    _align_cuda_builds("torchvision", "torchcodec")

    # --- 6. torchao 再アンインストール ------------------------------
    # 上の install 中に依存として復活することがあるため保険としてもう一度消す
    run_quiet(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=False,
    )

    # --- 7. sys.modules と sys.path の掃除 ---------------------------
    # 前回実行時に読まれた古い lerobot/torchao モジュールを追い出す
    for module_name in list(sys.modules):
        if (
            module_name == "lerobot"
            or module_name.startswith("lerobot.")
            or module_name == "torchao"
            or module_name.startswith("torchao.")
        ):
            del sys.modules[module_name]

    config.ensure_lerobot_on_path()
    importlib.invalidate_caches()

    # torchao が完全に消えたかチェック。残っていると SmolVLA が起動時に落ちる
    try:
        importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("torchaoの削除に失敗しました。")

    # 実際に import して、read 先が今 clone した src 配下であることを保証
    print("Importing lerobot (torch init may take up to a minute)...", flush=True)
    import lerobot  # noqa: F401
    import peft  # noqa: F401

    if config.LEROBOT_SRC.resolve() not in Path(lerobot.__file__).resolve().parents:
        raise RuntimeError("LeRobotの読込先が正しくありません。")

    print("LeRobot ready.")
