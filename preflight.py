"""Section 9 事前チェック (Section 9.0 相当).

Section 8 で自分なりの工夫をした結果、モデルディレクトリの形式が
壊れていないか事前に確認する。ここで失敗するなら Section 8 を見直す。
"""

from __future__ import annotations

from pathlib import Path

from . import config

_REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
]
# .safetensors の stats は名前が可変なので glob で検証
_REQUIRED_GLOBS = [
    "policy_preprocessor*.safetensors",
    "policy_postprocessor*.safetensors",
]


def check_policy_dir(policy_dir: Path, label: str) -> None:
    print(f"\n[{label}] {policy_dir}")
    if not policy_dir.is_dir():
        raise FileNotFoundError(
            f"{label}: ディレクトリが存在しません。"
            "train / merge を実行しましたか?"
        )
    missing = []
    for name in _REQUIRED_FILES:
        path = policy_dir / name
        ok = path.is_file()
        print(
            f"  {'✅' if ok else '❌'}  {name}"
            + (f"  ({path.stat().st_size / 1e6:.1f} MB)" if ok else "")
        )
        if not ok:
            missing.append(name)
    for pattern in _REQUIRED_GLOBS:
        matches = list(policy_dir.glob(pattern))
        ok = len(matches) > 0
        print(f"  {'✅' if ok else '❌'}  {pattern}  ({len(matches)} match)")
        if not ok:
            missing.append(pattern)
    if missing:
        raise FileNotFoundError(
            f"{label} に必要ファイルが不足: {missing}. "
            "merge (8.3) / baseline (8.4) を再実行してください。"
        )


def preflight() -> None:
    check_policy_dir(config.BASELINE_MODEL_DIR, "BASELINE_MODEL_DIR (対照)")
    check_policy_dir(config.MERGED_MODEL_DIR, "MERGED_MODEL_DIR (工夫版)")
    print("\n✅ 契約 OK — アドバンスド評価を実行できます")
