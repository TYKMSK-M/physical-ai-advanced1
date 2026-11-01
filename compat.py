"""Colab 依存 (drive mount / files.download / display) のローカルフォールバック.

Colab 環境で実行した場合は本来の動作にフォワードし、
ローカル/コンテナでは no-op / print に置き換える。
"""

from __future__ import annotations

from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_drive() -> None:
    """Section 1 の drive.mount 相当。ローカルでは CACHE_DIR を使うので no-op。"""
    if is_colab():
        from google.colab import drive

        drive.mount("/content/drive")
    else:
        from . import config

        print(f"(local) Drive マウント不要 — キャッシュは {config.CACHE_DIR} を使用")


def download(path: Path | str) -> None:
    """files.download 相当。ローカルではパスを表示するだけ。"""
    path = Path(path)
    if is_colab():
        from google.colab import files

        files.download(str(path))
    else:
        print(f"(local) 成果物: {path}")


def display_df(df) -> None:
    """IPython.display(df) 相当。ターミナルでは to_string で出力。"""
    print(df.round(1).to_string(index=False))


def show_video(path: Path | str) -> None:
    """IPython.display(Video(...)) 相当。ターミナルではパスを表示。"""
    print(f"(local) 動画: {path}")
