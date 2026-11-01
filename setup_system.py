"""システムパッケージ導入 (Section 2 相当).

Colab 版の `sudo apt-get` を、root / sudo 有無 / 非 Debian 系を判定して
実行できるようにした。コンテナイメージで事前導入する場合はスキップ可。
"""

from __future__ import annotations

import os
import shutil

from .proc import run_quiet

# 依存パッケージ内訳 (notebook と同じ):
#   ffmpeg                : LeRobot の動画デコード
#   git / unzip           : clone / 展開に使用
#   libgl1 / libglib2.0-0 : OpenGL 系 (MuJoCo/robosuite の描画)
#   libsm6 / libxext6     : X11 系ランタイム (描画ライブラリの依存)
#   libexpat1             : XML パーサ (MuJoCo の XML モデル読込)
#   libfontconfig1-dev    : matplotlib 用フォント設定
#   libmagickwand-dev     : Wand (ImageMagick binding) が LIBERO 経由で必要
SYSTEM_PACKAGES = [
    "ffmpeg",
    "git",
    "unzip",
    "libgl1",
    "libglib2.0-0",
    "libsm6",
    "libxext6",
    "libexpat1",
    "libfontconfig1-dev",
    "libmagickwand-dev",
]


def install_system_packages() -> None:
    if shutil.which("apt-get") is None:
        print("(skip) apt-get が見つからないためシステムパッケージ導入をスキップ。"
              "ffmpeg / libgl1 等はイメージ側で事前導入してください。")
        return

    if os.geteuid() == 0:
        base: list[str] = ["apt-get"]
    elif shutil.which("sudo"):
        base = ["sudo", "apt-get"]
    else:
        print("(warn) root でも sudo でもないため apt-get をスキップします。"
              "パッケージはイメージ側で事前導入してください。")
        return

    run_quiet([*base, "update", "-qq"])
    run_quiet([*base, "install", "-y", "-qq", *SYSTEM_PACKAGES])

    print("System packages ready.")
