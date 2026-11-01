"""外部コマンド実行ヘルパ (Section 2 相当).

- run_quiet            : 成功時は静か、失敗時は末尾 6000 文字を例外に載せる
- run_streaming        : 1 行ごとに print する streaming 版
- run_streaming_rolling: 直近 tail 行だけを "上書き更新" 表示する

Colab 版の ipywidgets HTML widget は、ローカル/コンテナでは
ANSI エスケープによるインプレース更新に置き換えた
(ipywidgets / IPython が無くても動く)。
"""

from __future__ import annotations

import subprocess
import sys
from collections import deque


def run_quiet(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(result.stdout[-6000:])

    return result


def run_streaming(
    command: list[str],
    *,
    prefix: str = "",
    check: bool = True,
) -> int:
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    recent: deque[str] = deque(maxlen=200)
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\n")
        recent.append(line)
        print(f"{prefix}{line}", flush=True)
    return_code = process.wait()
    if check and return_code != 0:
        raise RuntimeError("\n".join(recent) or f"command failed: {command}")
    return return_code


def run_streaming_rolling(
    command: list[str],
    *,
    prefix: str = "",
    tail: int = 10,
    check: bool = True,
) -> int:
    """長時間コマンド (pip install 等) 用: 直近 tail 行だけをインプレース表示.

    notebook 版では ipywidgets の HTML box を上書き更新していた。
    ターミナルでは同一領域を書き換えるため、ブロック末尾に
    確定した行を順次 flush するシンプルな方式にした
    (出力が数百行でもログが肥大化しすぎないよう 5 行ごとに 1 行代表を出す)。
    """
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    recent: deque[str] = deque(maxlen=tail)
    count = 0
    for raw in process.stdout:
        line = raw.rstrip("\n")
        recent.append(line)
        count += 1
        # 毎行 print するとログが肥大化するので 200 行ごとに 1 行進捗を出す
        if count % 200 == 0:
            print(f"{prefix}... {count} lines | {line}", flush=True)

    # 最終状態 (直近 tail 行) を確定出力
    for buffered in recent:
        print(f"{prefix}{buffered}")
    print(f"{prefix}done ({count} lines total)", flush=True)

    return_code = process.wait()
    if check and return_code != 0:
        raise RuntimeError("\n".join(recent) or f"command failed: {command}")
    return return_code


def python(*args: str) -> list[str]:
    """現 interpreter での pip / -c 実行コマンドを作るヘルパ."""
    return [sys.executable, *args]
