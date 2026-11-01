"""Hugging Face 取得のリトライ + キャッシュ優先ラッパ (Section 4 相当).

匿名アクセスは 429 (Too Many Requests) を食らいやすいので指数バックオフを
実装し、既存キャッシュがあれば実 DL を回避する。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

T = TypeVar("T")


def run_hf_with_retry(operation: Callable[[], T]) -> T:
    """429 のみ最大 6 回リトライ。それ以外のエラーは即再送出する."""
    last_error: BaseException | None = None

    for attempt in range(6):
        try:
            return operation()
        except (HfHubHTTPError, httpx.HTTPStatusError) as error:
            last_error = error
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)

            # 429 以外はリトライしない (404 などを叩き続けない)
            if status != 429 and "429" not in str(error):
                raise

            if attempt == 5:
                break

            # サーバが Retry-After を返せば尊重、無ければ指数バックオフ
            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(headers.get("Retry-After", 15)) + 1
            except (TypeError, ValueError):
                delay = min(120, 15 * (2**attempt) + random.random())

            time.sleep(delay)

    raise RuntimeError("Hugging Faceからの取得に失敗しました。") from last_error


def cached_or_downloaded_snapshot(
    repo_id: str,
    revision: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """まず local_files_only でキャッシュ確認、無ければ実 DL する.

    大きなモデル/データセットを何度も再取得しないための最適化。
    """
    try:
        # キャッシュヒット時はネットワークに触れず即返す
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=False,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        # キャッシュミス -> 429 リトライ付きで実 DL
        # max_workers=1 でサーバに優しくアクセス (429 回避)
        return Path(
            run_hf_with_retry(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=False,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    max_workers=1,
                )
            )
        )
