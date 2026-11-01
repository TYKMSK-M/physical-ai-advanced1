"""マージ済みモデルの zip 化 (Section 8.7 相当).

Colab の files.download は compat.download に委譲
(ローカルではパス表示のみ)。
"""

from __future__ import annotations

from zipfile import ZIP_STORED, ZipFile

from . import config
from .compat import download


def package_merged_model() -> None:
    # 既に zip があれば消す (再実行時のため)
    if config.MERGED_ZIP_PATH.exists():
        config.MERGED_ZIP_PATH.unlink()

    # ZIP_STORED (無圧縮) を選んでいるのは safetensors が既に圧縮済みで、
    # 再圧縮しても縮まないため。IO 時間だけ増えるので無圧縮で良い。
    with ZipFile(
        config.MERGED_ZIP_PATH,
        mode="w",
        compression=ZIP_STORED,
        allowZip64=True,  # safetensors が 4GB 超になるケースの保険
    ) as archive:
        for file_path in sorted(config.MERGED_MODEL_DIR.rglob("*")):
            if file_path.is_file():
                # zip 内では MERGED_MODEL_DIR の basename を root にする
                archive.write(
                    file_path,
                    arcname=(
                        config.MERGED_MODEL_DIR.name
                        / file_path.relative_to(config.MERGED_MODEL_DIR)
                    ),
                )

    print(f"Saved: {config.MERGED_ZIP_PATH}")
    download(config.MERGED_ZIP_PATH)
    download(config.COMPARISON_CSV_PATH)
