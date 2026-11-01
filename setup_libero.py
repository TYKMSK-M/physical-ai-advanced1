"""LIBERO-plus 評価環境のセットアップ (Section 5 相当).

手順 (notebook と同じ):
  1. MuJoCo/robosuite 系の依存を pin 付きで install
  2. LIBERO-plus fork を固定 SHA で clone/checkout
  3. assets.zip を HF から取得して展開
  4. ~/.libero/config.yaml を書いてアセット位置を教える
  5. lerobot_eval.py に進捗表示用のパッチを当てる
  6. sys.path / sys.modules を掃除して fork 側を優先ロード

Colab 版からの変更点:
- Drive キャッシュ (tar.gz / zip) -> config.CACHE_DIR 配下に同等実装
- ipywidgets の展開進捗 widget -> tqdm のみ (IPython 不要)
"""

from __future__ import annotations

import importlib
import shutil
import sys
import time
from pathlib import Path

from . import config
from .hf_fetch import run_hf_with_retry
from .proc import run_quiet


def _install_pinned_dependencies() -> None:
    # 既存 install を除去 (再実行安全性)
    run_quiet(
        [
            sys.executable, "-m", "pip", "uninstall", "-y",
            "hf-libero", "libero", "robosuite",
        ],
        check=False,
    )

    # バージョンを揃えないと robosuite / mujoco の API 差でクラッシュするため厳格に固定
    run_quiet(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "robosuite==1.4.1",  # LIBERO-plus が期待する固定版
            "bddl==1.0.1",
            "easydict==1.13",
            "mujoco==3.7.0",
            "matplotlib==3.10.8",
            "Wand==0.6.13",
            "scikit-image==0.25.2",
            "gym==0.26.2",
            "future",  # bddl 1.0.1 が依存宣言していない隠れ依存
        ]
    )

    # robosuite が期待どおりの版でロードされることを担保
    if importlib.metadata.version("robosuite") != "1.4.1":
        raise RuntimeError("robosuite 1.4.1 is required.")


def _clone_fork() -> None:
    if not (config.LIBERO_PLUS_DIR / ".git").is_dir():
        shutil.rmtree(config.LIBERO_PLUS_DIR, ignore_errors=True)
        run_quiet(
            [
                "git", "clone", "--quiet",
                "https://github.com/sylvestf/LIBERO-plus.git",
                str(config.LIBERO_PLUS_DIR),
            ]
        )

    # 固定 SHA へ checkout。既存 clone に対しても再実行安全になるよう、
    # checkout 失敗時のみ fetch でリカバリする
    checkout = run_quiet(
        [
            "git", "-C", str(config.LIBERO_PLUS_DIR),
            "checkout", "--quiet", config.LIBERO_PLUS_SHA,
        ],
        check=False,
    )
    if checkout.returncode != 0:
        # SHA がローカルに無い (shallow clone 等) なら fetch してから checkout
        run_quiet(
            [
                "git", "-C", str(config.LIBERO_PLUS_DIR),
                "fetch", "--quiet", "--depth", "1", "origin",
                config.LIBERO_PLUS_SHA,
            ]
        )
        run_quiet(
            [
                "git", "-C", str(config.LIBERO_PLUS_DIR),
                "checkout", "--quiet", config.LIBERO_PLUS_SHA,
            ]
        )

    # fork を editable install。--no-deps で pin 済みの依存を上書きされないようにする
    run_quiet(
        [
            sys.executable, "-m", "pip", "install",
            "--quiet", "--no-deps", "-e", str(config.LIBERO_PLUS_DIR),
        ]
    )


def _extract_tar_with_pbar(tar_path: Path, dest: Path) -> None:
    import tarfile

    from tqdm.auto import tqdm

    # tar.gz は zip と違って中央ディレクトリを持たないため、
    # 前回作成時の .count サイドカーがあれば tqdm の total に即使う。
    count_sidecar = tar_path.with_suffix(tar_path.suffix + ".count")
    total = None
    if count_sidecar.is_file():
        try:
            total = int(count_sidecar.read_text().strip())
            print(f"サイドカーから件数を取得: {total} files", flush=True)
        except Exception:
            total = None
    if total is None:
        print("件数を数えています (初回のみ、少し時間かかる)...", flush=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            total = sum(1 for _ in tf)

    with tarfile.open(tar_path, "r:gz") as tf:
        pbar = tqdm(total=total, desc="tar 展開", unit="files")
        while True:
            member = tf.next()
            if member is None:
                break
            tf.extract(member, dest)
            pbar.update(1)
        pbar.close()


def _prepare_assets() -> None:
    """assets (45 万小ファイル 6GB) を取得して展開する.

    優先度: (1) tar.gz キャッシュ (最速) > (2) zip キャッシュ > (3) HF DL
    tar.gz を共有キャッシュに持っておくと、次回以降は展開だけで数分。
    """
    if config.LIBERO_PLUS_ASSETS_DIR.is_dir():
        return

    assets_root = config.WORKDIR / "libero_plus_assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    cache_tar = config.CACHE_DIR / "libero_plus_assets.tar.gz"
    cache_zip = config.CACHE_DIR / "libero_plus_assets.zip"

    if cache_tar.is_file():
        # ---- 最速パス: tar.gz を共有キャッシュから復元 ----
        print(f"tar.gz キャッシュから復元中: {cache_tar}", flush=True)
        t0 = time.time()
        _extract_tar_with_pbar(cache_tar, assets_root)
        print(f"tar 展開完了 (in {time.time() - t0:.0f}s)", flush=True)

        # 古い zip キャッシュを削除 (容量節約)
        if cache_zip.is_file():
            try:
                cache_zip.unlink()
                print(f"古い zip キャッシュを削除: {cache_zip}", flush=True)
            except Exception:
                pass
        if config.LIBERO_PLUS_ASSETS_DIR.is_dir():
            return

    # ---- 中速パス: zip キャッシュから復元 or HF DL ----
    local_zip = assets_root / "assets.zip"

    if cache_zip.is_file() and not local_zip.is_file():
        print(f"キャッシュから assets.zip を復元: {cache_zip}", flush=True)
        shutil.copy2(cache_zip, local_zip)

    if local_zip.is_file():
        archive_path = local_zip
        print(f"既存の assets.zip を使用: {archive_path}", flush=True)
    else:
        print("HF から assets.zip をダウンロード中 (6GB, 時間かかる)...", flush=True)
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

        # env-var の pop は huggingface_hub が import 時に disabled を
        # キャッシュするため効かない。公式 API で明示的に切り替える。
        enable_progress_bars()
        try:
            t0 = time.time()
            archive_path = Path(
                run_hf_with_retry(
                    lambda: hf_hub_download(
                        repo_id="Sylvest/LIBERO-plus",
                        repo_type="dataset",
                        filename="assets.zip",
                        local_dir=assets_root,
                        token=False,
                    )
                )
            )
            dl_sec = time.time() - t0
        finally:
            disable_progress_bars()

        size_mb = archive_path.stat().st_size / 1e6
        print(
            f"HF DL 完了: {size_mb:.0f} MB in {dl_sec:.0f}s "
            f"({size_mb / max(dl_sec, 1):.1f} MB/s)",
            flush=True,
        )

    extract_dir = assets_root / "extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 展開は Python の zipfile + tqdm progress bar で行う
    # (subprocess の unzip -o だと 45 万行の出力になって表示が固まるため)
    import zipfile

    from tqdm.auto import tqdm

    print(f"Extracting {archive_path.name} to {extract_dir}...", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="unzip", unit="files"):
            zf.extract(member, extract_dir)
    print(f"unzip 完了 (in {time.time() - t0:.0f}s)", flush=True)

    # zip 内のディレクトリ階層は変わりうるので rglob で assets/ を探す。
    # ネストが浅いものを優先 (本物の assets ルートに近いと想定)
    candidates = sorted(
        (path for path in extract_dir.rglob("assets") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    if not candidates:
        raise FileNotFoundError("LIBERO-plus assets not found.")

    config.LIBERO_PLUS_ASSETS_DIR.parent.mkdir(parents=True, exist_ok=True)
    # 展開した本物の assets を fork 内の期待位置へ移動
    shutil.move(str(candidates[0]), str(config.LIBERO_PLUS_ASSETS_DIR))
    shutil.rmtree(extract_dir, ignore_errors=True)

    # ---- 次回実行用に tar.gz を共有キャッシュへ ----
    # zip + unzip のフルパスは重いので、展開後の assets を tar.gz 化して保存。
    # 次回は tar 展開だけになり数分で復元できる。
    try:
        import tarfile

        from tqdm.auto import tqdm as _tqdm

        t0 = time.time()
        print("共有キャッシュに tar.gz を作成中...", flush=True)
        entries = list(config.LIBERO_PLUS_ASSETS_DIR.rglob("*"))
        total_entries = len(entries) + 1  # +1: root dir 自身
        pbar = _tqdm(total=total_entries, desc="tar 作成", unit="files")

        def _tar_filter(tarinfo):
            pbar.update(1)
            return tarinfo

        with tarfile.open(cache_tar, "w:gz", compresslevel=1) as tf:
            tf.add(config.LIBERO_PLUS_ASSETS_DIR, arcname="assets", filter=_tar_filter)
        pbar.close()
        # 次回復元時に tqdm の total を即座に出せるよう件数サイドカーを保存
        cache_tar.with_suffix(cache_tar.suffix + ".count").write_text(
            str(total_entries) + "\n"
        )
        print(f"tar.gz キャッシュ完了 (in {time.time() - t0:.0f}s)", flush=True)
        if cache_zip.is_file():
            try:
                cache_zip.unlink()
                print(f"旧 zip キャッシュを削除: {cache_zip}", flush=True)
            except Exception:
                pass
    except Exception as error:
        print(f"tar.gz キャッシュ作成失敗 (無視して継続): {error}", flush=True)


def _write_libero_config() -> None:
    # ---- LIBERO の期待する assets パスへの symlink を作る ----
    # LIBERO の robosuite Arena XML 読込は fork 内の相対パス
    # ({LIBERO_PLUS_DIR}/libero/libero/assets/...) をハードコードで参照する。
    # 実体を LIBERO_PLUS_ASSETS_DIR に置いてあるので symlink で繋ぐ。
    expected_assets = (
        config.LIBERO_PLUS_DIR / "libero" / "libero" / "assets"
    )
    if expected_assets.is_symlink() or expected_assets.exists():
        if expected_assets.is_symlink():
            expected_assets.unlink()
        else:
            shutil.rmtree(expected_assets, ignore_errors=True)
    expected_assets.parent.mkdir(parents=True, exist_ok=True)
    expected_assets.symlink_to(config.LIBERO_PLUS_ASSETS_DIR)
    print(f"symlink: {expected_assets} -> {config.LIBERO_PLUS_ASSETS_DIR}")

    # LIBERO は起動時に ~/.libero/config.yaml を読んで
    # assets/bddl/datasets/init_files の場所を解決する
    libero_config_dir = Path.home() / ".libero"
    libero_config_dir.mkdir(parents=True, exist_ok=True)
    (libero_config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"assets: {config.LIBERO_PLUS_ASSETS_DIR}",
                f"bddl_files: {config.LIBERO_PLUS_PACKAGE_ROOT / 'bddl_files'}",
                f"datasets: {config.LIBERO_PLUS_PACKAGE_ROOT.parent / 'datasets'}",
                f"init_states: {config.LIBERO_PLUS_PACKAGE_ROOT / 'init_files'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_eval_script() -> None:
    """lerobot_eval.py に進捗表示パッチを当てる.

    デフォルトの評価スクリプトは冗長な log と tqdm を吐くので抑制。
    代わりに "EVAL_PROGRESS task=x/y episode=a/b" 形式を stdout に流し、
    こちら側でパースして表示する。
    """
    eval_script = config.LEROBOT_SRC / "lerobot" / "scripts" / "lerobot_eval.py"
    source = eval_script.read_text(encoding="utf-8")

    # 冗長な設定 pformat を DEBUG に落とす
    source = source.replace(
        "logging.info(pformat(asdict(cfg)))",
        "logging.debug(pformat(asdict(cfg)))",
        1,
    )
    # 録画時以外に rollout 動画を最大 10 本描画してしまう箇所を 0 に固定
    source = source.replace(
        "max_episodes_rendered = 0 if cfg.eval.recording else 10",
        "max_episodes_rendered = 0",
        1,
    )
    # tqdm を強制無効化
    source = source.replace("disable=inside_slurm()", "disable=True")

    # 現在の task/episode 進捗を保持するグローバル辞書をスクリプトへ注入
    progress_state = '_EVAL_PROGRESS = {"task_index": 0, "task_total": 0}'
    if progress_state not in source:
        import_anchor = "from tqdm import trange\n"
        if import_anchor not in source:
            raise RuntimeError("Evaluation progress import anchor not found.")
        source = source.replace(
            import_anchor, import_anchor + "\n" + progress_state + "\n", 1
        )

    # task ループの先頭で _EVAL_PROGRESS を更新するコードを差し込む
    task_loop_anchor = (
        "        for i, (task_group, task_id, env) in enumerate(tasks):\n"
    )
    task_loop_patch = (
        task_loop_anchor
        + '            _EVAL_PROGRESS["task_index"] = i + 1\n'
        + '            _EVAL_PROGRESS["task_total"] = len(tasks)\n'
    )
    if '_EVAL_PROGRESS["task_index"] = i + 1' not in source:
        if task_loop_anchor not in source:
            raise RuntimeError("Evaluation task-loop anchor not found.")
        source = source.replace(task_loop_anchor, task_loop_patch, 1)

    # episode ループの各周で進捗 1 行を stdout に流す
    episode_loop_anchor = "    for batch_ix in progbar:\n"
    episode_progress_line = (
        '        print('
        'f"EVAL_PROGRESS '
        "task={_EVAL_PROGRESS['task_index']}/"
        "{_EVAL_PROGRESS['task_total']} "
        'episode={batch_ix + 1}/{n_batches}", '
        "flush=True)\n"
    )
    if "EVAL_PROGRESS task=" not in source:
        if episode_loop_anchor not in source:
            raise RuntimeError("Evaluation episode-loop anchor not found.")
        source = source.replace(
            episode_loop_anchor, episode_loop_anchor + episode_progress_line, 1
        )

    eval_script.write_text(source, encoding="utf-8")


def _make_flexibly_patchable() -> None:
    """Section 8.8 (動画) 用に max_episodes_rendered を env 可変にする.

    notebook では動画セルで別途パッチしていた内容をセットアップに統合し、
    LEROBOT_MAX_RECORDED 環境変数で録画本数を制御できるようにする
    (idempotent)。
    """
    eval_script = config.LEROBOT_SRC / "lerobot" / "scripts" / "lerobot_eval.py"
    source = eval_script.read_text(encoding="utf-8")

    # eval_script は module top で os を import していないため
    # __import__("os") を使って self-contained な式にする
    flexible_patch = (
        "max_episodes_rendered = "
        'int(__import__("os").environ.get("LEROBOT_MAX_RECORDED", "0"))'
    )
    broken_patch = (
        "max_episodes_rendered = "
        'int(os.environ.get("LEROBOT_MAX_RECORDED", "0"))'
    )
    if flexible_patch not in source:
        if broken_patch in source:
            source = source.replace(broken_patch, flexible_patch, 1)
        else:
            old_patch = "max_episodes_rendered = 0"
            if old_patch not in source:
                raise RuntimeError(
                    "eval_script のパッチ地点が見つかりません。"
                    "setup-libero を再実行してください。"
                )
            source = source.replace(old_patch, flexible_patch, 1)
        eval_script.write_text(source, encoding="utf-8")


def _verify_fork_loaded() -> None:
    # pip install した LIBERO-plus fork が絶対に手前に来るようにする
    config.ensure_libero_on_path()

    # 古い LIBERO/robosuite モジュールキャッシュを追い出す
    for module_name in list(sys.modules):
        if (
            module_name == "libero"
            or module_name.startswith("libero.")
            or module_name == "robosuite"
            or module_name.startswith("robosuite.")
        ):
            del sys.modules[module_name]

    importlib.invalidate_caches()

    import libero
    from libero.libero import benchmark

    # libero パッケージの読込先が fork ディレクトリ配下であることを保証
    search_paths = [Path(path).resolve() for path in getattr(libero, "__path__", [])]
    if not any(
        config.LIBERO_PLUS_DIR.resolve() in path.parents
        or path == config.LIBERO_PLUS_DIR.resolve()
        for path in search_paths
    ):
        raise RuntimeError("LIBERO-plus fork was not loaded.")

    # benchmark モジュールについても同様に確認
    benchmark_path = Path(benchmark.__file__).resolve()
    if config.LIBERO_PLUS_DIR.resolve() not in benchmark_path.parents:
        raise RuntimeError("LIBERO-plus benchmark was not loaded.")


def setup_libero_plus() -> None:
    _install_pinned_dependencies()
    _clone_fork()
    _prepare_assets()
    _write_libero_config()
    _patch_eval_script()
    _make_flexibly_patchable()
    _verify_fork_loaded()

    print("LIBERO-plus ready.")
