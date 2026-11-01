"""実行パス・定数・学習/評価ハイパラの一元定義.

最終課題(Advanced).ipynb の Section 1 (環境変数) と Section 6 (定数)・
Section 8.0 (ハイパラ)・Section 9 (評価スイート) に対応する。

Colab 版からの主な変更点:
- /content/workdir        -> 環境変数 SMOLVLA_WORKDIR (既定: project/workdir)
- Google Drive キャッシュ -> 環境変数 SMOLVLA_CACHE_DIR (既定: <WORKDIR>/shared_cache)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_workdir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").is_file():
        return project_root / "workdir"
    return Path.cwd() / "workdir"


# --- WORKDIR -----------------------------------------------------
# notebook の /content/workdir 相当。Drive 直で作業すると 45 万小ファイルの
# 展開などが遅くなるため、高速なローカルディスクに置く (notebook と同じ方針)。
WORKDIR = Path(
    os.environ.get("SMOLVLA_WORKDIR", str(_default_workdir()))
).resolve()

# --- 共有アセットキャッシュ --------------------------------------
# notebook の Google Drive (MyDrive/smolvla_task) 相当。
# LIBERO-plus assets の tar.gz / zip キャッシュを置いて再実行を速くする。
CACHE_DIR = Path(
    os.environ.get("SMOLVLA_CACHE_DIR", str(WORKDIR / "shared_cache"))
).resolve()

# --- LeRobot / LIBERO-plus の配置先 ------------------------------
LEROBOT_DIR = WORKDIR / "lerobot"
LEROBOT_SRC = LEROBOT_DIR / "src"
LEROBOT_TAG = "v0.6.0"

LIBERO_PLUS_SHA = "4976dc3"
LIBERO_PLUS_DIR = WORKDIR / "LIBERO-plus"
LIBERO_PLUS_PACKAGE_ROOT = LIBERO_PLUS_DIR / "libero" / "libero"
LIBERO_PLUS_ASSETS_DIR = WORKDIR / "libero_plus_assets" / "assets"

# --- 出力パス ----------------------------------------------------
OUTPUT_DIR = WORKDIR / "model_tournament" / "smolvla_spatial_full_20261101"
MERGED_MODEL_DIR = WORKDIR / "smolvla_libero_plus_spatial_lora_merged"
BASELINE_MODEL_DIR = WORKDIR / "smolvla_libero_plus_baseline"

BASE_EVAL_DIR = WORKDIR / "eval" / "base"
FINETUNED_EVAL_DIR = WORKDIR / "eval" / "spatial_lora"
COMPARISON_CSV_PATH = WORKDIR / "libero_spatial_comparison.csv"
MERGED_ZIP_PATH = WORKDIR / "smolvla_libero_plus_spatial_lora_merged.zip"

VIDEO_OUT_DIR = WORKDIR / "eval" / "video"
ADVANCED_OUT_DIR = WORKDIR / "eval" / "advanced"

SUBMISSION_DIR = WORKDIR / "submission"

# --- Basic ルート (PyTorch 自作 SmolVLA) の出力パス -------------
PY_CHECKPOINT_DIR = WORKDIR / "py_train"
PY_VIDEO_DIR = WORKDIR / "py_videos"
PY_EVAL_RESULTS_PATH = WORKDIR / "py_eval_results.json"
PY_EXPERIMENTS_DIR = WORKDIR / "experiments"
PY_QUALIFIED_DIR = WORKDIR / "qualified"
PY_ABANDONED_DIR = WORKDIR / "abandoned"

# --- モデル / データセットの固定 revision ------------------------
# revision を commit hash で固定して再現性を担保 (notebook と同一)
BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"

# SmolVLA の Vision-Language backbone (500M パラメータ版)
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

# --- LIBERO-Spatial の 10 タスク (学習・評価で共通) -------------
# データセット側の task 名と厳密一致させる必要がある
SPATIAL_TASK_NAMES = [
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
]

# --- 学習・評価データ選抜 ----------------------------------------
TRAIN_EPISODES_PER_TASK = None  # Spatial 10 tasksの全episodeを使用
SEED = 42
EVAL_SEED = 2026
PY_EVAL_POLICY_SEED = 12026

# --- LoRA 追加学習用ハイパラ (lerobot-train) --------------------
STEPS = 5000  # screening学習の総step数
LOG_FREQ = 100  # loss print 頻度
BATCH_SIZE = 8  # L40S 46GB上のscreening batch
LEARNING_RATE = 1e-4  # action expert + state projection用peak learning rate
FINAL_LEARNING_RATE = 1e-5  # cosine decay 後の lr
WARMUP_STEPS = 100  # linear warmup step 数
LORA_R = 16  # LoRA 低ランク次元 (4/8/16/32 が定番)
LORA_ALPHA = 16  # LoRA スケーリング (通常 = r or 2r)

# --- LeRobot 評価用ハイパラ (lerobot-eval / Section 8.5) --------
EVAL_TASK_IDS = list(range(10))  # LIBERO-Spatial の task_id (全 10)
EVAL_EPISODES_PER_TASK = 1  # 10 task x 1 ep x 2 model = 20 rollout

# LIBERO 内部のカメラ名 -> SmolVLA が期待する名前へのマッピング
EVAL_CAMERA_MAPPING = {
    "agentview_image": "front",  # 俯瞰カメラ
    "robot0_eye_in_hand_image": "wrist",  # ハンドカメラ
}

# --- アドバンスド評価 (Section 9) -------------------------------
# 各 suite の 10 tasks (id 0-9) から等間隔に 3 個をサンプル (start/middle/end)
ADVANCED_TASK_IDS = [0, 4, 8]
ADVANCED_SUITES = {
    "libero_spatial": ADVANCED_TASK_IDS,  # 学習ドメイン
    "libero_object": ADVANCED_TASK_IDS,  # オブジェクト汎化
    "libero_goal": ADVANCED_TASK_IDS,  # 目標語彙汎化
    "libero_10": ADVANCED_TASK_IDS,  # long-horizon 汎化
}
ADVANCED_EPISODES_PER_TASK = 5  # A1: 1 -> 5 に増量 (1 ep はノイズ大で判定不能だった)

# rollout 動画 (Section 8.8)
VIDEO_TASK_ID = 0
VIDEO_N_EPISODES = 2

# --- Basic ルート (自作 SmolVLA) のモデル・学習・評価ハイパラ ---
# 最終課題(Basic).ipynb Section 8.5 / 9.3 / 10.1 と同一の値。
# 教材サイズ (from-scratch x 50 episode 向けに小型化)。
PY_ACTION_DIM = 7              # LIBERO: 6-DoF + gripper = 7 次元
PY_STATE_DIM = 8               # eef pos 3 + axis-angle 3 + gripper qpos 2
PY_CHUNK_SIZE = 8              # 8 step 予測 (0.4 秒)
PY_NUM_TRANSFORMER_LAYERS = 4  # 16 -> 4 に圧縮
PY_NUM_HEADS = 8               # transformer_hidden_dim=256 に合わせて (256/32)
PY_TRANSFORMER_HIDDEN_DIM = 256  # 720 -> 256 に圧縮

# PyTorch 版の学習ハイパラ (Basic Section 9.3)
# Stage 2 (CHANGES 2026-10-29): 12k → 24k (loss 0.84 で未収束のため約 4 epoch)。
# C1 cache 自動 ON 条件 (step×bs > N=6159) は引き続き満たす。
TRAIN_STEPS_PY = 24_000        # 6159 samples に対して約 4 epoch
TRAIN_BATCH_SIZE_PY = 1        # VLM が重いので batch=1 が無難
TRAIN_LR_PY = 3e-4             # peak lr
TRAIN_WARMUP_PY = 300          # linear warmup 区間
TRAIN_LOG_FREQ_PY = 100        # loss を print する頻度
TRAIN_NUM_WORKERS_PY = 0
TRAIN_CHECKPOINT_FREQ_PY = 1_000
TRAIN_VALIDATION_FREQ_PY = 1_000
PY_TRAIN_EPISODES_PER_TASK = 4
PY_VALIDATION_EPISODES_PER_TASK = 1
PY_VALIDATION_MAX_SAMPLES = 256
PY_VALIDATION_SEED = 2027
WANDB_PROJECT_PY = "smolvla-adv-basic"

# PyTorch 版の評価ハイパラ (Basic Section 10.1)
PY_EVAL_TASK_IDS = [0, 1]      # LIBERO-Spatial の task 0-1 だけ
PY_EVAL_EPISODES_PER_TASK = 5  # 各 task 5 episodes. 合計 10 rollout
PY_EVAL_MAX_STEPS = 300        # 1 episode の最大 env step
PY_EVAL_NUM_FLOW_STEPS = 16    # Flow matching の ODE step 数 (Stage 1: 10 -> 16)
# Stage 1 (CHANGES 2026-10-29): chunk 全部を使い切るのではなく
# receding step 個実行するごとに再予測する (receding horizon)。
# chunk_size (=8) と同じ値に戻すと notebook 既定の「使い切るまで開ループ」動作。
PY_EVAL_RECEDING_STEPS = 4

# --- 精度・性能向上のスイッチ (CHANGES 2026-10-28 の提案実装) -----
# C1: VLM 特徴の事前計算キャッシュ。VLM は完全凍結なので、学習データ分の
#     vlm_hidden を 1 回だけ計算して保存し、学習中の 500M forward を省略する。
PY_USE_VLM_CACHE = True
PY_VLM_CACHE_DIR = WORKDIR / "py_vlm_cache"
PY_VLM_CACHE_SHARD_SIZE = 512  # 1 shard あたりのサンプル数

# B4: 推論 ODE solver。"euler" (notebook 既定) or "heun" (2次精度, eval 2倍)
PY_EVAL_FLOW_SOLVER = "heun"


def seed_everything(seed: int = SEED) -> None:
    """Basic ルートの Python / NumPy / PyTorch 乱数を固定する."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_environment() -> None:
    """Section 1 相当の環境変数設定。torch / HF を触る前に必ず呼ぶ。"""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Hugging Face Hub のノイズ抑制 (ユーザーが先に設定した値は尊重)
    defaults = {
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "DIFFUSERS_VERBOSITY": "error",
        # HF / LeRobot のキャッシュを WORKDIR 配下へ固定
        "HF_HOME": str(WORKDIR / "hf_cache"),
        "HF_LEROBOT_HOME": str(WORKDIR / "lerobot_cache"),
        # 長時間学習で起こりやすい断片化 OOM を回避
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Colab のヘッドレス GPU 環境では EGL バックエンドで MuJoCo を描画
        "MUJOCO_GL": "egl",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def check_runtime(require_gpu: bool = True) -> None:
    """Section 1 の実行前チェック (Python バージョン / GPU)。"""
    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12以上が必要です。")

    if not require_gpu:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU が利用できません。NVIDIA GPU + CUDA 環境で実行してください。"
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def mixed_precision() -> str:
    """bf16 対応 GPU (A100/H100/L40/RTX40 系など) なら bf16、それ以外 fp16。"""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def ensure_lerobot_on_path() -> None:
    """clone した LeRobot src を sys.path の最優先に置く (Section 3 相当)。"""
    src = str(LEROBOT_SRC)
    sys.path[:] = [item for item in sys.path if item != src]
    sys.path.insert(0, src)


def ensure_libero_on_path() -> None:
    """LIBERO-plus fork を sys.path の最優先に置く (Section 5 相当)。"""
    root = str(LIBERO_PLUS_DIR)
    sys.path[:] = [item for item in sys.path if item != root]
    sys.path.insert(0, root)


def subprocess_env(**extra: str) -> dict[str, str]:
    """lerobot-train / lerobot-eval の子プロセスに渡す共通環境。

    - PATH: 現在のPython環境にあるLeRobot CLIを優先
    - PYTHONPATH: LIBERO-plus fork とパッチ済み LeRobot src を優先
    - 進捗系は無効化して、こちら側のパーサに出力を整理させる
    """
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = (
        str(LIBERO_PLUS_DIR)
        + os.pathsep
        + str(LEROBOT_SRC)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["MUJOCO_GL"] = "egl"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_VERBOSITY"] = "error"
    env["TQDM_DISABLE"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONUNBUFFERED"] = "1"
    env.update({key: str(value) for key, value in extra.items()})
    return env
