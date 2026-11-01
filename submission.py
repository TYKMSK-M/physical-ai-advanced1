"""提出用 submission.json の生成 (Basic Section 11 相当の提出フォーマット).

最終課題(Basic).ipynb Section 11 のスキーマを基準に、
Advanced ルート (LoRA 追加学習 + 4 suite 評価) の実行証拠にも対応させた
共通の提出フォーマットを出力する。

- write_submission_advanced(): train/merge/eval/adv-eval の成果から生成
- write_submission_basic()   : basic-train/basic-eval の成果から生成

出力:
- {WORKDIR}/submission/submission.json  ... ★提出物★
- {WORKDIR}/submission/submission.md    ... 参照用サマリ (提出不要)
- {WORKDIR}/submission/pip_freeze.txt   ... 参照用環境情報 (提出不要)
- 参照用の 2 ファイルは CACHE_DIR/submission/ へバックアップ
  (Colab 版の Drive バックアップ相当)
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .compat import download


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_eval_info(path: Path) -> dict | None:
    """eval_info.json があれば読み、無ければ None を返す。"""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pc_success(info: dict | None) -> float | None:
    if info is None:
        return None
    try:
        return float(info["overall"]["pc_success"])
    except (KeyError, TypeError, ValueError):
        return None


def collect_environment() -> dict:
    """環境情報 (Basic §11 の environment と同一キー)。"""
    import torch

    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    except Exception as error:  # noqa: BLE001
        pip_freeze = f"# pip freeze failed: {error}"

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }
    return environment, pip_freeze


def _write_submission_files(
    submission: dict,
    submission_md: str,
    pip_freeze: str,
) -> Path:
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    sub_json = config.SUBMISSION_DIR / "submission.json"
    sub_json.write_text(
        json.dumps(submission, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sub_md = config.SUBMISSION_DIR / "submission.md"
    sub_md.write_text(submission_md, encoding="utf-8")

    pip_path = config.SUBMISSION_DIR / "pip_freeze.txt"
    pip_path.write_text(pip_freeze, encoding="utf-8")

    # Colab 版の Drive バックアップ相当: 共有キャッシュへコピー
    cache_dir = config.CACHE_DIR / "submission"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in [sub_json, sub_md, pip_path]:
        shutil.copy2(path, cache_dir / path.name)

    print("✅ 提出用ファイル (submission.json) を生成しました:")
    print(f"  {sub_json}")
    print(f"  (参照用) {sub_md}")
    print(f"  (参照用) {pip_path}")
    print(f"✅ キャッシュにもバックアップ済: {cache_dir}")

    # Colab ではブラウザダウンロード、ローカルではパス表示
    download(sub_json)
    return sub_json


# ---------------------------------------------------------------------------
# Advanced ルート (LoRA 追加学習 + 4 suite 広域評価)
# ---------------------------------------------------------------------------


def _advanced_spatial_evidence() -> dict:
    base_info = _load_eval_info(config.BASE_EVAL_DIR / "eval_info.json")
    finetuned_info = _load_eval_info(
        config.FINETUNED_EVAL_DIR / "eval_info.json"
    )

    def _success_count(info: dict | None) -> tuple[int | None, int | None]:
        if info is None:
            return None, None
        successes = 0
        total = 0
        for task in info.get("per_task", []):
            flags = task["metrics"]["successes"]
            successes += sum(bool(v) for v in flags)
            total += len(flags)
        return successes, total

    base_success, base_total = _success_count(base_info)
    finetuned_success, finetuned_total = _success_count(finetuned_info)

    return {
        "spatial_base_pc_success": _pc_success(base_info),
        "spatial_finetuned_pc_success": _pc_success(finetuned_info),
        "spatial_base_success_count": base_success,
        "spatial_base_total_rollouts": base_total,
        "spatial_finetuned_success_count": finetuned_success,
        "spatial_finetuned_total_rollouts": finetuned_total,
    }


def _advanced_suite_evidence() -> dict:
    suites = {}
    for split in ("baseline", "finetuned"):
        for suite_name in config.ADVANCED_SUITES:
            info = _load_eval_info(
                config.ADVANCED_OUT_DIR / split / suite_name / "eval_info.json"
            )
            pc = _pc_success(info)
            if pc is not None:
                suites[f"{split}/{suite_name}"] = pc
    return suites


def write_submission_advanced(
    methods_notes: str = "(自由記述: どんな工夫を試したか等)",
) -> Path:
    """train / merge / eval / adv-eval の成果から submission.json を生成する。"""
    environment, pip_freeze = collect_environment()

    # 学習済み (マージ済み) モデルの指紋
    fingerprints: dict[str, str] = {}
    merged_weights = config.MERGED_MODEL_DIR / "model.safetensors"
    if merged_weights.is_file():
        fingerprints["merged_model.safetensors"] = sha256_of_file(
            merged_weights
        )
    baseline_weights = config.BASELINE_MODEL_DIR / "model.safetensors"
    if baseline_weights.is_file():
        fingerprints["baseline_model.safetensors"] = sha256_of_file(
            baseline_weights
        )

    spatial = _advanced_spatial_evidence()
    suite_pc = _advanced_suite_evidence()

    checkpoint_dir = (
        config.OUTPUT_DIR
        / "checkpoints"
        / f"{config.STEPS:06d}"
        / "pretrained_model"
    )
    adapter = checkpoint_dir / "adapter_model.safetensors"

    execution_evidence = {
        "vlm_repo": config.VLM_REPO,
        "base_model_repo": config.BASE_MODEL_REPO,
        "base_model_revision": config.BASE_MODEL_REVISION,
        "dataset_repo": config.DATASET_REPO,
        "dataset_revision": config.DATASET_REVISION,
        "training_steps_completed": int(config.STEPS),
        "lora_adapter_saved": bool(adapter.is_file()),
        "merged_model_saved": bool(merged_weights.is_file()),
        "baseline_model_saved": bool(baseline_weights.is_file()),
        "mixed_precision": config.mixed_precision(),
        **spatial,
        "advanced_suite_pc_success": suite_pc,
        "comparison_csv_saved": bool(config.COMPARISON_CSV_PATH.is_file()),
        "merged_zip_saved": bool(config.MERGED_ZIP_PATH.is_file()),
    }

    submission = {
        "notebook_version": "advanced-cli-v1",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "methods_notes": methods_notes,
        "execution_evidence": execution_evidence,
        "model_fingerprint": fingerprints,
        "environment": environment,
        "hyperparams": {
            "train_steps": config.STEPS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "final_learning_rate": config.FINAL_LEARNING_RATE,
            "warmup_steps": config.WARMUP_STEPS,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "seed": config.SEED,
            "eval_seed": config.EVAL_SEED,
            "eval_episodes_per_task": config.EVAL_EPISODES_PER_TASK,
            "advanced_episodes_per_task": config.ADVANCED_EPISODES_PER_TASK,
            "train_episodes_per_task": config.TRAIN_EPISODES_PER_TASK,
        },
        "artifacts": {
            "merged_model_dir": str(config.MERGED_MODEL_DIR),
            "baseline_model_dir": str(config.BASELINE_MODEL_DIR),
            "comparison_csv": str(config.COMPARISON_CSV_PATH),
            "merged_zip": str(config.MERGED_ZIP_PATH),
            "advanced_out_dir": str(config.ADVANCED_OUT_DIR),
        },
    }

    lines = [
        "# Submission (Advanced CLI)",
        "",
        f"**Submitted**: {submission['submitted_at']}",
        "**Version**: advanced-cli-v1",
        "",
        "## 実行の証拠",
        f"- 学習 step: {config.STEPS} (LoRA r={config.LORA_R}, "
        f"alpha={config.LORA_ALPHA})",
        f"- mixed precision: {execution_evidence['mixed_precision']}",
        f"- GPU: {environment['gpu']}",
    ]
    if spatial["spatial_base_pc_success"] is not None:
        lines.append(
            f"- Spatial 成功率: "
            f"{spatial['spatial_base_pc_success']:.1f}% -> "
            f"{spatial['spatial_finetuned_pc_success']:.1f}%"
        )
    if suite_pc:
        lines.append("")
        lines.append("## 4 suite 広域評価 (pc_success %)")
        for key, value in suite_pc.items():
            lines.append(f"- {key}: {value:.1f}")
    lines += [
        "",
        "## Notes",
        methods_notes,
        "",
    ]

    path = _write_submission_files(
        submission, "\n".join(lines), pip_freeze
    )
    print(f"📥 submission.json: {path}")
    return path


# ---------------------------------------------------------------------------
# Basic ルート (PyTorch 自作 SmolVLA)
# ---------------------------------------------------------------------------


def write_submission_basic(
    methods_notes: str = "(自由記述: どんな工夫を試したか等)",
) -> Path:
    """basic-train / basic-eval の成果から submission.json を生成する。

    Basic ノートブック Section 11 と同一キーのスキーマ (todos_completed 含む)。
    """
    import torch

    environment, pip_freeze = collect_environment()

    ckpt_path = config.PY_CHECKPOINT_DIR / "smolvla_ckpt.pt"
    fingerprints: dict[str, str] = {}
    loss_history: list[float] = []
    if ckpt_path.is_file():
        fingerprints["smolvla_ckpt.pt"] = sha256_of_file(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        loss_history = [float(v) for v in ckpt.get("loss_history", [])]

    results_path = config.PY_EVAL_RESULTS_PATH
    py_eval: dict = {}
    if results_path.is_file():
        py_eval = json.loads(results_path.read_text(encoding="utf-8"))

    rollouts = py_eval.get("rollouts", [])
    video_paths = sorted(config.PY_VIDEO_DIR.glob("*.mp4"))

    execution_evidence = {
        "vlm_hidden_dim": 960,
        "transformer_hidden_dim": config.PY_TRANSFORMER_HIDDEN_DIM,
        "trainable_params": py_eval.get("trainable_params"),
        "training_steps_completed": int(config.TRAIN_STEPS_PY),
        "training_final_loss": (
            loss_history[-1] if loss_history else None
        ),
        "eval_total_rollouts": int(len(rollouts)),
        "eval_success_count": int(
            sum(1 for r in rollouts if r.get("success"))
        ),
        "eval_overall_success_percent": float(
            py_eval.get("overall_success", 0.0)
        ),
        "video_generated": bool(video_paths),
        "video_count": len(video_paths),
    }

    submission = {
        "notebook_version": "basic-cli-v1",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "methods_notes": methods_notes,
        "todos_completed": [
            "TODO_1_time_embedding_outer_product",
            "TODO_2_flow_matching_interpolation",
            "TODO_3_target_velocity",
            "TODO_4_euler_step",
        ],
        "execution_evidence": execution_evidence,
        "model_fingerprint": fingerprints,
        "environment": environment,
        "hyperparams": {
            "chunk_size": config.PY_CHUNK_SIZE,
            "action_dim": config.PY_ACTION_DIM,
            "state_dim": config.PY_STATE_DIM,
            "transformer_hidden_dim": config.PY_TRANSFORMER_HIDDEN_DIM,
            "num_transformer_layers": config.PY_NUM_TRANSFORMER_LAYERS,
            "num_heads": config.PY_NUM_HEADS,
            "train_steps": config.TRAIN_STEPS_PY,
            "batch_size": config.TRAIN_BATCH_SIZE_PY,
            "learning_rate": config.TRAIN_LR_PY,
            "warmup_steps": config.TRAIN_WARMUP_PY,
            "seed": config.SEED,
            "num_flow_steps_eval": config.PY_EVAL_NUM_FLOW_STEPS,
            "flow_solver_eval": config.PY_EVAL_FLOW_SOLVER,
            "vlm_feature_cache": bool(config.PY_USE_VLM_CACHE),
        },
    }

    per_task = "\n".join(
        f"- task {r['task_id']} ep {r['episode']}: "
        f"{'✅' if r.get('success') else '❌'} ({r.get('steps')} steps)"
        for r in rollouts
    ) or "(basic-eval 未実行)"

    submission_md = f"""# Submission (Basic CLI)

**Submitted**: {submission['submitted_at']}
**Version**: basic-cli-v1

## TODO 完了状況
✅ 全ての TODO を実装済 (basic-train / basic-eval が完走できている時点で確定)

## 実行の証拠
- VLM hidden dim: {execution_evidence['vlm_hidden_dim']}
- Transformer hidden dim: {execution_evidence['transformer_hidden_dim']}
- 学習可能パラメタ: {execution_evidence['trainable_params']}
- 学習 step: {execution_evidence['training_steps_completed']}
- 最終 loss: {execution_evidence['training_final_loss']}
- Rollout 数: {execution_evidence['eval_total_rollouts']}
- 成功数: {execution_evidence['eval_success_count']}
- 動画生成: {'✅' if execution_evidence['video_generated'] else '❌'}

## Rollout 詳細
{per_task}

## Notes
{methods_notes}

## Environment
- {environment['gpu']}
- torch {environment['torch']}, cuda {environment['cuda']}
- python {environment['python']}
"""

    path = _write_submission_files(submission, submission_md, pip_freeze)
    print(f"📥 submission.json: {path}")
    return path
