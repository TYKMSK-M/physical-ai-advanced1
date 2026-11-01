"""事前学習済み SmolVLA のダウンロードと LoRA 追加学習 (Section 8.1-8.2 相当).

- download_base_model : 追加学習の初期重み (必要最小限ファイルのみ)
- train_lora          : lerobot-train を subprocess で起動し進捗をリアルタイム表示
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import deque
from pathlib import Path

from . import config
from .hf_fetch import cached_or_downloaded_snapshot


def download_base_model() -> Path:
    """必要最小限のファイルのみ取得する。README や eval/** (動画) は落とさない."""
    base_model_local = cached_or_downloaded_snapshot(
        config.BASE_MODEL_REPO,
        config.BASE_MODEL_REVISION,
        allow_patterns=[
            "config.json",  # モデル構成
            "model.safetensors",  # 本体 weight
            "train_config.json",  # 学習時設定 (再利用のため)
            "policy_preprocessor.json",  # 入力前処理設定
            "policy_preprocessor*.safetensors",  # 入力側 stats
            "policy_postprocessor.json",  # 出力後処理設定
            "policy_postprocessor*.safetensors",  # 出力側 stats
        ],
        ignore_patterns=[
            "README.md",
            "eval/**",  # 公開 repo に含まれる評価動画 (重い) は不要
        ],
    )

    # 本体 weight が確実に取れていることを確認
    if not (base_model_local / "model.safetensors").is_file():
        raise FileNotFoundError("Base model not found.")

    print("Base model ready.")
    return base_model_local


def train_lora(episode_indices: list[int], base_model_local: Path) -> None:
    """lerobot-train の CLI をそのまま呼んで LoRA 学習を実行.

    stdout をパースして step/loss/lr を LOG_FREQ ごとに 1 行だけ表示する。
    """
    # EPISODE_INDICES を CLI 引数として渡すため JSON 配列文字列に整形
    episodes_json = "[" + ",".join(map(str, episode_indices)) + "]"

    command = [
        "lerobot-train",
        # 初期重みと VLM backbone のパス
        f"--policy.path={base_model_local}",
        f"--policy.vlm_model_name={config.VLM_REPO}",
        # Hub への push は行わない
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        # 現datasetのfront/wrist、state 8D、action 7Dからfeaturesを推論
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        # LoRA 対象は expert 側のみ。VLM 側は凍結する
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        # 学習率と cosine スケジューラ
        f"--policy.optimizer_lr={config.LEARNING_RATE}",
        f"--policy.scheduler_decay_lr={config.FINAL_LEARNING_RATE}",
        f"--policy.scheduler_warmup_steps={config.WARMUP_STEPS}",
        f"--policy.scheduler_decay_steps={config.STEPS}",
        # データセットと選抜済み episode
        f"--dataset.repo_id={config.DATASET_REPO}",
        f"--dataset.revision={config.DATASET_REVISION}",
        f"--dataset.episodes={episodes_json}",
        "--dataset.eval_split=0.2",
        "--dataset.use_imagenet_stats=false",
        "--dataset.video_backend=torchcodec",
        # 出力ディレクトリ / ジョブ名
        f"--output_dir={config.OUTPUT_DIR}",
        "--job_name=smolvla_libero_plus_spatial_lora",
        # 学習ステップ / batch / DataLoader
        f"--steps={config.STEPS}",
        f"--batch_size={config.BATCH_SIZE}",
        "--num_workers=4",
        "--prefetch_factor=4",
        "--persistent_workers=true",
        # 学習中の env 評価は無効 (この CLI 側で別途評価する)
        "--env_eval_freq=0",
        "--eval_steps=1000",
        "--max_eval_samples=1000",
        f"--seed={config.SEED}",
        # checkpoint は最終 step のみローカル保存
        "--save_checkpoint=true",
        f"--save_freq={config.STEPS}",
        "--save_checkpoint_to_hub=false",
        f"--log_freq={config.LOG_FREQ}",
        "--wandb.enable=true",
        "--wandb.project=smolvla-adv-model-tournament",
        # LoRA 設定
        "--peft.method_type=LORA",
        f"--peft.r={config.LORA_R}",
        f"--peft.lora_alpha={config.LORA_ALPHA}",
    ]

    # 子プロセスの環境変数 (共通 + 学習固有)
    training_env = config.subprocess_env(
        ACCELERATE_MIXED_PRECISION=config.mixed_precision(),
    )

    # 出力ディレクトリはクリーンな状態から始める
    shutil.rmtree(config.OUTPUT_DIR, ignore_errors=True)

    print("Preparing data and starting training...")

    process = subprocess.Popen(
        command,
        cwd=config.LEROBOT_DIR,
        env=training_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # 直近ログ 80 行を保持しておき、失敗時にダンプする (診断用)
    recent_lines: deque[str] = deque(maxlen=80)
    report_step = config.LOG_FREQ

    assert process.stdout is not None

    # "step:... loss:... lr:..." 行だけを拾って整形表示する
    for raw_line in process.stdout:
        line = raw_line.replace("\r", "").strip()
        if not line:
            continue
        recent_lines.append(line)

        if "step:" in line and "loss:" in line:
            loss_match = re.search(r"loss:([0-9.eE+-]+)", line)
            lr_match = re.search(r"lr:([0-9.eE+-]+)", line)
            loss = loss_match.group(1) if loss_match else "n/a"
            lr = lr_match.group(1) if lr_match else "n/a"
            print(
                f"step {report_step:4d}/{config.STEPS}  loss={loss}  lr={lr}",
                flush=True,
            )
            report_step += config.LOG_FREQ

    return_code = process.wait()

    # 失敗時は末尾ログを吐いてから raise (原因調査用)
    if return_code != 0:
        print("\n".join(recent_lines))
        raise RuntimeError(f"Training failed: {return_code}")

    print("Training complete.")
