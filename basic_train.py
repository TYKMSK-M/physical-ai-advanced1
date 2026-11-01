"""PyTorch 自作 SmolVLA の学習ループ (Basic Section 9.3 相当).

- AdamW + linear warmup -> cosine decay + grad clip
- 学習対象は flow_head / vlm_projector / state_proj (VLM は凍結)
- checkpoint (state_dict + action stats + loss_history) を保存
- loss curve を PNG で保存 (notebook の matplotlib 表示に相当)
"""

from __future__ import annotations

import math
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


@torch.no_grad()
def evaluate_validation_loss(
    smolvla: torch.nn.Module,
    validation_dataset,
    normalize_actions,
    *,
    batch_size: int = 1,
    seed: int = 2026,
    max_samples: int | None = 256,
) -> tuple[float, int]:
    """固定sample/noise/timeで再現可能なvalidation lossを計算する."""
    from torch.utils.data import DataLoader, Subset

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or None")

    evaluated_dataset = validation_dataset
    if max_samples is not None and len(validation_dataset) > max_samples:
        indices = np.linspace(
            0, len(validation_dataset) - 1, num=max_samples, dtype=int
        ).tolist()
        evaluated_dataset = Subset(validation_dataset, indices)

    loader = DataLoader(
        evaluated_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = next(smolvla.flow_head.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_loss = 0.0
    evaluated_samples = 0
    was_training = smolvla.training
    smolvla.eval()
    try:
        for batch in loader:
            vlm_hidden_raw = batch.get("vlm_hidden")
            if vlm_hidden_raw is not None:
                imgs_list, tasks_list = None, None
                vlm_hidden_raw = vlm_hidden_raw.to(device)
            else:
                imgs_tensor = batch["observation.images.front"]
                if imgs_tensor.dtype.is_floating_point:
                    imgs_tensor = (
                        (imgs_tensor * 255).clamp(0, 255).to(torch.uint8)
                    )
                imgs_np = imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()
                imgs_list = [imgs_np[index] for index in range(imgs_np.shape[0])]
                tasks_list = batch["task"]
                if isinstance(tasks_list, str):
                    tasks_list = [tasks_list]

            actions = batch["action"].to(device, dtype=torch.float32)
            actions_norm = normalize_actions(actions)
            states = batch["observation.state"].to(device, dtype=torch.float32)
            noise = torch.randn(
                actions_norm.shape,
                generator=generator,
                device=device,
                dtype=actions_norm.dtype,
            )
            times = torch.rand(
                actions_norm.size(0),
                generator=generator,
                device=device,
                dtype=actions_norm.dtype,
            )
            loss = smolvla.compute_loss(
                imgs_list,
                tasks_list,
                states,
                actions_norm,
                vlm_hidden_raw=vlm_hidden_raw,
                noise=noise,
                t=times,
            )
            current_batch_size = actions_norm.size(0)
            total_loss += float(loss.item()) * current_batch_size
            evaluated_samples += current_batch_size
    finally:
        smolvla.train(was_training)

    if evaluated_samples == 0:
        raise ValueError("validation dataset must not be empty")
    return total_loss / evaluated_samples, evaluated_samples


def train_pytorch_smolvla(
    smolvla: torch.nn.Module,
    train_dataset,
    normalize_actions,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    *,
    output_dir: Path | None = None,
    plot_dir: Path | None = None,
    metrics_path: Path | None = None,
    train_steps: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    checkpoint_freq: int = 1_000,
    resume_path: Path | None = None,
    wandb_logger=None,
    validation_dataset=None,
    validation_metrics_path: Path | None = None,
    validation_freq: int = 1_000,
    validation_max_samples: int | None = 256,
    validation_seed: int = 2026,
) -> Path:
    """Basic Section 9.3 の学習ループをそのまま実行し、checkpoint パスを返す."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from . import config

    target_steps = train_steps or config.TRAIN_STEPS_PY
    effective_batch_size = batch_size or config.TRAIN_BATCH_SIZE_PY
    if target_steps <= 0:
        raise ValueError("train_steps must be positive")
    if effective_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if validation_dataset is not None and validation_freq <= 0:
        raise ValueError("validation_freq must be positive")

    legacy_output = output_dir is None
    output_dir = output_dir or config.PY_CHECKPOINT_DIR
    plot_dir = plot_dir or output_dir
    metrics_path = metrics_path or output_dir / "train_history.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_dataset is not None:
        validation_metrics_path = validation_metrics_path or (
            metrics_path.parent / "validation_history.jsonl"
        )
        validation_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- DataLoader ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS_PY,
        drop_last=True,
    )

    # ---- optimizer: 学習可能な params だけを渡す ----
    trainable_params = [p for p in smolvla.parameters() if p.requires_grad]
    device = next(smolvla.flow_head.parameters()).device
    print(
        f"Optimizer target: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M"
        " params"
    )

    optimizer = AdamW(
        trainable_params, lr=config.TRAIN_LR_PY, weight_decay=1e-4
    )

    # ---- LR schedule: linear warmup -> cosine decay (peak の 10% まで) ----
    def lr_lambda(step: int) -> float:
        if step < config.TRAIN_WARMUP_PY:
            return step / max(config.TRAIN_WARMUP_PY, 1)
        progress = (step - config.TRAIN_WARMUP_PY) / max(
            target_steps - config.TRAIN_WARMUP_PY, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- resume ----
    step = 0
    loss_history: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        _load_model_checkpoint(smolvla, checkpoint)
        if not torch.allclose(checkpoint["action_mean"], action_mean.cpu()):
            raise ValueError("resume checkpoint action_mean does not match dataset")
        if not torch.allclose(checkpoint["action_std"], action_std.cpu()):
            raise ValueError("resume checkpoint action_std does not match dataset")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, next(smolvla.parameters()).device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint["step"])
        loss_history = list(checkpoint.get("loss_history", []))
        validation_history = list(checkpoint.get("validation_history", []))
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", math.inf)
        )
        _restore_rng_state(checkpoint.get("rng_state"))
        if step >= target_steps:
            raise ValueError(
                f"resume step ({step}) must be smaller than target steps ({target_steps})"
            )
        print(f"Resumed training from {resume_path} at step {step}", flush=True)

    # ---- 学習ループ ----
    smolvla.train()
    start = time.time()

    print(
        f"\nTraining {target_steps} optimizer steps "
        f"(batch={effective_batch_size}, accum={gradient_accumulation_steps}, "
        f"effective_batch={effective_batch_size * gradient_accumulation_steps})...",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulated_micro_batches = 0
    while step < target_steps:
        for batch in train_loader:
            if step >= target_steps:
                break

            # C1: 事前計算 vlm_hidden がある場合、画像/タスク文は不要なので
            # 前処理 (GPU->CPU 転送・numpy 変換) ごとスキップする
            vlm_hidden_raw = batch.get("vlm_hidden")
            if vlm_hidden_raw is not None:
                imgs_list, tasks_list = None, None
                vlm_hidden_raw = vlm_hidden_raw.to(device)
            else:
                # 画像を numpy (H, W, C) uint8 に整形 (RealVLMEncoder 用)
                imgs_tensor = batch["observation.images.front"]
                if imgs_tensor.dtype.is_floating_point:
                    imgs_tensor = (imgs_tensor * 255).clamp(0, 255).to(torch.uint8)
                imgs_np = imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()
                imgs_list = [imgs_np[i] for i in range(imgs_np.shape[0])]

                # タスク説明文
                tasks_list = batch["task"]
                if isinstance(tasks_list, str):
                    tasks_list = [tasks_list]

            # action を GPU に + 正規化
            actions = batch["action"].to(device, dtype=torch.float32)
            actions_norm = normalize_actions(actions)

            # state (LIBERO は (B, 8))
            states = batch["observation.state"].to(device, dtype=torch.float32)

            # forward -> loss -> backward (optimizer step単位で勾配蓄積)
            loss = smolvla.compute_loss(
                imgs_list,
                tasks_list,
                states,
                actions_norm,
                vlm_hidden_raw=vlm_hidden_raw,
            )

            (loss / gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.item())
            accumulated_micro_batches += 1
            if accumulated_micro_batches < gradient_accumulation_steps:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            optimizer_loss = accumulated_loss / accumulated_micro_batches
            accumulated_loss = 0.0
            accumulated_micro_batches = 0
            loss_history.append(optimizer_loss)

            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start
            window = loss_history[-config.TRAIN_LOG_FREQ_PY :]
            avg_loss = sum(window) / len(window)
            metric = {
                "step": step,
                "loss": optimizer_loss,
                "loss_avg_100": avg_loss,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "seen_samples": (
                    step * effective_batch_size * gradient_accumulation_steps
                ),
                "elapsed_sec": elapsed,
            }
            _append_jsonl(metrics_path, metric)

            if wandb_logger is not None and (step == 1 or step % 10 == 0):
                wandb_logger.log(
                    {
                        "train/loss_raw": optimizer_loss,
                        "train/loss_avg_100": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/seen_samples": metric["seen_samples"],
                        "system/elapsed_sec": elapsed,
                    },
                    step=step,
                )

            if step % config.TRAIN_LOG_FREQ_PY == 0 or step == 1:
                print(
                    f"step {step:4d}/{target_steps}  "
                    f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if checkpoint_freq > 0 and step % checkpoint_freq == 0:
                periodic_path = output_dir / f"step_{step:06d}.pt"
                _save_checkpoint(
                    periodic_path,
                    smolvla,
                    optimizer,
                    scheduler,
                    step,
                    target_steps,
                    gradient_accumulation_steps,
                    loss_history,
                    action_mean,
                    action_std,
                    trainable_params,
                    validation_history,
                    best_validation_loss,
                )

            if validation_dataset is not None and (
                step % validation_freq == 0 or step == target_steps
            ):
                validation_loss, evaluated_samples = evaluate_validation_loss(
                    smolvla,
                    validation_dataset,
                    normalize_actions,
                    batch_size=effective_batch_size,
                    seed=validation_seed,
                    max_samples=validation_max_samples,
                )
                validation_metric = {
                    "step": step,
                    "loss": validation_loss,
                    "best_loss": min(best_validation_loss, validation_loss),
                    "evaluated_samples": evaluated_samples,
                    "seed": validation_seed,
                }
                validation_history.append(validation_metric)
                _append_jsonl(validation_metrics_path, validation_metric)
                print(
                    f"validation step {step}: loss={validation_loss:.6f} "
                    f"samples={evaluated_samples}",
                    flush=True,
                )
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "val/loss": validation_loss,
                            "val/best_loss": validation_metric["best_loss"],
                            "val/evaluated_samples": evaluated_samples,
                        },
                        step=step,
                    )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    validation_metric["best_loss"] = best_validation_loss
                    _save_checkpoint(
                        output_dir / "best_val.pt",
                        smolvla,
                        optimizer,
                        scheduler,
                        step,
                        target_steps,
                        gradient_accumulation_steps,
                        loss_history,
                        action_mean,
                        action_std,
                        trainable_params,
                        validation_history,
                        best_validation_loss,
                    )

    train_time = time.time() - start
    print(
        f"\nTraining done in {train_time:.1f}s ({train_time / 60:.1f} min)"
    )

    # ---- Loss curve を PNG 保存 (notebook 表示に相当) ----
    plot_path = _save_loss_curve(loss_history, plot_dir)

    # ---- checkpoint 保存 ----
    # flow_head / vlm_projector / state_proj すべて学習対象なので一緒に保存
    # (stats を保存しないと復元時に正規化が崩れる)
    ckpt_path = output_dir / ("smolvla_ckpt.pt" if legacy_output else "last.pt")
    _save_checkpoint(
        ckpt_path,
        smolvla,
        optimizer,
        scheduler,
        step,
        target_steps,
        gradient_accumulation_steps,
        loss_history,
        action_mean,
        action_std,
        trainable_params,
        validation_history,
        best_validation_loss,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    if plot_path is not None:
        print(f"Saved loss curve to {plot_path}")
    if validation_dataset is not None:
        validation_summary_path = validation_metrics_path.parent / "validation.json"
        validation_summary_path.write_text(
            json.dumps(
                {
                    "best_loss": best_validation_loss,
                    "last_loss": validation_history[-1]["loss"],
                    "evaluated_samples": validation_history[-1][
                        "evaluated_samples"
                    ],
                    "seed": validation_seed,
                    "history": validation_history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if wandb_logger is not None:
        final_window = loss_history[-config.TRAIN_LOG_FREQ_PY :]
        wandb_logger.set_summary(
            status="completed",
            final_step=step,
            final_loss=loss_history[-1],
            final_loss_avg=sum(final_window) / len(final_window),
            best_validation_loss=(
                best_validation_loss if validation_dataset is not None else None
            ),
        )
        wandb_logger.log_artifact(
            ckpt_path,
            name=f"smolvla-{ckpt_path.parent.parent.name}",
            artifact_type="model",
            aliases=["last"],
        )
        best_val_path = output_dir / "best_val.pt"
        if best_val_path.is_file():
            wandb_logger.log_artifact(
                best_val_path,
                name=f"smolvla-{ckpt_path.parent.parent.name}-best-val",
                artifact_type="model",
                aliases=["best-val"],
            )

    return ckpt_path


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_model_checkpoint(smolvla: torch.nn.Module, checkpoint: dict) -> None:
    smolvla.flow_head.load_state_dict(checkpoint["flow_head_state_dict"])
    projector_state = checkpoint["vlm_projector_state_dict"]
    if projector_state:
        smolvla.vlm_projector.load_state_dict(projector_state)
    smolvla.state_proj.load_state_dict(checkpoint["state_proj_state_dict"])
    if "state_mean" in checkpoint and "state_std" in checkpoint:
        smolvla.set_state_stats(checkpoint["state_mean"], checkpoint["state_std"])


def _save_checkpoint(
    path: Path,
    smolvla: torch.nn.Module,
    optimizer,
    scheduler,
    step: int,
    target_steps: int,
    gradient_accumulation_steps: int,
    loss_history: list[float],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    trainable_params: list[torch.nn.Parameter],
    validation_history: list[dict[str, Any]] | None = None,
    best_validation_loss: float = math.inf,
) -> None:
    checkpoint = {
        "flow_head_state_dict": smolvla.flow_head.state_dict(),
        "vlm_projector_state_dict": smolvla.vlm_projector.state_dict(),
        "state_proj_state_dict": smolvla.state_proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "target_steps": target_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "rng_state": _capture_rng_state(),
        "action_mean": action_mean.cpu(),
        "action_std": action_std.cpu(),
        "state_mean": smolvla.state_mean.cpu(),
        "state_std": smolvla.state_std.cpu(),
        "chunk_size": smolvla.chunk_size,
        "action_dim": smolvla.action_dim,
        "state_dim": smolvla.state_dim,
        "hidden_dim": smolvla.hidden_dim,
        "loss_history": loss_history,
        "validation_history": validation_history or [],
        "best_validation_loss": best_validation_loss,
        "trainable_params": int(sum(p.numel() for p in trainable_params)),
    }
    torch.save(checkpoint, path)


def _save_loss_curve(loss_history: list[float], out_dir: Path) -> Path | None:
    """loss curve を PNG に保存。matplotlib が無ければスキップ."""
    if not loss_history:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    from . import config

    x = np.arange(1, len(loss_history) + 1)
    y = np.array(loss_history)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="#8888cc", alpha=0.4, linewidth=0.8, label="loss (raw)")

    window = config.TRAIN_LOG_FREQ_PY
    if len(y) >= window:
        smooth = np.convolve(y, np.ones(window) / window, mode="valid")
        ax.plot(
            x[window - 1 :],
            smooth,
            color="#2244aa",
            linewidth=1.5,
            label=f"moving avg (window={window})",
        )

    ax.set_xlabel("step")
    ax.set_ylabel("Flow Matching loss (MSE)")
    ax.set_title(
        f"Training loss ({len(loss_history)} steps, "
        f"final={loss_history[-1]:.4f}, min={min(loss_history):.4f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    plot_path = out_dir / "loss_curve.png"
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    return plot_path
