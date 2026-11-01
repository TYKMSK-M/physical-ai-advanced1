"""LIBERO データセット読み込みと action 正規化 (Basic Section 9.1-9.2 相当).

- load_train_dataset : chunk_size ぶんの future action 付き LeRobotDataset
- compute_action_stats / make_normalizers : 全 action の mean/std で正規化
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_train_dataset(episode_indices: list[int], chunk_size: int):
    """50 episode から chunk_size 個の未来 action を並べて返す Dataset を作る."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from . import config

    # LIBERO の記録レート = 20 Hz。delta_timestamps は秒単位のオフセットなので
    # i-th 未来 step は i / fps 秒後。
    dataset_fps = 20.0
    delta_timestamps = {
        "action": [i / dataset_fps for i in range(chunk_size)],
    }

    print(f"Loading LIBERO dataset with {len(episode_indices)} episodes...")

    # 初回ロードは HF から動画をひっぱるため、この間だけ progress bar を有効化
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    enable_progress_bars()
    try:
        train_dataset = LeRobotDataset(
            config.DATASET_REPO,
            revision=config.DATASET_REVISION,
            episodes=episode_indices,
            delta_timestamps=delta_timestamps,
        )
    finally:
        disable_progress_bars()

    sample = train_dataset[0]
    print(f"training samples total: {len(train_dataset)}")
    state0 = sample["observation.state"]
    print(f"observation.state sample: shape={tuple(state0.shape)}")

    return train_dataset


def compute_action_stats(
    train_dataset, action_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """全 action の次元別 mean / std を GPU 上 (float32) で返す."""
    print("Collecting action statistics...", flush=True)
    all_actions = torch.stack(
        [train_dataset[i]["action"] for i in range(len(train_dataset))]
    ).view(-1, action_dim)  # (N, 7)

    action_mean = all_actions.mean(dim=0)                    # (7,)
    action_std = all_actions.std(dim=0).clamp_min(1e-3)      # (7,)

    action_mean = action_mean.to("cuda", dtype=torch.float32)
    action_std = action_std.to("cuda", dtype=torch.float32)

    print(f"action_mean: {action_mean.cpu().numpy().round(3)}")
    print(f"action_std : {action_std.cpu().numpy().round(3)}")
    return action_mean, action_std


def compute_state_stats(
    train_dataset, state_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """全 observation.state の次元別 mean / std を GPU 上 (float32) で返す.

    Stage 2 (CHANGES 2026-10-29): LIBERO の eef 位置はスケールが大きく、
    state_proj 前に正規化しないと条件トークンが action 側とアンバランス。
    """
    print("Collecting state statistics...", flush=True)
    all_states = torch.stack(
        [train_dataset[i]["observation.state"] for i in range(len(train_dataset))]
    ).view(-1, state_dim)  # (N, 8)

    state_mean = all_states.mean(dim=0)                   # (8,)
    state_std = all_states.std(dim=0).clamp_min(1e-3)     # (8,)

    state_mean = state_mean.to("cuda", dtype=torch.float32)
    state_std = state_std.to("cuda", dtype=torch.float32)

    print(f"state_mean: {state_mean.cpu().numpy().round(3)}")
    print(f"state_std : {state_std.cpu().numpy().round(3)}")
    return state_mean, state_std


def make_normalizers(action_mean: torch.Tensor, action_std: torch.Tensor):
    """(normalize, denormalize) の関数ペアを作る.

    学習前に (a - mean) / std、推論後に a * std + mean で元スケールへ戻す。
    """

    def normalize_actions(a: torch.Tensor) -> torch.Tensor:
        return (a - action_mean) / action_std

    def denormalize_actions(a: torch.Tensor) -> torch.Tensor:
        return a * action_std + action_mean

    return normalize_actions, denormalize_actions


def save_action_stats(
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    path: Path,
) -> None:
    """rollout 側で denormalize できるよう stats を磁盘へ保存する."""
    torch.save(
        {"action_mean": action_mean.cpu(), "action_std": action_std.cpu()},
        path,
    )


def load_action_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cuda")
    return (
        data["action_mean"].to("cuda", dtype=torch.float32),
        data["action_std"].to("cuda", dtype=torch.float32),
    )
