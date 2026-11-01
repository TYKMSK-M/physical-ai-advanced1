"""LIBERO-Spatial 10タスクからepisodeを等間隔選択 (Section 7 相当).

学習用サブセットを再現可能に構築する。
データセット側の task 名は表記揺れがありうるので正規化キーで突合し、
各taskのepisode indexから指定件数を等間隔に選び、episode単位で分割する。
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from . import config
from .hf_fetch import run_hf_with_retry


def normalize_task_name(value: str) -> str:
    """小文字化・記号除去・空白正規化で突合用キーを作る."""
    value = value.lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def task_name_from_cell(value) -> str:
    """メタデータの tasks 列は str/list どちらもあり得るので単一 str に落とす."""
    if isinstance(value, str):
        return value

    try:
        if len(value) > 0:
            return str(value[0])
    except TypeError:
        pass

    return str(value)


def choose_evenly_spaced(episode_indices: list[int], count: int) -> list[int]:
    """episode 群から等間隔に count 個を選ぶ (先頭と末尾を必ず含む)."""
    if len(episode_indices) < count:
        raise ValueError(
            f"Task には episode が {len(episode_indices)} 件しかないため、"
            f"要求数 {count} を満たせません。"
        )
    positions = [
        round(index * (len(episode_indices) - 1) / (count - 1))
        for index in range(count)
    ]
    return [episode_indices[position] for position in positions]


def _select_episode_indices_by_task(
    episodes_per_task: int | None = config.TRAIN_EPISODES_PER_TASK,
) -> dict[str, list[int]]:
    """Spatial各taskから選んだepisode indexをtask別に返す."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    # データセットのメタデータを取得 (429 リトライ付き)
    dataset_metadata = run_hf_with_retry(
        lambda: LeRobotDatasetMetadata(
            config.DATASET_REPO,
            revision=config.DATASET_REVISION,
        )
    )

    # 「task 名 -> その task に属する episode index リスト」を構築
    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    for episode_index, task_cell in enumerate(dataset_metadata.episodes["tasks"]):
        task_to_episodes[task_name_from_cell(task_cell)].append(int(episode_index))

    # 正規化キーから実際のタスク名 (表記揺れ含む) を引くための辞書
    available_by_normalized = {
        normalize_task_name(task_name): task_name
        for task_name in task_to_episodes
    }

    if episodes_per_task is not None and episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")

    # Spatialの10タスク各々から指定件数を選ぶ
    selected_by_task: dict[str, list[int]] = {}
    for task_name in config.SPATIAL_TASK_NAMES:
        actual_task = available_by_normalized.get(normalize_task_name(task_name))
        if actual_task is None:
            raise RuntimeError(f"Spatial task not found: {task_name}")
        available_episodes = task_to_episodes[actual_task]
        selected_by_task[actual_task] = (
            list(available_episodes)
            if episodes_per_task is None
            else choose_evenly_spaced(available_episodes, episodes_per_task)
        )

    return selected_by_task


def select_episode_indices(
    episodes_per_task: int | None = config.TRAIN_EPISODES_PER_TASK,
) -> list[int]:
    """Spatial 10 task分のepisode indexソート済みリストを返す."""
    selected_by_task = _select_episode_indices_by_task(episodes_per_task)

    episode_indices = sorted(
        episode_index
        for episode_indices in selected_by_task.values()
        for episode_index in episode_indices
    )

    expected = (
        len(episode_indices)
        if episodes_per_task is None
        else len(config.SPATIAL_TASK_NAMES) * episodes_per_task
    )
    if len(episode_indices) != expected:
        raise RuntimeError("Episode selection failed.")

    if episodes_per_task is None:
        print(
            f"Training data: all {expected} episodes from "
            f"{len(config.SPATIAL_TASK_NAMES)} tasks"
        )
    else:
        print(
            f"Training data: {len(config.SPATIAL_TASK_NAMES)} tasks × "
            f"{episodes_per_task} episodes = {expected} episodes"
        )
    return episode_indices


def select_train_validation_episode_indices(
    train_episodes_per_task: int = 4,
    validation_episodes_per_task: int = 1,
    seed: int = config.SEED,
) -> tuple[list[int], list[int]]:
    """taskごとにepisodeを分け、未来action chunkのtrain/validation漏洩を防ぐ."""
    if train_episodes_per_task <= 0 or validation_episodes_per_task <= 0:
        raise ValueError("train and validation episodes per task must be positive")

    selected_by_task = _select_episode_indices_by_task(
        train_episodes_per_task + validation_episodes_per_task
    )
    generator = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for episode_indices in selected_by_task.values():
        validation = set(
            generator.sample(episode_indices, validation_episodes_per_task)
        )
        validation_indices.extend(validation)
        train_indices.extend(
            episode_index
            for episode_index in episode_indices
            if episode_index not in validation
        )

    train_indices.sort()
    validation_indices.sort()
    print(
        "Training/validation data: "
        f"{len(train_indices)} / {len(validation_indices)} episodes "
        f"({train_episodes_per_task} / {validation_episodes_per_task} per task, "
        "episode-disjoint)"
    )
    return train_indices, validation_indices
