"""rollout 動画の録画 (Section 8.8 相当).

lerobot_eval.py の録画本数は setup_libero で LEROBOT_MAX_RECORDED 環境変数
から制御できるようにしてある。ここでは 1 タスク x N 本の rollout を
recording=true 相当の設定で実行し、できた mp4 を列挙する。

Colab 版の notebook 埋め込み表示は compat.show_video に委譲
(ローカルではパス表示、Colab では Video 埋め込み)。
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .compat import show_video
from .evaluate import run_evaluation


def record_rollout_videos(
    policy_dir: Path | None = None,
    *,
    task_id: int | None = None,
    n_episodes: int | None = None,
) -> list[Path]:
    if policy_dir is None:
        policy_dir = config.MERGED_MODEL_DIR
    if task_id is None:
        task_id = config.VIDEO_TASK_ID
    if n_episodes is None:
        n_episodes = config.VIDEO_N_EPISODES

    print(
        f"Recording {n_episodes} rollout(s) on task {task_id}...",
        flush=True,
    )
    run_evaluation(
        policy_dir,
        config.VIDEO_OUT_DIR,
        "video",
        task_ids=[task_id],
        n_episodes=n_episodes,
        # recording=false + LEROBOT_MAX_RECORDED>0 の組み合わせで、
        # eval_script が videos_dir に mp4 を吐き出すルートに乗る
        # (recording=true は HF dataset 形式の完全記録用で mp4 とは排他)。
        extra_env={"LEROBOT_MAX_RECORDED": str(n_episodes)},
    )

    mp4_files = sorted(config.VIDEO_OUT_DIR.rglob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(f"mp4 が {config.VIDEO_OUT_DIR} に見つかりません。")

    print(f"\nFound {len(mp4_files)} mp4 file(s):")
    for mp4 in mp4_files:
        print(f"  {mp4.relative_to(config.VIDEO_OUT_DIR)}")
        show_video(mp4)

    return mp4_files
