"""LiberoEnv での評価と rollout 動画 (Basic Section 10.1-10.2 相当).

- rollout_one_episode : 1 episode を実行し成功可否 + frame 列を返す
- evaluate_pytorch_smolvla : 設定した task / episode を回し結果 JSON を保存
- write_mp4 : frame 列を H.264 mp4 にエンコード (PyAV)

学習データと同じ observation.state 契約と LIBERO action 範囲を維持する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def wilson_interval(
    successes: int, total: int, z_score: float = 1.959963984540054
) -> tuple[float, float]:
    """二項成功率のWilson score intervalを割合(0-1)で返す."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z_score
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_rollouts(rollouts: list[dict]) -> dict:
    """成功率とclip率をrollout長で重み付けして集約する."""
    total = len(rollouts)
    successes = sum(int(rollout["success"]) for rollout in rollouts)
    successful_steps = [
        rollout["steps"] for rollout in rollouts if rollout["success"]
    ]
    predicted_clipped = sum(
        rollout["clipped_action_values"] for rollout in rollouts
    )
    predicted_values = sum(
        rollout["predicted_action_values"] for rollout in rollouts
    )
    executed_clipped = sum(
        rollout["executed_clipped_action_values"] for rollout in rollouts
    )
    executed_values = sum(
        rollout["executed_action_values"] for rollout in rollouts
    )
    predicted_by_dim = np.sum(
        [rollout["predicted_clipped_by_dim"] for rollout in rollouts], axis=0
    )
    executed_by_dim = np.sum(
        [rollout["executed_clipped_by_dim"] for rollout in rollouts], axis=0
    )
    predicted_actions_per_dim = sum(
        rollout["predicted_actions_per_dim"] for rollout in rollouts
    )
    executed_actions_per_dim = sum(
        rollout["executed_actions_per_dim"] for rollout in rollouts
    )
    lower, upper = wilson_interval(successes, total)
    return {
        "success_count": successes,
        "n_rollouts": total,
        "success_rate": successes / max(total, 1),
        "success_rate_wilson95": [lower, upper],
        "mean_steps_success": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "action_clip_rate_predicted": (
            predicted_clipped / max(predicted_values, 1)
        ),
        "action_clip_rate_executed": (
            executed_clipped / max(executed_values, 1)
        ),
        "action_clip_rate_predicted_by_dim": (
            predicted_by_dim / max(predicted_actions_per_dim, 1)
        ).tolist(),
        "action_clip_rate_executed_by_dim": (
            executed_by_dim / max(executed_actions_per_dim, 1)
        ).tolist(),
    }


def evaluation_metrics_for_wandb(results: dict) -> dict[str, float | int | None]:
    """評価結果をtask・action次元を含むW&B scalarへ変換する."""
    summary = results["summary"]
    metrics: dict[str, float | int | None] = {
        "eval/overall_success_rate": summary["success_rate"],
        "eval/success_count": summary["success_count"],
        "eval/mean_steps_success": summary["mean_steps_success"],
        "eval/action_clip_rate_predicted": summary[
            "action_clip_rate_predicted"
        ],
        "eval/action_clip_rate_executed": summary[
            "action_clip_rate_executed"
        ],
    }
    for task in results["per_task"]:
        metrics[f"eval/success_rate_by_task/{task['task_id']}"] = (
            task["success_rate"] / 100.0
        )
    for dimension, rate in enumerate(
        summary["action_clip_rate_predicted_by_dim"]
    ):
        metrics[f"eval/action_clip_rate_predicted_by_dim/{dimension}"] = rate
    for dimension, rate in enumerate(
        summary["action_clip_rate_executed_by_dim"]
    ):
        metrics[f"eval/action_clip_rate_executed_by_dim/{dimension}"] = rate
    return metrics


def observation_state_from_libero(obs: dict) -> np.ndarray:
    """LIBERO observation を学習データと同じ8次元 state に変換する."""
    try:
        robot_state = obs["robot_state"]
        eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
        eef_quat = np.asarray(robot_state["eef"]["quat"], dtype=np.float32)
        gripper_qpos = np.asarray(
            robot_state["gripper"]["qpos"], dtype=np.float32
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "LIBERO env observation から state を構築できません: "
            f"{error}. obs.keys={list(obs.keys())}"
        ) from error

    if (
        eef_pos.shape != (3,)
        or eef_quat.shape != (4,)
        or gripper_qpos.shape != (2,)
    ):
        raise RuntimeError(
            "LIBERO robot_state の shape が不正です: "
            f"eef.pos={eef_pos.shape}, eef.quat={eef_quat.shape}, "
            f"gripper.qpos={gripper_qpos.shape}"
        )

    w = float(np.clip(eef_quat[3], -1.0, 1.0))
    denominator = np.sqrt(max(1.0 - w * w, 0.0))
    eef_axisangle = np.zeros(3, dtype=np.float32)
    if denominator > 1e-10:
        angle = 2.0 * np.arccos(w)
        eef_axisangle = (eef_quat[:3] / denominator * angle).astype(np.float32)

    return np.concatenate(
        [eef_pos, eef_axisangle, gripper_qpos], axis=0
    ).astype(np.float32)


def clip_actions_for_libero(actions: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Action chunk を LIBERO の有効範囲に制限し、clip 値数を返す."""
    clipped = actions.clamp(-1.0, 1.0)
    clipped_count = int(torch.count_nonzero(clipped != actions).item())
    return clipped, clipped_count


def rollout_one_episode(
    model: torch.nn.Module,
    env,
    seed: int,
    denormalize_actions,
    max_steps: int,
    num_flow_steps: int,
    collect_frames: bool = False,
    solver: str = "euler",
    receding_steps: int | None = None,
    policy_seed: int | None = None,
) -> dict:
    """1 episode 実行して結果 (成功可否 + オプションで frame 列) を返す.

    action chunking: chunk 予測 -> receding_steps 個を順に env へ -> 再予測。
    receding_steps=None なら chunk_size (=chunk_buffer 全長) 使い切るまで開ループ
    (notebook 既定)。receding horizon (Stage 1, CHANGES 2026-10-29) では例 4 とし、
    誤差蓄積による action 飽和の悪循環を断つ。
    """
    model.eval()
    device = next(model.flow_head.parameters()).device
    resolved_policy_seed = seed if policy_seed is None else policy_seed
    torch.manual_seed(resolved_policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_policy_seed)

    obs, _info = env.reset(seed=seed)
    task_desc = env.task_description

    chunk_buffer = None  # 現在の chunk (predicted, denormalized numpy)
    chunk_step = 0
    success = False
    total_steps = 0
    clipped_action_values = 0
    predicted_action_values = 0
    executed_clipped_action_values = 0
    executed_action_values = 0
    predicted_clipped_by_dim = None
    executed_clipped_by_dim = None
    predicted_actions_per_dim = 0
    executed_actions_per_dim = 0
    frames = [] if collect_frames else None
    chunk_clip_mask = None

    while total_steps < max_steps:
        img = obs["pixels"]["image"]  # (H, W, C) uint8

        if collect_frames:
            # LIBERO は agentview を上下反転して返すので、動画用は向きを戻す
            frames.append(img[::-1, ::-1].copy())

        state_vec = observation_state_from_libero(obs)
        state_tensor = (
            torch.from_numpy(state_vec).unsqueeze(0).to(device)
        )  # (1, 8)

        # receding_steps 個実行したら (または初期) 新しい chunk を予測
        horizon = (
            receding_steps
            if receding_steps is not None
            else chunk_buffer.shape[0] if chunk_buffer is not None else 0
        )
        if chunk_buffer is None or chunk_step >= horizon:
            with torch.no_grad():
                pred_normalized = model.sample_actions(
                    [img], [task_desc], state_tensor,
                    num_steps=num_flow_steps,
                    solver=solver,
                )  # (1, chunk_size, 7)
            pred_denorm_raw = denormalize_actions(pred_normalized[0])
            pred_denorm, clipped_count = clip_actions_for_libero(pred_denorm_raw)
            chunk_clip_mask = pred_denorm != pred_denorm_raw
            clipped_action_values += clipped_count
            predicted_action_values += pred_denorm.numel()
            clipped_by_dim = chunk_clip_mask.sum(dim=0).to(torch.int64).cpu()
            if predicted_clipped_by_dim is None:
                predicted_clipped_by_dim = torch.zeros_like(clipped_by_dim)
                executed_clipped_by_dim = torch.zeros_like(clipped_by_dim)
            predicted_clipped_by_dim += clipped_by_dim
            predicted_actions_per_dim += pred_denorm.shape[0]
            chunk_buffer = pred_denorm.cpu().numpy()
            chunk_step = 0

        action = chunk_buffer[chunk_step]
        action_clip_mask = chunk_clip_mask[chunk_step]
        executed_clipped_action_values += int(action_clip_mask.sum().item())
        executed_action_values += int(action_clip_mask.numel())
        executed_clipped_by_dim += action_clip_mask.to(torch.int64).cpu()
        executed_actions_per_dim += 1
        chunk_step += 1
        total_steps += 1

        obs, _reward, terminated, _truncated, info = env.step(action)

        if terminated or _truncated:
            success = bool(info.get("success", terminated))
            break

    predicted_by_dim = (
        predicted_clipped_by_dim.float() / max(predicted_actions_per_dim, 1)
    ).tolist()
    executed_by_dim = (
        executed_clipped_by_dim.float() / max(executed_actions_per_dim, 1)
    ).tolist()
    return {
        "success": success,
        "steps": total_steps,
        "clipped_action_values": clipped_action_values,
        "predicted_action_values": predicted_action_values,
        "executed_clipped_action_values": executed_clipped_action_values,
        "executed_action_values": executed_action_values,
        "predicted_clip_rate": (
            clipped_action_values / max(predicted_action_values, 1)
        ),
        "executed_clip_rate": (
            executed_clipped_action_values / max(executed_action_values, 1)
        ),
        "predicted_clip_rate_by_dim": predicted_by_dim,
        "executed_clip_rate_by_dim": executed_by_dim,
        "predicted_clipped_by_dim": predicted_clipped_by_dim.tolist(),
        "executed_clipped_by_dim": executed_clipped_by_dim.tolist(),
        "predicted_actions_per_dim": predicted_actions_per_dim,
        "executed_actions_per_dim": executed_actions_per_dim,
        "env_seed": seed,
        "policy_seed": resolved_policy_seed,
        "frames": frames,
    }


def evaluate_pytorch_smolvla(
    model: torch.nn.Module,
    denormalize_actions,
    *,
    task_ids: list[int] | None = None,
    episodes_per_task: int | None = None,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    results_path: Path | None = None,
    video_dir: Path | None = None,
    wandb_logger=None,
) -> dict:
    """Basic Section 10.1: 設定した task / episode の結果を保存する."""
    from tqdm.auto import tqdm

    from . import config

    task_ids = list(config.PY_EVAL_TASK_IDS if task_ids is None else task_ids)
    episodes_per_task = (
        config.PY_EVAL_EPISODES_PER_TASK
        if episodes_per_task is None
        else episodes_per_task
    )
    env_seed = config.EVAL_SEED if env_seed is None else env_seed
    policy_seed = (
        getattr(config, "PY_EVAL_POLICY_SEED", config.EVAL_SEED + 10_000)
        if policy_seed is None
        else policy_seed
    )
    results_path = results_path or config.PY_EVAL_RESULTS_PATH
    video_dir = video_dir or config.PY_VIDEO_DIR
    if not task_ids or any(task_id not in range(10) for task_id in task_ids):
        raise ValueError("task_ids must contain LIBERO-Spatial task IDs 0-9")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")

    video_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Evaluating custom SmolVLA on {len(task_ids)} tasks x "
        f"{episodes_per_task} episodes...",
        flush=True,
    )

    # LIBERO benchmark suite を取得 (fork 側が優先ロードされている前提)
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark as libero_benchmark

    suite = libero_benchmark.get_benchmark_dict()["libero_spatial"]()

    results: dict = {
        "config": {
            "task_ids": task_ids,
            "episodes_per_task": episodes_per_task,
            "env_seed": env_seed,
            "policy_seed": policy_seed,
            "max_steps": config.PY_EVAL_MAX_STEPS,
            "num_flow_steps": config.PY_EVAL_NUM_FLOW_STEPS,
            "flow_solver": config.PY_EVAL_FLOW_SOLVER,
            "receding_steps": config.PY_EVAL_RECEDING_STEPS,
        },
        "per_task": [],
        "overall_success": None,
        "summary": None,
        "rollouts": [],
    }

    expected = len(task_ids) * episodes_per_task
    outer = tqdm(total=expected, desc="rollout", unit="ep")

    for task_id in task_ids:
        task_rollouts = []
        for ep_i in range(episodes_per_task):
            # env を毎 episode 新規作成 (状態を持ち越さない)
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name="libero_spatial",
                episode_index=ep_i,
                n_envs=episodes_per_task,
                observation_height=256,
                observation_width=256,
                camera_name=["agentview_image"],
                camera_name_mapping={"agentview_image": "image"},
                control_mode="relative",
                is_libero_plus=True,
                obs_type="pixels_agent_pos",  # obs に "robot_state" を含める
            )
            # 最初の episode だけ frame を集めて動画にする (メモリ節約)
            collect = task_id == task_ids[0] and ep_i == 0
            result = rollout_one_episode(
                model,
                env,
                seed=env_seed + task_id * 100 + ep_i,
                policy_seed=policy_seed + task_id * 100 + ep_i,
                denormalize_actions=denormalize_actions,
                max_steps=config.PY_EVAL_MAX_STEPS,
                num_flow_steps=config.PY_EVAL_NUM_FLOW_STEPS,
                collect_frames=collect,
                solver=config.PY_EVAL_FLOW_SOLVER,
                receding_steps=getattr(config, "PY_EVAL_RECEDING_STEPS", None),
            )
            env.close()

            rollout = {"task_id": task_id, "episode": ep_i, **result}
            task_rollouts.append(rollout)
            results["rollouts"].append(rollout)
            outer.update(1)
            outer.set_postfix_str(
                f"task {task_id} ep {ep_i}: "
                f"{'SUCCESS' if result['success'] else 'FAIL'} "
                f"({result['steps']} steps)"
            )

        task_summary = summarize_rollouts(task_rollouts)
        results["per_task"].append(
            {
                "task_id": task_id,
                "task": config.SPATIAL_TASK_NAMES[task_id],
                "success_rate": 100.0 * task_summary["success_rate"],
                "n_success": task_summary["success_count"],
                "n_episodes": task_summary["n_rollouts"],
                "success_rate_wilson95": [
                    100.0 * value
                    for value in task_summary["success_rate_wilson95"]
                ],
                "mean_steps_success": task_summary["mean_steps_success"],
                "action_clip_rate_predicted": task_summary[
                    "action_clip_rate_predicted"
                ],
                "action_clip_rate_executed": task_summary[
                    "action_clip_rate_executed"
                ],
            }
        )

    outer.close()
    summary = summarize_rollouts(results["rollouts"])
    results["summary"] = summary
    results["overall_success"] = 100.0 * summary["success_rate"]
    results["trainable_params"] = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    print(f"\n=== Custom SmolVLA overall: {results['overall_success']:.1f}% ===")

    # ---- 10.2: frames を持つ rollout を mp4 化 ----
    saved = 0
    for rollout in results["rollouts"]:
        frames = rollout["frames"]
        if not frames:
            continue
        mp4_path = (
            video_dir
            / f"task{rollout['task_id']}_ep{rollout['episode']}.mp4"
        )
        write_mp4(frames, mp4_path, fps=20)
        label = "SUCCESS" if rollout["success"] else "FAIL"
        print(
            f"\n▶ task {rollout['task_id']} ep {rollout['episode']}  "
            f"({label}, {rollout['steps']} steps)"
        )
        print(f"  mp4: {mp4_path}")
        saved += 1
    if saved:
        print(f"Saved {saved} mp4(s) to {video_dir}")

    # ---- 結果を JSON 保存 (frames は除く) + Drive 相当キャッシュへ ----
    serializable = {
        "per_task": results["per_task"],
        "overall_success": results["overall_success"],
        "summary": results["summary"],
        "config": results["config"],
        "trainable_params": results["trainable_params"],
        "rollouts": [
            {k: v for k, v in r.items() if k != "frames"}
            for r in results["rollouts"]
        ],
    }
    results_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if results_path == config.PY_EVAL_RESULTS_PATH:
        cache_path = config.CACHE_DIR / config.PY_EVAL_RESULTS_PATH.name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if wandb_logger is not None:
        wandb_logger.log(evaluation_metrics_for_wandb(results))
        wandb_logger.set_summary(
            status="completed",
            overall_success_rate=summary["success_rate"],
            success_count=summary["success_count"],
            action_clip_rate_predicted=summary["action_clip_rate_predicted"],
            action_clip_rate_executed=summary["action_clip_rate_executed"],
        )
        wandb_logger.log_artifact(
            results_path,
            name=f"eval-{results_path.parent.parent.name}",
            artifact_type="evaluation",
            aliases=["latest"],
        )
    print(f"Saved eval results to {results_path}")

    return results


def write_mp4(frames: list, path: Path, fps: int = 20) -> None:
    """RGB uint8 の frame 列を H.264 mp4 に書き出す (Basic Section 10.2)."""
    import av

    if not frames:
        raise RuntimeError("frames が空")

    h, w = frames[0].shape[:2]
    # H.264 の yuv420p は幅・高さが偶数である必要がある
    h -= h % 2
    w -= w % 2

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}

        for frame_array in frames:
            frame_array = frame_array[:h, :w]
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
