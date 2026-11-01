"""lerobot-eval による評価 (Section 8.5-8.6 / Section 9 相当).

- run_evaluation     : 1 モデルの Spatial 評価 (10 task x 1 ep) を回して
                       eval_info.json を返す
- compare_spatial    : 追加学習前後の成功率比較表 (CSV) を作る
- run_advanced_eval  : 4 suite x 3 task x 1 ep = 12 rollout / モデルの広域評価
- compare_advanced   : suite ごとの比較表 (CSV) を作る
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path

from . import config
from .compat import display_df

# Section 5 で仕込んだ "EVAL_PROGRESS task=x/y episode=a/b" 行を拾う正規表現
PROGRESS_PATTERN = re.compile(
    r"^EVAL_PROGRESS task=(\d+)/(\d+) episode=(\d+)/(\d+)$"
)


def build_eval_command(
    policy_path: Path,
    output_dir: Path,
    *,
    suite: str = "libero_spatial",
    task_ids: list[int] | None = None,
    n_episodes: int | None = None,
) -> list[str]:
    """lerobot-eval CLI 引数を組み立てる。ベース/マージ済みで共通の設定。"""
    if task_ids is None:
        task_ids = config.EVAL_TASK_IDS
    if n_episodes is None:
        n_episodes = config.EVAL_EPISODES_PER_TASK

    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        "--env.is_libero_plus=true",
        f"--env.task={suite}",
        "--env.task_ids=" + json.dumps(task_ids, separators=(",", ":")),
        "--env.camera_name_mapping="
        + json.dumps(config.EVAL_CAMERA_MAPPING, separators=(",", ":")),
        # 観測画像は 256x256 (SmolVLA の入力仕様に合わせる)
        "--env.observation_height=256",
        "--env.observation_width=256",
        # relative action 空間 (LIBERO-plus のデフォルト)
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",  # 逐次実行 (VRAM 節約)
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        "--eval.use_async_envs=false",
        "--eval.recording=false",
        f"--seed={config.EVAL_SEED}",
        f"--output_dir={output_dir}",
    ]


def run_evaluation(
    policy_path: Path,
    output_dir: Path,
    label: str,
    *,
    suite: str = "libero_spatial",
    task_ids: list[int] | None = None,
    n_episodes: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """1 モデル分の評価を回して eval_info.json を dict で返す."""
    if task_ids is None:
        task_ids = config.EVAL_TASK_IDS
    if n_episodes is None:
        n_episodes = config.EVAL_EPISODES_PER_TASK

    shutil.rmtree(output_dir, ignore_errors=True)

    process = subprocess.Popen(
        build_eval_command(
            policy_path,
            output_dir,
            suite=suite,
            task_ids=task_ids,
            n_episodes=n_episodes,
        ),
        cwd=config.LEROBOT_DIR,
        env=config.subprocess_env(**(extra_env or {})),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    # 失敗時ダンプ用の直近ログ
    recent_lines: deque[str] = deque(maxlen=120)

    from tqdm.auto import tqdm

    total_rollouts = len(task_ids) * n_episodes
    pbar = tqdm(total=total_rollouts, desc=f"eval[{label}]", unit="ep")

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.replace("\r", "").strip()
        if not line:
            continue
        recent_lines.append(line)
        match = PROGRESS_PATTERN.match(line)
        if match:
            ti, tt, ei, et = match.groups()
            pbar.update(1)
            pbar.set_postfix_str(f"task {ti}/{tt} ep {ei}/{et}")

    pbar.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("\n".join(recent_lines))

    # 評価結果 JSON を読んで返す
    result_path = output_dir / "eval_info.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    return json.loads(result_path.read_text(encoding="utf-8"))


def per_task_success(eval_info: dict) -> dict[int, float]:
    """eval_info.json の per_task から task_id -> 成功率(%) を作る."""
    result: dict[int, float] = {}
    for task_info in eval_info["per_task"]:
        task_id = int(task_info["task_id"])
        successes = task_info["metrics"]["successes"]
        # bool リスト -> True の割合を %
        result[task_id] = (
            100.0 * sum(bool(value) for value in successes) / len(successes)
        )
    return result


def compare_spatial(
    base_eval_info: dict,
    finetuned_eval_info: dict,
) -> None:
    """タスク単位の成功率と Δ (pp) を表にまとめて CSV に保存する."""
    import pandas as pd

    base_per_task = per_task_success(base_eval_info)
    finetuned_per_task = per_task_success(finetuned_eval_info)

    rows = []
    for task_id in config.EVAL_TASK_IDS:
        base_score = base_per_task[task_id]
        finetuned_score = finetuned_per_task[task_id]
        rows.append(
            {
                "Task ID": task_id,
                "Task": config.SPATIAL_TASK_NAMES[task_id],
                "Base (%)": base_score,
                "Spatial LoRA (%)": finetuned_score,
                # Δ (pp) は「追加学習後 - 追加学習前」のパーセンテージポイント差
                "Δ (pp)": finetuned_score - base_score,
            }
        )

    # Overall 行 (LIBERO-Spatial 全体の pc_success)
    base_overall = float(base_eval_info["overall"]["pc_success"])
    finetuned_overall = float(finetuned_eval_info["overall"]["pc_success"])
    rows.append(
        {
            "Task ID": "Overall",
            "Task": "LIBERO-Spatial",
            "Base (%)": base_overall,
            "Spatial LoRA (%)": finetuned_overall,
            "Δ (pp)": finetuned_overall - base_overall,
        }
    )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(config.COMPARISON_CSV_PATH, index=False)

    display_df(comparison_df)
    print(
        f"\nSaved: {config.COMPARISON_CSV_PATH}"
    )
    print(
        f"Overall: {base_overall:.1f}% -> {finetuned_overall:.1f}% "
        f"({finetuned_overall - base_overall:+.1f} pp)"
    )


def run_advanced_eval(policy_path: Path, out_root: Path, label: str) -> dict:
    """1 モデルについて全 suite を回して suite -> eval_info の dict を返す."""
    shutil.rmtree(out_root, ignore_errors=True)
    suite_results: dict[str, dict] = {}

    for suite_name, task_ids in config.ADVANCED_SUITES.items():
        sub_dir = out_root / suite_name
        print(
            f"\n[{label}] {suite_name} "
            f"({len(task_ids)} tasks x {config.ADVANCED_EPISODES_PER_TASK} ep)",
            flush=True,
        )
        info = run_evaluation(
            policy_path,
            sub_dir,
            f"{label} {suite_name}",
            suite=suite_name,
            task_ids=task_ids,
            n_episodes=config.ADVANCED_EPISODES_PER_TASK,
            extra_env={"LEROBOT_MAX_RECORDED": "0"},
        )
        suite_results[suite_name] = info

    return suite_results


def overall_across_suites(results: dict[str, dict]) -> float:
    """全 rollout 合計での成功率 (%) を返す."""
    total_success = 0.0
    total_rollouts = 0
    for info in results.values():
        for task in info["per_task"]:
            successes = task["metrics"]["successes"]
            total_success += sum(bool(v) for v in successes)
            total_rollouts += len(successes)
    return 100.0 * total_success / total_rollouts if total_rollouts else 0.0


def compare_advanced(base_results: dict, finetuned_results: dict) -> None:
    """suite ごとの集計テーブルを CSV に保存する."""
    import pandas as pd

    rows = []
    for suite_name in config.ADVANCED_SUITES:
        base = base_results[suite_name]["overall"]["pc_success"]
        fine = finetuned_results[suite_name]["overall"]["pc_success"]
        rows.append(
            {
                "Suite": suite_name,
                "Tasks": len(config.ADVANCED_SUITES[suite_name]),
                "Base (%)": base,
                "Spatial LoRA (%)": fine,
                "Δ (pp)": fine - base,
            }
        )

    base_overall = overall_across_suites(base_results)
    finetuned_overall = overall_across_suites(finetuned_results)
    rows.append(
        {
            "Suite": "Overall (12 rollouts)",
            "Tasks": sum(len(v) for v in config.ADVANCED_SUITES.values()),
            "Base (%)": base_overall,
            "Spatial LoRA (%)": finetuned_overall,
            "Δ (pp)": finetuned_overall - base_overall,
        }
    )

    adv_df = pd.DataFrame(rows)
    adv_csv = config.ADVANCED_OUT_DIR / "advanced_comparison.csv"
    adv_csv.parent.mkdir(parents=True, exist_ok=True)
    adv_df.to_csv(adv_csv, index=False)

    display_df(adv_df)
    print(f"\nSaved: {adv_csv}")
    print(
        f"Overall (12 rollouts): {base_overall:.1f}% -> "
        f"{finetuned_overall:.1f}% "
        f"({finetuned_overall - base_overall:+.1f} pp)"
    )
