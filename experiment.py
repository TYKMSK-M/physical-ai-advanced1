"""日時付き実験ディレクトリと再現情報を管理する."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "run"


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status and status != "unknown"),
    }


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    root: Path
    checkpoints: Path
    metrics: Path
    logs: Path
    plots: Path
    videos: Path
    wandb_dir: Path
    manifest_path: Path

    def update_manifest(self, **updates: Any) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(updates)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def create_experiment_run(
    workdir: Path,
    stage: str,
    name: str,
    config_values: dict[str, Any],
    command: list[str] | None = None,
) -> ExperimentRun:
    """日時付きrunを作成し、設定・Git・コマンドを記録する."""
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_slugify(stage)}_{_slugify(name)}"
    root = workdir / "experiments" / run_id
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")

    directories = {
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "logs": root / "logs",
        "plots": root / "plots",
        "videos": root / "videos",
        "wandb_dir": root / "wandb",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=False)

    project_root = Path(__file__).resolve().parents[3]
    git = _git_metadata(project_root)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": stage,
                "name": name,
                "status": "running",
                "started_at": now.isoformat(),
                "git": git,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(config_values, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (root / "git_commit.txt").write_text(
        f"commit={git['commit']}\nbranch={git['branch']}\ndirty={git['dirty']}\n",
        encoding="utf-8",
    )
    actual_command = command if command is not None else sys.argv
    (root / "command.txt").write_text(
        shlex.join(actual_command) + "\n",
        encoding="utf-8",
    )

    return ExperimentRun(
        run_id=run_id,
        root=root,
        manifest_path=manifest_path,
        **directories,
    )
