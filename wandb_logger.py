"""W&Bをoptionalに扱う小さな学習ログラッパ."""

from __future__ import annotations

import json
import netrc
import os
from pathlib import Path
from typing import Any


class WandbLogger:
    def __init__(self, run=None, mode: str = "disabled"):
        self._run = run
        self.mode = mode

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(values, step=step)

    def set_summary(self, **values: Any) -> None:
        if self._run is not None:
            for key, value in values.items():
                self._run.summary[key] = value

    def log_artifact(
        self,
        path: Path,
        name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
    ) -> None:
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def _wandb_credentials_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return credentials is not None


def init_wandb(
    experiment,
    mode: str,
    project: str,
    entity: str | None,
    config_values: dict[str, Any],
) -> WandbLogger:
    """W&B runを開始する。autoは認証なしならofflineへ切り替える."""
    if mode == "disabled":
        _write_metadata(experiment, mode="disabled", run=None)
        return WandbLogger(mode="disabled")

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "online" if _wandb_credentials_available() else "offline"

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B が未導入です: uv pip install -e '.[basic]' を再実行してください"
        ) from error

    run = wandb.init(
        project=project,
        entity=entity,
        name=experiment.run_id,
        dir=str(experiment.wandb_dir),
        mode=resolved_mode,
        config=config_values,
        tags=["basic", "smolvla", "flow-matching"],
    )
    _write_metadata(experiment, mode=resolved_mode, run=run)
    print(
        f"[W&B] mode={resolved_mode} run={run.id} "
        f"url={getattr(run, 'url', None) or '(offline: sync required)'}",
        flush=True,
    )
    return WandbLogger(run=run, mode=resolved_mode)


def _write_metadata(experiment, mode: str, run) -> None:
    metadata = {
        "mode": mode,
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "run_id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
    }
    (experiment.root / "wandb_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
