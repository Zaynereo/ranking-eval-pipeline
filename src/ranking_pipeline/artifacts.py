"""Run output: provenance, metrics JSON, configs (no untrusted pickle inputs)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PIPELINE_VERSION = "0.1.0"


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_provenance(
    out_dir: Path,
    *,
    seed: int,
    command: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "synthetic_data_label": "SYNTHETIC_DEMO_DATA",
        "pipeline_version": PIPELINE_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "seed": seed,
        "command": command,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def write_config(out_dir: Path, config: dict[str, Any]) -> Path:
    path = out_dir / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def write_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        return obj

    path.write_text(json.dumps(_sanitize(metrics), indent=2), encoding="utf-8")


def write_metrics_text(metrics: dict[str, Any], path: Path) -> None:
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
