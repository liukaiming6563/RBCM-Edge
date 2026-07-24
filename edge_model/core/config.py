"""Configuration helpers for edge model experiments.

The edge model scripts use YAML for persistent experiment settings and argparse
for command-line overrides. This keeps local PyCharm demos and server training
jobs on the same code path.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    """Write a YAML config snapshot with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a config dictionary and return a new dictionary."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve paths relative to the configured project root."""
    value = Path(value)
    if value.is_absolute():
        return value
    root = Path(config.get("paths", {}).get("project_root", "."))
    return (root / value).resolve()
