"""Checkpoint I/O helpers with PyTorch-version compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, *, map_location: Any = None) -> dict:
    """Load a training checkpoint that may contain optimizer, scaler, and RNG state."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
