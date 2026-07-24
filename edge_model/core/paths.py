"""Output path creation for edge detection runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EdgeRunPaths:
    """Structured output directories for one experiment."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    predictions: Path
    gate_heatmaps: Path
    visualizations: Path


def make_run_paths(output_root: str | Path, experiment_name: str) -> EdgeRunPaths:
    """Create and return all output folders for an experiment."""
    root = Path(output_root) / experiment_name
    paths = EdgeRunPaths(
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        metrics=root / "metrics",
        predictions=root / "predictions",
        gate_heatmaps=root / "gate_heatmaps",
        visualizations=root / "visualizations",
    )
    for path in paths.__dict__.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths
