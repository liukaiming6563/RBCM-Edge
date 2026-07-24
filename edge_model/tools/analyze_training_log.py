"""Create compact diagnostic plots from the CSV written by training."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze_training_log(
    log_csv: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Plot train/validation losses and validation edge metrics."""
    log_path = Path(log_csv)
    if not log_path.exists():
        raise FileNotFoundError(f"Cannot find training log: {log_path}")

    plot_dir = Path(output_dir) if output_dir is not None else log_path.parents[1] / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(log_path)
    if "epoch" not in log.columns or "split" not in log.columns:
        raise ValueError("train_log.csv must contain 'epoch' and 'split' columns")

    numeric_columns = [
        "total",
        "loss",
        "final_bce",
        "final_dice",
        "local",
        "side",
        "context",
        "ODS",
        "OIS",
        "AP",
    ]
    for column in numeric_columns:
        if column in log:
            log[column] = pd.to_numeric(log[column], errors="coerce")

    outputs = {
        "loss_curves": plot_dir / "loss_curves.png",
        "validation_metrics": plot_dir / "validation_metrics.png",
        "training_summary": plot_dir / "training_summary.png",
    }
    _plot_losses(log, outputs["loss_curves"])
    _plot_metrics(log, outputs["validation_metrics"])
    _plot_summary(log, outputs["training_summary"])
    return outputs


def _subset(log: pd.DataFrame, split: str) -> pd.DataFrame:
    return log[log["split"].astype(str).str.lower().eq(split)].copy()


def _has_values(frame: pd.DataFrame, column: str) -> bool:
    return column in frame and frame[column].notna().any()


def _plot_losses(log: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for split in ("train", "val"):
        frame = _subset(log, split)
        candidates = ("total", "loss") if split == "train" else ("loss", "total")
        column = next((name for name in candidates if _has_values(frame, name)), None)
        if column is not None:
            ax.plot(frame["epoch"], frame[column], marker="o", linewidth=1.8, label=f"{split} {column}")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_metrics(log: pd.DataFrame, output: Path) -> None:
    val = _subset(log, "val")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for metric in ("ODS", "OIS", "AP"):
        if _has_values(val, metric):
            ax.plot(val["epoch"], val[metric], marker="o", linewidth=1.8, label=metric)
    ax.set(xlabel="Epoch", ylabel="Score", title="Validation Edge Metrics")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_summary(log: pd.DataFrame, output: Path) -> None:
    train = _subset(log, "train")
    val = _subset(log, "val")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    if _has_values(train, "total"):
        axes[0].plot(train["epoch"], train["total"], marker="o", label="train")
    if _has_values(val, "loss"):
        axes[0].plot(val["epoch"], val["loss"], marker="o", label="val")
    for metric in ("ODS", "OIS", "AP"):
        if _has_values(val, metric):
            axes[1].plot(val["epoch"], val[metric], marker="o", label=metric)
    axes[0].set_title("Loss")
    axes[1].set_title("Validation Metrics")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
