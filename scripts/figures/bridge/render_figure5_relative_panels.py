#!/usr/bin/env python3
"""Render the two final Figure 5 panels from the relative-modulation audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "edge_outputs"
    / "rbcm"
    / "analyses"
    / "mea_rbcm_bridge"
    / "figure5_relative"
)
DEFAULT_OUTPUT = (
    ROOT
    / "edge_outputs"
    / "rbcm"
    / "figures"
    / "mea_rbcm_bridge"
    / "figure5_relative"
)

DATASET_ORDER = ["BIPED", "Multicue", "NYUDv2", "UDED"]

STATE_SPECS = [
    ("Relative\nenhancement", "enhance_fraction", "#E69A60", "#C95700", "#DF7625", "#A94400"),
    ("Relative\nsuppression", "suppress_fraction", "#9B8BC7", "#5E49A2", "#7E68B7", "#493586"),
    ("Relative\nnear-neutral", "neutral_fraction", "#BAC3C9", "#7D8C97", "#96A5AF", "#5D6E79"),
]
MEA_COLOR = "#348A80"
RBCM_COLOR = "#4C78A8"
MEAN_COLOR = "#D55E00"
GRID_COLOR = "#D9E0E5"
SPINE_COLOR = "#65747E"
TEXT_MUTED = "#5D6B75"
PANEL_FACE = "#FBFAF7"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 18.0,
            "axes.labelsize": 21.0,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 18.0,
            "legend.fontsize": 16.5,
            "axes.linewidth": 1.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": PANEL_FACE,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_sources(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mea = pd.read_csv(source / "mea_relative_rows.csv")
    model = pd.read_csv(source / "model_relative_rows.csv")
    states = ["enhance_fraction", "suppress_fraction", "neutral_fraction"]
    model_datasets = (
        model.groupby("dataset", as_index=False)[states]
        .mean()
        .set_index("dataset")
        .reindex(DATASET_ORDER)
        .reset_index()
    )
    model_uded = model.loc[model["dataset"] == "UDED"].copy()
    if len(mea) != 24 or len(model_datasets) != 4 or len(model_uded) != 30:
        raise RuntimeError(
            f"Unexpected source sizes: MEA={len(mea)}, datasets={len(model_datasets)}, UDED={len(model_uded)}"
        )
    return mea, model_datasets, model_uded


def symmetric_jitter(count: int, width: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-width, width, count)
    return values - values.mean()


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(width=1.2, length=5.0, pad=7)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1.0, alpha=0.9, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        ha="left",
        va="top",
    )


def draw_panel_a(
    ax: plt.Axes, mea: pd.DataFrame, model_datasets: pd.DataFrame
) -> None:
    category_x = np.arange(3, dtype=float)
    mea_offset, model_offset = -0.18, 0.18
    mea_jitter = symmetric_jitter(len(mea), 0.065, 20260807)
    model_jitter = np.array([-0.095, -0.045, 0.045, 0.095], dtype=float)

    for index, (_, column, mea_color, model_color, mea_mean, model_mean) in enumerate(
        STATE_SPECS
    ):
        mea_values = mea[column].to_numpy(dtype=float) * 100.0
        model_values = model_datasets[column].to_numpy(dtype=float) * 100.0
        ax.scatter(
            category_x[index] + mea_offset + mea_jitter,
            mea_values,
            s=80,
            marker="o",
            color=mea_color,
            alpha=0.70,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        ax.scatter(
            category_x[index] + model_offset + model_jitter,
            model_values,
            s=105,
            marker="s",
            color=model_color,
            alpha=0.95,
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
        for center, values, color in (
            (category_x[index] + mea_offset, mea_values, mea_mean),
            (category_x[index] + model_offset, model_values, model_mean),
        ):
            mean_value = float(values.mean())
            ax.hlines(
                mean_value,
                center - 0.13,
                center + 0.13,
                color=color,
                linewidth=3.4,
                zorder=5,
            )
            ax.scatter(
                center,
                mean_value,
                s=185,
                marker="D",
                color=color,
                edgecolors="white",
                linewidths=1.8,
                zorder=6,
            )
            text = ax.text(
                center,
                mean_value + 3.5,
                f"{mean_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=17.0,
                fontweight="bold",
                color="#20272C",
                zorder=7,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=3.0, foreground=PANEL_FACE)]
            )

    ax.set_ylabel("Proportion (%)", labelpad=8)
    ax.set_xticks(category_x, [item[0] for item in STATE_SPECS])
    ax.set_xlim(-0.50, 2.50)
    ax.set_ylim(0, 65)
    ax.set_yticks(np.arange(0, 61, 10))
    style_axis(ax)
    legend_color = "#71818B"
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9, color=legend_color, label="MEA group-direction observations"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=9, color=legend_color, label="H-RBCM datasets"),
        Line2D([0], [0], marker="D", linestyle="-", linewidth=2.0, markersize=9, color=legend_color, markeredgecolor="white", label="Mean"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        borderaxespad=0,
        frameon=False,
        handlelength=1.8,
        labelspacing=0.55,
    )
    ax.text(
        0.02,
        0.025,
        "MEA: 3 groups x 8 directions   |   H-RBCM: 4 datasets",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15.0,
        color="#71818B",
    )
    panel_label(ax, "A")


def draw_panel_b(ax: plt.Axes, mea: pd.DataFrame, model: pd.DataFrame) -> None:
    groups = [
        (mea["signed_balance_index"].to_numpy(dtype=float), MEA_COLOR, "o"),
        (model["signed_balance_index"].to_numpy(dtype=float), RBCM_COLOR, "s"),
    ]
    rng = np.random.default_rng(4517)
    for x, (values, color, marker) in enumerate(groups):
        violin = ax.violinplot(
            values,
            positions=[x],
            widths=0.70,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.35,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(1.8)
            body.set_alpha(0.15)
        ax.scatter(
            x + rng.uniform(-0.14, 0.14, len(values)),
            values,
            s=80,
            marker=marker,
            color=color,
            alpha=0.74,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        minimum = float(values.min())
        maximum = float(values.max())
        mean_value = float(values.mean())
        ax.vlines(x, minimum, maximum, color=color, linewidth=3.1, alpha=0.92, zorder=4)
        ax.hlines([minimum, maximum], x - 0.11, x + 0.11, color=color, linewidth=3.1, zorder=4)
        ax.scatter(
            [x],
            [mean_value],
            marker="D",
            s=210,
            color=MEAN_COLOR,
            edgecolor="white",
            linewidth=1.8,
            zorder=5,
        )
        ax.text(
            x + 0.18,
            mean_value,
            f"mean {mean_value:+.3f}",
            ha="left",
            va="center",
            fontsize=17.0,
            color=color,
            fontweight="bold",
        )

    ax.axhline(0.0, color=SPINE_COLOR, linewidth=2.0, linestyle="--", zorder=1)
    ax.set_xticks(
        [0, 1],
        ["MEA\n3 groups x 8 directions", "H-RBCM\n30 images"],
    )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks(np.arange(-0.4, 0.41, 0.2))
    ax.set_ylabel("Relative signed balance index, B", labelpad=8)
    style_axis(ax)
    ax.text(
        0.98,
        0.97,
        "Suppression-dominant",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=17.0,
        color=TEXT_MUTED,
    )
    ax.text(
        0.02,
        0.03,
        "Enhancement-dominant",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17.0,
        color=TEXT_MUTED,
    )
    panel_label(ax, "B")


def save_panel(
    fig: plt.Figure,
    filename: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / filename,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
    )


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    configure_style()
    mea, model_datasets, model_uded = load_sources(source)

    fig_a, ax_a = plt.subplots(figsize=(7.4, 6.0))
    fig_a.subplots_adjust(left=0.16, right=0.985, top=0.95, bottom=0.19)
    draw_panel_a(ax_a, mea, model_datasets)
    save_panel(
        fig_a,
        "05A_MEA_H-RBCM_relative_composition.png",
        output,
    )
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7.4, 6.0))
    fig_b.subplots_adjust(left=0.18, right=0.985, top=0.95, bottom=0.20)
    draw_panel_b(ax_b, mea, model_uded)
    save_panel(
        fig_b,
        "05B_MEA_H-RBCM_relative_balance.png",
        output,
    )
    plt.close(fig_b)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), gridspec_kw={"wspace": 0.18})
    draw_panel_a(axes[0], mea, model_datasets)
    draw_panel_b(axes[1], mea, model_uded)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.95, bottom=0.20)
    save_panel(
        fig,
        "05_MEA_H-RBCM_relative_combined.png",
        output,
    )
    plt.close(fig)

    print(f"Read final source tables from {source}")
    print(f"Wrote Figure 5 panels to {output}")


if __name__ == "__main__":
    main()
