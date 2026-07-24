"""Directional stability summary for MEA UME/CME trajectory metrics.

This module reads the existing permutation-test outputs for Trajectory MAE,
Shape MAE, and Derivative MAE. It summarizes whether the UME/CME difference is
stable across motion directions by plotting effect strength rather than only
the q < 0.05 pass fraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ID = "10_directional_stability_summary"

DIRECTION_ORDER = ["R", "RU", "U", "LU", "L", "LD", "D", "RD"]

METRIC_CONFIG = [
    {
        "metric_id": "trajectory_MAE",
        "label": "Trajectory MAE",
        "module_id": "05_trajectory_mae_permutation_test_summary",
        "real_col": "real_mean_trajectory_MAE",
        "null_col": "null_mean_trajectory_MAE",
        "null_std_col": "null_std_trajectory_MAE",
        "p_col": "p_trajectory_MAE",
        "q_col": "q_trajectory_MAE",
    },
]

DEFAULT_ARGS = {
    "mea_outputs_dir": PROJECT_ROOT / "MEA_outputs",
    "output_dir": PROJECT_ROOT / "MEA_outputs" / MODULE_ID,
    "q_threshold": 0.05,
    "jitter_seed": 20260603,
    "dpi": 300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mea-outputs-dir", type=Path, default=DEFAULT_ARGS["mea_outputs_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--q-threshold", type=float, default=DEFAULT_ARGS["q_threshold"])
    parser.add_argument("--jitter-seed", type=int, default=DEFAULT_ARGS["jitter_seed"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 21,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 19,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_condition_metrics(mea_outputs_dir: Path, q_threshold: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cfg in METRIC_CONFIG:
        table_path = (
            mea_outputs_dir
            / cfg["module_id"]
            / "tables"
            / f"{cfg['module_id']}_condition_tests.csv"
        )
        if not table_path.exists():
            raise FileNotFoundError(f"Required permutation-test table not found: {table_path}")
        source = pd.read_csv(table_path)
        required = [
            "condition_id",
            "pair_id",
            "pair_label",
            "grid_n",
            "direction_id",
            "direction_code",
            "direction_name",
            "valid_grid_count",
            cfg["real_col"],
            cfg["null_col"],
            cfg["null_std_col"],
            cfg["p_col"],
            cfg["q_col"],
        ]
        missing = [col for col in required if col not in source.columns]
        if missing:
            raise ValueError(f"{table_path} is missing columns: {missing}")

        out = source[required].copy()
        out = out.rename(
            columns={
                cfg["real_col"]: "real_mean",
                cfg["null_col"]: "null_mean",
                cfg["null_std_col"]: "null_std",
                cfg["p_col"]: "p_value",
                cfg["q_col"]: "q_value",
            }
        )
        out.insert(0, "metric_id", cfg["metric_id"])
        out.insert(1, "metric_label", cfg["label"])
        out["delta_real_null"] = out["real_mean"] - out["null_mean"]
        out["ratio_real_null"] = out["real_mean"] / out["null_mean"]
        out["z_real_null"] = np.divide(
            out["delta_real_null"],
            out["null_std"],
            out=np.full(len(out), np.nan, dtype=float),
            where=out["null_std"].to_numpy(dtype=float) > 0,
        )
        out["significant_q"] = out["q_value"] < q_threshold
        rows.append(out)

    data = pd.concat(rows, ignore_index=True)
    direction_rank = {direction: idx for idx, direction in enumerate(DIRECTION_ORDER)}
    data["direction_rank"] = data["direction_code"].map(direction_rank)
    if data["direction_rank"].isna().any():
        unknown = sorted(data.loc[data["direction_rank"].isna(), "direction_code"].unique())
        raise ValueError(f"Unknown direction code(s): {unknown}")
    data = data.sort_values(["metric_id", "direction_rank", "grid_n", "pair_id"]).reset_index(drop=True)
    return data


def summarize_by_direction(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        condition_metrics.groupby(
            ["metric_id", "metric_label", "direction_rank", "direction_id", "direction_code", "direction_name"],
            as_index=False,
        )
        .agg(
            condition_count=("condition_id", "size"),
            significant_count=("significant_q", "sum"),
            significant_fraction=("significant_q", "mean"),
            median_delta_real_null=("delta_real_null", "median"),
            mean_delta_real_null=("delta_real_null", "mean"),
            q1_delta_real_null=("delta_real_null", lambda x: float(np.nanpercentile(x, 25))),
            q3_delta_real_null=("delta_real_null", lambda x: float(np.nanpercentile(x, 75))),
            min_delta_real_null=("delta_real_null", "min"),
            max_delta_real_null=("delta_real_null", "max"),
            median_ratio_real_null=("ratio_real_null", "median"),
            median_z_real_null=("z_real_null", "median"),
            median_q_value=("q_value", "median"),
            min_q_value=("q_value", "min"),
            max_q_value=("q_value", "max"),
        )
        .sort_values(["metric_id", "direction_rank"])
        .reset_index(drop=True)
    )
    summary["significant_label"] = (
        summary["significant_count"].astype(int).astype(str)
        + "/"
        + summary["condition_count"].astype(int).astype(str)
    )
    return summary


def effect_colormap() -> mpl.colors.Colormap:
    colors = ["#f5f2e9", "#c9d989", "#4db3a2", "#196c7a", "#35215f"]
    return mpl.colors.LinearSegmentedColormap.from_list("real_null_delta", colors, N=256)


def plot_directional_summary(
    condition_metrics: pd.DataFrame,
    direction_summary: pd.DataFrame,
    output_path: Path,
    q_threshold: float,
    jitter_seed: int,
    dpi: int,
) -> dict[str, object]:
    rng = np.random.default_rng(jitter_seed)
    fig, ax = plt.subplots(figsize=(9.2, 6.1), constrained_layout=True)

    trajectory = condition_metrics[condition_metrics["metric_id"].eq("trajectory_MAE")].copy()
    y_positions = {direction: len(DIRECTION_ORDER) - 1 - idx for idx, direction in enumerate(DIRECTION_ORDER)}
    trajectory["y_base"] = trajectory["direction_code"].map(y_positions)
    trajectory["y_jitter"] = trajectory["y_base"] + rng.uniform(-0.16, 0.16, size=len(trajectory))

    sig = trajectory["significant_q"].to_numpy(dtype=bool)
    ax.scatter(
        trajectory.loc[~sig, "delta_real_null"],
        trajectory.loc[~sig, "y_jitter"],
        s=64,
        facecolors="white",
        edgecolors="#9a9a9a",
        linewidths=1.35,
        zorder=2,
        label="n.s.",
    )
    ax.scatter(
        trajectory.loc[sig, "delta_real_null"],
        trajectory.loc[sig, "y_jitter"],
        s=66,
        c="#2aa198",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.9,
        zorder=3,
        label=f"q < {q_threshold:g}",
    )

    trajectory_summary = direction_summary[direction_summary["metric_id"].eq("trajectory_MAE")]
    for row in trajectory_summary.itertuples(index=False):
        y = y_positions[row.direction_code]
        ax.hlines(
            y,
            row.q1_delta_real_null,
            row.q3_delta_real_null,
            color="#263238",
            lw=3.0,
            zorder=4,
        )
        ax.scatter(
            row.median_delta_real_null,
            y,
            marker="D",
            s=88,
            color="#c65d17",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        ax.text(
            ax.get_xlim()[1],
            y,
            row.significant_label,
            ha="right",
            va="center",
            fontsize=18,
            color="#333333",
        )

    x_min = min(0.0, float(np.nanmin(trajectory["delta_real_null"])))
    x_max = float(np.nanmax(trajectory["delta_real_null"]))
    pad = max((x_max - x_min) * 0.30, 0.014)
    ax.set_xlim(x_min - pad, x_max + pad)
    for text in ax.texts:
        text.set_x(x_max + pad * 0.78)
    ax.axvline(0, color="#6f6f6f", lw=1.3, ls="--", zorder=1)
    ax.set_yticks([y_positions[d] for d in DIRECTION_ORDER])
    ax.set_yticklabels(DIRECTION_ORDER)
    ax.set_ylim(-0.65, len(DIRECTION_ORDER) - 0.35)
    ax.set_xlabel("Real - null mean Trajectory MAE")
    ax.set_ylabel("Motion direction")
    ax.set_title("Directional stability of Trajectory MAE", pad=24)
    ax.text(
        0.98,
        1.005,
        "q<0.05 count",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=17,
        color="#444444",
    )
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d5d5d5", lw=0.8, alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {
        "figure_id": output_path.stem,
        "png": str(output_path),
        "q_threshold": float(q_threshold),
        "jitter_seed": int(jitter_seed),
        "trajectory_mae_direction_count": int(trajectory_summary["direction_code"].nunique()),
        "panel_count": 1,
        "scatter_color_significant": "#2aa198",
        "scatter_color_non_significant": "#9a9a9a",
    }


def main() -> None:
    args = parse_args()
    set_style()

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    condition_metrics = load_condition_metrics(args.mea_outputs_dir, args.q_threshold)
    direction_summary = summarize_by_direction(condition_metrics)

    figure_path = figure_dir / f"{MODULE_ID}.png"
    figure_row = plot_directional_summary(
        condition_metrics=condition_metrics,
        direction_summary=direction_summary,
        output_path=figure_path,
        q_threshold=args.q_threshold,
        jitter_seed=args.jitter_seed,
        dpi=args.dpi,
    )

    condition_metrics.to_csv(table_dir / f"{MODULE_ID}_condition_metrics.csv", index=False)
    direction_summary.to_csv(table_dir / f"{MODULE_ID}_direction_summary.csv", index=False)
    pd.DataFrame([figure_row]).to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "mea_outputs_dir": str(args.mea_outputs_dir),
        "output_dir": str(args.output_dir),
        "source_modules": [cfg["module_id"] for cfg in METRIC_CONFIG],
        "metrics": METRIC_CONFIG,
        "direction_order": DIRECTION_ORDER,
        "condition_definition": "paired retina x grid scale x motion direction",
        "conditions_per_direction_per_metric": 21,
        "condition_components_per_direction": "3 pairs x 7 grid scales",
        "q_threshold": float(args.q_threshold),
        "effect_strength": "delta_real_null = real_mean - null_mean for each metric",
        "figure_panel": "Trajectory MAE per-condition real-null deltas by direction; diamond = median, line = IQR",
        "jitter_seed": int(args.jitter_seed),
        "figure_formats": ["png"],
    }
    (args.output_dir / f"{MODULE_ID}_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved figure to {figure_path}")
    print(f"Saved tables to {table_dir}")


if __name__ == "__main__":
    main()
