"""Formal condition-level effect distribution summary for three MAE metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ID = "12_effect_distribution_summary"

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
        "color": "#2aa198",
    },
    {
        "metric_id": "shape_MAE",
        "label": "Shape MAE",
        "module_id": "07_shape_mae_permutation_test_summary",
        "real_col": "real_mean_shape_MAE",
        "null_col": "null_mean_shape_MAE",
        "null_std_col": "null_std_shape_MAE",
        "p_col": "p_shape_MAE",
        "q_col": "q_shape_MAE",
        "color": "#6c8fc9",
    },
    {
        "metric_id": "derivative_MAE",
        "label": "Derivative MAE",
        "module_id": "09_derivative_mae_permutation_test_summary",
        "real_col": "real_mean_derivative_MAE",
        "null_col": "null_mean_derivative_MAE",
        "null_std_col": "null_std_derivative_MAE",
        "p_col": "p_derivative_MAE",
        "q_col": "q_derivative_MAE",
        "color": "#c27a35",
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
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
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
            raise FileNotFoundError(f"Required table not found: {table_path}")
        source = pd.read_csv(table_path)
        required = [
            "condition_id",
            "pair_id",
            "pair_label",
            "grid_n",
            "direction_id",
            "direction_code",
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
    return pd.concat(rows, ignore_index=True)


def summarize_metrics(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        condition_metrics.groupby(["metric_id", "metric_label"], as_index=False)
        .agg(
            condition_count=("condition_id", "size"),
            significant_count=("significant_q", "sum"),
            significant_fraction=("significant_q", "mean"),
            median_delta_real_null=("delta_real_null", "median"),
            mean_delta_real_null=("delta_real_null", "mean"),
            q1_delta_real_null=("delta_real_null", lambda x: float(np.nanpercentile(x, 25))),
            q3_delta_real_null=("delta_real_null", lambda x: float(np.nanpercentile(x, 75))),
            median_ratio_real_null=("ratio_real_null", "median"),
            median_z_real_null=("z_real_null", "median"),
            median_q_value=("q_value", "median"),
        )
        .reset_index(drop=True)
    )
    summary["significant_label"] = (
        summary["significant_count"].astype(int).astype(str)
        + "/"
        + summary["condition_count"].astype(int).astype(str)
    )
    return summary


def plot_distribution_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    zero_line: float | None,
    rng: np.random.Generator,
) -> None:
    labels = [cfg["label"] for cfg in METRIC_CONFIG]
    colors = [cfg["color"] for cfg in METRIC_CONFIG]
    values = [data.loc[data["metric_label"].eq(label), value_col].dropna().to_numpy(dtype=float) for label in labels]
    positions = np.arange(1, len(labels) + 1)

    parts = ax.violinplot(values, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.22)

    for pos, vals, color in zip(positions, values, colors):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=13,
            color=color,
            alpha=0.48,
            linewidths=0,
            zorder=2,
        )
        q1, median, q3 = np.nanpercentile(vals, [25, 50, 75])
        ax.vlines(pos, q1, q3, color="#263238", lw=2.2, zorder=4)
        ax.scatter(pos, median, marker="D", s=36, color="#222222", edgecolors="white", linewidths=0.35, zorder=5)

    if zero_line is not None:
        ax.axhline(zero_line, color="#6f6f6f", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.65)


def plot_effect_distributions(
    condition_metrics: pd.DataFrame,
    output_path: Path,
    jitter_seed: int,
    dpi: int,
) -> dict[str, object]:
    rng = np.random.default_rng(jitter_seed)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), constrained_layout=True)
    plot_distribution_panel(
        axes[0],
        condition_metrics,
        "delta_real_null",
        "Real - null",
        "Delta effect",
        0.0,
        rng,
    )
    plot_distribution_panel(
        axes[1],
        condition_metrics,
        "ratio_real_null",
        "Real / null",
        "Ratio effect",
        1.0,
        rng,
    )
    plot_distribution_panel(
        axes[2],
        condition_metrics,
        "z_real_null",
        "Delta / null SD",
        "Null-standardized effect",
        0.0,
        rng,
    )
    fig.suptitle("Condition-level effect distributions across trajectory metrics", y=1.04, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {
        "figure_id": output_path.stem,
        "png": str(output_path),
        "metric_count": len(METRIC_CONFIG),
        "condition_count_total": int(len(condition_metrics)),
        "jitter_seed": int(jitter_seed),
    }


def main() -> None:
    args = parse_args()
    set_style()

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    condition_metrics = load_condition_metrics(args.mea_outputs_dir, args.q_threshold)
    metric_summary = summarize_metrics(condition_metrics)

    figure_path = figure_dir / f"{MODULE_ID}.png"
    figure_row = plot_effect_distributions(condition_metrics, figure_path, args.jitter_seed, args.dpi)

    condition_metrics.to_csv(table_dir / f"{MODULE_ID}_condition_metrics.csv", index=False)
    metric_summary.to_csv(table_dir / f"{MODULE_ID}_metric_summary.csv", index=False)
    pd.DataFrame([figure_row]).to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "mea_outputs_dir": str(args.mea_outputs_dir),
        "output_dir": str(args.output_dir),
        "source_modules": [cfg["module_id"] for cfg in METRIC_CONFIG],
        "metrics": METRIC_CONFIG,
        "q_threshold": float(args.q_threshold),
        "effect_metrics": [
            "delta_real_null = real_mean - null_mean",
            "ratio_real_null = real_mean / null_mean",
            "z_real_null = (real_mean - null_mean) / null_std",
        ],
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
