"""Plot two-dimensional unit spatial distributions for each MEA experiment.

This module reconstructs sorted-unit positions from Kilosort spike-level
coordinates. For each good or MUA cluster, the unit position is defined as the
median x/y position of all spikes assigned to that cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODULE_ID = "02_unit_spatial_distribution"

EXPERIMENT_STIMULUS = {
    "000031": "CME",
    "000032": "UME",
    "000033": "bar",
    "000034": "CME",
    "000035": "UME",
    "000036": "bar",
    "000037": "CME",
    "000038": "UME",
    "000039": "bar",
}

DEFAULT_ARGS = {
    "mea_data_dir": PROJECT_ROOT / "MEA_data",
    "output_dir": PROJECT_ROOT / "MEA_outputs" / MODULE_ID,
    "experiments": list(EXPERIMENT_STIMULUS),
    "dpi": 300,
    "point_size_min": 20.0,
    "point_size_max": 62.0,
}

COLORS = {
    "good": "#d65f2e",
    "mua": "#3b79a8",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mea-data-dir", type=Path, default=DEFAULT_ARGS["mea_data_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_ARGS["experiments"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    parser.add_argument("--point-size-min", type=float, default=DEFAULT_ARGS["point_size_min"])
    parser.add_argument("--point-size-max", type=float, default=DEFAULT_ARGS["point_size_max"])
    return parser.parse_args()


def set_style() -> None:
    """Apply a compact, paper-oriented matplotlib style."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_cluster_labels(ks_dir: Path) -> pd.DataFrame:
    """Read cluster labels from Kilosort/Phy TSV files."""

    label_files = [ks_dir / "cluster_group.tsv", ks_dir / "cluster_KSLabel.tsv"]
    merged: pd.DataFrame | None = None

    for path in label_files:
        if not path.exists():
            continue
        table = pd.read_csv(path, sep="\t")
        if "cluster_id" not in table.columns:
            table = table.rename(columns={table.columns[0]: "cluster_id"})
        label_columns = [col for col in table.columns if col != "cluster_id"]
        if not label_columns:
            continue
        label_col = label_columns[0]
        table = table[["cluster_id", label_col]].rename(columns={label_col: "label"})
        table["cluster_id"] = table["cluster_id"].astype(int)
        table["label"] = table["label"].astype(str).str.lower().str.strip()
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on="cluster_id", how="outer", suffixes=("", "_alt"))
            if "label_alt" in merged.columns:
                merged["label"] = merged["label"].where(merged["label"].notna(), merged["label_alt"])
                merged = merged.drop(columns=["label_alt"])

    if merged is None or merged.empty:
        raise FileNotFoundError(f"No cluster label TSV file found in {ks_dir}")

    merged = merged.drop_duplicates("cluster_id").sort_values("cluster_id").reset_index(drop=True)
    return merged


def load_unit_positions(exp_dir: Path, exp_id: str) -> pd.DataFrame:
    """Reconstruct good and MUA unit coordinates from Kilosort spike positions."""

    ks_dir = exp_dir / "kilosort4"
    required = [
        ks_dir / "spike_positions.npy",
        ks_dir / "spike_clusters.npy",
        ks_dir / "cluster_group.tsv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    labels = read_cluster_labels(ks_dir)
    labels = labels[labels["label"].isin(["good", "mua"])].copy()
    if labels.empty:
        raise ValueError(f"No good or MUA labels found for experiment {exp_id}")

    spike_clusters = np.load(ks_dir / "spike_clusters.npy").astype(int)
    spike_positions = np.load(ks_dir / "spike_positions.npy").astype(float)
    if spike_positions.ndim != 2 or spike_positions.shape[1] < 2:
        raise ValueError(f"{ks_dir / 'spike_positions.npy'} has invalid shape: {spike_positions.shape}")
    if spike_positions.shape[0] != spike_clusters.shape[0]:
        raise ValueError(f"{exp_id}: spike_positions and spike_clusters lengths do not match")

    target_ids = labels["cluster_id"].to_numpy(dtype=int)
    keep = np.isin(spike_clusters, target_ids)
    finite_xy = np.isfinite(spike_positions[:, 0]) & np.isfinite(spike_positions[:, 1])
    keep &= finite_xy
    if not np.any(keep):
        raise ValueError(f"{exp_id}: no finite spike positions for good/MUA clusters")

    spike_table = pd.DataFrame(
        {
            "cluster_id": spike_clusters[keep],
            "x": spike_positions[keep, 0],
            "y": spike_positions[keep, 1],
        }
    )
    positions = spike_table.groupby("cluster_id", as_index=False).agg(
        x=("x", "median"),
        y=("y", "median"),
        n_spikes=("x", "size"),
    )

    units = labels.merge(positions, on="cluster_id", how="left")
    missing = units[units["x"].isna() | units["y"].isna()]["cluster_id"].tolist()
    if missing:
        raise ValueError(f"{exp_id}: clusters missing spike positions: {missing[:10]}")

    units.insert(0, "experiment_id", exp_id)
    units.insert(1, "stimulus", EXPERIMENT_STIMULUS.get(exp_id, "unknown"))
    return units.sort_values(["label", "cluster_id"]).reset_index(drop=True)


def point_sizes(values: pd.Series, size_min: float, size_max: float) -> np.ndarray:
    """Scale point size by log spike count for visual readability."""

    log_counts = np.log1p(values.to_numpy(dtype=float))
    low, high = np.nanpercentile(log_counts, [5, 95])
    if not np.isfinite(high - low) or high <= low:
        return np.full(log_counts.shape, (size_min + size_max) / 2.0)
    scaled = np.clip((log_counts - low) / (high - low), 0.0, 1.0)
    return size_min + scaled * (size_max - size_min)


def add_gradient_swatch(fig: plt.Figure, bounds: list[float], color: str, label: str) -> None:
    """Add a small unlabeled vertical color swatch used as a class key."""

    ax = fig.add_axes(bounds)
    rgb = np.array(mpl.colors.to_rgb(color))
    white = np.ones(3)
    ramp = np.linspace(0.35, 1.0, 128)[:, None]
    image = white * (1.0 - ramp) + rgb * ramp
    ax.imshow(image.reshape(128, 1, 3), origin="lower", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1.45, 0.5, label, transform=ax.transAxes, ha="left", va="center", fontsize=12)


def plot_experiment(
    units: pd.DataFrame,
    output_stem: Path,
    dpi: int,
    size_min: float,
    size_max: float,
) -> dict[str, object]:
    """Plot a single experiment's unit positions."""

    exp_id = str(units["experiment_id"].iloc[0])
    stimulus = str(units["stimulus"].iloc[0])
    good = units[units["label"].eq("good")].copy()
    mua = units[units["label"].eq("mua")].copy()

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.82, bottom=0.12, top=0.92)

    for label, df in [("mua", mua), ("good", good)]:
        if df.empty:
            continue
        ax.scatter(
            df["x"],
            df["y"],
            s=point_sizes(df["n_spikes"], size_min, size_max),
            c=COLORS[label],
            alpha=0.72 if label == "good" else 0.62,
            edgecolors="white",
            linewidths=0.35,
            rasterized=True,
            zorder=2 if label == "good" else 1,
        )

    x_min, x_max = units["x"].min(), units["x"].max()
    y_min, y_max = units["y"].min(), units["y"].max()
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, color="#d6d6d6", linewidth=0.8, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.0, length=4, pad=6)
    ax.set_title(f"Experiment {exp_id} ({stimulus})", pad=8)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["good"], markeredgecolor="none", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["mua"], markeredgecolor="none", markersize=7),
    ]
    ax.legend(
        legend_handles,
        [f"good units  {len(good)}", f"MUA units  {len(mua)}"],
        frameon=False,
        loc="lower left",
        handletextpad=0.6,
        borderpad=0.2,
    )

    add_gradient_swatch(fig, [0.86, 0.56, 0.035, 0.28], COLORS["good"], "good")
    add_gradient_swatch(fig, [0.86, 0.22, 0.035, 0.28], COLORS["mua"], "MUA")

    png_path = output_stem.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)

    return {
        "experiment_id": exp_id,
        "stimulus": stimulus,
        "good_units": int(len(good)),
        "mua_units": int(len(mua)),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "png": str(png_path),
    }


def main() -> None:
    args = parse_args()
    set_style()

    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    all_units: list[pd.DataFrame] = []

    for exp_id in args.experiments:
        exp_dir = args.mea_data_dir / exp_id
        units = load_unit_positions(exp_dir, exp_id)
        output_stem = figure_dir / f"{MODULE_ID}_exp{exp_id}"
        manifest_rows.append(
            plot_experiment(
                units=units,
                output_stem=output_stem,
                dpi=args.dpi,
                size_min=args.point_size_min,
                size_max=args.point_size_max,
            )
        )
        all_units.append(units)
        print(
            f"Wrote spatial distribution for experiment {exp_id}: "
            f"good={int(units['label'].eq('good').sum())}, "
            f"MUA={int(units['label'].eq('mua').sum())}"
        )

    manifest = pd.DataFrame(manifest_rows)
    unit_table = pd.concat(all_units, ignore_index=True)

    manifest.to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)
    unit_table.to_csv(table_dir / f"{MODULE_ID}_unit_positions.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "mea_data_dir": str(args.mea_data_dir),
        "output_dir": str(args.output_dir),
        "experiments": list(args.experiments),
        "dpi": args.dpi,
        "point_size_min": args.point_size_min,
        "point_size_max": args.point_size_max,
        "position_definition": "median x/y of all finite spike_positions assigned to each good or MUA cluster",
        "figure_formats": ["png"],
        "random_seed": None,
    }
    (args.output_dir / f"{MODULE_ID}_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved figures to {figure_dir}")
    print(f"Saved tables to {table_dir}")


if __name__ == "__main__":
    main()
