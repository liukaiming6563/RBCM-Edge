"""Plot grid-level UME/CME trajectory MAE maps for paired MEA recordings.

For each paired retina, direction, and grid scale, this module computes:

1. good-unit log response trajectories:
   log((ON + epsilon) / (OFF + epsilon))
2. grid-level local population mean trajectories for UME and CME
3. trajectory MAE between the two grid-level mean trajectories

Each output PNG contains the three paired retinal preparations for a fixed
grid scale and motion direction.
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
MODULE_ID = "04_grid_trajectory_mae_maps"

PAIR_CONFIG = {
    "pair_31_32": {"CME": "000031", "UME": "000032", "label": "P31-32"},
    "pair_34_35": {"CME": "000034", "UME": "000035", "label": "P34-35"},
    "pair_37_38": {"CME": "000037", "UME": "000038", "label": "P37-38"},
}

DIRECTION_CONFIG = [
    {"id": 1, "code": "R", "name": "right"},
    {"id": 2, "code": "RU", "name": "up_right"},
    {"id": 3, "code": "U", "name": "up"},
    {"id": 4, "code": "LU", "name": "up_left"},
    {"id": 5, "code": "L", "name": "left"},
    {"id": 6, "code": "LD", "name": "down_left"},
    {"id": 7, "code": "D", "name": "down"},
    {"id": 8, "code": "RD", "name": "down_right"},
]

WINDOW_STEPS = {
    "UME": (0, 7),
    "CME": (0, 6),
}
STEPS_PER_DIRECTION = {
    "UME": 13,
    "CME": 11,
}
COMMON_PROGRESS = np.linspace(0.0, 1.0, 7)
SMALL = 1e-9

DEFAULT_ARGS = {
    "mea_data_dir": PROJECT_ROOT / "MEA_data",
    "output_dir": PROJECT_ROOT / "MEA_outputs" / MODULE_ID,
    "grid_scales": "8,10,12,16,20,25,30",
    "directions": ",".join(direction["code"] for direction in DIRECTION_CONFIG),
    "epsilon": 0.1,
    "min_units_per_grid_per_stim": 3,
    "dpi": 300,
    "vmax_percentile": 98.0,
}

UNIT_COLORS = {
    "UME": "#f4c430",
    "CME": "#48c7bf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mea-data-dir", type=Path, default=DEFAULT_ARGS["mea_data_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--grid-scales", default=DEFAULT_ARGS["grid_scales"])
    parser.add_argument("--directions", default=DEFAULT_ARGS["directions"])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_ARGS["epsilon"])
    parser.add_argument(
        "--min-units-per-grid-per-stim",
        type=int,
        default=DEFAULT_ARGS["min_units_per_grid_per_stim"],
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    parser.add_argument("--vmax-percentile", type=float, default=DEFAULT_ARGS["vmax_percentile"])
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_direction_list(text: str) -> list[str]:
    valid = {direction["code"] for direction in DIRECTION_CONFIG}
    directions = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(directions) - valid)
    if unknown:
        raise ValueError(f"Unknown direction code(s): {unknown}")
    return directions


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 21,
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


def mae_colormap() -> mpl.colors.Colormap:
    """Dark teal for small MAE, then green and yellow for larger MAE."""

    colors = ["#083f4a", "#0b5a63", "#16c783", "#f0ef5a", "#fff46a"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("trajectory_mae_teal_green_yellow", colors, N=256)
    cmap.set_bad("#083f4a")
    return cmap


def read_cluster_labels(ks_dir: Path) -> pd.DataFrame:
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

    return merged.drop_duplicates("cluster_id").sort_values("cluster_id").reset_index(drop=True)


def load_good_unit_positions(mea_data_dir: Path, exp_id: str) -> pd.DataFrame:
    """Load good-unit coordinates in the same unit order as good_on/off arrays."""

    exp_dir = mea_data_dir / exp_id
    ks_dir = exp_dir / "kilosort4"
    sorted_ids_path = exp_dir / "segment_result" / "origin_segment" / "sorted_cluster_ids.npy"
    required = [
        ks_dir / "spike_positions.npy",
        ks_dir / "spike_clusters.npy",
        ks_dir / "cluster_group.tsv",
        sorted_ids_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    labels = read_cluster_labels(ks_dir)
    good_ids = set(labels.loc[labels["label"].eq("good"), "cluster_id"].astype(int))
    sorted_cluster_ids = np.load(sorted_ids_path).astype(int)
    good_cluster_ids = [int(cluster_id) for cluster_id in sorted_cluster_ids if int(cluster_id) in good_ids]
    if not good_cluster_ids:
        raise ValueError(f"{exp_id}: no good clusters found in sorted_cluster_ids")

    spike_clusters = np.load(ks_dir / "spike_clusters.npy").astype(int)
    spike_positions = np.load(ks_dir / "spike_positions.npy").astype(float)
    if spike_positions.ndim != 2 or spike_positions.shape[1] < 2:
        raise ValueError(f"{ks_dir / 'spike_positions.npy'} has invalid shape: {spike_positions.shape}")
    if spike_positions.shape[0] != spike_clusters.shape[0]:
        raise ValueError(f"{exp_id}: spike_positions and spike_clusters lengths do not match")

    keep = np.isin(spike_clusters, np.asarray(good_cluster_ids, dtype=int))
    keep &= np.isfinite(spike_positions[:, 0]) & np.isfinite(spike_positions[:, 1])
    spike_table = pd.DataFrame(
        {
            "cluster_id": spike_clusters[keep],
            "x": spike_positions[keep, 0],
            "y": spike_positions[keep, 1],
        }
    )
    centers = spike_table.groupby("cluster_id", sort=False)[["x", "y"]].median()

    rows = []
    missing = []
    for unit_row_idx, cluster_id in enumerate(good_cluster_ids):
        if cluster_id not in centers.index:
            missing.append(cluster_id)
            continue
        rows.append(
            {
                "experiment_id": exp_id,
                "unit_row_idx": unit_row_idx,
                "cluster_id": int(cluster_id),
                "x": float(centers.loc[cluster_id, "x"]),
                "y": float(centers.loc[cluster_id, "y"]),
            }
        )

    if missing:
        raise ValueError(f"{exp_id}: good clusters missing spike positions: {missing[:10]}")
    return pd.DataFrame(rows).sort_values("unit_row_idx").reset_index(drop=True)


def load_phase_fr(mea_data_dir: Path, exp_id: str, phase: str) -> np.ndarray:
    phase_key = phase.lower()
    path = mea_data_dir / exp_id / "segment_result" / "processed_segment" / f"good_{phase_key}" / "output_fre.npy"
    if not path.exists():
        raise FileNotFoundError(f"Required firing-rate array not found: {path}")
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} has invalid shape {arr.shape}; expected repeat x unit x event")
    return arr


def event_indices(stimulus: str, direction_zero_based: int) -> tuple[np.ndarray, np.ndarray]:
    start, stop = WINDOW_STEPS[stimulus]
    steps = np.arange(start, stop)
    event_idx = direction_zero_based * STEPS_PER_DIRECTION[stimulus] + steps
    progress = np.linspace(0.0, 1.0, len(steps))
    return event_idx, progress


def unit_log_onoff_trajectory(
    mea_data_dir: Path,
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    epsilon: float,
) -> np.ndarray:
    events, progress = event_indices(stimulus, direction_zero_based)
    on = load_phase_fr(mea_data_dir, exp_id, "on")
    off = load_phase_fr(mea_data_dir, exp_id, "off")
    if int(events.max()) >= on.shape[2] or int(events.max()) >= off.shape[2]:
        raise IndexError(f"{exp_id} {stimulus}: event index exceeds firing-rate array shape")

    on_mean = np.nanmean(on[:, :, events], axis=0)
    off_mean = np.nanmean(off[:, :, events], axis=0)
    trajectory = np.log((on_mean + epsilon) / (off_mean + epsilon))

    if len(progress) == len(COMMON_PROGRESS) and np.allclose(progress, COMMON_PROGRESS):
        return trajectory

    out = np.empty((trajectory.shape[0], len(COMMON_PROGRESS)), dtype=float)
    for unit_idx in range(trajectory.shape[0]):
        out[unit_idx] = np.interp(COMMON_PROGRESS, progress, trajectory[unit_idx])
    return out


def assign_grid(units: pd.DataFrame, grid_n: int, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    x_min, x_max, y_min, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid coordinate bounds for grid assignment")

    out = units.copy()
    out["x_norm"] = (out["x"] - x_min) / (x_max - x_min)
    out["y_norm"] = (out["y"] - y_min) / (y_max - y_min)
    out["grid_x"] = np.clip(np.floor(out["x_norm"].to_numpy() * grid_n).astype(int), 0, grid_n - 1)
    out["grid_y"] = np.clip(np.floor(out["y_norm"].to_numpy() * grid_n).astype(int), 0, grid_n - 1)
    out["grid_id"] = out["grid_y"].astype(str).str.zfill(2) + "_" + out["grid_x"].astype(str).str.zfill(2)
    return out


def pair_bounds(ume_units: pd.DataFrame, cme_units: pd.DataFrame) -> tuple[float, float, float, float]:
    x_min = float(min(ume_units["x"].min(), cme_units["x"].min()))
    x_max = float(max(ume_units["x"].max(), cme_units["x"].max()))
    y_min = float(min(ume_units["y"].min(), cme_units["y"].min()))
    y_max = float(max(ume_units["y"].max(), cme_units["y"].max()))
    return x_min, x_max, y_min, y_max


def build_unit_table(units: pd.DataFrame, trajectories: np.ndarray, stimulus: str) -> pd.DataFrame:
    unit_indices = units["unit_row_idx"].to_numpy(dtype=int)
    if trajectories.shape[0] <= int(unit_indices.max()):
        raise ValueError("Unit position order exceeds trajectory array length")
    return pd.DataFrame(
        {
            "stimulus": stimulus,
            "grid_id": units["grid_id"].to_numpy(),
            "grid_x": units["grid_x"].to_numpy(dtype=int),
            "grid_y": units["grid_y"].to_numpy(dtype=int),
            "trajectory": list(trajectories[unit_indices]),
        }
    )


def pearson_vec(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if x.size < 2 or denominator == 0:
        return np.nan
    return float(1.0 - np.dot(x, y) / denominator)


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Use NumPy's available trapezoidal integration API without warnings."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def grid_trajectory_metrics(
    pair_id: str,
    pair_cfg: dict[str, str],
    grid_n: int,
    direction: dict[str, object],
    unit_df: pd.DataFrame,
    min_units_per_grid_per_stim: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for grid_id, group in unit_df.groupby("grid_id"):
        ume_group = group[group["stimulus"].eq("UME")]
        cme_group = group[group["stimulus"].eq("CME")]
        if len(ume_group) < min_units_per_grid_per_stim or len(cme_group) < min_units_per_grid_per_stim:
            continue

        ume = np.vstack(ume_group["trajectory"].to_numpy())
        cme = np.vstack(cme_group["trajectory"].to_numpy())
        ume_mean = np.nanmean(ume, axis=0)
        cme_mean = np.nanmean(cme, axis=0)
        diff = ume_mean - cme_mean
        abs_diff = np.abs(diff)
        ume_centered = ume_mean - np.nanmean(ume_mean)
        cme_centered = cme_mean - np.nanmean(cme_mean)
        shape_abs_diff = np.abs(ume_centered - cme_centered)
        derivative_abs_diff = np.abs(np.diff(ume_mean) - np.diff(cme_mean))
        response_level = np.nanmean((np.abs(ume_mean) + np.abs(cme_mean)) / 2.0)
        auc_diff = trapezoid_integral(ume_mean, COMMON_PROGRESS) - trapezoid_integral(cme_mean, COMMON_PROGRESS)
        peak_diff = float(np.nanmax(ume_mean) - np.nanmax(cme_mean))

        row: dict[str, object] = {
            "pair_id": pair_id,
            "pair_label": pair_cfg["label"],
            "UME_exp": pair_cfg["UME"],
            "CME_exp": pair_cfg["CME"],
            "grid_n": grid_n,
            "direction_id": int(direction["id"]),
            "direction_code": str(direction["code"]),
            "direction_name": str(direction["name"]),
            "grid_id": grid_id,
            "grid_x": int(ume_group["grid_x"].iloc[0]),
            "grid_y": int(ume_group["grid_y"].iloc[0]),
            "UME_unit_count": int(len(ume_group)),
            "CME_unit_count": int(len(cme_group)),
            "trajectory_MAE": float(np.nanmean(abs_diff)),
            "shape_MAE": float(np.nanmean(shape_abs_diff)),
            "derivative_MAE": float(np.nanmean(derivative_abs_diff)),
            "normalized_trajectory_MAE": float(np.nanmean(abs_diff) / (response_level + SMALL)),
            "trajectory_RMSE": float(np.sqrt(np.nanmean(diff**2))),
            "AUC_diff": auc_diff,
            "abs_AUC_diff": abs(auc_diff),
            "peak_diff": peak_diff,
            "abs_peak_diff": abs(peak_diff),
            "trajectory_corr": pearson_vec(ume_mean, cme_mean),
            "trajectory_cosine_distance": cosine_distance(ume_mean, cme_mean),
        }
        for idx, progress in enumerate(COMMON_PROGRESS):
            row[f"progress_{idx}"] = float(progress)
            row[f"UME_traj_{idx}"] = float(ume_mean[idx])
            row[f"CME_traj_{idx}"] = float(cme_mean[idx])
            row[f"abs_diff_{idx}"] = float(abs_diff[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def compute_all_metrics(
    mea_data_dir: Path,
    grid_scales: list[int],
    directions: list[dict[str, object]],
    epsilon: float,
    min_units_per_grid_per_stim: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    unit_summary_rows: list[dict[str, object]] = []
    unit_position_rows: list[pd.DataFrame] = []

    position_cache: dict[str, pd.DataFrame] = {}
    trajectory_cache: dict[tuple[str, str, int], np.ndarray] = {}

    for pair_id, pair_cfg in PAIR_CONFIG.items():
        ume_exp = pair_cfg["UME"]
        cme_exp = pair_cfg["CME"]
        if ume_exp not in position_cache:
            position_cache[ume_exp] = load_good_unit_positions(mea_data_dir, ume_exp)
        if cme_exp not in position_cache:
            position_cache[cme_exp] = load_good_unit_positions(mea_data_dir, cme_exp)

        ume_base = position_cache[ume_exp]
        cme_base = position_cache[cme_exp]
        unit_summary_rows.append(
            {
                "pair_id": pair_id,
                "pair_label": pair_cfg["label"],
                "UME_exp": ume_exp,
                "CME_exp": cme_exp,
                "UME_good_units": int(len(ume_base)),
                "CME_good_units": int(len(cme_base)),
            }
        )
        bounds = pair_bounds(ume_base, cme_base)
        x_min, x_max, y_min, y_max = bounds

        for stimulus, exp_id, base_units in [("UME", ume_exp, ume_base), ("CME", cme_exp, cme_base)]:
            unit_positions = base_units.copy()
            unit_positions["pair_id"] = pair_id
            unit_positions["pair_label"] = pair_cfg["label"]
            unit_positions["stimulus"] = stimulus
            unit_positions["experiment_id"] = exp_id
            unit_positions["x_norm"] = (unit_positions["x"] - x_min) / (x_max - x_min)
            unit_positions["y_norm"] = (unit_positions["y"] - y_min) / (y_max - y_min)
            unit_position_rows.append(
                unit_positions[
                    [
                        "pair_id",
                        "pair_label",
                        "stimulus",
                        "experiment_id",
                        "unit_row_idx",
                        "cluster_id",
                        "x",
                        "y",
                        "x_norm",
                        "y_norm",
                    ]
                ]
            )

        for grid_n in grid_scales:
            ume_units = assign_grid(ume_base, grid_n, bounds)
            cme_units = assign_grid(cme_base, grid_n, bounds)

            for direction in directions:
                direction_zero_based = int(direction["id"]) - 1
                ume_key = (ume_exp, "UME", direction_zero_based)
                cme_key = (cme_exp, "CME", direction_zero_based)
                if ume_key not in trajectory_cache:
                    trajectory_cache[ume_key] = unit_log_onoff_trajectory(
                        mea_data_dir,
                        ume_exp,
                        "UME",
                        direction_zero_based,
                        epsilon,
                    )
                if cme_key not in trajectory_cache:
                    trajectory_cache[cme_key] = unit_log_onoff_trajectory(
                        mea_data_dir,
                        cme_exp,
                        "CME",
                        direction_zero_based,
                        epsilon,
                    )

                if trajectory_cache[ume_key].shape[0] != len(ume_base):
                    raise ValueError(f"{ume_exp}: position count does not match good_on/off unit count")
                if trajectory_cache[cme_key].shape[0] != len(cme_base):
                    raise ValueError(f"{cme_exp}: position count does not match good_on/off unit count")

                unit_df = pd.concat(
                    [
                        build_unit_table(ume_units, trajectory_cache[ume_key], "UME"),
                        build_unit_table(cme_units, trajectory_cache[cme_key], "CME"),
                    ],
                    ignore_index=True,
                )
                metrics = grid_trajectory_metrics(
                    pair_id=pair_id,
                    pair_cfg=pair_cfg,
                    grid_n=grid_n,
                    direction=direction,
                    unit_df=unit_df,
                    min_units_per_grid_per_stim=min_units_per_grid_per_stim,
                )
                all_rows.append(metrics)

    grid_cells = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    unit_summary = pd.DataFrame(unit_summary_rows)
    unit_positions = pd.concat(unit_position_rows, ignore_index=True) if unit_position_rows else pd.DataFrame()
    return grid_cells, unit_summary, unit_positions


def summarize_metrics(grid_cells: pd.DataFrame) -> pd.DataFrame:
    if grid_cells.empty:
        return pd.DataFrame()
    return (
        grid_cells.groupby(["grid_n", "direction_code", "direction_id", "pair_id", "pair_label"], as_index=False)
        .agg(
            valid_grid_count=("grid_id", "size"),
            mean_trajectory_MAE=("trajectory_MAE", "mean"),
            median_trajectory_MAE=("trajectory_MAE", "median"),
            max_trajectory_MAE=("trajectory_MAE", "max"),
            mean_shape_MAE=("shape_MAE", "mean"),
            median_shape_MAE=("shape_MAE", "median"),
            max_shape_MAE=("shape_MAE", "max"),
            mean_derivative_MAE=("derivative_MAE", "mean"),
            median_derivative_MAE=("derivative_MAE", "median"),
            max_derivative_MAE=("derivative_MAE", "max"),
            mean_normalized_trajectory_MAE=("normalized_trajectory_MAE", "mean"),
            mean_trajectory_RMSE=("trajectory_RMSE", "mean"),
            mean_abs_AUC_diff=("abs_AUC_diff", "mean"),
            mean_abs_peak_diff=("abs_peak_diff", "mean"),
            mean_trajectory_cosine_distance=("trajectory_cosine_distance", "mean"),
        )
        .sort_values(["grid_n", "direction_id", "pair_id"])
        .reset_index(drop=True)
    )


def metric_matrix(grid_cells: pd.DataFrame, pair_id: str, grid_n: int, direction_code: str) -> np.ndarray:
    matrix = np.full((grid_n, grid_n), np.nan, dtype=float)
    sub = grid_cells[
        grid_cells["pair_id"].eq(pair_id)
        & grid_cells["grid_n"].eq(grid_n)
        & grid_cells["direction_code"].eq(direction_code)
    ]
    for row in sub.itertuples(index=False):
        matrix[int(row.grid_y), int(row.grid_x)] = float(row.trajectory_MAE)
    return matrix


def plot_map_figure(
    grid_cells: pd.DataFrame,
    unit_positions: pd.DataFrame,
    grid_n: int,
    direction: dict[str, object],
    vmax: float,
    output_path: Path,
    dpi: int,
) -> dict[str, object]:
    cmap = mae_colormap()
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    direction_code = str(direction["code"])

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.65), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.825, top=0.79, bottom=0.21, wspace=0.17)

    image = None
    for ax, (pair_id, pair_cfg) in zip(axes, PAIR_CONFIG.items()):
        matrix = metric_matrix(grid_cells, pair_id, grid_n, direction_code)
        display = np.nan_to_num(matrix, nan=0.0)
        image = ax.imshow(
            display,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            extent=(0, 1, 0, 1),
            aspect="equal",
        )

        pair_units = unit_positions[unit_positions["pair_id"].eq(pair_id)]
        for stimulus in ["CME", "UME"]:
            stim_units = pair_units[pair_units["stimulus"].eq(stimulus)]
            if stim_units.empty:
                continue
            ax.scatter(
                stim_units["x_norm"],
                stim_units["y_norm"],
                s=11,
                c=UNIT_COLORS[stimulus],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.18,
                zorder=4 if stimulus == "UME" else 3,
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=UNIT_COLORS["UME"],
                markeredgecolor="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=UNIT_COLORS["CME"],
                markeredgecolor="none",
                markersize=10,
            ),
        ]
        legend = ax.legend(
            legend_handles,
            [
                f"UME, n={int(pair_units['stimulus'].eq('UME').sum())}",
                f"CME, n={int(pair_units['stimulus'].eq('CME').sum())}",
            ],
            loc="upper left",
            bbox_to_anchor=(0.0, -0.04),
            frameon=False,
            fontsize=19,
            handletextpad=0.4,
            labelspacing=0.15,
            borderpad=0.2,
            borderaxespad=0.0,
        )
        for text in legend.get_texts():
            text.set_color("#222222")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_axes([0.865, 0.18, 0.028, 0.58])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Trajectory MAE", rotation=90, labelpad=14, fontsize=21)
    cbar.set_ticks([0, vmax / 2.0, vmax])
    cbar.ax.set_yticklabels([f"{0:g}", f"{vmax / 2.0:.2g}", f"{vmax:.2g}"])
    cbar.ax.tick_params(labelsize=19)

    fig.suptitle(
        f"{grid_n} x {grid_n} grid, direction {direction_code}",
        y=0.965,
        fontsize=23,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    sub = grid_cells[grid_cells["grid_n"].eq(grid_n) & grid_cells["direction_code"].eq(direction_code)]
    return {
        "figure_id": output_path.stem,
        "grid_n": grid_n,
        "direction_id": int(direction["id"]),
        "direction_code": direction_code,
        "valid_grid_count_total": int(len(sub)),
        "mean_trajectory_MAE": float(sub["trajectory_MAE"].mean()) if not sub.empty else np.nan,
        "vmax": float(vmax),
        "png": str(output_path),
    }


def main() -> None:
    args = parse_args()
    set_style()

    grid_scales = parse_int_list(args.grid_scales)
    direction_codes = parse_direction_list(args.directions)
    directions = [direction for direction in DIRECTION_CONFIG if direction["code"] in direction_codes]

    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    grid_cells, unit_summary, unit_positions = compute_all_metrics(
        mea_data_dir=args.mea_data_dir,
        grid_scales=grid_scales,
        directions=directions,
        epsilon=args.epsilon,
        min_units_per_grid_per_stim=args.min_units_per_grid_per_stim,
    )
    if grid_cells.empty:
        raise RuntimeError("No valid grid cells were computed")

    vmax = float(np.nanpercentile(grid_cells["trajectory_MAE"].to_numpy(dtype=float), args.vmax_percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(grid_cells["trajectory_MAE"].to_numpy(dtype=float)))
    vmax = max(vmax, SMALL)

    figure_rows = []
    for grid_n in grid_scales:
        for direction in directions:
            out_path = figure_dir / f"{MODULE_ID}_grid{grid_n:02d}_dir{direction['code']}.png"
            figure_rows.append(
                plot_map_figure(
                    grid_cells=grid_cells,
                    unit_positions=unit_positions,
                    grid_n=grid_n,
                    direction=direction,
                    vmax=vmax,
                    output_path=out_path,
                    dpi=args.dpi,
                )
            )
            print(f"Wrote {out_path}")

    summary = summarize_metrics(grid_cells)
    figure_manifest = pd.DataFrame(figure_rows)

    grid_cells.to_csv(table_dir / f"{MODULE_ID}_grid_cells.csv", index=False)
    summary.to_csv(table_dir / f"{MODULE_ID}_summary.csv", index=False)
    unit_summary.to_csv(table_dir / f"{MODULE_ID}_unit_summary.csv", index=False)
    unit_positions.to_csv(table_dir / f"{MODULE_ID}_unit_positions_overlay.csv", index=False)
    figure_manifest.to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "mea_data_dir": str(args.mea_data_dir),
        "output_dir": str(args.output_dir),
        "pairs": PAIR_CONFIG,
        "grid_scales": grid_scales,
        "directions": directions,
        "response_metric": f"log((ON + {args.epsilon}) / (OFF + {args.epsilon}))",
        "main_grid_metric": "trajectory_MAE",
        "trajectory_progress": COMMON_PROGRESS.tolist(),
        "window_steps": WINDOW_STEPS,
        "min_units_per_grid_per_stim": args.min_units_per_grid_per_stim,
        "position_definition": "median x/y of all finite spike_positions assigned to each good cluster; ordered by sorted_cluster_ids",
        "coordinate_transform": "none",
        "vmax_percentile": args.vmax_percentile,
        "vmax_used": vmax,
        "figure_formats": ["png"],
        "unit_position_overlay": {
            "included": True,
            "stimuli": ["UME", "CME"],
            "unit_label": "good",
            "coordinates": "pair-level normalized x/y coordinates over the shared UME/CME coordinate bounds",
        },
        "random_seed": None,
    }
    (args.output_dir / f"{MODULE_ID}_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(figure_rows)} figures to {figure_dir}")
    print(f"Saved tables to {table_dir}")


if __name__ == "__main__":
    main()
