"""Plot random local population trajectory examples for MEA UME/CME pairs.

This module uses the same response definition and grid assignment as the
grid-level map modules:
log((ON + 0.1) / (OFF + 0.1)) trajectories are computed for good units, then
UME and CME trajectories are compared within spatially matched grid cells.

Only a fixed-seed random subset of valid grid-cell conditions is plotted.
Each example shows the UME/CME mean local population trajectory with a shaded
mean +/- SEM band across good units in that grid cell.
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
MODULE_ID = "03_local_population_trajectory_examples"

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
    "n_examples": 50,
    "seed": 20260603,
    "dpi": 300,
}

COLORS = {
    "UME": "#d55e00",
    "CME": "#1b9e77",
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
    parser.add_argument("--n-examples", type=int, default=DEFAULT_ARGS["n_examples"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
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
            "axes.titlesize": 23,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


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
    phase_cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    events, progress = event_indices(stimulus, direction_zero_based)
    on_key = (exp_id, "on")
    off_key = (exp_id, "off")
    if on_key not in phase_cache:
        phase_cache[on_key] = load_phase_fr(mea_data_dir, exp_id, "on")
    if off_key not in phase_cache:
        phase_cache[off_key] = load_phase_fr(mea_data_dir, exp_id, "off")

    on = phase_cache[on_key]
    off = phase_cache[off_key]
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


def pair_bounds(ume_units: pd.DataFrame, cme_units: pd.DataFrame) -> tuple[float, float, float, float]:
    x_min = float(min(ume_units["x"].min(), cme_units["x"].min()))
    x_max = float(max(ume_units["x"].max(), cme_units["x"].max()))
    y_min = float(min(ume_units["y"].min(), cme_units["y"].min()))
    y_max = float(max(ume_units["y"].max(), cme_units["y"].max()))
    return x_min, x_max, y_min, y_max


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


def trajectory_for(
    mea_data_dir: Path,
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    epsilon: float,
    trajectory_cache: dict[tuple[str, str, int], np.ndarray],
    phase_cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    key = (exp_id, stimulus, direction_zero_based)
    if key not in trajectory_cache:
        trajectory_cache[key] = unit_log_onoff_trajectory(
            mea_data_dir=mea_data_dir,
            exp_id=exp_id,
            stimulus=stimulus,
            direction_zero_based=direction_zero_based,
            epsilon=epsilon,
            phase_cache=phase_cache,
        )
    return trajectory_cache[key]


def grid_arrays_for_condition(
    mea_data_dir: Path,
    pair_id: str,
    grid_n: int,
    direction: dict[str, object],
    epsilon: float,
    position_cache: dict[str, pd.DataFrame],
    trajectory_cache: dict[tuple[str, str, int], np.ndarray],
    phase_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    pair_cfg = PAIR_CONFIG[pair_id]
    ume_exp = pair_cfg["UME"]
    cme_exp = pair_cfg["CME"]
    if ume_exp not in position_cache:
        position_cache[ume_exp] = load_good_unit_positions(mea_data_dir, ume_exp)
    if cme_exp not in position_cache:
        position_cache[cme_exp] = load_good_unit_positions(mea_data_dir, cme_exp)

    ume_base = position_cache[ume_exp]
    cme_base = position_cache[cme_exp]
    bounds = pair_bounds(ume_base, cme_base)
    ume_units = assign_grid(ume_base, grid_n, bounds)
    cme_units = assign_grid(cme_base, grid_n, bounds)
    direction_zero_based = int(direction["id"]) - 1

    ume_traj = trajectory_for(
        mea_data_dir,
        ume_exp,
        "UME",
        direction_zero_based,
        epsilon,
        trajectory_cache,
        phase_cache,
    )
    cme_traj = trajectory_for(
        mea_data_dir,
        cme_exp,
        "CME",
        direction_zero_based,
        epsilon,
        trajectory_cache,
        phase_cache,
    )
    if ume_traj.shape[0] != len(ume_base):
        raise ValueError(f"{ume_exp}: position count does not match good_on/off unit count")
    if cme_traj.shape[0] != len(cme_base):
        raise ValueError(f"{cme_exp}: position count does not match good_on/off unit count")

    return ume_traj, cme_traj, ume_units, cme_units


def compute_candidates(
    mea_data_dir: Path,
    grid_scales: list[int],
    directions: list[dict[str, object]],
    epsilon: float,
    min_units_per_grid_per_stim: int,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[tuple[str, str, int], np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    position_cache: dict[str, pd.DataFrame] = {}
    trajectory_cache: dict[tuple[str, str, int], np.ndarray] = {}
    phase_cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for pair_id, pair_cfg in PAIR_CONFIG.items():
        for grid_n in grid_scales:
            for direction in directions:
                ume_traj, cme_traj, ume_units, cme_units = grid_arrays_for_condition(
                    mea_data_dir=mea_data_dir,
                    pair_id=pair_id,
                    grid_n=grid_n,
                    direction=direction,
                    epsilon=epsilon,
                    position_cache=position_cache,
                    trajectory_cache=trajectory_cache,
                    phase_cache=phase_cache,
                )

                for grid_id in sorted(set(ume_units["grid_id"]) | set(cme_units["grid_id"])):
                    ume_grid = ume_units[ume_units["grid_id"].eq(grid_id)]
                    cme_grid = cme_units[cme_units["grid_id"].eq(grid_id)]
                    if len(ume_grid) < min_units_per_grid_per_stim or len(cme_grid) < min_units_per_grid_per_stim:
                        continue

                    ume_arr = ume_traj[ume_grid["unit_row_idx"].to_numpy(dtype=int)]
                    cme_arr = cme_traj[cme_grid["unit_row_idx"].to_numpy(dtype=int)]
                    ume_mean = np.nanmean(ume_arr, axis=0)
                    cme_mean = np.nanmean(cme_arr, axis=0)
                    diff = ume_mean - cme_mean
                    abs_diff = np.abs(diff)
                    response_level = np.nanmean((np.abs(ume_mean) + np.abs(cme_mean)) / 2.0)

                    rows.append(
                        {
                            "pair_id": pair_id,
                            "pair_label": pair_cfg["label"],
                            "UME_exp": pair_cfg["UME"],
                            "CME_exp": pair_cfg["CME"],
                            "grid_n": grid_n,
                            "direction_id": int(direction["id"]),
                            "direction_code": str(direction["code"]),
                            "direction_name": str(direction["name"]),
                            "grid_id": grid_id,
                            "grid_x": int(ume_grid["grid_x"].iloc[0]),
                            "grid_y": int(ume_grid["grid_y"].iloc[0]),
                            "UME_unit_count": int(len(ume_grid)),
                            "CME_unit_count": int(len(cme_grid)),
                            "trajectory_MAE": float(np.nanmean(abs_diff)),
                            "normalized_trajectory_MAE": float(np.nanmean(abs_diff) / (response_level + SMALL)),
                        }
                    )

    candidates = pd.DataFrame(rows).sort_values(
        ["grid_n", "direction_id", "pair_id", "grid_y", "grid_x", "grid_id"]
    )
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))
    return candidates, position_cache, trajectory_cache, phase_cache


def select_examples(candidates: pd.DataFrame, n_examples: int, seed: int) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("No valid grid-cell candidates available")
    if n_examples > len(candidates):
        raise ValueError(f"Requested {n_examples} examples but only {len(candidates)} candidates are available")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=n_examples, replace=False)
    selected = candidates.iloc[chosen].copy().reset_index(drop=True)
    selected.insert(0, "example_id", [f"example_{idx:02d}" for idx in range(1, n_examples + 1)])
    selected.insert(1, "random_seed", seed)
    selected.insert(2, "random_draw_order", np.arange(1, n_examples + 1, dtype=int))
    return selected


def mean_sd_sem(arr: np.ndarray) -> pd.DataFrame:
    finite_count = np.isfinite(arr).sum(axis=0)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(finite_count), out=np.zeros_like(sd), where=finite_count > 1)
    return pd.DataFrame(
        {
            "progress_idx": np.arange(len(COMMON_PROGRESS), dtype=int),
            "progress": COMMON_PROGRESS,
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "band_lower": mean - sem,
            "band_upper": mean + sem,
            "n_units": finite_count.astype(int),
        }
    )


def arrays_for_selected_row(
    mea_data_dir: Path,
    row: pd.Series,
    epsilon: float,
    position_cache: dict[str, pd.DataFrame],
    trajectory_cache: dict[tuple[str, str, int], np.ndarray],
    phase_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    direction = next(direction for direction in DIRECTION_CONFIG if direction["code"] == row["direction_code"])
    ume_traj, cme_traj, ume_units, cme_units = grid_arrays_for_condition(
        mea_data_dir=mea_data_dir,
        pair_id=str(row["pair_id"]),
        grid_n=int(row["grid_n"]),
        direction=direction,
        epsilon=epsilon,
        position_cache=position_cache,
        trajectory_cache=trajectory_cache,
        phase_cache=phase_cache,
    )
    grid_id = str(row["grid_id"])
    ume_grid = ume_units[ume_units["grid_id"].eq(grid_id)]
    cme_grid = cme_units[cme_units["grid_id"].eq(grid_id)]
    ume_arr = ume_traj[ume_grid["unit_row_idx"].to_numpy(dtype=int)]
    cme_arr = cme_traj[cme_grid["unit_row_idx"].to_numpy(dtype=int)]
    return ume_arr, cme_arr


def plot_example(row: pd.Series, trajectory_table: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(4.55, 4.15), constrained_layout=True)

    for stimulus in ["UME", "CME"]:
        sub = trajectory_table[trajectory_table["stimulus"].eq(stimulus)]
        color = COLORS[stimulus]
        ax.fill_between(
            sub["progress"].to_numpy(dtype=float),
            sub["band_lower"].to_numpy(dtype=float),
            sub["band_upper"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            sub["progress"],
            sub["mean"],
            color=color,
            lw=3.8,
            label=stimulus,
        )

    y_values = np.r_[
        trajectory_table["band_lower"].to_numpy(dtype=float),
        trajectory_table["band_upper"].to_numpy(dtype=float),
        trajectory_table["mean"].to_numpy(dtype=float),
    ]
    finite = y_values[np.isfinite(y_values)]
    if finite.size:
        ymin = float(np.nanmin(finite))
        ymax = float(np.nanmax(finite))
        pad = max((ymax - ymin) * 0.15, 0.35)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.axhline(0, color="#b5b5b5", lw=0.8, alpha=0.85)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Normalized motion progress")
    ax.set_ylabel("log(ON/OFF) response")
    ax.set_title("Local population trajectory", pad=10)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.0, length=4)

    info = (
        f"{row['pair_label']}, {row['direction_code']}, {int(row['grid_n'])}x{int(row['grid_n'])}\n"
        f"Grid {row['grid_id']}, MAE={float(row['trajectory_MAE']):.2f}\n"
        f"n={int(row['UME_unit_count'])}/{int(row['CME_unit_count'])}"
    )
    ax.text(
        0.02,
        0.04,
        info,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        color="#333333",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()

    grid_scales = parse_int_list(args.grid_scales)
    direction_codes = parse_direction_list(args.directions)
    directions = [direction for direction in DIRECTION_CONFIG if direction["code"] in direction_codes]

    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    candidates, position_cache, trajectory_cache, phase_cache = compute_candidates(
        mea_data_dir=args.mea_data_dir,
        grid_scales=grid_scales,
        directions=directions,
        epsilon=args.epsilon,
        min_units_per_grid_per_stim=args.min_units_per_grid_per_stim,
    )
    selected = select_examples(candidates, args.n_examples, args.seed)

    figure_rows: list[dict[str, object]] = []
    trajectory_rows: list[pd.DataFrame] = []
    selected_rows = []

    for row in selected.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        ume_arr, cme_arr = arrays_for_selected_row(
            mea_data_dir=args.mea_data_dir,
            row=row_series,
            epsilon=args.epsilon,
            position_cache=position_cache,
            trajectory_cache=trajectory_cache,
            phase_cache=phase_cache,
        )
        stats = []
        for stimulus, arr in [("UME", ume_arr), ("CME", cme_arr)]:
            stim_stats = mean_sd_sem(arr)
            stim_stats.insert(0, "stimulus", stimulus)
            stim_stats.insert(0, "example_id", row.example_id)
            stats.append(stim_stats)
        trajectory_table = pd.concat(stats, ignore_index=True)
        trajectory_rows.append(trajectory_table)

        file_stem = (
            f"{MODULE_ID}_{row.example_id}_{row.pair_id}_"
            f"grid{int(row.grid_n):02d}_dir{row.direction_code}_cell{row.grid_id}"
        )
        png_path = figure_dir / f"{file_stem}.png"
        plot_example(row_series, trajectory_table, png_path, args.dpi)

        selected_dict = row_series.to_dict()
        selected_dict["png"] = str(png_path)
        selected_rows.append(selected_dict)
        figure_rows.append(
            {
                "figure_id": file_stem,
                "example_id": row.example_id,
                "candidate_id": int(row.candidate_id),
                "pair_id": row.pair_id,
                "grid_n": int(row.grid_n),
                "direction_code": row.direction_code,
                "grid_id": row.grid_id,
                "trajectory_MAE": float(row.trajectory_MAE),
                "png": str(png_path),
            }
        )
        print(f"Wrote {png_path}")

    selected_examples = pd.DataFrame(selected_rows)
    trajectory_summary = pd.concat(trajectory_rows, ignore_index=True)
    figure_manifest = pd.DataFrame(figure_rows)

    candidates.to_csv(table_dir / f"{MODULE_ID}_candidate_grid_cells.csv", index=False)
    selected_examples.to_csv(table_dir / f"{MODULE_ID}_selected_examples.csv", index=False)
    trajectory_summary.to_csv(table_dir / f"{MODULE_ID}_trajectory_summary.csv", index=False)
    figure_manifest.to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "mea_data_dir": str(args.mea_data_dir),
        "output_dir": str(args.output_dir),
        "pairs": PAIR_CONFIG,
        "grid_scales": grid_scales,
        "directions": directions,
        "response_metric": f"log((ON + {args.epsilon}) / (OFF + {args.epsilon}))",
        "trajectory_progress": COMMON_PROGRESS.tolist(),
        "window_steps": WINDOW_STEPS,
        "min_units_per_grid_per_stim": args.min_units_per_grid_per_stim,
        "candidate_count": int(len(candidates)),
        "n_examples": int(args.n_examples),
        "random_seed": int(args.seed),
        "selection_method": "uniform random sample without replacement from all valid grid-cell conditions",
        "band_definition": "mean +/- SEM across good-unit trajectories within the selected grid cell",
        "figure_formats": ["png"],
    }
    (args.output_dir / f"{MODULE_ID}_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(figure_rows)} figures to {figure_dir}")
    print(f"Saved tables to {table_dir}")


if __name__ == "__main__":
    main()
