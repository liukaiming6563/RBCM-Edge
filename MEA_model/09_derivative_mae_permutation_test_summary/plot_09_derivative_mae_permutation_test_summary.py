"""Permutation-test summary plot for MEA UME/CME Derivative MAE.

Each condition is one paired retina x grid scale x motion direction. For each
condition, UME/CME labels are permuted within every valid spatial grid while
preserving the original UME and CME unit counts. The test statistic is the mean
Derivative MAE across valid grids.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ID = "09_derivative_mae_permutation_test_summary"
BASE_MODULE_ID = "03_local_population_trajectory_examples"
BASE_MODULE_PATH = (
    PROJECT_ROOT
    / "MEA_model"
    / BASE_MODULE_ID
    / "plot_03_local_population_trajectory_examples.py"
)

DEFAULT_ARGS = {
    "mea_data_dir": PROJECT_ROOT / "MEA_data",
    "output_dir": PROJECT_ROOT / "MEA_outputs" / MODULE_ID,
    "grid_scales": "8,10,12,16,20,25,30",
    "directions": "R,RU,U,LU,L,LD,D,RD",
    "epsilon": 0.1,
    "min_units_per_grid_per_stim": 3,
    "n_perm": 5000,
    "seed": 20260603,
    "dpi": 300,
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
    parser.add_argument("--n-perm", type=int, default=DEFAULT_ARGS["n_perm"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    return parser.parse_args()


def load_base_module() -> ModuleType:
    if not BASE_MODULE_PATH.exists():
        raise FileNotFoundError(f"Base trajectory module not found: {BASE_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("mea_trajectory_base", BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base trajectory module from {BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    p = p_values.to_numpy(dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(p[finite])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def derivative_mae_from_means(ume_mean: np.ndarray, cme_mean: np.ndarray) -> float:
    return float(np.nanmean(np.abs(np.diff(ume_mean) - np.diff(cme_mean))))


def condition_grid_arrays(
    base: ModuleType,
    mea_data_dir: Path,
    pair_id: str,
    grid_n: int,
    direction: dict[str, object],
    epsilon: float,
    min_units_per_grid_per_stim: int,
    position_cache: dict[str, pd.DataFrame],
    trajectory_cache: dict[tuple[str, str, int], np.ndarray],
    phase_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[list[float], list[tuple[np.ndarray, int]], list[dict[str, object]]]:
    ume_traj, cme_traj, ume_units, cme_units = base.grid_arrays_for_condition(
        mea_data_dir=mea_data_dir,
        pair_id=pair_id,
        grid_n=grid_n,
        direction=direction,
        epsilon=epsilon,
        position_cache=position_cache,
        trajectory_cache=trajectory_cache,
        phase_cache=phase_cache,
    )

    real_grid_mae: list[float] = []
    perm_arrays: list[tuple[np.ndarray, int]] = []
    grid_rows: list[dict[str, object]] = []

    for grid_id in sorted(set(ume_units["grid_id"]) | set(cme_units["grid_id"])):
        ume_grid = ume_units[ume_units["grid_id"].eq(grid_id)]
        cme_grid = cme_units[cme_units["grid_id"].eq(grid_id)]
        if len(ume_grid) < min_units_per_grid_per_stim or len(cme_grid) < min_units_per_grid_per_stim:
            continue

        ume = ume_traj[ume_grid["unit_row_idx"].to_numpy(dtype=int)]
        cme = cme_traj[cme_grid["unit_row_idx"].to_numpy(dtype=int)]
        ume_mean = np.nanmean(ume, axis=0)
        cme_mean = np.nanmean(cme, axis=0)
        mae = derivative_mae_from_means(ume_mean, cme_mean)
        real_grid_mae.append(mae)
        perm_arrays.append((np.vstack([ume, cme]), len(ume_grid)))
        grid_rows.append(
            {
                "grid_id": grid_id,
                "grid_x": int(ume_grid["grid_x"].iloc[0]),
                "grid_y": int(ume_grid["grid_y"].iloc[0]),
                "UME_unit_count": int(len(ume_grid)),
                "CME_unit_count": int(len(cme_grid)),
                "derivative_MAE": mae,
            }
        )

    return real_grid_mae, perm_arrays, grid_rows


def permutation_test(
    real_grid_mae: list[float],
    perm_arrays: list[tuple[np.ndarray, int]],
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if not real_grid_mae or not perm_arrays:
        return {
            "real_mean_derivative_MAE": np.nan,
            "null_mean_derivative_MAE": np.nan,
            "null_std_derivative_MAE": np.nan,
            "p_derivative_MAE": np.nan,
        }

    real = float(np.nanmean(real_grid_mae))
    null_sum = np.zeros(n_perm, dtype=float)

    for arr, ume_n in perm_arrays:
        arr = np.asarray(arr, dtype=float)
        total_n = arr.shape[0]
        cme_n = total_n - ume_n
        if ume_n <= 0 or cme_n <= 0:
            continue

        # Generate all within-grid label permutations for this grid at once.
        # The smallest random scores become pseudo-UME; the remaining units
        # become pseudo-CME. This preserves the original group sizes exactly.
        order = np.argsort(rng.random((n_perm, total_n)), axis=1)
        ume_idx = order[:, :ume_n]
        ume_sum = arr[ume_idx].sum(axis=1)
        total_sum = arr.sum(axis=0)
        cme_sum = total_sum - ume_sum
        diff = np.diff(ume_sum / ume_n, axis=1) - np.diff(cme_sum / cme_n, axis=1)
        null_sum += np.nanmean(np.abs(diff), axis=1)

    null_values = null_sum / len(perm_arrays)

    p_value = float((np.sum(null_values >= real) + 1) / (n_perm + 1))
    return {
        "real_mean_derivative_MAE": real,
        "null_mean_derivative_MAE": float(np.nanmean(null_values)),
        "null_std_derivative_MAE": float(np.nanstd(null_values, ddof=1)),
        "p_derivative_MAE": p_value,
    }


def compute_all_tests(
    base: ModuleType,
    mea_data_dir: Path,
    grid_scales: list[int],
    directions: list[dict[str, object]],
    epsilon: float,
    min_units_per_grid_per_stim: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_cache: dict[str, pd.DataFrame] = {}
    trajectory_cache: dict[tuple[str, str, int], np.ndarray] = {}
    phase_cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    grid_rows_all: list[dict[str, object]] = []
    condition_idx = 0

    for grid_n in grid_scales:
        for direction in directions:
            for pair_id, pair_cfg in base.PAIR_CONFIG.items():
                condition_seed = int(seed + condition_idx)
                real_grid_mae, perm_arrays, grid_rows = condition_grid_arrays(
                    base=base,
                    mea_data_dir=mea_data_dir,
                    pair_id=pair_id,
                    grid_n=grid_n,
                    direction=direction,
                    epsilon=epsilon,
                    min_units_per_grid_per_stim=min_units_per_grid_per_stim,
                    position_cache=position_cache,
                    trajectory_cache=trajectory_cache,
                    phase_cache=phase_cache,
                )
                stats = permutation_test(
                    real_grid_mae=real_grid_mae,
                    perm_arrays=perm_arrays,
                    n_perm=n_perm,
                    rng=np.random.default_rng(condition_seed),
                )
                row = {
                    "condition_id": condition_idx,
                    "condition_seed": condition_seed,
                    "pair_id": pair_id,
                    "pair_label": pair_cfg["label"],
                    "UME_exp": pair_cfg["UME"],
                    "CME_exp": pair_cfg["CME"],
                    "grid_n": int(grid_n),
                    "direction_id": int(direction["id"]),
                    "direction_code": str(direction["code"]),
                    "direction_name": str(direction["name"]),
                    "valid_grid_count": int(len(real_grid_mae)),
                    "mean_grid_derivative_MAE": float(np.nanmean(real_grid_mae)) if real_grid_mae else np.nan,
                    "median_grid_derivative_MAE": float(np.nanmedian(real_grid_mae)) if real_grid_mae else np.nan,
                    **stats,
                }
                rows.append(row)
                for grid_row in grid_rows:
                    grid_rows_all.append({**row, **grid_row})
                condition_idx += 1
                print(
                    f"condition {condition_idx}: grid={grid_n}, "
                    f"dir={direction['code']}, pair={pair_id}, "
                    f"real={stats['real_mean_derivative_MAE']:.4f}, "
                    f"null={stats['null_mean_derivative_MAE']:.4f}, "
                    f"p={stats['p_derivative_MAE']:.6f}"
                )

    tests = pd.DataFrame(rows)
    tests["q_derivative_MAE"] = benjamini_hochberg(tests["p_derivative_MAE"])
    tests["significant_q05"] = tests["q_derivative_MAE"] < 0.05
    grid_details = pd.DataFrame(grid_rows_all)
    return tests, grid_details


def plot_summary(tests: pd.DataFrame, output_path: Path, dpi: int) -> None:
    sig = tests["q_derivative_MAE"] < 0.05
    q_clipped = tests["q_derivative_MAE"].clip(lower=0.0, upper=0.10)

    x = tests["null_mean_derivative_MAE"].to_numpy(dtype=float)
    y = tests["real_mean_derivative_MAE"].to_numpy(dtype=float)
    lim_min = float(np.nanmin(np.r_[x, y]))
    lim_max = float(np.nanmax(np.r_[x, y]))
    pad = max((lim_max - lim_min) * 0.08, 0.02)
    lims = (lim_min - pad, lim_max + pad)

    fig, ax = plt.subplots(figsize=(6.7, 6.05), constrained_layout=True)
    cmap = mpl.colormaps["viridis_r"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.10)

    ax.scatter(
        x[~sig],
        y[~sig],
        s=72,
        facecolors="white",
        edgecolors="#7a7a7a",
        linewidths=1.2,
        label="n.s.",
        zorder=2,
    )
    sc = ax.scatter(
        x[sig],
        y[sig],
        s=74,
        c=q_clipped[sig],
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.65,
        label="q < 0.05",
        zorder=3,
    )
    ax.plot(lims, lims, ls="--", lw=1.35, color="#666666", zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Null mean Derivative MAE")
    ax.set_ylabel("Real mean Derivative MAE")
    ax.set_title(f"Permutation test across {len(tests)} formal conditions", pad=12)

    sig_count = int(sig.sum())
    median_q = float(np.nanmedian(tests["q_derivative_MAE"]))
    ax.text(
        0.05,
        0.95,
        f"{sig_count}/{len(tests)} conditions q < 0.05\nmedian q = {median_q:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cmap(norm(0.0)), markeredgecolor="white", markersize=12, label="q < 0.05"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#7a7a7a", markersize=12, label="n.s."),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("q value", fontsize=21)
    cbar.set_ticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    cbar.ax.tick_params(labelsize=19)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    base = load_base_module()
    grid_scales = base.parse_int_list(args.grid_scales)
    direction_codes = base.parse_direction_list(args.directions)
    directions = [direction for direction in base.DIRECTION_CONFIG if direction["code"] in direction_codes]

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    tests, grid_details = compute_all_tests(
        base=base,
        mea_data_dir=args.mea_data_dir,
        grid_scales=grid_scales,
        directions=directions,
        epsilon=args.epsilon,
        min_units_per_grid_per_stim=args.min_units_per_grid_per_stim,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    tests.to_csv(table_dir / f"{MODULE_ID}_condition_tests.csv", index=False)
    grid_details.to_csv(table_dir / f"{MODULE_ID}_grid_details.csv", index=False)

    figure_path = figure_dir / f"{MODULE_ID}.png"
    plot_summary(tests, figure_path, args.dpi)
    pd.DataFrame(
        [
            {
                "figure_id": MODULE_ID,
                "condition_count": int(len(tests)),
                "q_lt_0_05_count": int((tests["q_derivative_MAE"] < 0.05).sum()),
                "median_q": float(np.nanmedian(tests["q_derivative_MAE"])),
                "png": str(figure_path),
            }
        ]
    ).to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "base_module_id": BASE_MODULE_ID,
        "mea_data_dir": str(args.mea_data_dir),
        "output_dir": str(args.output_dir),
        "pairs": base.PAIR_CONFIG,
        "grid_scales": grid_scales,
        "directions": directions,
        "response_metric": f"log((ON + {args.epsilon}) / (OFF + {args.epsilon}))",
        "main_grid_metric": "derivative_MAE",
        "main_grid_metric_formula": "mean_j |(T_UME_g(j+1) - T_UME_g(j)) - (T_CME_g(j+1) - T_CME_g(j))|",
        "test_statistic": "mean derivative_MAE across valid spatial grids within each condition",
        "permutation_unit": "shuffle UME/CME labels within each valid grid while preserving UME/CME unit counts",
        "min_units_per_grid_per_stim": int(args.min_units_per_grid_per_stim),
        "n_perm": int(args.n_perm),
        "random_seed": int(args.seed),
        "condition_seed_rule": "condition_seed = random_seed + condition_id",
        "fdr_method": "Benjamini-Hochberg over all plotted conditions",
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
