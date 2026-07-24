"""Formal representative permutation-null histograms for Trajectory MAE."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ID = "13_permutation_null_examples"
SOURCE_MODULE_ID = "05_trajectory_mae_permutation_test_summary"
SOURCE_SCRIPT = (
    PROJECT_ROOT
    / "MEA_model"
    / SOURCE_MODULE_ID
    / "plot_05_trajectory_mae_permutation_test_summary.py"
)

DEFAULT_ARGS = {
    "mea_data_dir": PROJECT_ROOT / "MEA_data",
    "mea_outputs_dir": PROJECT_ROOT / "MEA_outputs",
    "output_dir": PROJECT_ROOT / "MEA_outputs" / MODULE_ID,
    "epsilon": 0.1,
    "min_units_per_grid_per_stim": 3,
    "n_perm": 5000,
    "q_threshold": 0.05,
    "dpi": 300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mea-data-dir", type=Path, default=DEFAULT_ARGS["mea_data_dir"])
    parser.add_argument("--mea-outputs-dir", type=Path, default=DEFAULT_ARGS["mea_outputs_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_ARGS["epsilon"])
    parser.add_argument(
        "--min-units-per-grid-per-stim",
        type=int,
        default=DEFAULT_ARGS["min_units_per_grid_per_stim"],
    )
    parser.add_argument("--n-perm", type=int, default=DEFAULT_ARGS["n_perm"])
    parser.add_argument("--q-threshold", type=float, default=DEFAULT_ARGS["q_threshold"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 19,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 17.5,
            "ytick.labelsize": 17.5,
            "legend.fontsize": 17.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_source_module() -> ModuleType:
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(f"Source permutation script not found: {SOURCE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("trajectory_mae_permutation_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_condition_tests(mea_outputs_dir: Path) -> pd.DataFrame:
    table_path = (
        mea_outputs_dir
        / SOURCE_MODULE_ID
        / "tables"
        / f"{SOURCE_MODULE_ID}_condition_tests.csv"
    )
    if not table_path.exists():
        raise FileNotFoundError(f"Required condition table not found: {table_path}")
    tests = pd.read_csv(table_path)
    tests["delta_real_null"] = tests["real_mean_trajectory_MAE"] - tests["null_mean_trajectory_MAE"]
    return tests


def select_examples(tests: pd.DataFrame, q_threshold: float) -> pd.DataFrame:
    sig = tests[tests["q_trajectory_MAE"] < q_threshold].copy()
    if sig.empty:
        raise ValueError("No significant conditions available for representative examples")

    strong_idx = sig["delta_real_null"].idxmax()
    median_delta = float(sig["delta_real_null"].median())
    typical_idx = (sig["delta_real_null"] - median_delta).abs().idxmin()
    borderline_idx = (tests["q_trajectory_MAE"] - q_threshold).abs().idxmin()

    selected = tests.loc[[strong_idx, typical_idx, borderline_idx]].copy()
    selected.insert(0, "example_type", ["strong_effect", "typical_effect", "borderline_q"])
    selected = selected.drop_duplicates("condition_id").reset_index(drop=True)
    return selected


def permutation_null_values(
    real_grid_mae: list[float],
    perm_arrays: list[tuple[np.ndarray, int]],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, float]:
    if not real_grid_mae or not perm_arrays:
        return np.nan, np.full(n_perm, np.nan), np.nan

    real = float(np.nanmean(real_grid_mae))
    null_sum = np.zeros(n_perm, dtype=float)

    for arr, ume_n in perm_arrays:
        arr = np.asarray(arr, dtype=float)
        total_n = arr.shape[0]
        cme_n = total_n - ume_n
        if ume_n <= 0 or cme_n <= 0:
            continue

        order = np.argsort(rng.random((n_perm, total_n)), axis=1)
        ume_idx = order[:, :ume_n]
        ume_sum = arr[ume_idx].sum(axis=1)
        total_sum = arr.sum(axis=0)
        cme_sum = total_sum - ume_sum
        diff = (ume_sum / ume_n) - (cme_sum / cme_n)
        null_sum += np.nanmean(np.abs(diff), axis=1)

    null_values = null_sum / len(perm_arrays)
    p_value = float((np.sum(null_values >= real) + 1) / (n_perm + 1))
    return real, null_values, p_value


def compute_examples(
    source: ModuleType,
    selected: pd.DataFrame,
    mea_data_dir: Path,
    epsilon: float,
    min_units_per_grid_per_stim: int,
    n_perm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = source.load_base_module()
    position_cache: dict[str, pd.DataFrame] = {}
    trajectory_cache: dict[tuple[str, str, int], np.ndarray] = {}
    phase_cache: dict[tuple[str, str], np.ndarray] = {}

    summary_rows: list[dict[str, object]] = []
    null_rows: list[pd.DataFrame] = []

    direction_lookup = {direction["code"]: direction for direction in base.DIRECTION_CONFIG}
    for idx, row in selected.iterrows():
        direction = direction_lookup[str(row["direction_code"])]
        real_grid_mae, perm_arrays, grid_rows = source.condition_grid_arrays(
            base=base,
            mea_data_dir=mea_data_dir,
            pair_id=str(row["pair_id"]),
            grid_n=int(row["grid_n"]),
            direction=direction,
            epsilon=epsilon,
            min_units_per_grid_per_stim=min_units_per_grid_per_stim,
            position_cache=position_cache,
            trajectory_cache=trajectory_cache,
            phase_cache=phase_cache,
        )
        condition_seed = int(row["condition_seed"])
        real, null_values, p_recomputed = permutation_null_values(
            real_grid_mae=real_grid_mae,
            perm_arrays=perm_arrays,
            n_perm=n_perm,
            rng=np.random.default_rng(condition_seed),
        )
        example_id = f"example_{idx + 1}_{row['example_type']}"
        summary_rows.append(
            {
                "example_id": example_id,
                "example_type": row["example_type"],
                "condition_id": int(row["condition_id"]),
                "condition_seed": condition_seed,
                "pair_id": row["pair_id"],
                "pair_label": row["pair_label"],
                "grid_n": int(row["grid_n"]),
                "direction_code": row["direction_code"],
                "valid_grid_count": int(len(real_grid_mae)),
                "real_stat_recomputed": real,
                "null_mean_recomputed": float(np.nanmean(null_values)),
                "null_std_recomputed": float(np.nanstd(null_values, ddof=1)),
                "p_recomputed": p_recomputed,
                "real_stat_source": float(row["real_mean_trajectory_MAE"]),
                "null_mean_source": float(row["null_mean_trajectory_MAE"]),
                "p_source": float(row["p_trajectory_MAE"]),
                "q_source": float(row["q_trajectory_MAE"]),
                "delta_source": float(row["delta_real_null"]),
                "grid_count": len(grid_rows),
            }
        )
        null_rows.append(
            pd.DataFrame(
                {
                    "example_id": example_id,
                    "perm_idx": np.arange(n_perm, dtype=int),
                    "null_trajectory_MAE": null_values,
                }
            )
        )

    return pd.DataFrame(summary_rows), pd.concat(null_rows, ignore_index=True)


def plot_null_examples(
    summary: pd.DataFrame,
    null_values: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> dict[str, object]:
    n_examples = len(summary)
    fig, axes = plt.subplots(1, n_examples, figsize=(5.15 * n_examples, 4.35), constrained_layout=True)
    if n_examples == 1:
        axes = [axes]

    for plot_idx, (ax, row) in enumerate(zip(axes, summary.itertuples(index=False))):
        vals = null_values.loc[
            null_values["example_id"].eq(row.example_id), "null_trajectory_MAE"
        ].to_numpy(dtype=float)
        counts, bins, patches = ax.hist(
            vals,
            bins=44,
            color="#a9c8df",
            edgecolor="white",
            linewidth=0.45,
            alpha=0.36,
            label="permutation null",
        )
        count_max = float(np.nanmax(counts))
        x_min = float(np.nanmin(vals))
        x_max = max(float(np.nanmax(vals)), float(row.real_stat_recomputed))
        x_span = max(x_max - x_min, 1e-6)
        ax.set_xlim(x_min - 0.06 * x_span, x_max + 0.13 * x_span)
        ax.set_ylim(0, count_max * 1.15)

        ax.axvline(row.null_mean_recomputed, color="#455a64", lw=1.8, ls="--", alpha=0.82, label="null mean")
        real_top = count_max * 0.86
        ax.plot(
            [row.real_stat_recomputed, row.real_stat_recomputed],
            [0, real_top],
            color="#d15f00",
            lw=4.6,
            solid_capstyle="round",
            label="real statistic",
            zorder=6,
        )
        ax.scatter(
            [row.real_stat_recomputed],
            [real_top],
            marker="o",
            s=88,
            color="#d15f00",
            edgecolors="white",
            linewidths=0.9,
            zorder=7,
        )
        ax.annotate(
            "real",
            xy=(row.real_stat_recomputed, real_top),
            xytext=(-10, 18),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=17,
            color="#9f3f00",
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#d15f00",
                "lw": 1.35,
                "shrinkA": 1,
                "shrinkB": 4,
            },
            zorder=7,
        )
        title = (
            f"{row.example_type.replace('_', ' ')}\n"
            f"{row.pair_label}, grid {int(row.grid_n)}x{int(row.grid_n)}, dir {row.direction_code}"
        )
        ax.set_title(title, pad=9)
        ax.set_xlabel("Null mean Trajectory MAE")
        if plot_idx == 0:
            ax.set_ylabel("Permutation count")
        else:
            ax.set_ylabel("")
        ax.text(
            0.04,
            0.08,
            f"real={row.real_stat_recomputed:.3f}\n"
            f"null={row.null_mean_recomputed:.3f}\n"
            f"p={row.p_recomputed:.4f}\n"
            f"q={row.q_source:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=16.5,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "none", "alpha": 0.82},
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e1e1e1", lw=0.75, alpha=0.62)

    axes[0].legend(frameon=False, loc="upper left", handlelength=2.0)
    fig.suptitle("Representative permutation-null distributions for Trajectory MAE", y=1.055, fontsize=24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {
        "figure_id": output_path.stem,
        "png": str(output_path),
        "example_count": int(n_examples),
    }


def main() -> None:
    args = parse_args()
    set_style()

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_module()
    tests = load_condition_tests(args.mea_outputs_dir)
    selected = select_examples(tests, args.q_threshold)
    summary, null_values = compute_examples(
        source=source,
        selected=selected,
        mea_data_dir=args.mea_data_dir,
        epsilon=args.epsilon,
        min_units_per_grid_per_stim=args.min_units_per_grid_per_stim,
        n_perm=args.n_perm,
    )

    figure_path = figure_dir / f"{MODULE_ID}.png"
    figure_row = plot_null_examples(summary, null_values, figure_path, args.dpi)

    selected.to_csv(table_dir / f"{MODULE_ID}_selected_conditions.csv", index=False)
    summary.to_csv(table_dir / f"{MODULE_ID}_example_summary.csv", index=False)
    null_values.to_csv(table_dir / f"{MODULE_ID}_null_values.csv", index=False)
    pd.DataFrame([figure_row]).to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    run_config = {
        "module_id": MODULE_ID,
        "source_module": SOURCE_MODULE_ID,
        "source_script": str(SOURCE_SCRIPT),
        "mea_data_dir": str(args.mea_data_dir),
        "mea_outputs_dir": str(args.mea_outputs_dir),
        "output_dir": str(args.output_dir),
        "metric": "trajectory_MAE",
        "example_selection": [
            "max delta_real_null among q<0.05 conditions",
            "condition closest to median significant delta_real_null",
            "condition with q closest to 0.05",
        ],
        "epsilon": float(args.epsilon),
        "min_units_per_grid_per_stim": int(args.min_units_per_grid_per_stim),
        "n_perm": int(args.n_perm),
        "q_threshold": float(args.q_threshold),
        "random_seed": "uses original condition_seed from module 05 for each selected condition",
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
