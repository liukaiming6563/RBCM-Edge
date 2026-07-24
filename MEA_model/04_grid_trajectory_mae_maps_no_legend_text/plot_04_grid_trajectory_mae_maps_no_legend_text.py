"""Plot 04 grid trajectory MAE maps without in-panel unit-count legend text.

This is a visual-only derivative of module 04_grid_trajectory_mae_maps. It reads
the formal module-04 tables and run config, then redraws the same 56 trajectory
MAE maps with the same data, color scale, and unit-position overlays. The only
intentional visual change is removal of the lower-left in-panel legend text:
"UME, n=..." and "CME, n=...".
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODULE_ID = "04_grid_trajectory_mae_maps"
MODULE_ID = "04_grid_trajectory_mae_maps_no_legend_text"

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

UNIT_COLORS = {
    "UME": "#f4c430",
    "CME": "#48c7bf",
}

DEFAULT_ARGS = {
    "source_output_dir": PROJECT_ROOT / "MEA_outputs" / SOURCE_MODULE_ID,
    "output_dir": PROJECT_ROOT / "MEA_outputs" / MODULE_ID,
    "dpi": 300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, default=DEFAULT_ARGS["source_output_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    return parser.parse_args()


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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def mae_colormap() -> mpl.colors.Colormap:
    """Use the exact color sequence from module 04."""

    colors = ["#083f4a", "#0b5a63", "#16c783", "#f0ef5a", "#fff46a"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trajectory_mae_teal_green_yellow",
        colors,
        N=256,
    )
    cmap.set_bad("#083f4a")
    return cmap


def load_source_data(source_output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    table_dir = source_output_dir / "tables"
    run_config_path = source_output_dir / f"{SOURCE_MODULE_ID}_run_config.json"
    required = {
        "grid_cells": table_dir / f"{SOURCE_MODULE_ID}_grid_cells.csv",
        "summary": table_dir / f"{SOURCE_MODULE_ID}_summary.csv",
        "unit_summary": table_dir / f"{SOURCE_MODULE_ID}_unit_summary.csv",
        "unit_positions": table_dir / f"{SOURCE_MODULE_ID}_unit_positions_overlay.csv",
        "run_config": run_config_path,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required module-04 source output file(s) missing:\n" + "\n".join(missing))

    grid_cells = pd.read_csv(required["grid_cells"])
    summary = pd.read_csv(required["summary"])
    unit_summary = pd.read_csv(required["unit_summary"])
    unit_positions = pd.read_csv(required["unit_positions"])
    run_config = json.loads(required["run_config"].read_text(encoding="utf-8"))
    return grid_cells, summary, unit_summary, run_config | {"unit_positions_table": unit_positions}


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

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.825, top=0.80, bottom=0.08, wspace=0.17)

    image = None
    for ax, (pair_id, _pair_cfg) in zip(axes, PAIR_CONFIG.items()):
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


def copy_source_tables(
    source_output_dir: Path,
    output_dir: Path,
    figure_manifest: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    source_table_dir = source_output_dir / "tables"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_names = [
        "grid_cells",
        "summary",
        "unit_summary",
        "unit_positions_overlay",
    ]
    for name in table_names:
        source_path = source_table_dir / f"{SOURCE_MODULE_ID}_{name}.csv"
        dest_path = table_dir / f"{MODULE_ID}_{name}.csv"
        shutil.copyfile(source_path, dest_path)

    figure_manifest.to_csv(table_dir / f"{MODULE_ID}_figure_manifest.csv", index=False)

    out_config = dict(run_config)
    out_config.update(
        {
            "module_id": MODULE_ID,
            "source_module_id": SOURCE_MODULE_ID,
            "source_output_dir": str(source_output_dir),
            "output_dir": str(output_dir),
            "visual_derivative": True,
            "visual_change": "Removed in-panel UME/CME unit-count legend text; heatmap values and unit-position overlays are unchanged.",
            "unit_count_legend_text": "removed",
            "figure_formats": ["png"],
        }
    )
    out_config.pop("unit_positions_table", None)
    (output_dir / f"{MODULE_ID}_run_config.json").write_text(
        json.dumps(out_config, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_style()

    grid_cells, summary, unit_summary, config = load_source_data(args.source_output_dir)
    unit_positions = config["unit_positions_table"]

    grid_scales = [int(x) for x in config.get("grid_scales", sorted(grid_cells["grid_n"].unique()))]
    direction_codes = [str(direction["code"]) for direction in config.get("directions", DIRECTION_CONFIG)]
    directions = [direction for direction in DIRECTION_CONFIG if direction["code"] in direction_codes]
    if not directions:
        raise ValueError("No direction codes available for plotting")

    vmax = float(config.get("vmax_used", np.nan))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanpercentile(grid_cells["trajectory_MAE"].to_numpy(dtype=float), 98.0))

    figure_dir = args.output_dir / "figures"
    figure_rows: list[dict[str, object]] = []
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

    copy_source_tables(
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        figure_manifest=pd.DataFrame(figure_rows),
        run_config=config,
    )

    print(f"Saved {len(figure_rows)} figures to {figure_dir}")
    print(f"Saved copied data tables to {args.output_dir / 'tables'}")
    print(f"Source data rows: grid_cells={len(grid_cells)}, summary={len(summary)}, unit_summary={len(unit_summary)}")


if __name__ == "__main__":
    main()
