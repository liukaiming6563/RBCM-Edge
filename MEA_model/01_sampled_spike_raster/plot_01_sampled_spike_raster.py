"""Plot sampled raw spike rasters for all MEA experiments.

This is manuscript MEA figure module 01. It visualizes raw Kilosort spike
times for a fixed-size sampled subset of good and MUA units from each
experiment. The sampling seed is fixed so the same units are selected every
time the script is rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ARGS = {
    "mea_data_dir": PROJECT_ROOT / "MEA_data",
    "output_dir": PROJECT_ROOT / "MEA_outputs" / "01_sampled_spike_raster",
    "experiments": [f"{idx:06d}" for idx in range(31, 40)],
    "seed": 20260602,
    "sample_rate_hz": 20000.0,
    "max_good_units": 90,
    "max_mua_units": 45,
    "dpi": 300,
}

STIMULUS_BY_EXPERIMENT = {
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

COLORS = {
    "good": "#d45a32",
    "mua": "#3b75a8",
    "stimulus": "#d8c6e6",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mea-data-dir", type=Path, default=DEFAULT_ARGS["mea_data_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_ARGS["experiments"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_ARGS["sample_rate_hz"])
    parser.add_argument("--max-good-units", type=int, default=DEFAULT_ARGS["max_good_units"])
    parser.add_argument("--max-mua-units", type=int, default=DEFAULT_ARGS["max_mua_units"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_ARGS["dpi"])
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cluster_labels(kilosort_dir: Path) -> pd.DataFrame:
    """Load cluster labels from Kilosort/Phy TSV files."""
    tables: list[pd.DataFrame] = []
    for filename in ["cluster_group.tsv", "cluster_KSLabel.tsv"]:
        path = kilosort_dir / filename
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
        tables.append(table)

    if not tables:
        raise FileNotFoundError(f"No cluster label TSV files found in {kilosort_dir}")

    merged = tables[0].copy()
    for table in tables[1:]:
        merged = merged.merge(table, on="cluster_id", how="outer", suffixes=("", "_alt"))
        if "label_alt" in merged.columns:
            merged["label"] = merged["label"].fillna(merged["label_alt"])
            merged = merged.drop(columns=["label_alt"])

    merged["cluster_id"] = merged["cluster_id"].astype(int)
    merged["label"] = merged["label"].astype(str).str.lower()
    return merged.sort_values("cluster_id").reset_index(drop=True)


def load_sample_rate(exp_dir: Path, fallback_hz: float) -> float:
    """Read sample rate from metadata when available."""
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        return fallback_hz
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback_hz
    return float(metadata.get("sample_rate_hz", fallback_hz))


def load_stimulus_window(exp_dir: Path) -> tuple[float, float] | None:
    """Return the detected stimulus-video window in seconds when available."""
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    timing = metadata.get("stimulus_timing", {})
    start = timing.get("stim_start_sec_recording")
    end = timing.get("stim_end_sec_recording_detected")
    if start is None or end is None:
        return None
    return float(start), float(end)


def choose_units(
    labels: pd.DataFrame,
    exp_id: str,
    seed: int,
    max_good_units: int,
    max_mua_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select sampled good and MUA cluster ids with a fixed seed."""
    rng = np.random.default_rng(seed + int(exp_id))
    good_ids = labels.loc[labels["label"].eq("good"), "cluster_id"].to_numpy(dtype=int)
    mua_ids = labels.loc[labels["label"].eq("mua"), "cluster_id"].to_numpy(dtype=int)

    good_n = min(max_good_units, good_ids.size)
    mua_n = min(max_mua_units, mua_ids.size)
    good_selected = np.sort(rng.choice(good_ids, size=good_n, replace=False)) if good_n else np.array([], dtype=int)
    mua_selected = np.sort(rng.choice(mua_ids, size=mua_n, replace=False)) if mua_n else np.array([], dtype=int)
    return good_selected, mua_selected


def selected_spikes(
    spike_times_s: np.ndarray,
    spike_clusters: np.ndarray,
    unit_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    """Return spike times for selected units."""
    unit_ids = np.asarray(list(unit_ids), dtype=int)
    if unit_ids.size == 0:
        return {}
    keep = np.isin(spike_clusters, unit_ids)
    kept_times = spike_times_s[keep]
    kept_clusters = spike_clusters[keep].astype(int)
    return {
        int(unit_id): kept_times[kept_clusters == int(unit_id)]
        for unit_id in unit_ids
    }


def plot_raster(
    exp_id: str,
    stimulus: str,
    spike_times_s: np.ndarray,
    spike_clusters: np.ndarray,
    labels: pd.DataFrame,
    good_selected: np.ndarray,
    mua_selected: np.ndarray,
    stimulus_window: tuple[float, float] | None,
    output_stem: Path,
    dpi: int,
) -> dict[str, object]:
    """Plot and save one sampled raw spike raster."""
    good_total = int(labels["label"].eq("good").sum())
    mua_total = int(labels["label"].eq("mua").sum())
    selected_units = np.concatenate([good_selected, mua_selected])
    unit_spikes = selected_spikes(spike_times_s, spike_clusters, selected_units)

    good_n = int(good_selected.size)
    mua_n = int(mua_selected.size)
    total_n = good_n + mua_n
    max_time = float(np.nanmax(spike_times_s)) if spike_times_s.size else 0.0
    x_limit = max(1.0, np.ceil(max_time / 50.0) * 50.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)

    if stimulus_window is not None:
        ax.axvspan(
            stimulus_window[0],
            stimulus_window[1],
            color=COLORS["stimulus"],
            alpha=0.35,
            lw=0,
            zorder=0,
        )

    for row_idx, unit_id in enumerate(good_selected, start=1):
        times = unit_spikes.get(int(unit_id), np.array([], dtype=float))
        if times.size:
            ax.scatter(
                times,
                np.full(times.size, row_idx),
                s=1.0,
                marker=".",
                color=COLORS["good"],
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )

    for offset, unit_id in enumerate(mua_selected, start=1):
        row_idx = good_n + offset
        times = unit_spikes.get(int(unit_id), np.array([], dtype=float))
        if times.size:
            ax.scatter(
                times,
                np.full(times.size, row_idx),
                s=1.0,
                marker=".",
                color=COLORS["mua"],
                alpha=0.70,
                linewidths=0,
                rasterized=True,
            )

    ax.set_xlim(0, x_limit)
    ax.set_ylim(0.5, total_n + 0.5)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Sampled units", fontsize=13)
    ax.set_yticks([1, max(1, good_n), total_n])
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    good_y = max(1.5, min(good_n * 0.08, good_n - 1.0 if good_n > 3 else good_n))
    mua_y = total_n - max(1.0, mua_n * 0.10)
    ax.text(x_limit * 0.01, good_y, "Good", color=COLORS["good"], fontsize=11, weight="bold")
    ax.text(x_limit * 0.01, mua_y, "MUA", color=COLORS["mua"], fontsize=11, weight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["good"], markeredgecolor="none", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["mua"], markeredgecolor="none", markersize=6),
    ]
    ax.legend(
        legend_handles,
        [f"good: {good_n}/{good_total}", f"MUA: {mua_n}/{mua_total}"],
        frameon=False,
        loc="upper right",
        fontsize=10,
        handletextpad=0.5,
    )
    ax.set_title(f"Experiment {exp_id} ({stimulus})", fontsize=11, pad=8)

    png_path = output_stem.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)

    return {
        "experiment_id": exp_id,
        "stimulus": stimulus,
        "good_total": good_total,
        "mua_total": mua_total,
        "good_sampled": good_n,
        "mua_sampled": mua_n,
        "duration_s": max_time,
        "stimulus_start_s": stimulus_window[0] if stimulus_window else np.nan,
        "stimulus_end_s": stimulus_window[1] if stimulus_window else np.nan,
        "png": str(png_path),
    }


def main() -> None:
    """Generate sampled raw spike raster figures for all requested experiments."""
    args = parse_args()
    mea_data_dir = resolve_project_path(args.mea_data_dir)
    output_dir = resolve_project_path(args.output_dir)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    sampled_rows: list[dict[str, object]] = []

    for exp_id in args.experiments:
        exp_dir = mea_data_dir / exp_id
        kilosort_dir = exp_dir / "kilosort4"
        if not kilosort_dir.exists():
            raise FileNotFoundError(f"Missing Kilosort directory: {kilosort_dir}")

        labels = load_cluster_labels(kilosort_dir)
        sample_rate_hz = load_sample_rate(exp_dir, args.sample_rate_hz)
        spike_times_samples = np.load(kilosort_dir / "spike_times.npy").reshape(-1)
        spike_clusters = np.load(kilosort_dir / "spike_clusters.npy").reshape(-1).astype(int)
        spike_times_s = spike_times_samples.astype(np.float64) / sample_rate_hz
        stimulus_window = load_stimulus_window(exp_dir)

        good_selected, mua_selected = choose_units(
            labels,
            exp_id,
            args.seed,
            args.max_good_units,
            args.max_mua_units,
        )
        for label, unit_ids in [("good", good_selected), ("mua", mua_selected)]:
            for row_idx, cluster_id in enumerate(unit_ids, start=1):
                sampled_rows.append(
                    {
                        "experiment_id": exp_id,
                        "stimulus": STIMULUS_BY_EXPERIMENT.get(exp_id, "unknown"),
                        "label": label,
                        "sample_order_within_label": row_idx,
                        "cluster_id": int(cluster_id),
                        "seed": int(args.seed),
                    }
                )

        output_stem = figure_dir / f"01_sampled_spike_raster_exp{exp_id}"
        manifest_rows.append(
            plot_raster(
                exp_id=exp_id,
                stimulus=STIMULUS_BY_EXPERIMENT.get(exp_id, "unknown"),
                spike_times_s=spike_times_s,
                spike_clusters=spike_clusters,
                labels=labels,
                good_selected=good_selected,
                mua_selected=mua_selected,
                stimulus_window=stimulus_window,
                output_stem=output_stem,
                dpi=args.dpi,
            )
        )
        print(f"Wrote raster for experiment {exp_id}")

    pd.DataFrame(manifest_rows).to_csv(table_dir / "01_sampled_spike_raster_figure_manifest.csv", index=False)
    pd.DataFrame(sampled_rows).to_csv(table_dir / "01_sampled_spike_raster_sampled_units.csv", index=False)

    run_config = {
        "module": "01_sampled_spike_raster",
        "seed": int(args.seed),
        "experiments": list(args.experiments),
        "max_good_units": int(args.max_good_units),
        "max_mua_units": int(args.max_mua_units),
        "mea_data_dir": str(mea_data_dir),
        "output_dir": str(output_dir),
        "sample_rate_hz_default": float(args.sample_rate_hz),
    }
    (output_dir / "01_sampled_spike_raster_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
