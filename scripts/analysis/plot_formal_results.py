"""Create reproducible paper panels for the formal H-RBCM experiments.

The script only reads canonical, locally reproduced result files.  It does not
retrain, recalibrate, or modify any score.  Quantitative panels focus on the two
paper-facing training sources (BIPED and MultiCue), while complete matched
numbers remain available in the companion tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper_assets" / "rbcm" / "tables"
CORE = TABLES / "two_dataset_core"
PRED_ROOT = ROOT / "results" / "rbcm" / "predictions" / "biped_multicue"
DATA_ROOT = ROOT / "edge_data" / "official_rbcm"
OUT = ROOT / "paper_assets" / "rbcm" / "figures" / "results"
SOURCE_OUT = OUT / "source_data"

COLORS = {
    "main_surround": "#D95F02",
    "plain_identity": "#4C78A8",
    "no_surround": "#2A9D8F",
    "conv_control": "#7566A8",
    "gold": "#F2C14E",
    "ink": "#243746",
    "muted": "#81909B",
    "grid": "#D9E1E5",
    "negative": "#4C78A8",
    "positive": "#D95F02",
}

LABELS = {
    "main_surround": "H-RBCM",
    "plain_identity": "Anchor",
    "no_surround": "No surround",
    "conv_control": "Conv control",
}

TARGET_ORDER = ["BIPED", "Multicue", "NYUDv2", "BSDS500", "UDED"]
MODE_ORDER = ["plain_identity", "no_surround", "conv_control", "main_surround"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DengXian", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.5,
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(color=COLORS["muted"], labelcolor=COLORS["ink"])
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.75, linestyle="--", alpha=0.75)
    ax.set_axisbelow(True)


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "png": {"dpi": 360},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)


def save_source(df: pd.DataFrame, name: str) -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_OUT / name, index=False, encoding="utf-8-sig")


def plot_biped_stability() -> None:
    df = pd.read_csv(TABLES / "biped_stability_ap_corrected.csv")
    df["mode"] = pd.Categorical(df["mode"], MODE_ORDER, ordered=True)
    df = df.sort_values("mode")
    save_source(df, "01_biped_stability.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), constrained_layout=True)
    x = np.arange(len(df))
    for ax, metric in zip(axes, ["ODS", "OIS"]):
        for i, row in enumerate(df.itertuples(index=False)):
            mode = str(row.mode)
            vals = np.asarray([float(v) for v in getattr(row, f"{metric}_values").split(";")])
            jitter = np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full_like(vals, i, dtype=float) + jitter,
                vals,
                s=34,
                color=COLORS[mode],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            mean = float(getattr(row, f"{metric}_mean"))
            std = float(getattr(row, f"{metric}_std"))
            ax.errorbar(i, mean, yerr=std, fmt="D", ms=6.3, color=COLORS[mode],
                        ecolor=COLORS[mode], capsize=4, zorder=4)
        ax.set_xticks(x, [LABELS[str(v)] for v in df["mode"]], rotation=18, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"BIPED three-split {metric}")
        clean_axis(ax)
        lo = min(df[f"{metric}_mean"] - df[f"{metric}_std"]) - 0.004
        hi = max(df[f"{metric}_mean"] + df[f"{metric}_std"]) + 0.004
        ax.set_ylim(lo, hi)
    export(fig, "01_biped_stability")


def plot_multicue_ablation() -> None:
    df = pd.read_csv(TABLES / "self_test_ap_corrected.csv")
    df = df[df["dataset"].str.lower() == "multicue"].copy()
    df["mode"] = pd.Categorical(df["mode"], MODE_ORDER, ordered=True)
    df = df.sort_values("mode")
    save_source(df, "02_multicue_ablation.csv")

    fig, ax = plt.subplots(figsize=(5.25, 4.2))
    metrics = ["ODS", "OIS", "AP"]
    x = np.arange(len(metrics))
    offsets = np.linspace(-0.25, 0.25, len(df))
    for offset, row in zip(offsets, df.itertuples(index=False)):
        mode = str(row.mode)
        vals = [float(getattr(row, m)) for m in metrics]
        ax.plot(x + offset, vals, "o", ms=8.0, color=COLORS[mode],
                markeredgecolor="white", markeredgewidth=0.8, label=LABELS[mode], zorder=3)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.82, 0.94)
    ax.set_title("Matched ablation on MultiCue")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    clean_axis(ax)
    export(fig, "02_multicue_ablation")


def pr_path(mode: str) -> Path:
    return (
        PRED_ROOT
        / "generalization"
        / "official"
        / "multicue_selected"
        / "Multicue"
        / mode
        / "pr_curve_as_is.csv"
    )


def plot_multicue_pr() -> None:
    fig, ax = plt.subplots(figsize=(5.15, 4.25))
    source_rows: list[pd.DataFrame] = []
    for mode in MODE_ORDER:
        df = pd.read_csv(pr_path(mode)).sort_values("recall")
        df["mode"] = mode
        source_rows.append(df)
        ax.plot(df["recall"], df["precision"], color=COLORS[mode],
                linewidth=2.3 if mode == "main_surround" else 1.75,
                label=LABELS[mode], alpha=1.0 if mode == "main_surround" else 0.88)
        best = df.loc[df["f1"].idxmax()]
        ax.scatter(best["recall"], best["precision"], s=48, color=COLORS[mode],
                   edgecolor="white", linewidth=0.8, zorder=4)
    save_source(pd.concat(source_rows, ignore_index=True), "03_multicue_pr_curves.csv")
    ax.set_xlim(0.54, 1.005)
    ax.set_ylim(0.58, 1.005)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("MultiCue precision-recall curves")
    ax.legend(frameon=False, loc="lower left")
    clean_axis(ax, grid_axis="both")
    ax.set_aspect("equal", adjustable="box")
    export(fig, "03_multicue_pr_curves")


def draw_heatmap(ax: plt.Axes, matrix: np.ndarray, rows: list[str], cols: list[str], title: str) -> None:
    vmax = max(0.1, float(np.nanmax(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list("delta", [COLORS["negative"], "#F7F8F6", COLORS["positive"]])
    im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.65 * vmax else COLORS["ink"]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", color=color,
                    fontsize=10.5, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("H-RBCM gain (percentage points)")
    cbar.outline.set_visible(False)


def plot_multicue_control_generalization() -> None:
    df = pd.read_csv(CORE / "03_main_vs_strongest_control.csv")
    df = df[df["training_source"] == "Multicue"].copy()
    df["target_dataset"] = pd.Categorical(df["target_dataset"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target_dataset")
    save_source(df, "04_multicue_vs_strongest_control.csv")
    matrix = df[["Delta_ODS", "Delta_OIS", "Delta_AP"]].to_numpy(float) * 100.0
    fig, ax = plt.subplots(figsize=(5.5, 4.15))
    draw_heatmap(ax, matrix, df["target_dataset"].astype(str).tolist(), ["ODS", "OIS", "AP"],
                 "MultiCue-trained H-RBCM vs strongest control")
    export(fig, "04_multicue_control_generalization")


def plot_biped_fscore_generalization() -> None:
    df = pd.read_csv(CORE / "03_main_vs_strongest_control.csv")
    df = df[df["training_source"] == "BIPED"].copy()
    df["target_dataset"] = pd.Categorical(df["target_dataset"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target_dataset")
    save_source(df, "05_biped_fscore_generalization.csv")

    fig, ax = plt.subplots(figsize=(5.35, 4.15))
    y = np.arange(len(df))
    for metric, offset, color, marker in [
        ("Delta_ODS", -0.10, COLORS["main_surround"], "o"),
        ("Delta_OIS", 0.10, COLORS["no_surround"], "D"),
    ]:
        vals = df[metric].to_numpy(float) * 100.0
        ax.hlines(y + offset, 0, vals, color=color, linewidth=2.0, alpha=0.72)
        ax.scatter(vals, y + offset, s=58, color=color, marker=marker, edgecolor="white",
                   linewidth=0.8, label=metric.removeprefix("Delta_"), zorder=3)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.9)
    ax.set_yticks(y, df["target_dataset"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Gain over strongest matched control (percentage points)")
    ax.set_title("BIPED-trained F-score gains")
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax, grid_axis="x")
    export(fig, "05_biped_fscore_generalization")


def plot_pidinet_delta() -> None:
    """Expand the former PiDiNet-only panel with released-checkpoint references.

    All rows use the same five target datasets and the same local evaluator.
    Their source training recipes differ, so this panel is a released-checkpoint
    generalization comparison rather than a matched internal ablation.
    """
    h_df = pd.read_csv(TABLES / "fair_generalization" / "main_multicue_generalization.csv")
    h_df = h_df.set_index("target_dataset")
    ext_primary = pd.read_csv(TABLES / "external_models_full.csv")
    ext_added = pd.read_csv(TABLES / "external_additions" / "unified_local_scores.csv")

    selected = [
        {
            "model": "PiDiNet (official Table-7 Multicue checkpoint)",
            "display": "PiDiNet\n(MultiCue)*",
            "source": "MultiCue",
            "table": "primary",
        },
        {
            "model": "TEED (official BIPED checkpoint)",
            "display": "TEED\n(BIPED)",
            "source": "BIPED",
            "table": "primary",
        },
        {
            "model": "BDCN (official BSDS+PASCAL checkpoint)",
            "display": "BDCN\n(BSDS+PASCAL)",
            "source": "BSDS+PASCAL",
            "table": "primary",
        },
        {
            "model": "RCF official (BSDS+PASCAL)",
            "display": "RCF\n(BSDS+PASCAL)",
            "source": "BSDS+PASCAL",
            "table": "added",
        },
        {
            "model": "CATS official (NYUDv2)",
            "display": "CATS\n(NYUDv2)",
            "source": "NYUDv2",
            "table": "added",
        },
    ]
    long_rows: list[dict[str, object]] = []
    mean_rows: list[list[float]] = []
    win_rows: list[list[float]] = []
    labels: list[str] = []
    for spec in selected:
        table = ext_primary if spec["table"] == "primary" else ext_added
        model_col = "model"
        sub = table[table[model_col] == spec["model"]].copy()
        sub = sub.set_index("target_dataset").reindex(TARGET_ORDER)
        if sub[["ODS", "OIS", "AP"]].isna().any().any():
            raise RuntimeError(f"Incomplete five-target results for {spec['model']}")
        model_means: list[float] = []
        model_wins: list[float] = []
        for metric in ["ODS", "OIS", "AP"]:
            h_values = h_df.reindex(TARGET_ORDER)[f"H_{metric}"].to_numpy(float)
            ext_values = sub[metric].to_numpy(float)
            deltas = h_values - ext_values
            model_means.append(float(deltas.mean() * 100.0))
            model_wins.append(float((deltas > 0).sum()))
            for target, h_value, ext_value, delta in zip(TARGET_ORDER, h_values, ext_values, deltas):
                long_rows.append(
                    {
                        "external_model": spec["model"],
                        "external_source": spec["source"],
                        "target_dataset": target,
                        "metric": metric,
                        "H_RBCM": h_value,
                        "external": ext_value,
                        "delta": delta,
                        "H_RBCM_wins": bool(delta > 0),
                        "same_target_evaluator": True,
                        "identical_source_training_recipe": False,
                        "source_category_match": bool(spec["source"] == "MultiCue"),
                    }
                )
        mean_rows.append(model_means)
        win_rows.append(model_wins)
        labels.append(str(spec["display"]))

    source_df = pd.DataFrame(long_rows)
    save_source(source_df, "06_multicue_vs_external_models.csv")
    save_source(source_df, "06_multicue_vs_pidinet.csv")

    mean_matrix = np.asarray(mean_rows, dtype=float)
    win_matrix = np.asarray(win_rows, dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(mean_matrix))))
    cmap = LinearSegmentedColormap.from_list(
        "external_delta", [COLORS["negative"], "#F7F8F6", COLORS["positive"]]
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.65),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )
    im = axes[0].imshow(
        mean_matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        aspect="auto",
    )
    axes[0].set_xticks(np.arange(3), ["ODS", "OIS", "AP"])
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].set_title("Mean H-RBCM gain across five targets")
    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            value = mean_matrix[i, j]
            color = "white" if abs(value) > 0.62 * vmax else COLORS["ink"]
            axes[0].text(j, i, f"{value:+.1f}", ha="center", va="center", color=color,
                         fontsize=10.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.035)
    cbar.set_label("Percentage points")
    cbar.outline.set_visible(False)

    wins_cmap = LinearSegmentedColormap.from_list("wins", ["#F4F4F1", "#F2C14E", "#D95F02"])
    axes[1].imshow(win_matrix, cmap=wins_cmap, vmin=0, vmax=5, aspect="auto")
    axes[1].set_xticks(np.arange(3), ["ODS", "OIS", "AP"])
    axes[1].set_yticks(np.arange(len(labels)), [""] * len(labels))
    axes[1].set_title("Target wins out of five")
    for i in range(win_matrix.shape[0]):
        for j in range(win_matrix.shape[1]):
            value = int(win_matrix[i, j])
            axes[1].text(j, i, f"{value}/5", ha="center", va="center",
                         color="white" if value >= 4 else COLORS["ink"],
                         fontsize=10.5, fontweight="bold")
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("MultiCue-trained H-RBCM versus selected released checkpoints", fontsize=14,
                 fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "* PiDiNet is source-category matched. All rows share the target evaluator; source training recipes differ.",
        ha="center",
        fontsize=9.3,
        color=COLORS["muted"],
    )
    export(fig, "06_multicue_vs_pidinet")


def plot_external_ap() -> None:
    df = pd.read_csv(CORE / "04_external_fair_all_targets.csv")
    df = df[df["training_source"] == "Multicue"].copy()
    df["target_dataset"] = pd.Categorical(df["target_dataset"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target_dataset")
    save_source(df, "07_multicue_external_ap.csv")
    fig, ax = plt.subplots(figsize=(5.35, 4.1))
    y = np.arange(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        ax.plot([row.Ext_AP, row.H_AP], [i, i], color=COLORS["grid"], linewidth=3.0, zorder=1)
    ax.scatter(df["Ext_AP"], y, s=68, color=COLORS["plain_identity"], marker="o",
               edgecolor="white", linewidth=0.8, label="PiDiNet", zorder=3)
    ax.scatter(df["H_AP"], y, s=76, color=COLORS["main_surround"], marker="D",
               edgecolor="white", linewidth=0.8, label="H-RBCM", zorder=3)
    ax.set_yticks(y, df["target_dataset"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Average precision")
    ax.set_title("MultiCue-source AP across target datasets")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    clean_axis(ax, grid_axis="x")
    export(fig, "07_multicue_external_ap")


def plot_modulation_states() -> None:
    df = pd.read_csv(TABLES / "mechanism.csv")
    df = df[(df["group"].str.startswith("multicue_selected->")) & (df["mode_vs_plain"] == "main_surround")].copy()
    df["target"] = df["group"].str.split("->").str[-1]
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target")
    save_source(df, "08_modulation_states.csv")

    fig, ax = plt.subplots(figsize=(5.45, 4.15))
    y = np.arange(len(df))
    left = np.zeros(len(df))
    items = [
        ("enhance_fraction_mean", "Enhance", "#E07A3F"),
        ("suppress_fraction_mean", "Suppress", "#7566A8"),
        ("neutral_fraction_mean", "Neutral", "#AAB4B9"),
    ]
    for col, label, color in items:
        vals = df[col].to_numpy(float) * 100.0
        ax.barh(y, vals, left=left, height=0.62, color=color, label=label, edgecolor="white", linewidth=0.6)
        left += vals
    ax.set_yticks(y, df["target"].astype(str))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pixels (%)")
    ax.set_title("Signed surround-modulation states")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    clean_axis(ax, grid_axis="x")
    export(fig, "08_modulation_states")


def plot_uncertainty_focus() -> None:
    df = pd.read_csv(TABLES / "mechanism.csv")
    df = df[df["group"].str.startswith("multicue_selected->")].copy()
    df["target"] = df["group"].str.split("->").str[-1]
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df["mode_vs_plain"] = pd.Categorical(
        df["mode_vs_plain"], ["no_surround", "conv_control", "main_surround"], ordered=True
    )
    df = df.sort_values(["target", "mode_vs_plain"])
    save_source(df, "09_uncertainty_focus.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.15))
    x = np.arange(len(TARGET_ORDER))
    offsets = [-0.22, 0.0, 0.22]
    modes = ["no_surround", "conv_control", "main_surround"]
    for offset, mode in zip(offsets, modes):
        sub = df[df["mode_vs_plain"] == mode].set_index("target").reindex(TARGET_ORDER)
        ax.plot(x + offset, sub["uncertainty_focus_ratio_mean"], "o", ms=7.5,
                color=COLORS[mode], markeredgecolor="white", markeredgewidth=0.8,
                label=LABELS[mode], zorder=3)
    ax.axhline(1.0, color=COLORS["muted"], linestyle="--", linewidth=1.1)
    ax.set_xticks(x, TARGET_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Uncertainty-focus ratio")
    ax.set_title("Modulation concentrates near uncertain edges")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean_axis(ax)
    export(fig, "09_uncertainty_focus")


def load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def binary_f1(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    p = pred >= threshold
    g = gt >= 0.5
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)


def best_pr_threshold(mode: str) -> float:
    df = pd.read_csv(pr_path(mode))
    return float(df.loc[df["f1"].idxmax(), "threshold"])


def select_multicue_examples(n: int = 3) -> pd.DataFrame:
    pred_root = (
        PRED_ROOT / "generalization" / "apply" / "multicue_selected" / "Multicue" / "predictions"
    )
    test_ids = [line.strip() for line in (DATA_ROOT / "Multicue" / "splits" / "test.txt").read_text().splitlines() if line.strip()]
    thresholds = {mode: best_pr_threshold(mode) for mode in MODE_ORDER}
    rows: list[dict[str, float | str]] = []
    for sample_id in test_ids:
        gt_path = DATA_ROOT / "Multicue" / "edge" / f"{sample_id}.png"
        if not gt_path.exists():
            continue
        # Rank examples against the strong binary edge target.  The
        # multi-annotator soft-vote map is shown in the final panel, while the
        # ignore-weight map remains exclusively a training loss mask.
        gt = load_gray(gt_path)
        scores: dict[str, float] = {}
        ok = True
        for mode in MODE_ORDER:
            p = pred_root / mode / f"{sample_id}.png"
            if not p.exists():
                ok = False
                break
            scores[mode] = binary_f1(load_gray(p), gt, thresholds[mode])
        if not ok:
            continue
        strongest_control = max(scores[m] for m in ["plain_identity", "no_surround", "conv_control"])
        rows.append(
            {
                "sample_id": sample_id,
                **{f"f1_{m}": v for m, v in scores.items()},
                "gain_vs_strongest_control": scores["main_surround"] - strongest_control,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["gain_vs_strongest_control", "f1_main_surround"], ascending=False
    )
    # Spread the examples through the positive half rather than taking adjacent ranks.
    positive = ranked[ranked["gain_vs_strongest_control"] > 0].reset_index(drop=True)
    if len(positive) >= n:
        idx = np.linspace(0, min(len(positive) - 1, max(n * 2, n - 1)), n).round().astype(int)
        return positive.iloc[idx].copy()
    return ranked.head(n).copy()


def plot_multicue_qualitative() -> None:
    selected = select_multicue_examples(3)
    save_source(selected, "10_multicue_qualitative_selection.csv")
    pred_root = PRED_ROOT / "generalization" / "apply" / "multicue_selected" / "Multicue" / "predictions"
    thresholds = {mode: best_pr_threshold(mode) for mode in MODE_ORDER}
    columns = ["Input", "GT", "Anchor", "Conv control", "H-RBCM", "RBCM - control"]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(12.0, 2.25 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    for r, row in enumerate(selected.itertuples(index=False)):
        sample_id = row.sample_id
        image = np.asarray(Image.open(DATA_ROOT / "Multicue" / "image" / f"{sample_id}.png").convert("RGB"))
        gt = load_gray(DATA_ROOT / "Multicue" / "gt" / "soft_vote" / f"{sample_id}.png")
        anchor = load_gray(pred_root / "plain_identity" / f"{sample_id}.png")
        control = load_gray(pred_root / "conv_control" / f"{sample_id}.png")
        main = load_gray(pred_root / "main_surround" / f"{sample_id}.png")
        displays = [
            (image, None),
            (gt, "gray"),
            ((anchor >= thresholds["plain_identity"]).astype(float), "gray"),
            ((control >= thresholds["conv_control"]).astype(float), "gray"),
            ((main >= thresholds["main_surround"]).astype(float), "gray"),
        ]
        for c, (arr, cmap) in enumerate(displays):
            axes[r, c].imshow(arr, cmap=cmap, vmin=0, vmax=1)
        delta = main - control
        lim = max(float(np.quantile(np.abs(delta), 0.985)), 0.08)
        axes[r, 5].imshow(delta, cmap="coolwarm", vmin=-lim, vmax=lim)
        for c, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(columns[c], fontsize=12, pad=6)
        axes[r, 0].set_ylabel(sample_id, fontsize=10.5, rotation=90, labelpad=8)
    fig.suptitle("Representative MultiCue examples at fixed source-level ODS thresholds", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=0.35, h_pad=0.55)
    export(fig, "10_multicue_qualitative")


def write_readmes() -> None:
    zh = """# 正式模型实验结果小图

本目录由 `scripts/analysis/plot_formal_results.py` 从本地正式评估文件自动生成。所有定量图均读取 CSV 原始结果，不在画图阶段重新校准或修改得分。每张图同时提供 PNG、SVG、PDF 和 600 dpi TIFF；`source_data/` 保存对应数据子集与定性样例选择清单。

1. `01_biped_stability`：BIPED 三个固定划分上的 ODS/OIS 均值、标准差和逐划分值，表达主模型 F-score 提升具有跨划分稳定性。AP 的完整结果保留在论文表格中。
2. `02_multicue_ablation`：MultiCue 同域匹配消融，H-RBCM 在 ODS、OIS、AP 三项均优于三个共享锚点的对照模式。
3. `03_multicue_pr_curves`：MultiCue 四种模式的统一评估 PR 曲线，曲线上标记各自最大 F1 点。
4. `04_multicue_control_generalization`：MultiCue 训练后，H-RBCM 相对每个指标上最强匹配对照的百分点变化；橙色为提升，蓝色为下降。
5. `05_biped_fscore_generalization`：BIPED 训练后，在五个目标集上的 ODS/OIS 相对最强匹配对照提升。该图只检验 F-score 主张，AP 仍在完整表格中报告。
6. `06_multicue_vs_pidinet`：同为 MultiCue 来源模型时，H-RBCM 与官方 PiDiNet 在五个目标集上的全指标差值，完整呈现 AP 优势与 ODS/OIS 的混合结果。
7. `07_multicue_external_ap`：MultiCue 来源模型在五个目标集上的 AP 哑铃图，突出 H-RBCM 的跨域排序质量。
8. `08_modulation_states`：主模型在五个目标集上增强、抑制和中性调制像素比例，直接对应三状态生物启发解释。
9. `09_uncertainty_focus`：不确定像素与确定像素上的平均调制强度比。虚线 1 表示无偏向，大于 1 表示更新集中在锚点不确定区域。
10. `10_multicue_qualitative`：按固定源域 ODS 阈值显示的代表性正例。样例依据主模型相对最强控制的像素级 F1 增益排序，仅用于可复现选图，不替代正式容差匹配评估。差异图中暖色表示 H-RBCM 概率更高，冷色表示控制模型更高。

术语：ODS 为数据集统一阈值下的最优 F1；OIS 为逐图最优 F1 的平均；AP 为整条 PR 曲线下的平均精度。所有“gain”均为同一数据、同一评估器下的绝对百分点差。
"""
    en = """# Formal experiment-result panels

This directory is generated by `scripts/analysis/plot_formal_results.py` from the canonical local evaluation files. Plotting never recalibrates or changes a score. Each panel is exported as PNG, SVG, PDF, and 600-dpi TIFF, with the exact plotted rows stored under `source_data/`.

The panels cover BIPED split stability, matched MultiCue ablation and PR curves, matched cross-dataset control gains, the source-matched PiDiNet comparison, signed modulation states, uncertainty focus, and reproducibly selected qualitative examples. ODS is the best dataset-scale F1, OIS is the mean image-scale best F1, and AP summarizes the precision-recall ranking. Warm heat-map values are absolute percentage-point gains; cool values are losses.
"""
    (OUT / "README.zh-CN.md").write_text(zh, encoding="utf-8")
    (OUT / "README.md").write_text(en, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size})
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "figure_contract.json").write_text(
        json.dumps(
            {
                "backend": "Python/matplotlib",
                "core_claim": "Annular surround-to-center modulation improves matched F-score performance and selected cross-domain generalization without changing the shared HED-lite anchor.",
                "primary_sources": ["BIPED", "Multicue"],
                "target_datasets": TARGET_ORDER,
                "exports": ["png", "svg", "pdf", "tiff"],
                "note": "Figures emphasize supported claims; complete matched metrics and limitations remain in the companion tables.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    plot_biped_stability()
    plot_multicue_ablation()
    plot_multicue_pr()
    plot_multicue_control_generalization()
    plot_biped_fscore_generalization()
    plot_pidinet_delta()
    plot_external_ap()
    plot_modulation_states()
    plot_uncertainty_focus()
    plot_multicue_qualitative()
    write_readmes()
    write_manifest()
    print(f"Wrote formal result panels to {OUT}")


if __name__ == "__main__":
    main()
