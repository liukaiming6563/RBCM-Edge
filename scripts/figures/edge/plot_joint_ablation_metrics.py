#!/usr/bin/env python3
"""Plot primary ODS/OIS/AP comparisons for H-RBCM ablations.

BIPED is reported as the arithmetic mean of three fixed split evaluations.
MultiCue is a single strict held-out evaluation. The paper-facing plots do not
infer or display uncertainty bars; split-level BIPED values remain available
in the machine-readable evidence.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of the expected result files exists:\n" + "\n".join(map(str, paths)))


PRIVATE_TABLES = ROOT / "edge_outputs" / "rbcm" / "tables"
OUT = ROOT / "edge_outputs" / "rbcm" / "figures" / "publication" / "results" / "joint_metrics"
BIPED_TABLE = first_existing(
    PRIVATE_TABLES / "biped_stability_ap_corrected.csv",
)
STRICT_TABLE = first_existing(
    PRIVATE_TABLES / "strict_protocols" / "same_domain_strict.csv",
)
AP_DEFINITION = "endpoint-complete precision-envelope AP"

MODE_ORDER = ["plain_identity", "no_surround", "conv_control", "main_surround"]
MODE_LABELS = {
    "plain_identity": "Anchor\n(plain identity)",
    "no_surround": "No surround",
    "conv_control": "Conv control",
    "main_surround": "H-RBCM",
}
METRICS = ["ODS", "OIS", "AP"]
METRIC_COLORS = {
    "ODS": "#4C78A8",
    "OIS": "#2A9D8F",
    "AP": "#D95F02",
}
MODEL_COLORS = {
    "plain_identity": "#4C78A8",
    "no_surround": "#2A9D8F",
    "conv_control": "#756BB1",
    "main_surround": "#D95F02",
}
MODEL_LEGEND_LABELS = {
    "plain_identity": "Anchor",
    "no_surround": "No surround",
    "conv_control": "Conv control",
    "main_surround": "H-RBCM",
}
INK = "#243746"
MUTED = "#81909B"
GRID = "#D9E1E5"
MAIN_BAND = "#FFF3E8"
NEUTRAL_BAND = "#F7F9FA"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DengXian", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.5,
            "axes.titlesize": 14.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def prepare_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    for i, mode in enumerate(MODE_ORDER):
        color = MAIN_BAND if mode == "main_surround" else NEUTRAL_BAND
        alpha = 0.95 if mode == "main_surround" else 0.55
        ax.axvspan(i - 0.43, i + 0.43, color=color, alpha=alpha, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(color=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(MODE_ORDER) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Score")
    ax.set_xticks(np.arange(len(MODE_ORDER)), [MODE_LABELS[mode] for mode in MODE_ORDER])


def finish_layout(fig: plt.Figure, handles: list[mpl.lines.Line2D], title: str) -> None:
    fig.suptitle(title, y=0.985, fontsize=14.0, fontweight="bold")
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.205, top=0.805)


def prepare_metric_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    """Prepare an axis grouped by metric, with four models inside each group."""
    for i in range(len(METRICS)):
        ax.axvspan(i - 0.43, i + 0.43, color=NEUTRAL_BAND, alpha=0.72, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(color=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(METRICS) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Score")
    ax.set_xticks(np.arange(len(METRICS)), METRICS)


def finish_metric_layout(fig: plt.Figure, handles: list[mpl.lines.Line2D], title: str) -> None:
    fig.suptitle(title, y=0.985, fontsize=14.0, fontweight="bold")
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.175, top=0.805)


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        ".png": {"dpi": 360},
        ".svg": {},
        ".pdf": {},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }
    for suffix, kwargs in outputs.items():
        fig.savefig(OUT / f"{stem}{suffix}", bbox_inches="tight", pad_inches=0.05, **kwargs)
    plt.close(fig)


def load_biped_means() -> pd.DataFrame:
    source = pd.read_csv(BIPED_TABLE)
    if {"split", "mode", "ODS", "OIS", "AP"}.issubset(source.columns):
        return (
            source[source["mode"].isin(MODE_ORDER)]
            .groupby("mode", as_index=False)[METRICS]
            .mean()
            .set_index("mode")
            .loc[MODE_ORDER]
            .reset_index()
        )
    return source.set_index("mode").loc[MODE_ORDER].reset_index()


def biped_score(row: pd.Series, metric: str) -> float:
    direct = metric
    aggregate = f"{metric}_mean"
    if direct in row.index:
        return float(row[direct])
    return float(row[aggregate])


def plot_biped() -> pd.DataFrame:
    source = load_biped_means()
    x = np.arange(len(MODE_ORDER), dtype=float)
    offsets = {"ODS": -0.22, "OIS": 0.0, "AP": 0.22}
    rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    prepare_axis(ax, (0.922, 0.981))

    for model_index, row in source.iterrows():
        for metric in METRICS:
            mean = biped_score(row, metric)
            xpos = x[model_index] + offsets[metric]
            color = METRIC_COLORS[metric]

            ax.hlines(
                mean,
                xpos - 0.075,
                xpos + 0.075,
                color=color,
                linewidth=2.2,
                zorder=3,
            )
            ax.scatter(
                xpos,
                mean,
                marker="D",
                s=50,
                color=color,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
            ax.text(
                xpos,
                mean + 0.0022,
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                color=color,
                fontsize=8.6,
                fontweight="bold" if row["mode"] == "main_surround" else "normal",
            )
            rows.append(
                {
                    "dataset": "BIPED",
                    "mode": row["mode"],
                    "metric": metric,
                    "mean": mean,
                    "n_runs": 3,
                    "display": "arithmetic mean across three fixed splits",
                    "ap_definition": AP_DEFINITION,
                }
            )

    handles = [
        mpl.lines.Line2D([], [], color=METRIC_COLORS[metric], marker="D", linestyle="-", linewidth=1.8,
                         markersize=6, label=metric)
        for metric in METRICS
    ]
    ax.text(
        0.01,
        0.018,
        "Arithmetic mean of three fixed splits; AP uses endpoint-complete envelope",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_layout(fig, handles, "BIPED: validation-frozen ODS, OIS and AP")
    export(fig, "11_biped_joint_ods_ois_ap")
    return pd.DataFrame(rows)


def plot_multicue() -> pd.DataFrame:
    source = pd.read_csv(STRICT_TABLE)
    source = source[source["dataset"].str.lower() == "multicue"].set_index("mode").loc[MODE_ORDER].reset_index()
    x = np.arange(len(MODE_ORDER), dtype=float)
    offsets = {"ODS": -0.22, "OIS": 0.0, "AP": 0.22}
    rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    prepare_axis(ax, (0.875, 0.948))

    for model_index, row in source.iterrows():
        for metric in METRICS:
            score = float(row[metric])
            xpos = x[model_index] + offsets[metric]
            color = METRIC_COLORS[metric]

            ax.hlines(score, xpos - 0.075, xpos + 0.075, color=color, linewidth=2.2, zorder=3)
            ax.scatter(
                xpos,
                score,
                marker="D",
                s=50,
                color=color,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
            ax.text(
                xpos,
                score + 0.0040,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                color=color,
                fontsize=8.6,
                fontweight="bold" if row["mode"] == "main_surround" else "normal",
            )
            rows.append(
                {
                    "dataset": "MultiCue",
                    "mode": row["mode"],
                    "metric": metric,
                    "score": score,
                    "n_images": int(row["n_images"]),
                    "n_runs": int(row["n_runs"]),
                    "source": row.get("source_file", row.get("source", "")),
                    "protocol": row["protocol"],
                    "display": "exact score tick; no variance inferred",
                    "ap_definition": AP_DEFINITION,
                }
            )

    handles = [
        mpl.lines.Line2D([], [], color=METRIC_COLORS[metric], marker="D", linestyle="-", linewidth=1.8,
                         markersize=6, label=metric)
        for metric in METRICS
    ]
    ax.text(
        0.01,
        0.018,
        "One-pass held-out evaluation (20 sources); no error bar inferred",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_layout(fig, handles, "MultiCue: strict held-out ODS, OIS and AP")
    export(fig, "12_multicue_joint_ods_ois_ap")
    return pd.DataFrame(rows)


def plot_biped_metric_grouped() -> pd.DataFrame:
    source = load_biped_means().set_index("mode")
    offsets = dict(zip(MODE_ORDER, [-0.27, -0.09, 0.09, 0.27]))
    rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    prepare_metric_axis(ax, (0.922, 0.981))

    for metric_index, metric in enumerate(METRICS):
        for mode in MODE_ORDER:
            row = source.loc[mode]
            mean = biped_score(row, metric)
            xpos = metric_index + offsets[mode]
            color = MODEL_COLORS[mode]

            ax.hlines(
                mean,
                xpos - 0.060,
                xpos + 0.060,
                color=color,
                linewidth=2.2,
                zorder=3,
            )
            ax.scatter(
                xpos,
                mean,
                marker="D",
                s=50,
                color=color,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
            ax.text(
                xpos,
                mean + 0.0020,
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                color=color,
                fontsize=8.3,
                fontweight="bold" if mode == "main_surround" else "normal",
            )
            rows.append(
                {
                    "dataset": "BIPED",
                    "metric": metric,
                    "mode": mode,
                    "mean": mean,
                    "n_runs": 3,
                    "display": "metric-grouped arithmetic mean across three fixed splits",
                    "ap_definition": AP_DEFINITION,
                }
            )

    handles = [
        mpl.lines.Line2D(
            [], [], color=MODEL_COLORS[mode], marker="D", linestyle="-", linewidth=1.8,
            markersize=6, label=MODEL_LEGEND_LABELS[mode]
        )
        for mode in MODE_ORDER
    ]
    ax.text(
        0.01,
        0.018,
        "Arithmetic mean of three fixed splits; AP uses endpoint-complete envelope",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_metric_layout(fig, handles, "BIPED: validation-frozen ablations by metric")
    export(fig, "13_biped_metric_grouped_models")
    return pd.DataFrame(rows)


def plot_multicue_metric_grouped() -> pd.DataFrame:
    source = pd.read_csv(STRICT_TABLE)
    source = source[source["dataset"].str.lower() == "multicue"].set_index("mode").loc[MODE_ORDER]
    offsets = dict(zip(MODE_ORDER, [-0.27, -0.09, 0.09, 0.27]))
    rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    prepare_metric_axis(ax, (0.875, 0.948))

    for metric_index, metric in enumerate(METRICS):
        for mode in MODE_ORDER:
            row = source.loc[mode]
            score = float(row[metric])
            xpos = metric_index + offsets[mode]
            color = MODEL_COLORS[mode]

            ax.hlines(score, xpos - 0.060, xpos + 0.060, color=color, linewidth=2.2, zorder=3)
            ax.scatter(
                xpos,
                score,
                marker="D",
                s=50,
                color=color,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
            ax.text(
                xpos,
                score + 0.0038,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                color=color,
                fontsize=8.3,
                fontweight="bold" if mode == "main_surround" else "normal",
            )
            rows.append(
                {
                    "dataset": "MultiCue",
                    "metric": metric,
                    "mode": mode,
                    "score": score,
                    "n_images": int(row["n_images"]),
                    "n_runs": int(row["n_runs"]),
                    "source": row.get("source_file", row.get("source", "")),
                    "protocol": row["protocol"],
                    "display": "metric-grouped exact score tick; no variance inferred",
                    "ap_definition": AP_DEFINITION,
                }
            )

    handles = [
        mpl.lines.Line2D(
            [], [], color=MODEL_COLORS[mode], marker="D", linestyle="-", linewidth=1.8,
            markersize=6, label=MODEL_LEGEND_LABELS[mode]
        )
        for mode in MODE_ORDER
    ]
    ax.text(
        0.01,
        0.018,
        "One-pass held-out evaluation (20 sources); no error bar inferred",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_metric_layout(fig, handles, "MultiCue: strict held-out ablations by metric")
    export(fig, "14_multicue_metric_grouped_models")
    return pd.DataFrame(rows)


def write_manifest() -> None:
    entries = [
        {
            "figure": "11_biped_joint_ods_ois_ap",
            "dataset": "BIPED",
            "source": BIPED_TABLE.relative_to(ROOT).as_posix(),
            "interpretation": "Arithmetic mean of three validation-frozen, mode-specific fixed splits.",
        },
        {
            "figure": "12_multicue_joint_ods_ois_ap",
            "dataset": "MultiCue",
            "source": STRICT_TABLE.relative_to(ROOT).as_posix(),
            "interpretation": "Strict 68/12/20 protocol; exact one-pass held-out scores.",
        },
        {
            "figure": "13_biped_metric_grouped_models",
            "dataset": "BIPED",
            "source": BIPED_TABLE.relative_to(ROOT).as_posix(),
            "interpretation": "Metrics define the x-axis groups; scores are validation-frozen three-split means.",
        },
        {
            "figure": "14_multicue_metric_grouped_models",
            "dataset": "MultiCue",
            "source": STRICT_TABLE.relative_to(ROOT).as_posix(),
            "interpretation": "Metrics define the x-axis groups; exact strict held-out scores.",
        },
    ]
    with (OUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)


def write_readmes() -> None:
    """Write notes for the current strict paper-facing figures."""
    zh = """# BIPED 与 MultiCue 三指标联合消融图

- `11_biped_joint_ods_ois_ap`：四种模型各自同时显示 ODS、OIS 和 AP。BIPED 只报告三个固定划分的算术平均，不在正文图中绘制标准差。
- `12_multicue_joint_ods_ois_ap`：显示严格 68/12/20 独立协议下的精确分数。该协议只有一个随机种子，不虚构误差线。
- `13_biped_metric_grouped_models`：横轴改为 ODS、OIS 和 AP，每组内并列四种模型。
- `14_multicue_metric_grouped_models`：采用同样的指标分组布局，使用一次性独立测试的精确分数。
- 每张图均导出 PNG、SVG、PDF 和 600 dpi TIFF，并保存对应源数据。
"""
    en = """# Joint BIPED and MultiCue ablation metrics

- `11_biped_joint_ods_ois_ap` shows the arithmetic mean of three fixed splits for Anchor, No surround, Conv control, and H-RBCM. The paper-facing figure does not display standard deviations.
- `12_multicue_joint_ods_ois_ap` uses the strict 68/12/20 source-disjoint protocol. It has one seed, so exact scores are shown without invented error bars.
- `13_biped_metric_grouped_models` switches the x-axis to ODS, OIS, and AP, with four models offset inside each metric group.
- `14_multicue_metric_grouped_models` uses the same metric-grouped layout and exact one-pass held-out scores.
- Each figure is exported as PNG, SVG, PDF, and 600-dpi TIFF, with source data in this directory.
"""
    (OUT / "README.zh-CN.md").write_text(zh, encoding="utf-8")
    (OUT / "README.md").write_text(en, encoding="utf-8")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_biped().to_csv(OUT / "11_biped_joint_ods_ois_ap_source.csv", index=False, encoding="utf-8-sig")
    plot_multicue().to_csv(OUT / "12_multicue_joint_ods_ois_ap_source.csv", index=False, encoding="utf-8-sig")
    plot_biped_metric_grouped().to_csv(
        OUT / "13_biped_metric_grouped_models_source.csv", index=False, encoding="utf-8-sig"
    )
    plot_multicue_metric_grouped().to_csv(
        OUT / "14_multicue_metric_grouped_models_source.csv", index=False, encoding="utf-8-sig"
    )
    write_manifest()
    write_readmes()
    print(f"Wrote joint metric figures and source data to {OUT}")


if __name__ == "__main__":
    main()
