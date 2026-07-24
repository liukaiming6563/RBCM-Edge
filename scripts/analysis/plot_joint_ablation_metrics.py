#!/usr/bin/env python3
"""Plot endpoint-corrected ODS/OIS/AP comparisons for H-RBCM ablations.

BIPED contains three fixed split evaluations, so the figure shows every split,
the mean, and +/-1 sample SD. MultiCue has one formal selected-checkpoint
evaluation, so only exact score ticks are shown; no variance is inferred.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper_assets" / "rbcm" / "tables"
OUT = ROOT / "paper_assets" / "rbcm" / "figures" / "results" / "joint_metrics"
BIPED_TABLE = TABLES / "biped_stability_ap_corrected.csv"
SELF_TEST_TABLE = TABLES / "self_test_ap_corrected.csv"
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


def plot_biped() -> pd.DataFrame:
    source = pd.read_csv(BIPED_TABLE)
    source = source.set_index("mode").loc[MODE_ORDER].reset_index()
    x = np.arange(len(MODE_ORDER), dtype=float)
    offsets = {"ODS": -0.22, "OIS": 0.0, "AP": 0.22}
    rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    prepare_axis(ax, (0.922, 0.981))

    for model_index, row in source.iterrows():
        for metric in METRICS:
            values = np.asarray([float(value) for value in str(row[f"{metric}_values"]).split(";")])
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            xpos = x[model_index] + offsets[metric]
            jitter = np.linspace(-0.035, 0.035, len(values))
            color = METRIC_COLORS[metric]

            ax.scatter(
                xpos + jitter,
                values,
                s=25,
                color=color,
                alpha=0.58,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.errorbar(
                xpos,
                mean,
                yerr=std,
                fmt="D",
                markersize=6.2,
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4.0,
                capthick=1.6,
                markeredgecolor="white",
                markeredgewidth=0.65,
                zorder=4,
            )
            ax.text(
                xpos,
                mean + std + 0.0022,
                f"{mean:.3f}",
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
                    "std": std,
                    "values": ";".join(f"{value:.6f}" for value in values),
                    "n_runs": len(values),
                    "display": "raw split points, mean diamond, +/-1 sample SD",
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
        "Small circles: fixed splits   Diamond: mean   Whisker: +/-1 SD   AP: endpoint-complete envelope",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_layout(fig, handles, "BIPED: joint ODS, OIS and AP across three fixed splits")
    export(fig, "11_biped_joint_ods_ois_ap")
    return pd.DataFrame(rows)


def plot_multicue() -> pd.DataFrame:
    source = pd.read_csv(SELF_TEST_TABLE)
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
                f"{score:.3f}",
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
                    "source": row["source"],
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
        "Single selected-checkpoint evaluation (20 images); no error bar inferred; AP uses endpoint-complete envelope",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_layout(fig, handles, "MultiCue: joint ODS, OIS and AP under the formal evaluator")
    export(fig, "12_multicue_joint_ods_ois_ap")
    return pd.DataFrame(rows)


def plot_biped_metric_grouped() -> pd.DataFrame:
    source = pd.read_csv(BIPED_TABLE)
    source = source.set_index("mode").loc[MODE_ORDER]
    offsets = dict(zip(MODE_ORDER, [-0.27, -0.09, 0.09, 0.27]))
    rows: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    prepare_metric_axis(ax, (0.922, 0.981))

    for metric_index, metric in enumerate(METRICS):
        for mode in MODE_ORDER:
            row = source.loc[mode]
            values = np.asarray([float(value) for value in str(row[f"{metric}_values"]).split(";")])
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            xpos = metric_index + offsets[mode]
            jitter = np.linspace(-0.024, 0.024, len(values))
            color = MODEL_COLORS[mode]

            ax.scatter(
                xpos + jitter,
                values,
                s=25,
                color=color,
                alpha=0.58,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.errorbar(
                xpos,
                mean,
                yerr=std,
                fmt="D",
                markersize=6.2,
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4.0,
                capthick=1.6,
                markeredgecolor="white",
                markeredgewidth=0.65,
                zorder=4,
            )
            ax.text(
                xpos,
                mean + std + 0.0020,
                f"{mean:.3f}",
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
                    "std": std,
                    "values": ";".join(f"{value:.6f}" for value in values),
                    "n_runs": len(values),
                    "display": "metric-grouped raw split points, mean diamond, +/-1 sample SD",
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
        "Small circles: fixed splits   Diamond: mean   Whisker: +/-1 SD   AP: endpoint-complete envelope",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_metric_layout(fig, handles, "BIPED: four ablation models grouped by evaluation metric")
    export(fig, "13_biped_metric_grouped_models")
    return pd.DataFrame(rows)


def plot_multicue_metric_grouped() -> pd.DataFrame:
    source = pd.read_csv(SELF_TEST_TABLE)
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
                f"{score:.3f}",
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
                    "source": row["source"],
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
        "Single selected-checkpoint evaluation (20 images); no error bar inferred; AP uses endpoint-complete envelope",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
        ha="left",
        va="bottom",
    )
    finish_metric_layout(fig, handles, "MultiCue: four ablation models grouped by evaluation metric")
    export(fig, "14_multicue_metric_grouped_models")
    return pd.DataFrame(rows)


def write_manifest() -> None:
    entries = [
        {
            "figure": "11_biped_joint_ods_ois_ap",
            "dataset": "BIPED",
            "source": "paper_assets/rbcm/tables/biped_stability_ap_corrected.csv",
            "interpretation": "Three fixed splits; AP uses endpoint-complete precision-envelope integration.",
        },
        {
            "figure": "12_multicue_joint_ods_ois_ap",
            "dataset": "MultiCue",
            "source": "paper_assets/rbcm/tables/self_test_ap_corrected.csv",
            "interpretation": "One selected-checkpoint evaluation; AP uses endpoint-complete precision-envelope integration.",
        },
        {
            "figure": "13_biped_metric_grouped_models",
            "dataset": "BIPED",
            "source": "paper_assets/rbcm/tables/biped_stability_ap_corrected.csv",
            "interpretation": "Metrics define the x-axis groups; AP is endpoint-complete and precision-enveloped.",
        },
        {
            "figure": "14_multicue_metric_grouped_models",
            "dataset": "MultiCue",
            "source": "paper_assets/rbcm/tables/self_test_ap_corrected.csv",
            "interpretation": "Metrics define the x-axis groups; exact scores use the corrected common AP definition.",
        },
    ]
    with (OUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)


def _write_legacy_readmes() -> None:
    zh = """# BIPED 与 MultiCue 三指标联合消融图

- `11_biped_joint_ods_ois_ap`：横轴为 Anchor、No surround、Conv control 和 H-RBCM；每个模型同时显示 ODS、OIS、AP。小圆点为三次固定划分原始值，菱形为均值，误差短线为正负一个样本标准差。
- `12_multicue_joint_ods_ois_ap`：横轴和指标定义相同。MultiCue 只有一次正式选定 checkpoint 评估，因此短横线和菱形表示精确分数，不虚构误差条或重复实验分布。
- `13_biped_metric_grouped_models`：横轴改为 ODS、OIS、AP 三组，每组内部并列四种模型；统计含义与 BIPED 上一版相同。
- `14_multicue_metric_grouped_models`：横轴改为 ODS、OIS、AP 三组，每组内部并列四种模型；仅显示正式评估精确值。
- 每张图均导出 PNG、SVG、PDF 和 600 dpi TIFF，源数据保存在同一目录。
"""
    en = """# Joint BIPED and MultiCue ablation metrics

- `11_biped_joint_ods_ois_ap` shows ODS, OIS, and AP for Anchor, No surround, Conv control, and H-RBCM. Small circles are the three fixed-split values, diamonds are means, and whiskers are +/-1 sample SD.
- `12_multicue_joint_ods_ois_ap` uses the same model and metric layout. MultiCue has one formal selected-checkpoint evaluation, so ticks and diamonds are exact scores; no error distribution is inferred.
- `13_biped_metric_grouped_models` switches the x-axis to ODS, OIS, and AP, with four models offset inside each metric group. Statistical marks retain their BIPED meanings.
- `14_multicue_metric_grouped_models` uses the same metric-grouped layout and exact formal MultiCue scores.
- Each figure is exported as PNG, SVG, PDF, and 600-dpi TIFF, with source data in this directory.
"""
    (OUT / "README.zh-CN.md").write_text(zh, encoding="utf-8")
    (OUT / "README.md").write_text(en, encoding="utf-8")


def write_readmes() -> None:
    zh = """# BIPED 与 MultiCue 三指标联合消融图

- `11_biped_joint_ods_ois_ap`：横轴为四种消融模型，每种模型同时展示 ODS、OIS 和 AP。小圆点表示三个固定划分，菱形表示均值，误差线表示正负一个样本标准差。
- `12_multicue_joint_ods_ois_ap`：布局相同。MultiCue 只有一次选定 checkpoint 的 20 图像评估，因此只显示精确分数，不虚构误差线。
- `13_biped_metric_grouped_models`：横轴改为 ODS、OIS 和 AP，每组内并列四种模型。
- `14_multicue_metric_grouped_models`：采用同样的指标分组布局，显示四种模型的精确分数。
- 所有模型和划分统一使用补齐 PR 端点并构造 precision envelope 后的 AP；ODS 和 OIS 未改动。
- 每张图均导出 PNG、SVG、PDF 和 600 dpi TIFF，源数据保存在同一目录。
- 当前 MultiCue 的验证列表与测试列表相同，因此图中只称为“20 图像评估”，不能表述为独立测试集结果。
"""
    en = """# Joint BIPED and MultiCue ablation metrics

- `11_biped_joint_ods_ois_ap` shows ODS, OIS, and AP for Anchor, No surround, Conv control, and H-RBCM. Small circles are the three fixed-split values, diamonds are means, and whiskers are +/-1 sample SD.
- `12_multicue_joint_ods_ois_ap` uses the same layout. MultiCue has one selected-checkpoint 20-image evaluation, so ticks and diamonds are exact scores; no error distribution is inferred.
- `13_biped_metric_grouped_models` switches the x-axis to ODS, OIS, and AP, with four models offset inside each metric group.
- `14_multicue_metric_grouped_models` uses the same metric-grouped layout and exact MultiCue scores.
- AP is uniformly recomputed for every model with PR sentinels and a precision envelope; ODS and OIS are unchanged.
- Each figure is exported as PNG, SVG, PDF, and 600-dpi TIFF, with source data in this directory.
- MultiCue validation and test lists are currently identical, so these figures do not describe the 20-image evaluation as an independent test set.
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
