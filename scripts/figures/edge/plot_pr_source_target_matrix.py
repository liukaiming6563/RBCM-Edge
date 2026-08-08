"""Plot the complete H-RBCM source-target precision-recall matrix.

The matrix contains three training sources (BIPED, strict MultiCue, and
strict NYUDv2) and five evaluation targets (BIPED, MultiCue, NYUDv2,
BSDS500, and UDED).  Every visible line is loaded from a recorded
per-threshold CSV.  Scalar ODS/OIS/AP values are never used to reconstruct a
curve.

Released external checkpoints retain their own training sources.  They are
shown as broad target-evaluator references; the controlled comparison is the
four matched H-RBCM modes within each source-target panel.
"""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.plot_pr_comparison import (  # noqa: E402
    CurveSpec,
    MODE_COLORS,
    MODE_LABELS,
    MODES,
    configure_style,
    display_curve,
    external_path,
    load_spec_curve,
    p,
    style_axis,
)


OUT = ROOT / "edge_outputs" / "rbcm" / "figures" / "publication" / "pr_source_target_matrix"
SOURCE_OUT = OUT / "source_data"
ALL_PNG_OUT = OUT / "all_png"

SOURCES = ("BIPED", "Multicue", "NYUDv2")
TARGETS = ("BIPED", "Multicue", "NYUDv2", "BSDS500", "UDED")
SOURCE_LABELS = {
    "BIPED": "BIPED-trained",
    "Multicue": "MultiCue-trained",
    "NYUDv2": "NYUDv2-trained",
}
TARGET_LABELS = {
    "BIPED": "BIPED",
    "Multicue": "MultiCue",
    "NYUDv2": "NYUDv2",
    "BSDS500": "BSDS500",
    "UDED": "UDED",
}

SAME_DOMAIN_SCORES = (
    ROOT / "edge_outputs" / "rbcm" / "tables"
    / "strict_protocols"
    / "same_domain_strict.csv"
)


@dataclass(frozen=True)
class Panel:
    index: int
    source: str
    target: str

    @property
    def stem(self) -> str:
        source = self.source.lower().replace("v2", "v2")
        target = self.target.lower().replace("v2", "v2")
        return f"{self.index:02d}_{source}_to_{target}_pr"

    @property
    def title(self) -> str:
        relation = "in-domain" if self.source == self.target else "cross-domain"
        return (
            f"{SOURCE_LABELS[self.source]} -> "
            f"{TARGET_LABELS[self.target]} ({relation})"
        )


def same_domain_scores() -> dict[tuple[str, str], float]:
    frame = pd.read_csv(SAME_DOMAIN_SCORES)
    return {
        (str(row.dataset), str(row.mode)): float(row.ODS)
        for row in frame.itertuples(index=False)
    }


def internal_curve_path(
    source: str,
    target: str,
    mode: str,
) -> Path | tuple[Path, ...]:
    """Return the formal internal curve source for one source-target pair."""

    if source == "BIPED":
        if target == "BIPED":
            base = p(
                "results",
                "rbcm",
                "predictions",
                "integrity_ringfix_20260723",
                "stability",
                "official",
            )
            return tuple(
                base / f"split{split_index}" / mode / "pr_curve_as_is.csv"
                for split_index in range(3)
            )
        return p(
            "results",
            "rbcm",
            "predictions",
            "biped_multicue",
            "generalization",
            "official",
            "biped_selected",
            target,
            mode,
            "pr_curve_as_is.csv",
        )

    if source == "Multicue":
        if target == "Multicue":
            return p(
                "results",
                "rbcm",
                "runs",
                "multicue_strict_20260726",
                "minimal",
                "metrics",
                "independent_test",
                "official",
                mode,
                "pr_curve_as_is.csv",
            )
        return p(
            "results",
            "rbcm",
            "predictions",
            "multicue_strict_seed4517_generalization5",
            "official49",
            target,
            mode,
            "pr_curve_as_is.csv",
        )

    if source == "NYUDv2":
        if target == "NYUDv2":
            return p(
                "results",
                "rbcm",
                "evaluations",
                "pr_comparison_20260729",
                "NYUDv2",
                mode,
                "pr_curve_as_is.csv",
            )
        return p(
            "results",
            "rbcm",
            "predictions",
            "nyudv2_strict_seed4517_20260725_generalization5",
            "official49",
            target,
            mode,
            "pr_curve_as_is.csv",
        )

    raise KeyError((source, target, mode))


def traditional_specs(target: str) -> list[CurveSpec]:
    if target in {"BIPED", "Multicue", "NYUDv2"}:
        sobel = CurveSpec(
            "Sobel",
            external_path(
                "edge_outputs/external/evaluations/sobel_fixed_gradient",
                target,
            ),
            "traditional",
            "fixed non-trained operator",
            "#303030",
            (0, (2, 1.4)),
            1.45,
        )
        canny_root = (
            "edge_outputs/external/evaluations/canny_fixed_sweep"
            if target == "BIPED"
            else "edge_outputs/external/evaluations_fair_20260720/canny_fixed_sweep"
        )
        canny = CurveSpec(
            "Canny",
            external_path(canny_root, target),
            "traditional",
            "fixed non-trained operator",
            "#B33B66",
            "--",
            1.55,
        )
        return [sobel, canny]

    return [
        CurveSpec(
            "Canny",
            external_path(
                "edge_outputs/external/evaluations/canny_fixed_sweep",
                target,
            ),
            "traditional",
            "fixed non-trained operator",
            "#B33B66",
            "--",
            1.55,
        )
    ]


def external_specs(target: str) -> list[CurveSpec]:
    if target == "BIPED":
        return [
            CurveSpec(
                "PiDiNet (BSDS)",
                external_path(
                    "edge_outputs/external/evaluations/pidinet_official_bsds",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#E69F00",
                "--",
            ),
            CurveSpec(
                "TEED (BIPED)",
                external_path(
                    "edge_outputs/external/evaluations/teed_official_biped",
                    target,
                ),
                "external",
                "released BIPED checkpoint",
                "#C44E52",
                "--",
            ),
            CurveSpec(
                "DexiNed (BIPED)",
                external_path(
                    "edge_outputs/external/evaluations/dexined_official_biped",
                    target,
                ),
                "external",
                "released BIPED checkpoint",
                "#00A6D6",
                "--",
            ),
            CurveSpec(
                "RCF",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "rcf_official_bsds_pascal",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#4D7C0F",
                "--",
            ),
            CurveSpec(
                "BDCN",
                external_path(
                    "edge_outputs/external/evaluations/bdcn_official_bsds",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#D4C200",
                "--",
            ),
            CurveSpec(
                "UAED",
                external_path(
                    "edge_outputs/external/evaluations/uaed_official_bsds",
                    target,
                ),
                "external",
                "released BSDS checkpoint",
                "#8C8C8C",
                "--",
            ),
        ]

    if target == "Multicue":
        return [
            CurveSpec(
                "PiDiNet (MultiCue)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "pidinet_official_multicue",
                    target,
                ),
                "external",
                "released MultiCue checkpoint",
                "#E69F00",
                "--",
            ),
            CurveSpec(
                "CATS (BSDS)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "cats_official_bsds",
                    target,
                ),
                "external",
                "released BSDS checkpoint",
                "#C44E52",
                "--",
            ),
            CurveSpec(
                "DexiNed (BIPED)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "dexined_official_biped",
                    target,
                ),
                "external",
                "released BIPED checkpoint",
                "#00A6D6",
                "--",
            ),
            CurveSpec(
                "RCF",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "rcf_official_bsds_pascal",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#4D7C0F",
                "--",
            ),
            CurveSpec(
                "BDCN",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "bdcn_official_bsds",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#D4C200",
                "--",
            ),
            CurveSpec(
                "UAED",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "uaed_official_bsds",
                    target,
                ),
                "external",
                "released BSDS checkpoint",
                "#8C8C8C",
                "--",
            ),
        ]

    if target == "NYUDv2":
        return [
            CurveSpec(
                "PiDiNet (NYUDv2)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "pidinet_official_nyudv2",
                    target,
                ),
                "external",
                "released NYUDv2 RGB checkpoint",
                "#E69F00",
                "--",
            ),
            CurveSpec(
                "CATS (NYUDv2)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "cats_official_nyudv2",
                    target,
                ),
                "external",
                "released NYUDv2 RGB checkpoint",
                "#C44E52",
                "--",
            ),
            CurveSpec(
                "DexiNed (BIPED)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "dexined_official_biped",
                    target,
                ),
                "external",
                "released BIPED checkpoint",
                "#00A6D6",
                "--",
            ),
            CurveSpec(
                "RCF",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "rcf_official_bsds_pascal",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#4D7C0F",
                "--",
            ),
            CurveSpec(
                "BDCN",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "bdcn_official_bsds",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#D4C200",
                "--",
            ),
            CurveSpec(
                "UAED",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260720/"
                    "uaed_official_bsds",
                    target,
                ),
                "external",
                "released BSDS checkpoint",
                "#8C8C8C",
                "--",
            ),
        ]

    if target in {"BSDS500", "UDED"}:
        return [
            CurveSpec(
                "PiDiNet (BSDS)",
                external_path(
                    "edge_outputs/external/evaluations/pidinet_official_bsds",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#E69F00",
                "--",
            ),
            CurveSpec(
                "CATS (BSDS)",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "cats_official_bsds",
                    target,
                ),
                "external",
                "released BSDS checkpoint",
                "#C44E52",
                "--",
            ),
            CurveSpec(
                "DexiNed (BIPED)",
                external_path(
                    "edge_outputs/external/evaluations/dexined_official_biped",
                    target,
                ),
                "external",
                "released BIPED checkpoint",
                "#00A6D6",
                "--",
            ),
            CurveSpec(
                "TEED (BIPED)",
                external_path(
                    "edge_outputs/external/evaluations/teed_official_biped",
                    target,
                ),
                "external",
                "released BIPED checkpoint",
                "#7AA6A1",
                "--",
            ),
            CurveSpec(
                "RCF",
                external_path(
                    "edge_outputs/external/evaluations_fair_20260721/"
                    "rcf_official_bsds_pascal",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#4D7C0F",
                "--",
            ),
            CurveSpec(
                "BDCN",
                external_path(
                    "edge_outputs/external/evaluations/bdcn_official_bsds",
                    target,
                ),
                "external",
                "released BSDS+PASCAL checkpoint",
                "#D4C200",
                "--",
            ),
            CurveSpec(
                "UAED",
                external_path(
                    "edge_outputs/external/evaluations/uaed_official_bsds",
                    target,
                ),
                "external",
                "released BSDS checkpoint",
                "#8C8C8C",
                "--",
            ),
        ]

    raise KeyError(target)


def internal_specs(source: str, target: str) -> list[CurveSpec]:
    protocol = {
        "BIPED": "BIPED three-fixed-split protocol",
        "Multicue": "strict MultiCue 68/12/20 protocol",
        "NYUDv2": "strict NYUDv2 381/414/654 protocol",
    }[source]
    return [
        CurveSpec(
            MODE_LABELS[mode],
            internal_curve_path(source, target, mode),
            "internal",
            protocol,
            MODE_COLORS[mode],
            "-" if mode == "main_surround" else (0, (4, 2)),
            2.7 if mode == "main_surround" else 1.9,
            1.0,
            mode,
        )
        for mode in MODES
    ]


def panel_specs(source: str, target: str) -> list[CurveSpec]:
    specs = traditional_specs(target) + external_specs(target)
    specs += internal_specs(source, target)
    if len(specs) != 12:
        raise AssertionError(
            f"Expected 12 curves for {source}->{target}, got {len(specs)}"
        )
    return specs


def export(fig: plt.Figure, panel: Panel) -> None:
    panel_dir = OUT / f"{panel.source.lower()}_source"
    panel_dir.mkdir(parents=True, exist_ok=True)
    ALL_PNG_OUT.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / f"{panel.stem}.png"
    fig.savefig(panel_path, dpi=360)
    shutil.copy2(panel_path, ALL_PNG_OUT / panel_path.name)


def clean_obsolete_exports() -> None:
    """Keep the final PR directory PNG-only while retaining provenance data."""

    for suffix in ("*.pdf", "*.svg", "*.tif", "*.tiff"):
        for path in OUT.rglob(suffix):
            path.unlink()
    ALL_PNG_OUT.mkdir(parents=True, exist_ok=True)
    for path in ALL_PNG_OUT.glob("*.png"):
        path.unlink()


def draw_panel(
    panel: Panel,
    formal_scores: dict[tuple[str, str], float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.130, top=0.915)

    raw_rows: list[dict[str, object]] = []
    display_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for order, spec in enumerate(panel_specs(panel.source, panel.target)):
        raw = load_spec_curve(spec)
        curve = display_curve(raw)
        if (
            panel.source == panel.target
            and spec.canonical_mode is not None
            and (panel.target, spec.canonical_mode) in formal_scores
        ):
            display_f = formal_scores[(panel.target, spec.canonical_mode)]
            score_source = "formal same-domain ODS"
        else:
            display_f = float(raw["f1"].max())
            score_source = "maximum F1 on recorded threshold curve"

        label = f"[F={display_f:.3f}] {spec.label}"
        ax.plot(
            curve["recall"],
            curve["precision"],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            label=label,
            zorder=(
                5
                if spec.canonical_mode == "main_surround"
                else 4
                if spec.family == "internal"
                else 2
            ),
        )

        source_paths = spec.path if isinstance(spec.path, tuple) else (spec.path,)
        manifest_rows.append(
            {
                "panel_index": panel.index,
                "training_source": panel.source,
                "target_dataset": panel.target,
                "relation": "in_domain" if panel.source == panel.target else "cross_domain",
                "order": order,
                "model": spec.label,
                "family": spec.family,
                "source_protocol": spec.source_protocol,
                "display_f": display_f,
                "display_f_source": score_source,
                "curve_paths": ";".join(
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in source_paths
                ),
                "display_interpolation": (
                    "interpolated precision envelope + monotone Bernstein smoothing"
                ),
                "display_interpolation_changes_metrics": False,
            }
        )

        for row in raw.itertuples(index=False):
            raw_rows.append(
                {
                    "panel_index": panel.index,
                    "training_source": panel.source,
                    "target_dataset": panel.target,
                    "model": spec.label,
                    "family": spec.family,
                    "source_protocol": spec.source_protocol,
                    "display_f": display_f,
                    "threshold": float(row.threshold),
                    "precision": float(row.precision),
                    "recall": float(row.recall),
                    "f1": float(row.f1),
                }
            )
        for point_index, row in enumerate(curve.itertuples(index=False)):
            display_rows.append(
                {
                    "panel_index": panel.index,
                    "training_source": panel.source,
                    "target_dataset": panel.target,
                    "model": spec.label,
                    "family": spec.family,
                    "point_index": point_index,
                    "recall": float(row.recall),
                    "precision": float(row.precision),
                }
            )

    style_axis(ax, panel.title)
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.025, 0.025),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#555555",
        borderpad=0.55,
        labelspacing=0.22,
        handlelength=2.8,
        handletextpad=0.62,
    )
    legend.get_frame().set_linewidth(0.9)
    export(fig, panel)
    plt.close(fig)
    return raw_rows, display_rows, manifest_rows


def write_readmes() -> None:
    english = """# Source-target precision-recall matrix

This directory contains 15 independently exported PR panels: three H-RBCM
training sources (BIPED, strict MultiCue, and strict NYUDv2) evaluated on five
targets (BIPED, MultiCue, NYUDv2, BSDS500, and UDED). Each source subdirectory
contains the five target panels in PNG format. The `all_png/` directory also
contains a flat copy of all 15 panels for convenient browsing.

Each panel contains 12 curves: four internal H-RBCM modes and eight target-
specific references. BIPED, MultiCue, and NYUDv2 include Sobel and Canny;
BSDS500 and UDED include Canny plus seven released edge models. Every curve is
read from a recorded per-threshold CSV. No scalar score is used to reconstruct
a PR curve.

The visible line uses the same interpolated precision envelope and monotone
Bernstein display smoothing as the representative paper figures. This changes
only the rendered line. The measured threshold points and all quantitative
metrics are retained unchanged under `source_data/`.

The external models retain their released training sources and are broad
references under the same target-specific local evaluator. They are not
source-training-matched controls. The controlled mechanism comparison is among
Anchor, No surround, Conv control, and H-RBCM within each panel.
"""
    chinese = """# 训练源-目标集 Precision-Recall 曲线矩阵

本目录包含 15 张独立导出的 PR 图：三个 H-RBCM 训练源（BIPED、严格
MultiCue、严格 NYUDv2）分别在五个目标集（BIPED、MultiCue、NYUDv2、
BSDS500、UDED）上评估。每个训练源子目录均含五张 PNG 图；`all_png/`
另行平铺保存全部 15 张图，便于一次性浏览。

每张图包含 12 条曲线：4 个 H-RBCM 内部模式与 8 个按目标集选取的参照。
BIPED、MultiCue 和 NYUDv2 图均包含 Sobel 与 Canny；BSDS500 和 UDED 图
包含 Canny 与 7 个公开边缘检测模型。所有曲线均直接读取真实逐阈值 CSV，
没有用单个 ODS/OIS/AP 数值重建曲线。

图中可见线条采用与代表性论文图一致的插值精确率包络和单调 Bernstein
显示平滑。该步骤仅改善显示，不修改任何测量阈值点和定量指标；原始数据、
显示数据及每条曲线的来源清单均保存在 `source_data/`。

外部模型保留其公开权重原有的训练来源，只作为同一目标评估器下的广义参照，
不是训练源完全匹配的控制实验。严格的机制比较是每张图中的 Anchor、
No surround、Conv control 与 H-RBCM 四个内部模式。
"""
    (OUT / "README.md").write_text(english, encoding="utf-8")
    (OUT / "README.zh-CN.md").write_text(chinese, encoding="utf-8")


def validate_manifest(frame: pd.DataFrame) -> None:
    if len(frame) != 15 * 12:
        raise AssertionError(f"Expected 180 manifest rows, got {len(frame)}")
    panel_counts = frame.groupby("panel_index").size()
    if not (panel_counts == 12).all():
        raise AssertionError(f"Invalid panel counts: {panel_counts.to_dict()}")
    internal_counts = frame[frame["family"] == "internal"].groupby("panel_index").size()
    if not (internal_counts == 4).all():
        raise AssertionError(
            f"Invalid internal curve counts: {internal_counts.to_dict()}"
        )
    traditional_counts = (
        frame[frame["family"] == "traditional"].groupby("panel_index").size()
    )
    if not (traditional_counts >= 1).all():
        raise AssertionError("Every panel must include a non-trained operator")
    if not np.isfinite(frame["display_f"].to_numpy(float)).all():
        raise AssertionError("Non-finite display F value detected")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    clean_obsolete_exports()

    panels = [
        Panel(index=index, source=source, target=target)
        for index, (source, target) in enumerate(
            ((source, target) for source in SOURCES for target in TARGETS),
            start=1,
        )
    ]
    formal_scores = same_domain_scores()
    all_raw: list[dict[str, object]] = []
    all_display: list[dict[str, object]] = []
    all_manifest: list[dict[str, object]] = []

    for panel in panels:
        raw_rows, display_rows, manifest_rows = draw_panel(panel, formal_scores)
        all_raw.extend(raw_rows)
        all_display.extend(display_rows)
        all_manifest.extend(manifest_rows)

    raw_frame = pd.DataFrame(all_raw)
    display_frame = pd.DataFrame(all_display)
    manifest_frame = pd.DataFrame(all_manifest)
    validate_manifest(manifest_frame)

    if not raw_frame[["precision", "recall", "f1"]].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all():
        raise AssertionError("Raw curve values outside [0, 1]")
    if not display_frame[["precision", "recall"]].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all():
        raise AssertionError("Display curve values outside [0, 1]")

    raw_frame.to_csv(
        SOURCE_OUT / "pr_curve_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    display_frame.to_csv(
        SOURCE_OUT / "pr_curve_display_interpolation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (OUT / "curve_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_manifest[0]))
        writer.writeheader()
        writer.writerows(all_manifest)
    write_readmes()
    print(f"Wrote 15 PR panels and provenance data to {OUT}")


if __name__ == "__main__":
    main()
