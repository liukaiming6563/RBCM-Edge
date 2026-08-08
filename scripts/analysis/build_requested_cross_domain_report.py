"""Build the requested source-matched cross-domain comparison report.

This script performs no training and no inference. It reads stored evaluator
outputs, recomputes AP from precision-recall curves with the project metric,
and separates source-matched learned models from source-independent operators
and literature-only references.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_model.engine.metrics import average_precision_from_curve
from scripts.analysis import build_external_comparison_report as external_report


OUT_DIR = ROOT / "edge_outputs" / "rbcm" / "tables" / "requested_cross_domain_external"

REQUESTED_PATHS = (
    ("MultiCue", "BIPED"),
    ("MultiCue", "UDED"),
    ("NYUDv2", "BIPED"),
    ("NYUDv2", "UDED"),
    ("BIPED", "UDED"),
    ("BIPED", "MultiCue"),
)

H_RUNS = {
    ("MultiCue", "BIPED"): ROOT
    / "edge_outputs/rbcm/predictions/multicue_strict_seed4517_generalization5/official49/BIPED/main_surround",
    ("MultiCue", "UDED"): ROOT
    / "edge_outputs/rbcm/predictions/multicue_strict_seed4517_generalization5/official49/UDED/main_surround",
    ("NYUDv2", "BIPED"): ROOT
    / "edge_outputs/rbcm/predictions/nyudv2_strict_seed4517_20260725_generalization5/official49/BIPED/main_surround",
    ("NYUDv2", "UDED"): ROOT
    / "edge_outputs/rbcm/predictions/nyudv2_strict_seed4517_20260725_generalization5/official49/UDED/main_surround",
    ("BIPED", "UDED"): ROOT
    / "edge_outputs/rbcm/predictions/biped_multicue/generalization/official/biped_selected/UDED/main_surround",
    ("BIPED", "MultiCue"): ROOT
    / "edge_outputs/rbcm/predictions/biped_multicue/generalization/official/biped_selected/Multicue/main_surround",
}

SOURCE_MATCHED_MODELS = {
    "MultiCue": ("PiDiNet official Table-7 (MultiCue)",),
    "NYUDv2": (
        "PiDiNet official Table-6 (NYUDv2 RGB)",
        "CATS official (NYUDv2)",
    ),
    "BIPED": (
        "TEED official (BIPED)",
        "TEED local reproduction (BIPED)",
        "DexiNed official (BIPED)",
        "LDC official (BIPED)",
    ),
}

FIXED_OPERATORS = {
    "Canny": "canny_fixed_sweep",
    "Sobel": "sobel_fixed_gradient",
    "Prewitt": "prewitt_fixed_gradient",
    "Scharr": "scharr_fixed_gradient",
    "Roberts": "roberts_fixed_gradient",
    "Laplacian": "laplacian_fixed_gradient",
}

TARGET_ONLY_ROSTER = (
    "PiDiNet official Table-5 (BSDS+PASCAL)",
    "RCF official (BSDS+PASCAL)",
    "BDCN official (BSDS+PASCAL)",
    "UAED official (BSDS)",
    "CATS official (BSDS)",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def canonical_ap(curve_path: Path) -> float:
    curve = [
        {"recall": float(row["recall"]), "precision": float(row["precision"])}
        for row in read_csv(curve_path)
    ]
    return float(average_precision_from_curve(curve))


def load_summary(run_dir: Path) -> tuple[dict[str, Any], Path]:
    for filename in ("summary.csv", "summary.json"):
        path = run_dir / filename
        if path.exists():
            return external_report.load_summary(path), path
    raise FileNotFoundError(f"Missing summary in {run_dir}")


def h_row(source: str, target: str) -> dict[str, Any]:
    run_dir = H_RUNS[(source, target)]
    summary, summary_path = load_summary(run_dir)
    curve_path = run_dir / "pr_curve_as_is.csv"
    if not curve_path.exists():
        raise FileNotFoundError(curve_path)
    return {
        "method": "H-RBCM",
        "source_dataset": source,
        "target_dataset": target,
        "ODS": float(summary["ODS"]),
        "OIS": float(summary["OIS"]),
        "AP": canonical_ap(curve_path),
        "n_images": int(float(summary["n_images"])),
        "metric_backend": summary.get("metric_backend", "dilation"),
        "evidence_type": "internal source-matched unified evaluation",
        "summary_file": rel(summary_path),
        "curve_file": rel(curve_path),
        "official_source": "internal formal checkpoint",
    }


def metric_differences(main: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    deltas = {metric: 100.0 * (float(main[metric]) - float(reference[metric])) for metric in ("ODS", "OIS", "AP")}
    wins = {metric: deltas[metric] > 0 for metric in deltas}
    return {
        "H_minus_ODS_pp": deltas["ODS"],
        "H_minus_OIS_pp": deltas["OIS"],
        "H_minus_AP_pp": deltas["AP"],
        "H_wins_ODS": wins["ODS"],
        "H_wins_OIS": wins["OIS"],
        "H_wins_AP": wins["AP"],
        "H_win_count": sum(wins.values()),
    }


def source_matched_rows(external: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in REQUESTED_PATHS:
        main = h_row(source, target)
        for method in SOURCE_MATCHED_MODELS[source]:
            key = (method, "Multicue" if target == "MultiCue" else target)
            if key not in external:
                continue
            reference = external[key]
            row = {
                "experiment": f"{source}->{target}",
                "target_dataset": target,
                "H_method": "H-RBCM",
                "H_ODS": main["ODS"],
                "H_OIS": main["OIS"],
                "H_AP": main["AP"],
                "external_method": method,
                "external_source_dataset": source,
                "external_ODS": reference["ODS"],
                "external_OIS": reference["OIS"],
                "external_AP": reference["AP"],
                "same_source_dataset": True,
                "same_target_dataset": True,
                "same_local_evaluator": True,
                "external_evidence_type": "released checkpoint/prediction; unified local evaluation",
                "H_curve_file": main["curve_file"],
                "external_curve_file": reference["curve_file"],
                "external_official_source": reference["official_source"],
            }
            row.update(metric_differences(main, reference))
            rows.append(row)
    return rows


def fixed_operator_rows() -> list[dict[str, Any]]:
    applicable = defaultdict(list)
    for source, target in REQUESTED_PATHS:
        applicable[target].append(f"{source}->{target}")

    rows: list[dict[str, Any]] = []
    for target in ("BIPED", "UDED", "MultiCue"):
        disk_target = "Multicue" if target == "MultiCue" else target
        for method, run in FIXED_OPERATORS.items():
            run_dir = ROOT / "edge_outputs/external/evaluations" / run / disk_target
            curve_path = run_dir / "pr_curve_as_is.csv"
            if not curve_path.exists():
                continue
            summary, summary_path = load_summary(run_dir)
            rows.append(
                {
                    "target_dataset": target,
                    "applicable_requested_paths": ";".join(applicable[target]),
                    "method": method,
                    "source_dataset": "none (fixed operator)",
                    "ODS": float(summary["ODS"]),
                    "OIS": float(summary["OIS"]),
                    "AP": canonical_ap(curve_path),
                    "n_images": int(float(summary["n_images"])),
                    "metric_backend": summary.get("metric_backend", "dilation"),
                    "evidence_type": "source-independent unified local evaluation",
                    "summary_file": rel(summary_path),
                    "curve_file": rel(curve_path),
                    "eligible_as_source_matched_learned_model": False,
                }
            )
    return rows


def literature_rows() -> list[dict[str, Any]]:
    values = (
        ("Canny", 0.742, 0.743),
        ("DexiNed", 0.815, 0.826),
        ("PiDiNet", 0.812, 0.824),
        ("LDC", 0.817, 0.838),
        ("BDCN-B2", 0.821, 0.839),
        ("TIN", 0.803, 0.827),
        ("PiDiNet-small", 0.821, 0.834),
        ("PiDiNet-tiny-L", 0.821, 0.834),
        ("TEED", 0.828, 0.842),
        ("TEEDup", 0.834, 0.847),
    )
    source = (
        "https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/"
        "Soria_Tiny_and_Efficient_Model_for_the_Edge_Detection_Generalization_ICCVW_2023_paper.pdf"
    )
    return [
        {
            "experiment": "BIPED->UDED",
            "method": method,
            "reported_ODS": ods,
            "reported_OIS": ois,
            "reported_AP": "",
            "evidence_type": "paper-reported Table 2; separate evaluator",
            "directly_comparable_to_local_unified_rows": False,
            "source": source,
        }
        for method, ods, ois in values
    ]


def target_only_rows(external: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in ("BIPED", "UDED", "MultiCue"):
        disk_target = "Multicue" if target == "MultiCue" else target
        for method in TARGET_ONLY_ROSTER:
            reference = external.get((method, disk_target))
            if reference is None:
                continue
            rows.append(
                {
                    "target_dataset": target,
                    "method": method,
                    "source_dataset": reference["source_dataset"],
                    "ODS": reference["ODS"],
                    "OIS": reference["OIS"],
                    "AP": reference["AP"],
                    "evidence_type": "released checkpoint/prediction; unified target evaluation",
                    "eligible_for_requested_source_matched_claim": False,
                    "reason": "training source differs from the requested H-RBCM source",
                    "curve_file": reference["curve_file"],
                    "official_source": reference["official_source"],
                }
            )
    return rows


def availability_rows() -> list[dict[str, Any]]:
    return [
        {
            "model": "PiDiNet",
            "status": "included",
            "usable_sources": "MultiCue;NYUDv2",
            "reason": "source-specific released checkpoints/predictions already evaluated locally",
            "complexity": "none",
            "official_source": "https://github.com/hellozhuo/pidinet",
        },
        {
            "model": "CATS",
            "status": "included",
            "usable_sources": "NYUDv2",
            "reason": "NYUDv2 checkpoint/predictions already evaluated locally",
            "complexity": "none",
            "official_source": "https://github.com/WHUHLX/CATS",
        },
        {
            "model": "TEED",
            "status": "included",
            "usable_sources": "BIPED",
            "reason": "official BIPED predictions and a separately labelled local reproduction are available",
            "complexity": "none",
            "official_source": "https://github.com/xavysp/TEED",
        },
        {
            "model": "DexiNed",
            "status": "included",
            "usable_sources": "BIPED",
            "reason": "official BIPED predictions are available",
            "complexity": "none",
            "official_source": "https://github.com/xavysp/DexiNed",
        },
        {
            "model": "LDC",
            "status": "included",
            "usable_sources": "BIPED",
            "reason": "official BIPED checkpoint/predictions are available and already evaluated",
            "complexity": "none",
            "official_source": "https://github.com/xavysp/LDC",
        },
        {
            "model": "DiffusionEdge",
            "status": "skipped",
            "usable_sources": "potentially BIPED;NYUDv2",
            "reason": "large checkpoints plus a separate diffusion/Swin environment; isolated smoke test exposed Torch/torchvision conflicts",
            "complexity": "high; environment change and long inference",
            "official_source": "https://github.com/GuHuangAI/DiffusionEdge",
        },
        {
            "model": "BDCN",
            "status": "skipped",
            "usable_sources": "potentially NYUDv2",
            "reason": "legacy source-specific assets require old CUDA/Caffe-style adapters and were not immediately executable in the formal environment",
            "complexity": "high; legacy environment",
            "official_source": "https://github.com/pkuCactus/BDCN",
        },
        {
            "model": "RCF",
            "status": "skipped",
            "usable_sources": "potentially NYUDv2",
            "reason": "legacy source-specific assets require old framework and dataset adapters not present in the formal environment",
            "complexity": "high; legacy environment",
            "official_source": "https://github.com/yun-liu/RCF-PyTorch",
        },
        {
            "model": "EDTER",
            "status": "skipped",
            "usable_sources": "not verified for requested sources",
            "reason": "old mmseg stack and no immediately usable source-specific artifact for the requested paths",
            "complexity": "high; separate mmseg environment",
            "official_source": "https://github.com/mengyangpu/edter",
        },
        {
            "model": "BLEDNet",
            "status": "not eligible for numeric cross-domain table",
            "usable_sources": "none verified for requested paths",
            "reason": "the accessible archive contains retinal-vessel outputs rather than complete source-specific edge predictions for the requested paths",
            "complexity": "would require retraining or unreleased artifacts",
            "official_source": "https://doi.org/10.1016/j.engappai.2023.106530",
        },
        {
            "model": "BLCDNet",
            "status": "not eligible for numeric cross-domain table",
            "usable_sources": "none verified for requested paths",
            "reason": "code and same-domain BSDS/NYUD result folders exist, but no verified BIPED, MultiCue, or NYUDv2 source checkpoint with complete requested-target predictions",
            "complexity": "would require retraining or unreleased artifacts",
            "official_source": "https://github.com/zwuser1227/BLCDNet",
        },
        {
            "model": "XYW-Net",
            "status": "not eligible for numeric cross-domain table",
            "usable_sources": "none verified for requested paths",
            "reason": "the paper covers relevant datasets, but the public repository does not expose a verified source-specific checkpoint or complete predictions for the requested paths",
            "complexity": "would require retraining or unreleased artifacts",
            "official_source": "https://github.com/PXinTao/XYW-Net",
        },
        {
            "model": "MEDNet",
            "status": "not eligible for numeric cross-domain table",
            "usable_sources": "none verified for requested paths",
            "reason": "public result folders are same-domain/fixed predictions and do not provide the source-matched checkpoints needed for these cross-domain paths",
            "complexity": "would require retraining or unreleased artifacts",
            "official_source": "https://github.com/zwuser1227/MEDNet-Edge-detection",
        },
        {
            "model": "DPED",
            "status": "related-work only",
            "usable_sources": "none verified for requested paths",
            "reason": "bio-inspired edge detector relevant to the literature narrative, but no verified source-matched artifact was found for the requested paths",
            "complexity": "numeric inclusion would require retraining or unavailable predictions",
            "official_source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9606659/",
        },
        {
            "model": "FFED",
            "status": "related-work only",
            "usable_sources": "none verified for requested paths",
            "reason": "flow-field-guided edge detector is relevant background, but no immediately usable source-specific artifact was verified for the requested paths",
            "complexity": "numeric inclusion would require additional reproduction work",
            "official_source": "https://github.com/hanyuchen2022/Flow-field-guided-edge-detection-FFED-",
        },
        {
            "model": "BPVENet",
            "status": "not eligible",
            "usable_sources": "none",
            "reason": "adverse-condition object-detection enhancement model, not an edge detector evaluated on the requested edge datasets",
            "complexity": "task mismatch",
            "official_source": "https://doi.org/10.1007/s11760-025-03919-w",
        },
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def source_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                row["experiment"],
                row["external_method"],
                f'{row["H_ODS"]:.4f}/{row["H_OIS"]:.4f}/{row["H_AP"]:.4f}',
                f'{row["external_ODS"]:.4f}/{row["external_OIS"]:.4f}/{row["external_AP"]:.4f}',
                f'{row["H_minus_ODS_pp"]:+.2f}/{row["H_minus_OIS_pp"]:+.2f}/{row["H_minus_AP_pp"]:+.2f}',
                str(row["H_win_count"]),
            ]
        )
    return markdown_table(
        ["Path", "External model", "H-RBCM ODS/OIS/AP", "External ODS/OIS/AP", "H delta (pp)", "Wins"],
        body,
    )


def source_table_zh(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                row["experiment"],
                row["external_method"],
                f'{row["H_ODS"]:.4f}/{row["H_OIS"]:.4f}/{row["H_AP"]:.4f}',
                f'{row["external_ODS"]:.4f}/{row["external_OIS"]:.4f}/{row["external_AP"]:.4f}',
                f'{row["H_minus_ODS_pp"]:+.2f}/{row["H_minus_OIS_pp"]:+.2f}/{row["H_minus_AP_pp"]:+.2f}',
                str(row["H_win_count"]),
            ]
        )
    return markdown_table(
        ["路径", "外部模型", "H-RBCM ODS/OIS/AP", "外部模型 ODS/OIS/AP", "H-RBCM 差值（百分点）", "胜出指标数"],
        body,
    )


def build_readmes_v2(source_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]]) -> None:
    source_md = source_table(source_rows)
    source_md_zh = source_table_zh(source_rows)
    fixed_counts = defaultdict(int)
    for row in fixed_rows:
        fixed_counts[row["target_dataset"]] += 1
    fixed_summary_zh = "，".join(f"{name}：{count}" for name, count in fixed_counts.items())

    en = f"""# Requested cross-domain external comparisons

This directory is generated by `scripts/analysis/build_requested_cross_domain_report.py`.
The generator performs no training or inference and does not synthesize scores.
Local ODS/OIS values are read from preserved evaluations; AP is recomputed from
each saved `pr_curve_as_is.csv` using the project's endpoint-complete precision
envelope and trapezoidal integration.

## Requested paths

- MultiCue -> BIPED
- MultiCue -> UDED
- NYUDv2 -> BIPED
- NYUDv2 -> UDED
- BIPED -> UDED
- BIPED -> MultiCue

The original request repeated NYUDv2 -> UDED; it is represented once here.

## Source-matched learned models

The external model must use the same source dataset and target dataset as H-RBCM.
Training details may differ and must not be described as identical conditions.
Official and local-reproduction artifacts are kept as separate rows.

{source_md}

## Paper-facing interpretation

- MultiCue source: H-RBCM exceeds PiDiNet on BIPED and UDED for ODS, OIS, and AP.
- NYUDv2 source: H-RBCM exceeds PiDiNet on BIPED for all metrics, exceeds CATS
  on BIPED and UDED for all metrics, and exceeds PiDiNet on UDED for ODS/OIS
  while AP is 0.41 percentage points lower.
- BIPED source: official TEED, DexiNed, and LDC artifacts are stronger than
  H-RBCM on UDED and MultiCue. H-RBCM exceeds the separately labelled local TEED
  reproduction on all UDED metrics and on MultiCue ODS/OIS, but not AP.
- The evidence supports selected cross-domain advantages, not universal superiority.

## Source-independent operators

`source_independent_operators.csv` contains {sum(fixed_counts.values())} target
evaluations: {dict(fixed_counts)}. These methods have no training source and are
target-side classical references, not source-matched learned models.

## Separate-protocol literature context

`biped_to_uded_literature.csv` transcribes Table 2 of the TEED paper. It uses the
paper's evaluator and must not be merged into a local unified-evaluator ranking.

## Availability and exclusions

`external_availability_audit.csv` records inclusion status, official source, and
reproduction complexity. `target_only_references.csv` contains locally evaluated
references whose training source differs from the requested source.

BLEDNet, BLCDNet, XYW-Net, MEDNet, DPED, and FFED remain related work because no
verified source-specific artifact covers these exact paths. BPVENet is an
object-detection enhancement model and is ineligible. DiffusionEdge, BDCN, RCF,
and EDTER were stopped because they require large or legacy isolated environments,
additional backbone assets, or substantial adapters.

## Files

- `source_matched_learned.csv`: main source/target-matched learned evidence.
- `source_independent_operators.csv`: fixed target-side operators.
- `biped_to_uded_literature.csv`: TEED-paper values under its evaluator.
- `target_only_references.csv`: source-mismatched supplementary references.
- `external_availability_audit.csv`: provenance and availability audit.
"""

    zh = f"""# 指定跨域路径的外部模型比较

本目录由 `scripts/analysis/build_requested_cross_domain_report.py` 生成。脚本不训练、
不推理，也不合成分数。本地 ODS/OIS 来自已保存的评估结果；AP 从每个
`pr_curve_as_is.csv` 按工程统一的端点补齐、precision envelope 和梯形积分重新计算。

## 指定路径

- MultiCue -> BIPED
- MultiCue -> UDED
- NYUDv2 -> BIPED
- NYUDv2 -> UDED
- BIPED -> UDED
- BIPED -> MultiCue

原始任务中 NYUDv2 -> UDED 重复出现，本报告只保留一次。

## 训练源与目标集匹配的学习模型

主表要求外部模型与 H-RBCM 使用相同的训练来源数据集和相同的跨域目标集。
训练超参数、增强和具体划分可能不同，因此论文中应写“训练源与目标集匹配”，
不能写成“训练条件完全相同”。官方权重与本地复现权重分别列行。

{source_md_zh}

## 可用于论文的结论边界

- MultiCue 来源：H-RBCM 在 BIPED 和 UDED 的 ODS、OIS、AP 均高于 PiDiNet。
- NYUDv2 来源：H-RBCM 在 BIPED 三项均高于 PiDiNet；在 BIPED 和 UDED 三项均
  高于 CATS；在 UDED 的 ODS/OIS 高于 PiDiNet，但 AP 低 0.41 个百分点。
- BIPED 来源：TEED、DexiNed、LDC 的官方资产在 UDED 和 MultiCue 上整体更强。
  H-RBCM 在 UDED 三项均高于单独标注的 TEED 本地复现，在 MultiCue 的
  ODS/OIS 高于该本地复现，但 AP 较低。
- 这些证据支持“在若干跨域条件下有优势”，不支持“普遍领先”。

## 无训练源的传统算子

`source_independent_operators.csv` 共包含 {sum(fixed_counts.values())} 个目标集评估
（{fixed_summary_zh}）。这些方法没有训练源，只能作为目标侧传统方法参照。

## 不同评估协议的文献补充

`biped_to_uded_literature.csv` 转录 TEED 论文 Table 2。该表使用论文自身评估器，
不能与本地统一评估结果混成同一数值排名。

## 可用性与未纳入项目

`external_availability_audit.csv` 逐项记录纳入状态、官方来源和复杂度。
`target_only_references.csv` 收录本地已复评、但训练源不匹配的补充参照。

BLEDNet、BLCDNet、XYW-Net、MEDNet、DPED 和 FFED 可用于相关工作讨论，但没有
找到覆盖指定路径的可验证源域权重或完整预测。BPVENet 是目标检测增强模型，
不属于本表任务。DiffusionEdge、BDCN、RCF 和 EDTER 因需要大型或旧版独立环境、
额外骨干资产或复杂适配，按用户要求停止复现。

## 文件说明

- `source_matched_learned.csv`：训练源与目标集匹配的学习模型主证据。
- `source_independent_operators.csv`：Canny 等目标侧固定算子。
- `biped_to_uded_literature.csv`：TEED 论文自身协议下的数值。
- `target_only_references.csv`：训练源不匹配的补充参照。
- `external_availability_audit.csv`：来源与可用性审计。
"""
    (OUT_DIR / "README.md").write_text(en, encoding="utf-8")
    (OUT_DIR / "README.zh-CN.md").write_text(zh, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    external = external_report.build_external_rows()
    source_rows = source_matched_rows(external)
    fixed_rows = fixed_operator_rows()
    literature = literature_rows()
    target_only = target_only_rows(external)
    availability = availability_rows()

    write_csv(OUT_DIR / "source_matched_learned.csv", source_rows)
    write_csv(OUT_DIR / "source_independent_operators.csv", fixed_rows)
    write_csv(OUT_DIR / "biped_to_uded_literature.csv", literature)
    write_csv(OUT_DIR / "target_only_references.csv", target_only)
    write_csv(OUT_DIR / "external_availability_audit.csv", availability)
    build_readmes_v2(source_rows, fixed_rows)

    print(
        f"Wrote {len(source_rows)} source-matched, {len(fixed_rows)} fixed-operator, "
        f"{len(literature)} literature, and {len(target_only)} target-only rows to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
