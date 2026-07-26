"""Build paper-facing tables from the strict H-RBCM evidence sources.

This script intentionally keeps protocol provenance beside every score.  It
does not read the archived MultiCue validation/report-overlap results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER_OUT = ROOT / "paper_assets" / "rbcm" / "tables" / "strict_protocols"
RESULT_OUT = ROOT / "results" / "rbcm" / "scores" / "strict_protocols"

FORMAL_INDEX = (
    ROOT
    / "paper_assets"
    / "rbcm"
    / "tables"
    / "formal_index"
    / "formal_result_index.csv"
)
NYUD_EXTERNAL = (
    ROOT
    / "paper_assets"
    / "rbcm"
    / "tables"
    / "fair_generalization"
    / "source_matched_comparisons.csv"
)

MODES = (
    "plain_identity",
    "main_surround",
    "no_surround",
    "conv_control",
)
METRICS = ("ODS", "OIS", "AP")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strongest_control(
    rows: list[dict[str, str]], dataset: str, metric: str
) -> tuple[str, float]:
    controls = [
        row
        for row in rows
        if row["training_source"] == dataset
        and row["target_dataset"] == dataset
        and row["mode"] != "main_surround"
        and row["scope"] in {"same_domain_stability", "same_domain_strict"}
    ]
    if len(controls) != 3:
        raise RuntimeError(
            f"Expected three matched controls for {dataset}, found {len(controls)}"
        )
    winner = max(controls, key=lambda row: float(row[metric]))
    return winner["mode"], float(winner[metric])


def build_same_domain(
    formal: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_domain = [
        row
        for row in formal
        if row["scope"] in {"same_domain_stability", "same_domain_strict"}
        and row["mode"] in MODES
    ]

    compact: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for dataset in ("BIPED", "Multicue", "NYUDv2"):
        dataset_rows = [
            row
            for row in same_domain
            if row["training_source"] == dataset
            and row["target_dataset"] == dataset
        ]
        if len(dataset_rows) != 4:
            raise RuntimeError(
                f"Expected four formal same-domain rows for {dataset}, "
                f"found {len(dataset_rows)}"
            )
        for row in dataset_rows:
            compact.append(
                {
                    "dataset": dataset,
                    "mode": row["mode"],
                    "ODS": row["ODS"],
                    "OIS": row["OIS"],
                    "AP": row["AP"],
                    "n_images": row["n_images"],
                    "n_runs": row["n_runs"],
                    "thresholds": row["thresholds"],
                    "protocol": row["protocol_note"],
                    "source_file": row["source_file"],
                }
            )

        main = next(row for row in dataset_rows if row["mode"] == "main_surround")
        for metric in METRICS:
            control_mode, control_value = strongest_control(
                formal, dataset, metric
            )
            main_value = float(main[metric])
            deltas.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "main": f"{main_value:.12f}",
                    "strongest_control": f"{control_value:.12f}",
                    "control_mode": control_mode,
                    "delta": f"{main_value - control_value:.12f}",
                    "main_wins": str(main_value > control_value),
                }
            )
    return compact, deltas


def build_nyud_generalization(
    formal: list[dict[str, str]]
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in formal
        if row["scope"] == "frozen_transfer"
        and row["training_source"] == "NYUDv2"
        and row["mode"] in MODES
    ]
    if len(rows) != 20:
        raise RuntimeError(
            f"Expected 20 strict-NYUDv2 five-target rows, found {len(rows)}"
        )
    return [
        {
            "training_source": row["training_source"],
            "target_dataset": row["target_dataset"],
            "mode": row["mode"],
            "ODS": row["ODS"],
            "OIS": row["OIS"],
            "AP": row["AP"],
            "n_images": row["n_images"],
            "thresholds": row["thresholds"],
            "protocol": row["protocol_note"],
            "source_file": row["source_file"],
        }
        for row in rows
    ]


def build_nyud_external() -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(NYUD_EXTERNAL)
        if row["training_source"] == "NYUDv2"
        and row["external_model"] == "PiDiNet official Table-6 (NYUDv2 RGB)"
    ]
    if len(rows) != 5:
        raise RuntimeError(
            f"Expected five NYUDv2/PiDiNet comparison rows, found {len(rows)}"
        )
    return rows


def fmt(value: str | float) -> str:
    return f"{float(value):.5f}"


def write_readmes(
    same: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    nyud_external: list[dict[str, Any]],
) -> None:
    delta_lookup = {
        (row["dataset"], row["metric"]): row for row in deltas
    }
    nyud_same = next(
        row for row in nyud_external if row["target_dataset"] == "NYUDv2"
    )

    zh = f"""# H-RBCM 严格协议结果

本目录由 `scripts/analysis/build_strict_protocol_tables.py` 生成，只汇总可用于正式论文的独立协议。旧 MultiCue 验证/报告集合重合结果不参与这些表格。

## 主要同域结论

- BIPED：三套固定 170/30/50 划分。H-RBCM 相对逐指标最强控制的变化为 ODS {100 * float(delta_lookup[("BIPED", "ODS")]["delta"]):+.2f}、OIS {100 * float(delta_lookup[("BIPED", "OIS")]["delta"]):+.2f}、AP {100 * float(delta_lookup[("BIPED", "AP")]["delta"]):+.2f} 个百分点。
- MultiCue：严格 68 个训练源、12 个验证源、20 个一次性独立测试源。H-RBCM 的 ODS/OIS 分别提高 {100 * float(delta_lookup[("Multicue", "ODS")]["delta"]):+.2f}/{100 * float(delta_lookup[("Multicue", "OIS")]["delta"]):+.2f} 个百分点，AP 变化 {100 * float(delta_lookup[("Multicue", "AP")]["delta"]):+.2f} 个百分点。该结果支持 F-score 改善，但必须同时报告 AP 权衡。
- NYUDv2：严格 381/414/654 零重叠划分。H-RBCM 的 ODS/OIS/AP 分别提高 {100 * float(delta_lookup[("NYUDv2", "ODS")]["delta"]):+.2f}/{100 * float(delta_lookup[("NYUDv2", "OIS")]["delta"]):+.2f}/{100 * float(delta_lookup[("NYUDv2", "AP")]["delta"]):+.2f} 个百分点。

## 严格 NYUDv2 外部比较

在共享目标 evaluator 下，NYUDv2 同域 H-RBCM 为 {fmt(nyud_same["H_ODS"])}/{fmt(nyud_same["H_OIS"])}/{fmt(nyud_same["H_AP"])}，PiDiNet NYUDv2 RGB released checkpoint 为 {fmt(nyud_same["Ext_ODS"])}/{fmt(nyud_same["Ext_OIS"])}/{fmt(nyud_same["Ext_AP"])}。H-RBCM 三项分别变化 {100 * float(nyud_same["Delta_ODS"]):+.2f}/{100 * float(nyud_same["Delta_OIS"]):+.2f}/{100 * float(nyud_same["Delta_AP"]):+.2f} 个百分点。

跨域外部比较并非全面领先：BIPED 三项领先；UDED 的 ODS/OIS 领先而 AP 小幅落后；MultiCue 和 BSDS500 落后。因此论文只能声称严格 NYUDv2 同域优势和部分域迁移优势，不能声称普遍优于 PiDiNet。

## 文件

- `same_domain_strict.csv`：三个主要独立协议的四模式同域结果。
- `main_vs_strongest_control.csv`：主模型相对逐指标最强匹配控制的变化。
- `nyudv2_five_target_ablation.csv`：严格 NYUDv2 checkpoint 与候选冻结后的五目标四模式矩阵。
- `nyudv2_vs_pidinet.csv`：NYUDv2 训练来源匹配、目标 evaluator 统一的外部比较。
"""

    en = f"""# Strict-protocol H-RBCM results

This directory is generated by `scripts/analysis/build_strict_protocol_tables.py` and contains only independent paper-facing protocols. The archived MultiCue validation/report-overlap result is excluded.

## Primary same-domain findings

- BIPED: three fixed 170/30/50 splits. Relative to the strongest matched control per metric, H-RBCM changes ODS/OIS/AP by {100 * float(delta_lookup[("BIPED", "ODS")]["delta"]):+.2f}/{100 * float(delta_lookup[("BIPED", "OIS")]["delta"]):+.2f}/{100 * float(delta_lookup[("BIPED", "AP")]["delta"]):+.2f} percentage points.
- MultiCue: strict 68-source training, 12-source validation, and 20-source one-time held-out testing. H-RBCM improves ODS/OIS by {100 * float(delta_lookup[("Multicue", "ODS")]["delta"]):+.2f}/{100 * float(delta_lookup[("Multicue", "OIS")]["delta"]):+.2f} points and changes AP by {100 * float(delta_lookup[("Multicue", "AP")]["delta"]):+.2f} points. This supports an F-score benefit with an explicitly reported AP trade-off.
- NYUDv2: strict zero-overlap 381/414/654 split. H-RBCM improves ODS/OIS/AP by {100 * float(delta_lookup[("NYUDv2", "ODS")]["delta"]):+.2f}/{100 * float(delta_lookup[("NYUDv2", "OIS")]["delta"]):+.2f}/{100 * float(delta_lookup[("NYUDv2", "AP")]["delta"]):+.2f} points.

## Strict NYUDv2 external comparison

Under the shared target evaluator, same-domain NYUDv2 H-RBCM scores {fmt(nyud_same["H_ODS"])}/{fmt(nyud_same["H_OIS"])}/{fmt(nyud_same["H_AP"])} versus {fmt(nyud_same["Ext_ODS"])}/{fmt(nyud_same["Ext_OIS"])}/{fmt(nyud_same["Ext_AP"])} for the released PiDiNet NYUDv2 RGB checkpoint. The changes are {100 * float(nyud_same["Delta_ODS"]):+.2f}/{100 * float(nyud_same["Delta_OIS"]):+.2f}/{100 * float(nyud_same["Delta_AP"]):+.2f} percentage points.

Cross-domain external performance is mixed: all three metrics improve on BIPED; ODS/OIS improve on UDED with a small AP decrease; MultiCue and BSDS500 are lower. The defensible claim is strict same-domain NYUDv2 superiority and selected transfer advantages, not universal superiority over PiDiNet.

## Files

- `same_domain_strict.csv`: four-mode same-domain results for the three primary independent protocols.
- `main_vs_strongest_control.csv`: main-model change relative to the strongest matched control for each metric.
- `nyudv2_five_target_ablation.csv`: frozen strict-NYUDv2 checkpoint/candidate matrix across five targets.
- `nyudv2_vs_pidinet.csv`: source-matched NYUDv2 comparison under the shared target evaluator.
"""

    (PAPER_OUT / "README.md").write_text(en, encoding="utf-8")
    (PAPER_OUT / "README.zh-CN.md").write_text(zh, encoding="utf-8")


def main() -> None:
    formal = read_csv(FORMAL_INDEX)
    same, deltas = build_same_domain(formal)
    nyud_generalization = build_nyud_generalization(formal)
    nyud_external = build_nyud_external()

    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    write_csv(PAPER_OUT / "same_domain_strict.csv", same)
    write_csv(PAPER_OUT / "main_vs_strongest_control.csv", deltas)
    write_csv(
        PAPER_OUT / "nyudv2_five_target_ablation.csv",
        nyud_generalization,
    )
    write_csv(PAPER_OUT / "nyudv2_vs_pidinet.csv", nyud_external)
    write_readmes(same, deltas, nyud_external)

    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    for source in PAPER_OUT.iterdir():
        if source.is_file():
            (RESULT_OUT / source.name).write_bytes(source.read_bytes())
    print(f"Wrote strict-protocol tables to {PAPER_OUT}")
    print(f"Mirrored strict-protocol tables to {RESULT_OUT}")


if __name__ == "__main__":
    main()
