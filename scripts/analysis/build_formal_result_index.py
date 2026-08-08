"""Build the canonical machine-readable H-RBCM result index.

The index separates the paper's original, structure-appropriate
validation-selection strategy from the equal-budget sensitivity analysis.
Every paper-facing candidate is selected on validation data and frozen before
the corresponding test or transfer evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RBCM_OUTPUT_ROOT = ROOT / "edge_outputs" / "rbcm"
OUT_ROOT = RBCM_OUTPUT_ROOT / "tables" / "formal_index"
RESULT_OUT_ROOT = RBCM_OUTPUT_ROOT / "scores"

PRIMARY_SAME_PATH = RBCM_OUTPUT_ROOT / "scores" / "self_test.csv"
BIPED_STABILITY_PATH = RBCM_OUTPUT_ROOT / "scores" / "biped_stability.csv"
FAIR_PRIMARY_PATH = (
    RBCM_OUTPUT_ROOT
    / "reanalysis"
    / "fair_eval_20260729"
    / "summary"
    / "fair_primary_results.csv"
)
MULTICUE_TRANSFER_PATH = (
    RBCM_OUTPUT_ROOT
    / "predictions"
    / "multicue_strict_seed4517_generalization5"
    / "generalization_summary_official49.csv"
)
NYUD_TRANSFER_PATH = (
    RBCM_OUTPUT_ROOT
    / "predictions"
    / "nyudv2_strict_seed4517_20260725_generalization5"
    / "generalization_summary_official49_49.csv"
)
EXTERNAL_PATH = (
    ROOT / "edge_outputs" / "rbcm" / "tables"
    / "fair_generalization"
    / "external_all_models.csv"
)

PRIMARY_DATASETS = ("BIPED", "Multicue", "NYUDv2")
MODES = ("plain_identity", "main_surround", "no_surround", "conv_control")
MULTICUE_TARGETS = ("BIPED", "Multicue", "NYUDv2", "BSDS500", "UDED")

FIELDS = [
    "scope",
    "training_source",
    "target_dataset",
    "mode",
    "ODS",
    "OIS",
    "AP",
    "ODS_std",
    "OIS_std",
    "AP_std",
    "n_images",
    "n_runs",
    "thresholds",
    "independent_test",
    "paper_role",
    "protocol_note",
    "source_file",
    "metric_backend",
    "evaluation_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def empty_row() -> dict[str, Any]:
    return {field: "" for field in FIELDS}


def primary_same_domain_rows() -> list[dict[str, Any]]:
    if not PRIMARY_SAME_PATH.is_file():
        raise FileNotFoundError(PRIMARY_SAME_PATH)

    stability = {}
    if BIPED_STABILITY_PATH.is_file():
        stability = {
            row["mode"]: row for row in read_csv(BIPED_STABILITY_PATH)
        }

    rows: list[dict[str, Any]] = []
    for source in read_csv(PRIMARY_SAME_PATH):
        dataset = source["dataset"]
        if dataset not in PRIMARY_DATASETS or source["mode"] not in MODES:
            continue
        mode = source["mode"]
        row = empty_row()
        row.update(
            scope=(
                "same_domain_stability"
                if dataset == "BIPED"
                else "same_domain_strict"
            ),
            training_source=dataset,
            target_dataset=dataset,
            mode=mode,
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images"],
            n_runs=source["n_runs"],
            thresholds="99",
            independent_test="true",
            paper_role="primary",
            protocol_note=source["protocol"],
            source_file=relative(PRIMARY_SAME_PATH),
            metric_backend="dilation",
            evaluation_status="original_validation_frozen_strategy",
        )
        if dataset == "BIPED" and mode in stability:
            row["ODS_std"] = stability[mode]["ODS_std"]
            row["OIS_std"] = stability[mode]["OIS_std"]
            row["AP_std"] = stability[mode]["AP_std"]
        rows.append(row)
    return rows


def equal_budget_sensitivity_rows() -> list[dict[str, Any]]:
    if not FAIR_PRIMARY_PATH.is_file():
        return []

    rows: list[dict[str, Any]] = []
    for source in read_csv(FAIR_PRIMARY_PATH):
        dataset = source["dataset"]
        if dataset not in PRIMARY_DATASETS:
            continue
        row = empty_row()
        row.update(
            scope="same_domain_equal_budget_sensitivity",
            training_source=dataset,
            target_dataset=dataset,
            mode=source["mode"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images_per_split"],
            n_runs="3" if dataset == "BIPED" else "1",
            thresholds=source["thresholds"],
            independent_test="true",
            paper_role="supplementary_sensitivity",
            protocol_note=(
                "Equal candidate-count sensitivity analysis; validation-only "
                "selection and frozen test evaluation"
            ),
            source_file=relative(FAIR_PRIMARY_PATH),
            metric_backend=source["metric_backend"],
            evaluation_status="supplementary_equal_budget_reanalysis_20260729",
        )
        rows.append(row)
    return rows


def multicue_transfer_rows() -> list[dict[str, Any]]:
    if not MULTICUE_TRANSFER_PATH.is_file():
        return []

    rows: list[dict[str, Any]] = []
    for source in read_csv(MULTICUE_TRANSFER_PATH):
        if source["mode"] not in MODES:
            continue
        target = source["target_dataset"]
        if target not in {"BIPED", "NYUDv2", "BSDS500", "UDED"}:
            continue
        row = empty_row()
        row.update(
            scope="cross_domain_internal",
            training_source="Multicue",
            target_dataset=target,
            mode=source["mode"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images"],
            n_runs="1",
            thresholds=source["thresholds"],
            independent_test="true",
            paper_role=(
                "primary_generalization"
                if target in {"BIPED", "NYUDv2", "UDED"}
                else "supplementary_tradeoff"
            ),
            protocol_note=(
                "Strict MultiCue 68/12/20 source split; mode-specific "
                "candidates selected on the 12-source validation set and "
                "frozen unchanged for target inference"
            ),
            source_file=relative(MULTICUE_TRANSFER_PATH),
            metric_backend="dilation",
            evaluation_status="manuscript_v5_frozen_strategy",
        )
        rows.append(row)
    return rows


def nyud_transfer_rows() -> list[dict[str, Any]]:
    if not NYUD_TRANSFER_PATH.is_file():
        return []

    rows: list[dict[str, Any]] = []
    for source in read_csv(NYUD_TRANSFER_PATH):
        if source["mode"] not in MODES:
            continue
        target = source["dataset"]
        row = empty_row()
        row.update(
            scope=(
                "same_domain_common_evaluator"
                if target == "NYUDv2"
                else "cross_domain_internal"
            ),
            training_source="NYUDv2",
            target_dataset=target,
            mode=source["mode"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images"],
            n_runs="1",
            thresholds=source["thresholds"],
            independent_test="true",
            paper_role=(
                "primary_generalization"
                if target in {"NYUDv2", "BSDS500", "UDED"}
                else "supplementary"
            ),
            protocol_note=(
                "Strict NYUDv2 381/414/654 split; mode-specific candidates "
                "selected on the 414-image validation set and frozen unchanged "
                "for all targets"
            ),
            source_file=relative(NYUD_TRANSFER_PATH),
            metric_backend=source["metric_backend"],
            evaluation_status="original_validation_frozen_strategy",
        )
        rows.append(row)
    return rows


def external_rows() -> list[dict[str, Any]]:
    if not EXTERNAL_PATH.is_file():
        return []

    rows: list[dict[str, Any]] = []
    for source in read_csv(EXTERNAL_PATH):
        row = empty_row()
        row.update(
            scope="external_released_checkpoint_common_evaluator",
            training_source=source["source_dataset"],
            target_dataset=source["target_dataset"],
            mode=source["model"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images"],
            n_runs="1",
            thresholds=source["thresholds"],
            independent_test="true",
            paper_role="external_context",
            protocol_note=(
                f"{source['training_scope']}; released prediction/checkpoint "
                "reference evaluated on the shared target GT, NMS, threshold "
                "sweep, tolerance, and matcher"
            ),
            source_file=relative(EXTERNAL_PATH),
            metric_backend=source["metric_backend"],
            evaluation_status="common_evaluator_reference",
        )
        rows.append(row)
    return rows


def build_rows() -> list[dict[str, Any]]:
    return [
        *primary_same_domain_rows(),
        *multicue_transfer_rows(),
        *nyud_transfer_rows(),
        *external_rows(),
        *equal_budget_sensitivity_rows(),
    ]


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    primary = [
        row
        for row in rows
        if row["paper_role"] == "primary"
        and row["mode"] in MODES
    ]
    transfer = [
        row
        for row in rows
        if row["paper_role"] == "primary_generalization"
        and row["training_source"] == "Multicue"
    ]
    lines = [
        "# Canonical H-RBCM result index",
        "",
        "This file is generated by `scripts/analysis/build_formal_result_index.py`.",
        "The primary analysis uses structure-appropriate, mode-specific candidate",
        "sets selected only on validation data and frozen before test access.",
        "The equal-budget analysis is retained as a supplementary sensitivity",
        "analysis and does not overwrite the predeclared primary strategy.",
        "",
        "All rows use the project's shared near-official Python dilation matcher.",
        "The square annulus is the intended L-infinity implementation.",
        "",
        "## Primary same-domain evidence",
        "",
        "| Source/target | Mode | ODS | OIS | AP |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['training_source']} | {row['mode']} | "
            f"{float(row['ODS']):.5f} | {float(row['OIS']):.5f} | "
            f"{float(row['AP']):.5f} |"
        )
    lines.extend(
        [
            "",
            "## Strict MultiCue-source transfer evidence",
            "",
            "| Target | Mode | ODS | OIS | AP |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in transfer:
        lines.append(
            f"| {row['target_dataset']} | {row['mode']} | "
            f"{float(row['ODS']):.5f} | {float(row['OIS']):.5f} | "
            f"{float(row['AP']):.5f} |"
        )
    lines.extend(
        [
            "",
            "The CSV/JSON files also retain the complete external released-checkpoint",
            "matrix and equal-budget sensitivity rows. External rows share the",
            "target evaluator but may have different published training recipes;",
            "they are therefore checkpoint references, not identical-training runs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_ROOT / "formal_result_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_ROOT / "formal_result_index.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "evidence_freeze": "manuscript_v5_result_sources_20260808",
                "primary_candidate_policy": (
                    "Mode-specific validation selection; candidates frozen "
                    "before test or transfer evaluation"
                ),
                "sensitivity_candidate_policy": (
                    "Equal candidate counts retained as supplementary analysis"
                ),
                "annulus_geometry": "square_l_infinity",
                "metric_backend": "near_official_python_dilation",
                "canonical_primary_protocols": [
                    "BIPED_three_split",
                    "Multicue_strict_68_12_20",
                    "NYUDv2_strict_381_414_654",
                ],
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(rows, OUT_ROOT / "FORMAL_RESULT_INDEX.md")

    RESULT_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for source in OUT_ROOT.iterdir():
        if source.is_file():
            (RESULT_OUT_ROOT / source.name).write_bytes(source.read_bytes())
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
