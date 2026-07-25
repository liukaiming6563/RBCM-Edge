"""Build the canonical machine-readable H-RBCM result index.

The index keeps protocol status and evidence provenance beside every score so
that paper tables cannot silently mix strict, descriptive, and transfer
results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    ROOT / "paper_assets" / "rbcm" / "tables" / "formal_index"
)
BIPED_PATH = (
    ROOT / "paper_assets" / "rbcm" / "tables" / "two_dataset_core"
    / "01_biped_stability.csv"
)
MULTICUE_PATH = (
    ROOT / "paper_assets" / "rbcm" / "tables" / "two_dataset_core"
    / "02_internal_ablation_all_targets.csv"
)
NYUD_SUMMARY_PATH = ROOT / "weights" / "rbcm" / "nyudv2" / "main" / "formal_summary.json"
NYUD_TRANSFER_PATH = (
    ROOT / "weights" / "rbcm" / "nyudv2" / "main" / "generalization_summary.csv"
)

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
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def empty_row() -> dict[str, Any]:
    return {field: "" for field in FIELDS}


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for source in read_csv(BIPED_PATH):
        row = empty_row()
        row.update(
            scope="same_domain_stability",
            training_source="BIPED",
            target_dataset="BIPED",
            mode=source["mode"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            ODS_std=source["ODS_std"],
            OIS_std=source["OIS_std"],
            AP_std=source["AP_std"],
            n_images="50",
            n_runs=source["n_runs"],
            thresholds="49",
            independent_test="true",
            paper_role="primary",
            protocol_note="Three fixed 170/30/50 splits; mean and sample SD",
            source_file=relative(BIPED_PATH),
        )
        rows.append(row)

    for source in read_csv(MULTICUE_PATH):
        if source["training_source"] != "Multicue" or source["target_dataset"] != "Multicue":
            continue
        row = empty_row()
        row.update(
            scope="same_domain_descriptive",
            training_source="Multicue",
            target_dataset="Multicue",
            mode=source["mode"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images"],
            n_runs="1",
            thresholds=source["thresholds"],
            independent_test="false",
            paper_role="supplementary",
            protocol_note=(
                "Source-only frozen posthoc; validation and reporting use the "
                "same 20 images"
            ),
            source_file=relative(MULTICUE_PATH),
        )
        rows.append(row)

    strict = json.loads(NYUD_SUMMARY_PATH.read_text(encoding="utf-8"))
    nyud_modes = {
        "plain_identity": strict["baseline"]["test"],
        **{
            mode: payload["test"]
            for mode, payload in strict["selected"].items()
        },
    }
    for mode, metrics in nyud_modes.items():
        row = empty_row()
        row.update(
            scope="same_domain_strict",
            training_source="NYUDv2",
            target_dataset="NYUDv2",
            mode=mode,
            ODS=metrics["ODS"],
            OIS=metrics["OIS"],
            AP=metrics["AP"],
            n_images=strict["test_count"],
            n_runs="1",
            thresholds="99",
            independent_test="true",
            paper_role="primary",
            protocol_note=(
                "Strict 381-source train / 414 validation / 654 test; "
                "checkpoint and candidates selected on validation only"
            ),
            source_file=relative(NYUD_SUMMARY_PATH),
        )
        rows.append(row)

    for source in read_csv(NYUD_TRANSFER_PATH):
        row = empty_row()
        row.update(
            scope="frozen_transfer",
            training_source="NYUDv2",
            target_dataset=source["dataset"],
            mode=source["mode"],
            ODS=source["ODS"],
            OIS=source["OIS"],
            AP=source["AP"],
            n_images=source["n_images"],
            n_runs="1",
            thresholds=source["thresholds"],
            independent_test="true",
            paper_role="secondary",
            protocol_note=(
                "Strict NYUDv2 epoch-9 checkpoint and validation-selected "
                "candidate frozen across all targets"
            ),
            source_file=relative(NYUD_TRANSFER_PATH),
        )
        rows.append(row)

    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    primary = [
        row for row in rows
        if row["scope"] in {"same_domain_stability", "same_domain_strict"}
    ]
    lines = [
        "# Canonical H-RBCM result index",
        "",
        "This file is generated by `scripts/analysis/build_formal_result_index.py`.",
        "BIPED and strict NYUDv2 are primary independent-test evidence. MultiCue",
        "is supplementary because its validation and reporting lists overlap.",
        "",
        "| Scope | Source | Target | Mode | ODS | OIS | AP | Role |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in primary:
        lines.append(
            f"| {row['scope']} | {row['training_source']} | "
            f"{row['target_dataset']} | {row['mode']} | "
            f"{float(row['ODS']):.5f} | {float(row['OIS']):.5f} | "
            f"{float(row['AP']):.5f} | {row['paper_role']} |"
        )
    lines.extend(
        [
            "",
            "The complete CSV/JSON also includes the limitation-labeled MultiCue",
            "record and the frozen strict-NYUDv2 five-target matrix.",
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
                "schema_version": 1,
                "canonical_primary_protocols": ["BIPED", "NYUDv2_strict"],
                "supplementary_protocols": ["Multicue_descriptive"],
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(rows, OUT_ROOT / "FORMAL_RESULT_INDEX.md")
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
