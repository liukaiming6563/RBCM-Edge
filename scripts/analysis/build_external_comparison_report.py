"""Build reproducible H-RBCM versus external-model comparison tables.

The script does not run inference or training. It reads saved evaluator
summaries and precision-recall curves, recomputes endpoint-complete AP with the
project metric implementation, and records the provenance of every number.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_model.engine.metrics import average_precision_from_curve


OUT_DIR = ROOT / "edge_outputs" / "rbcm" / "tables" / "external_comparison"

CANONICAL_EXTERNAL = (
    ROOT / "edge_outputs" / "rbcm" / "tables" / "fair_generalization" / "external_all_models.csv"
)
EXPANDED_EXTERNAL = (
    ROOT / "edge_outputs" / "rbcm" / "tables" / "external_additions" / "external_models_expanded.csv"
)
STRICT_SAME_DOMAIN = (
    ROOT / "edge_outputs" / "rbcm" / "tables" / "strict_protocols" / "same_domain_strict.csv"
)


MODEL_URLS = {
    "TEED official (BIPED)": "https://github.com/xavysp/TEED",
    "TEED local reproduction (BIPED)": "https://github.com/xavysp/TEED",
    "DexiNed official (BIPED)": "https://github.com/xavysp/DexiNed",
    "LDC official (BIPED)": "https://github.com/xavysp/LDC",
    "PiDiNet official Table-5 (BSDS+PASCAL)": "https://github.com/hellozhuo/pidinet",
    "PiDiNet official Table-6 (NYUDv2 RGB)": "https://github.com/hellozhuo/pidinet",
    "PiDiNet official Table-7 (MultiCue)": "https://github.com/hellozhuo/pidinet",
    "PiDiNet local reproduction (BSDS)": "https://github.com/hellozhuo/pidinet",
    "RCF official (BSDS+PASCAL)": "https://github.com/yun-liu/RCF-PyTorch",
    "BDCN official (BSDS+PASCAL)": "https://github.com/pkuCactus/BDCN",
    "UAED official (BSDS)": "https://github.com/ZhouCX117/UAED",
    "CATS official (BSDS)": "https://github.com/WHUHLX/CATS",
    "CATS official (NYUDv2)": "https://github.com/WHUHLX/CATS",
    "MEDNet official predictions (BIPED)": "https://github.com/zwuser1227/MEDNet-Edge-detection",
}


SAME_DOMAIN_ROSTERS = {
    "BIPED": [
        "TEED official (BIPED)",
        "DexiNed official (BIPED)",
        "LDC official (BIPED)",
        "MEDNet official predictions (BIPED)",
        "PiDiNet official Table-5 (BSDS+PASCAL)",
        "RCF official (BSDS+PASCAL)",
        "BDCN official (BSDS+PASCAL)",
        "UAED official (BSDS)",
        "Sobel fixed gradient",
        "Canny fixed sweep",
    ],
    "Multicue": [
        "PiDiNet official Table-7 (MultiCue)",
        "PiDiNet official Table-5 (BSDS+PASCAL)",
        "BDCN official (BSDS+PASCAL)",
        "UAED official (BSDS)",
        "RCF official (BSDS+PASCAL)",
        "CATS official (BSDS)",
        "DexiNed official (BIPED)",
        "LDC official (BIPED)",
        "Sobel fixed gradient",
        "Canny fixed sweep",
    ],
    "NYUDv2": [
        "PiDiNet official Table-6 (NYUDv2 RGB)",
        "CATS official (NYUDv2)",
        "PiDiNet official Table-5 (BSDS+PASCAL)",
        "BDCN official (BSDS+PASCAL)",
        "RCF official (BSDS+PASCAL)",
        "UAED official (BSDS)",
        "TEED official (BIPED)",
        "DexiNed official (BIPED)",
        "Sobel fixed gradient",
        "Canny fixed sweep",
    ],
}


CROSS_TARGET_ROSTERS = {
    "BIPED": [
        "TEED official (BIPED)",
        "DexiNed official (BIPED)",
        "LDC official (BIPED)",
        "MEDNet official predictions (BIPED)",
        "PiDiNet official Table-5 (BSDS+PASCAL)",
        "PiDiNet official Table-6 (NYUDv2 RGB)",
        "PiDiNet official Table-7 (MultiCue)",
        "RCF official (BSDS+PASCAL)",
        "BDCN official (BSDS+PASCAL)",
        "UAED official (BSDS)",
        "Sobel fixed gradient",
        "Canny fixed sweep",
    ],
    "UDED": [
        "TEED official (BIPED)",
        "DexiNed official (BIPED)",
        "LDC official (BIPED)",
        "PiDiNet official Table-5 (BSDS+PASCAL)",
        "PiDiNet official Table-6 (NYUDv2 RGB)",
        "PiDiNet official Table-7 (MultiCue)",
        "RCF official (BSDS+PASCAL)",
        "BDCN official (BSDS+PASCAL)",
        "UAED official (BSDS)",
        "CATS official (BSDS)",
        "CATS official (NYUDv2)",
        "Canny fixed sweep",
    ],
}


STRICT_MULTICUE_GENERALIZATION = (
    ROOT / "results" / "rbcm" / "predictions" / "multicue_strict_seed4517_generalization5" / "official49"
)
STRICT_NYUD_GENERALIZATION = (
    ROOT / "results" / "rbcm" / "predictions" / "nyudv2_strict_seed4517_20260725_generalization5" / "official49"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_ap(curve_path: Path) -> float:
    rows = []
    for row in read_csv(curve_path):
        rows.append({"recall": float(row["recall"]), "precision": float(row["precision"])})
    return average_precision_from_curve(rows)


def load_summary(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("selected", payload)
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty summary: {path}")
    return rows[0]


def external_curve_index() -> dict[tuple[str, str], Path]:
    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in (ROOT / "results" / "external").rglob("pr_curve_as_is.csv"):
        if len(path.parents) < 2:
            continue
        run = path.parent.parent.name
        target = path.parent.name
        candidates[(run, target)].append(path)

    def priority(path: Path) -> tuple[int, float]:
        text = str(path).lower()
        score = 3 if "evaluations_fair_20260721" in text else 2 if "evaluations_fair_20260720" in text else 1
        return score, path.stat().st_mtime

    return {key: max(paths, key=priority) for key, paths in candidates.items()}


def normalize_source(source: str) -> str:
    if source.lower() in {"multicue", "multi-cue"}:
        return "Multicue"
    return source


def source_relation(source: str, target: str, h_source: str | None = None) -> str:
    source = normalize_source(source)
    if source.lower() == "none":
        return "non_trained"
    if h_source is not None and source == normalize_source(h_source):
        return "source_dataset_matched"
    if source == target:
        return "target_dataset_matched"
    return "cross_source_dataset"


def build_external_rows() -> dict[tuple[str, str], dict[str, Any]]:
    curves = external_curve_index()
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for row in read_csv(CANONICAL_EXTERNAL):
        merged[(row["model"], row["target_dataset"])] = dict(row)

    additions = ("LDC official", "RCF official", "CATS official", "MEDNet official")
    for row in read_csv(EXPANDED_EXTERNAL):
        if row["model"].startswith(additions):
            merged[(row["model"], row["target_dataset"])] = dict(row)

    for target in ("BIPED", "Multicue", "NYUDv2"):
        run = "sobel_fixed_gradient"
        summary_path = ROOT / "results" / "external" / "evaluations" / run / target / "summary.json"
        curve_path = ROOT / "results" / "external" / "evaluations" / run / target / "pr_curve_as_is.csv"
        summary = load_summary(summary_path)
        merged[("Sobel fixed gradient", target)] = {
            "model": "Sobel fixed gradient",
            "source_dataset": "none",
            "training_scope": "fixed non-learned gradient operator",
            "target_dataset": target,
            "ODS": summary["ODS"],
            "OIS": summary["OIS"],
            "AP": summary["AP"],
            "n_images": summary["n_images"],
            "match_tolerance": "",
            "gt_threshold": "",
            "thresholds": len(read_csv(curve_path)),
            "NMS": True,
            "metric_backend": "dilation",
            "orientation": "as_is",
            "prediction_run": run,
        }

    clean: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in merged.items():
        run = row["prediction_run"]
        curve_path = curves.get((run, row["target_dataset"]))
        if curve_path is None and run == "sobel_fixed_gradient":
            curve_path = (
                ROOT
                / "results"
                / "external"
                / "evaluations"
                / run
                / row["target_dataset"]
                / "pr_curve_as_is.csv"
            )
        if curve_path is None or not curve_path.exists():
            raise FileNotFoundError(f"Missing PR curve for {key}: run={run}")
        clean[key] = {
            "method": row["model"],
            "source_dataset": normalize_source(row["source_dataset"]),
            "training_scope": row["training_scope"],
            "target_dataset": row["target_dataset"],
            "ODS": float(row["ODS"]),
            "OIS": float(row["OIS"]),
            "AP": endpoint_ap(curve_path),
            "n_images": int(float(row["n_images"])),
            "thresholds": len(read_csv(curve_path)),
            "NMS": str(row["NMS"]).lower() == "true",
            "metric_backend": row["metric_backend"],
            "curve_file": curve_path.relative_to(ROOT).as_posix(),
            "official_source": MODEL_URLS.get(row["model"], "local fixed operator implementation"),
        }
    return clean


def win_fields(main: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    wins = {metric: float(main[metric]) > float(reference[metric]) for metric in ("ODS", "OIS", "AP")}
    return {
        "H_minus_ODS_pp": 100.0 * (float(main["ODS"]) - float(reference["ODS"])),
        "H_minus_OIS_pp": 100.0 * (float(main["OIS"]) - float(reference["OIS"])),
        "H_minus_AP_pp": 100.0 * (float(main["AP"]) - float(reference["AP"])),
        "H_wins_ODS": wins["ODS"],
        "H_wins_OIS": wins["OIS"],
        "H_wins_AP": wins["AP"],
        "H_win_count": sum(wins.values()),
    }


def h_same_domain_rows() -> dict[str, dict[str, Any]]:
    selected = {}
    for row in read_csv(STRICT_SAME_DOMAIN):
        if row["mode"] != "main_surround":
            continue
        selected[row["dataset"]] = {
            "method": "H-RBCM",
            "source_dataset": row["dataset"],
            "training_scope": row["protocol"],
            "target_dataset": row["dataset"],
            "ODS": float(row["ODS"]),
            "OIS": float(row["OIS"]),
            "AP": float(row["AP"]),
            "n_images": int(row["n_images"]),
            "thresholds": int(row["thresholds"]),
            "NMS": True,
            "metric_backend": "dilation",
            "curve_file": row["source_file"],
            "official_source": "internal formal protocol",
        }
    return selected


def h_cross_row(source: str, target: str) -> dict[str, Any]:
    base = STRICT_MULTICUE_GENERALIZATION if source == "Multicue" else STRICT_NYUD_GENERALIZATION
    summary_path = base / target / "main_surround" / "summary.csv"
    summary = load_summary(summary_path)
    curve_path = summary_path.with_name("pr_curve_as_is.csv")
    return {
        "method": "H-RBCM",
        "source_dataset": source,
        "training_scope": f"strict {source} source protocol; frozen calibration",
        "target_dataset": target,
        "ODS": float(summary["ODS"]),
        "OIS": float(summary["OIS"]),
        "AP": endpoint_ap(curve_path),
        "n_images": int(float(summary["n_images"])),
        "thresholds": len(read_csv(curve_path)),
        "NMS": True,
        "metric_backend": summary["metric_backend"],
        "curve_file": curve_path.relative_to(ROOT).as_posix(),
        "official_source": "internal strict generalization evaluation",
    }


def serialize_row(row: dict[str, Any], relation: str, experiment: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "target_dataset": row["target_dataset"],
        "method": row["method"],
        "source_dataset": row["source_dataset"],
        "source_relation": relation,
        "ODS": row["ODS"],
        "OIS": row["OIS"],
        "AP": row["AP"],
        "n_images": row["n_images"],
        "thresholds": row["thresholds"],
        "NMS": row["NMS"],
        "metric_backend": row["metric_backend"],
        "training_scope": row["training_scope"],
        "curve_file": row["curve_file"],
        "official_source": row["official_source"],
        "H_minus_ODS_pp": "",
        "H_minus_OIS_pp": "",
        "H_minus_AP_pp": "",
        "H_wins_ODS": "",
        "H_wins_OIS": "",
        "H_wins_AP": "",
        "H_win_count": "",
    }


def build_same_domain(external: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target, roster in SAME_DOMAIN_ROSTERS.items():
        main = h_same_domain_rows()[target]
        main_row = serialize_row(main, "internal_main", f"same_domain_{target}")
        output.append(main_row)
        for method in roster:
            reference = external[(method, target)]
            row = serialize_row(
                reference,
                source_relation(reference["source_dataset"], target),
                f"same_domain_{target}",
            )
            row.update(win_fields(main, reference))
            output.append(row)
    return output


def build_cross_domain(external: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in ("Multicue", "NYUDv2"):
        for target in ("BIPED", "UDED"):
            main = h_cross_row(source, target)
            experiment = f"{source}_to_{target}"
            output.append(serialize_row(main, "internal_main", experiment))
            for method in CROSS_TARGET_ROSTERS[target]:
                reference = external[(method, target)]
                row = serialize_row(
                    reference,
                    source_relation(reference["source_dataset"], target, h_source=source),
                    experiment,
                )
                row.update(win_fields(main, reference))
                output.append(row)
    return output


def markdown_table(rows: list[dict[str, Any]], include_relation: bool = True) -> str:
    headers = ["Method", "Source"]
    if include_relation:
        headers.append("Relation")
    headers += ["ODS", "OIS", "AP", "H wins"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        method = row["method"]
        metrics = [f"{float(row[key]):.4f}" for key in ("ODS", "OIS", "AP")]
        if method == "H-RBCM":
            metrics = [f"**{value}**" for value in metrics]
            won = "main"
        else:
            won_metrics = [key for key in ("ODS", "OIS", "AP") if row[f"H_wins_{key}"] is True]
            won = ", ".join(won_metrics) if won_metrics else "none"
        cells = [method, row["source_dataset"]]
        if include_relation:
            cells.append(row["source_relation"])
        cells += metrics + [won]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_readmes(same_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]]) -> None:
    by_same: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cross: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in same_rows:
        by_same[row["target_dataset"]].append(row)
    for row in cross_rows:
        by_cross[row["experiment"]].append(row)

    common_en = """# External comparison audit

All ODS/OIS values are read from saved evaluator summaries. AP is recomputed
from the saved `pr_curve_as_is.csv` files using endpoint sentinels, duplicate
recall consolidation, a precision envelope, and trapezoidal integration.
Every row uses the same target-side NMS, GT handling, tolerance, and dilation
matcher. Training sources are intentionally explicit: `source_dataset_matched`
is the cleanest available checkpoint comparison; `target_dataset_matched` and
`cross_source_dataset` rows are released-checkpoint context, not
identical-training claims; `non_trained`
denotes a fixed classical operator.

The formal H-RBCM same-domain rows use the registered paper protocols (99
thresholds). Released external checkpoints use 49 thresholds. Cross-domain
tables use 49 thresholds for both H-RBCM and external references. Threshold
density is recorded in the CSV and must be disclosed when quoting the broad
same-domain context.
"""
    common_zh = """# 外部模型统一复评审计

所有 ODS/OIS 均直接读取保存的评估摘要；AP 则从真实的
`pr_curve_as_is.csv` 重新计算，统一采用端点补齐、重复 recall 取最大
precision、precision envelope 和梯形积分。每个目标数据集上的 NMS、GT
处理、匹配容差和 dilation matcher 均保持一致。

`source_dataset_matched` 表示与 H-RBCM 使用相同训练数据集来源，但不代表
训练配方完全一致；`target_dataset_matched` 和 `cross_source_dataset` 只表示
统一目标评估器下的公开权重参照，
不能写成完全同训练条件；`non_trained` 为固定传统算子。

H-RBCM 同域正式结果采用论文登记的 99 阈值协议，公开外部权重采用 49
阈值；跨域表中 H-RBCM 与外部模型均为 49 阈值。CSV 已逐行记录阈值数，
引用同域广义外部参照时应披露这一差异。
"""

    en = [common_en, "\n## Same-domain target evaluations\n"]
    zh = [common_zh, "\n## 同域目标评估\n"]
    for target in ("BIPED", "Multicue", "NYUDv2"):
        en += [f"\n### {target}\n", markdown_table(by_same[target])]
        zh += [f"\n### {target}\n", markdown_table(by_same[target])]
    en.append("\n## Selected cross-domain evaluations\n")
    zh.append("\n## 选定跨域评估\n")
    for experiment in ("Multicue_to_BIPED", "Multicue_to_UDED", "NYUDv2_to_BIPED", "NYUDv2_to_UDED"):
        en += [f"\n### {experiment}\n", markdown_table(by_cross[experiment])]
        zh += [f"\n### {experiment}\n", markdown_table(by_cross[experiment])]

    (OUT_DIR / "README.md").write_text("\n".join(en) + "\n", encoding="utf-8")
    (OUT_DIR / "README.zh-CN.md").write_text("\n".join(zh) + "\n", encoding="utf-8")


def build_provenance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (row["method"], row["source_dataset"], row["target_dataset"], row["curve_file"])
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "method": row["method"],
                "source_dataset": row["source_dataset"],
                "target_dataset": row["target_dataset"],
                "training_scope": row["training_scope"],
                "curve_file": row["curve_file"],
                "official_source": row["official_source"],
                "n_images": row["n_images"],
                "thresholds": row["thresholds"],
                "NMS": row["NMS"],
                "metric_backend": row["metric_backend"],
                "AP_definition": "endpoint-complete precision-envelope trapezoid",
            }
        )
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    external = build_external_rows()
    same_rows = build_same_domain(external)
    cross_rows = build_cross_domain(external)
    write_csv(OUT_DIR / "same_domain_external.csv", same_rows)
    write_csv(OUT_DIR / "cross_domain_external.csv", cross_rows)
    write_csv(OUT_DIR / "provenance.csv", build_provenance(same_rows + cross_rows))
    build_readmes(same_rows, cross_rows)
    print(f"Wrote {len(same_rows)} same-domain rows and {len(cross_rows)} cross-domain rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
