"""Audit the fixed source-disjoint MultiCue confirmation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "edge_data" / "official_rbcm" / "Multicue"
DEFAULT_CONFIG = PROJECT_ROOT / "edge_model" / "configs" / "rbcm" / "multicue_strict.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "rbcm" / "audits" / "multicue_strict_20260726"
STEM_PATTERN = re.compile(r"^images_s[^_]+_r[^_]+_(?:flip_1_)?(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    return [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(stem: str) -> str:
    match = STEM_PATTERN.match(stem)
    return match.group(1) if match else stem


def resolve_split(config: dict, split: str) -> Path:
    value = config["dataset"]["split_files"][split]
    path = Path(value)
    if path.is_absolute():
        return path
    dataset_relative = DATASET_ROOT / path
    return dataset_relative if dataset_relative.exists() else PROJECT_ROOT / path


def check(name: str, condition: bool, detail: object) -> dict[str, object]:
    return {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config.get("dataset", {})
    train_cfg = config.get("train", {})
    gt = dataset.get("gt", {})

    paths = {split: resolve_split(config, split) for split in ("train", "val", "test")}
    rows = {split: read_split(path) for split, path in paths.items()}
    sources = {split: {source_id(stem) for stem in stems} for split, stems in rows.items()}
    manifest_path = paths["train"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checks: list[dict[str, object]] = []
    checks.append(check("row counts", [len(rows[key]) for key in ("train", "val", "test")] == [6528, 12, 20], {key: len(value) for key, value in rows.items()}))
    checks.append(check("source counts", [len(sources[key]) for key in ("train", "val", "test")] == [68, 12, 20], {key: len(value) for key, value in sources.items()}))
    checks.append(check("unique rows", all(len(value) == len(set(value)) for value in rows.values()), {key: len(value) - len(set(value)) for key, value in rows.items()}))
    overlap = {
        "train_val": sorted(sources["train"] & sources["val"]),
        "train_test": sorted(sources["train"] & sources["test"]),
        "val_test": sorted(sources["val"] & sources["test"]),
    }
    checks.append(check("zero source overlap", not any(overlap.values()), overlap))
    descendants = {source: 0 for source in sources["train"]}
    for stem in rows["train"]:
        descendants[source_id(stem)] += 1
    checks.append(check("complete augmentation groups", set(descendants.values()) == {96}, sorted(set(descendants.values()))))
    checks.append(check("validation originals only", all(stem == f"images_s1.0_r0.0_{source_id(stem)}" for stem in rows["val"]), rows["val"]))
    checks.append(check("fixed test preserved", bool(manifest.get("test_is_original_report_split")), manifest.get("test_is_original_report_split")))
    checks.append(check("split hashes match manifest", all(sha256(paths[key]) == manifest["split_hashes"][key] for key in paths), {key: sha256(path) for key, path in paths.items()}))

    missing: list[str] = []
    for split, stems in rows.items():
        variants = ["image", "edge", "gt/soft_vote"]
        if split == "train":
            variants.append("gt/multicue_ignore_weight")
        for stem in stems:
            for variant in variants:
                path = DATASET_ROOT / variant / f"{stem}.png"
                if not path.exists():
                    missing.append(str(path))
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
    checks.append(check("required assets exist", not missing, missing))

    checks.append(check("validation used during training", dataset.get("eval_split") == "val" and dataset.get("val_split") == "val", {"val_split": dataset.get("val_split"), "eval_split": dataset.get("eval_split")}))
    checks.append(check("explicit strict split mapping", set(dataset.get("split_files", {})) >= {"train", "val", "test"}, dataset.get("split_files")))
    checks.append(check("validation ODS checkpoint selection", str(train_cfg.get("checkpoint_score", {}).get("mode", "")).lower() == "ods", train_cfg.get("checkpoint_score")))
    checks.append(check("no test-driven strict replacement", train_cfg.get("strict_checkpoint_selection", {}).get("enabled") is False, train_cfg.get("strict_checkpoint_selection")))
    checks.append(check("MultiCue GT protocol", gt.get("train_variant") == "edge" and gt.get("train_mode") == "binary" and gt.get("eval_variant") == "soft_vote" and gt.get("eval_mode") == "soft", gt))
    checks.append(check("fixed seed and full epochs", int(config.get("seed", -1)) == 4517 and int(train_cfg.get("epochs", -1)) == 14, {"seed": config.get("seed"), "epochs": train_cfg.get("epochs")}))
    checks.append(check("no sample caps", all(dataset.get(key) is None for key in ("max_train_samples", "max_eval_samples", "max_val_samples")), {key: dataset.get(key) for key in ("max_train_samples", "max_eval_samples", "max_val_samples")}))
    checks.append(check("resumable training", bool(train_cfg.get("resumable_mid_epoch")) and int(train_cfg.get("checkpoint_interval_steps", 0)) > 0, {"resumable_mid_epoch": train_cfg.get("resumable_mid_epoch"), "checkpoint_interval_steps": train_cfg.get("checkpoint_interval_steps")}))

    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {
        "status": status,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# MultiCue Strict Protocol Audit", "", f"Overall status: **{status}**", ""]
    lines.extend(f"- [{item['status']}] {item['name']}: `{item['detail']}`" for item in checks)
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
