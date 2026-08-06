"""Audit the paper-facing NYUDv2 381/414/654 training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edge_model.data.build import make_dataset  # noqa: E402


ID_PATTERN = re.compile(r"(img_\d+)")
SCALE_PATTERN = re.compile(r"_scale(050|075|100)$")
EXPECTED_ROTATIONS = {"r0", "r90", "r180", "r270"}
EXPECTED_SCALES = {"050", "075", "100"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("edge_model/configs/rbcm/nyudv2_strict.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge_outputs/rbcm/audits/nyudv2_strict_20260724"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stems(path: Path) -> list[str]:
    return [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def base_id(stem: str) -> str:
    match = ID_PATTERN.search(stem)
    if match is None:
        raise ValueError(f"Cannot recover NYUD source ID from {stem}")
    return match.group(1)


def augmentation_signature(stem: str) -> tuple[str, str, str]:
    scale_match = SCALE_PATTERN.search(stem)
    if scale_match is None:
        raise ValueError(f"Cannot recover scale from {stem}")
    if "_r90_" in stem:
        rotation = "r90"
    elif "_r180_" in stem:
        rotation = "r180"
    elif "_r270_" in stem:
        rotation = "r270"
    else:
        rotation = "r0"
    flip = "flip" if "_flip_" in stem else "plain"
    return scale_match.group(1), rotation, flip


def file_index(folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        if path.stem in index:
            duplicates.append(path.stem)
        index[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate stems under {folder}: {duplicates[:5]}")
    return index


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.notes: list[str] = []

    def check(self, condition: bool, name: str, detail: str) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    @property
    def failures(self) -> list[dict]:
        return [row for row in self.checks if row["status"] == "FAIL"]


def audit_splits(audit: Audit, root: Path, manifest: dict) -> dict:
    strict_root = root / "splits" / "strict_381_414"
    paths = {name: strict_root / f"{name}.txt" for name in ("train", "val", "test")}
    rows = {name: read_stems(path) for name, path in paths.items()}
    ids = {name: [base_id(stem) for stem in values] for name, values in rows.items()}
    unique_ids = {name: set(values) for name, values in ids.items()}

    expected_rows = {"train": 381 * 24, "val": 414, "test": 654}
    expected_ids = {"train": 381, "val": 414, "test": 654}
    for name in ("train", "val", "test"):
        audit.check(
            len(rows[name]) == expected_rows[name],
            f"{name}_row_count",
            f"{len(rows[name])} rows; expected {expected_rows[name]}",
        )
        audit.check(
            len(unique_ids[name]) == expected_ids[name],
            f"{name}_source_count",
            f"{len(unique_ids[name])} unique source IDs; expected {expected_ids[name]}",
        )
        audit.check(
            len(rows[name]) == len(set(rows[name])),
            f"{name}_no_duplicate_stems",
            "Split stems are unique.",
        )
        expected_hash = manifest["splits"][name]["sha256"]
        audit.check(
            sha256(paths[name]) == expected_hash,
            f"{name}_manifest_hash",
            f"SHA-256 matches manifest: {expected_hash}",
        )

    overlaps = {
        "train_val": unique_ids["train"] & unique_ids["val"],
        "train_test": unique_ids["train"] & unique_ids["test"],
        "val_test": unique_ids["val"] & unique_ids["test"],
    }
    for name, overlap in overlaps.items():
        audit.check(not overlap, f"{name}_disjoint", f"Overlap count: {len(overlap)}")

    source_path = root / "splits" / "train.txt"
    source_rows = read_stems(source_path)
    audit.check(
        sha256(source_path) == manifest["source"]["local_sha256"],
        "ordered_source_hash",
        "The preserved 19,080-row PiDiNet/RCF list matches the frozen manifest.",
    )
    audit.check(
        len(source_rows) == 795 * 24,
        "ordered_source_size",
        f"{len(source_rows)} rows; expected 795 x 24.",
    )
    first_block_ids = [base_id(stem) for stem in source_rows[:795]]
    block_order_ok = len(set(first_block_ids)) == 795
    for block_index in range(24):
        start = block_index * 795
        block_ids = [base_id(stem) for stem in source_rows[start : start + 795]]
        block_order_ok = block_order_ok and block_ids == first_block_ids
    audit.check(
        block_order_ok,
        "ordered_24_block_identity",
        "All 24 augmentation blocks preserve the same 795-image source order.",
    )
    audit.check(
        unique_ids["train"] == set(first_block_ids[:381]),
        "train_boundary_recovery",
        "Strict train IDs equal the first 381 IDs in the frozen combined list.",
    )
    audit.check(
        unique_ids["val"] == set(first_block_ids[381:]),
        "validation_boundary_recovery",
        "Strict validation IDs equal the following 414 IDs.",
    )

    signatures: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    counts = Counter(ids["train"])
    for stem in rows["train"]:
        signatures[base_id(stem)].add(augmentation_signature(stem))
    expected_signatures = {
        (scale, rotation, flip)
        for scale in EXPECTED_SCALES
        for rotation in EXPECTED_ROTATIONS
        for flip in {"plain", "flip"}
    }
    augmentation_ok = all(
        counts[source_id] == 24 and signatures[source_id] == expected_signatures
        for source_id in unique_ids["train"]
    )
    audit.check(
        augmentation_ok,
        "train_24_way_augmentation",
        "Every one of 381 train IDs has exactly 3 scales x 4 rotations x 2 flips.",
    )
    audit.notes.append(
        "The 381/414 boundary is recovered from the frozen ordered PiDiNet/RCF "
        "train+val list. This is documented explicitly rather than described as "
        "a newly invented random split."
    )
    return {"paths": paths, "rows": rows, "ids": unique_ids}


def audit_files(audit: Audit, root: Path, split_data: dict) -> dict:
    images = file_index(root / "image")
    edges = file_index(root / "edge")
    soft_votes = file_index(root / "gt" / "soft_vote")
    all_rows = [stem for name in ("train", "val", "test") for stem in split_data["rows"][name]]
    missing_images = [stem for stem in all_rows if stem not in images]
    missing_edges = [stem for stem in all_rows if stem not in edges]
    audit.check(not missing_images, "all_images_present", f"Missing images: {missing_images[:5]}")
    audit.check(not missing_edges, "all_edges_present", f"Missing edge maps: {missing_edges[:5]}")

    mismatch: list[dict] = []
    dimension_counts: dict[str, Counter] = {name: Counter() for name in ("train", "val", "test")}
    for name in ("train", "val", "test"):
        for stem in split_data["rows"][name]:
            with Image.open(images[stem]) as image:
                image_size = image.size
                image_mode = image.mode
            with Image.open(edges[stem]) as edge:
                edge_size = edge.size
                edge_mode = edge.mode
            dimension_counts[name][image_size] += 1
            if image_size != edge_size or image_mode not in {"RGB", "RGBA"} or edge_mode not in {"L", "P"}:
                mismatch.append(
                    {
                        "stem": stem,
                        "image_size": image_size,
                        "edge_size": edge_size,
                        "image_mode": image_mode,
                        "edge_mode": edge_mode,
                    }
                )
    audit.check(
        not mismatch,
        "image_target_geometry",
        f"Checked {len(all_rows)} image/edge pairs; mismatches: {mismatch[:3]}",
    )

    test_rows = split_data["rows"]["test"]
    missing_soft = [stem for stem in test_rows if stem not in soft_votes]
    soft_mismatch: list[str] = []
    soft_alias_mismatch: list[str] = []
    soft_nonbinary = 0
    for stem in test_rows:
        if stem not in soft_votes:
            continue
        with (
            Image.open(images[stem]) as image,
            Image.open(edges[stem]) as edge,
            Image.open(soft_votes[stem]) as target,
        ):
            if image.size != target.size:
                soft_mismatch.append(stem)
            target_l = target.convert("L")
            colors = target_l.getcolors(maxcolors=256)
            if colors is not None and len(colors) > 2:
                soft_nonbinary += 1
            if ImageChops.difference(edge.convert("L"), target_l).getbbox() is not None:
                soft_alias_mismatch.append(stem)
    audit.check(not missing_soft, "test_soft_vote_present", f"Missing soft-vote GT: {missing_soft[:5]}")
    audit.check(not soft_mismatch, "test_soft_vote_geometry", f"Size mismatches: {soft_mismatch[:5]}")
    audit.check(
        soft_nonbinary == 0 and not soft_alias_mismatch,
        "test_legacy_gt_alias",
        (
            f"The historical soft_vote folder is binary and pixel-identical to edge "
            f"for all {len(test_rows)} test images; alias mismatches: {soft_alias_mismatch[:5]}"
        ),
    )
    return {
        "dimension_counts": {
            name: {f"{width}x{height}": count for (width, height), count in counts.items()}
            for name, counts in dimension_counts.items()
        }
    }


def audit_config(audit: Audit, config: dict, config_path: Path) -> dict:
    dataset = config["dataset"]
    loader = config["loader"]
    train = config["train"]
    protocol = config["paper_protocol"]
    final_test = protocol["final_test"]

    expected_split_files = {
        "train": "splits/strict_381_414/train.txt",
        "val": "splits/strict_381_414/val.txt",
        "test": "splits/strict_381_414/test.txt",
    }
    audit.check(
        dataset.get("split_files") == expected_split_files,
        "config_strict_split_files",
        f"Configured split files: {dataset.get('split_files')}",
    )
    audit.check(
        dataset.get("train_split") == "train"
        and dataset.get("val_split") == "val"
        and dataset.get("eval_split") == "val",
        "config_validation_only_selection",
        "Training evaluates the independent 414-image validation split, never test.",
    )
    audit.check(
        dataset.get("native_size_train") is True
        and dataset.get("native_size_eval") is True
        and dataset.get("random_crop") is False
        and dataset.get("horizontal_flip") is False
        and dataset.get("vertical_flip") is False,
        "config_no_duplicate_runtime_augmentation",
        "Uses native pre-generated images without extra crop/flip augmentation.",
    )
    audit.check(
        loader.get("batch_size") == 1
        and loader.get("persistent_workers") is False,
        "config_variable_size_loader",
        "Batch size 1 supports native variable shapes; workers are restartable.",
    )
    seed = config.get("seed")
    deterministic = bool(config.get("deterministic", True))
    audit.check(
        isinstance(seed, int),
        "config_seeded_runtime",
        (
            f"Fixed seed={seed}; each epoch also uses a fixed sampler seed. "
            f"CUDA deterministic algorithms={deterministic}. When false, cuDNN "
            "benchmarking improves throughput but independent runs are not "
            "guaranteed to be bitwise identical."
        ),
    )
    audit.check(
        train.get("resumable_mid_epoch") is True
        and int(train.get("checkpoint_interval_steps", 0)) <= 500
        and float(train.get("periodic_checkpoint_minutes", 0.0)) <= 10.0,
        "config_mid_epoch_recovery",
        "Full-state recovery is refreshed every <=500 steps or <=10 minutes.",
    )
    audit.check(
        int(train.get("epochs", 0)) == 14
        and train.get("scheduler") == "multistep"
        and train.get("scheduler_milestones") == [8, 12],
        "config_schedule",
        "14 epochs with fixed MultiStep milestones at 8 and 12.",
    )
    audit.check(
        train.get("checkpoint_score", {}).get("mode", "").lower() == "ods"
        and train.get("strict_checkpoint_selection", {}).get("enabled") is False,
        "config_checkpoint_selection",
        "Best checkpoint is selected once per epoch by validation ODS.",
    )
    audit.check(
        int(train.get("fast_metric_tolerance_pixels", 0)) == 9
        and train.get("apply_nms") is True,
        "config_validation_tolerance",
        "640x480 validation uses NMS and 9 px, the rounded 0.011 x 800 diagonal.",
    )
    audit.check(
        final_test.get("split") == "test"
        and final_test.get("gt_variant") == "edge"
        and int(final_test.get("thresholds", 0)) == 99
        and final_test.get("apply_nms") is True
        and abs(float(final_test.get("match_tolerance", 0.0)) - 0.011) < 1e-12
        and final_test.get("restore_original_size") is True,
        "config_final_test_contract",
        "Frozen checkpoint: 654 test images, binary edge GT, 99 thresholds, NMS, tolerance 0.011.",
    )
    audit.check(
        config["model"].get("host") == "hed_lite"
        and config["model"].get("variant") == "plain",
        "config_anchor_training_role",
        "The trainable source is the shared HED-lite anchor; H-RBCM modes are calibrated afterward.",
    )
    audit.check(
        dataset.get("max_train_samples") is None and dataset.get("max_eval_samples") is None,
        "config_no_sample_caps",
        "Formal config has no train/validation sample cap.",
    )

    runtime_config = json.loads(json.dumps(config))
    runtime_config.setdefault("paths", {})["project_root"] = str(PROJECT_ROOT)
    train_dataset = make_dataset(runtime_config, "NYUDv2", "train", training=True)
    val_dataset = make_dataset(runtime_config, "NYUDv2", "val", training=False)
    train_shapes = [list(train_dataset[index]["image"].shape) for index in (0, 381, 762)]
    val_shape = list(val_dataset[0]["image"].shape)
    audit.check(
        len(train_dataset) == 9144 and len(val_dataset) == 414,
        "runtime_dataset_lengths",
        f"Runtime train={len(train_dataset)}, validation={len(val_dataset)}.",
    )
    audit.check(
        train_shapes == [[3, 480, 640], [3, 640, 480], [3, 480, 640]]
        and val_shape == [3, 480, 640],
        "runtime_native_shapes",
        f"Representative train shapes={train_shapes}; validation shape={val_shape}.",
    )
    return {
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "representative_train_shapes": train_shapes,
        "representative_val_shape": val_shape,
    }


def markdown_report(payload: dict, chinese: bool) -> str:
    title = "NYUDv2 严格训练协议审计" if chinese else "NYUDv2 Strict Training Protocol Audit"
    status_label = "总体状态" if chinese else "Overall status"
    check_label = "检查项" if chinese else "Check"
    detail_label = "证据" if chinese else "Evidence"
    config_label = "配置" if chinese else "Config"
    split_label = "划分" if chinese else "Split"
    augmentation_label = "增广" if chinese else "Augmentation"
    evaluation_label = "最终 NYUD 评估" if chinese else "Final NYUD evaluation"
    lines = [
        f"# {title}",
        "",
        f"- {status_label}: **{payload['status']}**",
        f"- {config_label}: `{payload['runtime']['config']}`",
        f"- {split_label}: `381 train / 414 validation / 654 test`",
        f"- {augmentation_label}: `3 scales x 4 rotations x 2 flips = 24` (train only)",
        f"- {evaluation_label}: `NMS + 99 thresholds + maxDist 0.011`",
        "",
        f"| {check_label} | Status | {detail_label} |",
        "|---|---:|---|",
    ]
    for row in payload["checks"]:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| `{row['name']}` | {row['status']} | {detail} |")
    lines.extend(["", "## Notes" if not chinese else "## 说明", ""])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Primary protocol references" if not chinese else "## 主流协议依据",
            "",
            "- HED: canonical NYUD split `381/414/654`, full-resolution processing, maxDist `0.011`.",
            "- RCF/PiDiNet: train+val package with `2 flips x 3 scales x 4 rotations`; this run retains only the 381 training IDs and reserves the following 414 IDs for validation.",
            "- The 654-image test split is never used for checkpoint or calibration selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config).resolve()
    output = resolve(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = PROJECT_ROOT / "edge_data" / "official_rbcm" / "NYUDv2"
    manifest_path = dataset_root / "splits" / "strict_381_414" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    audit = Audit()
    split_data = audit_splits(audit, dataset_root, manifest)
    file_details = audit_files(audit, dataset_root, split_data)
    runtime = audit_config(audit, config, config_path)
    payload = {
        "status": "FAIL" if audit.failures else "PASS",
        "checks": audit.checks,
        "notes": audit.notes,
        "runtime": runtime,
        "files": file_details,
        "manifest": manifest,
    }
    (output / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(markdown_report(payload, chinese=False), encoding="utf-8")
    (output / "README.zh-CN.md").write_text(markdown_report(payload, chinese=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
