"""Create a source-disjoint 68/12/20 MultiCue split.

The normalized MultiCue training package contains 80 source images, each with
96 pre-generated descendants. The existing 20-image report split is retained
as the fixed test set. Twelve source images are selected deterministically from
the original 80 for validation; every descendant follows its source assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "edge_data" / "official_rbcm" / "Multicue"
SOURCE_TRAIN = DATASET_ROOT / "splits" / "train.txt"
SOURCE_TEST = DATASET_ROOT / "splits" / "test.txt"
OUTPUT_DIR = DATASET_ROOT / "splits" / "strict_68_12_20"

SEED = 4517
SOURCE_COUNT = 80
VALIDATION_SOURCES = 12
DESCENDANTS_PER_SOURCE = 96
TEST_SOURCES = 20
STEM_PATTERN = re.compile(r"^images_s[^_]+_r[^_]+_(?:flip_1_)?(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, default=SOURCE_TRAIN)
    parser.add_argument("--source-test", type=Path, default=SOURCE_TEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
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
    if match is None:
        return stem
    return match.group(1)


def stable_key(source: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source}".encode("utf-8")).hexdigest()


def original_stem(source: str) -> str:
    return f"images_s1.0_r0.0_{source}"


def assert_assets(stems: list[str], *, require_weight: bool) -> None:
    required_dirs = [
        DATASET_ROOT / "image",
        DATASET_ROOT / "edge",
        DATASET_ROOT / "gt" / "soft_vote",
    ]
    if require_weight:
        required_dirs.append(DATASET_ROOT / "gt" / "multicue_ignore_weight")
    missing: list[str] = []
    for stem in stems:
        for directory in required_dirs:
            if not (directory / f"{stem}.png").exists():
                missing.append(str(directory / f"{stem}.png"))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError("Missing MultiCue assets:\n" + "\n".join(missing))


def write_split(path: Path, stems: list[str]) -> None:
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_train = args.source_train.resolve()
    source_test = args.source_test.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_split(source_train)
    test_rows = read_split(source_test)
    if len(train_rows) != SOURCE_COUNT * DESCENDANTS_PER_SOURCE:
        raise ValueError(
            f"Expected {SOURCE_COUNT * DESCENDANTS_PER_SOURCE} augmented rows, "
            f"found {len(train_rows)}"
        )
    if len(test_rows) != TEST_SOURCES:
        raise ValueError(f"Expected {TEST_SOURCES} fixed test images, found {len(test_rows)}")

    groups: dict[str, list[str]] = {}
    for stem in train_rows:
        groups.setdefault(source_id(stem), []).append(stem)
    if len(groups) != SOURCE_COUNT:
        raise ValueError(f"Expected {SOURCE_COUNT} source groups, found {len(groups)}")
    bad_groups = {key: len(value) for key, value in groups.items() if len(value) != DESCENDANTS_PER_SOURCE}
    if bad_groups:
        raise ValueError(f"Each source must have {DESCENDANTS_PER_SOURCE} descendants: {bad_groups}")

    ranked_sources = sorted(groups, key=lambda item: stable_key(item, int(args.seed)))
    validation_sources = set(ranked_sources[:VALIDATION_SOURCES])
    training_sources = set(ranked_sources[VALIDATION_SOURCES:])
    test_sources = set(test_rows)
    if training_sources & validation_sources:
        raise RuntimeError("Training and validation source IDs overlap")
    if (training_sources | validation_sources) & test_sources:
        raise RuntimeError("Original training and fixed test source IDs overlap")

    strict_train = [stem for stem in train_rows if source_id(stem) in training_sources]
    strict_val = [original_stem(source) for source in ranked_sources[:VALIDATION_SOURCES]]
    strict_test = list(test_rows)
    if any(stem not in groups[source_id(stem)] for stem in strict_val):
        raise RuntimeError("A validation source is missing its unaugmented scale-1 original")

    assert_assets(strict_train, require_weight=True)
    assert_assets(strict_val, require_weight=False)
    assert_assets(strict_test, require_weight=False)

    split_paths = {
        "train": output_dir / "train.txt",
        "val": output_dir / "val.txt",
        "test": output_dir / "test.txt",
    }
    write_split(split_paths["train"], strict_train)
    write_split(split_paths["val"], strict_val)
    write_split(split_paths["test"], strict_test)

    assignment_path = output_dir / "source_assignment.csv"
    with assignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "descendant_count"])
        writer.writeheader()
        for source in sorted(training_sources):
            writer.writerow(
                {"source_id": source, "split": "train", "descendant_count": len(groups[source])}
            )
        for source in sorted(validation_sources):
            writer.writerow({"source_id": source, "split": "val", "descendant_count": 1})
        for source in sorted(test_sources):
            writer.writerow({"source_id": source, "split": "test", "descendant_count": 1})

    manifest = {
        "protocol": "MultiCue fixed source-disjoint 68/12/20 confirmation",
        "seed": int(args.seed),
        "selection_rule": "ascending SHA256(seed:source_id); first 12 of original 80 are validation",
        "source_train": str(source_train),
        "source_test": str(source_test),
        "source_hashes": {
            "train": sha256(source_train),
            "test": sha256(source_test),
        },
        "counts": {
            "train_sources": len(training_sources),
            "train_rows": len(strict_train),
            "val_sources": len(validation_sources),
            "val_rows": len(strict_val),
            "test_sources": len(test_sources),
            "test_rows": len(strict_test),
            "descendants_per_training_source": DESCENDANTS_PER_SOURCE,
        },
        "split_hashes": {name: sha256(path) for name, path in split_paths.items()},
        "source_assignment_sha256": sha256(assignment_path),
        "test_is_original_report_split": strict_test == test_rows,
        "source_overlap": {
            "train_val": sorted(training_sources & validation_sources),
            "train_test": sorted(training_sources & test_sources),
            "val_test": sorted(validation_sources & test_sources),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
