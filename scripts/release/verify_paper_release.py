"""Verify the RBCM-Edge source, checkpoint, edge-data, and MEA-data packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_RELEASE_ROOT = PROJECT_ROOT / "release"
if (DEVELOPMENT_RELEASE_ROOT / "github" / "RBCM-Edge").is_dir():
    CODE_ROOT = DEVELOPMENT_RELEASE_ROOT / "github" / "RBCM-Edge"
    RELEASE_ROOT = DEVELOPMENT_RELEASE_ROOT
else:
    CODE_ROOT = PROJECT_ROOT
    RELEASE_ROOT = PROJECT_ROOT.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    parser.add_argument(
        "--checkpoint-package",
        type=Path,
        default=RELEASE_ROOT / "checkpoints" / "RBCM-Edge-Checkpoints",
    )
    parser.add_argument(
        "--dataset-package",
        type=Path,
        default=RELEASE_ROOT / "datasets" / "RBCM-Edge-Datasets",
    )
    parser.add_argument(
        "--mea-data-package",
        type=Path,
        default=RELEASE_ROOT / "datasets" / "RBCM-Edge-MEA-Data",
    )
    parser.add_argument(
        "--full-data-index",
        action="store_true",
        help="Check the existence and byte size of every indexed dataset file.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_vcs_metadata(path: Path, root: Path) -> bool:
    return ".git" in path.relative_to(root).parts


def verify_sha_manifest(root: Path) -> int:
    manifest = root / "MANIFEST_SHA256.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    indexed: set[str] = set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            indexed.add(row["path"])
            path = root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_size = int(row["bytes"])
            if path.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch: {path}")
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
            checked += 1
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest and not is_vcs_metadata(path, root)
    }
    if actual != indexed:
        missing = sorted(indexed - actual)
        extra = sorted(actual - indexed)
        raise RuntimeError(
            f"Release manifest membership mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return checked


def verify_dataset(package: Path, full_index: bool) -> tuple[int, int, int]:
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    expected_files = int(manifest["file_count"])
    expected_bytes = int(manifest["total_bytes"])

    indexed_files = 0
    indexed_bytes = 0
    with (package / "FILE_INDEX.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            indexed_files += 1
            size = int(row["bytes"])
            indexed_bytes += size
            if full_index:
                path = package / "edge_data" / row["path"]
                if not path.is_file() or path.stat().st_size != size:
                    raise RuntimeError(f"Dataset index mismatch: {path}")
    if (indexed_files, indexed_bytes) != (expected_files, expected_bytes):
        raise RuntimeError(
            "Dataset manifest/index mismatch: "
            f"{indexed_files}/{indexed_bytes} vs {expected_files}/{expected_bytes}"
        )

    protocol_files = 0
    with (package / "PROTOCOL_SHA256.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            path = package / "edge_data" / row["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                raise RuntimeError(f"Dataset protocol mismatch: {path}")
            protocol_files += 1
    if protocol_files != int(manifest["protocol_file_count"]):
        raise RuntimeError("Dataset protocol file count differs from MANIFEST.json")
    return indexed_files, indexed_bytes, protocol_files


def verify_mea_data(package: Path, full_index: bool) -> tuple[int, int]:
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    expected_files = int(manifest["file_count"])
    expected_bytes = int(manifest["total_bytes"])
    indexed_files = 0
    indexed_bytes = 0
    with (package / "FILE_INDEX.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            indexed_files += 1
            size = int(row["bytes"])
            indexed_bytes += size
            if full_index:
                path = package / "MEA_data" / row["path"]
                if not path.is_file() or path.stat().st_size != size:
                    raise RuntimeError(f"MEA data index mismatch: {path}")
                if sha256(path) != row["sha256"]:
                    raise RuntimeError(f"MEA data SHA-256 mismatch: {path}")
    if (indexed_files, indexed_bytes) != (expected_files, expected_bytes):
        raise RuntimeError(
            "MEA manifest/index mismatch: "
            f"{indexed_files}/{indexed_bytes} vs {expected_files}/{expected_bytes}"
        )
    return indexed_files, indexed_bytes


def verify_strict_result_metadata(code_root: Path) -> None:
    required_configs = (
        code_root / "edge_model/configs/rbcm/multicue_strict.yaml",
        code_root / "edge_model/configs/rbcm/nyudv2_strict.yaml",
    )
    for path in required_configs:
        if not path.is_file():
            raise FileNotFoundError(path)

    strict_specs = {
        "multicue": code_root / "docs/results/strict/multicue/formal_summary.json",
        "nyudv2": code_root / "docs/results/strict/nyudv2/formal_summary.json",
    }
    for dataset, path in strict_specs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if dataset == "multicue":
            modes = set(payload.get("modes", {}))
        else:
            modes = {"plain_identity", *payload.get("selected", {}).keys()}
        expected_modes = {
            "plain_identity",
            "main_surround",
            "no_surround",
            "conv_control",
        }
        if modes != expected_modes:
            raise RuntimeError(
                f"{dataset} strict summary modes differ: {sorted(modes)}"
            )

    with (code_root / "docs/results/formal_result_index.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for dataset in ("Multicue", "NYUDv2"):
        selected = [
            row for row in rows
            if row["training_source"] == dataset
            and row["scope"] == "same_domain_strict"
        ]
        if len(selected) != 4 or any(
            row["independent_test"].lower() != "true" for row in selected
        ):
            raise RuntimeError(
                f"Expected four independent strict rows for {dataset}, "
                f"found {len(selected)}"
            )
    if any(
        row["training_source"] == "Multicue"
        and "descriptive" in row["scope"].lower()
        for row in rows
    ):
        raise RuntimeError("Archival MultiCue descriptive rows leaked into formal index")

    # Construct the markers at runtime so the verifier does not embed and
    # subsequently flag the exact private path strings in its own public copy.
    forbidden_markers = (
        "/workspace/" + "RBCM-Edge",
        "D:\\" + "study\\project\\RBCM-Edge",
    )
    for path in code_root.rglob("*"):
        if (
            path.is_file()
            and not is_vcs_metadata(path, code_root)
            and path.suffix.lower() in {
            ".py", ".yaml", ".yml", ".json", ".csv", ".md", ".txt"
            }
        ):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in forbidden_markers):
                raise RuntimeError(f"Machine-specific path leaked into public source: {path}")


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    checkpoint_package = args.checkpoint_package.resolve()
    dataset_package = args.dataset_package.resolve()
    mea_data_package = args.mea_data_package.resolve()

    forbidden = {
        "backup",
        "MEA_data",
        "MEA_outputs",
        "external",
        "legacy_edge",
        "paper_assets",
        "results",
        "weights",
    }
    present = sorted(
        path.name
        for path in code_root.iterdir()
        if path.is_dir() and path.name in forbidden
    )
    if present:
        raise RuntimeError(f"Historical/non-release directories in source package: {present}")
    if list(code_root.rglob("*.pt")):
        raise RuntimeError("GitHub source package must not contain checkpoint binaries.")
    for name in ("archive_exploratory", "tmp1_pair_stability_summary",
                 "tmp2_permutation_null_examples", "tmp3_effect_distribution_summary"):
        if list(code_root.rglob(name)):
            raise RuntimeError(f"Historical MEA directory in source package: {name}")

    code_files = verify_sha_manifest(code_root)
    verify_strict_result_metadata(code_root)
    checkpoint_files = verify_sha_manifest(checkpoint_package)
    expected_checkpoints = {
        "pretrained/biped/main/best.pt",
        "pretrained/biped/split0/best.pt",
        "pretrained/biped/split1/best.pt",
        "pretrained/biped/split2/best.pt",
        "pretrained/multicue_strict/best.pt",
        "pretrained/nyudv2_strict/best.pt",
    }
    checkpoint_paths = {
        path.relative_to(checkpoint_package).as_posix()
        for path in checkpoint_package.rglob("*.pt")
    }
    if checkpoint_paths != expected_checkpoints:
        raise RuntimeError(
            "Selected checkpoint membership differs: "
            f"missing={sorted(expected_checkpoints - checkpoint_paths)}, "
            f"extra={sorted(checkpoint_paths - expected_checkpoints)}"
        )
    for relative in (
        "pretrained/multicue_strict/CANDIDATES_FROZEN.sha256",
        "pretrained/multicue_strict/protocol_manifest.json",
        "pretrained/multicue_strict/formal_summary.json",
        "pretrained/multicue_strict/checkpoint_provenance.json",
        "pretrained/nyudv2_strict/protocol_manifest.json",
        "pretrained/nyudv2_strict/formal_summary.json",
    ):
        if not (checkpoint_package / relative).is_file():
            raise FileNotFoundError(checkpoint_package / relative)

    data_files, data_bytes, protocol_files = verify_dataset(
        dataset_package,
        args.full_data_index,
    )
    mea_files, mea_bytes = verify_mea_data(
        mea_data_package,
        args.full_data_index,
    )
    print(
        "release_verify=PASS "
        f"code_manifest_files={code_files} "
        f"checkpoint_manifest_files={checkpoint_files} "
        f"checkpoint_count={len(checkpoint_paths)} "
        f"dataset_files={data_files} "
        f"dataset_bytes={data_bytes} "
        f"protocol_files={protocol_files} "
        f"mea_files={mea_files} "
        f"mea_bytes={mea_bytes} "
        f"full_data_index={int(args.full_data_index)}"
    )


if __name__ == "__main__":
    main()
