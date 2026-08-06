"""Verify the V5-aligned RBCM-Edge code-and-configuration release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = root / "MANIFEST_SHA256.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    indexed: dict[str, tuple[int, str]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            indexed[row["path"]] = (int(row["bytes"]), row["sha256"])
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest
        and ".git" not in path.relative_to(root).parts
    }
    if set(actual) != set(indexed):
        raise RuntimeError(
            "Manifest membership mismatch; "
            f"missing={sorted(set(indexed) - set(actual))[:5]}, "
            f"extra={sorted(set(actual) - set(indexed))[:5]}"
        )
    for relative, path in actual.items():
        expected_size, expected_hash = indexed[relative]
        if path.stat().st_size != expected_size or sha256(path) != expected_hash:
            raise RuntimeError(f"Manifest mismatch: {relative}")
    return len(actual)


def verify_membership(root: Path) -> None:
    required = (
        "README.md",
        "README.zh-CN.md",
        "REPRODUCE.md",
        "REPRODUCE.zh-CN.md",
        "edge_model/configs/rbcm/biped.yaml",
        "edge_model/configs/rbcm/multicue_strict.yaml",
        "edge_model/configs/rbcm/nyudv2_strict.yaml",
        "edge_model/models/build.py",
        "src/rbcm_edge/models/networks/anchor.py",
        "src/rbcm_edge/models/losses.py",
        "scripts/experiments/calibrate.py",
        "scripts/experiments/evaluate_generalization.py",
        "scripts/analysis/evaluate_nyud_strict_generalization.py",
        "scripts/analysis/reproduce_figure5_relative_statistics.py",
        "scripts/figures/bridge/render_figure5_relative_panels.py",
        "scripts/release/smoke_paper_release.py",
        "scripts/release/verify_paper_release.py",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)

    forbidden_top_level = {
        "backup",
        "legacy_edge",
        "docs",
        "paper_assets",
        "MEA_data",
        "MEA_outputs",
        "edge_data",
        "edge_outputs",
        "outputs",
        "results",
        "pretrained",
        "checkpoints",
        "figures",
        "weights",
    }
    leaked_dirs = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name in forbidden_top_level
    )
    if leaked_dirs:
        raise RuntimeError(f"Non-source directories leaked into release: {leaked_dirs[:10]}")
    historical_dirs = {
        "legacy_edge",
        "archive_exploratory",
        "tmp1_pair_stability_summary",
        "tmp2_permutation_null_examples",
        "tmp3_effect_distribution_summary",
    }
    historical = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and path.name in historical_dirs
    )
    if historical:
        raise RuntimeError(f"Historical source directories leaked into release: {historical}")

    forbidden_suffixes = {
        ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".xlsx", ".pt", ".pth", ".ckpt", ".onnx", ".npy", ".npz",
        ".h5", ".mat", ".json",
    }
    leaked_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    )
    if leaked_files:
        raise RuntimeError(f"Binary/result files leaked into release: {leaked_files[:10]}")
    csv_files = [
        path for path in root.rglob("*.csv")
        if path.name != "MANIFEST_SHA256.csv"
    ]
    if csv_files:
        raise RuntimeError(
            "Experimental CSV files leaked into release: "
            + ", ".join(path.relative_to(root).as_posix() for path in csv_files[:10])
        )


def load_config(root: Path, name: str) -> dict:
    path = root / "edge_model" / "configs" / "rbcm" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_v5_configs(root: Path) -> None:
    configs = {
        "biped": load_config(root, "biped.yaml"),
        "multicue": load_config(root, "multicue_strict.yaml"),
        "nyudv2": load_config(root, "nyudv2_strict.yaml"),
    }
    for dataset, config in configs.items():
        model = config["model"]
        expected = {
            "host": "hed_lite",
            "variant": "plain",
            "in_channels": 3,
            "feature_channels": 48,
            "decoder_channels": 64,
            "norm": "gn",
            "gn_groups": 8,
            "activation": "relu",
        }
        if any(model.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{dataset} model config differs from V5: {model}")
        inactive_history = {
            "rbcm_reduction", "ring_token_extractor", "surround_attention", "modulation"
        }
        if inactive_history.intersection(model):
            raise RuntimeError(f"Historical trainable RBCM fields remain in {dataset}")
        loss = config["loss"]
        expected_loss = {
            "dice_weight": 1.0,
            "local_weight": 0.24,
            "side_weight": 0.18,
            "context_weight": 0.08,
            "density_weight": 0.002,
            "tversky_weight": 0.22,
            "far_background_weight": 0.03,
        }
        if any(float(loss.get(key, -1)) != value for key, value in expected_loss.items()):
            raise RuntimeError(f"{dataset} principal loss weights differ from V5")
    multicue = configs["multicue"]
    protocol = multicue["paper_protocol"]
    if (
        int(protocol["train_source_images"]) != 68
        or int(protocol["validation_source_images"]) != 12
        or int(protocol["test_images"]) != 20
    ):
        raise RuntimeError("Strict MultiCue protocol is not 68/12/20")
    nyud = configs["nyudv2"]["paper_protocol"]
    if (
        int(nyud["train_source_images"]) != 381
        or int(nyud["validation_images"]) != 414
        or int(nyud["test_images"]) != 654
    ):
        raise RuntimeError("Strict NYUDv2 protocol is not 381/414/654")


def verify_v5_runtime(root: Path) -> tuple[int, tuple[int, ...]]:
    import torch

    for import_root in (root, root / "src"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from edge_model.models.build import build_model
    from rbcm_edge.models.losses import EdgeDetectionLoss
    from scripts.analysis.reproduce_figure5_relative_statistics import (
        mean_mad_relative,
        state_row,
    )
    from scripts.experiments.calibrate import Candidate, apply_candidate

    config = load_config(root, "multicue_strict.yaml")
    model = build_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 6_638_807:
        raise RuntimeError(f"Unexpected HED-lite parameter count: {parameter_count}")
    if model.modulator is not None:
        raise RuntimeError("The released anchor unexpectedly contains a trainable calibrator")
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 48, 64))
    if tuple(outputs["logits"].shape) != (1, 1, 48, 64):
        raise RuntimeError("Unexpected final Anchor output shape")
    side_logits = outputs.get("side_logits")
    if not isinstance(side_logits, list) or len(side_logits) != 4:
        raise RuntimeError("The released Anchor must expose four side logits")
    pyramid = outputs.get("pyramid_features")
    if not isinstance(pyramid, tuple) or [int(item.shape[1]) for item in pyramid] != [48, 96, 192, 384]:
        raise RuntimeError("The released Anchor feature hierarchy differs from V5")
    loss_config = config["loss"]
    criterion = EdgeDetectionLoss(
        dice_weight=float(loss_config["dice_weight"]),
        local_weight=float(loss_config["local_weight"]),
        side_weight=float(loss_config["side_weight"]),
        context_weight=float(loss_config["context_weight"]),
        context_dilation=int(loss_config["context_dilation"]),
        context_gamma=float(loss_config["context_gamma"]),
        gate_sparsity_weight=float(loss_config["gate_sparsity_weight"]),
        density_weight=float(loss_config["density_weight"]),
        density_target_multiplier=float(loss_config["density_target_multiplier"]),
        density_floor=float(loss_config["density_floor"]),
        tversky_weight=float(loss_config["tversky_weight"]),
        tversky_alpha=float(loss_config["tversky_alpha"]),
        tversky_beta=float(loss_config["tversky_beta"]),
        tversky_gamma=float(loss_config["tversky_gamma"]),
        far_background_weight=float(loss_config["far_background_weight"]),
        far_background_dilation=int(loss_config["far_background_dilation"]),
        far_background_gamma=float(loss_config["far_background_gamma"]),
        mix_balance_weight=float(loss_config["mix_balance_weight"]),
        mix_prior=loss_config["mix_prior"],
        target_threshold=float(loss_config["target_threshold"]),
    )
    train_outputs = model(torch.rand(1, 3, 32, 40))
    target = (torch.rand(1, 1, 32, 40) > 0.9).float()
    loss_parts = criterion(
        train_outputs["logits"],
        target,
        local_logits=train_outputs["local_logits"],
        context_logits=train_outputs["context_logits"],
        side_logits=train_outputs["side_logits"],
        gate=train_outputs["gate"],
        mix_weights=train_outputs["mix_weights"],
    )
    loss_parts["total"].backward()
    if not any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("The V5 Anchor forward/backward smoke test failed")

    probability = np.linspace(0.05, 0.95, 48 * 64, dtype=np.float32).reshape(48, 64)
    edge = np.flipud(probability).copy()
    candidate = Candidate(
        mode="main_surround",
        ring="1-7-7-17",
        alpha=0.2,
        edge_weight=0.6,
        prob_weight=0.4,
        uncertainty_power=1.0,
        temperature=1.0,
        bias=0.0,
        sharpen=0.0,
    )
    calibrated = apply_candidate(
        {"sample_id": "audit", "prob": probability, "edge": edge},
        candidate,
    )
    if calibrated.shape != probability.shape or not np.isfinite(calibrated).all():
        raise RuntimeError("V5 annular calibration smoke test failed")

    values = np.asarray([-3.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
    z = mean_mad_relative(values)
    row = state_row(values)
    if not np.isclose(float(z.mean()), 0.0, atol=1e-6):
        raise RuntimeError("Figure 5 relative transform is not mean-centered")
    total = sum(float(row[column]) for column in (
        "enhance_fraction", "suppress_fraction", "neutral_fraction"
    ))
    if not np.isclose(total, 1.0):
        raise RuntimeError("Figure 5 state fractions do not sum to one")
    return parameter_count, calibrated.shape


def verify_text_safety(root: Path) -> None:
    private_markers = (
        "/workspace/" + "RBCM-Edge",
        "D:\\" + "study\\project\\RBCM-Edge",
    )
    for path in root.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in private_markers):
                raise RuntimeError(f"Machine-specific path leaked into {path}")
            if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text):
                raise RuntimeError(f"Drive-specific path leaked into {path}")


def main() -> None:
    root = parse_args().code_root.resolve()
    verify_membership(root)
    verify_v5_configs(root)
    parameter_count, output_shape = verify_v5_runtime(root)
    verify_text_safety(root)
    manifest_files = verify_manifest(root)
    print(
        "release_verify=PASS "
        f"manifest_files={manifest_files} "
        f"hed_lite_parameters={parameter_count} "
        f"calibration_shape={output_shape} "
        "v5_figure5_definition=mean_mad_relative "
        "generated_results_in_repository=0"
    )


if __name__ == "__main__":
    main()
