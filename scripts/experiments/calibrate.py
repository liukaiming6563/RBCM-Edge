"""Search validation-selected center-surround logit calibration on BIPED.

This is a fast diagnostic/HPO script for the HED-lite path.  It does not train
or change a checkpoint.  Instead, it treats an existing HED-lite checkpoint as
the center edge evidence and searches compact surround-to-center logit
calibration parameters on the BIPED train-tail validation split.  The selected
parameters are then evaluated once on the BIPED test split.

The goal is to determine whether a clean, validation-selected annular surround
calibration can produce a materially larger gap than the learned plugin-style
RBCM variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_model.core.checkpoint_io import load_checkpoint
from edge_model.core.config import deep_update, load_config, project_path
from edge_model.data.build import make_dataset, make_loader
from edge_model.engine.train_loop import load_original_target_array, restore_sample_array
from edge_model.engine.visualize import save_probability_map
from edge_model.models.build import build_model
from scripts.baselines.evaluate_official_edges import (
    average_precision_from_curve,
    dilation_metrics_from_arrays,
    maybe_nms,
)


@dataclass(frozen=True)
class Candidate:
    mode: str
    ring: str
    alpha: float
    edge_weight: float
    prob_weight: float
    uncertainty_power: float
    temperature: float
    bias: float
    sharpen: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--thresholds", type=int, default=49)
    parser.add_argument("--coarse-thresholds", type=int, default=21)
    parser.add_argument("--coarse-max-side", type=int, default=192)
    parser.add_argument("--match-tolerance", type=float, default=0.0075)
    parser.add_argument(
        "--gt-threshold",
        type=float,
        default=0.0,
        help="Threshold applied to soft ground truth by the metric (for example 0.3 for MultiCue).",
    )
    parser.add_argument(
        "--eval-gt-variant",
        default=None,
        help="Override dataset.gt.eval_variant, for example soft_vote.",
    )
    parser.add_argument(
        "--eval-gt-mode",
        choices=["binary", "soft"],
        default=None,
        help="Override dataset.gt.eval_mode.",
    )
    parser.add_argument("--nms-low-threshold", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-val-candidates", type=int, default=2000)
    parser.add_argument("--search-seed", type=int, default=4517)
    parser.add_argument("--modes", nargs="+", default=["main_surround", "no_surround", "conv_control"])
    parser.add_argument(
        "--dataset-name",
        default="auto",
        help=(
            "Dataset to collect val/test samples from. Use 'auto' to infer from "
            "the training config's val_dataset/train_dataset."
        ),
    )
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--grid-profile",
        choices=["default", "focused_main", "focused_main_strong", "focused_main_refine"],
        default="default",
    )
    parser.add_argument("--save-test-predictions", action="store_true")
    return parser.parse_args()


def infer_dataset_name(config: dict, requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    dataset_config = config.get("dataset", {})
    for key in ("val_dataset", "eval_dataset", "train_dataset"):
        value = dataset_config.get(key)
        if value:
            return str(value)
    raise ValueError("Could not infer dataset name from config; pass --dataset-name explicitly.")


def load_config_for_eval(config_path: Path, checkpoint: dict, args: argparse.Namespace) -> dict:
    base = load_config(config_path)
    checkpoint_config = checkpoint.get("config", base)
    merged = deep_update(
        checkpoint_config,
        {
            "paths": {
                "project_root": str(PROJECT_ROOT),
                "edge_data_root": checkpoint_config.get("paths", {}).get("edge_data_root", "edge_data"),
            },
            "dataset": {
                "input_size": checkpoint_config.get("dataset", {}).get("input_size", 300),
                "preserve_aspect_eval": True,
                "gt": {
                    "eval_variant": checkpoint_config.get("dataset", {}).get("gt", {}).get("eval_variant", "edge"),
                    "eval_mode": checkpoint_config.get("dataset", {}).get("gt", {}).get("eval_mode", "soft"),
                    "binarize_eval_edges": False,
                },
            },
            "loader": {
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "pin_memory": True,
                "persistent_workers": int(args.num_workers) > 0,
            },
        },
    )
    gt_config = merged.setdefault("dataset", {}).setdefault("gt", {})
    if getattr(args, "eval_gt_variant", None):
        gt_config["eval_variant"] = str(args.eval_gt_variant)
    if getattr(args, "eval_gt_mode", None):
        gt_config["eval_mode"] = str(args.eval_gt_mode)
        gt_config["binarize_eval_edges"] = str(args.eval_gt_mode) == "binary"
    return merged


def edge_energy_from_image(path: str | Path) -> np.ndarray:
    """Return deterministic 3x3 Sobel energy for an RGB input image.

    This implementation intentionally does not change with optional OpenCV
    availability.  The previous fallback used ``numpy.gradient``, which is a
    different operator and made fixed calibration candidates environment
    dependent.
    """
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    padded = np.pad(arr, 1, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    kernel_x = np.asarray(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    kernel_y = kernel_x.T
    gx = np.einsum("ijkl,kl->ij", windows, kernel_x, optimize=True)
    gy = np.einsum("ijkl,kl->ij", windows, kernel_y, optimize=True)
    energy = np.sqrt(gx * gx + gy * gy + 1.0e-8)
    mean = float(energy.mean())
    std = float(energy.std())
    return np.clip(energy / (mean + 2.0 * std + 1.0e-6), 0.0, 3.0).astype(np.float32) / 3.0


def blur(arr: np.ndarray, kernel: int) -> np.ndarray:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    try:
        import cv2

        return cv2.blur(arr.astype(np.float32), (kernel, kernel), borderType=cv2.BORDER_REFLECT_101)
    except Exception:
        pad = kernel // 2
        padded = np.pad(arr.astype(np.float32), pad, mode="reflect")
        out = np.empty_like(arr, dtype=np.float32)
        area = float(kernel * kernel)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                out[y, x] = padded[y : y + kernel, x : x + kernel].sum() / area
        return out


def ring_mean(arr: np.ndarray, inner: int, outer: int) -> np.ndarray:
    if inner < 1 or outer <= inner:
        raise ValueError(f"Invalid annulus: inner={inner}, outer={outer}")
    outer_mean = blur(arr, outer)
    inner_mean = blur(arr, inner)
    outer_area = float(outer * outer)
    inner_area = float(inner * inner)
    return ((outer_mean * outer_area) - (inner_mean * inner_area)) / max(1.0, outer_area - inner_area)


def local_surround(arr: np.ndarray, mode: str, ring: tuple[int, int, int, int]) -> np.ndarray:
    inner1, outer1, inner2, outer2 = ring
    if mode == "main_surround":
        near = ring_mean(arr, inner1, outer1)
        far = ring_mean(arr, inner2, outer2)
        return 0.62 * near + 0.38 * far
    if mode == "no_surround":
        near = blur(arr, 3)
        mid = blur(arr, 5)
        return 0.68 * arr + 0.24 * near + 0.08 * mid
    if mode == "conv_control":
        near = blur(arr, 3)
        mid = blur(near, 3)
        far = blur(mid, 3)
        return 0.45 * near + 0.35 * mid + 0.20 * far
    raise ValueError(f"Unknown mode: {mode}")


def contrast(center: np.ndarray, surround: np.ndarray) -> np.ndarray:
    return np.clip((center - surround) / (center + surround + 0.06), -1.0, 1.0).astype(np.float32)


def logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(np.float32), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def resize_float_array(arr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = arr.shape
    max_side = max(16, int(max_side))
    current = max(height, width)
    if current <= max_side:
        return arr.astype(np.float32)
    scale = max_side / float(current)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = Image.fromarray(np.clip(arr, 0.0, 1.0).astype(np.float32))
    return np.asarray(
        image.resize((new_width, new_height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).clip(0.0, 1.0)


def coarse_sample(sample: dict[str, np.ndarray | str], max_side: int) -> dict[str, np.ndarray | str]:
    key = f"_coarse_{max_side}"
    cached = sample.get(key)
    if isinstance(cached, dict):
        return cached  # type: ignore[return-value]
    small = {
        "sample_id": str(sample["sample_id"]),
        "prob": resize_float_array(sample["prob"], max_side),  # type: ignore[arg-type]
        "target": resize_float_array(sample["target"], max_side),  # type: ignore[arg-type]
        "edge": resize_float_array(sample["edge"], max_side),  # type: ignore[arg-type]
    }
    sample[key] = small  # type: ignore[assignment]
    return small


def clear_candidate_caches(samples: list[dict[str, np.ndarray | str]]) -> None:
    """Release mode-specific calibration arrays while retaining source samples."""

    cache_prefixes = ("_state_",)
    cache_keys = {"_uncertainty", "_logit", "_local_diff"}
    for sample in samples:
        nested_samples = [
            value
            for key, value in sample.items()
            if key.startswith("_coarse_") and isinstance(value, dict)
        ]
        for target in [sample, *nested_samples]:
            for key in list(target):
                if key in cache_keys or key.startswith(cache_prefixes):
                    target.pop(key, None)


def apply_candidate(sample: dict[str, np.ndarray | str], candidate: Candidate) -> np.ndarray:
    prob = sample["prob"]  # type: ignore[assignment]
    edge = sample["edge"]  # type: ignore[assignment]
    assert isinstance(prob, np.ndarray)
    assert isinstance(edge, np.ndarray)
    prob_state, edge_state = cached_states(sample, candidate)
    state = candidate.edge_weight * edge_state + candidate.prob_weight * prob_state
    state = np.clip(state, -1.0, 1.0)
    uncertainty = sample.get("_uncertainty")
    if not isinstance(uncertainty, np.ndarray):
        uncertainty = np.clip(4.0 * prob * (1.0 - prob), 0.0, 1.0)
        sample["_uncertainty"] = uncertainty
    if candidate.uncertainty_power != 1.0:
        uncertainty = np.power(np.clip(uncertainty, 0.0, 1.0), candidate.uncertainty_power)
    base = sample.get("_logit")
    if not isinstance(base, np.ndarray):
        base = logit(prob)
        sample["_logit"] = base
    base_logit = base * candidate.temperature + candidate.bias
    if candidate.sharpen != 0.0:
        local_diff = sample.get("_local_diff")
        if not isinstance(local_diff, np.ndarray):
            local_diff = prob - blur(prob, 3)
            sample["_local_diff"] = local_diff
        base_logit = base_logit + candidate.sharpen * local_diff
    return sigmoid(base_logit + candidate.alpha * uncertainty * state)


def cached_states(sample: dict[str, np.ndarray | str], candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    key = f"_state_{candidate.mode}_{candidate.ring}"
    cached = sample.get(key)
    if isinstance(cached, tuple):
        return cached  # type: ignore[return-value]
    prob = sample["prob"]
    edge = sample["edge"]
    assert isinstance(prob, np.ndarray)
    assert isinstance(edge, np.ndarray)
    rings = parse_ring(candidate.ring)
    prob_surround = local_surround(prob, candidate.mode, rings)
    edge_surround = local_surround(edge, candidate.mode, rings)
    prob_state = contrast(prob, prob_surround)
    edge_state = contrast(edge, edge_surround)
    sample[key] = (prob_state, edge_state)  # type: ignore[assignment]
    return prob_state, edge_state


def parse_ring(text: str) -> tuple[int, int, int, int]:
    values = [int(item) for item in text.split("-")]
    if len(values) != 4:
        raise ValueError(f"Ring must have four dash-separated integers: {text}")
    inner1, outer1, inner2, outer2 = values
    if any(value < 1 or value % 2 == 0 for value in values):
        raise ValueError(f"Ring sizes must be positive odd integers: {text}")
    if not (inner1 < outer1 <= inner2 < outer2):
        raise ValueError(
            "Ring sizes must satisfy inner1 < outer1 <= inner2 < outer2; "
            f"got {text}"
        )
    return values[0], values[1], values[2], values[3]


@torch.no_grad()
def collect_samples(
    *,
    model: torch.nn.Module,
    config: dict,
    dataset_name: str,
    split: str,
    device: torch.device,
) -> list[dict[str, np.ndarray | str]]:
    dataset = make_dataset(config, dataset_name=dataset_name, split=split, training=False)
    loader = make_loader(dataset, config, shuffle=False)
    model.eval()
    rows: list[dict[str, np.ndarray | str]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.sigmoid(outputs["logits"]).float().detach().cpu().numpy()
        targets = batch["edge"].detach().cpu().numpy()
        for idx in range(probs.shape[0]):
            prob = restore_sample_array(probs[idx, 0], batch, idx, is_target=False)
            truth = load_original_target_array(batch, idx)
            if truth is None:
                truth = restore_sample_array(targets[idx, 0], batch, idx, is_target=True)
            sample_id = str(batch["sample_id"][idx])
            image_path = str(batch["image_path"][idx])
            edge = edge_energy_from_image(image_path)
            if edge.shape != prob.shape:
                edge = np.asarray(
                    Image.fromarray((edge * 255.0).astype(np.uint8)).resize(
                        (prob.shape[1], prob.shape[0]),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                ) / 255.0
            rows.append({"sample_id": sample_id, "prob": prob, "target": truth, "edge": edge})
    return rows


def metric_for_probs(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    thresholds: int,
    match_tolerance: float,
    nms_low_threshold: float,
    apply_nms: bool,
    device: str,
    gt_threshold: float = 0.0,
) -> dict[str, float]:
    probs = probabilities
    if apply_nms:
        probs = maybe_nms(probs, device_name=device, low_threshold=nms_low_threshold)
    threshold_values = np.linspace(0.01, 0.99, max(2, int(thresholds)))
    metrics, _ = dilation_metrics_from_arrays(
        probs,
        targets,
        threshold_values,
        match_tolerance=match_tolerance,
        gt_threshold=float(gt_threshold),
    )
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (float, int))}


def proxy_metric_for_probs(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    thresholds: int,
    gt_threshold: float = 0.0,
) -> dict[str, float]:
    """Fast coarse proxy used only to rank many validation candidates.

    The refined stage still uses the same dilation/NMS metric as the near-official
    evaluator.  This proxy avoids per-threshold prediction dilation, which is the
    expensive part when evaluating thousands of candidates.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("cv2 is required for the coarse proxy metric") from exc
    threshold_values = np.linspace(0.03, 0.97, max(2, int(thresholds)))
    rows = []
    per_image_best = []
    for threshold in threshold_values:
        rows.append({"threshold": float(threshold), "mp": 0.0, "pc": 0.0, "mt": 0.0, "tc": 0.0})
    for prob, target in zip(probabilities, targets):
        gt = (target > float(gt_threshold)).astype(np.uint8)
        if gt.sum() == 0:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gt_dil = cv2.dilate(gt, kernel) > 0
        image_best = 0.0
        truth_count = float(gt.sum())
        for index, threshold in enumerate(threshold_values):
            pred = prob >= float(threshold)
            pred_count = float(pred.sum())
            matched_pred = float((pred & gt_dil).sum()) if pred_count > 0.0 else 0.0
            # Strict matched truth is a cheap recall proxy.  Full dilation recall
            # is applied in the refined top-k stage.
            matched_truth = float((pred & (gt > 0)).sum()) if pred_count > 0.0 else 0.0
            row = rows[index]
            row["mp"] += matched_pred
            row["pc"] += pred_count
            row["mt"] += matched_truth
            row["tc"] += truth_count
            precision = matched_pred / max(pred_count, 1.0)
            recall = matched_truth / max(truth_count, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            image_best = max(image_best, f1)
        per_image_best.append(image_best)
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_threshold = 0.0
    curve_precision = []
    curve_recall = []
    for row in rows:
        precision = row["mp"] / max(row["pc"], 1.0)
        recall = row["mt"] / max(row["tc"], 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        curve_precision.append(precision)
        curve_recall.append(recall)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_precision = float(precision)
            best_recall = float(recall)
            best_threshold = float(row["threshold"])
    ap = average_precision_from_curve(
        [
            {"recall": recall, "precision": precision}
            for recall, precision in zip(curve_recall, curve_precision)
        ]
    )
    return {
        "ODS": best_f1,
        "OIS": float(np.mean(per_image_best)) if per_image_best else 0.0,
        "AP": ap,
        "ODS_threshold": best_threshold,
        "precision_at_ODS": best_precision,
        "recall_at_ODS": best_recall,
    }


def candidate_grid(mode: str, profile: str = "default") -> Iterable[Candidate]:
    if profile == "focused_main_refine" and mode == "main_surround":
        rings = ["1-7-7-17", "3-7-7-17"]
        alphas = [-4.20, -4.00, -3.80, -3.60, -3.40, -3.20]
        uncertainty_powers = [0.15, 0.20, 0.25, 0.30]
        temperatures = [0.96, 1.00, 1.04]
        biases = [-0.22, -0.20, -0.18, -0.16]
        sharpens = [-1.50, -1.25, -1.00, -0.75]
        weights = [(0.95, 0.05), (0.90, 0.10), (0.85, 0.15)]
    elif profile == "focused_main_strong" and mode == "main_surround":
        rings = ["1-7-7-17", "3-7-7-17", "3-9-9-21", "5-9-9-21"]
        alphas = [-4.20, -3.60, -3.00, -2.60, -2.20, -1.80]
        uncertainty_powers = [0.20, 0.35, 0.50, 0.70]
        temperatures = [0.84, 0.92, 1.00, 1.08]
        biases = [-0.24, -0.18, -0.12, -0.06, 0.0]
        sharpens = [-1.25, -1.00, -0.75, -0.50, -0.25]
        weights = [(1.00, 0.00), (0.90, 0.10), (0.80, 0.20)]
    elif profile == "focused_main" and mode == "main_surround":
        rings = ["1-5-5-13", "1-7-7-17", "1-9-9-21", "3-7-7-17", "3-9-9-21"]
        alphas = [-2.40, -2.00, -1.60, -1.20, -0.90, -0.60]
        uncertainty_powers = [0.35, 0.55, 0.75, 1.00]
        temperatures = [0.92, 1.00, 1.08, 1.16]
        biases = [-0.18, -0.12, -0.08, -0.04, 0.0]
        sharpens = [-0.75, -0.55, -0.35, -0.15, 0.0]
        weights = [(1.00, 0.00), (0.90, 0.10), (0.75, 0.25)]
    else:
        rings = ["1-5-5-15", "1-7-7-17", "3-9-9-21", "5-11-11-25"]
        alphas = [-1.20, -0.90, -0.60, -0.35, 0.35, 0.60, 0.90, 1.20]
        uncertainty_powers = [0.55, 0.85, 1.15, 1.60]
        temperatures = [0.88, 1.00, 1.12]
        biases = [-0.08, 0.0, 0.08]
        sharpens = [-0.35, 0.0, 0.35]
        weights = [(0.80, 0.20), (0.60, 0.40), (0.40, 0.60), (1.00, 0.00)]
    for ring in rings:
        for alpha in alphas:
            for edge_weight, prob_weight in weights:
                for uncertainty_power in uncertainty_powers:
                    for temperature in temperatures:
                        for bias in biases:
                            for sharpen in sharpens:
                                yield Candidate(
                                    mode=mode,
                                    ring=ring,
                                    alpha=alpha,
                                    edge_weight=edge_weight,
                                    prob_weight=prob_weight,
                                    uncertainty_power=uncertainty_power,
                                    temperature=temperature,
                                    bias=bias,
                                    sharpen=sharpen,
                                )


def evaluate_candidate(
    samples: list[dict[str, np.ndarray | str]],
    candidate: Candidate,
    args: argparse.Namespace,
    *,
    apply_nms: bool,
) -> dict[str, float | str]:
    probs = [apply_candidate(sample, candidate) for sample in samples]
    targets = [sample["target"] for sample in samples]
    metrics = metric_for_probs(
        probs,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        thresholds=args.thresholds,
        match_tolerance=args.match_tolerance,
        nms_low_threshold=args.nms_low_threshold,
        apply_nms=apply_nms,
        device=args.device,
        gt_threshold=args.gt_threshold,
    )
    return {**asdict(candidate), **metrics}


def evaluate_candidate_proxy(
    samples: list[dict[str, np.ndarray | str]],
    candidate: Candidate,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    small_samples = [coarse_sample(sample, int(args.coarse_max_side)) for sample in samples]
    probs = [apply_candidate(sample, candidate) for sample in small_samples]
    targets = [sample["target"] for sample in small_samples]
    metrics = proxy_metric_for_probs(
        probs,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        thresholds=args.coarse_thresholds,
        gt_threshold=args.gt_threshold,
    )
    return {**asdict(candidate), **metrics}


def search_mode(
    samples: list[dict[str, np.ndarray | str]],
    mode: str,
    args: argparse.Namespace,
) -> tuple[Candidate, dict[str, float | str], list[dict[str, float | str]]]:
    coarse_rows: list[dict[str, float | str]] = []
    candidates = list(candidate_grid(mode, profile=str(args.grid_profile)))
    rng = random.Random(int(args.search_seed) + sum(ord(ch) for ch in mode))
    rng.shuffle(candidates)
    for index, candidate in enumerate(candidates):
        if index >= int(args.max_val_candidates):
            break
        row = evaluate_candidate_proxy(samples, candidate, args)
        coarse_rows.append(row)
    coarse_rows.sort(key=lambda row: (float(row["ODS"]), float(row["AP"])), reverse=True)
    top_rows = coarse_rows[: max(1, int(args.top_k))]
    refined_rows: list[dict[str, float | str]] = []
    for row in top_rows:
        candidate = Candidate(
            mode=str(row["mode"]),
            ring=str(row["ring"]),
            alpha=float(row["alpha"]),
            edge_weight=float(row["edge_weight"]),
            prob_weight=float(row["prob_weight"]),
            uncertainty_power=float(row["uncertainty_power"]),
            temperature=float(row["temperature"]),
            bias=float(row["bias"]),
            sharpen=float(row["sharpen"]),
        )
        refined_rows.append(evaluate_candidate(samples, candidate, args, apply_nms=True))
    refined_rows.sort(key=lambda row: (float(row["ODS"]), float(row["AP"])), reverse=True)
    best_row = refined_rows[0]
    best = Candidate(
        mode=str(best_row["mode"]),
        ring=str(best_row["ring"]),
        alpha=float(best_row["alpha"]),
        edge_weight=float(best_row["edge_weight"]),
        prob_weight=float(best_row["prob_weight"]),
        uncertainty_power=float(best_row["uncertainty_power"]),
        temperature=float(best_row["temperature"]),
        bias=float(best_row["bias"]),
        sharpen=float(best_row["sharpen"]),
    )
    return best, best_row, refined_rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    samples: list[dict[str, np.ndarray | str]],
    candidate: Candidate,
    pred_dir: Path,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample_id = str(sample["sample_id"])
        prob = apply_candidate(sample, candidate)
        save_probability_map(prob, pred_dir / f"{sample_id}.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = load_config_for_eval(args.config, checkpoint, args)
    dataset_name = infer_dataset_name(config, args.dataset_name)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    val_samples = collect_samples(
        model=model,
        config=config,
        dataset_name=dataset_name,
        split=args.val_split,
        device=device,
    )
    print(f"collected val samples: {len(val_samples)}", flush=True)
    test_samples = collect_samples(
        model=model,
        config=config,
        dataset_name=dataset_name,
        split=args.test_split,
        device=device,
    )
    print(f"collected test samples: {len(test_samples)}", flush=True)

    baseline_val = metric_for_probs(
        [sample["prob"] for sample in val_samples],  # type: ignore[list-item]
        [sample["target"] for sample in val_samples],  # type: ignore[list-item]
        thresholds=args.thresholds,
        match_tolerance=args.match_tolerance,
        nms_low_threshold=args.nms_low_threshold,
        apply_nms=True,
        device=args.device,
        gt_threshold=args.gt_threshold,
    )
    baseline_test = metric_for_probs(
        [sample["prob"] for sample in test_samples],  # type: ignore[list-item]
        [sample["target"] for sample in test_samples],  # type: ignore[list-item]
        thresholds=args.thresholds,
        match_tolerance=args.match_tolerance,
        nms_low_threshold=args.nms_low_threshold,
        apply_nms=True,
        device=args.device,
        gt_threshold=args.gt_threshold,
    )

    summary_rows: list[dict[str, float | str]] = [
        {"mode": "plain_identity", "split": "val", **baseline_val},
        {"mode": "plain_identity", "split": "test", **baseline_test},
    ]
    selected: dict[str, dict[str, object]] = {}
    for mode in list(args.modes):
        print(f"searching {mode}", flush=True)
        best, best_val, refined = search_mode(val_samples, mode, args)
        write_csv(args.output_dir / f"{mode}_val_top_refined.csv", refined)
        # Full-resolution ring states can occupy several gigabytes.  They are
        # no longer needed after validation has selected the fixed candidate.
        clear_candidate_caches(val_samples)
        test_row = evaluate_candidate(test_samples, best, args, apply_nms=True)
        selected[mode] = {"candidate": asdict(best), "val": best_val, "test": test_row}
        summary_rows.append({"mode": mode, "split": "val", **best_val})
        summary_rows.append({"mode": mode, "split": "test", **test_row})
        print(
            f"{mode}: val ODS={float(best_val['ODS']):.4f}, "
            f"test ODS={float(test_row['ODS']):.4f}, AP={float(test_row['AP']):.4f}",
            flush=True,
        )
        if args.save_test_predictions:
            save_predictions(test_samples, best, args.output_dir / "predictions" / mode)
        clear_candidate_caches(test_samples)

    write_csv(args.output_dir / "summary.csv", summary_rows)
    report = {
        "dataset_name": dataset_name,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "eval_gt_variant": config.get("dataset", {}).get("gt", {}).get("eval_variant"),
        "eval_gt_mode": config.get("dataset", {}).get("gt", {}).get("eval_mode"),
        "gt_threshold": float(args.gt_threshold),
        "baseline": {"val": baseline_val, "test": baseline_test},
        "selected": selected,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
