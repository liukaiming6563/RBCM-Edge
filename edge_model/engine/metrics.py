"""Evaluation metrics for soft or binary edge probability maps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sigmoid_to_numpy(logits: torch.Tensor) -> np.ndarray:
    """Convert `[B, 1, H, W]` logits to NumPy probabilities."""
    return torch.sigmoid(logits).detach().cpu().numpy()


def nms_probabilities(probabilities: torch.Tensor, low_threshold: float = 0.0) -> torch.Tensor:
    """Thin edge probability maps with gradient-direction non-maximum suppression.

    This is a lightweight torch implementation for evaluation-time edge
    thinning. It follows the Canny/HED-style idea of keeping local maxima along
    the response-gradient direction, while remaining GPU-friendly for batched
    tensors. Official BSDS benchmark wrappers can still be used later for exact
    paper reproduction.
    """
    if probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("probabilities must have shape [B, 1, H, W]")
    prob = probabilities.float().clamp(0.0, 1.0)
    sobel_x = prob.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = prob.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(prob, sobel_x, padding=1)
    grad_y = F.conv2d(prob, sobel_y, padding=1)
    angle = torch.remainder(torch.rad2deg(torch.atan2(grad_y, grad_x)), 180.0)

    padded = F.pad(prob, (1, 1, 1, 1), mode="replicate")
    center = padded[:, :, 1:-1, 1:-1]
    east = padded[:, :, 1:-1, 2:]
    west = padded[:, :, 1:-1, :-2]
    north = padded[:, :, :-2, 1:-1]
    south = padded[:, :, 2:, 1:-1]
    north_east = padded[:, :, :-2, 2:]
    south_west = padded[:, :, 2:, :-2]
    north_west = padded[:, :, :-2, :-2]
    south_east = padded[:, :, 2:, 2:]

    bin_0 = (angle < 22.5) | (angle >= 157.5)
    bin_45 = (angle >= 22.5) & (angle < 67.5)
    bin_90 = (angle >= 67.5) & (angle < 112.5)
    bin_135 = (angle >= 112.5) & (angle < 157.5)
    keep = (
        (bin_0 & (center >= east) & (center >= west))
        | (bin_45 & (center >= north_east) & (center >= south_west))
        | (bin_90 & (center >= north) & (center >= south))
        | (bin_135 & (center >= north_west) & (center >= south_east))
    )
    if low_threshold > 0.0:
        keep = keep & (center >= float(low_threshold))
    return (center * keep.to(dtype=center.dtype)).to(dtype=probabilities.dtype)


class FastEdgeMetricAccumulator:
    """GPU-friendly approximate edge metrics for training-time validation.

    The strict metric below restores every prediction to original image size and
    uses SciPy KD-tree matching for each threshold. That is useful for final
    reporting but slow inside training. This accumulator keeps validation on the
    active torch device and uses dilation-based tolerant matching. It is intended
    for checkpoint selection and debugging, not as the paper's final benchmark.
    """

    def __init__(
        self,
        thresholds: int | list[float] | np.ndarray | torch.Tensor | None = None,
        tolerance_pixels: int = 4,
    ) -> None:
        if thresholds is None:
            values = torch.linspace(0.01, 0.99, 49)
        elif isinstance(thresholds, int):
            values = torch.linspace(0.01, 0.99, max(2, int(thresholds)))
        elif isinstance(thresholds, torch.Tensor):
            values = thresholds.detach().float().cpu()
        else:
            values = torch.as_tensor(thresholds, dtype=torch.float32)

        self.threshold_values = values.detach().float().clone()
        self.thresholds = [float(value) for value in self.threshold_values.tolist()]
        self.tolerance_pixels = max(0, int(round(tolerance_pixels)))
        n = len(self.thresholds)
        self.matched_pred = torch.zeros(n, dtype=torch.float64)
        self.pred_count = torch.zeros(n, dtype=torch.float64)
        self.matched_truth = torch.zeros(n, dtype=torch.float64)
        self.truth_count = torch.zeros(n, dtype=torch.float64)
        self.per_image_best: list[float] = []
        self.pred_prob_sum = 0.0
        self.pred_edge_count = 0.0
        self.target_edge_count = 0.0
        self.valid_count = 0.0

    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one validation batch.

        Args:
            probabilities: `[B, 1, H, W]` sigmoid probabilities.
            targets: `[B, 1, H, W]` binary or continuous soft-vote edge targets.
            valid_mask: Optional `[B, 1, H, W]` mask that excludes letterbox
                padding from density and metric counts.
        """
        prob = probabilities.detach().float().clamp(0.0, 1.0)
        truth = targets.detach().float().clamp(0.0, 1.0)
        if valid_mask is None:
            mask = torch.ones_like(truth, dtype=torch.bool)
        else:
            mask = valid_mask.to(device=prob.device, dtype=torch.bool)
        mask_float = mask.float()
        truth = truth * mask_float

        valid_count = float(mask.sum().detach().cpu())
        self.valid_count += valid_count
        self.pred_prob_sum += float((prob * mask_float).sum().detach().cpu())
        pred_at_half = (prob >= 0.5) & mask
        self.pred_edge_count += float(pred_at_half.sum().detach().cpu())
        self.target_edge_count += float(truth.sum().detach().cpu())

        thresholds = self.threshold_values.to(device=prob.device, dtype=prob.dtype).view(-1, 1, 1, 1, 1)
        pred = (prob.unsqueeze(0) >= thresholds) & mask.unsqueeze(0)
        truth_expanded = truth.unsqueeze(0).expand_as(pred).float()
        if self.tolerance_pixels > 0:
            truth_neighborhood = _dilate_soft(truth, self.tolerance_pixels).unsqueeze(0).expand_as(pred)
            pred_neighborhood = _dilate_binary(
                pred.reshape(-1, *pred.shape[2:]),
                self.tolerance_pixels,
            ).view_as(pred)
        else:
            truth_neighborhood = truth_expanded
            pred_neighborhood = pred

        pred_float = pred.float()
        matched_pred = (pred_float * truth_neighborhood.float()).sum(dim=(2, 3, 4)).float()
        pred_count = pred_float.sum(dim=(2, 3, 4)).float()
        matched_truth = (truth_expanded * pred_neighborhood.float()).sum(dim=(2, 3, 4)).float()
        truth_count = truth_expanded.sum(dim=(2, 3, 4)).float()

        self.matched_pred += matched_pred.sum(dim=1).detach().cpu().double()
        self.pred_count += pred_count.sum(dim=1).detach().cpu().double()
        self.matched_truth += matched_truth.sum(dim=1).detach().cpu().double()
        self.truth_count += truth_count.sum(dim=1).detach().cpu().double()

        truth_per_image = truth.sum(dim=(1, 2, 3)).float()
        precision = matched_pred / pred_count.clamp_min(1.0)
        recall = matched_truth / truth_per_image.view(1, -1).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        batch_best = f1.max(dim=0).values
        self.per_image_best.extend(float(value) for value in batch_best.detach().cpu().tolist())

    def compute(self) -> dict[str, float]:
        """Return approximate ODS/OIS/AP and density diagnostics."""
        rows: list[dict[str, float]] = []
        for index, threshold in enumerate(self.thresholds):
            precision = float(self.matched_pred[index] / max(float(self.pred_count[index]), 1.0))
            recall = float(self.matched_truth[index] / max(float(self.truth_count[index]), 1.0))
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            rows.append(
                {
                    "threshold": threshold,
                    "threshold_index": float(index),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "tp": float(min(float(self.matched_pred[index]), float(self.matched_truth[index]))),
                    "fp": float(max(float(self.pred_count[index] - self.matched_pred[index]), 0.0)),
                    "fn": float(max(float(self.truth_count[index] - self.matched_truth[index]), 0.0)),
                }
            )

        best_row = max(rows, key=lambda row: row["f1"]) if rows else {}
        pred_edge_density = self.pred_edge_count / max(self.valid_count, 1.0)
        best_index = int(best_row.get("threshold_index", 0.0)) if rows else 0
        pred_edge_density_at_ods = float(self.pred_count[best_index] / max(self.valid_count, 1.0)) if rows else 0.0
        target_edge_density = self.target_edge_count / max(self.valid_count, 1.0)
        return {
            "ODS": float(best_row.get("f1", 0.0)),
            "OIS": float(np.mean(self.per_image_best)) if self.per_image_best else 0.0,
            "AP": _average_precision_from_curve(rows),
            "ODS_threshold": float(best_row.get("threshold", 0.0)),
            "precision_at_ODS": float(best_row.get("precision", 0.0)),
            "recall_at_ODS": float(best_row.get("recall", 0.0)),
            "pred_prob_density": self.pred_prob_sum / max(self.valid_count, 1.0),
            "pred_edge_density": pred_edge_density,
            "pred_edge_density_at_ODS": pred_edge_density_at_ods,
            "target_edge_density": target_edge_density,
            "density_ratio": pred_edge_density / max(target_edge_density, 1e-6),
            "density_ratio_at_ODS": pred_edge_density_at_ods / max(target_edge_density, 1e-6),
        }


def edge_metrics_from_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    thresholds: np.ndarray | None = None,
    match_tolerance: float | None = 0.0075,
) -> dict[str, float]:
    """Compute ODS, OIS, and AP for edge maps.

    By default this uses BSDS-style tolerant boundary matching: a predicted edge
    pixel counts as correct when it falls within `match_tolerance` times the
    image diagonal of a ground-truth edge pixel. Set `match_tolerance=None` for
    the older exact-pixel development metric.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    if not probabilities:
        return {
            "ODS": 0.0,
            "OIS": 0.0,
            "AP": 0.0,
            "ODS_threshold": 0.0,
            "precision_at_ODS": 0.0,
            "recall_at_ODS": 0.0,
        }

    if match_tolerance is None:
        flat_prob, flat_target = _flatten_arrays(probabilities, targets)
        curve = [
            {"threshold": float(threshold), **_precision_recall_f1(flat_prob, flat_target, threshold)}
            for threshold in thresholds
        ]
        best_row = max(curve, key=lambda row: float(row["f1"])) if curve else {}
        ods = float(best_row.get("f1", 0.0))
        ap = 0.0 if flat_target.max() == flat_target.min() else float(_average_precision(flat_target, flat_prob))
    else:
        curve = threshold_curve_from_arrays(probabilities, targets, thresholds, match_tolerance=match_tolerance)
        best_row = max(curve, key=lambda row: float(row["f1"])) if curve else {}
        ods = float(best_row.get("f1", 0.0))
        ap = _average_precision_from_curve(curve)

    per_image_best = []
    for prob, target in zip(probabilities, targets):
        if match_tolerance is None:
            prob_flat = prob.reshape(-1)
            target_flat = np.asarray(target, dtype=np.float32).reshape(-1).clip(0.0, 1.0)
            best = max(_f1_at_threshold(prob_flat, target_flat, threshold) for threshold in thresholds)
        else:
            best = max(
                _precision_recall_f1_image(prob, target, threshold, match_tolerance)["f1"]
                for threshold in thresholds
            )
        per_image_best.append(best)
    ois = float(np.mean(per_image_best)) if per_image_best else 0.0
    density_stats = _density_stats_at_threshold(
        probabilities,
        targets,
        threshold=float(best_row.get("threshold", 0.0)),
    )

    return {
        "ODS": float(ods),
        "OIS": float(ois),
        "AP": ap,
        "ODS_threshold": float(best_row.get("threshold", 0.0)),
        "precision_at_ODS": float(best_row.get("precision", 0.0)),
        "recall_at_ODS": float(best_row.get("recall", 0.0)),
        **density_stats,
    }


def _fast_tolerant_counts(
    pred: torch.Tensor,
    truth: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate tolerant boundary counts per image on a torch device."""
    pred = pred.bool()
    truth = truth.bool()
    if radius > 0:
        truth_neighborhood = _dilate_binary(truth, radius)
        pred_neighborhood = _dilate_binary(pred, radius)
    else:
        truth_neighborhood = truth
        pred_neighborhood = pred

    matched_pred = (pred & truth_neighborhood).sum(dim=(1, 2, 3)).float()
    pred_count = pred.sum(dim=(1, 2, 3)).float()
    matched_truth = (truth & pred_neighborhood).sum(dim=(1, 2, 3)).float()
    truth_count = truth.sum(dim=(1, 2, 3)).float()
    return matched_pred, pred_count, matched_truth, truth_count


def _dilate_binary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a `[B, 1, H, W]` binary mask by a square radius."""
    kernel_size = 2 * int(radius) + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=int(radius)) > 0


def _dilate_soft(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Max-dilate a `[B, 1, H, W]` soft target map by a square radius."""
    kernel_size = 2 * int(radius) + 1
    return F.max_pool2d(mask.float().clamp(0.0, 1.0), kernel_size=kernel_size, stride=1, padding=int(radius))


def threshold_curve_from_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    thresholds: np.ndarray | None = None,
    match_tolerance: float | None = 0.0075,
) -> list[dict[str, float]]:
    """Return precision/recall/F1 values across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    if not probabilities:
        return []

    if match_tolerance is not None:
        rows: list[dict[str, float]] = []
        for threshold in thresholds:
            counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            for prob, target in zip(probabilities, targets):
                row = _precision_recall_f1_image(prob, target, threshold, match_tolerance)
                counts["tp"] += row["tp"]
                counts["fp"] += row["fp"]
                counts["fn"] += row["fn"]
            precision, recall, f1 = _scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    **counts,
                }
            )
        return rows

    flat_prob, flat_target = _flatten_arrays(probabilities, targets)
    return [
        {"threshold": float(threshold), **_precision_recall_f1(flat_prob, flat_target, threshold)}
        for threshold in thresholds
    ]


def per_image_metrics_from_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    sample_ids: list[str] | None = None,
    thresholds: np.ndarray | None = None,
    match_tolerance: float | None = 0.0075,
) -> list[dict[str, float | str]]:
    """Return best-threshold metrics and density stats for each image."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    if sample_ids is None:
        sample_ids = [str(index) for index in range(len(probabilities))]

    rows: list[dict[str, float | str]] = []
    for sample_id, prob, target in zip(sample_ids, probabilities, targets):
        target_float = np.asarray(target, dtype=np.float32).clip(0.0, 1.0)
        target_flat = target_float.reshape(-1)
        if match_tolerance is None:
            prob_flat = prob.reshape(-1)
            candidates = [
                {"threshold": float(threshold), **_precision_recall_f1(prob_flat, target_flat, threshold)}
                for threshold in thresholds
            ]
        else:
            candidates = [
                {"threshold": float(threshold), **_precision_recall_f1_image(prob, target, threshold, match_tolerance)}
                for threshold in thresholds
            ]
        best = max(candidates, key=lambda row: float(row["f1"]))
        prob_flat = prob.reshape(-1)
        rows.append(
            {
                "sample_id": sample_id,
                "best_threshold": best["threshold"],
                "best_f1": best["f1"],
                "precision": best["precision"],
                "recall": best["recall"],
                "gt_density": float(target_flat.mean()),
                "gt_support_density": float((target_flat > 0).mean()),
                "pred_mean": float(prob_flat.mean()),
                "pred_std": float(prob_flat.std()),
            }
        )
    return rows


def probability_stats_from_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float,
) -> dict[str, float]:
    """Summarize prediction distribution and foreground density."""
    if not probabilities:
        return {}
    flat_prob, flat_target = _flatten_arrays(probabilities, targets)
    pred_binary = flat_prob >= threshold
    truth = flat_target.clip(0.0, 1.0)
    return {
        "threshold": float(threshold),
        "pred_mean": float(flat_prob.mean()),
        "pred_std": float(flat_prob.std()),
        "pred_min": float(flat_prob.min()),
        "pred_max": float(flat_prob.max()),
        "pred_density": float(pred_binary.mean()),
        "gt_density": float(truth.mean()),
        "gt_support_density": float((truth > 0).mean()),
        "density_ratio": float(pred_binary.mean() / max(truth.mean(), 1e-8)),
    }


def _density_stats_at_threshold(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float,
) -> dict[str, float]:
    """Return aggregate density diagnostics at a selected threshold."""
    if not probabilities:
        return {
            "pred_edge_density_at_ODS": 0.0,
            "target_edge_density": 0.0,
            "density_ratio_at_ODS": 0.0,
        }
    total_pixels = float(sum(np.asarray(prob).size for prob in probabilities))
    pred_count = float(sum((np.asarray(prob) >= float(threshold)).sum() for prob in probabilities))
    target_mass = float(
        sum(np.asarray(target, dtype=np.float32).clip(0.0, 1.0).sum() for target in targets)
    )
    pred_density = pred_count / max(total_pixels, 1.0)
    target_density = target_mass / max(total_pixels, 1.0)
    return {
        "pred_edge_density_at_ODS": pred_density,
        "target_edge_density": target_density,
        "density_ratio_at_ODS": pred_density / max(target_density, 1e-8),
    }


def _f1_at_threshold(prob: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """Compute binary F1 at one probability threshold."""
    return _precision_recall_f1(prob, target, threshold)["f1"]


def _precision_recall_f1(prob: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute weighted precision, recall, and F1 at one probability threshold."""
    pred = np.asarray(prob >= threshold, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32).clip(0.0, 1.0)
    tp = float((pred * truth).sum())
    fp = float((pred * (1.0 - truth)).sum())
    fn = float(((1.0 - pred) * truth).sum())
    precision, recall, f1 = _scores_from_counts(float(tp), float(fp), float(fn))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _precision_recall_f1_image(
    prob: np.ndarray,
    target: np.ndarray,
    threshold: float,
    match_tolerance: float,
) -> dict[str, float]:
    """Compute tolerant boundary precision/recall/F1 for one image."""
    pred = np.asarray(prob >= threshold, dtype=bool)
    truth = np.asarray(target, dtype=np.float32).clip(0.0, 1.0)
    tp, fp, fn = _soft_tolerant_boundary_counts(pred, truth, match_tolerance)
    precision, recall, f1 = _scores_from_counts(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _soft_tolerant_boundary_counts(pred: np.ndarray, truth: np.ndarray, match_tolerance: float) -> tuple[float, float, float]:
    """Return tolerant counts where GT contributes its soft vote weight.

    Candidate predictions are greedily matched to unique GT pixels by distance,
    so duplicate predictions around the same soft-vote edge remain false
    positives. A matched GT pixel contributes its soft-vote value as fractional
    true-positive mass. This keeps soft-vote targets continuous instead of
    collapsing them with a 0.5 threshold while retaining BSDS-style tolerance.
    """
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=np.float32).clip(0.0, 1.0)
    pred_count = float(pred.sum())
    truth_count = float(truth.sum())
    if pred_count == 0 and truth_count == 0:
        return 0.0, 0.0, 0.0
    if pred_count == 0:
        return 0.0, 0.0, truth_count
    if truth_count == 0:
        return 0.0, pred_count, 0.0

    pred_points = np.argwhere(pred)
    truth_mask = truth > 0.0
    truth_points = np.argwhere(truth_mask)
    truth_values = truth[truth_mask].astype(np.float64)
    tolerance_pixels = _tolerance_pixels(pred.shape, match_tolerance)
    distances, truth_indices = _nearest_truth_indices(pred_points, truth_points, tolerance_pixels)
    valid_pred = np.isfinite(distances) & (truth_indices < len(truth_points))
    if not valid_pred.any():
        return 0.0, pred_count, truth_count

    candidate_order = np.argsort(distances[valid_pred], kind="mergesort")
    ordered_truth_indices = truth_indices[valid_pred][candidate_order]

    used_truth: set[int] = set()
    tp = 0.0
    for truth_index in ordered_truth_indices:
        truth_index_int = int(truth_index)
        if truth_index_int in used_truth:
            continue
        used_truth.add(truth_index_int)
        tp += float(truth_values[truth_index_int])

    # Each predicted pixel contributes at most one unit of precision mass.  A
    # soft-vote GT match gives fractional true-positive credit, and duplicate
    # predictions around the same GT pixel remain false positives.
    fp = max(pred_count - tp, 0.0)
    fn = max(truth_count - tp, 0.0)
    return float(tp), float(fp), float(fn)


def _tolerant_boundary_counts(pred: np.ndarray, truth: np.ndarray, match_tolerance: float) -> tuple[float, float, float]:
    """Return approximate one-to-one tolerant boundary matching counts."""
    pred_count = float(pred.sum())
    truth_count = float(truth.sum())
    if pred_count == 0 and truth_count == 0:
        return 0.0, 0.0, 0.0
    if pred_count == 0:
        return 0.0, 0.0, truth_count
    if truth_count == 0:
        return 0.0, pred_count, 0.0

    pred_points = np.argwhere(pred)
    truth_points = np.argwhere(truth)
    tolerance_pixels = _tolerance_pixels(pred.shape, match_tolerance)
    distances, truth_indices = _nearest_truth_indices(pred_points, truth_points, tolerance_pixels)
    valid = np.isfinite(distances) & (truth_indices < len(truth_points))
    if not valid.any():
        return 0.0, pred_count, truth_count

    candidate_order = np.argsort(distances[valid], kind="mergesort")
    candidate_truth_indices = truth_indices[valid][candidate_order]
    used_truth: set[int] = set()
    matched = 0
    for truth_index in candidate_truth_indices:
        truth_index_int = int(truth_index)
        if truth_index_int in used_truth:
            continue
        used_truth.add(truth_index_int)
        matched += 1

    tp = float(matched)
    return tp, pred_count - tp, truth_count - tp


def _nearest_truth_indices(
    pred_points: np.ndarray,
    truth_points: np.ndarray,
    tolerance_pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest GT index for each predicted point within tolerance.

    SciPy's cKDTree is the intended path for paper-facing strict evaluation.
    The NumPy fallback keeps smoke tests and small strict-subset checks usable in
    minimal environments, but it can be slower on large full-resolution runs.
    """
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError:
        return _nearest_truth_indices_numpy(pred_points, truth_points, tolerance_pixels)

    truth_tree = cKDTree(truth_points)
    return truth_tree.query(
        pred_points,
        k=1,
        distance_upper_bound=float(tolerance_pixels),
        workers=-1,
    )


def _nearest_truth_indices_numpy(
    pred_points: np.ndarray,
    truth_points: np.ndarray,
    tolerance_pixels: float,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Small-environment fallback for nearest GT lookup without SciPy."""
    pred_points = np.asarray(pred_points, dtype=np.float32)
    truth_points = np.asarray(truth_points, dtype=np.float32)
    distances = np.full((len(pred_points),), np.inf, dtype=np.float64)
    indices = np.full((len(pred_points),), len(truth_points), dtype=np.int64)
    if len(pred_points) == 0 or len(truth_points) == 0:
        return distances, indices

    tolerance_sq = float(tolerance_pixels) ** 2
    chunk = max(1, int(chunk_size))
    truth_y = truth_points[:, 0][None, :]
    truth_x = truth_points[:, 1][None, :]
    for start in range(0, len(pred_points), chunk):
        block = pred_points[start : start + chunk]
        dy = block[:, 0:1] - truth_y
        dx = block[:, 1:2] - truth_x
        dist_sq = dy * dy + dx * dx
        best = np.argmin(dist_sq, axis=1)
        best_dist_sq = dist_sq[np.arange(len(block)), best]
        within = best_dist_sq <= tolerance_sq
        if within.any():
            out = slice(start, start + len(block))
            out_distances = distances[out]
            out_indices = indices[out]
            out_distances[within] = np.sqrt(best_dist_sq[within]).astype(np.float64)
            out_indices[within] = best[within].astype(np.int64)
            distances[out] = out_distances
            indices[out] = out_indices
    return distances, indices


def _tolerance_pixels(shape: tuple[int, int], match_tolerance: float) -> float:
    """Convert relative or absolute tolerance into pixels."""
    height, width = shape
    if match_tolerance < 1.0:
        return max(1.0, float(match_tolerance) * float(np.hypot(height, width)))
    return float(match_tolerance)


def _scores_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Convert TP/FP/FN counts into precision, recall, and F1."""
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return float(precision), float(recall), float(f1)


def _average_precision(target: np.ndarray, prob: np.ndarray) -> float:
    """Compute non-interpolated average precision with NumPy only."""
    order = np.argsort(-prob, kind="mergesort")
    target_sorted = target[order].astype(np.float64)
    true_positive = np.cumsum(target_sorted)
    false_positive = np.cumsum(1.0 - target_sorted)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / max(float(target_sorted.sum()), 1.0)
    recall_step = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * recall_step))


def average_precision_from_curve(rows: list[dict[str, float]]) -> float:
    """Integrate a finite PR curve with benchmark-style endpoint handling.

    Finite threshold sweeps do not necessarily contain the conventional
    ``(recall=0, precision=1)`` and ``(recall=1, precision=0)`` sentinels.
    Omitting those points makes AP depend strongly on score saturation at the
    largest sampled threshold.  Collapse duplicate recalls, add both
    sentinels, and integrate the monotone precision envelope so every
    evaluator and calibration search uses the same stable convention.
    """
    if not rows:
        return 0.0

    points = sorted((float(row["recall"]), float(row["precision"])) for row in rows)
    recall_values = np.asarray([point[0] for point in points], dtype=np.float64)
    precision_values = np.asarray([point[1] for point in points], dtype=np.float64)
    unique_recall = np.unique(recall_values)
    unique_precision = np.asarray(
        [precision_values[recall_values == value].max() for value in unique_recall],
        dtype=np.float64,
    )

    recall = np.concatenate(([0.0], unique_recall, [1.0]))
    precision = np.concatenate(([1.0], unique_precision, [0.0]))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.trapezoid(envelope, recall))


def _average_precision_from_curve(rows: list[dict[str, float]]) -> float:
    """Backward-compatible private alias for older metric call sites."""

    return average_precision_from_curve(rows)


def _flatten_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten image arrays into aligned prediction and target vectors."""
    flat_prob = np.concatenate([p.reshape(-1) for p in probabilities]).astype(np.float32)
    flat_target = np.concatenate([np.asarray(t, dtype=np.float32).reshape(-1) for t in targets]).clip(0.0, 1.0)
    return flat_prob, flat_target
