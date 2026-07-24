"""Loss functions for multi-scale edge detection with RBCM."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def class_balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Class-balanced BCE with normalized pixel weights.

    Edge labels are sparse, so plain BCE is dominated by background pixels. A
    raw `pos_weight=negative/positive` also makes the loss scale jump between
    batches. This variant balances foreground/background and normalizes by the
    active weight sum for steadier optimization.
    """
    target = target.float().clamp(0.0, 1.0)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    positive = (target * sample_weight).sum()
    negative = ((1.0 - target) * sample_weight).sum()
    total = (positive + negative).clamp_min(epsilon)
    positive_weight = (negative / total).clamp_min(epsilon)
    negative_weight = (positive / total).clamp_min(epsilon)
    weights = (target * positive_weight + (1.0 - target) * negative_weight) * sample_weight
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(epsilon)


def dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Dice loss computed on sigmoid probabilities."""
    prob = torch.sigmoid(logits)
    target = target.float().clamp(0.0, 1.0)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    numerator = 2.0 * (prob * target * sample_weight).sum(dim=(1, 2, 3)) + epsilon
    denominator = ((prob + target) * sample_weight).sum(dim=(1, 2, 3)) + epsilon
    return (1.0 - numerator / denominator).mean()


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Precision-biased Tversky loss for sparse edge maps.

    Edge detection often fails by producing overly dense texture responses. A
    larger `alpha` penalizes false positives more strongly than false negatives,
    while still keeping the objective differentiable on the active torch device.
    """
    prob = torch.sigmoid(logits)
    truth = target.float().clamp(0.0, 1.0)
    sample_weight = _prepare_sample_weight(sample_weight, truth)
    dims = (1, 2, 3)
    tp = (prob * truth * sample_weight).sum(dim=dims)
    fp = (prob * (1.0 - truth) * sample_weight).sum(dim=dims)
    fn = ((1.0 - prob) * truth * sample_weight).sum(dim=dims)
    score = (tp + epsilon) / (tp + float(alpha) * fp + float(beta) * fn + epsilon)
    loss = (1.0 - score).clamp_min(0.0)
    if gamma != 1.0:
        loss = loss.pow(float(gamma))
    return loss.mean()


def far_background_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    dilation_radius: int = 3,
    gamma: float = 1.5,
    target_threshold: float = 0.01,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Suppress confident predictions far away from any annotated edge.

    The target is dilated before building the background mask, so predictions
    near an edge are tolerated. This is closer to boundary benchmark matching
    than a global density penalty, and it directly attacks the high-density
    false-positive failure observed in RBCM runs.
    """
    prob = torch.sigmoid(logits)
    truth = _positive_target_mask(target, target_threshold)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    radius = max(0, int(dilation_radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        edge_band = F.max_pool2d(truth, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        edge_band = truth
    far_background = (1.0 - edge_band).clamp(0.0, 1.0)
    effective_weight = far_background * sample_weight
    penalty = prob.clamp(0.0, 1.0).pow(float(gamma)) * effective_weight
    return penalty.sum() / effective_weight.sum().clamp_min(epsilon)


def background_blob_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    dilation_radius: int = 3,
    pool_size: int = 9,
    margin: float = 0.06,
    gamma: float = 2.0,
    target_threshold: float = 0.01,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Penalize broad far-background probability blobs.

    NYUD indoor scenes exposed a failure mode where walls/floors received
    smooth, high-area probability mass. Pixelwise background losses suppress
    individual responses; this term looks at a local average and only penalizes
    it outside a dilated GT band. Thin true contours and their immediate
    tolerance neighborhood remain exempt.
    """
    prob = torch.sigmoid(logits).float()
    truth = _positive_target_mask(target, target_threshold)
    sample_weight = _prepare_sample_weight(sample_weight, target)

    radius = max(0, int(dilation_radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        edge_band = F.max_pool2d(truth, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        edge_band = truth
    far_background = (1.0 - edge_band).clamp(0.0, 1.0)

    pool = max(1, int(pool_size))
    if pool % 2 == 0:
        pool += 1
    local_mass = F.avg_pool2d(prob, kernel_size=pool, stride=1, padding=pool // 2)
    effective_weight = far_background * sample_weight
    penalty = F.relu(local_mass - float(margin)).pow(float(gamma)) * effective_weight
    return penalty.sum() / effective_weight.sum().clamp_min(epsilon)


def context_support_target(
    target: torch.Tensor,
    dilation_radius: int = 3,
    gamma: float = 1.0,
    target_threshold: float = 0.01,
) -> torch.Tensor:
    """Build a soft context-support target from annotated edge pixels.

    The context branch is not meant to replace the final thin edge target. It
    learns where boundary evidence exists in a tolerance band around the edge,
    which gives the modulation block an explicit, inspectable context signal.
    """
    del target_threshold
    truth = target.float().clamp(0.0, 1.0)
    radius = max(0, int(dilation_radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        support = F.max_pool2d(truth, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        support = truth
    if float(gamma) != 1.0:
        support = support.clamp(0.0, 1.0).pow(float(gamma))
    return support.clamp(0.0, 1.0)


def isolated_directional_response_loss(
    isolated_excess: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    dilation_radius: int = 3,
    gamma: float = 1.5,
    target_threshold: float = 0.01,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Penalize isolated texture responses away from the tolerated GT band.

    `isolated_excess` is produced by the directional-continuity head as
    max(probability - same-direction support, 0). Penalizing it only in far
    background suppresses short speckle-like edge fragments without discouraging
    true edges or valid endpoints near annotations.
    """
    isolated = isolated_excess.float().clamp_min(0.0)
    truth = _positive_target_mask(target, target_threshold)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    radius = max(0, int(dilation_radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        edge_band = F.max_pool2d(truth, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        edge_band = truth
    far_background = (1.0 - edge_band).clamp(0.0, 1.0)
    effective_weight = far_background * sample_weight
    penalty = isolated.pow(float(gamma)) * effective_weight
    return penalty.sum() / effective_weight.sum().clamp_min(epsilon)


def margin_calibration_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    edge_margin: float = 0.55,
    background_margin: float = 0.04,
    background_dilation: int = 3,
    target_threshold: float = 0.01,
    edge_weight: float = 0.35,
    background_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrate probabilities near true edges and far from true edges.

    Distance-soft labels make training tolerant, but NYUD showed a calibration
    failure where low-confidence texture filled the probability map while
    thresholded edges stayed sparse. This margin objective keeps true edge
    pixels above a modest confidence floor and far-background pixels below a
    small probability ceiling.
    """
    prob = torch.sigmoid(logits).float()
    truth = _positive_target_mask(target, target_threshold)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    edge_weight_map = truth * sample_weight
    edge_loss = F.relu(float(edge_margin) - prob).pow(2) * edge_weight_map
    edge_loss = edge_loss.sum() / edge_weight_map.sum().clamp_min(epsilon)

    radius = max(0, int(background_dilation))
    if radius > 0:
        kernel_size = 2 * radius + 1
        edge_band = F.max_pool2d(truth, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        edge_band = truth
    far_background = (1.0 - edge_band).clamp(0.0, 1.0)
    background_weight_map = far_background * sample_weight
    background_loss = F.relu(prob - float(background_margin)).pow(2) * background_weight_map
    background_loss = background_loss.sum() / background_weight_map.sum().clamp_min(epsilon)

    total = float(edge_weight) * edge_loss + float(background_weight) * background_loss
    return total, edge_loss, background_loss


def soft_active_density_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    min_multiplier: float = 0.35,
    max_multiplier: float = 1.45,
    density_floor: float = 0.002,
    active_threshold: float = 0.5,
    temperature: float = 0.06,
    target_threshold: float = 0.01,
) -> torch.Tensor:
    """Keep high-confidence edge density inside a soft target-relative range."""
    prob = torch.sigmoid(logits).float()
    soft_active = torch.sigmoid((prob - float(active_threshold)) / max(float(temperature), 1e-4))
    active_density = soft_active.mean(dim=(1, 2, 3))
    target_density = _positive_target_mask(target, target_threshold).mean(dim=(1, 2, 3))
    lower = float(min_multiplier) * target_density + float(density_floor)
    upper = float(max_multiplier) * target_density + float(density_floor)
    low_loss = F.relu(lower - active_density).pow(2)
    high_loss = F.relu(active_density - upper).pow(2)
    return (low_loss + high_loss).mean()


def continuity_support_loss(
    support: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    support_target: float = 0.14,
    dilation_radius: int = 1,
    gamma: float = 1.0,
    target_threshold: float = 0.01,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Encourage annotated edge pixels to have same-direction neighborhood support."""
    support = support.float().clamp(0.0, 1.0)
    truth = _positive_target_mask(target, target_threshold)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    radius = max(0, int(dilation_radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        truth = F.max_pool2d(truth, kernel_size=kernel_size, stride=1, padding=radius)
    effective_weight = truth * sample_weight
    penalty = F.relu(float(support_target) - support).pow(float(gamma)) * effective_weight
    return penalty.sum() / effective_weight.sum().clamp_min(epsilon)


def state_supervision_loss(
    mix_weights: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    edge_dilation: int = 1,
    suppress_dilation: int = 5,
    target_threshold: float = 0.01,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Supervise RBCM enhance/suppress/neutral states from edge morphology.

    Enhance is assigned to the annotated edge center band. Suppress is assigned
    to a local surround band around edges, excluding the center. Neutral is used
    elsewhere. This is intentionally coarse: it teaches the modulation block an
    interpretable center-surround role without changing the final evaluator.
    """
    if mix_weights.ndim != 4 or mix_weights.shape[1] != 3:
        return torch.zeros((), device=mix_weights.device, dtype=mix_weights.dtype)
    truth = _positive_target_mask(target, target_threshold)
    edge_radius = max(0, int(edge_dilation))
    if edge_radius > 0:
        edge_band = F.max_pool2d(truth, kernel_size=2 * edge_radius + 1, stride=1, padding=edge_radius)
    else:
        edge_band = truth
    suppress_radius = max(edge_radius + 1, int(suppress_dilation))
    surround_support = F.max_pool2d(truth, kernel_size=2 * suppress_radius + 1, stride=1, padding=suppress_radius)
    suppress_band = (surround_support - edge_band).clamp(0.0, 1.0)
    neutral = (1.0 - torch.maximum(edge_band, suppress_band)).clamp(0.0, 1.0)
    state_target = torch.cat([edge_band, suppress_band, neutral], dim=1)
    state_target = state_target / state_target.sum(dim=1, keepdim=True).clamp_min(epsilon)
    prob = mix_weights.float().clamp_min(epsilon)
    loss_map = -(state_target.float() * prob.log()).sum(dim=1, keepdim=True)
    sample_weight = _prepare_sample_weight(sample_weight, target)
    return (loss_map * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)


def edge_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined BCE+Dice loss and its components."""
    bce = class_balanced_bce_with_logits(logits, target, sample_weight=sample_weight)
    dice = dice_loss_with_logits(logits, target, sample_weight=sample_weight)
    return bce + dice_weight * dice, bce, dice


def _prepare_sample_weight(sample_weight: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Return a clamped `[B, 1, H, W]` weight tensor on the reference device."""
    if sample_weight is None:
        return torch.ones_like(reference, dtype=reference.dtype, device=reference.device)
    weight = sample_weight.to(device=reference.device, dtype=reference.dtype)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[-2:] != reference.shape[-2:]:
        weight = F.interpolate(weight, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    if weight.shape[1] != reference.shape[1]:
        weight = weight.mean(dim=1, keepdim=True)
    return weight.clamp(0.0, 1.0)


def _positive_target_mask(target: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Return a binary support mask for soft or hard edge targets.

    The main BCE/Dice terms still consume the continuous soft-vote target.
    Auxiliary region losses, however, need a binary "near annotated edge" mask.
    A fixed 0.5 threshold is too strict after resizing one-pixel soft labels, so
    the default treats any meaningful annotation support as positive.
    """
    value = target.float().clamp(0.0, 1.0)
    threshold = max(0.0, float(threshold))
    if threshold <= 0.0:
        return (value > 0.0).float()
    return (value > threshold).float()


class EdgeDetectionLoss(nn.Module):
    """Combined loss for final, local, and multi-scale side edge predictions."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        local_weight: float = 0.3,
        side_weight: float = 0.4,
        context_weight: float = 0.0,
        context_dilation: int = 3,
        context_gamma: float = 1.0,
        gate_sparsity_weight: float = 1e-4,
        density_weight: float = 0.0,
        density_target_multiplier: float = 2.0,
        density_floor: float = 0.005,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 1.0,
        far_background_weight: float = 0.0,
        far_background_dilation: int = 3,
        far_background_gamma: float = 1.5,
        background_blob_weight: float = 0.0,
        background_blob_dilation: int = 3,
        background_blob_pool_size: int = 9,
        background_blob_margin: float = 0.06,
        background_blob_gamma: float = 2.0,
        aux_tversky_weight: float = 0.0,
        aux_far_background_weight: float = 0.0,
        aux_far_background_dilation: int | None = None,
        aux_far_background_gamma: float | None = None,
        mix_balance_weight: float = 0.0,
        mix_prior: list[float] | tuple[float, ...] | None = None,
        separation_weight: float = 0.0,
        uncertainty_prior_weight: float = 0.0,
        uncertainty_prior: float = 0.35,
        hf_residual_weight: float = 0.0,
        hf_gate_sparsity_weight: float = 0.0,
        continuity_residual_weight: float = 0.0,
        continuity_gate_sparsity_weight: float = 0.0,
        continuity_isolated_weight: float = 0.0,
        continuity_isolated_dilation: int | None = None,
        continuity_isolated_gamma: float = 1.5,
        continuity_support_weight: float = 0.0,
        continuity_support_target: float = 0.14,
        continuity_support_dilation: int = 1,
        continuity_support_gamma: float = 1.0,
        calibration_weight: float = 0.0,
        calibration_edge_margin: float = 0.55,
        calibration_background_margin: float = 0.04,
        calibration_background_dilation: int = 3,
        calibration_edge_weight: float = 0.35,
        calibration_background_weight: float = 1.0,
        active_density_weight: float = 0.0,
        active_density_min_multiplier: float = 0.35,
        active_density_max_multiplier: float = 1.45,
        active_density_floor: float = 0.002,
        active_density_threshold: float = 0.5,
        active_density_temperature: float = 0.06,
        state_supervision_weight: float = 0.0,
        state_edge_dilation: int = 1,
        state_suppress_dilation: int = 5,
        target_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.local_weight = local_weight
        self.side_weight = side_weight
        self.context_weight = context_weight
        self.context_dilation = int(context_dilation)
        self.context_gamma = context_gamma
        self.gate_sparsity_weight = gate_sparsity_weight
        self.density_weight = density_weight
        self.density_target_multiplier = density_target_multiplier
        self.density_floor = density_floor
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma
        self.far_background_weight = far_background_weight
        self.far_background_dilation = int(far_background_dilation)
        self.far_background_gamma = far_background_gamma
        self.background_blob_weight = background_blob_weight
        self.background_blob_dilation = int(background_blob_dilation)
        self.background_blob_pool_size = int(background_blob_pool_size)
        self.background_blob_margin = background_blob_margin
        self.background_blob_gamma = background_blob_gamma
        self.aux_tversky_weight = aux_tversky_weight
        self.aux_far_background_weight = aux_far_background_weight
        self.aux_far_background_dilation = (
            self.far_background_dilation
            if aux_far_background_dilation is None
            else int(aux_far_background_dilation)
        )
        self.aux_far_background_gamma = (
            self.far_background_gamma if aux_far_background_gamma is None else float(aux_far_background_gamma)
        )
        self.mix_balance_weight = mix_balance_weight
        self.mix_prior = tuple(float(value) for value in mix_prior) if mix_prior is not None else None
        self.separation_weight = separation_weight
        self.uncertainty_prior_weight = uncertainty_prior_weight
        self.uncertainty_prior = uncertainty_prior
        self.hf_residual_weight = hf_residual_weight
        self.hf_gate_sparsity_weight = hf_gate_sparsity_weight
        self.continuity_residual_weight = continuity_residual_weight
        self.continuity_gate_sparsity_weight = continuity_gate_sparsity_weight
        self.continuity_isolated_weight = continuity_isolated_weight
        self.continuity_isolated_dilation = (
            self.far_background_dilation
            if continuity_isolated_dilation is None
            else int(continuity_isolated_dilation)
        )
        self.continuity_isolated_gamma = continuity_isolated_gamma
        self.continuity_support_weight = continuity_support_weight
        self.continuity_support_target = continuity_support_target
        self.continuity_support_dilation = int(continuity_support_dilation)
        self.continuity_support_gamma = continuity_support_gamma
        self.calibration_weight = calibration_weight
        self.calibration_edge_margin = calibration_edge_margin
        self.calibration_background_margin = calibration_background_margin
        self.calibration_background_dilation = int(calibration_background_dilation)
        self.calibration_edge_weight = calibration_edge_weight
        self.calibration_background_weight = calibration_background_weight
        self.active_density_weight = active_density_weight
        self.active_density_min_multiplier = active_density_min_multiplier
        self.active_density_max_multiplier = active_density_max_multiplier
        self.active_density_floor = active_density_floor
        self.active_density_threshold = active_density_threshold
        self.active_density_temperature = active_density_temperature
        self.state_supervision_weight = state_supervision_weight
        self.state_edge_dilation = int(state_edge_dilation)
        self.state_suppress_dilation = int(state_suppress_dilation)
        self.target_threshold = target_threshold

    def forward(
        self,
        final_logits: torch.Tensor,
        target: torch.Tensor,
        local_logits: torch.Tensor | None = None,
        context_logits: torch.Tensor | None = None,
        side_logits: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        sample_weight: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        mix_weights: torch.Tensor | None = None,
        local_feature: torch.Tensor | None = None,
        context_feature: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        hf_logit_residual: torch.Tensor | None = None,
        hf_gate: torch.Tensor | None = None,
        continuity_logit_residual: torch.Tensor | None = None,
        continuity_gate: torch.Tensor | None = None,
        continuity_support: torch.Tensor | None = None,
        continuity_isolated_excess: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        final_loss, final_bce, final_dice = edge_loss_with_logits(
            final_logits,
            target,
            dice_weight=self.dice_weight,
            sample_weight=sample_weight,
        )
        total = final_loss

        local_loss = torch.zeros((), device=final_logits.device)
        if local_logits is not None:
            local_loss, _, _ = edge_loss_with_logits(
                local_logits,
                target,
                dice_weight=self.dice_weight,
                sample_weight=sample_weight,
            )
            total = total + self.local_weight * local_loss

        context_loss = torch.zeros((), device=final_logits.device)
        if context_logits is not None and self.context_weight > 0:
            context_target = context_support_target(
                target,
                dilation_radius=self.context_dilation,
                gamma=self.context_gamma,
            )
            context_loss = class_balanced_bce_with_logits(
                context_logits,
                context_target,
                sample_weight=sample_weight,
            )
            total = total + self.context_weight * context_loss

        side_loss = torch.zeros((), device=final_logits.device)
        if side_logits:
            side_values = [
                edge_loss_with_logits(logit, target, dice_weight=self.dice_weight, sample_weight=sample_weight)[0]
                for logit in side_logits
            ]
            side_loss = torch.stack(side_values).mean()
            total = total + self.side_weight * side_loss

        aux_logits: list[torch.Tensor] = []
        if local_logits is not None:
            aux_logits.append(local_logits)
        if side_logits:
            aux_logits.extend(list(side_logits))

        gate_loss = torch.zeros((), device=final_logits.device)
        if gate is not None:
            gate_loss = gate.abs().mean()
            total = total + self.gate_sparsity_weight * gate_loss

        density_loss = torch.zeros((), device=final_logits.device)
        if self.density_weight > 0:
            prob_density = torch.sigmoid(final_logits).mean(dim=(1, 2, 3))
            target_density = target.float().clamp(0.0, 1.0).mean(dim=(1, 2, 3))
            density_limit = self.density_target_multiplier * target_density + self.density_floor
            density_loss = F.relu(prob_density - density_limit).pow(2).mean()
            total = total + self.density_weight * density_loss

        active_density_loss = torch.zeros((), device=final_logits.device)
        if self.active_density_weight > 0:
            active_density_loss = soft_active_density_loss_with_logits(
                final_logits,
                target,
                min_multiplier=self.active_density_min_multiplier,
                max_multiplier=self.active_density_max_multiplier,
                density_floor=self.active_density_floor,
                active_threshold=self.active_density_threshold,
                temperature=self.active_density_temperature,
                target_threshold=self.target_threshold,
            )
            total = total + self.active_density_weight * active_density_loss

        tversky_loss = torch.zeros((), device=final_logits.device)
        if self.tversky_weight > 0:
            tversky_loss = tversky_loss_with_logits(
                final_logits,
                target,
                sample_weight=sample_weight,
                alpha=self.tversky_alpha,
                beta=self.tversky_beta,
                gamma=self.tversky_gamma,
            )
            total = total + self.tversky_weight * tversky_loss

        far_background_loss = torch.zeros((), device=final_logits.device)
        if self.far_background_weight > 0:
            far_background_loss = far_background_loss_with_logits(
                final_logits,
                target,
                sample_weight=sample_weight,
                dilation_radius=self.far_background_dilation,
                gamma=self.far_background_gamma,
                target_threshold=self.target_threshold,
            )
            total = total + self.far_background_weight * far_background_loss

        background_blob_loss = torch.zeros((), device=final_logits.device)
        if self.background_blob_weight > 0:
            background_blob_loss = background_blob_loss_with_logits(
                final_logits,
                target,
                sample_weight=sample_weight,
                dilation_radius=self.background_blob_dilation,
                pool_size=self.background_blob_pool_size,
                margin=self.background_blob_margin,
                gamma=self.background_blob_gamma,
                target_threshold=self.target_threshold,
            )
            total = total + self.background_blob_weight * background_blob_loss

        calibration_loss = torch.zeros((), device=final_logits.device)
        edge_confidence_loss = torch.zeros((), device=final_logits.device)
        background_margin_loss = torch.zeros((), device=final_logits.device)
        if self.calibration_weight > 0:
            calibration_loss, edge_confidence_loss, background_margin_loss = margin_calibration_loss_with_logits(
                final_logits,
                target,
                sample_weight=sample_weight,
                edge_margin=self.calibration_edge_margin,
                background_margin=self.calibration_background_margin,
                background_dilation=self.calibration_background_dilation,
                target_threshold=self.target_threshold,
                edge_weight=self.calibration_edge_weight,
                background_weight=self.calibration_background_weight,
            )
            total = total + self.calibration_weight * calibration_loss

        aux_tversky_loss = torch.zeros((), device=final_logits.device)
        if self.aux_tversky_weight > 0 and aux_logits:
            aux_tversky_loss = torch.stack(
                [
                    tversky_loss_with_logits(
                        logit,
                        target,
                        sample_weight=sample_weight,
                        alpha=self.tversky_alpha,
                        beta=self.tversky_beta,
                        gamma=self.tversky_gamma,
                    )
                    for logit in aux_logits
                ]
            ).mean()
            total = total + self.aux_tversky_weight * aux_tversky_loss

        aux_far_background_loss = torch.zeros((), device=final_logits.device)
        if self.aux_far_background_weight > 0 and aux_logits:
            aux_far_background_loss = torch.stack(
                [
                    far_background_loss_with_logits(
                        logit,
                        target,
                        sample_weight=sample_weight,
                        dilation_radius=self.aux_far_background_dilation,
                        gamma=self.aux_far_background_gamma,
                        target_threshold=self.target_threshold,
                    )
                    for logit in aux_logits
                ]
            ).mean()
            total = total + self.aux_far_background_weight * aux_far_background_loss

        mix_balance_loss = torch.zeros((), device=final_logits.device)
        if self.mix_balance_weight > 0 and mix_weights is not None:
            mean_mix = mix_weights.mean(dim=(0, 2, 3))
            if self.mix_prior is not None and len(self.mix_prior) == mean_mix.numel():
                target_mix = torch.tensor(self.mix_prior, dtype=mean_mix.dtype, device=mean_mix.device)
            else:
                target_mix = torch.full_like(mean_mix, 1.0 / max(mean_mix.numel(), 1))
            mix_balance_loss = (mean_mix - target_mix).pow(2).mean()
            total = total + self.mix_balance_weight * mix_balance_loss

        state_supervision_term = torch.zeros((), device=final_logits.device)
        if self.state_supervision_weight > 0 and mix_weights is not None:
            state_supervision_term = state_supervision_loss(
                mix_weights,
                target,
                sample_weight=sample_weight,
                edge_dilation=self.state_edge_dilation,
                suppress_dilation=self.state_suppress_dilation,
                target_threshold=self.target_threshold,
            )
            total = total + self.state_supervision_weight * state_supervision_term

        separation_loss = torch.zeros((), device=final_logits.device)
        if self.separation_weight > 0 and local_feature is not None and context_feature is not None:
            local_flat = local_feature.flatten(2)
            context_flat = context_feature.flatten(2)
            local_flat = local_flat - local_flat.mean(dim=2, keepdim=True)
            context_flat = context_flat - context_flat.mean(dim=2, keepdim=True)
            local_flat = F.normalize(local_flat, dim=2)
            context_flat = F.normalize(context_flat, dim=2)
            separation_loss = (local_flat * context_flat).sum(dim=2).pow(2).mean()
            total = total + self.separation_weight * separation_loss

        uncertainty_prior_loss = torch.zeros((), device=final_logits.device)
        if self.uncertainty_prior_weight > 0 and uncertainty is not None:
            uncertainty_mean = uncertainty.float().mean()
            target_uncertainty = torch.as_tensor(
                self.uncertainty_prior,
                dtype=uncertainty_mean.dtype,
                device=uncertainty_mean.device,
            )
            uncertainty_prior_loss = (uncertainty_mean - target_uncertainty).pow(2)
            total = total + self.uncertainty_prior_weight * uncertainty_prior_loss

        hf_residual_loss = torch.zeros((), device=final_logits.device)
        if self.hf_residual_weight > 0 and hf_logit_residual is not None:
            hf_residual_loss = hf_logit_residual.float().abs().mean()
            total = total + self.hf_residual_weight * hf_residual_loss

        hf_gate_loss = torch.zeros((), device=final_logits.device)
        if self.hf_gate_sparsity_weight > 0 and hf_gate is not None:
            hf_gate_loss = hf_gate.float().mean()
            total = total + self.hf_gate_sparsity_weight * hf_gate_loss

        continuity_residual_loss = torch.zeros((), device=final_logits.device)
        if self.continuity_residual_weight > 0 and continuity_logit_residual is not None:
            continuity_residual_loss = continuity_logit_residual.float().abs().mean()
            total = total + self.continuity_residual_weight * continuity_residual_loss

        continuity_gate_loss = torch.zeros((), device=final_logits.device)
        if self.continuity_gate_sparsity_weight > 0 and continuity_gate is not None:
            continuity_gate_loss = continuity_gate.float().mean()
            total = total + self.continuity_gate_sparsity_weight * continuity_gate_loss

        continuity_isolated_loss = torch.zeros((), device=final_logits.device)
        if self.continuity_isolated_weight > 0 and continuity_isolated_excess is not None:
            continuity_isolated_loss = isolated_directional_response_loss(
                continuity_isolated_excess,
                target,
                sample_weight=sample_weight,
                dilation_radius=self.continuity_isolated_dilation,
                gamma=self.continuity_isolated_gamma,
                target_threshold=self.target_threshold,
            )
            total = total + self.continuity_isolated_weight * continuity_isolated_loss

        continuity_support_term = torch.zeros((), device=final_logits.device)
        if self.continuity_support_weight > 0 and continuity_support is not None:
            continuity_support_term = continuity_support_loss(
                continuity_support,
                target,
                sample_weight=sample_weight,
                support_target=self.continuity_support_target,
                dilation_radius=self.continuity_support_dilation,
                gamma=self.continuity_support_gamma,
                target_threshold=self.target_threshold,
            )
            total = total + self.continuity_support_weight * continuity_support_term

        return {
            "total": total,
            "final_bce": final_bce,
            "final_dice": final_dice,
            "local": local_loss,
            "context": context_loss,
            "side": side_loss,
            "gate_sparsity": gate_loss,
            "density": density_loss,
            "active_density": active_density_loss,
            "tversky": tversky_loss,
            "far_background": far_background_loss,
            "background_blob": background_blob_loss,
            "calibration": calibration_loss,
            "edge_confidence": edge_confidence_loss,
            "background_margin": background_margin_loss,
            "aux_tversky": aux_tversky_loss,
            "aux_far_background": aux_far_background_loss,
            "mix_balance": mix_balance_loss,
            "state_supervision": state_supervision_term,
            "separation": separation_loss,
            "uncertainty_prior": uncertainty_prior_loss,
            "hf_residual_reg": hf_residual_loss,
            "hf_gate_sparsity": hf_gate_loss,
            "continuity_residual_reg": continuity_residual_loss,
            "continuity_gate_sparsity": continuity_gate_loss,
            "continuity_isolated": continuity_isolated_loss,
            "continuity_support": continuity_support_term,
        }
