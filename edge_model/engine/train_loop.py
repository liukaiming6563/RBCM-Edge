"""Training and evaluation loops for edge detection."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from PIL import Image
from tqdm import tqdm

from edge_model.engine.metrics import (
    FastEdgeMetricAccumulator,
    edge_metrics_from_arrays,
    nms_probabilities,
)
from edge_model.engine.visualize import (
    save_gate_heatmap,
    save_probability_map,
    save_triplet_visualization,
)

METRIC_FIELD_ORDER = [
    "epoch",
    "split",
    "total",
    "final_bce",
    "final_dice",
    "local",
    "context",
    "side",
    "gate_sparsity",
    "density",
    "active_density",
    "tversky",
    "far_background",
    "background_blob",
    "calibration",
    "edge_confidence",
    "background_margin",
    "aux_tversky",
    "aux_far_background",
    "mix_balance",
    "state_supervision",
    "separation",
    "uncertainty_prior",
    "hf_residual_reg",
    "hf_gate_sparsity",
    "continuity_residual_reg",
    "continuity_gate_sparsity",
    "continuity_isolated",
    "continuity_support",
    "gate_mean",
    "gate_abs_mean",
    "enhance_gate_mean",
    "suppress_gate_mean",
    "mix_0_mean",
    "mix_1_mean",
    "mix_2_mean",
    "uncertainty_mean",
    "uncertainty_std",
    "hf_scale",
    "hf_residual_abs_mean",
    "hf_gate_mean",
    "continuity_scale",
    "continuity_residual_abs_mean",
    "continuity_gate_mean",
    "continuity_support_mean",
    "continuity_isolated_excess_mean",
    "dog_local_scale",
    "dog_surround_scale",
    "dog_strength_mean",
    "dog_raw_strength_mean",
    "dog_confidence_mean",
    "dog_ring_strength_mean",
    "dog_ring_confidence_mean",
    "dog_local_delta_abs_mean",
    "alpha",
    "context_prob_density",
    "context_edge_density",
    "pred_prob_density",
    "pred_edge_density",
    "target_edge_density",
    "density_ratio",
    "loss_weight_mean",
    "loss_weight_min",
    "loss_weight_max",
    "checkpoint_score",
    "ODS",
    "OIS",
    "AP",
    "ODS_threshold",
    "precision_at_ODS",
    "recall_at_ODS",
    "loss",
    "checkpoint",
    "dataset",
]


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    epoch: int,
    log_interval: int,
    gradient_clip_norm: float | None = None,
    diagnostic_interval: int = 10,
    resume_state: dict | None = None,
    checkpoint_interval_steps: int = 0,
    checkpoint_interval_seconds: float = 0.0,
    checkpoint_callback: Callable[[int, dict], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict:
    """Train one epoch and return metrics plus resumable accumulator state.

    A resumed epoch recreates the same deterministic DataLoader order and
    consumes already-completed batches without updating the model. The caller
    is responsible for constructing that loader with the same epoch seed.
    """
    model.train()
    resume_state = resume_state or {}
    resume_step = max(0, int(resume_state.get("step_in_epoch", resume_state.get("steps", 0))))
    loss_totals = restore_scalar_totals(resume_state.get("loss_totals"), device)
    stat_totals = restore_scalar_totals(resume_state.get("stat_totals"), device)
    steps = resume_step
    stat_steps = max(0, int(resume_state.get("stat_steps", 0)))
    diagnostic_interval = max(1, int(diagnostic_interval))
    amp_overflow_steps = max(0, int(resume_state.get("amp_overflow_steps", 0)))
    max_amp_overflow_steps = max(8, len(loader) // 4) if scaler is not None else 0
    checkpoint_interval_steps = max(0, int(checkpoint_interval_steps))
    checkpoint_interval_seconds = max(0.0, float(checkpoint_interval_seconds))
    last_checkpoint_time = time.monotonic()
    description = f"train epoch {epoch}"
    if resume_step:
        description += f" resume@{resume_step}"
    progress = tqdm(loader, desc=description, leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        if batch_index <= resume_step:
            continue
        step = batch_index
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["edge"].to(device, non_blocking=True)
        sample_weight = batch_loss_weight(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            losses = criterion(
                final_logits=outputs["logits"],
                target=targets,
                local_logits=outputs.get("local_logits"),
                context_logits=outputs.get("context_logits"),
                side_logits=outputs.get("side_logits"),
                sample_weight=sample_weight,
                gate=outputs.get("gate"),
                mix_weights=outputs.get("mix_weights"),
                local_feature=outputs.get("local_feature"),
                context_feature=outputs.get("context_feature"),
                uncertainty=outputs.get("uncertainty"),
                hf_logit_residual=outputs.get("hf_logit_residual"),
                hf_gate=outputs.get("hf_gate"),
                continuity_logit_residual=outputs.get("continuity_logit_residual"),
                continuity_gate=outputs.get("continuity_gate"),
                continuity_support=outputs.get("continuity_support"),
                continuity_isolated_excess=outputs.get("continuity_isolated_excess"),
            )
        loss = losses["total"]
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(build_nonfinite_training_report(epoch, step, batch, outputs, losses))

        if scaler is not None:
            scaler.scale(loss).backward()
            grad_norm = clip_and_check_gradients(model, optimizer, gradient_clip_norm, scaler=scaler)
            if grad_norm is not None and not torch.isfinite(grad_norm.detach()):
                amp_overflow_steps += 1
                if amp_overflow_steps <= 3:
                    print(
                        f"AMP gradient overflow at epoch={epoch}, step={step}; "
                        f"GradScaler will skip this optimizer step and lower the scale.",
                        flush=True,
                    )
                scaler.step(optimizer)
                scaler.update()
                if amp_overflow_steps > max_amp_overflow_steps:
                    raise FloatingPointError(build_nonfinite_gradient_report(epoch, step, batch, model, grad_norm))
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            grad_norm = clip_and_check_gradients(model, optimizer, gradient_clip_norm, scaler=None)
            if grad_norm is not None and not torch.isfinite(grad_norm.detach()):
                raise FloatingPointError(build_nonfinite_gradient_report(epoch, step, batch, model, grad_norm))
            optimizer.step()

        steps = step
        accumulate_scalar_tensors(loss_totals, losses)
        if step == 1 or step % diagnostic_interval == 0 or step == len(loader):
            accumulate_scalar_tensors(stat_totals, output_scalar_stat_tensors(outputs))
            accumulate_scalar_tensors(stat_totals, loss_weight_stat_tensors(sample_weight))
            stat_steps += 1
        if steps % max(1, log_interval) == 0:
            progress.set_postfix(total=float((loss_totals["total"] / steps).detach().cpu()))

        now = time.monotonic()
        checkpoint_due = checkpoint_callback is not None and (
            (checkpoint_interval_steps > 0 and step % checkpoint_interval_steps == 0)
            or (checkpoint_interval_seconds > 0.0 and (now - last_checkpoint_time) >= checkpoint_interval_seconds)
        )
        requested_stop = bool(stop_requested is not None and stop_requested())
        if checkpoint_callback is not None and (checkpoint_due or requested_stop):
            checkpoint_callback(
                step,
                build_train_epoch_state(
                    step=step,
                    loss_totals=loss_totals,
                    stat_totals=stat_totals,
                    stat_steps=stat_steps,
                    amp_overflow_steps=amp_overflow_steps,
                ),
            )
            last_checkpoint_time = now
        if requested_stop:
            state = build_train_epoch_state(
                step=step,
                loss_totals=loss_totals,
                stat_totals=stat_totals,
                stat_steps=stat_steps,
                amp_overflow_steps=amp_overflow_steps,
            )
            return {
                "metrics": average_train_epoch_state(state),
                "state": state,
                "completed": False,
            }

    state = build_train_epoch_state(
        step=steps,
        loss_totals=loss_totals,
        stat_totals=stat_totals,
        stat_steps=stat_steps,
        amp_overflow_steps=amp_overflow_steps,
    )
    return {
        "metrics": average_train_epoch_state(state),
        "state": state,
        "completed": True,
    }


def restore_scalar_totals(values: dict | None, device: torch.device) -> dict[str, torch.Tensor]:
    """Restore scalar accumulator values from a CPU checkpoint."""
    if not isinstance(values, dict):
        return {}
    return {
        str(key): torch.as_tensor(float(value), device=device, dtype=torch.float32)
        for key, value in values.items()
    }


def build_train_epoch_state(
    *,
    step: int,
    loss_totals: dict[str, torch.Tensor],
    stat_totals: dict[str, torch.Tensor],
    stat_steps: int,
    amp_overflow_steps: int,
) -> dict:
    """Serialize the small accumulator needed for exact mid-epoch continuation."""
    return {
        "step_in_epoch": int(step),
        "loss_totals": {
            key: float(value.detach().float().cpu())
            for key, value in loss_totals.items()
        },
        "stat_totals": {
            key: float(value.detach().float().cpu())
            for key, value in stat_totals.items()
        },
        "stat_steps": int(stat_steps),
        "amp_overflow_steps": int(amp_overflow_steps),
    }


def average_train_epoch_state(state: dict) -> dict[str, float]:
    """Convert a serialized training accumulator into averaged metrics."""
    steps = max(1, int(state.get("step_in_epoch", 0)))
    stat_steps = max(1, int(state.get("stat_steps", 0)))
    metrics = {
        str(key): float(value) / steps
        for key, value in (state.get("loss_totals", {}) or {}).items()
    }
    metrics.update(
        {
            str(key): float(value) / stat_steps
            for key, value in (state.get("stat_totals", {}) or {}).items()
        }
    )
    metrics["amp_overflow_steps"] = float(state.get("amp_overflow_steps", 0))
    return metrics


def clip_and_check_gradients(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None,
    scaler: GradScaler | None,
) -> torch.Tensor | None:
    """Unscale AMP grads, clip finite grads when requested, and return the pre-clip norm.

    AMP can legitimately overflow early while GradScaler is finding a safe scale.
    Compute the norm before clipping so an infinite norm does not turn Inf gradients
    into NaNs via an ``Inf * 0`` clip coefficient.
    """
    if gradient_clip_norm is None:
        return None
    max_norm = float(gradient_clip_norm)
    if max_norm <= 0:
        return None
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = total_gradient_norm(model)
    if torch.isfinite(grad_norm.detach()):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, error_if_nonfinite=False)
    return grad_norm


def total_gradient_norm(model: nn.Module) -> torch.Tensor:
    """Return a finite-aware global L2 gradient norm without modifying gradients."""
    norms: list[torch.Tensor] = []
    fallback_device: torch.device | None = None
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        fallback_device = grad.device
        detached = grad.detach()
        if not torch.isfinite(detached).all():
            return torch.full((), float("inf"), device=detached.device)
        norms.append(torch.linalg.vector_norm(detached.float(), ord=2))
    if norms:
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return torch.zeros((), device=fallback_device or torch.device("cpu"))


def build_nonfinite_training_report(
    epoch: int,
    step: int,
    batch: dict,
    outputs: dict,
    losses: dict[str, torch.Tensor],
) -> str:
    """Build a concise report for a non-finite forward/loss failure."""
    lines = [
        f"Non-finite training loss detected at epoch={epoch}, step={step}.",
        sample_id_report(batch),
        "Loss tensors:",
    ]
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            lines.append(tensor_finite_summary(name, value))
    lines.append("Selected output tensors:")
    for name in (
        "logits",
        "local_logits",
        "context_logits",
        "side_logits",
        "gate",
        "enhance_gate",
        "suppress_gate",
        "mix_weights",
        "mix_logits",
        "anchor_feature",
        "rbcm_feature",
        "residual_delta",
        "residual_gate",
        "local_feature",
        "context_feature",
    ):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            lines.append(tensor_finite_summary(name, value))
    return "\n".join(line for line in lines if line)


def build_nonfinite_gradient_report(
    epoch: int,
    step: int,
    batch: dict,
    model: nn.Module,
    grad_norm: torch.Tensor,
) -> str:
    """Build a concise report for a non-finite backward/gradient failure."""
    lines = [
        f"Non-finite gradient detected at epoch={epoch}, step={step}.",
        sample_id_report(batch),
        tensor_finite_summary("global_grad_norm", grad_norm),
        "First non-finite parameter gradients:",
    ]
    count = 0
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None or torch.isfinite(grad.detach()).all():
            continue
        lines.append(tensor_finite_summary(f"grad:{name}", grad))
        count += 1
        if count >= 8:
            break
    if count == 0:
        lines.append("No individual non-finite gradient tensor was found after global norm check.")
    return "\n".join(line for line in lines if line)


def tensor_finite_summary(name: str, value: torch.Tensor) -> str:
    """Summarize finite/non-finite content for one tensor without hiding NaNs."""
    detached = value.detach()
    finite = torch.isfinite(detached)
    total = detached.numel()
    nonfinite_count = int((~finite).sum().detach().cpu())
    if total == 0:
        return f"- {name}: empty tensor"
    if bool(finite.any().detach().cpu()):
        finite_values = detached[finite].float()
        finite_min = float(finite_values.min().detach().cpu())
        finite_max = float(finite_values.max().detach().cpu())
        finite_mean = float(finite_values.mean().detach().cpu())
        return (
            f"- {name}: nonfinite={nonfinite_count}/{total}, "
            f"finite_min={finite_min:.6g}, finite_max={finite_max:.6g}, finite_mean={finite_mean:.6g}"
        )
    return f"- {name}: nonfinite={nonfinite_count}/{total}, no finite values"


def sample_id_report(batch: dict, max_items: int = 4) -> str:
    """Return a short sample-id line for debugging a bad batch."""
    values = batch.get("sample_id")
    if values is None:
        return ""
    if isinstance(values, torch.Tensor):
        items = values.detach().cpu().flatten().tolist()[:max_items]
    elif isinstance(values, (list, tuple)):
        items = list(values)[:max_items]
    else:
        items = [values]
    return "Sample ids: " + ", ".join(str(item) for item in items)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module | None,
    device: torch.device,
    visual_dir: Path | None = None,
    pred_dir: Path | None = None,
    gate_dir: Path | None = None,
    max_visual_samples: int = 0,
    metric_mode: str = "strict",
    fast_metric_thresholds: int = 49,
    fast_metric_tolerance_pixels: int = 4,
    apply_nms: bool = False,
    nms_low_threshold: float = 0.0,
    diagnostic_interval: int = 10,
) -> dict[str, float]:
    """Evaluate a model and optionally save predictions and gate maps."""
    model.eval()
    use_fast_metrics = str(metric_mode).lower() in {"fast", "fast_gpu", "gpu", "approx"}
    fast_metrics = (
        FastEdgeMetricAccumulator(
            thresholds=int(fast_metric_thresholds),
            tolerance_pixels=int(fast_metric_tolerance_pixels),
        )
        if use_fast_metrics
        else None
    )
    probabilities: list = []
    targets: list = []
    loss_total: torch.Tensor | None = None
    loss_steps = 0
    saved = 0
    gate_saved = 0
    stat_totals: dict[str, torch.Tensor] = {}
    stat_steps = 0
    diagnostic_interval = max(1, int(diagnostic_interval))

    progress = tqdm(loader, desc="eval", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["edge"].to(device, non_blocking=True)
        sample_weight = batch_loss_weight(batch, device)
        outputs = model(images)
        probabilities_tensor = torch.sigmoid(outputs["logits"]).float()
        if apply_nms:
            probabilities_tensor = nms_probabilities(probabilities_tensor, low_threshold=float(nms_low_threshold))
        if batch_index == 1 or batch_index % diagnostic_interval == 0 or batch_index == len(loader):
            accumulate_scalar_tensors(stat_totals, output_scalar_stat_tensors(outputs))
            accumulate_scalar_tensors(stat_totals, loss_weight_stat_tensors(sample_weight))
            stat_steps += 1

        if fast_metrics is not None:
            fast_metrics.update(
                probabilities=probabilities_tensor,
                targets=target,
                valid_mask=valid_content_mask(batch, target.shape, device),
            )

        if criterion is not None:
            losses = criterion(
                final_logits=outputs["logits"],
                target=target,
                local_logits=outputs.get("local_logits"),
                context_logits=outputs.get("context_logits"),
                side_logits=outputs.get("side_logits"),
                sample_weight=sample_weight,
                gate=outputs.get("gate"),
                mix_weights=outputs.get("mix_weights"),
                local_feature=outputs.get("local_feature"),
                context_feature=outputs.get("context_feature"),
                uncertainty=outputs.get("uncertainty"),
                hf_logit_residual=outputs.get("hf_logit_residual"),
                hf_gate=outputs.get("hf_gate"),
                continuity_logit_residual=outputs.get("continuity_logit_residual"),
                continuity_gate=outputs.get("continuity_gate"),
                continuity_support=outputs.get("continuity_support"),
                continuity_isolated_excess=outputs.get("continuity_isolated_excess"),
            )
            detached_loss = losses["total"].detach().float()
            loss_total = detached_loss if loss_total is None else loss_total + detached_loss
            loss_steps += 1

        need_numpy_arrays = (fast_metrics is None) or pred_dir is not None or visual_dir is not None
        if not need_numpy_arrays:
            continue

        batch_prob = probabilities_tensor.detach().cpu().numpy()
        batch_target = target.detach().cpu().numpy()
        for idx in range(batch_prob.shape[0]):
            raw_prob = batch_prob[idx, 0]
            raw_truth = batch_target[idx, 0]
            prob = restore_sample_array(raw_prob, batch, idx, is_target=False)
            truth = load_original_target_array(batch, idx)
            if truth is None:
                truth = restore_sample_array(raw_truth, batch, idx, is_target=True)
            if fast_metrics is None:
                probabilities.append(prob)
                targets.append(truth)

            sample_id = str(batch["sample_id"][idx])
            if pred_dir is not None:
                save_probability_map(prob, pred_dir / f"{sample_id}.png")
            if gate_dir is not None and "gate" in outputs and gate_saved < max_visual_samples:
                save_gate_heatmap(restore_gate_tensor(outputs["gate"][idx], batch, idx), gate_dir / f"{sample_id}.png")
                gate_saved += 1
            if visual_dir is not None and saved < max_visual_samples:
                save_triplet_visualization(
                    image_tensor=restore_image_tensor(batch["image"][idx], batch, idx).cpu(),
                    target_tensor=torch.from_numpy(truth).unsqueeze(0),
                    probability=prob,
                    output_path=visual_dir / f"{sample_id}.png",
                )
                saved += 1

    metrics = fast_metrics.compute() if fast_metrics is not None else edge_metrics_from_arrays(probabilities, targets)
    if fast_metrics is None and probabilities:
        pred_prob_density = float(np.mean([prob.mean() for prob in probabilities]))
        pred_edge_density = float(np.mean([(prob >= 0.5).mean() for prob in probabilities]))
        target_edge_density = float(np.mean([truth.mean() for truth in targets]))
        metrics["pred_prob_density"] = pred_prob_density
        metrics["pred_edge_density"] = pred_edge_density
        metrics["target_edge_density"] = target_edge_density
        metrics["density_ratio"] = pred_edge_density / max(target_edge_density, 1e-6)
    if stat_steps:
        for key, value in stat_totals.items():
            metrics[key] = float((value / stat_steps).detach().cpu())
    if loss_steps and loss_total is not None:
        metrics["loss"] = float((loss_total / loss_steps).detach().cpu())
    return metrics


def load_original_target_array(batch: dict, idx: int) -> np.ndarray | None:
    """Load the raw soft-vote edge map for strict eval without binarizing it."""
    if "edge_path" not in batch:
        return None
    path_value = _batch_item(batch, "edge_path", idx)
    if path_value is None:
        return None
    edge_path = Path(str(path_value))
    if not edge_path.exists():
        return None
    with Image.open(edge_path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def accumulate_scalar_tensors(totals: dict[str, torch.Tensor], values: dict[str, torch.Tensor | float | int]) -> None:
    """Accumulate detached scalar tensors without forcing per-key CPU syncs."""
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            detached = value.detach().float()
        else:
            device = totals[key].device if key in totals else None
            detached = torch.as_tensor(float(value), device=device)
        totals[key] = detached if key not in totals else totals[key] + detached


def output_scalar_stat_tensors(outputs: dict) -> dict[str, torch.Tensor]:
    """Collect lightweight diagnostic scalar tensors on the active device."""
    stats: dict[str, torch.Tensor] = {}
    gate = outputs.get("gate")
    if isinstance(gate, torch.Tensor):
        detached = gate.detach().float()
        stats["gate_mean"] = detached.mean()
        stats["gate_abs_mean"] = detached.abs().mean()
    for name in ("enhance_gate", "suppress_gate"):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            stats[f"{name}_mean"] = value.detach().float().mean()
    mix = outputs.get("mix_weights")
    if isinstance(mix, torch.Tensor):
        mix_mean = mix.detach().float().mean(dim=(0, 2, 3))
        for index in range(min(3, mix_mean.numel())):
            stats[f"mix_{index}_mean"] = mix_mean[index]
    context_logits = outputs.get("context_logits")
    if isinstance(context_logits, torch.Tensor):
        context_prob = torch.sigmoid(context_logits.detach()).float()
        stats["context_prob_density"] = context_prob.mean()
        stats["context_edge_density"] = (context_prob >= 0.5).float().mean()
    uncertainty = outputs.get("uncertainty")
    if isinstance(uncertainty, torch.Tensor):
        detached = uncertainty.detach().float()
        stats["uncertainty_mean"] = detached.mean()
        stats["uncertainty_std"] = detached.std(unbiased=False)
    residual = outputs.get("hf_logit_residual")
    if isinstance(residual, torch.Tensor):
        stats["hf_residual_abs_mean"] = residual.detach().float().abs().mean()
    hf_gate = outputs.get("hf_gate")
    if isinstance(hf_gate, torch.Tensor):
        stats["hf_gate_mean"] = hf_gate.detach().float().mean()
    continuity_residual = outputs.get("continuity_logit_residual")
    if isinstance(continuity_residual, torch.Tensor):
        stats["continuity_residual_abs_mean"] = continuity_residual.detach().float().abs().mean()
    continuity_gate = outputs.get("continuity_gate")
    if isinstance(continuity_gate, torch.Tensor):
        stats["continuity_gate_mean"] = continuity_gate.detach().float().mean()
    continuity_support = outputs.get("continuity_support")
    if isinstance(continuity_support, torch.Tensor):
        stats["continuity_support_mean"] = continuity_support.detach().float().mean()
    isolated_excess = outputs.get("continuity_isolated_excess")
    if isinstance(isolated_excess, torch.Tensor):
        stats["continuity_isolated_excess_mean"] = isolated_excess.detach().float().mean()
    for name in ("hf_scale", "alpha"):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            stats[name] = value.detach().float().mean()
    value = outputs.get("continuity_scale")
    if isinstance(value, torch.Tensor):
        stats["continuity_scale"] = value.detach().float().mean()
    for key in ("dog_local_scale", "dog_surround_scale"):
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            stats[key] = value.detach().float().mean()
    for key, stat_name in (
        ("dog_strength", "dog_strength_mean"),
        ("dog_raw_strength", "dog_raw_strength_mean"),
        ("dog_confidence", "dog_confidence_mean"),
        ("dog_ring_strength", "dog_ring_strength_mean"),
        ("dog_ring_confidence", "dog_ring_confidence_mean"),
    ):
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            stats[stat_name] = value.detach().float().mean()
    value = outputs.get("dog_local_delta")
    if isinstance(value, torch.Tensor):
        stats["dog_local_delta_abs_mean"] = value.detach().float().abs().mean()
    return stats


def output_scalar_stats(outputs: dict) -> dict[str, float]:
    """Collect lightweight diagnostic scalars from model outputs."""
    return {key: float(value.detach().cpu()) for key, value in output_scalar_stat_tensors(outputs).items()}


def batch_loss_weight(batch: dict, device: torch.device) -> torch.Tensor | None:
    """Return an optional batch loss-weight tensor on the active device."""
    weight = batch.get("loss_weight")
    if not isinstance(weight, torch.Tensor):
        return None
    return weight.to(device, non_blocking=True)


def loss_weight_stat_tensors(weight: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Summarize optional uncertainty weights as scalar tensors."""
    if weight is None:
        return {}
    detached = weight.detach().float()
    return {
        "loss_weight_mean": detached.mean(),
        "loss_weight_min": detached.min(),
        "loss_weight_max": detached.max(),
    }


def loss_weight_stats(weight: torch.Tensor | None) -> dict[str, float]:
    """Summarize optional uncertainty weights for logs."""
    return {key: float(value.detach().cpu()) for key, value in loss_weight_stat_tensors(weight).items()}


def append_metrics_csv(path: str | Path, row: dict) -> None:
    """Append one metrics row while keeping a stable CSV schema.

    Training rows contain loss components such as `total` and `final_bce`,
    while validation rows contain edge metrics such as `ODS`, `OIS`, `AP`, and
    `loss`. The first implementation wrote each row with its own field order,
    which made validation values land under the training-loss columns. This
    helper keeps one union header for the whole file and expands that header if
    new columns appear in later rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict[str, str]] = []
    existing_fields: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fieldnames = _merge_metric_fields(existing_fields, row.keys())
    rows_to_write = existing_rows + [row]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_to_write)


def _merge_metric_fields(existing_fields: list[str], new_fields: Iterable[str]) -> list[str]:
    """Return a deterministic union of existing and new CSV columns."""
    merged: list[str] = []
    all_fields = list(existing_fields) + list(new_fields)
    for field in METRIC_FIELD_ORDER + all_fields:
        if field in all_fields and field not in merged:
            merged.append(field)
    return merged


def restore_sample_array(array: np.ndarray, batch: dict, idx: int, is_target: bool) -> np.ndarray:
    """Remove eval padding and resize a probability/target map to original size."""
    height, width = array.shape
    pad_top = _batch_int(batch, "pad_top", idx, 0)
    pad_left = _batch_int(batch, "pad_left", idx, 0)
    content_height = _batch_int(batch, "content_height", idx, height)
    content_width = _batch_int(batch, "content_width", idx, width)
    original_height = _batch_int(batch, "original_height", idx, content_height)
    original_width = _batch_int(batch, "original_width", idx, content_width)

    cropped = array[
        pad_top : min(pad_top + content_height, height),
        pad_left : min(pad_left + content_width, width),
    ]
    if cropped.shape == (original_height, original_width):
        return np.clip(cropped, 0.0, 1.0).astype(np.float32)

    if is_target:
        image = Image.fromarray(np.clip(cropped.astype(np.float32), 0.0, 1.0))
        resized = image.resize((original_width, original_height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32).clip(0.0, 1.0)

    image = Image.fromarray(cropped.astype(np.float32))
    resized = image.resize((original_width, original_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).clip(0.0, 1.0)


def restore_gate_tensor(gate: torch.Tensor, batch: dict, idx: int) -> torch.Tensor:
    """Remove eval padding from a gate tensor and resize it to original size."""
    _, height, width = gate.shape
    pad_top = _batch_int(batch, "pad_top", idx, 0)
    pad_left = _batch_int(batch, "pad_left", idx, 0)
    content_height = _batch_int(batch, "content_height", idx, height)
    content_width = _batch_int(batch, "content_width", idx, width)
    original_height = _batch_int(batch, "original_height", idx, content_height)
    original_width = _batch_int(batch, "original_width", idx, content_width)
    cropped = gate[
        :,
        pad_top : min(pad_top + content_height, height),
        pad_left : min(pad_left + content_width, width),
    ]
    if cropped.shape[-2:] == (original_height, original_width):
        return cropped
    return F.interpolate(
        cropped.unsqueeze(0),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def restore_image_tensor(image: torch.Tensor, batch: dict, idx: int) -> torch.Tensor:
    """Remove eval padding from a normalized image tensor for visualization."""
    _, height, width = image.shape
    pad_top = _batch_int(batch, "pad_top", idx, 0)
    pad_left = _batch_int(batch, "pad_left", idx, 0)
    content_height = _batch_int(batch, "content_height", idx, height)
    content_width = _batch_int(batch, "content_width", idx, width)
    original_height = _batch_int(batch, "original_height", idx, content_height)
    original_width = _batch_int(batch, "original_width", idx, content_width)
    cropped = image[
        :,
        pad_top : min(pad_top + content_height, height),
        pad_left : min(pad_left + content_width, width),
    ]
    if cropped.shape[-2:] == (original_height, original_width):
        return cropped
    return F.interpolate(
        cropped.unsqueeze(0),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def valid_content_mask(batch: dict, shape: torch.Size | tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Return a `[B, 1, H, W]` mask for non-padding pixels in letterboxed batches."""
    batch_size, _, height, width = shape
    pad_top = _batch_tensor_int(batch, "pad_top", batch_size, device, default=0).clamp(0, height)
    pad_left = _batch_tensor_int(batch, "pad_left", batch_size, device, default=0).clamp(0, width)
    content_height = _batch_tensor_int(batch, "content_height", batch_size, device, default=height).clamp(0, height)
    content_width = _batch_tensor_int(batch, "content_width", batch_size, device, default=width).clamp(0, width)

    y = torch.arange(height, device=device).view(1, 1, height, 1)
    x = torch.arange(width, device=device).view(1, 1, 1, width)
    y1 = pad_top.view(batch_size, 1, 1, 1)
    x1 = pad_left.view(batch_size, 1, 1, 1)
    y2 = (pad_top + content_height).clamp(0, height).view(batch_size, 1, 1, 1)
    x2 = (pad_left + content_width).clamp(0, width).view(batch_size, 1, 1, 1)
    return (y >= y1) & (y < y2) & (x >= x1) & (x < x2)


def _batch_int(batch: dict, key: str, idx: int, default: int) -> int:
    """Read an integer metadata value from a collated DataLoader batch."""
    value = _batch_item(batch, key, idx)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _batch_tensor_int(batch: dict, key: str, batch_size: int, device: torch.device, default: int) -> torch.Tensor:
    """Read a batched integer metadata field as a device tensor."""
    if key not in batch:
        return torch.full((batch_size,), int(default), device=device, dtype=torch.long)
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long).view(-1)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor([int(item) for item in value], device=device, dtype=torch.long)
    return torch.full((batch_size,), int(value), device=device, dtype=torch.long)


def _batch_item(batch: dict, key: str, idx: int):
    """Read one metadata item from a collated DataLoader batch."""
    if key not in batch:
        return None
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value[idx]
    if isinstance(value, (list, tuple)):
        item = value[idx]
        return item
    return value
