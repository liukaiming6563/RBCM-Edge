"""Evaluate the uncalibrated HED-lite Anchor stored in a formal checkpoint.

This entry point never applies the post-hoc H-RBCM calibration.  Use
``scripts/experiments/evaluate_generalization.py`` followed by
``scripts/baselines/evaluate_official_edges.py`` for paper-facing H-RBCM,
no-surround, and convolution-control results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.env import append_workspace_local_packages

append_workspace_local_packages(PROJECT_ROOT)

from edge_model.core.checkpoint_io import load_checkpoint
from edge_model.core.config import deep_update, load_config, project_path, save_config
from edge_model.core.paths import make_run_paths
from edge_model.data.build import make_dataset, make_loader
from edge_model.engine.train_loop import append_metrics_csv, evaluate
from edge_model.models.build import build_model
from rbcm_edge.models.losses import EdgeDetectionLoss

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "eval" / "final_eval_test.yaml",
    "checkpoint": None,
    "eval_dataset": None,
    "experiment_name": None,
    "output_root": None,
    "batch_size": None,
    "input_size": None,
    "device": None,
}


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARGS["checkpoint"])
    parser.add_argument("--eval-dataset", default=DEFAULT_ARGS["eval_dataset"])
    parser.add_argument("--experiment-name", default=DEFAULT_ARGS["experiment_name"])
    parser.add_argument("--output-root", default=DEFAULT_ARGS["output_root"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_ARGS["input_size"])
    parser.add_argument("--device", default=DEFAULT_ARGS["device"])
    parser.add_argument("--metric-mode", choices=["strict", "fast_gpu"], default=None)
    parser.add_argument("--fast-metric-thresholds", type=int, default=None)
    parser.add_argument("--fast-metric-tolerance-pixels", type=int, default=None)
    parser.add_argument("--apply-nms", action="store_true", default=None)
    parser.add_argument("--no-apply-nms", action="store_false", dest="apply_nms")
    parser.add_argument("--nms-low-threshold", type=float, default=None)
    parser.add_argument(
        "--output-mode",
        choices=["plain_identity"],
        default="plain_identity",
        help="This checkpoint evaluator is intentionally Anchor-only.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run Anchor checkpoint evaluation and optional prediction saving."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    config = deep_update(
        config,
        {
            "experiment_name": args.experiment_name,
            "device": args.device,
            "dataset": {"eval_dataset": args.eval_dataset, "input_size": args.input_size},
            "paths": {"output_root": args.output_root},
            "loader": {"batch_size": args.batch_size},
            "eval": {
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "metric_mode": args.metric_mode,
                "fast_metric_thresholds": args.fast_metric_thresholds,
                "fast_metric_tolerance_pixels": args.fast_metric_tolerance_pixels,
                "apply_nms": args.apply_nms,
                "nms_low_threshold": args.nms_low_threshold,
            },
        },
    )

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint_path = project_path(config, config["eval"]["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    checkpoint_config = deep_update(checkpoint_config, {"paths": config["paths"], "dataset": config["dataset"]})

    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    dataset = make_dataset(
        config,
        dataset_name=config["dataset"]["eval_dataset"],
        split=config["dataset"].get("eval_split", "all"),
        training=False,
    )
    loader = make_loader(dataset, config, shuffle=False)

    output_root = project_path(config, config["paths"].get("output_root", "results/rbcm/runs"))
    run_paths = make_run_paths(output_root, config.get("experiment_name", "eval"))
    save_config(run_paths.root / "config.yaml", config)
    loss_cfg = checkpoint_config.get("loss", config.get("loss", {}))
    criterion = EdgeDetectionLoss(
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        local_weight=float(loss_cfg.get("local_weight", 0.3)),
        side_weight=float(loss_cfg.get("side_weight", 0.4)),
        context_weight=float(loss_cfg.get("context_weight", 0.0)),
        context_dilation=int(loss_cfg.get("context_dilation", 3)),
        context_gamma=float(loss_cfg.get("context_gamma", 1.0)),
        gate_sparsity_weight=float(loss_cfg.get("gate_sparsity_weight", 1e-4)),
        density_weight=float(loss_cfg.get("density_weight", 0.0)),
        density_target_multiplier=float(loss_cfg.get("density_target_multiplier", 2.0)),
        density_floor=float(loss_cfg.get("density_floor", 0.005)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 0.0)),
        tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.7)),
        tversky_beta=float(loss_cfg.get("tversky_beta", 0.3)),
        tversky_gamma=float(loss_cfg.get("tversky_gamma", 1.0)),
        far_background_weight=float(loss_cfg.get("far_background_weight", 0.0)),
        far_background_dilation=int(loss_cfg.get("far_background_dilation", 3)),
        far_background_gamma=float(loss_cfg.get("far_background_gamma", 1.5)),
        background_blob_weight=float(loss_cfg.get("background_blob_weight", 0.0)),
        background_blob_dilation=int(loss_cfg.get("background_blob_dilation", 3)),
        background_blob_pool_size=int(loss_cfg.get("background_blob_pool_size", 9)),
        background_blob_margin=float(loss_cfg.get("background_blob_margin", 0.06)),
        background_blob_gamma=float(loss_cfg.get("background_blob_gamma", 2.0)),
        aux_tversky_weight=float(loss_cfg.get("aux_tversky_weight", 0.0)),
        aux_far_background_weight=float(loss_cfg.get("aux_far_background_weight", 0.0)),
        aux_far_background_dilation=loss_cfg.get("aux_far_background_dilation"),
        aux_far_background_gamma=loss_cfg.get("aux_far_background_gamma"),
        mix_balance_weight=float(loss_cfg.get("mix_balance_weight", 0.0)),
        mix_prior=loss_cfg.get("mix_prior"),
        separation_weight=float(loss_cfg.get("separation_weight", loss_cfg.get("sep_weight", 0.0))),
        uncertainty_prior_weight=float(loss_cfg.get("uncertainty_prior_weight", 0.0)),
        uncertainty_prior=float(loss_cfg.get("uncertainty_prior", 0.35)),
        hf_residual_weight=float(loss_cfg.get("hf_residual_weight", 0.0)),
        hf_gate_sparsity_weight=float(loss_cfg.get("hf_gate_sparsity_weight", 0.0)),
        continuity_residual_weight=float(loss_cfg.get("continuity_residual_weight", 0.0)),
        continuity_gate_sparsity_weight=float(loss_cfg.get("continuity_gate_sparsity_weight", 0.0)),
        continuity_isolated_weight=float(loss_cfg.get("continuity_isolated_weight", 0.0)),
        continuity_isolated_dilation=loss_cfg.get("continuity_isolated_dilation"),
        continuity_isolated_gamma=float(loss_cfg.get("continuity_isolated_gamma", 1.5)),
        continuity_support_weight=float(loss_cfg.get("continuity_support_weight", 0.0)),
        continuity_support_target=float(loss_cfg.get("continuity_support_target", 0.14)),
        continuity_support_dilation=int(loss_cfg.get("continuity_support_dilation", 1)),
        continuity_support_gamma=float(loss_cfg.get("continuity_support_gamma", 1.0)),
        calibration_weight=float(loss_cfg.get("calibration_weight", 0.0)),
        calibration_edge_margin=float(loss_cfg.get("calibration_edge_margin", 0.55)),
        calibration_background_margin=float(loss_cfg.get("calibration_background_margin", 0.04)),
        calibration_background_dilation=int(loss_cfg.get("calibration_background_dilation", 3)),
        calibration_edge_weight=float(loss_cfg.get("calibration_edge_weight", 0.35)),
        calibration_background_weight=float(loss_cfg.get("calibration_background_weight", 1.0)),
        active_density_weight=float(loss_cfg.get("active_density_weight", 0.0)),
        active_density_min_multiplier=float(loss_cfg.get("active_density_min_multiplier", 0.35)),
        active_density_max_multiplier=float(loss_cfg.get("active_density_max_multiplier", 1.45)),
        active_density_floor=float(loss_cfg.get("active_density_floor", 0.002)),
        active_density_threshold=float(loss_cfg.get("active_density_threshold", 0.5)),
        active_density_temperature=float(loss_cfg.get("active_density_temperature", 0.06)),
        state_supervision_weight=float(loss_cfg.get("state_supervision_weight", 0.0)),
        state_edge_dilation=int(loss_cfg.get("state_edge_dilation", 1)),
        state_suppress_dilation=int(loss_cfg.get("state_suppress_dilation", 5)),
        target_threshold=float(loss_cfg.get("target_threshold", 0.01)),
    )

    eval_cfg = config.get("eval", {})
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        visual_dir=run_paths.visualizations if eval_cfg.get("save_visualizations", False) else None,
        pred_dir=run_paths.predictions if eval_cfg.get("save_predictions", False) else None,
        gate_dir=run_paths.gate_heatmaps if eval_cfg.get("save_gate_heatmaps", False) else None,
        max_visual_samples=int(eval_cfg.get("max_visual_samples", 32)),
        metric_mode=str(eval_cfg.get("metric_mode", "fast_gpu")),
        fast_metric_thresholds=int(eval_cfg.get("fast_metric_thresholds", 49)),
        fast_metric_tolerance_pixels=int(eval_cfg.get("fast_metric_tolerance_pixels", 4)),
        apply_nms=bool(eval_cfg.get("apply_nms", False)),
        nms_low_threshold=float(eval_cfg.get("nms_low_threshold", 0.0)),
        diagnostic_interval=int(eval_cfg.get("diagnostic_interval", 10)),
    )
    row = {"checkpoint": str(checkpoint_path), "dataset": config["dataset"]["eval_dataset"], **metrics}
    append_metrics_csv(run_paths.metrics / "eval_metrics.csv", row)
    print(row)


if __name__ == "__main__":
    main(parse_args())
