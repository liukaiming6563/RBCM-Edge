"""Apply fixed HED-lite post-hoc surround candidates to arbitrary test datasets.

This script is intentionally different from ``calibrate.py``:
it does not tune candidates on the target dataset.  Instead, it loads candidate
parameters selected elsewhere and applies them unchanged to a requested split.
That makes it suitable for cross-dataset generalization checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_model.engine.visualize import save_probability_map
from edge_model.models.build import build_model
from scripts.experiments.calibrate import (
    Candidate,
    apply_candidate,
    collect_samples,
    load_config_for_eval,
    metric_for_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "Explicit split file for the requested target dataset. This "
            "overrides a source config's split mapping and prevents a "
            "source-specific split from leaking into cross-dataset inference."
        ),
    )
    parser.add_argument("--edge-data-root", default="edge_data/official_rbcm")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional target-split cap for smoke tests only.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--thresholds", type=int, default=49)
    parser.add_argument("--match-tolerance", type=float, default=0.0075)
    parser.add_argument("--gt-threshold", type=float, default=0.0)
    parser.add_argument("--eval-gt-variant", default="edge")
    parser.add_argument("--eval-gt-mode", choices=["binary", "soft"], default="binary")
    parser.add_argument("--nms-low-threshold", type=float, default=0.02)
    parser.add_argument("--modes", nargs="+", default=["main_surround", "no_surround", "conv_control"])
    parser.add_argument("--candidate-split", default="val")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--skip-internal-metrics",
        action="store_true",
        help="Only write fixed-candidate predictions; official saved-prediction evaluation is run separately.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_candidates(paths: list[Path], modes: list[str], split: str) -> dict[str, Candidate]:
    wanted = set(modes)
    selected: dict[str, Candidate] = {}
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mode = str(row.get("mode", ""))
                row_split = str(row.get("split", ""))
                if mode not in wanted or row_split != split or mode in selected:
                    continue
                selected[mode] = Candidate(
                    mode=mode,
                    ring=str(row["ring"]),
                    alpha=float(row["alpha"]),
                    edge_weight=float(row["edge_weight"]),
                    prob_weight=float(row["prob_weight"]),
                    uncertainty_power=float(row["uncertainty_power"]),
                    temperature=float(row["temperature"]),
                    bias=float(row["bias"]),
                    sharpen=float(row["sharpen"]),
                )
    missing = [mode for mode in modes if mode not in selected]
    if missing:
        raise ValueError(f"Missing fixed candidates for modes {missing} from {paths} split={split}")
    return selected


def save_probs(samples: list[dict[str, np.ndarray | str]], probs: list[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample, prob in zip(samples, probs, strict=True):
        save_one_prob(sample, prob, output_dir)


def save_one_prob(sample: dict[str, np.ndarray | str], prob: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(sample["sample_id"])
    stem = Path(sample_id).stem
    save_probability_map(np.asarray(prob, dtype=np.float32), output_dir / f"{stem}.png")


def clear_transient_state(sample: dict[str, np.ndarray | str]) -> None:
    for key in list(sample.keys()):
        if key.startswith("_"):
            del sample[key]


def evaluate_probs(
    probs: list[np.ndarray],
    samples: list[dict[str, np.ndarray | str]],
    args: argparse.Namespace,
) -> dict[str, float]:
    targets = [sample["target"] for sample in samples]
    return metric_for_probs(
        probs,
        targets,  # type: ignore[arg-type]
        thresholds=int(args.thresholds),
        match_tolerance=float(args.match_tolerance),
        nms_low_threshold=float(args.nms_low_threshold),
        apply_nms=True,
        device=str(args.device),
        gt_threshold=float(args.gt_threshold),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config_for_eval(args.config, checkpoint, args)
    config.setdefault("paths", {})["edge_data_root"] = str(args.edge_data_root)
    if args.input_size is not None:
        config.setdefault("dataset", {})["input_size"] = int(args.input_size)
    if args.max_samples is not None:
        config.setdefault("dataset", {})["max_eval_samples"] = int(args.max_samples)
    if args.split_file is not None:
        config.setdefault("dataset", {}).setdefault("split_files", {})[str(args.split)] = str(args.split_file)
    gt_config = config.setdefault("dataset", {}).setdefault("gt", {})
    # A dataset-specific rule has higher priority than ``eval_variant`` in the
    # dataset factory.  Update the requested dataset explicitly so CLI GT
    # overrides cannot be silently shadowed by a checkpoint's old mapping.
    dataset_variants = gt_config.setdefault("dataset_variants", {})
    if not isinstance(dataset_variants, dict):
        dataset_variants = {}
        gt_config["dataset_variants"] = dataset_variants
    dataset_variants[str(args.dataset_name)] = str(args.eval_gt_variant)
    gt_config["eval_variant"] = str(args.eval_gt_variant)
    gt_config["eval_mode"] = str(args.eval_gt_mode)
    gt_config["binarize_eval_edges"] = str(args.eval_gt_mode) == "binary"

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    samples = collect_samples(
        model=model,
        config=config,
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        device=device,
    )

    candidates = load_candidates(args.candidate_csv, list(args.modes), str(args.candidate_split))
    rows: list[dict[str, object]] = []

    plain_probs = [sample["prob"] for sample in samples]
    plain_metrics = {} if args.skip_internal_metrics else evaluate_probs(plain_probs, samples, args)  # type: ignore[arg-type]
    rows.append(
        {
            "source_label": args.source_label,
            "dataset": args.dataset_name,
            "split": args.split,
            "mode": "plain_identity",
            "metrics_skipped": bool(args.skip_internal_metrics),
            **plain_metrics,
        }
    )
    if args.save_predictions:
        if args.skip_internal_metrics:
            for sample in samples:
                save_one_prob(sample, sample["prob"], args.output_dir / "predictions" / "plain_identity")  # type: ignore[arg-type]
        else:
            save_probs(samples, plain_probs, args.output_dir / "predictions" / "plain_identity")  # type: ignore[arg-type]

    selected_report: dict[str, dict[str, object]] = {}
    for mode, candidate in candidates.items():
        if args.skip_internal_metrics:
            probs = []
            metrics = {}
            if args.save_predictions:
                for sample in samples:
                    prob = apply_candidate(sample, candidate)
                    save_one_prob(sample, prob, args.output_dir / "predictions" / mode)
                    clear_transient_state(sample)
            else:
                for sample in samples:
                    _ = apply_candidate(sample, candidate)
                    clear_transient_state(sample)
        else:
            probs = [apply_candidate(sample, candidate) for sample in samples]
            metrics = evaluate_probs(probs, samples, args)
        rows.append(
            {
                "source_label": args.source_label,
                "dataset": args.dataset_name,
                "split": args.split,
                **asdict(candidate),
                "metrics_skipped": bool(args.skip_internal_metrics),
                **metrics,
            }
        )
        selected_report[mode] = asdict(candidate)
        if args.save_predictions and not args.skip_internal_metrics:
            save_probs(samples, probs, args.output_dir / "predictions" / mode)

    write_csv(args.output_dir / "summary.csv", rows)
    report = {
        "source_label": args.source_label,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "candidate_csv": [str(path) for path in args.candidate_csv],
        "candidate_split": args.candidate_split,
        "n_samples": len(samples),
        "selected_candidates": selected_report,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
