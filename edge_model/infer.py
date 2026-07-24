"""Run explicit Anchor or fixed-candidate H-RBCM inference on image folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.checkpoint_io import load_checkpoint
from edge_model.core.config import load_config, project_path
from edge_model.data.transforms import make_eval_transform
from edge_model.engine.train_loop import restore_sample_array
from edge_model.engine.visualize import save_probability_map, save_signed_array_heatmap
from edge_model.models.build import build_model
from scripts.experiments.calibrate import apply_candidate, edge_energy_from_image, logit
from scripts.experiments.evaluate_generalization import load_candidates

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "rbcm" / "biped.yaml",
    "checkpoint": None,
    "image_dir": Path("edge_data/official_rbcm/BIPED/image"),
    "output_dir": Path("results/rbcm/inference"),
    "input_size": 300,
    "device": "cuda",
    "mode": "plain_identity",
}


def parse_args() -> argparse.Namespace:
    """Parse inference arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARGS["checkpoint"])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_ARGS["image_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_ARGS["input_size"])
    parser.add_argument("--device", default=DEFAULT_ARGS["device"])
    parser.add_argument(
        "--mode",
        choices=["plain_identity", "main_surround", "no_surround", "conv_control"],
        default=DEFAULT_ARGS["mode"],
        help="Prediction function to export. plain_identity is the uncalibrated HED-lite Anchor.",
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        type=Path,
        default=[],
        help="Validation-selected candidate table; required for every non-identity mode.",
    )
    parser.add_argument("--candidate-split", default="val")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run inference and save the explicitly requested probability maps."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    if args.checkpoint is None:
        raise SystemExit("Please provide --checkpoint or edit DEFAULT_ARGS['checkpoint'].")

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = load_checkpoint(project_path(config, args.checkpoint), map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    image_dir = project_path(config, args.image_dir)
    output_dir = project_path(config, args.output_dir)
    pred_dir = output_dir / "predictions"
    probability_delta_dir = output_dir / "probability_delta"
    logit_modulation_dir = output_dir / "logit_modulation"
    pred_dir.mkdir(parents=True, exist_ok=True)
    probability_delta_dir.mkdir(parents=True, exist_ok=True)
    logit_modulation_dir.mkdir(parents=True, exist_ok=True)

    candidates = {}
    if args.mode != "plain_identity":
        if not args.candidate_csv:
            raise SystemExit(f"--candidate-csv is required for --mode {args.mode}.")
        candidate_paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in args.candidate_csv]
        candidates = load_candidates(candidate_paths, [str(args.mode)], str(args.candidate_split))

    dataset_cfg = config.get("dataset", {})
    transform = make_eval_transform(
        input_size=args.input_size,
        preserve_aspect=bool(dataset_cfg.get("preserve_aspect_eval", True)),
        binarize_edges=bool(dataset_cfg.get("binarize_edges", True)),
        return_meta=True,
    )
    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    image_paths = sorted(image_paths)
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            dummy_edge = Image.new("L", image.size, 0)
            image_tensor, _, meta = transform(image, dummy_edge)
            outputs = model(image_tensor.unsqueeze(0).to(device))
            meta_batch = {key: torch.tensor([value]) for key, value in meta.items()}
            anchor_probability = restore_sample_array(
                torch.sigmoid(outputs["logits"])[0, 0].cpu().numpy(),
                meta_batch,
                idx=0,
                is_target=False,
            )
            probability = np.asarray(anchor_probability, dtype=np.float32)
            if args.mode != "plain_identity":
                sample = {
                    "sample_id": image_path.stem,
                    "prob": probability,
                    "edge": edge_energy_from_image(image_path),
                }
                probability = apply_candidate(sample, candidates[str(args.mode)])
            save_probability_map(probability, pred_dir / f"{image_path.stem}.png")
            probability_delta = probability - anchor_probability
            logit_modulation = logit(probability) - logit(anchor_probability)
            save_signed_array_heatmap(
                probability_delta,
                probability_delta_dir / f"{image_path.stem}.png",
                label=f"{args.mode}: probability - Anchor",
            )
            save_signed_array_heatmap(
                logit_modulation,
                logit_modulation_dir / f"{image_path.stem}.png",
                label=f"{args.mode}: logit modulation",
            )
            print(f"Saved {image_path.stem}")

    manifest = {
        "mode": str(args.mode),
        "checkpoint": str(project_path(config, args.checkpoint)),
        "config": str(config_path),
        "image_dir": str(image_dir),
        "input_size": int(args.input_size),
        "candidate_csv": [str(path) for path in args.candidate_csv],
        "candidate_split": str(args.candidate_split),
        "n_images": len(image_paths),
        "diagnostics": {
            "probability_delta": "final probability minus HED-lite Anchor probability",
            "logit_modulation": "logit(final probability) minus logit(Anchor probability)",
        },
        "note": (
            "plain_identity exports the HED-lite Anchor; non-identity modes apply a fixed "
            "validation-selected calibration candidate to the restored Anchor probability map."
        ),
    }
    (output_dir / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
