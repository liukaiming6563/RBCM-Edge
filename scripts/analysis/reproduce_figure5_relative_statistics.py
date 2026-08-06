#!/usr/bin/env python3
"""Reproduce the V5 Figure 5 relative-modulation source tables.

The script uses the frozen strict MultiCue H-RBCM candidate and Anchor
probability maps. It computes the pure surround-dependent term
``delta_RBCM = alpha * U * C`` and applies the same within-sample transform to
MEA group-direction effects and model-image effects:

``z = (d - mean(d)) / (median(abs(d - mean(d))) + epsilon)``.

Relative enhancement, suppression, and near-neutral states are defined by
``z > 1``, ``z < -1``, and ``-1 <= z <= 1``. Ground truth is not used for this
state classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.calibrate import (  # noqa: E402
    contrast,
    edge_energy_from_image,
    local_surround,
    parse_ring,
)


DATASET_ORDER = ("BIPED", "Multicue", "NYUDv2", "UDED")
EXPECTED_MODEL_COUNTS = {
    "BIPED": 50,
    "Multicue": 20,
    "NYUDv2": 654,
    "UDED": 30,
}
GRID_SIZES = (12, 16)
THRESHOLD = 1.0
EPSILON = 1.0e-8
PROBABILITY_EPSILON = 1.0e-4
STATE_COLUMNS = (
    "enhance_fraction",
    "suppress_fraction",
    "neutral_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mea-grid-source",
        type=Path,
        default=(
            ROOT
            / "MEA_outputs/grid_full_analysis/tables/"
            "grid_cell_level_results_all_scales_extended.csv"
        ),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=(
            ROOT
            / "edge_outputs/rbcm/predictions/"
            "multicue_strict_seed4517_generalization5/apply"
        ),
        help=(
            "Directory containing <dataset>/predictions/plain_identity/*.png "
            "from the frozen strict MultiCue checkpoint."
        ),
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=ROOT / "pretrained/multicue_strict/calibration_candidates.csv",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "edge_data/official_rbcm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "edge_outputs/rbcm/analyses/mea_rbcm_bridge/"
            "figure5_relative"
        ),
    )
    return parser.parse_args()


def load_main_candidate(path: Path) -> dict[str, float | str]:
    candidates = [path]
    if path.name == "calibration_candidates.csv":
        candidates.append(path.with_name("fixed_candidates.csv"))
    existing = next((candidate for candidate in candidates if candidate.is_file()), None)
    if existing is None:
        raise FileNotFoundError(
            "No frozen strict MultiCue candidate table found; checked "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    with existing.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("mode") == "main_surround"
            and row.get("split", "val") == "val"
        ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one frozen validation main_surround row in {existing}, found {len(rows)}"
        )
    row = rows[0]
    return {
        "ring": str(row["ring"]),
        "alpha": float(row["alpha"]),
        "edge_weight": float(row["edge_weight"]),
        "prob_weight": float(row["prob_weight"]),
        "uncertainty_power": float(row["uncertainty_power"]),
    }


def mean_mad_relative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    center = float(values.mean())
    scale = float(np.median(np.abs(values - center)))
    return (values - center) / max(scale, EPSILON)


def signed_balance(enhance: float, suppress: float) -> float:
    denominator = enhance + suppress
    if denominator <= 0.0:
        return 0.0
    return float((suppress - enhance) / denominator)


def state_row(values: np.ndarray, **metadata: object) -> dict[str, object]:
    z = mean_mad_relative(values)
    enhance = float((z > THRESHOLD).mean())
    suppress = float((z < -THRESHOLD).mean())
    neutral = float(1.0 - enhance - suppress)
    return {
        **metadata,
        "method": "mean_mad_relative",
        "threshold_multiplier": THRESHOLD,
        "n_elements": int(z.size),
        "enhance_fraction": enhance,
        "suppress_fraction": suppress,
        "neutral_fraction": neutral,
        "signed_balance_index": signed_balance(enhance, suppress),
    }


def build_mea_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    valid = frame["valid_grid"].astype(str).str.lower().eq("true")
    selected = frame.loc[
        valid & frame["grid_n"].astype(int).isin(GRID_SIZES)
    ].copy()
    # The source table stores UME - CME; V5 defines d_MEA = FR_CME - FR_UME.
    selected["signed_effect"] = -pd.to_numeric(
        selected["delta_mean_fr_hz"], errors="coerce"
    )
    selected = selected[np.isfinite(selected["signed_effect"])].copy()
    rows: list[dict[str, object]] = []
    for (pair_id, direction_code), group in selected.groupby(
        ["pair_id", "direction_code"], sort=True
    ):
        rows.append(
            state_row(
                group["signed_effect"].to_numpy(dtype=np.float32),
                system="MEA",
                pair_id=str(pair_id),
                direction_code=str(direction_code),
            )
        )
    if len(rows) != 24:
        raise RuntimeError(f"Expected 3 groups x 8 directions, found {len(rows)} rows")
    return pd.DataFrame(rows)


def load_probability(path: Path) -> np.ndarray:
    probability = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def find_input_image(image_root: Path, dataset: str, prediction_name: str) -> Path:
    image_dir = image_root / dataset / "image"
    matches = sorted(image_dir.glob(f"{Path(prediction_name).stem}.*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one image for {dataset}/{prediction_name}, found {len(matches)}"
        )
    return matches[0]


def pure_rbcm_delta(
    probability: np.ndarray,
    image_path: Path,
    candidate: dict[str, float | str],
) -> np.ndarray:
    edge = edge_energy_from_image(image_path)
    if edge.shape != probability.shape:
        edge = np.asarray(
            Image.fromarray(edge.astype(np.float32)).resize(
                (probability.shape[1], probability.shape[0]),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    ring = parse_ring(str(candidate["ring"]))
    probability_state = contrast(
        probability,
        local_surround(probability, "main_surround", ring),
    )
    edge_state = contrast(edge, local_surround(edge, "main_surround", ring))
    context = np.clip(
        float(candidate["edge_weight"]) * edge_state
        + float(candidate["prob_weight"]) * probability_state,
        -1.0,
        1.0,
    )
    uncertainty = np.clip(4.0 * probability * (1.0 - probability), 0.0, 1.0)
    uncertainty = np.power(uncertainty, float(candidate["uncertainty_power"]))
    return (float(candidate["alpha"]) * uncertainty * context).astype(np.float32)


def build_model_rows(
    prediction_root: Path,
    image_root: Path,
    candidate: dict[str, float | str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        plain_root = prediction_root / dataset / "predictions" / "plain_identity"
        prediction_paths = sorted(plain_root.glob("*.png"))
        expected = EXPECTED_MODEL_COUNTS[dataset]
        if len(prediction_paths) != expected:
            raise RuntimeError(
                f"Expected {expected} Anchor predictions for {dataset}, found {len(prediction_paths)}"
            )
        for probability_path in prediction_paths:
            probability = load_probability(probability_path)
            image_path = find_input_image(
                image_root, dataset, probability_path.name
            )
            delta = pure_rbcm_delta(probability, image_path, candidate)
            rows.append(
                state_row(
                    delta,
                    system="H-RBCM",
                    dataset=dataset,
                    image=probability_path.name,
                    signed_effect="pure_rbcm_delta_alpha_u_c",
                )
            )
    return pd.DataFrame(rows)


def composition(frame: pd.DataFrame) -> dict[str, float]:
    return {column: float(frame[column].mean()) for column in STATE_COLUMNS}


def balance(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["signed_balance_index"].to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    mea = build_mea_rows(args.mea_grid_source.resolve())
    candidate = load_main_candidate(args.candidate_csv.resolve())
    model = build_model_rows(
        args.prediction_root.resolve(),
        args.image_root.resolve(),
        candidate,
    )
    model_dataset = (
        model.groupby("dataset", as_index=False)[list(STATE_COLUMNS)]
        .mean()
        .set_index("dataset")
        .reindex(DATASET_ORDER)
        .reset_index()
    )
    model_uded = model.loc[model["dataset"] == "UDED"].copy()
    summary = {
        "figure": "Figure 5",
        "manuscript_version": "V5",
        "definition": {
            "method": "mean_mad_relative",
            "transform": "z = (d - mean(d)) / (median(abs(d - mean(d))) + epsilon)",
            "threshold_multiplier": THRESHOLD,
            "mea_signed_effect": "FR_CME - FR_UME",
            "model_signed_effect": "pure RBCM term delta_RBCM = alpha * U * C",
            "ground_truth_used_for_state_classification": False,
        },
        "counts": {
            "mea_group_direction_observations": 24,
            "model_images_by_dataset": EXPECTED_MODEL_COUNTS,
            "model_uded_images_for_balance_panel": 30,
        },
        "panel_a": {
            "mea_mean_composition": composition(mea),
            "model_equal_dataset_mean_composition": {
                column: float(model_dataset[column].mean())
                for column in STATE_COLUMNS
            },
        },
        "panel_b": {
            "mea_signed_balance": balance(mea),
            "model_uded_signed_balance": balance(model_uded),
            "formula": "B = (N_suppress - N_enhance) / (N_suppress + N_enhance)",
        },
        "interpretation_boundary": (
            "Normalized within-sample relative heterogeneity only; not equality "
            "of raw distributions or neuron-to-pixel correspondence."
        ),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mea.to_csv(output / "mea_relative_rows.csv", index=False)
    model.to_csv(output / "model_relative_rows.csv", index=False)
    model_dataset.to_csv(output / "model_dataset_summary.csv", index=False)
    (output / "figure5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
