"""Near-official Python edge evaluation for official baseline reproduction.

The exact BSDS/PiDiNet protocol is Matlab based. This script keeps the same
diagnostic spirit in Python: original image sizes, BSDS-style localization
tolerance, threshold sweep, ODS/OIS/AP, optional NMS, and explicit prediction
orientation checks. Use it to detect protocol/data failures before spending
time on RBCM changes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.engine.metrics import (
    average_precision_from_curve,
    edge_metrics_from_arrays,
    nms_probabilities,
    threshold_curve_from_arrays,
)


try:
    import scipy.io as sio
except ModuleNotFoundError:  # pragma: no cover - server/PiDiNet env should have scipy
    # The release bundle keeps optional compiled evaluation dependencies under
    # ``.local_pkgs``.  Append (rather than prepend) it so a CUDA-enabled torch
    # installation in the active environment is never shadowed by the CPU
    # fallback bundled for document/release checks.
    local_packages = PROJECT_ROOT / ".local_pkgs"
    if local_packages.exists():
        sys.path.append(str(local_packages))
        try:
            import scipy.io as sio
        except ModuleNotFoundError:
            sio = None
    else:
        sio = None

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - server env installs cv2 for TEED
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["BIPED", "BSDS500", "Multicue", "NYUDv2", "UDED"])
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--biped-root", type=Path, default=PROJECT_ROOT / "edge_data" / "official_repro" / "BIPED")
    parser.add_argument("--bsds-root", type=Path, default=PROJECT_ROOT / "edge_data" / "official_repro" / "BSDS500")
    parser.add_argument("--multicue-root", type=Path, default=PROJECT_ROOT / "edge_data" / "official_rbcm" / "Multicue")
    parser.add_argument("--nyud-root", type=Path, default=PROJECT_ROOT / "edge_data" / "official_rbcm" / "NYUDv2")
    parser.add_argument("--uded-root", type=Path, default=PROJECT_ROOT / "edge_data" / "official_repro" / "UDED")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit evaluation split. For MultiCue and NYUDv2 "
            "paper-facing runs, pass the strict test list instead of relying "
            "on a dataset root's top-level splits/test.txt."
        ),
    )
    parser.add_argument(
        "--orientation",
        choices=["auto", "as_is", "inverted"],
        default="as_is",
        help=(
            "Prediction polarity. Paper-facing evaluation must use a fixed value "
            "(normally as_is). 'auto' compares both polarities on the target GT "
            "and is diagnostic only."
        ),
    )
    parser.add_argument("--apply-nms", action="store_true")
    parser.add_argument("--nms-low-threshold", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--thresholds", type=int, default=99)
    parser.add_argument("--match-tolerance", type=float, default=0.0075)
    parser.add_argument("--skip-dense-orientation", type=float, default=0.35)
    parser.add_argument(
        "--metric-backend",
        choices=["dilation", "greedy_one_to_one", "strict_kdtree"],
        default="dilation",
    )
    parser.add_argument("--gt-threshold", type=float, default=0.0)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Permit incomplete prediction sets. Paper-facing evaluation fails on missing samples by default.",
    )
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="Permit prediction stems outside the requested evaluation split. Paper-facing evaluation fails by default.",
    )
    return parser.parse_args()


def normalize_image_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        if sio is None:
            raise ModuleNotFoundError("scipy is required to read .mat predictions")
        mat = sio.loadmat(path)
        if "img" in mat:
            arr = np.asarray(mat["img"], dtype=np.float32)
        else:
            candidates = [value for key, value in mat.items() if not key.startswith("__") and np.asarray(value).ndim >= 2]
            if not candidates:
                raise ValueError(f"No image-like variable in {path}")
            arr = np.asarray(candidates[0], dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)
    return normalize_image_array(path)


def collect_predictions(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    priority = {".mat": 0, ".png": 1, ".jpg": 2, ".jpeg": 2}
    chosen: dict[str, tuple[int, Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in priority:
            continue
        rel_parts = {part.lower() for part in path.relative_to(root).parts}
        if "all_edges" in rel_parts:
            continue
        key = path.stem
        score = priority[suffix]
        if key in chosen and score == chosen[key][0] and path.resolve() != chosen[key][1].resolve():
            raise RuntimeError(
                f"Ambiguous duplicate predictions for sample {key!r}: "
                f"{chosen[key][1]} and {path}"
            )
        if key not in chosen or score < chosen[key][0]:
            chosen[key] = (score, path)
    return {key: value[1] for key, value in chosen.items()}


def validate_samples(samples: list[tuple[str, Path]]) -> None:
    stems = [stem for stem, _ in samples]
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate evaluation sample IDs: {duplicates[:20]}")
    missing_gt = [str(path) for _, path in samples if not path.exists()]
    if missing_gt:
        raise FileNotFoundError(f"Missing ground-truth files ({len(missing_gt)}): {missing_gt[:20]}")


def biped_samples(root: Path, split_file: Path | None = None) -> list[tuple[str, Path]]:
    if split_file is not None:
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        stems = [
            line.strip().split()[0]
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [(stem, root / "edge" / f"{stem}.png") for stem in stems]

    pair_file = root / "test_pair.lst"
    if pair_file.exists():
        pairs = json.loads(pair_file.read_text(encoding="utf-8"))
        return [(Path(img).stem, root / gt) for img, gt in pairs]

    default_split = root / "splits" / "test.txt"
    if not default_split.exists():
        raise FileNotFoundError(
            f"BIPED requires --split-file, {pair_file}, or {default_split}"
        )
    stems = [
        line.strip().split()[0]
        for line in default_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [(stem, root / "edge" / f"{stem}.png") for stem in stems]


def _iter_bsds_boundaries(raw: object) -> Iterable[np.ndarray]:
    arr = np.asarray(raw).squeeze()
    for item in arr.flat:
        if hasattr(item, "Boundaries"):
            yield np.asarray(item.Boundaries, dtype=np.float32)
            continue
        obj = np.asarray(item).squeeze()
        if obj.dtype.names and "Boundaries" in obj.dtype.names:
            yield np.asarray(obj["Boundaries"].item(), dtype=np.float32)


def load_bsds_gt(path: Path) -> np.ndarray:
    if sio is None:
        fallback = PROJECT_ROOT / "edge_data" / "official_rbcm" / "BSDS500" / "edge" / f"{path.stem}.png"
        if fallback.exists():
            return normalize_image_array(fallback)
        raise ModuleNotFoundError(
            "scipy is required to read BSDS500 .mat ground-truth files; "
            f"fallback PNG not found: {fallback}"
        )
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "groundTruth" not in mat:
        raise ValueError(f"Missing groundTruth in {path}")
    maps = [boundary for boundary in _iter_bsds_boundaries(mat["groundTruth"])]
    if not maps:
        raise ValueError(f"No Boundaries fields found in {path}")
    base_shape = maps[0].shape
    maps = [m if m.shape == base_shape else np.asarray(Image.fromarray(m).resize(base_shape[::-1])) for m in maps]
    gt = np.mean(np.stack([(m > 0).astype(np.float32) for m in maps], axis=0), axis=0)
    return np.clip(gt, 0.0, 1.0)


def bsds_samples(root: Path) -> list[tuple[str, Path]]:
    list_path = root / "HED-BSDS" / "test.lst"
    stems = [Path(line.strip()).stem for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(stem, root / "groundTruth" / "test" / f"{stem}.mat") for stem in stems]


def multicue_samples(root: Path, split_file: Path | None = None) -> list[tuple[str, Path]]:
    split_path = split_file if split_file is not None else root / "splits" / "test.txt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    stems = [line.strip().split()[0] for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(stem, root / "gt" / "soft_vote" / f"{stem}.png") for stem in stems]


def nyud_samples(root: Path, split_file: Path | None = None) -> list[tuple[str, Path]]:
    split_path = split_file if split_file is not None else root / "splits" / "test.txt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    stems = [line.strip().split()[0] for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    soft_dir = root / "gt" / "soft_vote"
    gt_dir = soft_dir if soft_dir.exists() else root / "edge"
    return [(stem, gt_dir / f"{stem}.png") for stem in stems]


def uded_samples(root: Path) -> list[tuple[str, Path]]:
    """Return UDED samples from the TEED-style test pair list."""
    list_path = root / "test_pair.lst"
    if not list_path.exists():
        raise FileNotFoundError(list_path)
    text = list_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    if text[0] == "[":
        import json

        raw = json.loads(text)
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
    else:
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return [(Path(image_rel).stem, root / gt_rel) for image_rel, gt_rel in pairs]


def resize_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr.astype(np.float32)
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.asarray(img.resize((shape[1], shape[0]), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def maybe_nms(arrays: list[np.ndarray], *, device_name: str, low_threshold: float) -> list[np.ndarray]:
    if not arrays:
        return []
    import torch

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    out = []
    for arr in arrays:
        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32).view(1, 1, *arr.shape)
        thinned = nms_probabilities(tensor, low_threshold=low_threshold)
        out.append(thinned.squeeze().detach().cpu().numpy().astype(np.float32))
    return out


def tolerance_radius(shape: tuple[int, int], match_tolerance: float) -> int:
    height, width = shape
    if match_tolerance < 1.0:
        value = float(match_tolerance) * float(np.hypot(height, width))
    else:
        value = float(match_tolerance)
    return max(1, int(round(value)))


def dilation_metrics_from_arrays(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    thresholds: np.ndarray,
    match_tolerance: float,
    gt_threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if cv2 is None:
        raise ModuleNotFoundError("cv2 is required for dilation metric backend")

    agg = []
    per_image_best = []
    for threshold in thresholds:
        agg.append({"threshold": float(threshold), "mp": 0.0, "pc": 0.0, "mt": 0.0, "tc": 0.0})

    for prob, target in zip(probabilities, targets):
        gt = np.asarray(target > gt_threshold, dtype=np.uint8)
        if gt.sum() == 0:
            continue
        radius = tolerance_radius(gt.shape, match_tolerance)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        gt_dil = cv2.dilate(gt, kernel) > 0
        image_rows = []
        for index, threshold in enumerate(thresholds):
            pred = np.asarray(prob >= float(threshold), dtype=np.uint8)
            pred_count = float(pred.sum())
            truth_count = float(gt.sum())
            if pred_count > 0.0:
                matched_pred = float(((pred > 0) & gt_dil).sum())
                pred_dil = cv2.dilate(pred, kernel) > 0
                matched_truth = float(((gt > 0) & pred_dil).sum())
            else:
                matched_pred = 0.0
                matched_truth = 0.0
            row = agg[index]
            row["mp"] += matched_pred
            row["pc"] += pred_count
            row["mt"] += matched_truth
            row["tc"] += truth_count
            precision = matched_pred / max(pred_count, 1.0)
            recall = matched_truth / max(truth_count, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            image_rows.append(f1)
        per_image_best.append(max(image_rows) if image_rows else 0.0)

    curve = []
    for row in agg:
        precision = row["mp"] / max(row["pc"], 1.0)
        recall = row["mt"] / max(row["tc"], 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        curve.append(
            {
                "threshold": row["threshold"],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": float(min(row["mp"], row["mt"])),
                "fp": float(max(row["pc"] - row["mp"], 0.0)),
                "fn": float(max(row["tc"] - row["mt"], 0.0)),
            }
        )
    best = max(curve, key=lambda item: item["f1"]) if curve else {}
    metrics = {
        "ODS": float(best.get("f1", 0.0)),
        "OIS": float(np.mean(per_image_best)) if per_image_best else 0.0,
        "AP": average_precision_from_curve(curve),
        "ODS_threshold": float(best.get("threshold", 0.0)),
        "precision_at_ODS": float(best.get("precision", 0.0)),
        "recall_at_ODS": float(best.get("recall", 0.0)),
        "metric_backend": "dilation",
    }
    return metrics, curve


def evaluate_orientation(
    probs: list[np.ndarray],
    targets: list[np.ndarray],
    orientation: str,
    thresholds: np.ndarray,
    tolerance: float,
    *,
    apply_nms: bool,
    device: str,
    nms_low_threshold: float,
    skip_dense_orientation: float,
    metric_backend: str,
    gt_threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    oriented = [1.0 - p if orientation == "inverted" else p for p in probs]
    if apply_nms:
        oriented = maybe_nms(oriented, device_name=device, low_threshold=nms_low_threshold)
    density_at_half = float(np.mean([(arr >= 0.5).mean() for arr in oriented]))
    if density_at_half > skip_dense_orientation:
        return (
            {
                "ODS": 0.0,
                "OIS": 0.0,
                "AP": 0.0,
                "ODS_threshold": 0.0,
                "precision_at_ODS": 0.0,
                "recall_at_ODS": 0.0,
                "orientation": orientation,
                "n_images": len(oriented),
                "pred_edge_density_at_0_5": density_at_half,
                "skipped_dense_orientation": True,
            },
            [],
        )
    if metric_backend in {"greedy_one_to_one", "strict_kdtree"}:
        binary_targets = [
            np.asarray(target > float(gt_threshold), dtype=np.float32)
            for target in targets
        ]
        metrics = edge_metrics_from_arrays(
            oriented,
            binary_targets,
            thresholds=thresholds,
            match_tolerance=tolerance,
        )
        curve = threshold_curve_from_arrays(
            oriented,
            binary_targets,
            thresholds=thresholds,
            match_tolerance=tolerance,
        )
        metrics["metric_backend"] = "greedy_one_to_one"
    else:
        metrics, curve = dilation_metrics_from_arrays(oriented, targets, thresholds, tolerance, gt_threshold)
    metrics["orientation"] = orientation
    metrics["n_images"] = len(oriented)
    metrics["pred_edge_density_at_0_5"] = density_at_half
    metrics["skipped_dense_orientation"] = False
    return metrics, curve


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = collect_predictions(args.prediction_dir.resolve())
    if args.dataset == "BIPED":
        samples = biped_samples(
            args.biped_root.resolve(),
            args.split_file.resolve() if args.split_file else None,
        )
    elif args.dataset == "BSDS500":
        samples = bsds_samples(args.bsds_root.resolve())
    elif args.dataset == "Multicue":
        samples = multicue_samples(
            args.multicue_root.resolve(),
            args.split_file.resolve() if args.split_file else None,
        )
    elif args.dataset == "NYUDv2":
        samples = nyud_samples(
            args.nyud_root.resolve(),
            args.split_file.resolve() if args.split_file else None,
        )
    else:
        samples = uded_samples(args.uded_root.resolve())
    if args.max_samples:
        samples = samples[: args.max_samples]
    validate_samples(samples)
    requested_stems = {stem for stem, _ in samples}
    extra = sorted(set(predictions) - requested_stems)
    if extra and not args.allow_extra:
        raise RuntimeError(
            f"Prediction directory contains {len(extra)} samples outside the requested split; "
            f"first 20={extra[:20]}. Use a split-specific prediction directory, or pass "
            "--allow-extra only for diagnostics."
        )

    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    missing: list[str] = []
    for stem, gt_path in samples:
        pred_path = predictions.get(stem)
        if pred_path is None:
            missing.append(stem)
            continue
        target = load_bsds_gt(gt_path) if args.dataset == "BSDS500" else normalize_image_array(gt_path)
        pred = resize_to(load_prediction(pred_path), target.shape)
        probs.append(pred)
        targets.append(target)

    if not probs:
        raise RuntimeError(f"No matched predictions under {args.prediction_dir}")
    if missing and not args.allow_missing:
        raise RuntimeError(
            f"Incomplete prediction set: matched {len(probs)}/{len(samples)} samples; "
            f"missing first 20={missing[:20]}. Pass --allow-missing only for diagnostics."
        )

    thresholds = np.linspace(0.01, 0.99, max(2, int(args.thresholds)))
    orientations = ["as_is", "inverted"] if args.orientation == "auto" else [args.orientation]
    summaries = []
    curves = {}
    for orient in orientations:
        metrics, curve = evaluate_orientation(
            probs,
            targets,
            orient,
            thresholds,
            args.match_tolerance,
            apply_nms=args.apply_nms,
            device=args.device,
            nms_low_threshold=args.nms_low_threshold,
            skip_dense_orientation=args.skip_dense_orientation,
            metric_backend=args.metric_backend,
            gt_threshold=args.gt_threshold,
        )
        summaries.append(metrics)
        curves[orient] = curve

    best = max(summaries, key=lambda row: (float(row["ODS"]), float(row["AP"])))
    summary = {
        "dataset": args.dataset,
        "prediction_dir": str(args.prediction_dir.resolve()),
        "n_requested": len(samples),
        "n_matched": len(probs),
        "missing_count": len(missing),
        "missing_first20": missing[:20],
        "extra_count": len(extra),
        "extra_first20": extra[:20],
        "apply_nms": bool(args.apply_nms),
        "nms_low_threshold": args.nms_low_threshold,
        "match_tolerance": args.match_tolerance,
        "metric_backend": args.metric_backend,
        "gt_threshold": args.gt_threshold,
        "requested_orientation": args.orientation,
        "selected_orientation": best["orientation"],
        "selected": best,
        "all_orientations": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", summaries)
    for orient, curve in curves.items():
        write_csv(args.output_dir / f"pr_curve_{orient}.csv", curve)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
