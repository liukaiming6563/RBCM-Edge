"""Factories for datasets and data loaders."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_model.data.edge_dataset import EdgeFolderDataset, collect_edge_pairs, split_pairs
from edge_model.data.transforms import make_eval_transform, make_train_transform


def make_dataset(config: dict, dataset_name: str, split: str, training: bool) -> EdgeFolderDataset:
    """Build an edge dataset from the configured edge_data folder."""
    project_root = Path(config.get("paths", {}).get("project_root", ".")).resolve()
    image_root = Path(config.get("paths", {}).get("edge_data_root", "edge_data"))
    if not image_root.is_absolute():
        image_root = project_root / image_root

    dataset_cfg = config.get("dataset", {})
    gt_cfg = dataset_cfg.get("gt", {})
    if not isinstance(gt_cfg, dict):
        gt_cfg = {}

    dataset_root = image_root / dataset_name
    edge_variant = _resolve_gt_variant(gt_cfg, dataset_name=dataset_name, training=training)
    weight_variant = _resolve_loss_weight_variant(gt_cfg, dataset_name=dataset_name, training=training)
    split_file = _resolve_split_file(dataset_root, dataset_cfg, split, project_root)
    required_stems = _read_split_stems(split_file) if split_file is not None else None
    pairs = collect_edge_pairs(
        dataset_root,
        dataset_name,
        edge_variant=edge_variant,
        weight_variant=weight_variant,
        required_stems=required_stems,
    )
    pairs = split_pairs(
        pairs,
        split=split,
        val_fraction=float(dataset_cfg.get("val_fraction", 0.15)),
        seed=int(config.get("seed", 42)),
        split_file=split_file,
    )
    pairs = _limit_pairs(pairs, dataset_cfg, training=training)
    gt_mode = str(gt_cfg.get("train_mode" if training else "eval_mode", gt_cfg.get("mode", "binary")))
    distance_soft_sigma = float(gt_cfg.get("distance_soft_sigma", 1.0))
    distance_soft_max_distance = int(gt_cfg.get("distance_soft_max_distance", 3))
    binarize_edges = bool(
        gt_cfg.get(
            "binarize_train_edges" if training else "binarize_eval_edges",
            dataset_cfg.get("binarize_edges", True),
        )
    )

    if training:
        transform = make_train_transform(
            input_size=int(dataset_cfg.get("input_size", 384)),
            random_crop=bool(dataset_cfg.get("random_crop", True)),
            horizontal_flip=bool(dataset_cfg.get("horizontal_flip", True)),
            vertical_flip=bool(dataset_cfg.get("vertical_flip", False)),
            preserve_aspect=bool(dataset_cfg.get("preserve_aspect_train", True)),
            native_size=bool(dataset_cfg.get("native_size_train", False)),
            binarize_edges=binarize_edges,
            gt_mode=gt_mode,
            distance_soft_sigma=distance_soft_sigma,
            distance_soft_max_distance=distance_soft_max_distance,
        )
    else:
        transform = make_eval_transform(
            input_size=int(dataset_cfg.get("input_size", 384)),
            preserve_aspect=bool(dataset_cfg.get("preserve_aspect_eval", True)),
            native_size=bool(dataset_cfg.get("native_size_eval", False)),
            binarize_edges=binarize_edges,
            gt_mode=gt_mode,
            distance_soft_sigma=distance_soft_sigma,
            distance_soft_max_distance=distance_soft_max_distance,
            return_meta=True,
        )

    return EdgeFolderDataset(pairs, transform=transform)


def _read_split_stems(split_file: Path) -> set[str]:
    """Read split membership early so GT completeness is split-specific."""
    stems: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                stems.append(line.strip().split()[0])
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise ValueError(f"Split file contains duplicate sample IDs: {duplicates[:5]}")
    return set(stems)


def _resolve_gt_variant(gt_cfg: dict, dataset_name: str, training: bool) -> str:
    """Return the configured GT folder variant for this dataset and stage.

    `edge` is the primary backwards-compatible label folder. Multi-annotator
    datasets can override training and validation/test to use continuous
    targets such as `soft_vote`.
    """
    stage = "train" if training else "eval"
    variant = gt_cfg.get(f"{stage}_variant", gt_cfg.get("variant", "edge"))
    dataset_variants = gt_cfg.get("dataset_variants", gt_cfg.get("variants", {}))
    if isinstance(dataset_variants, dict):
        dataset_rule = None
        for key in (dataset_name, dataset_name.lower(), dataset_name.upper()):
            if key in dataset_variants:
                dataset_rule = dataset_variants[key]
                break
        if isinstance(dataset_rule, dict):
            variant = dataset_rule.get(stage, dataset_rule.get(f"{stage}_variant", variant))
        elif dataset_rule:
            variant = dataset_rule
    return str(variant or "edge")


def _resolve_loss_weight_variant(gt_cfg: dict, dataset_name: str, training: bool) -> str | None:
    """Return an optional GT folder containing pixel-wise loss weights."""
    stage = "train" if training else "eval"
    variant = gt_cfg.get(
        f"loss_weight_{stage}_variant",
        gt_cfg.get(f"{stage}_loss_weight_variant", gt_cfg.get("loss_weight_variant")),
    )
    dataset_variants = gt_cfg.get(
        "dataset_loss_weight_variants",
        gt_cfg.get("loss_weight_dataset_variants", {}),
    )
    if isinstance(dataset_variants, dict):
        dataset_rule = None
        for key in (dataset_name, dataset_name.lower(), dataset_name.upper()):
            if key in dataset_variants:
                dataset_rule = dataset_variants[key]
                break
        if isinstance(dataset_rule, dict):
            variant = dataset_rule.get(stage, dataset_rule.get(f"{stage}_variant", variant))
        elif dataset_rule is not None:
            variant = dataset_rule
    if variant is None:
        return None
    normalized = str(variant).replace("\\", "/").strip("/")
    if normalized.lower() in {"", "none", "null", "false", "off", "disabled"}:
        return None
    return normalized


def make_loader(
    dataset: EdgeFolderDataset,
    config: dict,
    shuffle: bool,
    *,
    generator: torch.Generator | None = None,
    persistent_workers: bool | None = None,
) -> DataLoader:
    """Create a PyTorch DataLoader with config-controlled defaults."""
    loader_cfg = config.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", 2))
    loader_kwargs = {
        "batch_size": int(loader_cfg.get("batch_size", 4)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", True)),
        "drop_last": shuffle,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator
    if num_workers > 0:
        keep_workers = (
            bool(loader_cfg.get("persistent_workers", True))
            if persistent_workers is None
            else bool(persistent_workers)
        )
        loader_kwargs["persistent_workers"] = keep_workers
        prefetch_factor = loader_cfg.get("prefetch_factor", 2)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def _resolve_split_file(dataset_root: Path, dataset_cfg: dict, split: str, project_root: Path) -> Path | None:
    """Resolve an optional official/fixed split file for the requested split."""
    split_files = dataset_cfg.get("split_files", {})
    split_file = split_files.get(split) if isinstance(split_files, dict) else None
    if split_file is None:
        candidates = [
            dataset_root / "splits" / f"{split}.txt",
            project_root / "edge_model" / "splits" / dataset_root.name / f"{split}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    split_path = Path(split_file)
    if not split_path.is_absolute():
        dataset_relative = dataset_root / split_path
        project_relative = project_root / split_path
        split_path = dataset_relative if dataset_relative.exists() else project_relative
    return split_path


def _limit_pairs(pairs: list, dataset_cfg: dict, training: bool) -> list:
    """Apply optional sample caps for short local probes."""
    key = "max_train_samples" if training else "max_eval_samples"
    limit = dataset_cfg.get(key)
    if limit is None and not training:
        limit = dataset_cfg.get("max_val_samples")
    if limit is None:
        return pairs
    return pairs[: max(0, int(limit))]
