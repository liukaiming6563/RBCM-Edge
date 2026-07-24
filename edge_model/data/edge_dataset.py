"""Dataset implementation for local edge detection folders.

Each dataset is expected to contain:

```text
DatasetName/
  image/
  edge/
  gt/
    soft_vote/
    uncertainty_weight/
    annotation_uncertainty/
  overlapping/
```

By default `image/` and `edge/` are used. Training and evaluation configs may
also point to a named `gt/<variant>/` folder. Some normalized datasets expose
`gt/uncertainty_weight/`, which can be returned as a pixel-wise loss-weight map
while keeping the soft-vote edge map as the training target.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EdgePair:
    """One input image and its ground-truth edge map."""

    image_path: Path
    edge_path: Path
    weight_path: Path | None
    dataset_name: str
    sample_id: str
    edge_variant: str = "edge"
    weight_variant: str | None = None


def collect_edge_pairs(
    dataset_root: str | Path,
    dataset_name: str,
    edge_variant: str = "edge",
    weight_variant: str | None = None,
    required_stems: set[str] | None = None,
) -> list[EdgePair]:
    """Collect image-edge pairs by matching file stems.

    Args:
        dataset_root: Folder containing `image/` and `edge/` or `gt/<variant>/`.
        dataset_name: Human-readable dataset name written into each sample.
        edge_variant: `edge` for the primary map, otherwise a folder name under
            `gt/`.
        weight_variant: Optional folder name under `gt/` used as a pixel-wise
            loss-weight map. Pass `None` or an empty string to disable weights.
        required_stems: Optional split membership. When provided, completeness
            is enforced only for the requested samples. This supports eval-only
            GT variants such as NYUDv2 `soft_vote` without requiring generated
            training augmentations to duplicate that target folder.

    Returns:
        Sorted list of matched image-edge pairs.
    """
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "image"
    edge_variant = str(edge_variant or "edge")
    edge_dir = _resolve_edge_dir(dataset_root, edge_variant)
    if not image_dir.exists() or not edge_dir.exists():
        raise FileNotFoundError(f"Expected image/ and {edge_dir.relative_to(dataset_root)} under {dataset_root}")
    weight_variant = _normalize_optional_variant(weight_variant)
    weight_dir = _resolve_edge_dir(dataset_root, weight_variant) if weight_variant else None
    if weight_dir is not None and not weight_dir.exists():
        raise FileNotFoundError(
            f"Expected loss-weight folder {weight_dir.relative_to(dataset_root)} under {dataset_root}"
        )

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if required_stems is not None:
        images = [path for path in images if path.stem in required_stems]
    edges = [p for p in edge_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    image_by_stem = _unique_files_by_stem(images, f"{dataset_name} images")
    edge_by_stem = _unique_files_by_stem(edges, f"{dataset_name} {edge_variant} targets")
    weight_by_stem: dict[str, Path] = {}
    if weight_dir is not None:
        weights = [p for p in weight_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        weight_by_stem = _unique_files_by_stem(weights, f"{dataset_name} {weight_variant} weights")

    pairs: list[EdgePair] = []
    missing_edges: list[str] = []
    missing_weights: list[str] = []
    for image_path in sorted(image_by_stem.values(), key=lambda p: p.stem):
        edge_path = edge_by_stem.get(image_path.stem)
        if edge_path is None:
            missing_edges.append(image_path.name)
            continue
        weight_path = weight_by_stem.get(image_path.stem) if weight_dir is not None else None
        if weight_dir is not None and weight_path is None:
            missing_weights.append(image_path.name)
            continue
        pairs.append(
            EdgePair(
                image_path=image_path,
                edge_path=edge_path,
                weight_path=weight_path,
                dataset_name=dataset_name,
                sample_id=image_path.stem,
                edge_variant=edge_variant,
                weight_variant=weight_variant,
            )
        )

    if missing_edges:
        preview = ", ".join(missing_edges[:5])
        raise ValueError(f"{dataset_name} has images without {edge_variant} edge maps: {preview}")
    if missing_weights:
        preview = ", ".join(missing_weights[:5])
        raise ValueError(f"{dataset_name} has images without {weight_variant} weight maps: {preview}")
    if required_stems is not None:
        found = {pair.sample_id for pair in pairs}
        missing_required = sorted(required_stems - found)
        if missing_required:
            preview = ", ".join(missing_required[:5])
            raise ValueError(f"{dataset_name} split references unavailable image/target pairs: {preview}")
    return pairs


def _unique_files_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    """Index files without silently choosing one of two same-stem inputs."""
    by_stem: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        previous = by_stem.get(path.stem)
        if previous is not None:
            duplicates.setdefault(path.stem, [previous]).append(path)
        else:
            by_stem[path.stem] = path
    if duplicates:
        preview = "; ".join(
            f"{stem}: {', '.join(str(path.name) for path in files)}"
            for stem, files in list(sorted(duplicates.items()))[:5]
        )
        raise ValueError(f"Duplicate stems in {label}: {preview}")
    return by_stem


def _resolve_edge_dir(dataset_root: Path, edge_variant: str) -> Path:
    """Resolve the folder that contains labels for a configured GT variant."""
    normalized = str(edge_variant or "edge").replace("\\", "/").strip("/")
    if normalized in {"", "edge", "primary", "default"}:
        return dataset_root / "edge"
    if normalized.startswith("gt/"):
        return dataset_root / normalized
    return dataset_root / "gt" / normalized


def _normalize_optional_variant(variant: str | None) -> str | None:
    """Normalize optional GT variants where common false-like values disable maps."""
    if variant is None:
        return None
    normalized = str(variant).replace("\\", "/").strip("/")
    if normalized.lower() in {"", "none", "null", "false", "off", "disabled"}:
        return None
    return normalized


def split_pairs(
    pairs: list[EdgePair],
    split: str,
    val_fraction: float = 0.15,
    seed: int = 42,
    split_file: str | Path | None = None,
) -> list[EdgePair]:
    """Return a deterministic train/val/all split.

    When `split_file` is provided, it is interpreted as a newline-delimited list
    of sample stems. Otherwise the function uses a deterministic random split
    for local demos and supports `all` for cross-dataset testing.
    """
    if split_file is not None:
        requested = _read_split_stems(split_file)
        duplicate_requests = sorted(stem for stem, count in Counter(requested).items() if count > 1)
        if duplicate_requests:
            raise ValueError(f"Split file contains duplicate sample IDs: {duplicate_requests[:5]}")
        by_stem = {pair.sample_id: pair for pair in pairs}
        missing = sorted(set(requested) - by_stem.keys())
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Split file references missing samples: {preview}")
        return [by_stem[stem] for stem in requested]

    if split == "all":
        return pairs
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    import random

    shuffled = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items if split == "train" else val_items


def _read_split_stems(split_file: str | Path) -> list[str]:
    """Read non-empty sample stems from a split text file."""
    path = Path(split_file)
    stems: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stems.append(line.strip().split()[0])
    return stems


class EdgeFolderDataset(Dataset):
    """PyTorch dataset for RGB images and single-channel edge maps."""

    def __init__(
        self,
        pairs: list[EdgePair],
        transform: Callable | None = None,
    ) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        edge = Image.open(pair.edge_path).convert("L")
        weight = Image.open(pair.weight_path).convert("L") if pair.weight_path is not None else None
        if self.transform is not None:
            transformed = self.transform(image, edge, weight)
        else:
            raise RuntimeError("A transform must be provided to convert PIL images to tensors.")

        image_tensor, edge_tensor = transformed[:2]
        sample = {
            "image": image_tensor,
            "edge": edge_tensor,
            "dataset": pair.dataset_name,
            "sample_id": pair.sample_id,
            "image_path": str(pair.image_path),
            "edge_path": str(pair.edge_path),
            "edge_variant": pair.edge_variant,
        }
        if pair.weight_path is not None:
            sample["weight_path"] = str(pair.weight_path)
            sample["weight_variant"] = pair.weight_variant or ""
        if len(transformed) == 3:
            sample.update(transformed[2])
        return sample
