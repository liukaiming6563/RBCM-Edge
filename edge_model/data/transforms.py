"""Image and edge-map transforms for local training.

The transforms avoid torchvision so the project has fewer moving parts. Images
are converted to float tensors in `[0, 1]`, then normalized with ImageNet mean
and standard deviation. Edge maps can be kept soft or binarized by config.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGENET_FILL = tuple(int(round(v * 255)) for v in (0.485, 0.456, 0.406))


@dataclass
class EdgeTransform:
    """Joint transform for an RGB image and its edge map."""

    input_size: int = 384
    random_crop: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    preserve_aspect: bool = True
    native_size: bool = False
    binarize_edges: bool = True
    gt_mode: str = "binary"
    distance_soft_sigma: float = 1.0
    distance_soft_max_distance: int = 3
    return_meta: bool = False

    def __call__(self, image: Image.Image, edge: Image.Image, weight: Image.Image | None = None):
        """Apply resize/crop/flip and convert to tensors."""
        original_width, original_height = image.size
        image, edge, weight, meta = self._resize_or_crop(image, edge, weight)

        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if weight is not None:
                weight = weight.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.vertical_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            edge = edge.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if weight is not None:
                weight = weight.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        meta.setdefault("original_width", original_width)
        meta.setdefault("original_height", original_height)
        image_tensor = self._image_to_tensor(image)
        edge_tensor = self._edge_to_tensor(
            edge,
            binarize_edges=self.binarize_edges,
            gt_mode=self.gt_mode,
            distance_soft_sigma=self.distance_soft_sigma,
            distance_soft_max_distance=self.distance_soft_max_distance,
        )
        extras: dict[str, object] = {}
        if weight is not None:
            extras["loss_weight"] = self._weight_to_tensor(weight)
        if self.return_meta:
            extras.update(meta)
        if extras:
            return image_tensor, edge_tensor, extras
        return image_tensor, edge_tensor

    def _resize_or_crop(
        self,
        image: Image.Image,
        edge: Image.Image,
        weight: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image, Image.Image | None, dict]:
        """Resize validation images and random-crop training images."""
        if self.native_size:
            width, height = image.size
            return image, edge, weight, self._full_meta(width=width, height=height)

        size = int(self.input_size)
        if self.random_crop:
            image, edge, weight = self._resize_short_side(image, edge, weight, size)
            width, height = image.size
            if width == size and height == size:
                return image, edge, weight, self._full_meta(width=width, height=height)
            left = random.randint(0, max(0, width - size))
            top = random.randint(0, max(0, height - size))
            box = (left, top, left + size, top + size)
            return (
                image.crop(box),
                edge.crop(box),
                weight.crop(box) if weight is not None else None,
                self._full_meta(width=size, height=size),
            )

        if self.preserve_aspect:
            return self._letterbox(image, edge, weight, size)

        return (
            image.resize((size, size), Image.Resampling.BILINEAR),
            self._resize_edge(edge, (size, size)),
            self._resize_weight(weight, (size, size)) if weight is not None else None,
            self._full_meta(width=size, height=size),
        )

    def _resize_short_side(
        self,
        image: Image.Image,
        edge: Image.Image,
        weight: Image.Image | None,
        size: int,
    ) -> tuple[Image.Image, Image.Image, Image.Image | None]:
        """Resize so both dimensions are at least `size`, preserving aspect ratio."""
        width, height = image.size
        scale = max(size / width, size / height)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return (
            image.resize(new_size, Image.Resampling.BILINEAR),
            self._resize_edge(edge, new_size),
            self._resize_weight(weight, new_size) if weight is not None else None,
        )

    def _letterbox(
        self,
        image: Image.Image,
        edge: Image.Image,
        weight: Image.Image | None,
        size: int,
    ) -> tuple[Image.Image, Image.Image, Image.Image | None, dict]:
        """Fit the whole image inside a square canvas without distorting aspect."""
        width, height = image.size
        scale = size / max(width, height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_edge = self._resize_edge(edge, (resized_width, resized_height))
        resized_weight = self._resize_weight(weight, (resized_width, resized_height)) if weight is not None else None

        pad_left = (size - resized_width) // 2
        pad_top = (size - resized_height) // 2
        pad_right = size - resized_width - pad_left
        pad_bottom = size - resized_height - pad_top
        # Replicate image border pixels instead of adding a constant-color frame;
        # otherwise the letterbox boundary becomes an artificial edge.
        image_array = np.asarray(resized_image.convert("RGB"), dtype=np.uint8)
        image_canvas = Image.fromarray(
            np.pad(
                image_array,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            ),
            mode="RGB",
        )
        edge_canvas = Image.new("L", (size, size), 0)
        edge_canvas.paste(resized_edge, (pad_left, pad_top))
        weight_canvas = None
        if resized_weight is not None:
            # Padding has no original annotation support, so ignore it in the
            # optional validation loss instead of treating it as certain background.
            weight_canvas = Image.new("L", (size, size), 0)
            weight_canvas.paste(resized_weight, (pad_left, pad_top))
        return (
            image_canvas,
            edge_canvas,
            weight_canvas,
            {
                "original_width": width,
                "original_height": height,
                "content_width": resized_width,
                "content_height": resized_height,
                "pad_left": pad_left,
                "pad_top": pad_top,
            },
        )

    @staticmethod
    def _full_meta(width: int, height: int) -> dict:
        """Metadata for samples whose full tensor area is valid image content."""
        return {
            "content_width": width,
            "content_height": height,
            "pad_left": 0,
            "pad_top": 0,
        }

    def _resize_edge(self, edge: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Resize binary/thin edge labels without erasing one-pixel contours."""
        resampling = Image.Resampling.BILINEAR if self._uses_soft_vote_resize() else Image.Resampling.NEAREST
        return edge.resize(size, resampling)

    @staticmethod
    def _resize_weight(weight: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
        """Resize continuous loss-weight maps with value-preserving interpolation."""
        if weight is None:
            return None
        return weight.resize(size, Image.Resampling.BILINEAR)

    def _uses_soft_vote_resize(self) -> bool:
        """Return true only when the input edge map should keep soft intensities."""
        mode = str(self.gt_mode).lower().replace("-", "_")
        return (not self.binarize_edges) and mode in {"soft", "soft_vote"}

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL RGB image to normalized CHW tensor."""
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    @staticmethod
    def _edge_to_tensor(
        edge: Image.Image,
        binarize_edges: bool,
        gt_mode: str = "binary",
        distance_soft_sigma: float = 1.0,
        distance_soft_max_distance: int = 3,
    ) -> torch.Tensor:
        """Convert PIL grayscale edge map to `[1, H, W]` float tensor."""
        array = np.asarray(edge, dtype=np.float32) / 255.0
        mode = str(gt_mode).lower().replace("-", "_")
        if mode in {"distance_soft", "distance"}:
            array = _distance_soft_label(
                (array > 0.5).astype(np.uint8),
                sigma=float(distance_soft_sigma),
                max_distance=int(distance_soft_max_distance),
            )
        elif binarize_edges or mode in {"binary", "hard", "union", "consensus"}:
            array = (array > 0.5).astype(np.float32)
        else:
            array = np.clip(array, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).contiguous()

    @staticmethod
    def _weight_to_tensor(weight: Image.Image) -> torch.Tensor:
        """Convert an uncertainty-derived loss-weight map to `[1, H, W]`."""
        array = np.asarray(weight, dtype=np.float32) / 255.0
        array = np.clip(array, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).contiguous()


def _distance_soft_label(binary_edge: np.ndarray, sigma: float = 1.0, max_distance: int = 3) -> np.ndarray:
    """Create a soft edge band from a binary edge map.

    Pixels on annotated edges remain 1.0. Nearby pixels receive a Gaussian
    falloff based on Euclidean distance, while far background stays 0. This is
    only for training targets; final evaluation still thresholds the raw GT.
    """
    binary = (binary_edge > 0).astype(np.uint8)
    if binary.max() == 0:
        return binary.astype(np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
    except ModuleNotFoundError:
        return _distance_soft_label_local(binary, sigma=sigma, max_distance=max_distance)

    distance = distance_transform_edt(1 - binary).astype(np.float32)
    sigma = max(float(sigma), 1e-6)
    soft = np.exp(-(distance**2) / (2.0 * sigma**2)).astype(np.float32)
    if max_distance is not None and int(max_distance) >= 0:
        soft[distance > int(max_distance)] = 0.0
    soft[binary > 0] = 1.0
    return soft.astype(np.float32)


def _distance_soft_label_local(binary: np.ndarray, sigma: float, max_distance: int) -> np.ndarray:
    """Small-radius distance-soft fallback that does not require SciPy.

    The project normally uses `scipy.ndimage.distance_transform_edt`, but local
    Windows environments may not have SciPy available. The configured edge
    tolerance radius is small, so a deterministic shifted-neighborhood fallback
    is sufficient and prevents distance-soft supervision from silently
    degenerating into hard binary labels.
    """
    radius = max(0, int(max_distance))
    if radius <= 0:
        return binary.astype(np.float32)
    height, width = binary.shape
    sigma = max(float(sigma), 1e-6)
    soft = binary.astype(np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = float((dy * dy + dx * dx) ** 0.5)
            if distance <= 0.0 or distance > float(radius):
                continue
            weight = float(np.exp(-(distance**2) / (2.0 * sigma**2)))
            shifted = np.zeros((height, width), dtype=np.float32)
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = binary[src_y0:src_y1, src_x0:src_x1] * weight
            soft = np.maximum(soft, shifted)
    return soft.astype(np.float32)


def make_train_transform(
    input_size: int,
    random_crop: bool,
    horizontal_flip: bool,
    vertical_flip: bool,
    preserve_aspect: bool = True,
    native_size: bool = False,
    binarize_edges: bool = True,
    gt_mode: str = "binary",
    distance_soft_sigma: float = 1.0,
    distance_soft_max_distance: int = 3,
) -> EdgeTransform:
    """Create the transform used for training."""
    return EdgeTransform(
        input_size=input_size,
        random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        preserve_aspect=preserve_aspect,
        native_size=native_size,
        binarize_edges=binarize_edges,
        gt_mode=gt_mode,
        distance_soft_sigma=distance_soft_sigma,
        distance_soft_max_distance=distance_soft_max_distance,
        return_meta=False,
    )


def make_eval_transform(
    input_size: int,
    preserve_aspect: bool = True,
    native_size: bool = False,
    binarize_edges: bool = True,
    gt_mode: str = "binary",
    distance_soft_sigma: float = 1.0,
    distance_soft_max_distance: int = 3,
    return_meta: bool = True,
) -> EdgeTransform:
    """Create deterministic transform used for validation, testing, and inference."""
    return EdgeTransform(
        input_size=input_size,
        random_crop=False,
        horizontal_flip=False,
        vertical_flip=False,
        preserve_aspect=preserve_aspect,
        native_size=native_size,
        binarize_edges=binarize_edges,
        gt_mode=gt_mode,
        distance_soft_sigma=distance_soft_sigma,
        distance_soft_max_distance=distance_soft_max_distance,
        return_meta=return_meta,
    )
