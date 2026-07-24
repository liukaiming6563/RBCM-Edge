"""Checkpoint-compatible HED-lite anchor for the final RBCM release."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def make_norm(channels: int, norm: str = "gn", groups: int = 8) -> nn.Module:
    key = str(norm).lower()
    if key in {"gn", "groupnorm", "group_norm"}:
        count = min(int(groups), int(channels))
        while int(channels) % count != 0 and count > 1:
            count -= 1
        return nn.GroupNorm(count, int(channels))
    if key in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(int(channels))
    if key in {"none", "identity", ""}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


def make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key in {"relu", "relu_"}:
        return nn.ReLU(inplace=True)
    if key in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if key in {"none", "identity", ""}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: str = "gn",
        groups: int = 8,
        activation: str = "relu",
    ) -> None:
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=int(kernel_size) // 2,
                bias=False,
            ),
            make_norm(int(out_channels), norm=norm, groups=groups),
            make_activation(activation),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        dilation: int = 1,
        norm: str = "gn",
        groups: int = 8,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        out_channels = int(channels if out_channels is None else out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                int(channels),
                int(channels),
                kernel_size=3,
                padding=int(dilation),
                dilation=int(dilation),
                groups=int(channels),
                bias=False,
            ),
            make_norm(int(channels), norm=norm, groups=groups),
            make_activation(activation),
            nn.Conv2d(int(channels), out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, norm=norm, groups=groups),
            make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGGConvBlock(nn.Module):
    def __init__(self, channels: int, repeats: int, norm: str, groups: int, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            *[
                ConvNormAct(channels, channels, norm=norm, groups=groups, activation=activation)
                for _ in range(max(1, int(repeats)))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GenericHostEncoder(nn.Module):
    def __init__(
        self,
        host: str = "hed_lite",
        in_channels: int = 3,
        base_channels: int = 48,
        norm: str = "gn",
        groups: int = 8,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if str(host).lower() not in {"hed_lite", "hed", "hed_style", "hed_edge"}:
            raise ValueError(f"RBCM requires the HED-lite host, got {host!r}")
        self.host = "hed_lite"
        c1 = int(base_channels)
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 8
        self.stem = nn.Sequential(
            ConvNormAct(int(in_channels), c1, norm=norm, groups=groups, activation=activation),
            VGGConvBlock(c1, repeats=2, norm=norm, groups=groups, activation=activation),
        )
        self.down2 = ConvNormAct(c1, c2, stride=2, norm=norm, groups=groups, activation=activation)
        self.stage2 = VGGConvBlock(c2, repeats=2, norm=norm, groups=groups, activation=activation)
        self.down3 = ConvNormAct(c2, c3, stride=2, norm=norm, groups=groups, activation=activation)
        self.stage3 = VGGConvBlock(c3, repeats=3, norm=norm, groups=groups, activation=activation)
        self.down4 = ConvNormAct(c3, c4, stride=2, norm=norm, groups=groups, activation=activation)
        self.stage4 = VGGConvBlock(c4, repeats=3, norm=norm, groups=groups, activation=activation)
        self.out_channels = (c1, c2, c3, c4)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(image)
        c2 = self.stage2(self.down2(c1))
        c3 = self.stage3(self.down3(c2))
        c4 = self.stage4(self.down4(c3))
        return c1, c2, c3, c4


class LocalFuseBranch(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int, norm: str, groups: int, activation: str) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvNormAct(ch, out_channels, kernel_size=1, norm=norm, groups=groups, activation=activation) for ch in in_channels]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels * len(in_channels), out_channels, norm=norm, groups=groups, activation=activation),
            DepthwiseSeparableConv(out_channels, out_channels, norm=norm, groups=groups, activation=activation),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections, strict=True):
            y = projection(feature)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            projected.append(y)
        return self.fuse(torch.cat(projected, dim=1))


class DeepContextFuseBranch(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int, norm: str, groups: int, activation: str) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvNormAct(ch, out_channels, kernel_size=1, norm=norm, groups=groups, activation=activation) for ch in in_channels]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels * len(in_channels), out_channels, norm=norm, groups=groups, activation=activation),
            ConvNormAct(out_channels, out_channels, norm=norm, groups=groups, activation=activation),
        )

    def forward(self, features: Sequence[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        resized = []
        for feature, projection in zip(features, self.projections, strict=True):
            y = projection(feature)
            if y.shape[-2:] != output_size:
                y = F.interpolate(y, size=output_size, mode="bilinear", align_corners=False)
            resized.append(y)
        return self.fuse(torch.cat(resized, dim=1))


class DifferentialContextBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, groups: int, activation: str) -> None:
        super().__init__()
        self.local = DepthwiseSeparableConv(in_channels, out_channels, norm=norm, groups=groups, activation=activation)
        self.dilated = DepthwiseSeparableConv(
            in_channels, out_channels, dilation=2, norm=norm, groups=groups, activation=activation
        )
        self.fuse = ConvNormAct(out_channels * 2, out_channels, kernel_size=1, norm=norm, groups=groups, activation=activation)

    def forward(self, shallow: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.local(shallow), self.dilated(shallow)], dim=1))


class EdgeAnchor(nn.Module):
    """HED-lite center-edge anchor used by the formal RBCM model."""

    def __init__(
        self,
        host: str = "hed_lite",
        variant: str = "plain",
        in_channels: int = 3,
        feature_channels: int = 48,
        decoder_channels: int = 64,
        norm: str = "gn",
        gn_groups: int = 8,
        activation: str = "relu",
        **_: object,
    ) -> None:
        super().__init__()
        if str(variant).lower() != "plain":
            raise ValueError("The formal RBCM anchor only supports variant='plain'.")
        self.host = "hed_lite"
        self.variant = "plain"
        self.encoder = GenericHostEncoder(host, in_channels, feature_channels, norm, gn_groups, activation)
        channels = self.encoder.out_channels
        self.local_branch = LocalFuseBranch(channels, decoder_channels, norm, gn_groups, activation)
        self.deep_context = DeepContextFuseBranch(channels[1:], decoder_channels, norm, gn_groups, activation)
        self.diff_context = DifferentialContextBranch(channels[0], decoder_channels, norm, gn_groups, activation)
        self.plain_context_fusion = nn.Sequential(
            ConvNormAct(decoder_channels * 2, decoder_channels, kernel_size=1, norm=norm, groups=gn_groups, activation=activation),
            DepthwiseSeparableConv(decoder_channels, decoder_channels, norm=norm, groups=gn_groups, activation=activation),
        )
        self.plain_feature_fusion = nn.Sequential(
            ConvNormAct(decoder_channels * 2, decoder_channels, norm=norm, groups=gn_groups, activation=activation),
            ConvNormAct(decoder_channels, decoder_channels, norm=norm, groups=gn_groups, activation=activation),
        )
        self.modulator = None
        self.local_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.context_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.final_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.side_projections = nn.ModuleList(
            [ConvNormAct(ch, decoder_channels, kernel_size=1, norm=norm, groups=gn_groups, activation=activation) for ch in channels]
        )
        self.side_heads = nn.ModuleList([nn.Conv2d(decoder_channels, 1, kernel_size=1) for _ in channels])

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        features = self.encoder(image)
        output_size = image.shape[-2:]
        c1, c2, c3, c4 = features
        f_local = self.local_branch(features)
        f_deep = self.deep_context((c2, c3, c4), output_size=f_local.shape[-2:])
        f_diff = self.diff_context(c1)
        f_context = self.plain_context_fusion(torch.cat([f_deep, f_diff], dim=1))
        feature = self.plain_feature_fusion(torch.cat([f_local, f_context], dim=1))
        local_logits = self.local_head(f_local)
        context_logits = self.context_head(f_context)
        final_logits = self.final_head(feature)
        side_features = []
        for feature_map, projection in zip(features, self.side_projections, strict=True):
            side = projection(feature_map)
            if side.shape[-2:] != output_size:
                side = F.interpolate(side, size=output_size, mode="bilinear", align_corners=False)
            side_features.append(side)
        side_logits = [head(side) for side, head in zip(side_features, self.side_heads, strict=True)]
        batch, _, height, width = final_logits.shape
        zero = final_logits.new_zeros((batch, 1, height, width))
        neutral = final_logits.new_ones((batch, 1, height, width))
        mix = final_logits.new_zeros((batch, 3, height, width))
        mix[:, 2:3] = neutral
        return {
            "logits": final_logits,
            "local_logits": local_logits,
            "context_logits": context_logits,
            "side_logits": side_logits,
            "pyramid_features": features,
            "scale_weights": final_logits.new_full((batch, 4, height, width), 0.25),
            "feature": feature,
            "gate": zero,
            "enhance_gate": zero,
            "suppress_gate": zero,
            "mix_weights": mix,
            "mix_logits": mix,
            "neutral_weight": neutral,
            "alpha": final_logits.new_tensor(0.0),
            "local_feature": f_local,
            "raw_local_feature": f_local,
            "context_feature": f_context,
            "surround_feature": f_context,
            "c_enhance": f_context,
            "c_suppress": f_context,
            "c_neutral": f_context,
        }


__all__ = ["EdgeAnchor"]
