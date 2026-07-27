"""Model factory for the formal RBCM release.

The trainable network in RBCM is the plain HED-lite anchor. The paper's
``main_surround``, ``no_surround``, and ``conv_control`` modes are fixed
validation-selected logit calibration functions applied after anchor
inference; they are implemented by the RBCM experiment scripts rather than
as separately trained neural networks.
"""

from __future__ import annotations

from rbcm_edge.models.networks import EdgeAnchor


def build_model(config: dict) -> EdgeAnchor:
    """Build the HED-lite anchor used by every formal RBCM mode."""

    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "baseline_host_edge"))
    host = str(model_cfg.get("host", "hed_lite"))
    variant = str(model_cfg.get("variant", "plain"))

    if name not in {"baseline_host_edge", "baseline_host", "edge_baseline_host"}:
        raise ValueError(
            f"The formal RBCM release only supports the edge-anchor factory; got name={name!r}."
        )
    if host not in {"hed_lite", "hed", "hed_style", "hed_edge"}:
        raise ValueError(f"The formal RBCM release requires host='hed_lite'; got {host!r}.")
    if variant != "plain":
        raise ValueError(
            f"The trainable RBCM anchor requires variant='plain'; got {variant!r}. "
            "Surround and matched-control modes are applied during fixed logit calibration."
        )

    return EdgeAnchor(
        host="hed_lite",
        variant="plain",
        in_channels=int(model_cfg.get("in_channels", 3)),
        feature_channels=int(model_cfg.get("feature_channels", 48)),
        decoder_channels=int(model_cfg.get("decoder_channels", 64)),
        norm=str(model_cfg.get("norm", "gn")),
        gn_groups=int(model_cfg.get("gn_groups", 8)),
        activation=str(model_cfg.get("activation", "relu")),
    )
