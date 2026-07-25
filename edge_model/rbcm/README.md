# RBCM

RBCM is the final paper-facing edge model:

1. `anchor.py` predicts a center-edge logit with a 6.639 M-parameter
   HED-lite network.
2. `calibrate.py` computes image Sobel energy and near/far annular summaries.
3. A validation-selected uncertainty-gated signed residual calibrates the
   center logit.
4. `plain_identity`, `no_surround`, and `conv_control` share the same anchor
   and evaluator with `main_surround`.

## Code

- Anchor: `src/rbcm_edge/models/networks/anchor.py`
- Factory: `edge_model/models/build.py`
- Calibration: `scripts/experiments/calibrate.py`
- Generalization: `scripts/experiments/evaluate_generalization.py`
- Near-official shared evaluator: `scripts/baselines/evaluate_official_edges.py`

The evaluator restores original image sizes, applies NMS, sweeps fixed
thresholds, and uses target-specific localization tolerance. Its dilation
matcher is not the exact BSDS Matlab bipartite matcher, so manuscripts must
name the backend and must not mix these scores with official leaderboard
numbers as if they were identical.

## Assets

- Configs: `edge_model/configs/rbcm`
- Weights and fixed calibration rows: `weights/rbcm`
- Results: `results/rbcm`
- Documentation: `docs/edge/en`

Legacy ResNet/PiDiNet/A-G implementations are not dependencies of this route.
