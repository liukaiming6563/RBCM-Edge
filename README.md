# RBCM-Edge

This public source package contains the final code for two connected parts of
the study:

1. retinal MEA analysis of UME and CME local population trajectories;
2. H-RBCM edge detection with annular surround-to-center logit modulation.

Historical ResNet, PiDiNet, A-G, probe, and exploratory variants are excluded.
The repository contains code and configuration only. Raw recordings, image
datasets, checkpoints, and generated results are distributed separately.

## Source layout

- `MEA_analysis`: final MEA trajectory analysis and shared data loaders;
- `MEA_model`: paper-facing MEA figure and statistical-summary modules;
- `edge_model`: H-RBCM training, inference, calibration, and evaluation;
- `src/rbcm_edge`: importable H-RBCM implementation;
- `scripts`: reproducibility, integrity, and pipeline entry points.
- `docs/results`: canonical machine-readable paper scores with protocol labels.

The Python evaluator is a shared near-official implementation with original
image sizes, NMS, threshold sweeping, and target-specific localization
tolerance. Its dilation matcher is not the exact BSDS benchmark Matlab
bipartite matcher. Use one backend consistently within a comparison and label
the protocol explicitly in reported tables.

## External packages

Place the separately distributed directories at the repository root:

- `pretrained/` from `RBCM-Edge-Checkpoints`;
- `edge_data/` from `RBCM-Edge-Datasets`;
- `MEA_data/` from `RBCM-Edge-MEA-Data`.

Download links and archive hashes can be filled into `DOWNLOADS.md` after the
three large packages are uploaded.

## Environment

Use Python 3.10 or newer and install a PyTorch build matching the local CUDA
runtime:

```bash
pip install -e .
pip install opencv-python-headless
```

## H-RBCM

H-RBCM trains one HED-lite center-edge anchor and derives four matched outputs:
`plain_identity`, `main_surround`, `no_surround`, and `conv_control`.
Calibration candidates are selected on validation data and then frozen.
The formal score index is `docs/results/formal_result_index.csv`; it records
the protocol role and evidence source beside every score.

```bash
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained
python scripts/checks/audit_nyud_strict_protocol.py   --config edge_model/configs/rbcm/nyudv2_strict.yaml
python scripts/analysis/evaluate_nyud_strict_generalization.py   --config edge_model/configs/rbcm/nyudv2_strict.yaml   --checkpoint pretrained/nyudv2_strict/best.pt   --formal-summary pretrained/nyudv2_strict/formal_summary.json   --run-tag nyudv2_strict_reproduction   --datasets BIPED Multicue NYUDv2 BSDS500 UDED   --device cuda --batch-size 1 --num-workers 2
```

Training is resumable:

```bash
python edge_model/train.py   --config edge_model/configs/rbcm/nyudv2_strict.yaml --resume
```

## MEA analysis

List or run the formal sequence with:

```bash
python scripts/analysis/run_mea_pipeline.py --list
python scripts/analysis/run_mea_pipeline.py
```

The pipeline reads `MEA_data/` and writes reproducible tables, reports, and
figures under `MEA_outputs/`.

## Licensing

No code license is assigned automatically. Dataset and recording
redistribution remains governed by the original sources and ethics/data-use
requirements. Review these terms before public distribution.
