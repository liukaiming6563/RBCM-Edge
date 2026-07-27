# Reproducing RBCM-Edge

This guide covers the final paper-facing MEA analysis and H-RBCM edge-detection
experiments. Historical model variants are not part of the reproduction target.

## 1. Obtain the source and assets

Clone the public `release` branch and place the separately distributed assets
at the repository root:

```text
RBCM-Edge/
  pretrained/   # selected checkpoints and frozen validation candidates
  edge_data/    # formal edge datasets and split files
  MEA_data/     # authorized MEA inputs
```

Verify each downloaded archive against the SHA-256 value in `DOWNLOADS.md`
before extracting it. Dataset and MEA archives may only be redistributed when
the original license, ethics, consent, and institutional data-sharing terms
permit it.

The released MEA input starts from Kilosort outputs and downstream derived
matrices. Continuous acquisition files such as `data.raw.h5` and converted
`data.raw.bin` are not part of the download and are not required by the
released trajectory, statistics, and figure pipeline.

## 2. Create the environment

The strict NYUDv2 run and local verification used Python 3.10.16,
PyTorch 2.7.1+cu126, and torchvision 0.22.1+cu126. The strict MultiCue run used
the server image `cuda128_torch280_py312`. Install the PyTorch build appropriate
for the target CUDA runtime, then install the tested non-PyTorch dependencies:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

Small numerical differences can arise from GPU, CUDA, cuDNN, and NMS
implementations. Use the supplied split hashes, candidate files, prediction
orientation, and evaluator backend.

## 3. Verify the release before evaluation

```bash
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained --dataset all
python scripts/checks/audit_multicue_strict_protocol.py \
  --config edge_model/configs/rbcm/multicue_strict.yaml
python scripts/checks/audit_nyud_strict_protocol.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
python scripts/checks/run_rbcm_paper_preflight.py --check-data
```

The expected formal data protocols are:

- BIPED: fixed 170 train / 30 validation / 50 test, with three fixed repeats
  for the stability table;
- MultiCue: 68 train sources / 12 validation sources / 20 held-out test
  sources, all source-disjoint; checkpoint and calibration candidates are
  frozen before the test set is opened once;
- NYUDv2 RGB: 381 train / 414 validation / 654 held-out test images;
- BSDS500 and UDED: evaluation or transfer targets in the current paper
  package, not primary training evidence.

## 4. Reproduce checkpoint inference and scores

For a single image directory, export any of the four matched modes:

```bash
python edge_model/infer.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml \
  --checkpoint pretrained/nyudv2_strict/best.pt \
  --image-dir edge_data/official_rbcm/NYUDv2/image \
  --output-dir reproduced/nyudv2_main \
  --mode main_surround \
  --candidate-csv pretrained/nyudv2_strict/fixed_candidates.csv
```

Run the frozen strict-NYUDv2 checkpoint on all five targets:

```bash
python scripts/analysis/evaluate_nyud_strict_generalization.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml \
  --checkpoint pretrained/nyudv2_strict/best.pt \
  --formal-summary pretrained/nyudv2_strict/formal_summary.json \
  --run-tag nyudv2_strict_reproduction \
  --datasets BIPED Multicue NYUDv2 BSDS500 UDED \
  --device cuda --batch-size 1 --num-workers 2
```

The paper-facing evaluator restores original image size, optionally applies
NMS, sweeps fixed thresholds, and uses one target-specific localization
tolerance per dataset. It is a shared near-official Python dilation matcher,
not the exact BSDS Matlab bipartite matcher. Do not compare its absolute scores
to a different backend without an explicit protocol label.

## 5. Rebuild canonical tables and figures

```bash
python scripts/analysis/build_formal_result_index.py
python scripts/analysis/build_strict_protocol_tables.py
python scripts/analysis/plot_joint_ablation_metrics.py
```

The current formal tables contain BIPED, strict MultiCue, and strict NYUDv2.
Scripts that reconstruct the retired overlapping MultiCue route require an
explicit archival flag and must not be used for the main paper table.

## 6. Reproduce training

Training is resumable because `last.pt` stores the model, optimizer, scheduler,
AMP scaler, best score, configuration, and RNG state.

```bash
python edge_model/train.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
python edge_model/train.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml --resume
```

For MultiCue, use `edge_model/configs/rbcm/multicue_strict.yaml` and preserve
the supplied strict split. Candidate selection must use validation data only;
the test split must remain inaccessible until the candidate table is frozen.
One seed is supplied for strict MultiCue and NYUDv2, so the paper must not
claim multi-seed statistical significance for those rows.

## 7. Reproduce MEA analysis

```bash
python scripts/analysis/run_mea_pipeline.py --list
python scripts/analysis/run_mea_pipeline.py
```

The pipeline reads `MEA_data/` and writes final tables, statistical summaries,
and figures to `MEA_outputs/`. It performs a spatially matched local-population
analysis of UME and CME recordings, not one-to-one cell matching between
recordings.

## 8. Expected evidence

Canonical machine-readable results are in `docs/results/strict/`. The supplied
checkpoints should reproduce nearby values rather than byte-identical floating
point output on every hardware stack. A result is considered consistent when
the same protocol, split identities, candidate rows, evaluator backend, and
metric definitions are used and numerical deviations are limited to ordinary
hardware/library variation.
