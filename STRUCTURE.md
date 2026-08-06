# Repository structure

```text
RBCM-Edge/
  MEA_analysis/        formal UME/CME analysis and shared data loaders
  MEA_model/           reproducible MEA figure scripts
  edge_model/          H-RBCM training, inference, and evaluation framework
  src/rbcm_edge/       importable model and loss package
  scripts/
    analysis/          evaluation, Figure 5 statistics, and MEA pipeline
    baselines/         shared edge-evaluation backend
    checks/            strict-protocol checks
    data/              fixed MultiCue split preparation
    experiments/       H-RBCM calibration and generalization evaluation
    figures/           Figure 5 and metric plotting code
    release/           source verification and reproduction entrypoints
```

Large datasets and pretrained checkpoints are distributed separately through
the link in `DOWNLOADS.md`. Manuscripts, rendered figures, result tables,
predictions, and historical code are not part of this branch. After extraction,
follow `REPRODUCE.md` to regenerate results locally.
