# Repository structure

```text
RBCM-Edge/
  MEA_analysis/        formal UME/CME analysis and shared data loaders
  MEA_model/           reproducible MEA figure scripts
  edge_model/          H-RBCM training, inference, and evaluation framework
  src/rbcm_edge/       importable model and loss package
  scripts/
    analysis/          formal result tables, figures, and MEA pipeline
    baselines/         shared edge-evaluation backend
    checks/            strict-protocol and release checks
    data/              fixed MultiCue split preparation
    experiments/       H-RBCM calibration and generalization evaluation
    release/           package verification and reproduction entrypoints
  docs/
    results/           frozen formal result summaries and protocol evidence
```

Large datasets and pretrained checkpoints are distributed separately through
the link in `DOWNLOADS.md`. After extraction, follow `REPRODUCE.md` to verify
the packages and reproduce the formal MEA and H-RBCM results.
