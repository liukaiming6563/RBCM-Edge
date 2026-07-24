# Formal MEA analysis

The final MEA analysis compares local RGC sorted-unit population trajectories
between UME and CME recordings from three paired retinal preparations. The
main script uses original coordinates, eight motion directions, the
approach-to-center movement window, grid-level population trajectories,
within-grid label permutation, and Benjamini-Hochberg FDR correction.

Run:

```bash
python MEA_analysis/run_MEA_final_UME_CME_trajectory_analysis.py
python scripts/analysis/run_mea_pipeline.py
```

Inputs are read from `MEA_data/`; generated artifacts are written to
`MEA_outputs/`. This is a spatially matched local-population analysis, not a
one-to-one paired-cell analysis across recordings.
