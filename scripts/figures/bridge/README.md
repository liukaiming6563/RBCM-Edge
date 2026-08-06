# V5 MEA-to-H-RBCM bridge

The released bridge workflow implements only the Figure 5 definition accepted
in manuscript V5. It does not contain the earlier absolute-threshold figures or
the exploratory definition search.

The model signal is the pure surround-dependent correction
`delta_RBCM = alpha * U * C` from the frozen strict MultiCue H-RBCM candidate.
The MEA signal is `FR_CME - FR_UME`. Within every MEA group-direction sample
and every model image, the signed values are mean-centered and divided by the
median absolute residual about that mean. Relative enhancement, suppression,
and near-neutral states use symmetric thresholds at `+1`, `-1`, and the
interval between them.

First generate the source tables from the formal MEA table, strict MultiCue
Anchor predictions, frozen candidate file, and source images:

```bash
python scripts/analysis/reproduce_figure5_relative_statistics.py \
  --candidate-csv pretrained/multicue_strict/calibration_candidates.csv
```

Then render the two panels and combined layout:

```bash
python scripts/figures/bridge/render_figure5_relative_panels.py
```

Both commands write only under `edge_outputs/`. The repository does not ship
the generated source rows or rendered PNG files.
