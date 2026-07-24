# Repository Structure

```text
RBCM-Edge/
  MEA_analysis/               formal UME/CME analysis and shared loaders
  MEA_model/                  formal MEA figures and statistical summaries
  edge_model/                 active training and evaluation framework
    rbcm/                     formal-model index
    configs/rbcm/             canonical dataset configs
  src/rbcm_edge/              importable source package
  edge_data/
    official_rbcm/            normalized image/edge/GT/split contract
    official_repro/           native or near-official evaluator assets
  weights/rbcm/               final checkpoints and calibration rows
  results/
    rbcm/                     formal scores, logs, predictions, and figures
    external/                 external-model scores and provenance artifacts
  paper_assets/rbcm/          concise manuscript-writing copy
  scripts/
    experiments/              calibration and evaluation-only reproduction
    analysis/                 tables, mechanism statistics, model document
    checks/                   release, weight, and dataloader checks
  docs/
    edge/en/                  final English edge-model documentation
    edge/zh/                  equivalent Chinese documentation
    edge/raw/                 immutable audits and source evidence
    manuscript/               formal manuscript workspace
  backup/legacy_edge/         all obsolete edge-model history
  MEA_data/, MEA_outputs/     local-only MEA inputs and generated outputs
  release/                    generated public source and Baidu packages
```

The public source branch includes only `MEA_analysis`, `MEA_model`,
`edge_model`, `src`, required scripts/configs, and concise documentation.
Large data, checkpoints, generated results, manuscript drafts, and all history
remain local or in separate Baidu Netdisk packages.
