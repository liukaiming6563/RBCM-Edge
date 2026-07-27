# Baidu Netdisk publication plan

## Upload after final packing

1. `RBCM-Edge-Data.tar.gz`
   - Exact `edge_data/official_rbcm`, `edge_data/official_repro`, and
     `MEA_data` trees.
   - Final edge-model YAML files, bilingual protocol documentation, split
     files, processed GT, data indexes, and protocol hashes.
   - Uploaded archive: 18,443,403,941 bytes.
   - SHA-256:
     `d1c9dd8694dc16f1c190047c82b0e9689e5f781d86273af0ec95f2ac97dcfc60`.
   - The MEA subtree starts from Kilosort outputs and excludes
     `data.raw.h5` and converted `data.raw.bin`.
   - Upload the edge image/GT content only after confirming redistribution
     rights for every upstream dataset. Upload MEA data only after ethics,
     consent, institutional, and source-data sharing approval. If either class
     of data cannot be redistributed, use official download/preparation
     instructions or controlled access for that subtree.

2. `RBCM-Edge-Pretrained.tar.gz`
   - Selected BIPED, strict MultiCue, and strict NYUDv2 checkpoints.
   - Frozen validation candidates, portable and source configs, protocol
     manifests, training summaries, and per-file SHA-256 records.
   - Uploaded archive: 392,322,360 bytes.
   - SHA-256:
     `5afdbfcd066ffead18654578cc083df640b25d850b690cb8ef7343867352b65b`.
   - The package contains project checkpoints only; third-party external-model
     weights are excluded.

Upload `SHA256SUMS.txt` beside the two archives.

Baidu Netdisk: https://pan.baidu.com/s/1vdzNH616H7_eu80oCMXptg

Extraction code: `i8uc`

## Publish on GitHub, not Netdisk

- Public `release` source branch;
- model/data preparation, training, inference, evaluation, plotting, and MEA scripts;
- strict split lists and hashes;
- compact score tables and public protocol manifests;
- `REPRODUCE.md`, environment files, and download instructions.

## Do not redistribute directly

- external baseline checkpoints unless their upstream license explicitly permits it;
- private repository history, discarded checkpoints, debug dumps, raw server runs,
  or historical model variants;
- raw datasets or MEA recordings without redistribution authorization.

For external models, publish the model name, upstream URL, upstream checkpoint
hash, adapter command, and evaluator command. The public `DOWNLOADS.md`
contains the verified link, extraction code, byte sizes, and archive hashes.
