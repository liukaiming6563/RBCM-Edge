# Baidu Netdisk publication plan

## Upload after final packing

1. `RBCM-Edge-Checkpoints`
   - Selected BIPED, strict MultiCue, and strict NYUDv2 checkpoints.
   - Frozen validation candidates, configs, protocol manifests, and SHA-256 manifest.
   - Current unpacked size: 425,987,732 bytes (about 0.397 GiB).
   - This package is authored by the project and is the highest-priority upload.

2. `RBCM-Edge-Datasets`
   - Exact `edge_data/official_rbcm` and `edge_data/official_repro` trees.
   - Split files, processed GT, file index, and protocol hashes.
   - Current unpacked size: 17,236,266,110 bytes (about 16.05 GiB).
   - Upload the full image/GT archive only after confirming redistribution rights
     for every upstream dataset. If redistribution is not allowed, publish only
     the split/protocol package and provide official download plus preparation
     instructions.

3. `RBCM-Edge-MEA-Data`
   - Final MEA inputs and per-file SHA-256 manifest.
   - Current unpacked size: 8,858,482,461 bytes (about 8.25 GiB).
   - Upload only after ethics, consent, institutional, and source-data sharing
     approval. Otherwise use controlled access or a data-availability request.

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
hash, adapter command, and evaluator command. After packing each Netdisk archive,
compute a new archive-level SHA-256 and enter its link, extraction code, byte
size, and hash in `DOWNLOADS.md`.
