# Tested Environment

The final strict NYUDv2 training and local reproduction checks were run with:

- Windows 11, Python 3.10.16;
- PyTorch 2.7.1+cu126, CUDA 12.6, cuDNN 9.7;
- NVIDIA GeForce RTX 3070 Laptop GPU for the strict local run;
- NumPy 2.2.4, pandas 2.2.3, SciPy 1.15.3;
- scikit-image 0.25.2, scikit-learn 1.6.x;
- OpenCV 4.11.0, Pillow 11.1.0, PyYAML 6.0.2;
- matplotlib 3.10.1.

The earlier paper-facing BIPED and MultiCue checkpoints were trained on an
NVIDIA RTX 4090. Exact wall-clock speed is hardware dependent; model scores
should be reproduced from the supplied checkpoints before retraining.

Install a PyTorch build that matches the host CUDA runtime, then install the
remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the release smoke test and protocol audit before evaluation:

```bash
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained
python scripts/checks/audit_nyud_strict_protocol.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
```

Small numerical differences can arise from GPU, CUDA, cuDNN, and NMS
implementations. Use the supplied split hashes, fixed calibration candidates,
fixed `as_is` prediction orientation, and the same evaluator backend.
