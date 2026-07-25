# 已测试环境

最终严格 NYUDv2 训练和本地复现检查使用：

- Windows 11，Python 3.10.16；
- PyTorch 2.7.1+cu126，CUDA 12.6，cuDNN 9.7；
- 严格本地训练使用 NVIDIA GeForce RTX 3070 Laptop GPU；
- NumPy 2.2.4，pandas 2.2.3，SciPy 1.15.3；
- scikit-image 0.25.2，scikit-learn 1.6.x；
- OpenCV 4.11.0，Pillow 11.1.0，PyYAML 6.0.2；
- matplotlib 3.10.1。

较早的论文用 BIPED 和 MultiCue checkpoint 在 NVIDIA RTX 4090 上训练。
实际耗时依赖硬件；建议先用发布的 checkpoint 复现分数，再考虑重新训练。

先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

评估前运行发布包 smoke test 和严格协议审计：

```bash
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained
python scripts/checks/audit_nyud_strict_protocol.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
```

不同 GPU、CUDA、cuDNN 和 NMS 实现可能产生小幅数值差异。复现时必须使用发布的
split 哈希、冻结校准候选、固定 `as_is` 预测方向和同一个评估后端。
