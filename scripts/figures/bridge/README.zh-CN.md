# V5 MEA-H-RBCM 桥接分析

公开版只保留论文 V5 最终采用的图5定义，不包含早期绝对阈值图或探索性定义搜索。

模型侧使用冻结严格版 MultiCue H-RBCM 候选的纯周边调制项
`delta_RBCM = alpha * U * C`，MEA 侧使用 `FR_CME - FR_UME`。对每个
MEA“实验组 × 方向”样本和每张模型图像，先减去样本均值，再除以相对于该
均值的绝对残差中位数；大于 `+1`、小于 `-1`、位于两者之间分别定义为相对
增强、相对抑制和相对近中性。

先用正式 MEA 表、严格 MultiCue Anchor 预测、冻结候选和输入图像生成源表：

```bash
python scripts/analysis/reproduce_figure5_relative_statistics.py \
  --candidate-csv pretrained/multicue_strict/calibration_candidates.csv
```

再生成两个面板和组合图：

```bash
python scripts/figures/bridge/render_figure5_relative_panels.py
```

两个命令都只向 `edge_outputs/` 写入结果；GitHub 仓库不携带生成的源表或 PNG。
