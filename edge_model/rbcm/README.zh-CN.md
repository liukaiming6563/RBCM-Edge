# RBCM

RBCM 是最终论文使用的边缘模型：

1. `anchor.py` 用 6.639 M 参数的 HED-lite 网络预测中心边缘 logit。
2. `calibrate.py` 计算原图 Sobel 能量和近/远环带摘要。
3. 验证协议选择的不确定性门控有符号残差校准中心 logit。
4. `plain_identity`、`no_surround`、`conv_control` 与
   `main_surround` 共用同一 anchor 和 evaluator。

## 代码

- Anchor：`src/rbcm_edge/models/networks/anchor.py`
- 模型工厂：`edge_model/models/build.py`
- 校准：`scripts/experiments/calibrate.py`
- 泛化：`scripts/experiments/evaluate_generalization.py`
- 统一近官方 evaluator：`scripts/baselines/evaluate_official_edges.py`

评估器恢复原始尺寸、执行 NMS、扫描固定阈值，并采用目标数据集特定的空间容差。
其中 dilation matcher 不是 BSDS 官方 Matlab 精确二分匹配器，因此论文必须写明
评估后端，不能把这里的分数与官方榜单分数当作完全相同的协议混合比较。

## 资产

- 配置：`edge_model/configs/rbcm`
- 权重与固定候选：`weights/rbcm`
- 结果：`results/rbcm`
- 文档：`docs/edge/zh`
