# 仓库结构

```text
RBCM-Edge/
  MEA_analysis/        正式 UME/CME 分析与共享数据读取代码
  MEA_model/           可复现的 MEA 作图脚本
  edge_model/          H-RBCM 训练、推理与评估框架
  src/rbcm_edge/       可导入的模型与损失函数包
  scripts/
    analysis/          正式结果表、图片与 MEA 流水线
    baselines/         统一边缘评估后端
    checks/            严格协议与发布检查
    data/              固定 MultiCue 划分准备
    experiments/       H-RBCM 校准与泛化评估
    release/           数据包验证与复现入口
  docs/
    results/           冻结的正式结果摘要与协议证据
```

大体量数据集和预训练权重通过 `DOWNLOADS.zh-CN.md` 中的网盘链接单独
发布。解压后按照 `REPRODUCE.zh-CN.md` 核验数据包并复现正式 MEA 与
H-RBCM 结果。
