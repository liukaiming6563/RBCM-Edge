# 仓库结构

```text
RBCM-Edge/
  MEA_analysis/        正式 UME/CME 分析与共享数据读取代码
  MEA_model/           可复现的 MEA 作图脚本
  edge_model/          H-RBCM 训练、推理与评估框架
  src/rbcm_edge/       可导入的模型与损失函数包
  scripts/
    analysis/          评估、图5统计与 MEA 流水线
    baselines/         统一边缘评估后端
    checks/            严格协议检查
    data/              固定 MultiCue 划分准备
    experiments/       H-RBCM 校准与泛化评估
    figures/           图5与指标作图代码
    release/           源码校验与复现入口
```

大体量数据集和预训练权重通过 `DOWNLOADS.zh-CN.md` 中的网盘链接单独
发布。论文、已生成图片、结果表、预测结果和历史代码不属于本分支。解压后
按照 `REPRODUCE.zh-CN.md` 在本地重新生成结果。
