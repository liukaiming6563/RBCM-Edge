# 工程目录结构

```text
RBCM-Edge/
  MEA_analysis/               正式 UME/CME 主分析与共享读取函数
  MEA_model/                  正式 MEA 作图和统计汇总模块
  edge_model/                 当前训练与评估框架
    rbcm/                     正式模型索引
    configs/rbcm/             各数据集正式配置
  src/rbcm_edge/              可导入源码包
  edge_data/
    official_rbcm/            统一 image/edge/GT/split 结构
    official_repro/           原生或近官方 evaluator 资产
  weights/rbcm/               最终 checkpoint 与校准行
  results/
    rbcm/                     正式得分、日志、预测和图片
    external/                 外部模型得分与来源证据
  paper_assets/rbcm/          简洁的论文写作副本
  scripts/
    experiments/              校准和只评估复现脚本
    analysis/                 表格、机制统计和模型文档
    checks/                   发布、权重与 dataloader 检查
  docs/
    edge/zh/                  最终中文边缘模型文档
    edge/en/                  内容一致的英文文档
    edge/raw/                 不改写的审计与来源证据
    manuscript/               正式论文写作区
  backup/legacy_edge/         所有废弃的边缘模型历史
  MEA_data/, MEA_outputs/     仅本地保存的 MEA 输入和生成结果
  release/                    自动生成的公开源码包和网盘包
```

公开源码分支只包含 `MEA_analysis`、`MEA_model`、`edge_model`、`src`、
必要脚本/配置和精简说明。数据、权重、生成结果、论文草稿与全部历史内容
仅保存在本地或单独的百度网盘包中。
