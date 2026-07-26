# RBCM-Edge

本公开源码包只包含论文最终定版的两部分代码：

1. UME 与 CME 局部群体轨迹的视网膜 MEA 分析；
2. 采用环形周边到中心 logit 调制的 H-RBCM 边缘检测模型。

历史 ResNet、PiDiNet、A-G、探针和探索性方案均不进入公开包。仓库只包含
代码与配置；原始记录、图像数据集、checkpoint 和生成结果单独发布。

## 源码结构

- `MEA_analysis`：最终 MEA 轨迹主分析和共享读取函数；
- `MEA_model`：论文使用的 MEA 作图与统计汇总模块；
- `edge_model`：H-RBCM 训练、推理、校准和评估；
- `src/rbcm_edge`：可导入的 H-RBCM 实现；
- `scripts`：复现、完整性检查和流水线入口。
- `docs/results`：带协议标签的唯一正式论文数字索引。

Python 评估器采用统一的近官方流程：恢复原始尺寸、执行 NMS、扫描固定
阈值，并使用目标数据集特定的空间容差。其中 dilation matcher 不是
BSDS 官方 Matlab 精确二分匹配器。比较时必须统一使用同一后端，并在
论文表格中明确标注协议。

## 外部大文件包

将单独发布的目录放在代码根目录：

- `RBCM-Edge-Checkpoints` 中的 `pretrained/`；
- `RBCM-Edge-Datasets` 中的 `edge_data/`；
- `RBCM-Edge-MEA-Data` 中的 `MEA_data/`。

三个包上传完成后，在 `DOWNLOADS.md` 填写下载链接与压缩包哈希。

完整复现步骤见 `REPRODUCE.zh-CN.md`；对应英文版为 `REPRODUCE.md`。

## 环境

使用 Python 3.10 或更高版本，并安装与本机 CUDA 匹配的 PyTorch：

```bash
pip install -e .
pip install opencv-python-headless
```

## H-RBCM

H-RBCM 训练一个共享 HED-lite 中心边缘 anchor，并得到
`plain_identity`、`main_surround`、`no_surround`、`conv_control`
四种严格匹配输出。校准候选只在验证集选择，随后冻结。
正式数字入口为 `docs/results/formal_result_index.csv`；每行同时记录协议
角色和原始证据路径。

边缘检测复现命令与严格 NYUDv2 五目标评估命令见英文 README。

## MEA 分析

```bash
python scripts/analysis/run_mea_pipeline.py --list
python scripts/analysis/run_mea_pipeline.py
```

流水线读取 `MEA_data/`，并在 `MEA_outputs/` 下生成可复现表格、报告和图片。

## 许可证

构建脚本不会自动替作者选择代码许可证。数据集和 MEA 记录的公开传播仍受
原始来源、伦理与数据使用要求约束，上传前必须逐项核对。
