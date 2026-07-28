# RBCM-Edge

本公开源码包只包含论文最终定版的两部分代码：

1. UME 与 CME 局部群体轨迹的视网膜 MEA 分析；
2. 采用环形周边到中心 logit 调制的 H-RBCM 边缘检测模型。

仓库只包含代码与配置；原始记录、图像数据集、checkpoint 和生成结果单独
发布。

## 从本仓库开始复现

对外只需要分享本 GitHub 仓库链接。本页已经集中提供源码、资源包下载、
完整性哈希、环境说明和完整复现文档的入口。

1. 克隆公开仓库：

   ```bash
   git clone https://github.com/liukaiming6563/RBCM-Edge.git
   cd RBCM-Edge
   ```

2. 从百度网盘下载两个大文件资源包：

   - 链接：<https://pan.baidu.com/s/1vdzNH616H7_eu80oCMXptg>
   - 提取码：`i8uc`

   | 压缩包 | 内容 | SHA-256 |
   |---|---|---|
   | `RBCM-Edge-Data.tar.gz` | 边缘检测数据集、固定划分、评估资源和 Kilosort 衍生 MEA 输入 | `d1c9dd8694dc16f1c190047c82b0e9689e5f781d86273af0ec95f2ac97dcfc60` |
   | `RBCM-Edge-Pretrained.tar.gz` | 精选 H-RBCM 权重、冻结的验证集候选、配置与清单 | `5afdbfcd066ffead18654578cc083df640b25d850b690cb8ef7343867352b65b` |

3. 按上述哈希校验并解压，把 `edge_data/`、`MEA_data/` 和
   `pretrained/` 放到仓库根目录。

4. 按 [REPRODUCE.zh-CN.md](REPRODUCE.zh-CN.md) 完成环境创建、协议审计、
   checkpoint 推理评估、论文表图重建和 MEA 分析复现。英文说明见
   [REPRODUCE.md](REPRODUCE.md)。

压缩包大小、包内目录和不同系统的校验方法也保存在
[DOWNLOADS.zh-CN.md](DOWNLOADS.zh-CN.md)。

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

大文件按两个压缩包发布：

- `RBCM-Edge-Data.tar.gz`：`edge_data/`、从 Kilosort 输出开始的
  `MEA_data/`、协议配置与完整性清单；
- `RBCM-Edge-Pretrained.tar.gz`：精选 H-RBCM 权重、验证集冻结候选、
  配置与完整性清单。

同一百度网盘链接、提取码、文件大小和 SHA-256 也记录在
`DOWNLOADS.zh-CN.md`。

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
