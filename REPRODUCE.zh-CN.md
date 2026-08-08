# RBCM-Edge 复现说明

本文档覆盖论文最终使用的 MEA 分析和 H-RBCM 边缘检测实验，不包含已废弃的历史模型路线。

## 1. 获取代码和大文件

克隆公开仓库的 `release` 分支，并把单独发布的资源解压到代码根目录：

```text
RBCM-Edge/
  pretrained/   # 精选 checkpoint 与验证集冻结候选
  edge_data/    # 正式边缘检测数据和划分文件
  MEA_data/     # 经授权可共享的 MEA 输入
```

解压前必须用 `DOWNLOADS.md` 中的 SHA-256 核验压缩包。只有在原始数据许可、
伦理、知情同意和机构数据共享要求允许时，才能公开分发数据集或 MEA 数据。

公开 MEA 输入从 Kilosort 输出和下游衍生矩阵开始；连续采集文件
`data.raw.h5` 及转换后的 `data.raw.bin` 不在下载包内，也不是复现公开
轨迹分析、统计检验和结果图所必需的输入。

## 2. 创建环境

严格 NYUDv2 本地训练和验证环境为 Python 3.10.16、PyTorch
2.7.1+cu126、torchvision 0.22.1+cu126；严格 MultiCue 使用服务器镜像
`cuda128_torch280_py312`。先安装与本机 CUDA 匹配的 PyTorch，再安装经过
测试的其余依赖：

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

不同 GPU、CUDA、cuDNN 和 NMS 实现可能带来很小的数值差异。必须保持划分哈希、
候选文件、预测方向和评估后端一致。

## 3. 评估前完整性检查

```bash
python scripts/release/verify_paper_release.py --code-root .
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained --dataset all
python scripts/checks/audit_multicue_strict_protocol.py \
  --config edge_model/configs/rbcm/multicue_strict.yaml
python scripts/checks/audit_nyud_strict_protocol.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
python scripts/checks/run_rbcm_paper_preflight.py --check-data
```

正式协议为：

- BIPED：固定 170 train / 30 validation / 50 test；稳定性表使用三个固定重复划分；
- MultiCue：68 个训练源、12 个验证源、20 个一次性独立测试源，三组源图像完全互斥；
  checkpoint 和校准候选在读取测试集前冻结；
- NYUDv2 RGB：381 train / 414 validation / 654 held-out test；
- BSDS500 与 UDED：当前论文中作为评估或迁移目标，不是主要训练证据。

## 4. 用给定权重复现推理和得分

对图像目录导出四种匹配模式中的任意一种：

```bash
python edge_model/infer.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml \
  --checkpoint pretrained/nyudv2_strict/best.pt \
  --image-dir edge_data/official_rbcm/NYUDv2/image \
  --output-dir reproduced/nyudv2_main \
  --mode main_surround \
  --candidate-csv pretrained/nyudv2_strict/fixed_candidates.csv
```

用冻结的严格 NYUDv2 checkpoint 完成五目标评估：

```bash
python scripts/analysis/evaluate_nyud_strict_generalization.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml \
  --checkpoint pretrained/nyudv2_strict/best.pt \
  --formal-summary pretrained/nyudv2_strict/formal_summary.json \
  --run-tag nyudv2_strict_reproduction \
  --datasets BIPED Multicue NYUDv2 BSDS500 UDED \
  --device cuda --batch-size 1 --num-workers 2
```

论文评估器会恢复原图尺寸、按配置执行 NMS、扫描固定阈值，并按数据集使用固定空间
容差。它是统一的近官方 Python dilation matcher，不是 BSDS 官方 Matlab 精确二分
匹配器。不同后端的绝对分数不能在不标注协议的情况下直接混比。

## 5. 在本地重建结果表和指标图

```bash
python scripts/analysis/build_formal_result_index.py
python scripts/analysis/build_strict_protocol_tables.py
python scripts/analysis/build_requested_cross_domain_report.py
python scripts/analysis/build_v5_result_tables.py
python scripts/figures/edge/plot_joint_ablation_metrics.py
```

`build_v5_result_tables.py` 直接从正式评估输出重建 V5 稿件使用的七张 CSV 表，
脚本中没有硬编码论文分数。
当前正式表只使用 BIPED、严格 MultiCue 和严格 NYUDv2。重建旧 MultiCue 重合协议
的脚本必须显式传入归档参数，不能用于正文主表。
公开分支不跟踪任何已生成表格或图片。

## 6. 从头训练

`last.pt` 保存模型、优化器、调度器、AMP scaler、最佳分数、配置和 RNG 状态，因此
支持断点恢复：

```bash
python edge_model/train.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
python edge_model/train.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml --resume
```

MultiCue 使用 `edge_model/configs/rbcm/multicue_strict.yaml`，必须保持提供的严格
划分。校准候选只能用验证集选择，测试集必须在候选冻结后才能读取。`calibrate.py`
默认只运行验证集选择；论文测试应把冻结候选 CSV 交给 `evaluate_generalization.py`。
旧的组合入口只有显式传入 `--evaluate-test`，且候选文件完成哈希冻结后，才会读取测试集。严格 MultiCue
和 NYUDv2 均只提供一个随机种子，因此论文不能声称这两项具有多 seed 统计显著性。

## 7. 复现 MEA 分析

```bash
python scripts/analysis/run_mea_pipeline.py --list
python scripts/analysis/run_mea_pipeline.py
```

流水线读取 `MEA_data/`，并向 `MEA_outputs/` 写入最终表格、统计摘要和图片。该分析
比较 UME/CME 条件下空间匹配的局部群体，不是跨 recording 的单细胞一一配对。

### 复现 V5 图5分析

先用冻结的严格 MultiCue checkpoint 为 BIPED、MultiCue、NYUDv2 和 UDED 生成
Anchor 预测，再从正式 MEA 表、输入图像和验证集冻结候选计算 V5 源表：

```bash
python scripts/figures/edge/generate_multicue_strict_pr_curves.py \
  --checkpoint-root pretrained --device cuda
python scripts/analysis/reproduce_figure5_relative_statistics.py \
  --candidate-csv pretrained/multicue_strict/calibration_candidates.csv
python scripts/figures/bridge/render_figure5_relative_panels.py \
  --source-dir edge_outputs/rbcm/analyses/mea_rbcm_bridge/figure5_relative \
  --output-dir edge_outputs/rbcm/figures/mea_rbcm_bridge/figure5_relative
```

脚本使用纯 H-RBCM 项 `alpha * U * C`，状态分类不使用目标 GT；图中比较的是
样本内标准化后的相对空间异质性，不表示 MEA 与网络原始绝对效应分布相同。

## 8. 预期结果边界

重新生成的预测、表格、统计结果和图片都写入被忽略的输出目录，不随 GitHub 分支
发布。不同硬件上应复现接近的数值，而不要求所有浮点输出逐字节一致。只有在协议、
划分身份、冻结候选、评估后端和指标定义一致时，才可把数值视为同一实验的复现。
