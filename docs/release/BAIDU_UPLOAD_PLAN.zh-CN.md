# 百度网盘发布清单

## 最终打包后上传

1. `RBCM-Edge-Data.tar.gz`
   - 完整的 `edge_data/official_rbcm`、`edge_data/official_repro` 和
     `MEA_data`；
   - 最终边缘模型 YAML、中英文协议说明、划分文件、处理后 GT、数据索引和协议
     哈希；
   - 已上传压缩包大小：18,443,403,941 字节；
   - SHA-256：
     `d1c9dd8694dc16f1c190047c82b0e9689e5f781d86273af0ec95f2ac97dcfc60`；
   - MEA 子树从 Kilosort 输出开始，不包含 `data.raw.h5` 和转换后的
     `data.raw.bin`；
   - 边缘图像和 GT 只有在逐项确认上游数据集允许再分发后才能上传；MEA 数据只有
     在伦理、知情同意、机构和来源数据共享要求允许后才能上传。若任一类数据不能
     直接再分发，应为对应子目录提供官方下载与准备方法，或采用受控访问。

2. `RBCM-Edge-Pretrained.tar.gz`
   - BIPED、严格 MultiCue 和严格 NYUDv2 的精选 checkpoint；
   - 冻结的验证集候选、可移植配置与原始运行配置、协议清单、训练汇总和逐文件
     SHA-256；
   - 已上传压缩包大小：392,322,360 字节；
   - SHA-256：
     `5afdbfcd066ffead18654578cc083df640b25d850b690cb8ef7343867352b65b`；
   - 只包含本项目 checkpoint，不包含第三方外部模型权重。

在两个压缩包旁同时上传 `SHA256SUMS.txt`。

百度网盘：https://pan.baidu.com/s/1vdzNH616H7_eu80oCMXptg

提取码：`i8uc`

## 放在 GitHub，不放网盘

- 公开 `release` 源码分支；
- 模型与数据准备、训练、推理、评估、绘图和 MEA 脚本；
- 严格划分列表及哈希；
- 小型得分表和公开协议清单；
- `REPRODUCE.md`、环境文件和下载说明。

## 不直接再分发

- 未明确获得上游许可的外部模型权重；
- 私有仓库历史、废弃 checkpoint、调试输出、原始服务器 run 和历史模型版本；
- 未获再分发授权的原始数据集或 MEA 记录。

外部模型只发布名称、上游链接、上游 checkpoint 哈希、适配命令和统一评估
命令。公开 `DOWNLOADS.md` 保存已经核验的链接、提取码、字节数和压缩包
SHA-256。
