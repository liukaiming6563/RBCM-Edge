# 外部大文件下载

大文件不会上传 GitHub。

**百度网盘：** https://pan.baidu.com/s/1vdzNH616H7_eu80oCMXptg

**提取码：** `i8uc`

| 压缩包 | 内容 | 字节数 | SHA-256 |
|---|---|---:|---|
| `RBCM-Edge-Data.tar.gz` | 边缘数据集、Kilosort 衍生 MEA 数据、配置、划分文件和清单 | 18,443,403,941 | `d1c9dd8694dc16f1c190047c82b0e9689e5f781d86273af0ec95f2ac97dcfc60` |
| `RBCM-Edge-Pretrained.tar.gz` | 精选 H-RBCM 权重、冻结候选、配置和清单 | 392,322,360 | `5afdbfcd066ffead18654578cc083df640b25d850b690cb8ef7343867352b65b` |

同时下载 `SHA256SUMS.txt`。先核对两个压缩包，再按包内中英文 README
完成安装。MEA 数据从 Kilosort 输出开始，不包含 `data.raw.h5` 和
`data.raw.bin` 连续采集文件。
