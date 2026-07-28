# H-RBCM 严格协议结果

本目录由 `scripts/analysis/build_strict_protocol_tables.py` 生成，只汇总可用于正式论文的独立协议。

## 主要同域结论

- BIPED：三套固定 170/30/50 划分。H-RBCM 相对逐指标最强控制的变化为 ODS +0.78、OIS +0.77、AP +0.48 个百分点。
- MultiCue：严格 68 个训练源、12 个验证源、20 个一次性独立测试源。H-RBCM 的 ODS/OIS 分别提高 +2.22/+1.76 个百分点，AP 变化 -0.92 个百分点。该结果支持 F-score 改善，但必须同时报告 AP 权衡。
- NYUDv2：严格 381/414/654 零重叠划分。H-RBCM 的 ODS/OIS/AP 分别提高 +0.82/+0.74/+0.94 个百分点。

## 严格 NYUDv2 外部比较

在共享目标 evaluator 下，NYUDv2 同域 H-RBCM 为 0.84291/0.85342/0.87266，PiDiNet NYUDv2 RGB released checkpoint 为 0.83224/0.84163/0.86016。H-RBCM 三项分别变化 +1.07/+1.18/+1.25 个百分点。

跨域外部比较并非全面领先：BIPED 三项领先；UDED 的 ODS/OIS 领先而 AP 小幅落后；MultiCue 和 BSDS500 落后。因此论文只能声称严格 NYUDv2 同域优势和部分域迁移优势，不能声称普遍优于 PiDiNet。

## 文件

- `same_domain_strict.csv`：三个主要独立协议的四模式同域结果。
- `main_vs_strongest_control.csv`：主模型相对逐指标最强匹配控制的变化。
- `nyudv2_five_target_ablation.csv`：严格 NYUDv2 checkpoint 与候选冻结后的五目标四模式矩阵。
- `nyudv2_vs_pidinet.csv`：NYUDv2 训练来源匹配、目标 evaluator 统一的外部比较。
