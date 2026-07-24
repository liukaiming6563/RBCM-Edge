# 正式 MEA 分析

最终 MEA 分析比较三对配对视网膜记录中 UME 与 CME 条件下的局部 RGC
sorted-unit 群体响应轨迹。主分析固定使用原始坐标、八个运动方向、靠近中心
的运动窗口、网格级群体轨迹、网格内标签置换和 Benjamini-Hochberg FDR。

运行：

```bash
python MEA_analysis/run_MEA_final_UME_CME_trajectory_analysis.py
python scripts/analysis/run_mea_pipeline.py
```

输入从 `MEA_data/` 读取，结果写入 `MEA_outputs/`。该分析是空间匹配的局部
群体分析，不是在不同 recording 之间做同一细胞的一一配对。
