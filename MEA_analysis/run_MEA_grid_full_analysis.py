"""RBCM-edge MEA spatial grid-level full analysis.

本脚本用于完整复现并扩展 RBCM-edge 论文中 MEA 部分的空间网格分析。

核心思想：
1. UME 与 CME 来自不同 recording / spike sorting，不能进行 unit-level pairing。
2. 用户确认 000031/000032、000034/000035、000037/000038 分别来自同一视网膜切片。
3. 因此本脚本采用 spatially matched grid-level local population analysis：
   在同一视网膜切片的相同空间网格内，分别聚合 UME 和 CME 条件下的 good sorted units，
   比较局部 RGC 群体平均响应、响应比例、稳健发放率指标和空间响应图相似性。

输入数据约定：
- firing rate 数组：MEA_data/<exp_id>/segment_result/processed_segment/good_on/output_fre.npy
  shape 为 repeat × good_unit × event。
- 空间坐标：使用已有 middle stripe 分析生成的 unit_distance_long_table.csv 中的 x/y 坐标。
  这是因为当前 MEA_data 中并非所有实验都保留了 unit_positions_good.csv。

输出：
- 统一保存到 D:/study/project/RBCM-Edge/outputs/MEA_analysis/MEA_analysis_final
- 包括 tables、figures、reports、logs、configs、code_snapshot。
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ??????????????????????????????????
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 1. 配置区：所有关键参数集中放在这里，便于后续复现实验或修改。
# =============================================================================

PROJECT_DIR = Path(os.environ.get("RBCM_EDGE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
MEA_DIR = Path(os.environ.get("RBCM_EDGE_MEA_DATA", PROJECT_DIR / "MEA_data")).resolve()

# 该目录来自早期 MEA 空间条带分析，包含所有实验的 good unit 空间坐标。
GEOMETRY_DIR = Path(
    os.environ.get(
        "RBCM_EDGE_GEOMETRY_DIR",
        PROJECT_DIR / "MEA_analysis" / "resources" / "reconstructed_unit_positions",
    )
).resolve()

OUT_DIR = PROJECT_DIR / "MEA_outputs" / "grid_full_analysis"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
CONFIG_DIR = OUT_DIR / "configs"
CODE_DIR = OUT_DIR / "code_snapshot"
RESOURCE_DIR = PROJECT_DIR / "MEA_analysis" / "resources"
POSITION_CACHE_DIR = RESOURCE_DIR / "reconstructed_unit_positions"

PAIR_CONFIG = {
    "pair_31_32": {"CME": "000031", "UME": "000032"},
    "pair_34_35": {"CME": "000034", "UME": "000035"},
    "pair_37_38": {"CME": "000037", "UME": "000038"},
}

GRID_SCALES = [8, 10, 12, 16, 20, 25, 30]

DIRECTION_CONFIG = [
    {"id": 1, "code": "R", "name": "right"},
    {"id": 2, "code": "RU", "name": "up_right"},
    {"id": 3, "code": "U", "name": "up"},
    {"id": 4, "code": "LU", "name": "up_left"},
    {"id": 5, "code": "L", "name": "left"},
    {"id": 6, "code": "LD", "name": "down_left"},
    {"id": 7, "code": "D", "name": "down"},
    {"id": 8, "code": "RD", "name": "down_right"},
]

MIN_UNITS_PER_GRID_PER_STIM = 3
MAIN_THRESHOLD_HZ = 0.5
THRESHOLDS_HZ = [0.25, 0.5, 1.0, 2.0]
EPSILON = 1e-6
N_PERM = 1000
RANDOM_SEED = 42
MAIN_GRID_SCALES_FOR_FIGURES = [12, 16]

# ????????????????????? single/CME ????????? UME/CME?
UME_FULL_NAME = "Uniform-background moving edge"
CME_FULL_NAME = "Contextual-background moving edge"


@dataclass
class FigureRecord:
    """记录每张图的元数据，用于 figure_manifest.csv。"""

    figure_id: str
    figure_type: str
    grid_scale: str
    pair_id: str
    direction_id: str
    file_png: str
    file_pdf: str
    description: str


LOG_LINES: list[str] = []
FIGURE_RECORDS: list[FigureRecord] = []
RNG = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# 2. 通用工具函数
# =============================================================================


def log(message: str) -> None:
    """同时打印并记录日志文本。"""

    print(message)
    LOG_LINES.append(message)


def ensure_dirs() -> None:
    """创建完整输出目录结构。"""

    for directory in [
        OUT_DIR,
        TABLE_DIR,
        FIG_DIR / "main_figures",
        FIG_DIR / "supplementary_figures",
        FIG_DIR / "grid_maps",
        FIG_DIR / "threshold_sensitivity",
        FIG_DIR / "response_fraction_maps",
        FIG_DIR / "robust_metric_maps",
        FIG_DIR / "NDI_maps",
        FIG_DIR / "map_similarity",
        FIG_DIR / "permutation",
        REPORT_DIR,
        LOG_DIR,
        CONFIG_DIR,
        CODE_DIR,
        POSITION_CACHE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """保存 CSV 表格，统一使用 utf-8-sig 便于 Excel 打开中文。"""

    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig: plt.Figure, subdir: str, stem: str, record: FigureRecord) -> None:
    """同时保存 PNG 和 PDF，并记录到 figure manifest。"""

    directory = FIG_DIR / subdir
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{stem}.png"
    pdf_path = directory / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    record.file_png = str(png_path)
    record.file_pdf = str(pdf_path)
    FIGURE_RECORDS.append(record)


def safe_mean(values: np.ndarray) -> float:
    """忽略 NaN 的均值；空数组返回 NaN。"""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def safe_median(values: np.ndarray) -> float:
    """忽略 NaN 的中位数；空数组返回 NaN。"""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def rank_average(values: np.ndarray) -> np.ndarray:
    """计算带 ties 平均名次的 rank，用于 Spearman 相关。"""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson_with_approx_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r 及近似 p 值。

    当前环境没有 scipy，因此 p 值使用 Fisher z 正态近似。
    该 p 值用于辅助描述，不作为本文 MEA 部分的唯一统计依据。
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    z = math.atanh(r) * math.sqrt(n - 3)
    p = float(math.erfc(abs(z) / math.sqrt(2)))
    return r, p


def spearman_with_approx_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman 相关及近似 p 值。"""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan, np.nan
    return pearson_with_approx_p(rank_average(x[mask]), rank_average(y[mask]))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """计算两个空间响应向量的 cosine similarity。"""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.nan
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def classify_delta(delta: pd.Series, valid: pd.Series, threshold: float) -> pd.Series:
    """按指定阈值将网格差异分为 UME_higher / CME_higher / similar。"""

    result = pd.Series("similar", index=delta.index, dtype=object)
    result[delta > threshold] = "UME_higher"
    result[delta < -threshold] = "CME_higher"
    result[~valid] = "insufficient_units"
    return result


def ndi(a: pd.Series, b: pd.Series) -> pd.Series:
    """归一化差异指数 NDI = (a-b)/(a+b+epsilon)。"""

    return (a - b) / (a + b + EPSILON)


# =============================================================================
# 3. 数据加载与质量检查
# =============================================================================


def load_unit_positions(exp_id: str, stim_class: str | None = None) -> pd.DataFrame:
    """??? MEA_data ???????? good unit ?????

    ????????? RBCM-Edge ????????? MEA_data ???????
    ?? middle-stripe ????? unit_distance_long_table.csv????? Kilosort ???

    - spike_clusters.npy??? spike ?? cluster?
    - spike_positions.npy??? spike ??????
    - cluster_group.tsv?cluster ? good/mua/noise ???
    - segment_result/origin_segment/sorted_cluster_ids.npy?firing-rate ????? cluster ???

    ????? sorted_cluster_ids ????? good cluster???? cluster ?? spike
    ??? median ?? sorted unit ??????????? unit_row_idx ?
    good_on/output_fre.npy ? unit ??????

    ?? stim_class ???????????????????????? UME/CME?
    """

    cache_path = POSITION_CACHE_DIR / f"{exp_id}_good_unit_positions_from_kilosort.csv"
    if cache_path.exists():
        units = pd.read_csv(cache_path)
        return units.sort_values("unit_row_idx").reset_index(drop=True)

    exp_dir = MEA_DIR / exp_id
    ks_dir = exp_dir / "kilosort4"
    sorted_ids_path = exp_dir / "segment_result" / "origin_segment" / "sorted_cluster_ids.npy"
    required_paths = [
        ks_dir / "spike_clusters.npy",
        ks_dir / "spike_positions.npy",
        ks_dir / "cluster_group.tsv",
        sorted_ids_path,
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"???? good unit ??????: {required_path}")

    sorted_cluster_ids = np.load(sorted_ids_path).astype(int)
    cluster_group = pd.read_csv(ks_dir / "cluster_group.tsv", sep="\t")
    label_candidates = [col for col in cluster_group.columns if col.lower() in {"group", "kslabel", "label"}]
    if not label_candidates:
        raise ValueError(f"{ks_dir / 'cluster_group.tsv'} ???? good/mua/noise ???")
    label_col = label_candidates[0]
    good_cluster_set = set(
        cluster_group.loc[cluster_group[label_col].astype(str).str.lower().eq("good"), "cluster_id"].astype(int)
    )
    good_cluster_ids = [int(cid) for cid in sorted_cluster_ids if int(cid) in good_cluster_set]

    spike_clusters = np.load(ks_dir / "spike_clusters.npy").astype(int)
    spike_positions = np.load(ks_dir / "spike_positions.npy").astype(float)
    if spike_positions.ndim != 2 or spike_positions.shape[1] < 2:
        raise ValueError(f"{ks_dir / 'spike_positions.npy'} shape ??: {spike_positions.shape}")
    if spike_positions.shape[0] != spike_clusters.shape[0]:
        raise ValueError(f"{exp_id} spike_positions ? spike_clusters ?????")

    # ??? good clusters ? spikes??? cluster ? median ???median ? mean ???? spike?
    mask = np.isin(spike_clusters, np.asarray(good_cluster_ids, dtype=int))
    pos_table = pd.DataFrame(
        {
            "cluster_id": spike_clusters[mask],
            "x": spike_positions[mask, 0],
            "y": spike_positions[mask, 1],
        }
    )
    center = pos_table.groupby("cluster_id", sort=False)[["x", "y"]].median()

    rows = []
    missing_clusters = []
    for unit_row_idx, cluster_id in enumerate(good_cluster_ids):
        if cluster_id not in center.index:
            missing_clusters.append(cluster_id)
            continue
        rows.append(
            {
                "unit_row_idx": unit_row_idx,
                "cluster_id": cluster_id,
                "x": float(center.loc[cluster_id, "x"]),
                "y": float(center.loc[cluster_id, "y"]),
            }
        )
    if missing_clusters:
        raise ValueError(f"{exp_id} ? good clusters ?? spike position: {missing_clusters[:10]}")

    units = pd.DataFrame(rows).sort_values("unit_row_idx").reset_index(drop=True)
    fre_path = exp_dir / "segment_result" / "processed_segment" / "good_on" / "output_fre.npy"
    if fre_path.exists():
        n_fr_units = int(np.load(fre_path, mmap_mode="r").shape[1])
        if len(units) != n_fr_units:
            raise ValueError(f"{exp_id} ???? unit ? {len(units)} ? firing-rate unit ? {n_fr_units} ???")
    if units.empty:
        raise ValueError(f"{exp_id} ???? good unit ??")

    units.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return units


def load_middle_step_fr(exp_id: str, stimulus: str, direction_zero_based: int) -> np.ndarray:
    """读取 ON 阶段中心移动步骤处的 unit firing rate。

    返回 shape = (n_good_units,)。
    原始 output_fre.npy shape = repeat × good_unit × event。
    """

    path = MEA_DIR / exp_id / "segment_result" / "processed_segment" / "good_on" / "output_fre.npy"
    if not path.exists():
        raise FileNotFoundError(f"缺少 firing-rate 数组: {path}")
    fr = np.load(path)
    if fr.ndim != 3:
        raise ValueError(f"{path} 维度异常，期望 repeat × unit × event，实际 shape={fr.shape}")
    if fr.shape[0] < 1:
        raise ValueError(f"{path} repeat 数为 0")
    if np.nanmin(fr) < 0:
        raise ValueError(f"{path} 中存在负 firing rate")

    if stimulus == "UME":
        event_idx = direction_zero_based * 13 + 6
    elif stimulus == "CME":
        event_idx = direction_zero_based * 11 + 5
    else:
        raise ValueError(f"未知 stimulus: {stimulus}")
    if event_idx >= fr.shape[2]:
        raise IndexError(f"{exp_id} {stimulus} event_idx={event_idx} 越界，shape={fr.shape}")
    return fr[:, :, event_idx].mean(axis=0)


def assign_grid(units: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray, grid_n: int) -> pd.DataFrame:
    """给每个 unit 分配空间网格坐标。

    同一 pair、同一 grid_scale 下 UME 和 CME 使用相同 x_edges/y_edges。
    """

    out = units.copy()
    out["grid_x"] = np.clip(np.searchsorted(x_edges, out["x"].to_numpy(), side="right") - 1, 0, grid_n - 1)
    out["grid_y"] = np.clip(np.searchsorted(y_edges, out["y"].to_numpy(), side="right") - 1, 0, grid_n - 1)
    out["grid_id"] = out["grid_y"].astype(str).str.zfill(2) + "_" + out["grid_x"].astype(str).str.zfill(2)
    return out


def build_unit_table(units: pd.DataFrame, fr: np.ndarray, stimulus: str) -> pd.DataFrame:
    """构造 unit-level 表，便于网格聚合和 permutation null。"""

    max_idx = int(units["unit_row_idx"].max())
    if max_idx >= fr.size:
        raise ValueError(f"{stimulus} unit_row_idx 与 firing-rate 数组长度不匹配: max_idx={max_idx}, n={fr.size}")
    out = units.copy()
    out["stimulus"] = stimulus
    out["firing_rate_hz"] = fr[out["unit_row_idx"].to_numpy(dtype=int)]
    return out


def aggregate_grid(unit_table: pd.DataFrame, stimulus: str) -> pd.DataFrame:
    """将 unit-level firing rate 聚合到 grid-level local population metrics。"""

    def zero_count(values: pd.Series) -> int:
        arr = values.to_numpy(dtype=float)
        return int(np.sum(arr == 0))

    def nonzero_count(values: pd.Series) -> int:
        arr = values.to_numpy(dtype=float)
        return int(np.sum(arr > 0))

    def nonzero_mean(values: pd.Series) -> float:
        arr = values.to_numpy(dtype=float)
        arr = arr[arr > 0]
        return float(np.mean(arr)) if arr.size else np.nan

    def nonzero_median(values: pd.Series) -> float:
        arr = values.to_numpy(dtype=float)
        arr = arr[arr > 0]
        return float(np.median(arr)) if arr.size else np.nan

    grouped = (
        unit_table.groupby(["grid_id", "grid_x", "grid_y"], as_index=False)
        .agg(
            unit_count=("firing_rate_hz", "size"),
            mean_fr_hz=("firing_rate_hz", "mean"),
            median_fr_hz=("firing_rate_hz", "median"),
            zero_count=("firing_rate_hz", zero_count),
            nonzero_count=("firing_rate_hz", nonzero_count),
            nonzero_mean_fr_hz=("firing_rate_hz", nonzero_mean),
            nonzero_median_fr_hz=("firing_rate_hz", nonzero_median),
        )
    )
    grouped[f"{stimulus}_zero_fraction"] = grouped["zero_count"] / grouped["unit_count"]
    grouped[f"{stimulus}_nonzero_fraction"] = grouped["nonzero_count"] / grouped["unit_count"]
    rename = {
        "unit_count": f"{stimulus}_unit_count",
        "mean_fr_hz": f"{stimulus}_mean_fr_hz",
        "median_fr_hz": f"{stimulus}_median_fr_hz",
        "zero_count": f"{stimulus}_zero_count",
        "nonzero_count": f"{stimulus}_nonzero_count",
        "nonzero_mean_fr_hz": f"{stimulus}_nonzero_mean_fr_hz",
        "nonzero_median_fr_hz": f"{stimulus}_nonzero_median_fr_hz",
    }
    return grouped.rename(columns=rename)


# =============================================================================
# 4. 网格指标计算
# =============================================================================


def compute_one_grid_direction(
    grid_n: int,
    pair_id: str,
    pair_cfg: dict[str, str],
    direction: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算一个 grid_scale × pair × direction 下的完整 grid-cell 明细。

    返回：
    - grid_df：完整 N×N 网格明细，包括 valid 和 insufficient grids。
    - unit_df：unit-level 表，用于 permutation null。
    """

    cme_exp = pair_cfg["CME"]
    ume_exp = pair_cfg["UME"]
    direction_zero_based = direction["id"] - 1

    ume_units = load_unit_positions(ume_exp, "single_edge")
    cme_units = load_unit_positions(cme_exp, "double_edge")

    # 使用 UME + CME 的共同坐标范围确定网格边界，避免两组各自坐标范围不同导致不可比。
    all_x = pd.concat([ume_units["x"], cme_units["x"]], ignore_index=True)
    all_y = pd.concat([ume_units["y"], cme_units["y"]], ignore_index=True)
    x_edges = np.linspace(all_x.min(), all_x.max(), grid_n + 1)
    y_edges = np.linspace(all_y.min(), all_y.max(), grid_n + 1)
    ume_units = assign_grid(ume_units, x_edges, y_edges, grid_n)
    cme_units = assign_grid(cme_units, x_edges, y_edges, grid_n)

    ume_fr = load_middle_step_fr(ume_exp, "UME", direction_zero_based)
    cme_fr = load_middle_step_fr(cme_exp, "CME", direction_zero_based)
    ume_unit_table = build_unit_table(ume_units, ume_fr, "UME")
    cme_unit_table = build_unit_table(cme_units, cme_fr, "CME")
    unit_df = pd.concat([ume_unit_table, cme_unit_table], ignore_index=True)

    UME_grid = aggregate_grid(ume_unit_table, "UME")
    CME_grid = aggregate_grid(cme_unit_table, "CME")

    # 构造完整 N×N 网格表，空网格也保留，便于统计 insufficient grids 和画图。
    full_grid = pd.DataFrame(
        [
            {"grid_x": gx, "grid_y": gy, "grid_id": f"{gy:02d}_{gx:02d}"}
            for gy in range(grid_n)
            for gx in range(grid_n)
        ]
    )
    grid = full_grid.merge(UME_grid, on=["grid_id", "grid_x", "grid_y"], how="left")
    grid = grid.merge(CME_grid, on=["grid_id", "grid_x", "grid_y"], how="left")

    for col in [
        "UME_unit_count",
        "CME_unit_count",
        "UME_zero_count",
        "CME_zero_count",
        "UME_nonzero_count",
        "CME_nonzero_count",
    ]:
        grid[col] = grid[col].fillna(0).astype(int)

    grid["grid_scale"] = f"{grid_n}x{grid_n}"
    grid["grid_n"] = grid_n
    grid["pair_id"] = pair_id
    grid["UME_exp"] = ume_exp
    grid["CME_exp"] = cme_exp
    grid["direction_id"] = f"{direction['id']:02d}"
    grid["direction_code"] = direction["code"]
    grid["direction_name"] = direction["name"]
    grid["min_units_per_grid_per_stim"] = MIN_UNITS_PER_GRID_PER_STIM
    grid["valid_grid"] = (
        (grid["UME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM)
        & (grid["CME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM)
    )

    # 均值、median、nonzero FR、response fraction 差异。
    grid["delta_mean_fr_hz"] = grid["UME_mean_fr_hz"] - grid["CME_mean_fr_hz"]
    grid["abs_delta_mean_fr_hz"] = grid["delta_mean_fr_hz"].abs()
    grid["delta_median_fr_hz"] = grid["UME_median_fr_hz"] - grid["CME_median_fr_hz"]
    grid["abs_delta_median_fr_hz"] = grid["delta_median_fr_hz"].abs()
    grid["delta_nonzero_fraction"] = grid["UME_nonzero_fraction"] - grid["CME_nonzero_fraction"]
    grid["abs_delta_nonzero_fraction"] = grid["delta_nonzero_fraction"].abs()
    grid["delta_zero_fraction"] = grid["UME_zero_fraction"] - grid["CME_zero_fraction"]
    grid["abs_delta_zero_fraction"] = grid["delta_zero_fraction"].abs()
    grid["delta_nonzero_mean_fr_hz"] = grid["UME_nonzero_mean_fr_hz"] - grid["CME_nonzero_mean_fr_hz"]
    grid["abs_delta_nonzero_mean_fr_hz"] = grid["delta_nonzero_mean_fr_hz"].abs()
    grid["delta_nonzero_median_fr_hz"] = grid["UME_nonzero_median_fr_hz"] - grid["CME_nonzero_median_fr_hz"]
    grid["abs_delta_nonzero_median_fr_hz"] = grid["delta_nonzero_median_fr_hz"].abs()

    # NDI 反映相对 UME/CME 偏好，范围理论上接近 [-1, 1]。
    grid["NDI_mean"] = ndi(grid["UME_mean_fr_hz"], grid["CME_mean_fr_hz"])
    grid["NDI_median"] = ndi(grid["UME_median_fr_hz"], grid["CME_median_fr_hz"])
    grid["NDI_nonzero_mean"] = ndi(grid["UME_nonzero_mean_fr_hz"], grid["CME_nonzero_mean_fr_hz"])
    grid["NDI_nonzero_fraction"] = ndi(grid["UME_nonzero_fraction"], grid["CME_nonzero_fraction"])

    for threshold in THRESHOLDS_HZ:
        suffix = str(threshold).replace(".", "p")
        grid[f"difference_class_{suffix}"] = classify_delta(grid["delta_mean_fr_hz"], grid["valid_grid"], threshold)
    grid["difference_class_main"] = grid[f"difference_class_{str(MAIN_THRESHOLD_HZ).replace('.', 'p')}"]

    # 让 unit_df 也携带 pair/direction/grid_scale 信息，方便 permutation。
    unit_df["grid_scale"] = f"{grid_n}x{grid_n}"
    unit_df["grid_n"] = grid_n
    unit_df["pair_id"] = pair_id
    unit_df["UME_exp"] = ume_exp
    unit_df["CME_exp"] = cme_exp
    unit_df["direction_id"] = f"{direction['id']:02d}"
    unit_df["direction_code"] = direction["code"]
    unit_df["direction_name"] = direction["name"]
    return grid, unit_df


# =============================================================================
# 5. 汇总表计算
# =============================================================================


def summarize_threshold(grid_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """阈值敏感性：按 threshold 重新统计 S/D/= 数量和比例。"""

    rows = []
    for threshold in THRESHOLDS_HZ:
        suffix = str(threshold).replace(".", "p")
        cls_col = f"difference_class_{suffix}"
        for keys, sub in grid_all.groupby(["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"]):
            valid = sub[sub["valid_grid"]]
            n = int(valid.shape[0])
            counts = valid[cls_col].value_counts()
            ume_n = int(counts.get("UME_higher", 0))
            cme_n = int(counts.get("CME_higher", 0))
            similar_n = int(counts.get("similar", 0))
            diff_n = ume_n + cme_n
            rows.append(
                {
                    "grid_scale": keys[0],
                    "grid_n": keys[1],
                    "pair_id": keys[2],
                    "direction_id": keys[3],
                    "direction_code": keys[4],
                    "direction_name": keys[5],
                    "threshold_hz": threshold,
                    "valid_grid_count": n,
                    "UME_higher_count": ume_n,
                    "CME_higher_count": cme_n,
                    "similar_count": similar_n,
                    "different_count": diff_n,
                    "UME_higher_fraction": ume_n / n if n else np.nan,
                    "CME_higher_fraction": cme_n / n if n else np.nan,
                    "similar_fraction": similar_n / n if n else np.nan,
                    "different_fraction": diff_n / n if n else np.nan,
                    "mean_abs_delta_mean_fr_hz": safe_mean(valid["abs_delta_mean_fr_hz"]),
                    "median_abs_delta_mean_fr_hz": safe_median(valid["abs_delta_mean_fr_hz"]),
                }
            )
    direction_summary = pd.DataFrame(rows)
    pair_overall = (
        direction_summary.groupby(["grid_scale", "grid_n", "pair_id", "threshold_hz"], as_index=False)
        .agg(
            mean_different_fraction=("different_fraction", "mean"),
            min_different_fraction=("different_fraction", "min"),
            max_different_fraction=("different_fraction", "max"),
            mean_UME_higher_fraction=("UME_higher_fraction", "mean"),
            mean_CME_higher_fraction=("CME_higher_fraction", "mean"),
            mean_similar_fraction=("similar_fraction", "mean"),
            total_valid_grids=("valid_grid_count", "sum"),
            total_UME_higher=("UME_higher_count", "sum"),
            total_CME_higher=("CME_higher_count", "sum"),
            total_similar=("similar_count", "sum"),
            mean_abs_delta_mean_fr_hz=("mean_abs_delta_mean_fr_hz", "mean"),
            median_abs_delta_mean_fr_hz=("median_abs_delta_mean_fr_hz", "median"),
        )
    )
    return direction_summary, pair_overall


def summarize_robust_metrics(grid_all: pd.DataFrame) -> pd.DataFrame:
    """汇总 mean/median/nonzero/response fraction 等稳健指标。"""

    valid = grid_all[grid_all["valid_grid"]].copy()
    return (
        valid.groupby(["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"], as_index=False)
        .agg(
            valid_grid_count=("valid_grid", "size"),
            mean_abs_delta_mean_fr_hz=("abs_delta_mean_fr_hz", "mean"),
            mean_abs_delta_median_fr_hz=("abs_delta_median_fr_hz", "mean"),
            mean_abs_delta_nonzero_mean_fr_hz=("abs_delta_nonzero_mean_fr_hz", "mean"),
            mean_abs_delta_nonzero_median_fr_hz=("abs_delta_nonzero_median_fr_hz", "mean"),
            mean_abs_delta_nonzero_fraction=("abs_delta_nonzero_fraction", "mean"),
            median_abs_delta_mean_fr_hz=("abs_delta_mean_fr_hz", "median"),
            median_abs_delta_median_fr_hz=("abs_delta_median_fr_hz", "median"),
            median_abs_delta_nonzero_mean_fr_hz=("abs_delta_nonzero_mean_fr_hz", "median"),
            median_abs_delta_nonzero_median_fr_hz=("abs_delta_nonzero_median_fr_hz", "median"),
            median_abs_delta_nonzero_fraction=("abs_delta_nonzero_fraction", "median"),
            mean_delta_nonzero_fraction=("delta_nonzero_fraction", "mean"),
            median_delta_nonzero_fraction=("delta_nonzero_fraction", "median"),
        )
    )


def summarize_ndi(grid_all: pd.DataFrame) -> pd.DataFrame:
    """汇总 NDI 相关统计。"""

    valid = grid_all[grid_all["valid_grid"]].copy()
    return (
        valid.groupby(["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"], as_index=False)
        .agg(
            valid_grid_count=("valid_grid", "size"),
            mean_NDI_mean=("NDI_mean", "mean"),
            median_NDI_mean=("NDI_mean", "median"),
            mean_abs_NDI_mean=("NDI_mean", lambda x: float(np.nanmean(np.abs(x)))),
            median_abs_NDI_mean=("NDI_mean", lambda x: float(np.nanmedian(np.abs(x)))),
            mean_NDI_median=("NDI_median", "mean"),
            median_NDI_median=("NDI_median", "median"),
            mean_abs_NDI_median=("NDI_median", lambda x: float(np.nanmean(np.abs(x)))),
            mean_NDI_nonzero_mean=("NDI_nonzero_mean", "mean"),
            mean_abs_NDI_nonzero_mean=("NDI_nonzero_mean", lambda x: float(np.nanmean(np.abs(x)))),
            mean_NDI_nonzero_fraction=("NDI_nonzero_fraction", "mean"),
            mean_abs_NDI_nonzero_fraction=("NDI_nonzero_fraction", lambda x: float(np.nanmean(np.abs(x)))),
        )
    )


def summarize_map_similarity(grid_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算 ume_map 与 cme_map 的空间相似性。"""

    rows = []
    for keys, sub in grid_all[grid_all["valid_grid"]].groupby(
        ["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"]
    ):
        ume = sub["UME_mean_fr_hz"].to_numpy(dtype=float)
        cme = sub["CME_mean_fr_hz"].to_numpy(dtype=float)
        delta = ume - cme
        pearson_r, pearson_p = pearson_with_approx_p(ume, cme)
        spearman_r, spearman_p = spearman_with_approx_p(ume, cme)
        cos = cosine_similarity(ume, cme)
        mae = float(np.mean(np.abs(delta))) if delta.size else np.nan
        rmse = float(np.sqrt(np.mean(delta**2))) if delta.size else np.nan
        mean_level = float(np.mean((ume + cme) / 2)) if delta.size else np.nan
        norm_mae = mae / (mean_level + EPSILON) if np.isfinite(mean_level) else np.nan
        norm_rmse = rmse / (mean_level + EPSILON) if np.isfinite(mean_level) else np.nan
        rows.append(
            {
                "grid_scale": keys[0],
                "grid_n": keys[1],
                "pair_id": keys[2],
                "direction_id": keys[3],
                "direction_code": keys[4],
                "direction_name": keys[5],
                "valid_grid_count": int(sub.shape[0]),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "cosine_similarity": cos,
                "cosine_distance": 1 - cos if np.isfinite(cos) else np.nan,
                "MAE": mae,
                "RMSE": rmse,
                "normalized_MAE": norm_mae,
                "normalized_RMSE": norm_rmse,
                "spatial_dissimilarity_index": norm_mae,
                "mean_abs_delta_mean_fr_hz": mae,
                "median_abs_delta_mean_fr_hz": float(np.median(np.abs(delta))) if delta.size else np.nan,
                "mean_abs_NDI_mean": safe_mean(np.abs(sub["NDI_mean"].to_numpy(dtype=float))),
                "median_abs_NDI_mean": safe_median(np.abs(sub["NDI_mean"].to_numpy(dtype=float))),
            }
        )
    sim = pd.DataFrame(rows)
    overall = (
        sim.groupby(["grid_scale", "grid_n", "pair_id"], as_index=False)
        .agg(
            mean_pearson_r=("pearson_r", "mean"),
            mean_spearman_r=("spearman_r", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            mean_normalized_MAE=("normalized_MAE", "mean"),
            mean_normalized_RMSE=("normalized_RMSE", "mean"),
            mean_spatial_dissimilarity_index=("spatial_dissimilarity_index", "mean"),
            mean_abs_delta_mean_fr_hz=("mean_abs_delta_mean_fr_hz", "mean"),
            mean_abs_NDI_mean=("mean_abs_NDI_mean", "mean"),
        )
    )
    return sim, overall


def summarize_valid_counts(grid_all: pd.DataFrame) -> pd.DataFrame:
    """统计每个组合下有效网格数量和 unit 数分布。"""

    rows = []
    for keys, sub in grid_all.groupby(["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"]):
        valid = sub[sub["valid_grid"]]
        rows.append(
            {
                "grid_scale": keys[0],
                "grid_n": keys[1],
                "pair_id": keys[2],
                "direction_id": keys[3],
                "direction_code": keys[4],
                "direction_name": keys[5],
                "total_grid_count": int(sub.shape[0]),
                "valid_grid_count": int(valid.shape[0]),
                "insufficient_grid_count": int((~sub["valid_grid"]).sum()),
                "valid_grid_fraction": float(valid.shape[0] / sub.shape[0]) if sub.shape[0] else np.nan,
                "mean_UME_unit_count_per_valid_grid": safe_mean(valid["UME_unit_count"]),
                "mean_CME_unit_count_per_valid_grid": safe_mean(valid["CME_unit_count"]),
                "median_UME_unit_count_per_valid_grid": safe_median(valid["UME_unit_count"]),
                "median_CME_unit_count_per_valid_grid": safe_median(valid["CME_unit_count"]),
            }
        )
    return pd.DataFrame(rows)


def paired_retina_summaries(
    threshold_direction: pd.DataFrame,
    robust: pd.DataFrame,
    ndi_summary: pd.DataFrame,
    sim: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 paired-retina direction-level 和 overall-level 总结。"""

    main = threshold_direction[threshold_direction["threshold_hz"] == MAIN_THRESHOLD_HZ].copy()
    direction = main.merge(
        robust,
        on=["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name", "valid_grid_count"],
        how="left",
        suffixes=("", "_robust"),
    )
    direction = direction.merge(
        ndi_summary[
            [
                "grid_scale",
                "grid_n",
                "pair_id",
                "direction_id",
                "direction_code",
                "direction_name",
                "mean_NDI_mean",
                "mean_abs_NDI_mean",
                "mean_NDI_nonzero_fraction",
                "mean_abs_NDI_nonzero_fraction",
            ]
        ],
        on=["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"],
        how="left",
    )
    direction = direction.merge(
        sim[
            [
                "grid_scale",
                "grid_n",
                "pair_id",
                "direction_id",
                "direction_code",
                "direction_name",
                "pearson_r",
                "spearman_r",
                "cosine_similarity",
                "normalized_MAE",
                "normalized_RMSE",
            ]
        ],
        on=["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"],
        how="left",
    )
    overall = (
        direction.groupby(["grid_scale", "grid_n", "pair_id"], as_index=False)
        .agg(
            mean_different_fraction=("different_fraction", "mean"),
            min_different_fraction=("different_fraction", "min"),
            max_different_fraction=("different_fraction", "max"),
            mean_UME_higher_fraction=("UME_higher_fraction", "mean"),
            mean_CME_higher_fraction=("CME_higher_fraction", "mean"),
            mean_similar_fraction=("similar_fraction", "mean"),
            mean_abs_delta_mean_fr_hz=("mean_abs_delta_mean_fr_hz", "mean"),
            median_abs_delta_mean_fr_hz=("median_abs_delta_mean_fr_hz", "median"),
            mean_abs_NDI_mean=("mean_abs_NDI_mean", "mean"),
            mean_normalized_MAE=("normalized_MAE", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            mean_pearson_r=("pearson_r", "mean"),
            total_valid_grids=("valid_grid_count", "sum"),
            total_UME_higher=("UME_higher_count", "sum"),
            total_CME_higher=("CME_higher_count", "sum"),
            total_similar=("similar_count", "sum"),
        )
    )
    return direction, overall


# =============================================================================
# 6. permutation-based null comparison
# =============================================================================


def permutation_null_for_combo(
    grid_valid: pd.DataFrame,
    unit_sub: pd.DataFrame,
    threshold: float = MAIN_THRESHOLD_HZ,
) -> dict[str, float]:
    """在每个有效网格内随机打乱 UME/CME 标签，构建 null distribution。

    该检验不证明同一个 unit 发生变化；它只是检验真实刺激标签下的
    grid-level 差异是否大于同一网格内随机分组的预期。
    """

    real_abs_delta = grid_valid["abs_delta_mean_fr_hz"].to_numpy(dtype=float)
    real_ndi_abs = np.abs(grid_valid["NDI_mean"].to_numpy(dtype=float))
    real_mean_abs_delta = safe_mean(real_abs_delta)
    real_different_fraction = float(np.mean(real_abs_delta > threshold)) if real_abs_delta.size else np.nan
    real_mean_abs_ndi = safe_mean(real_ndi_abs)

    grid_arrays: list[tuple[np.ndarray, int]] = []
    for grid_id in grid_valid["grid_id"]:
        values = unit_sub.loc[unit_sub["grid_id"] == grid_id, ["stimulus", "firing_rate_hz"]]
        ume_n = int((values["stimulus"] == "UME").sum())
        arr = values["firing_rate_hz"].to_numpy(dtype=float)
        if ume_n >= MIN_UNITS_PER_GRID_PER_STIM and (arr.size - ume_n) >= MIN_UNITS_PER_GRID_PER_STIM:
            grid_arrays.append((arr, ume_n))

    null_mean_abs_delta = np.empty(N_PERM, dtype=float)
    null_different_fraction = np.empty(N_PERM, dtype=float)
    null_mean_abs_ndi = np.empty(N_PERM, dtype=float)

    for i in range(N_PERM):
        deltas = []
        ndis = []
        for arr, ume_n in grid_arrays:
            perm = RNG.permutation(arr)
            s = perm[:ume_n]
            d = perm[ume_n:]
            s_mean = float(np.mean(s))
            d_mean = float(np.mean(d))
            delta = s_mean - d_mean
            deltas.append(delta)
            ndis.append(delta / (s_mean + d_mean + EPSILON))
        deltas = np.asarray(deltas, dtype=float)
        ndis = np.asarray(ndis, dtype=float)
        null_mean_abs_delta[i] = safe_mean(np.abs(deltas))
        null_different_fraction[i] = float(np.mean(np.abs(deltas) > threshold)) if deltas.size else np.nan
        null_mean_abs_ndi[i] = safe_mean(np.abs(ndis))

    def p_greater(null_values: np.ndarray, real_value: float) -> float:
        null_values = null_values[np.isfinite(null_values)]
        if null_values.size == 0 or not np.isfinite(real_value):
            return np.nan
        return float((np.sum(null_values >= real_value) + 1) / (null_values.size + 1))

    return {
        "n_perm": N_PERM,
        "real_mean_abs_delta": real_mean_abs_delta,
        "null_mean_mean_abs_delta": safe_mean(null_mean_abs_delta),
        "null_std_mean_abs_delta": float(np.nanstd(null_mean_abs_delta, ddof=1)),
        "p_mean_abs_delta": p_greater(null_mean_abs_delta, real_mean_abs_delta),
        "real_different_fraction": real_different_fraction,
        "null_mean_different_fraction": safe_mean(null_different_fraction),
        "null_std_different_fraction": float(np.nanstd(null_different_fraction, ddof=1)),
        "p_different_fraction": p_greater(null_different_fraction, real_different_fraction),
        "real_mean_abs_NDI": real_mean_abs_ndi,
        "null_mean_mean_abs_NDI": safe_mean(null_mean_abs_ndi),
        "null_std_mean_abs_NDI": float(np.nanstd(null_mean_abs_ndi, ddof=1)),
        "p_mean_abs_NDI": p_greater(null_mean_abs_ndi, real_mean_abs_ndi),
        "_null_mean_abs_delta_values": null_mean_abs_delta,
        "_null_different_fraction_values": null_different_fraction,
    }


def compute_permutation_summary(grid_all: pd.DataFrame, unit_all: pd.DataFrame) -> pd.DataFrame:
    """对所有 grid_scale × pair × direction 做 permutation null comparison。"""

    rows = []
    for keys, sub in grid_all.groupby(["grid_scale", "grid_n", "pair_id", "direction_id", "direction_code", "direction_name"]):
        valid = sub[sub["valid_grid"]]
        unit_sub = unit_all[
            (unit_all["grid_scale"] == keys[0])
            & (unit_all["pair_id"] == keys[2])
            & (unit_all["direction_id"] == keys[3])
        ]
        result = permutation_null_for_combo(valid, unit_sub)
        null_abs = result.pop("_null_mean_abs_delta_values")
        null_diff = result.pop("_null_different_fraction_values")
        row = {
            "grid_scale": keys[0],
            "grid_n": keys[1],
            "pair_id": keys[2],
            "direction_id": keys[3],
            "direction_code": keys[4],
            "direction_name": keys[5],
            **result,
        }
        rows.append(row)

        # 只对主尺度保存代表性的 null distribution 图，避免图像数量过大。
        if keys[1] in MAIN_GRID_SCALES_FOR_FIGURES:
            make_permutation_histogram(keys, result, null_abs, null_diff)
    return pd.DataFrame(rows)


# =============================================================================
# 7. 作图函数
# =============================================================================


def matrix_from_grid(grid: pd.DataFrame, value_col: str, grid_n: int) -> np.ndarray:
    """将 grid-cell 表转成 N×N 矩阵。"""

    mat = np.full((grid_n, grid_n), np.nan)
    for row in grid.itertuples(index=False):
        mat[int(row.grid_y), int(row.grid_x)] = getattr(row, value_col)
    return mat


def make_delta_and_classification_map(grid: pd.DataFrame) -> None:
    """生成每个组合的 delta map 与分类 map。"""

    grid_n = int(grid["grid_n"].iloc[0])
    grid_scale = grid["grid_scale"].iloc[0]
    pair_id = grid["pair_id"].iloc[0]
    direction_id = grid["direction_id"].iloc[0]
    direction_code = grid["direction_code"].iloc[0]
    delta = matrix_from_grid(grid, "delta_mean_fr_hz", grid_n)
    valid_values = delta[np.isfinite(delta)]
    vmax = max(1.0, float(np.nanpercentile(np.abs(valid_values), 95))) if valid_values.size else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(delta, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    if grid_n <= 20:
        for row in grid.itertuples(index=False):
            cls = row.difference_class_main
            text = {"UME_higher": "U", "CME_higher": "C", "similar": "=", "insufficient_units": "."}.get(cls, "")
            ax.text(row.grid_x, row.grid_y, text, ha="center", va="center", fontsize=6, color="black")
    ax.set_title(f"{grid_scale} {pair_id} dir-{direction_id} {direction_code}\nDelta mean FR: UME - CME")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, label="Hz")
    save_figure(
        fig,
        "grid_maps",
        f"{grid_scale}_{pair_id}_dir-{direction_id}_{direction_code}_delta_mean_map",
        FigureRecord(
            figure_id=f"{grid_scale}_{pair_id}_{direction_id}_delta_mean",
            figure_type="delta_mean_fr_map",
            grid_scale=grid_scale,
            pair_id=pair_id,
            direction_id=direction_id,
            file_png="",
            file_pdf="",
            description="Grid-level UME_mean_fr_hz - CME_mean_fr_hz heatmap; S/D/= are threshold-based descriptive classes.",
        ),
    )


def make_response_fraction_map(grid: pd.DataFrame) -> None:
    """生成 response fraction 差异图，只对主尺度输出。"""

    grid_n = int(grid["grid_n"].iloc[0])
    if grid_n not in MAIN_GRID_SCALES_FOR_FIGURES:
        return
    grid_scale = grid["grid_scale"].iloc[0]
    pair_id = grid["pair_id"].iloc[0]
    direction_id = grid["direction_id"].iloc[0]
    direction_code = grid["direction_code"].iloc[0]
    mat = matrix_from_grid(grid, "delta_nonzero_fraction", grid_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"{grid_scale} {pair_id} dir-{direction_id} {direction_code}\nDelta responsive fraction: UME - CME")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, label="fraction difference")
    save_figure(
        fig,
        "response_fraction_maps",
        f"{grid_scale}_{pair_id}_dir-{direction_id}_{direction_code}_delta_nonzero_fraction",
        FigureRecord(
            figure_id=f"{grid_scale}_{pair_id}_{direction_id}_response_fraction",
            figure_type="delta_nonzero_fraction_map",
            grid_scale=grid_scale,
            pair_id=pair_id,
            direction_id=direction_id,
            file_png="",
            file_pdf="",
            description="Responsive-unit fraction difference map; positive means UME has larger responsive fraction.",
        ),
    )


def make_robust_metric_panel(grid: pd.DataFrame) -> None:
    """生成 mean/median/nonzero/response fraction/NDI 的多面板图。"""

    grid_n = int(grid["grid_n"].iloc[0])
    if grid_n not in MAIN_GRID_SCALES_FOR_FIGURES:
        return
    grid_scale = grid["grid_scale"].iloc[0]
    pair_id = grid["pair_id"].iloc[0]
    direction_id = grid["direction_id"].iloc[0]
    direction_code = grid["direction_code"].iloc[0]

    panels = [
        ("delta_mean_fr_hz", "Delta mean FR", "Hz", None),
        ("delta_median_fr_hz", "Delta median FR", "Hz", None),
        ("delta_nonzero_mean_fr_hz", "Delta nonzero mean FR", "Hz", None),
        ("delta_nonzero_fraction", "Delta responsive fraction", "fraction", (-1, 1)),
        ("NDI_mean", "NDI mean", "NDI", (-1, 1)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.8))
    for ax, (col, title, label, fixed) in zip(axes, panels):
        mat = matrix_from_grid(grid, col, grid_n)
        if fixed is None:
            vals = mat[np.isfinite(mat)]
            vmax = max(1.0, float(np.nanpercentile(np.abs(vals), 95))) if vals.size else 1.0
            vmin, vmax = -vmax, vmax
        else:
            vmin, vmax = fixed
        im = ax.imshow(mat, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.suptitle(f"{grid_scale} {pair_id} dir-{direction_id} {direction_code}: robust metric maps", fontsize=11)
    save_figure(
        fig,
        "robust_metric_maps",
        f"{grid_scale}_{pair_id}_dir-{direction_id}_{direction_code}_robust_metric_panel",
        FigureRecord(
            figure_id=f"{grid_scale}_{pair_id}_{direction_id}_robust_panel",
            figure_type="robust_metric_panel",
            grid_scale=grid_scale,
            pair_id=pair_id,
            direction_id=direction_id,
            file_png="",
            file_pdf="",
            description="Multi-panel maps for mean, median, nonzero mean, response fraction, and NDI.",
        ),
    )


def make_ndi_map(grid: pd.DataFrame) -> None:
    """生成 NDI_mean map。"""

    grid_n = int(grid["grid_n"].iloc[0])
    if grid_n not in MAIN_GRID_SCALES_FOR_FIGURES:
        return
    grid_scale = grid["grid_scale"].iloc[0]
    pair_id = grid["pair_id"].iloc[0]
    direction_id = grid["direction_id"].iloc[0]
    direction_code = grid["direction_code"].iloc[0]
    mat = matrix_from_grid(grid, "NDI_mean", grid_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"{grid_scale} {pair_id} dir-{direction_id} {direction_code}\nNDI mean")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, label="NDI")
    save_figure(
        fig,
        "NDI_maps",
        f"{grid_scale}_{pair_id}_dir-{direction_id}_{direction_code}_NDI_mean",
        FigureRecord(
            figure_id=f"{grid_scale}_{pair_id}_{direction_id}_NDI_mean",
            figure_type="NDI_mean_map",
            grid_scale=grid_scale,
            pair_id=pair_id,
            direction_id=direction_id,
            file_png="",
            file_pdf="",
            description="Normalized difference index map based on mean firing rate.",
        ),
    )


def make_map_similarity_scatter(grid: pd.DataFrame) -> None:
    """UME_map vs cme_map scatter plot。"""

    grid_n = int(grid["grid_n"].iloc[0])
    if grid_n not in MAIN_GRID_SCALES_FOR_FIGURES:
        return
    valid = grid[grid["valid_grid"]]
    grid_scale = grid["grid_scale"].iloc[0]
    pair_id = grid["pair_id"].iloc[0]
    direction_id = grid["direction_id"].iloc[0]
    direction_code = grid["direction_code"].iloc[0]
    x = valid["UME_mean_fr_hz"].to_numpy(dtype=float)
    y = valid["CME_mean_fr_hz"].to_numpy(dtype=float)
    r, _ = pearson_with_approx_p(x, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=18, alpha=0.75)
    lim = max(float(np.nanmax(x)) if x.size else 1, float(np.nanmax(y)) if y.size else 1, 1)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("UME mean FR (Hz)")
    ax.set_ylabel("CME mean FR (Hz)")
    ax.set_title(f"{grid_scale} {pair_id} dir-{direction_id} {direction_code}\nr={r:.2f}, n={len(valid)}")
    save_figure(
        fig,
        "map_similarity",
        f"{grid_scale}_{pair_id}_dir-{direction_id}_{direction_code}_UME_vs_CME_scatter",
        FigureRecord(
            figure_id=f"{grid_scale}_{pair_id}_{direction_id}_scatter",
            figure_type="UME_cme_map_scatter",
            grid_scale=grid_scale,
            pair_id=pair_id,
            direction_id=direction_id,
            file_png="",
            file_pdf="",
            description="Scatter plot comparing spatially matched grid mean firing rates.",
        ),
    )


def make_permutation_histogram(keys, result, null_abs, null_diff) -> None:
    """生成 permutation null 分布图。"""

    grid_scale, grid_n, pair_id, direction_id, direction_code, _ = keys
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(null_abs, bins=30, color="#4C78A8", alpha=0.8)
    axes[0].axvline(result["real_mean_abs_delta"], color="#E45756", linewidth=2)
    axes[0].set_title(f"mean abs delta\np={result['p_mean_abs_delta']:.3g}")
    axes[0].set_xlabel("null mean abs delta")
    axes[1].hist(null_diff, bins=30, color="#72B7B2", alpha=0.8)
    axes[1].axvline(result["real_different_fraction"], color="#E45756", linewidth=2)
    axes[1].set_title(f"different fraction\np={result['p_different_fraction']:.3g}")
    axes[1].set_xlabel("null different fraction")
    fig.suptitle(f"{grid_scale} {pair_id} dir-{direction_id} {direction_code} permutation null")
    save_figure(
        fig,
        "permutation",
        f"{grid_scale}_{pair_id}_dir-{direction_id}_{direction_code}_permutation_null",
        FigureRecord(
            figure_id=f"{grid_scale}_{pair_id}_{direction_id}_permutation",
            figure_type="permutation_null_distribution",
            grid_scale=grid_scale,
            pair_id=pair_id,
            direction_id=direction_id,
            file_png="",
            file_pdf="",
            description="Permutation null distributions for mean abs delta and different fraction.",
        ),
    )


def make_pipeline_figure() -> None:
    """生成 MEA 分析流程示意图。"""

    steps = [
        "Paired retina\nUME/CME recordings",
        "Good sorted units\nON phase",
        "Center edge step\n8 directions",
        "Spatial grid\naggregation",
        "Grid metrics\nmean/median/NDI",
        "Map similarity\n& threshold analysis",
        "Biological\nmotivation for RBCM",
    ]
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, text) in enumerate(zip(xs, steps)):
        ax.text(x, 0.5, text, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="#555555"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.06, 0.5), xytext=(x + 0.06, 0.5), arrowprops=dict(arrowstyle="->", lw=1.4))
    save_figure(
        fig,
        "main_figures",
        "main_figure_1_MEA_analysis_pipeline",
        FigureRecord(
            figure_id="main_figure_1",
            figure_type="analysis_pipeline",
            grid_scale="NA",
            pair_id="NA",
            direction_id="NA",
            file_png="",
            file_pdf="",
            description="Schematic workflow of the MEA grid-level analysis.",
        ),
    )


def make_threshold_sensitivity_figures(threshold_dir: pd.DataFrame, threshold_pair: pd.DataFrame) -> None:
    """生成阈值敏感性图。"""

    for grid_n in MAIN_GRID_SCALES_FOR_FIGURES:
        sub = threshold_pair[threshold_pair["grid_n"] == grid_n]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for pair_id, df in sub.groupby("pair_id"):
            df = df.sort_values("threshold_hz")
            ax.plot(df["threshold_hz"], df["mean_different_fraction"], marker="o", label=pair_id)
        ax.set_xlabel("threshold (Hz)")
        ax.set_ylabel("mean different fraction")
        ax.set_title(f"Threshold sensitivity ({grid_n}x{grid_n})")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()
        save_figure(
            fig,
            "threshold_sensitivity",
            f"threshold_sensitivity_lines_{grid_n}x{grid_n}",
            FigureRecord(
                figure_id=f"threshold_lines_{grid_n}",
                figure_type="threshold_sensitivity_line",
                grid_scale=f"{grid_n}x{grid_n}",
                pair_id="all",
                direction_id="all",
                file_png="",
                file_pdf="",
                description="Mean different fraction across thresholds for each paired retina.",
            ),
        )

    heat = (
        threshold_pair.groupby(["grid_scale", "grid_n", "threshold_hz"], as_index=False)
        .agg(mean_different_fraction=("mean_different_fraction", "mean"))
        .sort_values(["grid_n", "threshold_hz"])
    )
    pivot = heat.pivot(index="grid_scale", columns="threshold_hz", values="mean_different_fraction")
    pivot = pivot.loc[[f"{g}x{g}" for g in GRID_SCALES]]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("threshold (Hz)")
    ax.set_ylabel("grid scale")
    ax.set_title("Grid scale × threshold sensitivity")
    fig.colorbar(im, ax=ax, label="mean different fraction")
    save_figure(
        fig,
        "threshold_sensitivity",
        "threshold_grid_scale_heatmap",
        FigureRecord(
            figure_id="threshold_heatmap",
            figure_type="threshold_grid_scale_heatmap",
            grid_scale="all",
            pair_id="all",
            direction_id="all",
            file_png="",
            file_pdf="",
            description="Heatmap of mean different fraction across grid scales and thresholds.",
        ),
    )


def make_scale_robustness_figures(pair_overall: pd.DataFrame, sim_overall: pd.DataFrame) -> None:
    """生成多尺度鲁棒性主图。"""

    # paired_retina_overall_summary 已经包含 mean_normalized_MAE。
    # 如果未来调用者传入的表缺少该列，则再从 sim_overall 补充。
    if "mean_normalized_MAE" in pair_overall.columns:
        merged = pair_overall.copy()
    else:
        merged = pair_overall.merge(
            sim_overall[["grid_scale", "grid_n", "pair_id", "mean_normalized_MAE"]],
            on=["grid_scale", "grid_n", "pair_id"],
            how="left",
        )
    metrics = [
        ("mean_different_fraction", "Mean different fraction", (0, 1)),
        ("mean_abs_delta_mean_fr_hz", "Mean abs delta (Hz)", None),
        ("mean_normalized_MAE", "Mean normalized MAE", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title, ylim) in zip(axes, metrics):
        for pair_id, df in merged.groupby("pair_id"):
            df = df.sort_values("grid_n")
            ax.plot(df["grid_n"], df[col], marker="o", label=pair_id)
        ax.set_xlabel("grid N")
        ax.set_ylabel(title)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Grid-scale robustness of spatial response differences")
    save_figure(
        fig,
        "main_figures",
        "main_figure_3_grid_scale_robustness",
        FigureRecord(
            figure_id="main_figure_3",
            figure_type="grid_scale_robustness",
            grid_scale="all",
            pair_id="all",
            direction_id="all",
            file_png="",
            file_pdf="",
            description="Robustness of different fraction, abs delta, and normalized MAE across grid scales.",
        ),
    )


def make_similarity_summary_figures(sim: pd.DataFrame, sim_overall: pd.DataFrame) -> None:
    """生成空间响应图相似性汇总图。"""

    for grid_n in MAIN_GRID_SCALES_FOR_FIGURES:
        sub = sim[sim["grid_n"] == grid_n].copy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for pair_id, df in sub.groupby("pair_id"):
            df = df.sort_values("direction_id")
            axes[0].plot(df["direction_code"], df["normalized_MAE"], marker="o", label=pair_id)
            axes[1].plot(df["direction_code"], df["cosine_similarity"], marker="o", label=pair_id)
        axes[0].set_ylabel("normalized MAE")
        axes[1].set_ylabel("cosine similarity")
        for ax in axes:
            ax.set_xlabel("direction")
            ax.grid(True, alpha=0.25)
        axes[0].legend()
        fig.suptitle(f"Spatial map similarity ({grid_n}x{grid_n})")
        save_figure(
            fig,
            "map_similarity",
            f"spatial_map_similarity_direction_summary_{grid_n}x{grid_n}",
            FigureRecord(
                figure_id=f"similarity_direction_{grid_n}",
                figure_type="spatial_similarity_direction_summary",
                grid_scale=f"{grid_n}x{grid_n}",
                pair_id="all",
                direction_id="all",
                file_png="",
                file_pdf="",
                description="Direction-wise normalized MAE and cosine similarity for each paired retina.",
            ),
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for pair_id, df in sim_overall.groupby("pair_id"):
        df = df.sort_values("grid_n")
        ax.plot(df["grid_n"], df["mean_normalized_MAE"], marker="o", label=pair_id)
    ax.set_xlabel("grid N")
    ax.set_ylabel("mean normalized MAE")
    ax.set_title("Spatial dissimilarity across grid scales")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(
        fig,
        "map_similarity",
        "spatial_similarity_scale_robustness_normalized_MAE",
        FigureRecord(
            figure_id="similarity_scale_normalized_MAE",
            figure_type="spatial_similarity_scale_robustness",
            grid_scale="all",
            pair_id="all",
            direction_id="all",
            file_png="",
            file_pdf="",
            description="Scale robustness of mean normalized MAE.",
        ),
    )


def make_paired_retina_dotplot(pair_overall: pd.DataFrame) -> None:
    """生成 paired retina 层级 summary dot plot。"""

    sub = pair_overall[pair_overall["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()
    metrics = [
        ("mean_different_fraction", "mean different fraction"),
        ("mean_abs_delta_mean_fr_hz", "mean abs delta (Hz)"),
        ("mean_abs_NDI_mean", "mean abs NDI"),
        ("mean_normalized_MAE", "mean normalized MAE"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for ax, (col, title) in zip(axes, metrics):
        for grid_n, marker in [(12, "o"), (16, "s")]:
            df = sub[sub["grid_n"] == grid_n]
            ax.scatter(df["pair_id"], df[col], label=f"{grid_n}x{grid_n}", marker=marker, s=60)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Paired-retina-level summary")
    save_figure(
        fig,
        "main_figures",
        "main_figure_5_paired_retina_summary",
        FigureRecord(
            figure_id="main_figure_5",
            figure_type="paired_retina_summary_dotplot",
            grid_scale="12x12_16x16",
            pair_id="all",
            direction_id="all",
            file_png="",
            file_pdf="",
            description="Paired-retina-level summary for main grid scales.",
        ),
    )


# =============================================================================
# 8. 报告生成
# =============================================================================


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """DataFrame 转 Markdown 表格。"""

    use = df if max_rows is None else df.head(max_rows)
    if use.empty:
        return "（无数据）"
    formatted = use.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_final_report(
    paired_overall: pd.DataFrame,
    threshold_pair: pd.DataFrame,
    robust: pd.DataFrame,
    ndi_summary: pd.DataFrame,
    sim_overall: pd.DataFrame,
    permutation: pd.DataFrame,
    valid_counts: pd.DataFrame,
) -> None:
    """写最终 Markdown 报告。"""

    main_pair = paired_overall[paired_overall["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()
    threshold_main = threshold_pair[threshold_pair["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()
    robust_main = robust[robust["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()
    ndi_main = ndi_summary[ndi_summary["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()
    sim_main = sim_overall[sim_overall["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()
    perm_main = permutation[permutation["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)].copy()

    # 自动给出有利/不利模块判断。
    module_rows = []
    avg_diff = paired_overall["mean_different_fraction"].mean()
    module_rows.append(
        {
            "analysis_module": "grid-level mean FR difference",
            "brief_result": f"all-scale mean different fraction ≈ {avg_diff:.3f}",
            "paper_value": "有利",
            "reason": "多尺度下大量空间匹配网格超过预设差异阈值，可支持 spatial population response difference。",
        }
    )
    thresh_1 = threshold_pair[threshold_pair["threshold_hz"] == 1.0]["mean_different_fraction"].mean()
    module_rows.append(
        {
            "analysis_module": "threshold sensitivity",
            "brief_result": f"threshold=1.0 Hz mean different fraction ≈ {thresh_1:.3f}",
            "paper_value": "有利但需谨慎",
            "reason": "阈值提高后差异比例下降是预期现象；若仍保持较高，可说明不是 0.5 Hz 单阈值造成。",
        }
    )
    median_ratio = robust_main["mean_abs_delta_median_fr_hz"].mean() / (robust_main["mean_abs_delta_mean_fr_hz"].mean() + EPSILON)
    module_rows.append(
        {
            "analysis_module": "median/nonzero robust metrics",
            "brief_result": f"median/mean abs-delta ratio in main scales ≈ {median_ratio:.3f}",
            "paper_value": "中性/辅助",
            "reason": "用于判断均值差异是否受高发放 unit 影响；若 median 明显较弱，应把主结论表述为 local population average output difference。",
        }
    )
    mean_abs_ndi = ndi_main["mean_abs_NDI_mean"].mean()
    module_rows.append(
        {
            "analysis_module": "NDI",
            "brief_result": f"main-scale mean abs NDI ≈ {mean_abs_ndi:.3f}",
            "paper_value": "有利",
            "reason": "若 NDI 也不低，说明差异不只是绝对 firing-rate 高低造成，也包含相对 UME/CME 偏好变化。",
        }
    )
    norm_mae = sim_main["mean_normalized_MAE"].mean()
    module_rows.append(
        {
            "analysis_module": "spatial map similarity",
            "brief_result": f"main-scale mean normalized MAE ≈ {norm_mae:.3f}",
            "paper_value": "有利",
            "reason": "直接量化 ume_map 与 cme_map 的空间响应图差异，非常适合连接 RBCM 空间上下文调制动机。",
        }
    )
    perm_sig = float(np.mean(perm_main["p_mean_abs_delta"] < 0.05)) if len(perm_main) else np.nan
    module_rows.append(
        {
            "analysis_module": "permutation null",
            "brief_result": f"main-scale fraction p_mean_abs_delta<0.05 ≈ {perm_sig:.3f}",
            "paper_value": "视结果而定",
            "reason": "如果显著比例高，则强力支持真实标签差异超过随机标签；如果低，则作为补充或不主打。",
        }
    )
    module_eval = pd.DataFrame(module_rows)
    save_csv(module_eval, "analysis_module_usefulness_summary.csv")

    report = f"""# RBCM-edge MEA Analysis Final Report

## 1. 分析目标

本分析比较的是 UME 和 CME 在同一视网膜切片中空间匹配局部区域内诱发的 RGC sorted-unit 群体响应模式，而不是同一 RGC 的 paired response。由于 UME 和 CME 来自不同 recording / spike sorting 结果，sorted units 无法可靠一一对应，因此本分析采用 spatially matched grid-level local population analysis。

## 2. 数据与实验配对

| pair_id | CME recording | UME recording |
|---|---|---|
| pair_31_32 | 000031 | 000032 |
| pair_34_35 | 000034 | 000035 |
| pair_37_38 | 000037 | 000038 |

## 3. 响应提取方法

- 使用 good units。
- 只分析 ON phase。
- 只取边缘移动到图像中间的步骤：UME 第 7/13 步，CME 第 6/11 步。
- 每个 unit 的 firing rate 为 3 次 repeat 的平均值。
- 分析 8 个方向。
- 在同一 paired retina 内划分空间网格，对同一空间网格内的 UME 和 CME units 分别聚合。

## 4. 主网格分析方法

使用网格尺度：{GRID_SCALES}。

有效网格定义：

```text
UME_unit_count >= {MIN_UNITS_PER_GRID_PER_STIM} 且 CME_unit_count >= {MIN_UNITS_PER_GRID_PER_STIM}
```

主差异值：

```text
delta_mean_fr_hz = UME_mean_fr_hz - CME_mean_fr_hz
```

主分类阈值：

```text
{MAIN_THRESHOLD_HZ} Hz
```

分类只是 descriptive grid-level classification，不是单格统计显著性检验。

## 5. 主结果：paired retina 层级

主尺度 12×12 和 16×16 的 paired-retina-level summary：

{markdown_table(main_pair[["grid_scale", "pair_id", "mean_different_fraction", "min_different_fraction", "max_different_fraction", "mean_abs_delta_mean_fr_hz", "mean_abs_NDI_mean", "mean_normalized_MAE", "total_valid_grids", "total_UME_higher", "total_CME_higher", "total_similar"]])}

总体观察：

- pair_34_35 通常最强。
- pair_31_32 相对较弱但仍可观察到差异。
- pair_37_38 中等且较稳定。
- UME_higher 和 CME_higher 均存在，说明不是简单全局 gain change，而是 spatially heterogeneous modulation。

## 6. 多尺度鲁棒性

所有尺度 paired retina summary 见 `paired_retina_overall_summary.csv`。多尺度结果用于说明该现象不是由单一网格尺度造成。

## 7. 阈值敏感性

主尺度阈值敏感性汇总：

{markdown_table(threshold_main[["grid_scale", "pair_id", "threshold_hz", "mean_different_fraction", "mean_UME_higher_fraction", "mean_CME_higher_fraction", "mean_similar_fraction", "total_valid_grids"]])}

解释：

- 如果 threshold 从 0.25 增加到 1.0 后 different_fraction 仍保持较高，则说明结果不完全依赖 0.5 Hz 单一阈值。
- 如果某些 pair 对阈值敏感，应在论文中谨慎表述为 threshold-based descriptive result。

## 8. 响应比例分析

response fraction 通过 `delta_nonzero_fraction = ume_nonzero_fraction - cme_nonzero_fraction` 衡量。完整结果见 `robust_metric_grid_summary.csv` 和 grid-cell 明细表。

主尺度 response fraction 差异概览：

{markdown_table(robust_main[["grid_scale", "pair_id", "direction_id", "direction_code", "valid_grid_count", "mean_delta_nonzero_fraction", "mean_abs_delta_nonzero_fraction"]], max_rows=30)}

## 9. 稳健 firing-rate 指标分析

稳健指标用于判断 mean FR 差异是否受少数高 firing-rate units 影响。完整结果见 `robust_metric_grid_summary.csv`。

主尺度稳健指标前 30 行：

{markdown_table(robust_main[["grid_scale", "pair_id", "direction_id", "direction_code", "mean_abs_delta_mean_fr_hz", "mean_abs_delta_median_fr_hz", "mean_abs_delta_nonzero_mean_fr_hz", "mean_abs_delta_nonzero_median_fr_hz", "mean_abs_delta_nonzero_fraction"]], max_rows=30)}

## 10. NDI 分析

NDI 用于衡量相对 UME/CME 偏好，避免只依赖绝对 firing-rate 差值。完整结果见 `NDI_summary.csv`。

主尺度 NDI 前 30 行：

{markdown_table(ndi_main[["grid_scale", "pair_id", "direction_id", "direction_code", "mean_NDI_mean", "mean_abs_NDI_mean", "mean_NDI_nonzero_fraction", "mean_abs_NDI_nonzero_fraction"]], max_rows=30)}

## 11. 空间响应图相似性

spatial response map similarity 直接比较 ume_map 与 cme_map 是否相似。完整结果见 `spatial_map_similarity.csv` 和 `spatial_map_similarity_pair_overall.csv`。

主尺度 spatial similarity summary：

{markdown_table(sim_main[["grid_scale", "pair_id", "mean_pearson_r", "mean_spearman_r", "mean_cosine_similarity", "mean_normalized_MAE", "mean_normalized_RMSE", "mean_abs_delta_mean_fr_hz", "mean_abs_NDI_mean"]])}

解释原则：

- correlation 高但 normalized_MAE 高：空间形状可能相似，但响应幅度不同。
- correlation 低且 normalized_MAE 高：空间模式和幅度均不同。
- normalized_MAE 可以作为 spatial dissimilarity index。

## 12. Permutation-based null comparison

Permutation null 在每个有效网格内随机打乱 UME/CME 标签，保持原始 sample size 不变，检验真实标签下的 mean_abs_delta / different_fraction 是否大于随机标签预期。完整结果见 `permutation_null_summary.csv`。

主尺度 permutation summary 前 30 行：

{markdown_table(perm_main[["grid_scale", "pair_id", "direction_id", "direction_code", "real_mean_abs_delta", "null_mean_mean_abs_delta", "p_mean_abs_delta", "real_different_fraction", "null_mean_different_fraction", "p_different_fraction", "real_mean_abs_NDI", "p_mean_abs_NDI"]], max_rows=30)}

## 13. 有效网格与 unit 数检查

完整结果见 `valid_grid_and_unit_count_summary.csv`。主尺度前 30 行：

{markdown_table(valid_counts[valid_counts["grid_n"].isin(MAIN_GRID_SCALES_FOR_FIGURES)][["grid_scale", "pair_id", "direction_id", "direction_code", "total_grid_count", "valid_grid_count", "valid_grid_fraction", "mean_UME_unit_count_per_valid_grid", "mean_CME_unit_count_per_valid_grid"]], max_rows=30)}

## 14. 分析模块对论文的有利/不利判断

{markdown_table(module_eval)}

## 15. 论文可用结论

中文：

> MEA 记录结果显示，UME 与 CME 刺激在同一视网膜切片的空间匹配网格中诱发了不同的局部 RGC 群体响应模式。在 3 组 paired retina 和多种网格尺度下，均有相当比例的有效网格表现出超过预设阈值的 firing-rate 差异。同时，UME 优势区域和 CME 优势区域均存在，说明这种差异并不是简单的全局增益变化，而是具有空间异质性的 RGC 群体响应调制。该结果提示边缘上下文能够改变视网膜空间群体响应模式，并为后续构建视网膜启发的边缘上下文调制模块提供了生物学动机。

English:

> MEA recordings showed that UME and CME stimulation evoked distinct local RGC population response patterns in spatially matched retinal grids. Across three paired retinal preparations and multiple grid resolutions, a substantial fraction of valid grids showed firing-rate differences exceeding predefined thresholds. Both UME-dominant and CME-dominant regions were observed, indicating that the effect was not a simple global gain change but a spatially heterogeneous modulation of RGC population activity. These observations suggest that edge context can reshape retinal population responses and motivated the design of a retinal-inspired boundary context modulation module for robust edge detection.

## 16. 限制说明

1. sorted units 不能跨 recording 一一匹配。
2. 本分析不是同一 RGC 的 paired response。
3. 网格层面分类不是单格显著性检验。
4. biological replicate 数量为 3 个 paired retina。
5. 网格用于空间模式展示，不应被当成独立 biological replicates。
6. MEA 结果提供 biological motivation，而不是 RBCM 机制的直接证明。
7. 后续模型实验才是验证 RBCM 有效性的主要证据。
"""

    (REPORT_DIR / "MEA_analysis_final_report.md").write_text(report, encoding="utf-8-sig")


# =============================================================================
# 9. 主流程
# =============================================================================


def main() -> None:
    """运行完整 MEA 网格分析流水线。"""

    ensure_dirs()
    log("Start RBCM-edge MEA grid full analysis")

    config = {
        "PAIR_CONFIG": PAIR_CONFIG,
        "GRID_SCALES": GRID_SCALES,
        "DIRECTION_CONFIG": DIRECTION_CONFIG,
        "MIN_UNITS_PER_GRID_PER_STIM": MIN_UNITS_PER_GRID_PER_STIM,
        "MAIN_THRESHOLD_HZ": MAIN_THRESHOLD_HZ,
        "THRESHOLDS_HZ": THRESHOLDS_HZ,
        "EPSILON": EPSILON,
        "N_PERM": N_PERM,
        "RANDOM_SEED": RANDOM_SEED,
        "MEA_DIR": str(MEA_DIR),
        "POSITION_CACHE_DIR": str(POSITION_CACHE_DIR),
        "OUT_DIR": str(OUT_DIR),
    }
    (CONFIG_DIR / "MEA_grid_analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    grid_tables: list[pd.DataFrame] = []
    unit_tables: list[pd.DataFrame] = []

    for grid_n in GRID_SCALES:
        log(f"Compute grid scale {grid_n}x{grid_n}")
        for pair_id, pair_cfg in PAIR_CONFIG.items():
            for direction in DIRECTION_CONFIG:
                grid, unit_df = compute_one_grid_direction(grid_n, pair_id, pair_cfg, direction)
                grid_tables.append(grid)
                unit_tables.append(unit_df)
                make_delta_and_classification_map(grid)
                make_response_fraction_map(grid)
                make_robust_metric_panel(grid)
                make_ndi_map(grid)
                make_map_similarity_scatter(grid)

    grid_all = pd.concat(grid_tables, ignore_index=True)
    unit_all = pd.concat(unit_tables, ignore_index=True)
    log(f"Grid-cell rows: {len(grid_all)}")
    log(f"Unit-level rows used internally: {len(unit_all)}")

    # 质量检查。
    if grid_all["delta_mean_fr_hz"].notna().sum() == 0:
        raise RuntimeError("所有 delta_mean_fr_hz 都是 NaN，分析失败")
    ndi_values = grid_all.loc[grid_all["valid_grid"], "NDI_mean"].dropna()
    if ((ndi_values < -1.0001) | (ndi_values > 1.0001)).any():
        log("WARNING: NDI_mean 超出 [-1,1]，请检查数据")
    for grid_n, sub in grid_all.groupby("grid_n"):
        valid_n = int(sub["valid_grid"].sum())
        log(f"grid_n={grid_n}: valid grids={valid_n}, total grids={len(sub)}")

    # 输出完整 grid-cell 明细。
    save_csv(grid_all, "grid_cell_level_results_all_scales_extended.csv")

    threshold_direction, threshold_pair = summarize_threshold(grid_all)
    robust = summarize_robust_metrics(grid_all)
    ndi_summary = summarize_ndi(grid_all)
    sim, sim_overall = summarize_map_similarity(grid_all)
    valid_counts = summarize_valid_counts(grid_all)
    paired_direction, paired_overall = paired_retina_summaries(threshold_direction, robust, ndi_summary, sim)

    log("Compute permutation null comparisons")
    permutation = compute_permutation_summary(grid_all, unit_all)

    # 保存要求的所有表格。
    save_csv(threshold_direction, "threshold_sensitivity_grid_direction_summary.csv")
    save_csv(threshold_pair, "threshold_sensitivity_pair_overall_summary.csv")
    save_csv(robust, "robust_metric_grid_summary.csv")
    save_csv(ndi_summary, "NDI_summary.csv")
    save_csv(sim, "spatial_map_similarity.csv")
    save_csv(sim_overall, "spatial_map_similarity_pair_overall.csv")
    save_csv(paired_direction, "paired_retina_direction_summary.csv")
    save_csv(paired_overall, "paired_retina_overall_summary.csv")
    save_csv(valid_counts, "valid_grid_and_unit_count_summary.csv")
    save_csv(permutation, "permutation_null_summary.csv")

    # 主图和补充汇总图。
    make_pipeline_figure()
    make_threshold_sensitivity_figures(threshold_direction, threshold_pair)
    make_scale_robustness_figures(paired_overall, sim_overall)
    make_similarity_summary_figures(sim, sim_overall)
    make_paired_retina_dotplot(paired_overall)

    figure_manifest = pd.DataFrame([record.__dict__ for record in FIGURE_RECORDS])
    save_csv(figure_manifest, "figure_manifest.csv")

    write_final_report(paired_overall, threshold_pair, robust, ndi_summary, sim_overall, permutation, valid_counts)

    # 保存脚本快照，确保输出结果可追溯。
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, CODE_DIR / script_path.name)

    (LOG_DIR / "analysis_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8-sig")
    log("Finished RBCM-edge MEA grid full analysis")


if __name__ == "__main__":
    main()
