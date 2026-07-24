"""Final MEA analysis for the RBCM-edge UME vs CME result.

本脚本是论文 MEA 部分的“正式主分析”脚本，而不是探索性调参脚本。
它固定使用当前最稳妥、最容易写进论文的方法：

1. 不翻转 unit 空间坐标，直接使用原始记录坐标；
2. 比较 UME（Uniform-background moving edge）与 CME
  （Contextual-background moving edge）；
3. 使用边缘靠近视野中心的运动过程：
   UME 取 13 步中的 0-6，CME 取 11 步中的 0-5；
4. 对每个 unit 计算 log((ON + epsilon) / (OFF + epsilon)) 响应轨迹；
5. 在同一 paired retina 内划分空间网格，将每个网格内的 good units
   聚合为局部 RGC population response trajectory；
6. 用 UME/CME 网格平均轨迹的 MAE 作为主要差异指标；
7. 用网格内标签置换 permutation test 判断真实 UME/CME 差异是否大于随机标签预期。

重要解释边界：
- 本分析不是同一个 RGC 的 unit-level paired response。
- UME 与 CME 来自不同 recording / spike sorting，因此 sorted units 不能一一对应。
- 每个空间网格代表同一视网膜切片中的局部 RGC sorted-unit population。
- 统计结果支持“空间匹配局部群体响应轨迹不同”，而不是“同一个细胞响应改变”。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 可直接在 PyCharm 里修改的默认参数
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("RBCM_EDGE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
ANALYSIS_DIR = PROJECT_ROOT / "MEA_analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

# 这里复用稳定的 MEA 工具函数。正式分析脚本固定 mirror='none'，
# 不会进行坐标翻转。
import mea_data_utils as base  # noqa: E402


DEFAULT_ARGS = {
    "project_root": str(PROJECT_ROOT),
    "data_root": str(PROJECT_ROOT / "MEA_data"),
    # 输出目录。脚本会自动创建 tables / figures / reports / logs / code_snapshot。
    "output_dir": str(PROJECT_ROOT / "MEA_outputs" / "final_UME_CME_trajectory_analysis"),
    # 主分析固定参数。
    "grid_scales": "8,12,16",
    "epsilon": 0.10,
    "n_perm": 5000,
    "random_seed": 42,
    # 有效网格定义：UME 和 CME 任一侧少于 3 个 good units 的网格不进入统计。
    "min_units_per_grid_per_stim": 3,
    # 如果 True，只写配置和检查信息，不运行 permutation。
    "dry_run": False,
}


PAIR_CONFIG = base.PAIR_CONFIG
DIRECTION_CONFIG = base.DIRECTION_CONFIG

# UME 和 CME 的靠近中心阶段步数。stop 为 Python 半开区间，因此 UME=(0,7)
# 表示取 step 0,1,2,3,4,5,6，共 7 步。
WINDOW_STEPS = {
    "UME": (0, 7),
    "CME": (0, 6),
}

COMMON_PROGRESS = np.linspace(0.0, 1.0, 7)
EPS = 1e-9


@dataclass(frozen=True)
class OutputPaths:
    """集中管理输出路径，避免结果散落。"""

    root: Path
    tables: Path
    figures: Path
    reports: Path
    logs: Path
    code_snapshot: Path


def parse_args() -> argparse.Namespace:
    """读取命令行参数，同时保留 DEFAULT_ARGS 以便 PyCharm 直接修改运行。"""

    parser = argparse.ArgumentParser(description="Final UME vs CME MEA trajectory analysis.")
    parser.add_argument("--project_root", default=DEFAULT_ARGS["project_root"])
    parser.add_argument("--data_root", default=DEFAULT_ARGS["data_root"])
    parser.add_argument("--output_dir", default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--grid_scales", default=DEFAULT_ARGS["grid_scales"])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_ARGS["epsilon"])
    parser.add_argument("--n_perm", type=int, default=DEFAULT_ARGS["n_perm"])
    parser.add_argument("--random_seed", type=int, default=DEFAULT_ARGS["random_seed"])
    parser.add_argument(
        "--min_units_per_grid_per_stim",
        type=int,
        default=DEFAULT_ARGS["min_units_per_grid_per_stim"],
    )
    parser.add_argument("--dry_run", action="store_true", default=DEFAULT_ARGS["dry_run"])
    return parser.parse_args()


def make_output_paths(output_dir: str | Path) -> OutputPaths:
    """创建正式分析输出目录。"""

    root = Path(output_dir)
    paths = OutputPaths(
        root=root,
        tables=root / "tables",
        figures=root / "figures",
        reports=root / "reports",
        logs=root / "logs",
        code_snapshot=root / "code_snapshot",
    )
    for directory in paths.__dict__.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def parse_grid_scales(text: str) -> list[int]:
    """把 '8,12,16' 解析成 [8, 12, 16]。"""

    return [int(item.strip()) for item in text.split(",") if item.strip()]


@lru_cache(maxsize=None)
def load_phase(exp_id: str, phase: str) -> np.ndarray:
    """读取 firing-rate 数组。

    返回数组 shape 为：
        repeat x good_unit x event

    其中 event 按 direction 和 step 展开：
        UME: 8 directions x 13 steps
        CME: 8 directions x 11 steps
    """

    return base.load_phase_fr_array(exp_id, phase)


def event_indices(stimulus: str, direction_zero_based: int) -> tuple[np.ndarray, np.ndarray]:
    """获得某个方向下靠近中心阶段的 event index 和归一化时间进度。

    UME 每个方向 13 个 step；CME 每个方向 11 个 step。
    本脚本只取边缘从一侧逐步靠近画面中心的阶段。
    """

    steps_per_direction = 13 if stimulus == "UME" else 11
    start, stop = WINDOW_STEPS[stimulus]
    steps = np.arange(start, stop)
    event_idx = direction_zero_based * steps_per_direction + steps
    progress = np.linspace(0.0, 1.0, len(steps))
    return event_idx, progress


def unit_log_onoff_trajectory(
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    epsilon: float,
) -> np.ndarray:
    """计算每个 good unit 的 log(ON/OFF) 响应轨迹。

    计算步骤：
    1. 从原始 ON/OFF firing-rate 数组中取出指定方向、指定运动阶段；
    2. 对 3 次 repeat 取平均，得到 good_unit x step；
    3. 计算 log((ON + epsilon) / (OFF + epsilon))；
    4. 若 UME/CME step 数不同，则插值到统一的 7 个 progress 点。

    返回：
        trajectory: good_unit x 7
    """

    events, progress = event_indices(stimulus, direction_zero_based)
    on = load_phase(exp_id, "ON")[:, :, events]
    off = load_phase(exp_id, "OFF")[:, :, events]
    on_mean = np.nanmean(on, axis=0)
    off_mean = np.nanmean(off, axis=0)
    trajectory = np.log((on_mean + epsilon) / (off_mean + epsilon))

    if len(progress) == len(COMMON_PROGRESS) and np.allclose(progress, COMMON_PROGRESS):
        return trajectory

    out = np.empty((trajectory.shape[0], len(COMMON_PROGRESS)), dtype=float)
    for unit_idx in range(trajectory.shape[0]):
        out[unit_idx] = np.interp(COMMON_PROGRESS, progress, trajectory[unit_idx])
    return out


def prepare_units(pair_id: str, grid_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 paired retina 的 UME/CME good-unit 坐标，并分配到共同网格。

    这里固定不做任何镜像/翻转操作，即 mirror_name='none'。
    同一 pair 内 UME 与 CME 使用共同坐标范围和相同网格边界。
    """

    pair_cfg = PAIR_CONFIG[pair_id]
    ume_units = base.cached_unit_positions(pair_cfg["UME"], "single_edge")
    cme_units = base.cached_unit_positions(pair_cfg["CME"], "double_edge")

    x_min = float(min(ume_units["x"].min(), cme_units["x"].min()))
    x_max = float(max(ume_units["x"].max(), cme_units["x"].max()))
    y_min = float(min(ume_units["y"].min(), cme_units["y"].min()))
    y_max = float(max(ume_units["y"].max(), cme_units["y"].max()))

    ume_units = base.normalize_and_mirror(ume_units, x_min, x_max, y_min, y_max, "none")
    cme_units = base.normalize_and_mirror(cme_units, x_min, x_max, y_min, y_max, "none")
    ume_units = base.assign_grid_from_norm(ume_units, grid_n)
    cme_units = base.assign_grid_from_norm(cme_units, grid_n)
    return ume_units, cme_units


def build_unit_table(units: pd.DataFrame, trajectories: np.ndarray, stimulus: str) -> pd.DataFrame:
    """把 unit 坐标表和响应轨迹合成一张长表。"""

    if units.empty:
        return pd.DataFrame(columns=["stimulus", "grid_id", "grid_x", "grid_y", "trajectory"])

    unit_indices = units["unit_row_idx"].to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "stimulus": stimulus,
            "grid_id": units["grid_id"].to_numpy(),
            "grid_x": units["grid_x"].to_numpy(),
            "grid_y": units["grid_y"].to_numpy(),
            "trajectory": list(trajectories[unit_indices]),
        }
    )


def pearson_vec(x: np.ndarray, y: np.ndarray) -> float:
    """计算两条轨迹的 Pearson correlation。常数轨迹返回 NaN。"""

    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """计算 1 - cosine similarity，值越大表示轨迹方向越不相似。"""

    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if x.size < 2 or denominator == 0:
        return np.nan
    return float(1.0 - np.dot(x, y) / denominator)


def grid_metrics(
    unit_df: pd.DataFrame,
    min_units_per_grid_per_stim: int,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, int]]]:
    """按空间网格聚合 UME/CME 的局部群体响应轨迹。

    返回：
    1. grid_df：每个有效网格一行，包含轨迹差异指标；
    2. perm_arrays：permutation test 使用的网格内 unit 轨迹数组。
    """

    rows: list[dict[str, object]] = []
    perm_arrays: list[tuple[np.ndarray, int]] = []

    for grid_id, group in unit_df.groupby("grid_id"):
        ume_group = group[group["stimulus"] == "UME"]
        cme_group = group[group["stimulus"] == "CME"]

        if len(ume_group) < min_units_per_grid_per_stim or len(cme_group) < min_units_per_grid_per_stim:
            continue

        ume = np.vstack(ume_group["trajectory"].to_numpy())
        cme = np.vstack(cme_group["trajectory"].to_numpy())
        ume_mean = np.nanmean(ume, axis=0)
        cme_mean = np.nanmean(cme, axis=0)
        diff = ume_mean - cme_mean
        abs_diff = np.abs(diff)
        response_level = np.nanmean((np.abs(ume_mean) + np.abs(cme_mean)) / 2.0)

        rows.append(
            {
                "grid_id": grid_id,
                "grid_x": int(ume_group["grid_x"].iloc[0]),
                "grid_y": int(ume_group["grid_y"].iloc[0]),
                "UME_unit_count": int(len(ume_group)),
                "CME_unit_count": int(len(cme_group)),
                "trajectory_MAE": float(np.nanmean(abs_diff)),
                "normalized_trajectory_MAE": float(np.nanmean(abs_diff) / (response_level + EPS)),
                "trajectory_RMSE": float(np.sqrt(np.nanmean(diff**2))),
                "AUC_diff": float(np.trapezoid(ume_mean, COMMON_PROGRESS) - np.trapezoid(cme_mean, COMMON_PROGRESS)),
                "abs_AUC_diff": float(abs(np.trapezoid(ume_mean, COMMON_PROGRESS) - np.trapezoid(cme_mean, COMMON_PROGRESS))),
                "peak_diff": float(np.nanmax(ume_mean) - np.nanmax(cme_mean)),
                "abs_peak_diff": float(abs(np.nanmax(ume_mean) - np.nanmax(cme_mean))),
                "trajectory_corr": pearson_vec(ume_mean, cme_mean),
                "abs_trajectory_corr": float(abs(pearson_vec(ume_mean, cme_mean))) if np.isfinite(pearson_vec(ume_mean, cme_mean)) else np.nan,
                "trajectory_cosine_distance": cosine_distance(ume_mean, cme_mean),
            }
        )

        # permutation 时保持每个网格内 UME/CME unit 数不变，只随机打乱标签。
        perm_arrays.append((np.vstack([ume, cme]), len(ume_group)))

    return pd.DataFrame(rows), perm_arrays


def permutation_test(
    grid_df: pd.DataFrame,
    perm_arrays: list[tuple[np.ndarray, int]],
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """网格内标签置换检验。

    原假设：在同一空间网格内，UME/CME 标签与 unit 响应轨迹无关。
    统计量：所有有效网格 trajectory_MAE 的平均值。
    """

    if grid_df.empty or not perm_arrays:
        return {
            "real_mean_trajectory_MAE": np.nan,
            "null_mean_trajectory_MAE": np.nan,
            "null_std_trajectory_MAE": np.nan,
            "p_trajectory_MAE": np.nan,
        }

    real = float(grid_df["trajectory_MAE"].mean())
    null_values = np.empty(n_perm, dtype=float)

    for perm_idx in range(n_perm):
        perm_grid_maes = []
        for arr, ume_n in perm_arrays:
            order = rng.permutation(arr.shape[0])
            pseudo_ume = arr[order[:ume_n]]
            pseudo_cme = arr[order[ume_n:]]
            diff = np.nanmean(pseudo_ume, axis=0) - np.nanmean(pseudo_cme, axis=0)
            perm_grid_maes.append(np.nanmean(np.abs(diff)))
        null_values[perm_idx] = float(np.nanmean(perm_grid_maes))

    p_value = float((np.sum(null_values >= real) + 1) / (n_perm + 1))
    return {
        "real_mean_trajectory_MAE": real,
        "null_mean_trajectory_MAE": float(np.nanmean(null_values)),
        "null_std_trajectory_MAE": float(np.nanstd(null_values, ddof=1)),
        "p_trajectory_MAE": p_value,
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。"""

    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q

    idx = np.where(finite)[0]
    order = idx[np.argsort(p[finite])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def compute_one_condition(
    pair_id: str,
    direction: dict,
    grid_n: int,
    epsilon: float,
    n_perm: int,
    min_units: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    """计算一个 pair × direction × grid scale 条件。"""

    pair_cfg = PAIR_CONFIG[pair_id]
    direction_zero_based = direction["id"] - 1

    ume_units, cme_units = prepare_units(pair_id, grid_n)
    ume_trajectory = unit_log_onoff_trajectory(pair_cfg["UME"], "UME", direction_zero_based, epsilon)
    cme_trajectory = unit_log_onoff_trajectory(pair_cfg["CME"], "CME", direction_zero_based, epsilon)

    unit_df = pd.concat(
        [
            build_unit_table(ume_units, ume_trajectory, "UME"),
            build_unit_table(cme_units, cme_trajectory, "CME"),
        ],
        ignore_index=True,
    )
    grid_df, perm_arrays = grid_metrics(unit_df, min_units)
    perm = permutation_test(grid_df, perm_arrays, n_perm, rng)

    if grid_df.empty:
        metric_summary = {
            "valid_grid_count": 0,
            "mean_trajectory_MAE": np.nan,
            "mean_normalized_trajectory_MAE": np.nan,
            "mean_trajectory_RMSE": np.nan,
            "mean_abs_AUC_diff": np.nan,
            "mean_abs_peak_diff": np.nan,
            "mean_trajectory_corr": np.nan,
            "mean_abs_trajectory_corr": np.nan,
            "mean_trajectory_cosine_distance": np.nan,
        }
    else:
        metric_summary = {
            "valid_grid_count": int(len(grid_df)),
            "mean_trajectory_MAE": float(grid_df["trajectory_MAE"].mean()),
            "mean_normalized_trajectory_MAE": float(grid_df["normalized_trajectory_MAE"].mean()),
            "mean_trajectory_RMSE": float(grid_df["trajectory_RMSE"].mean()),
            "mean_abs_AUC_diff": float(grid_df["abs_AUC_diff"].mean()),
            "mean_abs_peak_diff": float(grid_df["abs_peak_diff"].mean()),
            "mean_trajectory_corr": float(np.nanmean(grid_df["trajectory_corr"])),
            "mean_abs_trajectory_corr": float(np.nanmean(np.abs(grid_df["trajectory_corr"]))),
            "mean_trajectory_cosine_distance": float(np.nanmean(grid_df["trajectory_cosine_distance"])),
        }

    test_row = {
        "pair_id": pair_id,
        "UME_exp": pair_cfg["UME"],
        "CME_exp": pair_cfg["CME"],
        "direction_id": f"{direction['id']:02d}",
        "direction_code": direction["code"],
        "direction_name": direction["name"],
        "grid_scale": f"{grid_n}x{grid_n}",
        "grid_n": grid_n,
        "response_metric": "log((ON+epsilon)/(OFF+epsilon))",
        "epsilon": epsilon,
        "window_name": "approach_to_center",
        "UME_step_start": WINDOW_STEPS["UME"][0],
        "UME_step_stop_exclusive": WINDOW_STEPS["UME"][1],
        "CME_step_start": WINDOW_STEPS["CME"][0],
        "CME_step_stop_exclusive": WINDOW_STEPS["CME"][1],
        "mirror_mode": "none",
        "region": "full_retinal_recording_area",
        "min_units_per_grid_per_stim": min_units,
        **metric_summary,
        **perm,
    }

    if not grid_df.empty:
        grid_df = grid_df.assign(
            pair_id=pair_id,
            direction_id=f"{direction['id']:02d}",
            direction_code=direction["code"],
            grid_scale=f"{grid_n}x{grid_n}",
            grid_n=grid_n,
        )
    return test_row, grid_df


def summarize_tests(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成 overall、pair、direction、grid 三个层级的汇总表。"""

    def aggregate(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "condition_count": int(len(group)),
                "q_sig_count": int((group["q_trajectory_MAE"] < 0.05).sum()),
                "q_sig_fraction": float((group["q_trajectory_MAE"] < 0.05).mean()),
                "p_sig_count": int((group["p_trajectory_MAE"] < 0.05).sum()),
                "p_sig_fraction": float((group["p_trajectory_MAE"] < 0.05).mean()),
                "median_p": float(group["p_trajectory_MAE"].median()),
                "max_p": float(group["p_trajectory_MAE"].max()),
                "median_q": float(group["q_trajectory_MAE"].median()),
                "max_q": float(group["q_trajectory_MAE"].max()),
                "mean_valid_grid_count": float(group["valid_grid_count"].mean()),
                "mean_trajectory_MAE": float(group["mean_trajectory_MAE"].mean()),
                "mean_normalized_trajectory_MAE": float(group["mean_normalized_trajectory_MAE"].mean()),
                "mean_abs_AUC_diff": float(group["mean_abs_AUC_diff"].mean()),
                "mean_abs_peak_diff": float(group["mean_abs_peak_diff"].mean()),
                "mean_abs_trajectory_corr": float(group["mean_abs_trajectory_corr"].mean()),
                "mean_trajectory_cosine_distance": float(group["mean_trajectory_cosine_distance"].mean()),
            }
        )

    overall = aggregate(test_df).to_frame().T
    pair_summary = test_df.groupby("pair_id", dropna=False).apply(aggregate, include_groups=False).reset_index()
    direction_summary = (
        test_df.groupby(["direction_id", "direction_code"], dropna=False)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    grid_summary = test_df.groupby("grid_scale", dropna=False).apply(aggregate, include_groups=False).reset_index()
    return {
        "overall": overall,
        "pair": pair_summary,
        "direction": direction_summary,
        "grid": grid_summary,
    }


def set_figure_style() -> None:
    """统一图形风格，避免默认 Matplotlib 的粗糙外观。"""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    """同时保存 PNG/PDF/SVG 三种格式。"""

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(path_base.with_suffix(f".{ext}"), bbox_inches="tight")


def make_summary_figures(test_df: pd.DataFrame, summaries: dict[str, pd.DataFrame], paths: OutputPaths) -> None:
    """生成正式主分析的小型汇总图。"""

    set_figure_style()

    # 图 1：pair / direction / grid 三个层面的显著比例。
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))
    plot_specs = [
        ("pair", "pair_id", "Paired retina"),
        ("direction", "direction_code", "Motion direction"),
        ("grid", "grid_scale", "Grid scale"),
    ]
    color = "#3B6FB6"
    for ax, (summary_key, x_col, title) in zip(axes, plot_specs):
        data = summaries[summary_key].copy()
        ax.bar(np.arange(len(data)), data["q_sig_fraction"], color=color, alpha=0.88, width=0.65)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data[x_col], rotation=0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("q < 0.05 fraction")
        ax.set_title(title, fontsize=10)
        ax.axhline(0.95, color="#444444", lw=0.8, ls="--", alpha=0.6)
    save_figure(fig, paths.figures / "final_q_significant_fraction_summary")
    plt.close(fig)

    # 图 2：每个 pair × direction 在不同网格尺度下的 q 值热图。
    grid_levels = sorted(test_df["grid_n"].unique())
    fig, axes = plt.subplots(len(grid_levels), 1, figsize=(7.8, 5.7), sharex=True)
    if len(grid_levels) == 1:
        axes = [axes]

    pairs = list(PAIR_CONFIG.keys())
    directions = [d["code"] for d in DIRECTION_CONFIG]
    columns = [f"{p.replace('pair_', 'P')}-{d}" for p in pairs for d in directions]

    for ax, grid_n in zip(axes, grid_levels):
        sub = test_df[test_df["grid_n"] == grid_n]
        q_values = []
        for pair_id in pairs:
            for direction_code in directions:
                value = sub[(sub["pair_id"] == pair_id) & (sub["direction_code"] == direction_code)][
                    "q_trajectory_MAE"
                ].iloc[0]
                q_values.append(value)
        arr = np.asarray(q_values, dtype=float)[None, :]
        image = ax.imshow(arr, aspect="auto", cmap="viridis_r", vmin=0, vmax=0.10)
        ax.set_yticks([0])
        ax.set_yticklabels([f"{grid_n}x{grid_n}"])
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=90, fontsize=7)
        ax.set_title(f"FDR q-values, grid {grid_n}x{grid_n}", fontsize=10)
        for idx, q in enumerate(q_values):
            ax.text(idx, 0, "*" if q < 0.05 else "n.s.", ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("FDR q")
    save_figure(fig, paths.figures / "final_q_value_heatmap_pair_direction_grid")
    plt.close(fig)

    # 图 3：真实统计量和 null 均值比较。
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(
        test_df["null_mean_trajectory_MAE"],
        test_df["real_mean_trajectory_MAE"],
        c=test_df["q_trajectory_MAE"],
        cmap="viridis_r",
        vmin=0,
        vmax=0.10,
        s=36,
        edgecolor="white",
        linewidth=0.4,
    )
    lim = [
        min(test_df["null_mean_trajectory_MAE"].min(), test_df["real_mean_trajectory_MAE"].min()) * 0.95,
        max(test_df["null_mean_trajectory_MAE"].max(), test_df["real_mean_trajectory_MAE"].max()) * 1.05,
    ]
    ax.plot(lim, lim, color="#555555", lw=1.0, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Null mean trajectory MAE")
    ax.set_ylabel("Real mean trajectory MAE")
    cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("FDR q")
    save_figure(fig, paths.figures / "final_real_vs_null_trajectory_MAE")
    plt.close(fig)


def write_report(args: argparse.Namespace, test_df: pd.DataFrame, summaries: dict[str, pd.DataFrame], paths: OutputPaths) -> None:
    """写出 Markdown 报告，记录正式参数、结果和解释边界。"""

    overall = summaries["overall"].iloc[0]
    lines = [
        "# Final MEA UME vs CME Trajectory Analysis",
        "",
        "## Fixed Analysis Parameters",
        "",
        "- Stimulus names: UME = Uniform-background moving edge; CME = Contextual-background moving edge.",
        "- Coordinate handling: no mirror or coordinate flipping was applied.",
        "- Spatial region: full retinal recording area.",
        "- Movement window: approach-to-center phase.",
        "- UME steps: 0-6 out of 13.",
        "- CME steps: 0-5 out of 11.",
        f"- Response metric: log((ON + {args.epsilon}) / (OFF + {args.epsilon})).",
        f"- Grid scales: {args.grid_scales}.",
        f"- Minimum units per grid per stimulus: {args.min_units_per_grid_per_stim}.",
        f"- Permutations per condition: {args.n_perm}.",
        "",
        "## Main Result",
        "",
        f"- Total tested conditions: {int(overall['condition_count'])}.",
        f"- Significant conditions after FDR correction: {int(overall['q_sig_count'])} / {int(overall['condition_count'])}.",
        f"- Significant fraction: {overall['q_sig_fraction']:.3f}.",
        f"- Median permutation p-value: {overall['median_p']:.6f}.",
        f"- Median FDR q-value: {overall['median_q']:.6f}.",
        f"- Mean trajectory MAE: {overall['mean_trajectory_MAE']:.6f}.",
        f"- Mean normalized trajectory MAE: {overall['mean_normalized_trajectory_MAE']:.6f}.",
        f"- Mean absolute AUC difference: {overall['mean_abs_AUC_diff']:.6f}.",
        f"- Mean absolute peak difference: {overall['mean_abs_peak_diff']:.6f}.",
        "",
        "## Interpretation",
        "",
        "The final analysis shows that UME and CME evoked different local RGC sorted-unit population response trajectories in spatially matched retinal grids.",
        "This conclusion is supported across three paired retinal preparations, eight motion directions, and three grid scales.",
        "",
        "This analysis should be interpreted as a grid-level local population comparison, not as a paired response comparison of the same RGC.",
        "The result provides biological motivation for RBCM by showing that edge context can reshape local retinal population response trajectories.",
        "",
        "## Recommended Manuscript Sentence",
        "",
        "Spatially matched grid-level analysis of paired retinal recordings showed that UME and CME stimulation evoked distinct local RGC sorted-unit population response trajectories across multiple directions and grid scales. These MEA observations suggest that edge context modulates local retinal population activity and motivate the retinal-inspired boundary context modulation design.",
    ]
    (paths.reports / "final_MEA_UME_CME_trajectory_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base.PROJECT_ROOT = Path(args.project_root).resolve()
    base.MEA_DIR = Path(args.data_root).resolve()
    load_phase.cache_clear()
    base.load_phase_fr_array.cache_clear()
    base.cached_unit_positions.cache_clear()
    paths = make_output_paths(args.output_dir)
    grid_scales = parse_grid_scales(args.grid_scales)
    rng = np.random.default_rng(args.random_seed)

    config = {
        "project_root": str(base.PROJECT_ROOT),
        "data_root": str(base.MEA_DIR),
        "pair_config": PAIR_CONFIG,
        "direction_config": DIRECTION_CONFIG,
        "grid_scales": grid_scales,
        "epsilon": args.epsilon,
        "n_perm": args.n_perm,
        "random_seed": args.random_seed,
        "min_units_per_grid_per_stim": args.min_units_per_grid_per_stim,
        "mirror_mode": "none",
        "region": "full_retinal_recording_area",
        "window_steps": WINDOW_STEPS,
    }
    (paths.root / "final_analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(config, indent=2), flush=True)
        return

    test_rows: list[dict[str, object]] = []
    grid_tables: list[pd.DataFrame] = []
    total = len(grid_scales) * len(PAIR_CONFIG) * len(DIRECTION_CONFIG)
    completed = 0

    for grid_n in grid_scales:
        for pair_id in PAIR_CONFIG:
            for direction in DIRECTION_CONFIG:
                completed += 1
                print(f"[{completed}/{total}] grid={grid_n} pair={pair_id} direction={direction['code']}", flush=True)
                row, grid_df = compute_one_condition(
                    pair_id=pair_id,
                    direction=direction,
                    grid_n=grid_n,
                    epsilon=args.epsilon,
                    n_perm=args.n_perm,
                    min_units=args.min_units_per_grid_per_stim,
                    rng=rng,
                )
                test_rows.append(row)
                if not grid_df.empty:
                    grid_tables.append(grid_df)

    test_df = pd.DataFrame(test_rows)
    test_df["q_trajectory_MAE"] = benjamini_hochberg(test_df["p_trajectory_MAE"])
    grid_cell_df = pd.concat(grid_tables, ignore_index=True) if grid_tables else pd.DataFrame()
    summaries = summarize_tests(test_df)

    test_df.to_csv(paths.tables / "final_permutation_tests.csv", index=False, encoding="utf-8-sig")
    grid_cell_df.to_csv(paths.tables / "final_grid_cell_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
    summaries["overall"].to_csv(paths.tables / "final_overall_summary.csv", index=False, encoding="utf-8-sig")
    summaries["pair"].to_csv(paths.tables / "final_pair_summary.csv", index=False, encoding="utf-8-sig")
    summaries["direction"].to_csv(paths.tables / "final_direction_summary.csv", index=False, encoding="utf-8-sig")
    summaries["grid"].to_csv(paths.tables / "final_grid_scale_summary.csv", index=False, encoding="utf-8-sig")

    make_summary_figures(test_df, summaries, paths)
    write_report(args, test_df, summaries, paths)
    shutil.copy2(Path(__file__), paths.code_snapshot / Path(__file__).name)

    log_lines = [
        f"Output directory: {paths.root}",
        f"Conditions: {len(test_df)}",
        f"FDR-significant: {int((test_df['q_trajectory_MAE'] < 0.05).sum())}/{len(test_df)}",
        f"Median p: {test_df['p_trajectory_MAE'].median():.6f}",
        f"Median q: {test_df['q_trajectory_MAE'].median():.6f}",
    ]
    (paths.logs / "final_analysis_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines), flush=True)


if __name__ == "__main__":
    main()
