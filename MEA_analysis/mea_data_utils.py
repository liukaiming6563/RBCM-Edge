"""Shared MEA data-loading utilities for RBCM-edge analyses.

这个文件只放稳定的通用工具函数：读取 firing-rate 数组、读取 good-unit
空间坐标、归一化坐标和网格分配。正式分析脚本依赖本文件，而探索性调参脚本
可以被归档，不会影响主分析复现。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import run_MEA_grid_full_analysis as grid_base


PROJECT_ROOT = Path(os.environ.get("RBCM_EDGE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
MEA_DIR = Path(os.environ.get("RBCM_EDGE_MEA_DATA", PROJECT_ROOT / "MEA_data")).resolve()

PAIR_CONFIG = grid_base.PAIR_CONFIG
DIRECTION_CONFIG = grid_base.DIRECTION_CONFIG
MIN_UNITS_PER_GRID_PER_STIM = grid_base.MIN_UNITS_PER_GRID_PER_STIM


@lru_cache(maxsize=None)
def load_phase_fr_array(exp_id: str, phase: str) -> np.ndarray:
    """读取某次实验、某个阶段的 firing-rate 数组。

    参数：
        exp_id: 实验编号，例如 "000031"。
        phase: "ON" 或 "OFF"。

    返回：
        shape = repeat x good_unit x event 的 numpy 数组。
    """

    phase_key = phase.lower()
    if phase_key not in {"on", "off"}:
        raise ValueError(f"phase must be ON or OFF, got {phase!r}")

    path = MEA_DIR / exp_id / "segment_result" / "processed_segment" / f"good_{phase_key}" / "output_fre.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing firing-rate array: {path}")

    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected repeat x unit x event array, got {arr.shape} at {path}")
    if arr.shape[0] < 1:
        raise ValueError(f"No repeats found in {path}")
    if np.nanmin(arr) < 0:
        raise ValueError(f"Negative firing rates found in {path}")
    return arr


@lru_cache(maxsize=None)
def cached_unit_positions(exp_id: str, stim_class: str) -> pd.DataFrame:
    """读取并缓存 good-unit 空间坐标。

    stim_class 保留是为了兼容旧脚本的调用形式；实际坐标读取由
    run_MEA_grid_full_analysis.load_unit_positions 负责。
    """

    return grid_base.load_unit_positions(exp_id, stim_class).copy()


def normalize_and_mirror(
    units: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    mirror_name: str = "none",
) -> pd.DataFrame:
    """把 unit 坐标归一化到 [0, 1]，并按需执行镜像。

    正式主分析固定使用 mirror_name="none"。
    """

    out = units.copy()
    out["x_norm"] = (out["x"] - x_min) / (x_max - x_min)
    out["y_norm"] = (out["y"] - y_min) / (y_max - y_min)

    if mirror_name == "none":
        pass
    elif mirror_name == "flip_x":
        out["x_norm"] = 1.0 - out["x_norm"]
    elif mirror_name == "flip_y":
        out["y_norm"] = 1.0 - out["y_norm"]
    elif mirror_name == "flip_xy":
        out["x_norm"] = 1.0 - out["x_norm"]
        out["y_norm"] = 1.0 - out["y_norm"]
    else:
        raise ValueError(f"Unknown mirror mode: {mirror_name}")

    return out


def assign_grid_from_norm(units: pd.DataFrame, grid_n: int) -> pd.DataFrame:
    """根据归一化坐标把 unit 分配到 N x N 网格。"""

    out = units.copy()
    out["grid_x"] = np.clip(np.floor(out["x_norm"].to_numpy() * grid_n).astype(int), 0, grid_n - 1)
    out["grid_y"] = np.clip(np.floor(out["y_norm"].to_numpy() * grid_n).astype(int), 0, grid_n - 1)
    out["grid_id"] = out["grid_y"].astype(str).str.zfill(2) + "_" + out["grid_x"].astype(str).str.zfill(2)
    return out


def motion_progress(x: pd.Series, y: pd.Series, direction_code: str) -> pd.Series:
    """计算某方向下从运动起始侧到终止侧的归一化空间进度。

    该函数主要供探索性区域筛选脚本使用；正式主分析使用全视网膜区域。
    """

    direction_code = direction_code.upper()
    if direction_code == "R":
        return x
    if direction_code == "L":
        return 1.0 - x
    if direction_code == "U":
        return 1.0 - y
    if direction_code == "D":
        return y
    if direction_code == "RU":
        return (x + (1.0 - y)) / 2.0
    if direction_code == "LU":
        return ((1.0 - x) + (1.0 - y)) / 2.0
    if direction_code == "LD":
        return ((1.0 - x) + y) / 2.0
    if direction_code == "RD":
        return (x + y) / 2.0
    raise ValueError(f"Unknown direction code: {direction_code}")
