#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光学图像 -> ISAR 效果（YOLO 2.0 版）

本版本说明（接口/文件名/函数名/权重调用方式与旧版完全一致，仅替换核心算法）：
1) 核心算法已替换为：光学图像 → 目标分割(Otsu)/YOLO掩码 → 散射点云 →
   K分布海杂波 → RDA成像（回波正演+距离压缩+RCMC+方位压缩）→ 40dB 显示，
   与"双输入 SAR/ISAR 演示系统 v2.2.2"部署系统逐位一致
2) YOLO 权重调用方式不变：默认 <脚本上级目录>/optical2sar/weights/optyolo/best.pt，
   直接替换 best.pt 文件即换模型；权重缺失/未检出时自动回退内置 Otsu 分割
3) 旧版显著性/散点参数（--saliency-threshold/--feather/--white-ratio/
   --randomness/--brightness）保留占位兼容旧调用，新算法不再使用

依赖：
  pip install opencv-python numpy ultralytics pillow
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

import cv2
# Windows 下避免 OpenMP 运行时重复初始化告警（与部署系统一致）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from flask import Flask, request, jsonify, send_file, Blueprint

# 将 Flask 应用改为 Blueprint
optical2sar_bp = Blueprint('optical2sar', __name__)

# 相对路径解析基准（修复旧版 base 未定义问题）
base = Path(__file__).resolve().parent


# =========================================================================
# 0. 常量与雷达参数
# =========================================================================

C_LIGHT = 2.9979e8

# 11 维雷达参数默认值（与部署系统一致）：
# [平台高度Hc(m), 斜视角thetaSQ(deg), 侧视角thetaSL(deg), 平台速度vx(m/s),
#  距离向天线孔径D_rg(m), 方位向天线孔径D_az(m), PRF(Hz), 脉宽T(s),
#  载频f0(Hz), 发射带宽Br(Hz), 距离向采样率fs(Hz)]
DEFAULT_PARAMETERS = np.array(
    [4999.0, 0.0, 45.0, 100.0, 2.0, 8.0, 30.0, 5e-6, 1e9, 80e6, 100e6],
    dtype=np.float64,
)

PARAMETER_FIELDS = {
    "Hc": 0, "thetaSQ": 1, "thetaSL": 2, "vx": 3, "D_rg": 4, "D_az": 5,
    "PRF": 6, "T": 7, "f0": 8, "Br": 9, "fs": 10,
}


def validate_parameters(parameters: np.ndarray | list[float] | None) -> np.ndarray:
    """校验并返回 float64 的 11 维雷达参数向量（None 时用默认值）。"""
    params = (
        DEFAULT_PARAMETERS.copy()
        if parameters is None
        else np.asarray(parameters, dtype=np.float64).ravel()
    )
    if params.size != 11:
        raise ValueError("radar parameters must contain exactly 11 values")
    if not np.isfinite(params).all():
        raise ValueError("radar parameters must be finite")
    positive_indices = (0, 3, 4, 5, 6, 7, 8, 9, 10)
    if np.any(params[list(positive_indices)] <= 0.0):
        raise ValueError("Hc, speed, antenna sizes, PRF, pulse width, f0, Br and fs must be positive")
    if abs(float(params[2])) >= 89.0:
        raise ValueError("side-looking angle must be between -89 and 89 degrees")
    return params


def load_image_as_array(image_path: str | Path) -> np.ndarray:
    """读取图像文件（jpg/png/bmp/...）为 numpy 数组。"""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return np.asarray(im)
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {image_path}") from e


def _to_array(source: Any) -> np.ndarray:
    """路径 / ndarray / PIL.Image 统一转 ndarray。"""
    if isinstance(source, (str, Path)):
        return load_image_as_array(source)
    return np.asarray(source)


# =========================================================================
# 1. 目标分割（auto Otsu / manual 阈值 / 外部掩码）
#    转写自 scene_segmentation.py（仅保留 auto 与 manual，不含 onnx/hybrid_v2）
#
# 【理论速览】
#   · auto 模式：Otsu 最大类间方差自动取阈值；亮目标(舰体亮于海面)与
#     暗目标(烈日反光下舰体暗于海面)两种极性各跑一遍 Otsu+连通域，
#     按评分取更优——自动适配两种海面场景。
#   · 连通域评分：从阈值掩码的连通域（scipy.ndimage.label）里挑"最像
#     舰船"的那块：面积占比的对数高斯偏好（约 5% 面积最理想）+
#     居中度 + 外接框填充率 - 边界接触惩罚（贴边多半是码头/暗角）。
#   · 外部掩码：YOLO 等前端检测出的舰体掩码可直接喂入，跳过内置分割。
# =========================================================================


@dataclass
class SegmentationResult:
    """一次分割的结果：掩码 + 用于散射点幅度的信号图 + 诊断信息。"""

    mask: np.ndarray
    gray: np.ndarray
    signal: np.ndarray
    method: str
    polarity: str
    threshold: float | None
    warning: str | None = None
    score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB/灰度图 → [0,255] float64 灰度（ITU-R 601 加权，与原实现一致）。"""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        if img.shape[2] < 3:
            raise ValueError("color image must have at least three channels")
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    elif img.ndim != 2:
        raise ValueError("image must be 2-D gray or 3-D color")
    return np.clip(img, 0.0, 255.0)


def otsu_threshold(gray01: np.ndarray) -> float:
    """对 [0,1] 归一化灰度做 256-bin Otsu，返回阈值。

    Otsu（1979 最大类间方差）原理：把像素按阈值分成前景/背景两类，
    类间方差 σb²(t) = w0(t)·w1(t)·[μ0(t)-μ1(t)]²，使 σb² 最大的 t 即
    最优二值化阈值（类间分得最开 = 类内最紧）。本实现用 256 级直方图
    + 累加和一次扫描完成，复杂度 O(N+256)。
    """
    g = np.asarray(gray01, dtype=np.float64).ravel()
    g = g[np.isfinite(g)]
    if g.size == 0:
        return 0.5
    hist, edges = np.histogram(g, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    level = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum_all = np.cumsum(hist * level)
    mu0 = np.divide(sum_all, w0, out=np.zeros_like(sum_all), where=w0 > 0)
    mu1 = np.divide(sum_all[-1] - sum_all, w1, out=np.zeros_like(sum_all), where=w1 > 0)
    var_between = w0 * w1 * (mu0 - mu1) ** 2
    best = int(np.argmax(var_between))
    return float(np.clip(level[best], 0.0, 1.0))


def _apply_roi(mask: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """把掩码限制到 ROI 矩形 [x0,y0,x1,y1] 内（None=整幅）。"""
    if roi is None:
        return np.asarray(mask, dtype=bool)
    h, w = mask.shape
    x0, y0, x1, y1 = map(int, roi)
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have positive width and height")
    out = np.zeros_like(mask, dtype=bool)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def _component_candidates(mask: np.ndarray) -> list[np.ndarray]:
    """连通域候选：取面积最大的前 12 个（scipy 可用；缺失时整幅返回）。"""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count == 0:
            return []
        sizes = np.bincount(labels.ravel())
        order = np.argsort(sizes[1:])[::-1][: min(count, 12)] + 1
        return [labels == int(i) for i in order if sizes[int(i)] > 0]
    except Exception:
        return [mask]


def _score_component(mask: np.ndarray) -> float:
    """连通域目标评分（面积占比偏好 + 居中度 + 填充率 - 边界接触惩罚）。"""
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return -np.inf
    ratio = area / float(h * w)
    if ratio < 0.0003 or ratio > 0.72:
        return -100.0 - abs(ratio - 0.15)
    yy, xx = np.nonzero(mask)
    cx = float(xx.mean()) / max(w - 1, 1)
    cy = float(yy.mean()) / max(h - 1, 1)
    center = 1.0 - min(1.0, np.hypot(cx - 0.5, cy - 0.5) / 0.7071)
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    border_fraction = float(border.mean())
    bw = int(xx.max() - xx.min() + 1)
    bh = int(yy.max() - yy.min() + 1)
    fill = area / float(max(1, bw * bh))
    area_pref = np.exp(-((np.log10(max(ratio, 1e-6)) + 1.25) / 1.3) ** 2)
    return 2.2 * area_pref + 1.1 * center + 0.5 * fill - 3.0 * border_fraction


def _best_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """从连通域候选里选评分最高的作为目标掩码。"""
    candidates = _component_candidates(mask)
    if not candidates:
        return np.zeros_like(mask, dtype=bool), -np.inf
    scores = [_score_component(c) for c in candidates]
    best = int(np.argmax(scores))
    return candidates[best], float(scores[best])


def mask_diagnostics(mask: np.ndarray, score: float | None = None) -> dict[str, Any]:
    """掩码质量诊断（面积/填充率/细长度/边界接触与告警）。"""
    binary = np.asarray(mask, dtype=bool)
    h, w = binary.shape
    area = int(binary.sum())
    if area == 0:
        return {
            "area_pixels": 0, "area_ratio": 0.0, "border_fraction": 0.0,
            "fill_ratio": 0.0, "elongation": 0.0, "confidence": "invalid",
            "warnings": ["分割结果为空"],
        }
    yy, xx = np.nonzero(binary)
    bw = int(xx.max() - xx.min() + 1)
    bh = int(yy.max() - yy.min() + 1)
    fill = area / float(max(1, bw * bh))
    border_pixels = int(binary[0].sum() + binary[-1].sum() + binary[:, 0].sum() + binary[:, -1].sum())
    border_fraction = border_pixels / float(max(1, 2 * (h + w)))
    coords = np.column_stack((xx - xx.mean(), yy - yy.mean())).astype(np.float64)
    if coords.shape[0] >= 3:
        eig = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
        elongation = float(np.sqrt(max(eig[-1], 1e-12) / max(eig[0], 1e-12)))
    else:
        elongation = 0.0
    area_ratio = area / float(h * w)
    warnings: list[str] = []
    if border_pixels:
        warnings.append("目标蒙版接触图像/ROI边界，可能包含码头、尾迹或被截断目标")
    if area_ratio > 0.35:
        warnings.append("目标面积占比过大，可能把背景并入目标")
    if elongation > 7.0 and fill < 0.35:
        warnings.append("蒙版存在细长低填充结构，可能包含尾迹、吊臂或码头")
    if fill < 0.12:
        warnings.append("目标蒙版较稀疏，可能由反光或纹理碎片组成")
    if score is not None and np.isfinite(score) and score < 2.75:
        warnings.append("自动候选评分偏低，建议在叠加预览中人工确认")
    confidence = "low" if warnings else ("high" if score is None or score >= 3.0 else "medium")
    return {
        "area_pixels": area,
        "area_ratio": float(area_ratio),
        "bbox_xywh": [int(xx.min()), int(yy.min()), bw, bh],
        "border_fraction": float(border_fraction),
        "fill_ratio": float(fill),
        "elongation": elongation,
        "candidate_score": None if score is None else float(score),
        "confidence": confidence,
        "warnings": warnings,
    }


def _finalize_result(result: SegmentationResult) -> SegmentationResult:
    """补齐诊断信息并汇总告警（与原 _finalize_result 等价，edits 不再支持）。"""
    if not result.mask.any():
        raise ValueError("mask removed the complete target")
    result.diagnostics = mask_diagnostics(result.mask, result.score)
    warnings = list(result.diagnostics.get("warnings", []))
    if result.warning:
        warnings = str(result.warning).split("；") + warnings
    warnings = list(dict.fromkeys(item for item in warnings if item))
    result.warning = "；".join(warnings) if warnings else None
    return result


def segment_auto(image: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> SegmentationResult:
    """自动分割：Otsu 阈值 + 亮/暗双极性各自选最优连通域，取评分高者。"""
    gray = to_gray(image)
    peak = float(gray.max())
    if peak <= 0:
        raise ValueError("image contains no positive intensity")
    g01 = gray / peak
    bright_thr = max(0.12, otsu_threshold(g01) * 0.95)
    dark_signal = 1.0 - g01
    dark_thr = max(0.12, otsu_threshold(dark_signal) * 0.95)

    bright, bright_score = _best_mask(_apply_roi(g01 >= bright_thr, roi))
    dark, dark_score = _best_mask(_apply_roi(dark_signal >= dark_thr, roi))
    if not bright.any() and not dark.any():
        raise ValueError("automatic segmentation found no target")
    if dark_score > bright_score:
        mask, polarity, threshold, signal, score = (
            dark, "dark", (1.0 - dark_thr) * peak, 255.0 - gray, dark_score,
        )
    else:
        mask, polarity, threshold, signal, score = (
            bright, "bright", bright_thr * peak, gray, bright_score,
        )
    return _finalize_result(SegmentationResult(
        mask=mask,
        gray=gray,
        signal=np.clip(signal, 0.0, 255.0),
        method="auto",
        polarity=polarity,
        threshold=float(threshold),
        score=float(score),
    ))


def segment_manual(
    image: np.ndarray,
    threshold: float,
    polarity: str = "bright",
    largest_component: bool = True,
    roi: tuple[int, int, int, int] | None = None,
) -> SegmentationResult:
    """人工阈值分割：polarity=bright 取亮目标，dark 取暗目标（先反转）。"""
    gray = to_gray(image)
    polarity = str(polarity).lower()
    if polarity not in {"bright", "dark"}:
        raise ValueError("polarity must be 'bright' or 'dark'")
    threshold = float(threshold)
    signal = gray if polarity == "bright" else 255.0 - gray
    mask = signal >= threshold
    mask = _apply_roi(mask, roi)
    score = None
    if largest_component:
        mask, score = _best_mask(mask)
    if not mask.any():
        raise ValueError("manual segmentation found no target")
    return _finalize_result(SegmentationResult(
        mask=mask,
        gray=gray,
        signal=np.clip(signal, 0.0, 255.0),
        method="manual",
        polarity=polarity,
        threshold=threshold,
        score=score,
    ))


def load_external_mask(mask: Any, image: np.ndarray, polarity: str = "bright") -> SegmentationResult:
    """把外部掩码（如 YOLO 舰船检测结果）包装成分割结果。

    mask 可为：
      * np.ndarray —— bool，或数值（非 0 即目标）；3 通道时取第 1 通道；
      * str / Path —— 掩码图像文件（PNG 等，非 0 像素视为目标）。

    掩码尺寸必须与输入图像一致。散射点幅度取自光学灰度：
    polarity="bright" 时幅度随亮度增大；舰船比海面暗的场景用 "dark"。
    """
    if isinstance(mask, (str, Path)):
        from PIL import Image

        with Image.open(mask) as mask_image:
            arr = np.asarray(mask_image.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
    if arr.dtype == bool:
        bool_mask = arr
    else:
        bool_mask = arr > 0
    if bool_mask.shape != np.asarray(image).shape[:2]:
        raise ValueError(
            f"mask size {bool_mask.shape} does not match the optical source {np.asarray(image).shape[:2]}"
        )
    if not bool_mask.any():
        raise ValueError("external mask is empty")
    polarity = str(polarity).lower()
    if polarity not in {"bright", "dark"}:
        raise ValueError("mask polarity must be 'bright' or 'dark'")
    gray = to_gray(image)
    signal = gray if polarity == "bright" else 255.0 - gray
    return SegmentationResult(
        mask=bool_mask,
        gray=gray,
        signal=np.clip(signal, 0.0, 255.0),
        method="external_mask",
        polarity=polarity,
        threshold=None,
    )


# =========================================================================
# 2. 掩码 → 散射点云 → 波束中心放置
#
# 【理论速览】光学图像没有天然的"散射系数"，这里采用模板驱动假设：
#   掩码内每个抽样像素视为一个点散射体，其后向散射幅度 σ 正比于该像素
#   的光学灰度（亮屋顶=强散射，暗舷侧=弱散射）；点坐标 (x,y) 按像素
#   步长×pixel_size 换算成米，并平移到波束中心地面投影处，使场景落入
#   天线波束（否则 RDA 会报"无散射点在波束内"）。
# =========================================================================


def _auto_step(mask: np.ndarray, requested: int | None, max_points: int = 30_000) -> int:
    """未指定抽稀步长时逐步加倍抽样直到点数不超过上限（默认 30000）。"""
    if requested is not None:
        if int(requested) < 1:
            raise ValueError("step must be >= 1")
        return int(requested)
    step = 1
    while step < 32 and int(mask[::step, ::step].sum()) > int(max_points):
        step += 1
    return step


def mask_to_points(
    segmentation: SegmentationResult,
    pixel_size_x: float = 1.0,
    pixel_size_y: float = 1.0,
    z0: float = 0.0,
    step: int = 1,
    center: bool = True,
) -> np.ndarray:
    """掩码 → 散射点云 N×4 [方位x, 地距y, z, 幅度]。

    幅度取分割信号图（灰度或反转灰度）并做下限截断：
    正幅度 5% 分位的 0.1 倍（避免零幅度点）。
    """
    if int(step) < 1:
        raise ValueError("step must be >= 1")
    step = int(step)
    mask = segmentation.mask[::step, ::step]
    signal = segmentation.signal[::step, ::step]
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    x = xx.astype(np.float64) * step * float(pixel_size_x)
    y = yy.astype(np.float64) * step * float(pixel_size_y)
    if center:
        x -= (x.min() + x.max()) / 2.0
        y -= (y.min() + y.max()) / 2.0
    amp = signal[mask].astype(np.float64)
    positive = amp[amp > 0]
    floor = float(np.percentile(positive, 5)) if positive.size else 1.0
    amp = np.maximum(amp, max(floor * 0.1, 1e-6))
    z = np.full_like(x, float(z0))
    return np.column_stack((x, y, z, amp))


def place_scene_at_beam(P: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """平移点云，使图像中心落在波束中心地面投影 (xc,yc)。

    (x_c, y_c) = (Hc/cos(thetaSL)*tan(thetaSQ), Hc*tan(thetaSL))。
    """
    p = np.asarray(P, dtype=np.float64).copy()
    if p.ndim != 2 or p.shape[1] != 4 or p.shape[0] == 0:
        return p
    params = np.asarray(parameters, dtype=np.float64).ravel()
    Hc, thetaSQ_deg, thetaSL_deg = map(float, params[:3])
    thetaSQ = np.deg2rad(thetaSQ_deg)
    thetaSL = np.deg2rad(thetaSL_deg)
    xc = Hc / np.cos(thetaSL) * np.tan(thetaSQ)
    yc = Hc * np.tan(thetaSL)
    p[:, 0] += xc
    p[:, 1] += yc
    return p


def save_points_csv(P: np.ndarray, csv_path: str | Path) -> None:
    """点云 N×4 存 CSV（表头 x,y,z,amplitude）。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_path,
        P,
        delimiter=",",
        fmt="%.10g",
        header="x,y,z,amplitude",
        comments="",
    )


# =========================================================================
# 3. YOLO 舰船检测（权重调用方式与旧版 optical_to_isar_v2_yolo 保持一致）
#    默认权重路径: <脚本上级目录>/optical2sar/weights/optyolo/best.pt，
#    直接替换 best.pt 文件即可换用新训练的权重；
#    亦可用 --yolo-weights / options.yolo.weights / 环境变量
#    OPTICAL2SAR_YOLO_WEIGHTS 指定其他路径。
# =========================================================================


def default_yolo_weights_path() -> Path:
    """旧版约定的默认 YOLO 权重路径：<脚本上级目录>/optical2sar/weights/optyolo/best.pt。"""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "optical2sar" / "weights" / "optyolo" / "best.pt"


def resolve_yolo_weights(yolo_weights: str | Path | None = None) -> Path:
    """权重路径解析顺序：显式参数 → 环境变量 OPTICAL2SAR_YOLO_WEIGHTS → 旧版默认路径。"""
    if yolo_weights:
        return Path(yolo_weights)
    env = os.environ.get("OPTICAL2SAR_YOLO_WEIGHTS")
    if env:
        return Path(env)
    return default_yolo_weights_path()


# 权重模型缓存: {路径: (文件 mtime, model)}——运行中替换 best.pt 会自动重载
_YOLO_MODEL_CACHE: dict[str, tuple[float, Any]] = {}


def _load_yolo_model(yolo_weights: str | Path):
    """加载 ultralytics YOLO 权重（与旧接口同名同参；文件未变更时复用缓存）。"""
    weights = Path(yolo_weights)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO 权重不存在: {weights}")
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise ImportError("缺少 ultralytics：pip install ultralytics") from e
    mtime = weights.stat().st_mtime
    cached = _YOLO_MODEL_CACHE.get(str(weights))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    model = YOLO(str(weights))
    _YOLO_MODEL_CACHE[str(weights)] = (mtime, model)
    return model


def _det_to_numpy(value: Any) -> np.ndarray:
    """兼容 torch 张量 / numpy 数组两种 YOLO 结果字段。"""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rasterize_polygons(polygons: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """多边形栅格化并集（用 Pillow 实现，避免引入 opencv）。"""
    from PIL import Image, ImageDraw

    layer = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(layer)
    for pts in polygons:
        ints = [(float(x), float(y)) for x, y in np.asarray(pts, dtype=np.float64).reshape(-1, 2)]
        if len(ints) >= 3:
            draw.polygon(ints, fill=255)
    return (np.asarray(layer) > 0).astype(np.uint8)


def _yolo_ship_mask(
    optical_img: np.ndarray,
    yolo_model: Any,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.45,
    use_obb: bool = True,
    max_dets: int = 1,
) -> np.ndarray | None:
    """运行 YOLO 得到舰体掩码（0/1 uint8），与旧版同名同参同语义。

    优先 OBB（旋转框多边形填充），其次实例分割掩码（新增支持），
    最后退化为 HBB（水平框矩形填充）。未检出任何目标时返回 None。
    """
    h, w = np.asarray(optical_img).shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # YOLO 需要 3 通道图；与旧版 cv2 管线一致按 BGR 通道序送入
    img = np.asarray(optical_img)
    if img.ndim == 2:
        img_bgr = np.repeat(img[..., None], 3, axis=2)
    else:
        img_bgr = np.ascontiguousarray(img[..., ::-1])

    det = yolo_model(img_bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]

    # OBB 优先
    if use_obb and getattr(det, "obb", None) is not None and len(det.obb) > 0:
        obb = det.obb
        if getattr(obb, "conf", None) is not None:
            confs = _det_to_numpy(obb.conf).reshape(-1)
        else:
            confs = np.ones((len(obb),), dtype=np.float32)
        polygons = []
        for j in np.argsort(-confs)[: max(1, int(max_dets))]:
            polygons.append(_det_to_numpy(obb.xyxyxyxy)[j].reshape(4, 2))
        return _rasterize_polygons(polygons, w, h)

    # 实例分割掩码
    masks = getattr(det, "masks", None)
    if masks is not None and getattr(masks, "data", None) is not None and len(masks) > 0:
        data = _det_to_numpy(masks.data)
        if getattr(masks, "conf", None) is not None:
            confs = _det_to_numpy(masks.conf).reshape(-1)
        else:
            confs = np.ones((data.shape[0],), dtype=np.float32)
        union = np.zeros((h, w), dtype=bool)
        for j in np.argsort(-confs)[: max(1, int(max_dets))]:
            single = np.asarray(data[j] > 0.5, dtype=np.uint8)
            if single.shape != (h, w):
                from PIL import Image

                single = (np.asarray(
                    Image.fromarray(single * 255).resize((w, h), Image.Resampling.NEAREST)
                ) > 0)
            union |= single.astype(bool)
        return union.astype(np.uint8)

    # HBB 退化
    if getattr(det, "boxes", None) is not None and len(det.boxes) > 0:
        boxes = det.boxes
        if getattr(boxes, "conf", None) is not None:
            order = np.argsort(-_det_to_numpy(boxes.conf).reshape(-1))
        else:
            order = np.arange(len(boxes))
        for j in order[: max(1, int(max_dets))]:
            x1, y1, x2, y2 = _det_to_numpy(boxes.xyxy)[j].tolist()
            xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
            xi2, yi2 = min(w, int(round(x2)) + 1), min(h, int(round(y2)) + 1)
            if xi2 > xi1 and yi2 > yi1:
                mask[yi1:yi2, xi1:xi2] = 1
        return mask

    return None


def yolo_ship_mask(
    image: Any,
    yolo_model: Any = None,
    yolo_weights: str | Path | None = None,
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.45,
    use_obb: bool = True,
    max_dets: int = 1,
) -> np.ndarray | None:
    """对光学图像跑 YOLO 舰船检测，返回 0/1 掩码（未检出为 None）。

    yolo_model 不传时按 resolve_yolo_weights() 的路径约定自动加载权重。
    """
    img = _to_array(image)
    model = yolo_model if yolo_model is not None else _load_yolo_model(resolve_yolo_weights(yolo_weights))
    return _yolo_ship_mask(
        img, model, imgsz=imgsz, conf=conf, iou=iou, use_obb=use_obb, max_dets=max_dets
    )


# =========================================================================
# 4. 真实感配置与海杂波（转写自 radar_realism.py，纯 numpy）
#
# 【理论速览】三档预设对应不同的"真实感层级"：
#   clean     确定性链路（无杂波/无噪声/无斑点），可逐位复现，用于对照；
#   realistic 相干斑 + Hamming 加窗 + 30dB 热噪声 + K 分布海杂波 + 40dB 显示；
#   rough-sea 更粗的杂波纹理(k=0.7)、更强杂波(0.18)与噪声(24dB)、45dB 显示。
# =========================================================================


@dataclass(frozen=True)
class RealismConfig:
    """真实感配置：预设 + 各项开关（字段与部署系统一一对应）。"""

    preset: str = "clean"
    seed: int = 7
    speckle: bool = False
    speckle_phase_strength: float = 1.0
    window: bool = False
    window_type: str = "none"
    kaiser_beta: float = 8.6
    noise_snr_db: float | None = None
    display_db: float | None = None
    sea_clutter: bool = False
    clutter_shape: float = 1.5
    clutter_scale: float = 0.08
    clutter_margin: float = 1.5
    clutter_max_points: int = 6000
    clutter_near_gain: float = 1.15
    clutter_far_gain: float = 0.70
    clutter_temporal_correlation: float = 0.95
    phase_noise_std_deg: float = 0.0
    phase_noise_correlation: float = 0.90
    timing_jitter_std_ns: float = 0.0
    motion_position_std_m: float = 0.0
    velocity_drift_mps: float = 0.0


def realism_preset(name: str = "realistic", seed: int = 7) -> RealismConfig:
    """三档真实感预设：clean（确定性）/ realistic（默认）/ rough-sea。"""
    name = str(name).lower().replace("_", "-")
    if name == "clean":
        return RealismConfig(preset="clean", seed=int(seed))
    if name == "realistic":
        return RealismConfig(
            preset="realistic",
            seed=int(seed),
            speckle=True,
            speckle_phase_strength=0.10,
            window=True,
            window_type="hamming",
            noise_snr_db=30.0,
            display_db=40.0,
            sea_clutter=True,
            clutter_shape=1.5,
            clutter_scale=0.08,
            clutter_margin=1.5,
        )
    if name in {"rough", "rough-sea"}:
        return RealismConfig(
            preset="rough-sea",
            seed=int(seed),
            speckle=True,
            speckle_phase_strength=0.25,
            window=True,
            window_type="hamming",
            noise_snr_db=24.0,
            display_db=45.0,
            sea_clutter=True,
            clutter_shape=0.7,
            clutter_scale=0.18,
            clutter_margin=2.0,
            clutter_max_points=8000,
        )
    raise ValueError("realism preset must be clean/realistic/rough-sea")


def realism_from_dict(raw: dict | None, preset: str = "realistic", seed: int = 7) -> RealismConfig:
    """在命名预设上合并经校验的自定义字段（未知字段报错）。"""
    preset_key = str(preset).lower().replace("_", "-")
    config = (
        replace(realism_preset("realistic", seed=seed), preset="custom")
        if preset_key == "custom" else realism_preset(preset_key, seed=seed)
    )
    if not raw:
        return config
    allowed = set(asdict(config))
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("unknown realism fields: " + ", ".join(unknown))
    values = {key: raw[key] for key in raw}
    config = replace(config, **values)
    if "window_type" in raw:
        config = replace(config, window=str(config.window_type).lower() != "none")
    elif "window" in raw:
        config = replace(config, window_type=("hamming" if bool(config.window) else "none"))
    if not 0.0 <= float(config.speckle_phase_strength) <= 1.0:
        raise ValueError("speckle_phase_strength must be between 0 and 1")
    if config.noise_snr_db is not None and not -20.0 <= float(config.noise_snr_db) <= 100.0:
        raise ValueError("noise_snr_db must be between -20 and 100")
    if not 0.0 <= float(config.clutter_temporal_correlation) < 1.0:
        raise ValueError("clutter_temporal_correlation must be in [0,1)")
    if not -0.999 <= float(config.phase_noise_correlation) <= 0.999:
        raise ValueError("phase_noise_correlation must be in [-0.999,0.999]")
    if str(config.window_type).lower() not in {"none", "hamming", "blackman", "kaiser"}:
        raise ValueError("window_type must be none/hamming/blackman/kaiser")
    nonnegative = (
        "clutter_shape", "clutter_scale", "clutter_margin", "clutter_near_gain",
        "clutter_far_gain", "phase_noise_std_deg", "timing_jitter_std_ns",
        "motion_position_std_m", "kaiser_beta",
    )
    for name in nonnegative:
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if int(config.clutter_max_points) < 0:
        raise ValueError("clutter_max_points must be non-negative")
    if config.display_db is not None and float(config.display_db) <= 0.0:
        raise ValueError("display_db must be positive")
    return config


def generate_sea_clutter(
    target_points: np.ndarray,
    parameters: np.ndarray,
    config: RealismConfig,
) -> tuple[np.ndarray, dict]:
    """生成 K 分布近似海杂波点云（Gamma 纹理 × Rayleigh 散斑 × 距离梯度）。

    【K 分布海杂波模型】低擦地角下海面杂波幅度常用复合 K 分布描述：
      幅度 = sqrt(Gamma纹理) × Rayleigh散斑
      · Gamma(shape=k, scale=1/k)（均值归一）建模海面涌浪等大尺度起伏
        （纹理分量，k 越小拖尾越重、尖峰越多）；
      · Rayleigh = sqrt((n1²+n2²)/2)（n 为标准正态）建模分辨单元内
        子散射体相干干涉（散斑分量）；
      · 再乘以近距→远距增益梯度，模拟俯角随距离变化引起的后向散射强弱。
    杂波参考电平取目标幅度的中位数 × clutter_scale，保证场景间相对亮度一致。
    杂波点与目标点一起进入 RDA 成像，因此会自然形成带噪底/杂波带的 SAR 图像。

    返回点沿用工程约定 [方位x, 地距y, z, 幅度]；复相位由回波仿真阶段生成。
    """
    P = np.asarray(target_points, dtype=np.float64)
    if not config.sea_clutter or P.size == 0:
        return np.empty((0, 4), dtype=np.float64), {"count": 0, "enabled": False}
    params = np.asarray(parameters, dtype=np.float64).ravel()
    if params.size < 11:
        raise ValueError("parameters must contain 11 radar values")
    c = C_LIGHT
    step_az = max(float(params[5]) / 2.0, 1.0)
    step_rg = max(c / (2.0 * float(params[10])), 0.75)
    lo = P[:, :2].min(axis=0)
    hi = P[:, :2].max(axis=0)
    size = np.maximum(hi - lo, [step_az * 4.0, step_rg * 4.0])
    centre = (lo + hi) / 2.0
    half = size * float(config.clutter_margin) / 2.0
    az = np.arange(centre[0] - half[0], centre[0] + half[0] + step_az, step_az)
    rg = np.arange(centre[1] - half[1], centre[1] + half[1] + step_rg, step_rg)
    aa, rr = np.meshgrid(az, rg, indexing="xy")
    coords = np.column_stack((aa.ravel(), rr.ravel()))
    if coords.shape[0] > config.clutter_max_points:
        stride = int(np.ceil(coords.shape[0] / config.clutter_max_points))
        coords = coords[::stride]

    rng = np.random.default_rng(int(config.seed) + 101)
    texture = rng.gamma(
        shape=max(float(config.clutter_shape), 1e-3),
        scale=1.0 / max(float(config.clutter_shape), 1e-3),
        size=coords.shape[0],
    )
    rayleigh = np.sqrt(
        (rng.standard_normal(coords.shape[0]) ** 2 + rng.standard_normal(coords.shape[0]) ** 2) / 2.0
    )
    target_positive = P[:, 3][P[:, 3] > 0]
    target_reference = float(np.median(target_positive)) if target_positive.size else 1.0
    rg_norm = (coords[:, 1] - coords[:, 1].min()) / max(float(np.ptp(coords[:, 1])), 1e-9)
    range_gradient = (
        float(config.clutter_near_gain)
        + (float(config.clutter_far_gain) - float(config.clutter_near_gain)) * rg_norm
    )
    amplitude = (
        target_reference
        * float(config.clutter_scale)
        * np.sqrt(texture)
        * rayleigh
        * range_gradient
    )
    clutter = np.column_stack((coords, np.zeros(coords.shape[0]), amplitude))
    metadata = {
        "enabled": True,
        "count": int(clutter.shape[0]),
        "shape": float(config.clutter_shape),
        "scale": float(config.clutter_scale),
        "near_gain": float(config.clutter_near_gain),
        "far_gain": float(config.clutter_far_gain),
        "temporal_correlation": float(config.clutter_temporal_correlation),
        "amplitude_mean": float(amplitude.mean()) if amplitude.size else 0.0,
        "amplitude_std": float(amplitude.std()) if amplitude.size else 0.0,
    }
    return clutter, metadata


# =========================================================================
# 5. RDA 成像核心（转写自 rda_pipeline.py，公式与 MATLAB 版一致）
#
# 【SAR 成像基本原理速览——帮助理解下面每一步在做什么】
#   · 几何：正侧视条带 SAR。平台沿 x 轴匀速 vx 飞行、高度 Hc，天线以侧视角
#     thetaSL、斜视角 thetaSQ 照射地面。场景中某散射点在慢时间 η 上的斜距
#       R(η) = sqrt(Rc² + (vx·η)² - 2·Rc·vx·η·sin(thetaSQ))
#     是一条双曲线——它是方位向聚焦的全部信息来源，也是距离徙动的根源。
#   · 距离向（快时间 t）：发射 LFM 线性调频脉冲 chirp = exp(jπ·kr·t²)，
#     kr = Br/T。匹配滤波（脉冲压缩）后分辨率 ρr = c/(2·Br)，与脉宽无关。
#   · 方位向（慢时间 η）：相位历史 exp(-j·4π·R(η)/λ) 的瞬时频率是多普勒
#     fD = -(2/λ)·dR/dη ≈ fDC + ka·η（线性调频），把它再做一次匹配滤波
#     即得方位聚焦，分辨率 ρa = D_az/2——这就是"合成孔径"：孔径越长
#     （LSAR = Rc·θaz），多普勒带宽越宽，分辨率越高。
#   · 距离-方位耦合：R(η) 的变化量以光速尺度换算成快时间，导致同一个点
#     的回波随 η 在距离向"走动+弯曲"（距离徙动 RCM）；大斜视角下距离向
#     调频率还会随多普勒改变（二次距离压缩 SRC）。因此必须按
#       距离压缩 → 走动校正(RWC) → 二维频域(SRC+Hr)(RGC) → RCMC(RMC)
#       → 方位压缩(Im)
#     的次序处理——正是本节 rda_numpy/rda_numpy_enhanced 的实现顺序。
# =========================================================================


def _prepare_input_layout(P: np.ndarray, input_layout: str) -> np.ndarray:
    """把调用方点云统一为内部 [方位x, 地距y, z, 幅度] 顺序。"""
    if input_layout == "xy":
        return np.asarray(P, dtype=np.float64)
    if input_layout == "yx":
        P2 = np.asarray(P, dtype=np.float64).copy()
        P2[:, [0, 1]] = P2[:, [1, 0]]
        return P2
    raise ValueError("input_layout must be 'xy' or 'yx'.")


def _calc_geometry(P: np.ndarray, parameters: np.ndarray):
    """由点云与 11 维参数推导公共雷达几何常量。"""
    c = C_LIGHT
    Hc, thetaSQ, thetaSL, vx, D_rg, D_az, PRF, T, f0, Br, fs = map(float, parameters[:11])

    thetaSQ = np.deg2rad(thetaSQ)
    thetaSL = np.deg2rad(thetaSL)
    vx = float(vx)
    if vx == 0.0:
        raise ValueError("vx cannot be 0.")
    if fs <= 0 or PRF <= 0:
        raise ValueError("PRF and fs must be positive.")

    kr = Br / T                 # 调频率：chirp exp(jπ·kr·t²) 的扫频速率 = Br/T
    PRI = 1.0 / PRF             # 脉冲重复间隔（慢时间采样间隔）
    Rc = Hc / (np.cos(thetaSQ) * np.cos(thetaSL))   # 场景中心斜距（两个角度的投影链）
    lam = c / f0                # 波长
    # 矩形口径天线的瑞利波束宽度：0.886 = sinc 主瓣半功率近似因子（λ/D 弧度）
    theta_az = 0.886 * lam / D_az   # 方位向波束宽度（决定合成孔径长度与 ρa）
    theta_rg = 0.886 * lam / D_rg   # 距离向波束宽度（决定照射地带宽度，用于波束内判据）
    rg_near = np.sqrt((Hc / np.cos(thetaSL - theta_rg / 2) / np.cos(thetaSQ)) ** 2 - Hc**2)
    rg_far = np.sqrt((Hc / np.cos(thetaSL + theta_rg / 2) / np.cos(thetaSQ)) ** 2 - Hc**2)
    LSAR = Rc * theta_az / np.cos(thetaSQ)  # 合成孔径长度：波束照射的地面弧长投影

    # 雷达轨迹（沿 x 匀速，高度 Hc）
    # 方位脉冲数 Na 的确定：场景方位跨度两端各再外扩 0.6·LSAR，
    # 保证场景内每个散射点在合成孔径全程都处于波束内（不丢相位历史）。
    Pm = P.T
    num = Pm.shape[1]
    Na1 = int(np.round((np.min(Pm[0, :]) - 0.6 * LSAR) / (vx * PRI)))
    Na2 = int(np.round((np.max(Pm[0, :]) + 0.6 * LSAR) / (vx * PRI)))
    if Na1 % 2 != 0:
        Na1 += 1
    if Na2 % 2 != 0:
        Na2 += 1
    Na = Na2 - Na1

    ta = np.arange(Na1, Na2, dtype=np.float64) * PRI
    Q0 = np.array([[0.0], [0.0], [Hc]], dtype=np.float64)
    Q = Q0 + np.array([[vx], [0.0], [0.0]], dtype=np.float64) * ta + 0.0 * (ta**2)
    Qprojection = np.vstack(
        (
            Q[0, :] + Q[2, :] / np.cos(thetaSL) * np.tan(thetaSQ),
            Q[1, :] + Q[2, :] * np.tan(thetaSL),
            np.zeros(Na, dtype=np.float64),
        )
    )

    N_LFM = int(np.round(T * fs))
    if N_LFM % 2 != 0:
        N_LFM += 1
    tLFM = (np.arange(-N_LFM / 2, N_LFM / 2, dtype=np.float64) / fs)
    return (
        c, Hc, thetaSQ, thetaSL, vx, fs, D_rg, D_az, PRF, T, f0, Br,
        kr, PRI, Rc, lam, theta_az, theta_rg, rg_near, rg_far, LSAR,
        Na, Na1, ta, Q, Qprojection, N_LFM, tLFM, Pm, num, vx * np.cos(thetaSQ),
    )


def rda_numpy(
    P: np.ndarray,
    parameters: np.ndarray,
    input_layout: str = "xy",
    apply_rcmc_to_image: bool = False,
):
    """NumPy RDA 成像（确定性，无任何随机项）。

    返回 (echo, RWC, RGC, RMC, Im)。
    apply_rcmc_to_image=True 时方位压缩使用经 RCMC 校正的谱（与 MATLAB 一致），
    亦为部署系统统一仿真 API 的默认选择。
    """
    P = _prepare_input_layout(P, input_layout)

    (
        c, Hc, thetaSQ, thetaSL, vx, fs, _D_rg, D_az, _PRF, _T, f0, Br,
        kr, PRI, Rc, lam, theta_az, theta_rg, _rg_near, _rg_far, LSAR,
        Na, Na1, ta, Q, Qprojection, N_LFM, tLFM, Pm, num, vxa,
    ) = _calc_geometry(P, parameters)

    # 内存安全：逐脉冲统计全局最近距离与波束内可见最远距离，推算 Nr
    global_r_min = np.inf
    global_r_max = 0.0
    visible_r_max = 0.0
    visible_pairs = 0
    amp = Pm[3, :]
    chirp = np.exp(1j * np.pi * kr * (tLFM**2))
    cos_thetax = np.cos(theta_az / 2)
    cos_thetay = np.cos(theta_rg / 2)
    for i in range(Na):
        R_i = np.linalg.norm(Pm[:3, :] - Q[:, i][:, None], axis=0)
        global_r_min = min(global_r_min, float(np.min(R_i)))
        global_r_max = max(global_r_max, float(np.max(R_i)))

        Q_i = Q[:, i]
        qp_i = Qprojection[:, i]
        QQ = qp_i - Q_i
        QQ_norm = np.linalg.norm(QQ)
        if QQ_norm <= 0:
            continue

        QPx = np.vstack((Pm[0, :], np.full(num, qp_i[1]), Pm[2, :])) - Q_i[:, None]
        QPy = np.vstack((np.full(num, qp_i[0]), Pm[1, :], Pm[2, :])) - Q_i[:, None]
        norm_QPx = np.linalg.norm(QPx, axis=0)
        norm_QPy = np.linalg.norm(QPy, axis=0)

        cosx = np.dot(QQ, QPx) / (QQ_norm * norm_QPx)
        cosy = np.dot(QQ, QPy) / (QQ_norm * norm_QPy)
        valid = (norm_QPx > 0) & (norm_QPy > 0)
        if not np.any(valid):
            continue
        cosx = np.clip(cosx[valid], -1.0, 1.0)
        cosy = np.clip(cosy[valid], -1.0, 1.0)
        mask_i = (cosx >= cos_thetax) & (cosy >= cos_thetay)

        if not np.any(mask_i):
            continue
        visible_pairs += int(mask_i.sum())
        valid_idx = np.where(valid)[0][mask_i]
        if valid_idx.size > 0:
            visible_r_max = max(visible_r_max, float(np.max(R_i[valid_idx])))

    if visible_pairs == 0:
        raise RuntimeError(
            "No scatter point falls inside the antenna beam: "
            f"scene must lie near the beam-centre ground point "
            f"(y ~= Hc*tan(thetaSL) = {Hc * np.tan(thetaSL):.1f} m, "
            f"x ~= Hc/cos(thetaSL)*tan(thetaSQ) = "
            f"{Hc / np.cos(thetaSL) * np.tan(thetaSQ):.1f} m). "
            "Use place_scene_at_beam(), or widen D_rg/D_az or reduce thetaSL."
        )
    if not np.isfinite(global_r_min):
        raise ValueError("Failed to compute valid range distances.")
    tRmin = 2 * global_r_min / c
    if visible_r_max > 0.0:
        tRmax = 2 * visible_r_max / c
    else:
        tRmax = 2 * global_r_max / c
    Nr = int(np.round((tRmax - tRmin) * fs)) + N_LFM
    if Nr % 2 != 0:
        Nr += 1

    # ---- 回波正演（点散射模型 + "停-走"假设）----
    # 第 i 个脉冲接收到的第 m 个散射点回波：
    #   s_i(t) = σ_m · exp(-j·4π·R_im/λ) · exp(jπ·kr·(t - 2R_im/c)²)
    # 其中 exp(-j4πR/λ) 是随斜距变化的相位历史（方位多普勒的来源），
    # exp(jπ·kr·t²) 是 LFM chirp 包络（距离脉冲压缩的对象）。
    # 实现方式：把每个散射点的 N_LFM 点 chirp 片段按快时间延迟
    # delay = rint((2R/c - tRmin)·fs) 累加到 echo[i] 的对应列上。
    echo = np.zeros((Na, Nr), dtype=np.complex128)
    for i in range(Na):
        R_i = np.linalg.norm(Pm[:3, :] - Q[:, i][:, None], axis=0)
        Q_i = Q[:, i]
        qp_i = Qprojection[:, i]
        QQ = qp_i - Q_i
        QQ_norm = np.linalg.norm(QQ)
        if QQ_norm <= 0:
            continue

        QPx = np.vstack((Pm[0, :], np.full(num, qp_i[1]), Pm[2, :])) - Q_i[:, None]
        QPy = np.vstack((np.full(num, qp_i[0]), Pm[1, :], Pm[2, :])) - Q_i[:, None]
        norm_QPx = np.linalg.norm(QPx, axis=0)
        norm_QPy = np.linalg.norm(QPy, axis=0)

        cosx = np.dot(QQ, QPx) / (QQ_norm * norm_QPx)
        cosy = np.dot(QQ, QPy) / (QQ_norm * norm_QPy)
        valid = (norm_QPx > 0) & (norm_QPy > 0)
        if not np.any(valid):
            continue

        cosx = np.clip(cosx[valid], -1.0, 1.0)
        cosy = np.clip(cosy[valid], -1.0, 1.0)
        vis_idx = np.where(valid)[0][(cosx >= cos_thetax) & (cosy >= cos_thetay)]
        if vis_idx.size == 0:
            continue

        delays = np.rint((2 * R_i[vis_idx] / c - tRmin) * fs).astype(np.int64)
        start = delays
        end = delays + N_LFM
        in_range = (start >= 0) & (end <= Nr)
        if not np.any(in_range):
            continue
        vis_idx = vis_idx[in_range]
        delays = delays[in_range]

        pulse = amp[vis_idx][:, None] * np.exp(-1j * 4 * np.pi * R_i[vis_idx] / lam)[:, None]
        col = delays[:, None] + np.arange(N_LFM, dtype=np.int64)[None, :]
        pulse = pulse * chirp[None, :]
        np.add.at(echo[i], col.ravel(), pulse.ravel())

    # ---- 步骤1：距离压缩 + 距离走动(一次RCM)校正 → RWC ----
    # fDC = 2·vx·sin(thetaSQ)/λ 为多普勒中心（斜视角引起）。
    # Hsl = exp(-j·4π/c · (fDC/2)·ta·(f0+fr))：斜视角下回波包络随慢时间
    # 线性平移（距离走动 R(η)≈Rc - vx·sinθSQ·η 的一阶项），在二维频域
    # 乘 Hsl 等价于把每个脉冲的距离包络插值搬回对齐位置。
    # RWC = IFFT_r[ FFT_r(echo) · Hsl ]
    fr = (np.arange(-Nr / 2, Nr / 2, dtype=np.float64) * fs / Nr)
    fr = np.fft.fftshift(fr)[None, :]
    fr = np.repeat(fr, Na, axis=0)
    ta2 = (np.arange(-Na / 2, Na / 2, dtype=np.float64) * PRI)[:, None]
    fDC = 2 * vx * np.sin(thetaSQ) / lam
    Hsl = np.exp(-1j * 4 * np.pi / c * lam * fDC / 2 * ta2 * (f0 + fr))
    RWC = np.fft.ifft(np.fft.fft(echo, axis=1) * Hsl, axis=1)

    # ---- 步骤2：二维频域 → 距离压缩精化 + 二次距离压缩 → RGC ----
    # fa：多普勒（方位频率）轴；Rr：各快时间采样对应的斜距。
    # RCMF = sqrt(1 - λ²·fa²/(4·vxa²))：距离徙动因子，即 sqrt(1-(fa/Bd)²)，
    #   描述多普勒 fa 处斜距相对中心斜距的投影收缩（双曲线的一阶归一化）。
    # Ksrc = 2·vxa²·f0³·RCMF³ / (c·R0·fa²)：二次距离压缩(SRC)调频率。
    #   大带宽×大斜视角时，距离脉冲响应随 fa 散开（距离-方位耦合的二次项），
    #   需在二维频域补偿；fa=0 处奇异（Ksrc→∞，Hsrc→1 即不补偿）。
    # Hr   = exp(+jπ·fr²/kr)    距离向匹配滤波（脉冲压缩的频域实现）
    # Hsrc = exp(-jπ·fr²/Ksrc)  二次距离压缩滤波
    # spec = FFT2(RWC)·Hsrc·Hr；RGC = IFFT2(spec)
    fa = np.fft.fftshift(fa)
    fa = np.repeat(fa, Nr, axis=1)
    fr = np.fft.fftshift((np.arange(-Nr / 2, Nr / 2, dtype=np.float64) * fs / Nr)[None, :])
    fr = np.repeat(fr, Na, axis=0)
    Rr = ((np.arange(-Nr / 2, Nr / 2, dtype=np.float64) / fs * c / 2 + Rc)[None, :])
    R0 = Rr.copy()
    Rr = np.repeat(Rr, Na, axis=0)

    Rref = Rc
    RCMF = np.sqrt(np.maximum(1 - lam**2 * fa**2 / (4 * vxa**2), 0.0))
    Ksrc = np.zeros_like(fa, dtype=np.float64)
    fa_nonzero = fa != 0
    Ksrc[fa_nonzero] = 2 * vxa**2 * f0**3 * RCMF[fa_nonzero] ** 3 / (c * Rref * fa[fa_nonzero] ** 2)
    Ksrc[~fa_nonzero] = np.inf

    Hr = np.exp(1j * np.pi * fr**2 / kr)
    Hsrc = np.exp(-1j * np.pi * fr**2 / Ksrc)
    spec = np.fft.fft2(RWC) * Hsrc * Hr
    RGC = np.fft.ifft2(spec)

    # ---- 步骤3：距离徙动校正（RCMC）→ RMC ----
    # 多普勒 fa 处散射点的斜距 R(fa) = Rc/RCMF（双曲线在频域的表达）。
    # Hrcmc = exp(j·4π·fr·(Rc/RCMF - Rc)/c)：在距离频域给不同 fa 的分量
    # 一个随 fr 线性的相位，等价于把"弯曲"的距离包络沿快时间平移拉直。
    # RMC = IFFT2(spec·Hrcmc)
    Hrcmc = np.exp(1j * 4 * np.pi * fr * (Rref / RCMF - Rc) / c)
    spec_rcmc = spec * Hrcmc
    RMC = np.fft.ifft2(spec_rcmc)

    # ---- 步骤4：方位压缩 → Im ----
    # Ha = exp(+j·4π·R0·RCMF/λ)：方位匹配滤波 = 多普勒调频斜率的共轭。
    #   相位历史 exp(-j4πR(η)/λ) 展开后是调频率 ka = 2v²/(λ·R0) 的线性
    #   调频信号，共轭相乘再 IFFT 即完成方位聚焦，分辨率 ρa = D_az/2。
    # 默认聚焦经 RCMC 校正后的谱（apply_rcmc_to_image=True，与 MATLAB 一致）。
    # 注意 Im 的行=方位（对应光学图的列），列=斜距（对应光学图的行），
    # 即 Im 是光学布局的转置——显示层会再转回来。
    Ha = np.exp(1j * 4 * np.pi / lam * R0 * RCMF)
    focus_spec = spec_rcmc if apply_rcmc_to_image else spec
    IFFT_r = np.fft.ifft(focus_spec, axis=1) * Ha
    Im = np.fft.ifft(IFFT_r, axis=0)

    return echo, RWC, RGC, RMC, Im


def rda_numpy_enhanced(
    P: np.ndarray,
    parameters: np.ndarray,
    input_layout: str = "xy",
    speckle: bool = False,
    window: bool = False,
    noise_snr_db: float | None = None,
    seed: int = 7,
    apply_rcmc_to_image: bool = False,
    window_type: str | None = None,
    kaiser_beta: float = 8.6,
    speckle_phase_strength: float = 1.0,
    phase_noise_std_deg: float = 0.0,
    phase_noise_correlation: float = 0.9,
    timing_jitter_std_ns: float = 0.0,
    motion_position_std_m: float = 0.0,
    velocity_drift_mps: float = 0.0,
    correct_complex_noise_power: bool = False,
    correct_window_frequency_order: bool = False,
    return_metadata: bool = False,
):
    """带真实感增强的 RDA：斑点相位 / 加窗 / 热噪声 / 慢时间误差 / 定时抖动。

    全部增强关闭时结果与 rda_numpy 一致（但不再返回 RGC/RMC 中间体）。
    返回 (echo, RWC, Im[, error_meta])。
    """
    P = _prepare_input_layout(P, input_layout)
    (
        c, Hc, thetaSQ, thetaSL, vx, fs, _D_rg, D_az, _PRF, _T, f0, Br,
        kr, PRI, Rc, lam, theta_az, theta_rg, _rg_near, _rg_far, LSAR,
        Na, _Na1, ta, Q, Qprojection, N_LFM, tLFM, Pm, num, vxa,
    ) = _calc_geometry(P, parameters)

    amp = Pm[3, :]
    chirp = np.exp(1j * np.pi * kr * (tLFM**2))
    cos_thetax = np.cos(theta_az / 2)
    cos_thetay = np.cos(theta_rg / 2)
    rng = np.random.default_rng(seed)
    phase_strength = float(np.clip(speckle_phase_strength, 0.0, 1.0))
    if speckle:
        # 相干斑（speckle）：真实 SAR 是相干成像，一个分辨单元内多个子散射体
        # 相干叠加，等效于给每个散射点一个 [0,2π) 均匀随机初相 phase0；
        # 聚焦后幅度呈 Rayleigh 起伏的颗粒状纹理——单视 SAR 的标志性外观。
        # 部分强度(<1)时初相以零为中心收缩为有界扰动，弱化斑纹对比。
        phase_draw = rng.uniform(0.0, 2.0 * np.pi, num)
        phase0 = phase_draw if phase_strength == 1.0 else (phase_draw - np.pi) * phase_strength
    else:
        phase0 = np.zeros(num)

    r_all_min = np.inf
    visible_r_max = 0.0
    for i in range(Na):
        R_i = np.linalg.norm(Pm[:3, :] - Q[:, i][:, None], axis=0)
        r_all_min = min(r_all_min, float(R_i.min()))

        Q_i = Q[:, i]
        qp_i = Qprojection[:, i]
        QQ = qp_i - Q_i
        QQ_norm = np.linalg.norm(QQ)
        if QQ_norm <= 0:
            continue
        QPx = np.vstack((Pm[0, :], np.full(num, qp_i[1]), Pm[2, :])) - Q_i[:, None]
        QPy = np.vstack((np.full(num, qp_i[0]), Pm[1, :], Pm[2, :])) - Q_i[:, None]
        norm_QPx = np.linalg.norm(QPx, axis=0)
        norm_QPy = np.linalg.norm(QPy, axis=0)
        cosx = np.clip(np.dot(QQ, QPx) / (QQ_norm * norm_QPx), -1.0, 1.0)
        cosy = np.clip(np.dot(QQ, QPy) / (QQ_norm * norm_QPy), -1.0, 1.0)
        valid = (norm_QPx > 0) & (norm_QPy > 0)
        if not np.any(valid):
            continue
        mask_i = (cosx[valid] >= cos_thetax) & (cosy[valid] >= cos_thetay)
        if not np.any(mask_i):
            continue
        visible_r_max = max(visible_r_max, float(np.max(R_i[valid][mask_i])))
    if not np.isfinite(r_all_min):
        raise ValueError("Failed to compute valid range distances.")
    tRmin = 2 * r_all_min / c
    tRmax = (
        2 * visible_r_max / c
        if visible_r_max > 0.0
        else 2 * float(np.max(np.linalg.norm(Pm[:3, :] - Q[:, -1][:, None], axis=0))) / c
    )
    Nr = int(np.round((tRmax - tRmin) * fs)) + N_LFM
    if Nr % 2 != 0:
        Nr += 1

    echo = np.zeros((Na, Nr), dtype=np.complex128)
    for i in range(Na):
        R_i = np.linalg.norm(Pm[:3, :] - Q[:, i][:, None], axis=0)
        Q_i = Q[:, i]
        qp_i = Qprojection[:, i]
        QQ = qp_i - Q_i
        QQ_norm = np.linalg.norm(QQ)
        if QQ_norm <= 0:
            continue
        QPx = np.vstack((Pm[0, :], np.full(num, qp_i[1]), Pm[2, :])) - Q_i[:, None]
        QPy = np.vstack((np.full(num, qp_i[0]), Pm[1, :], Pm[2, :])) - Q_i[:, None]
        norm_QPx = np.linalg.norm(QPx, axis=0)
        norm_QPy = np.linalg.norm(QPy, axis=0)
        cosx = np.clip(np.dot(QQ, QPx) / (QQ_norm * norm_QPx), -1.0, 1.0)
        cosy = np.clip(np.dot(QQ, QPy) / (QQ_norm * norm_QPy), -1.0, 1.0)
        valid = (norm_QPx > 0) & (norm_QPy > 0)
        if not np.any(valid):
            continue
        vis_idx = np.where(valid)[0][(cosx[valid] >= cos_thetax) & (cosy[valid] >= cos_thetay)]
        if vis_idx.size == 0:
            continue
        delays = np.rint((2 * R_i[vis_idx] / c - tRmin) * fs).astype(np.int64)
        in_range = (delays >= 0) & (delays + N_LFM <= Nr)
        vis_idx, delays = vis_idx[in_range], delays[in_range]
        if vis_idx.size == 0:
            continue
        pulse = (amp[vis_idx] * np.exp(
            -1j * 4 * np.pi * R_i[vis_idx] / lam + 1j * phase0[vis_idx]))[:, None]
        col = delays[:, None] + np.arange(N_LFM, dtype=np.int64)[None, :]
        pulse = pulse * chirp[None, :]
        np.add.at(echo[i], col.ravel(), pulse.ravel())

    error_meta: dict[str, float | str | bool] = {
        "speckle_phase_strength": phase_strength if speckle else 0.0,
        "phase_noise_rms_deg": 0.0,
        "timing_jitter_rms_ns": 0.0,
        "motion_error_rms_m": 0.0,
        "actual_snr_db": float("inf"),
    }
    # 慢时间相位误差：相位噪声（AR(1)）+ 位置误差 / 速度漂移
    slow_errors = np.zeros(Na, dtype=np.float64)
    phase_std = max(0.0, float(phase_noise_std_deg))
    if phase_std > 0.0:
        rho = float(np.clip(phase_noise_correlation, -0.999, 0.999))
        innovation = rng.standard_normal(Na) * np.deg2rad(phase_std) * np.sqrt(1.0 - rho * rho)
        phase_error = np.zeros(Na, dtype=np.float64)
        phase_error[0] = rng.normal(0.0, np.deg2rad(phase_std))
        for idx in range(1, Na):
            phase_error[idx] = rho * phase_error[idx - 1] + innovation[idx]
        slow_errors += phase_error
        error_meta["phase_noise_rms_deg"] = float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
    pos_std = max(0.0, float(motion_position_std_m))
    drift = float(velocity_drift_mps)
    if pos_std > 0.0 or drift != 0.0:
        pos_error = rng.normal(0.0, pos_std, Na) + drift * (ta - float(np.mean(ta)))
        slow_errors += -4.0 * np.pi * pos_error / lam
        error_meta["motion_error_rms_m"] = float(np.sqrt(np.mean(pos_error**2)))
    if np.any(slow_errors):
        echo *= np.exp(1j * slow_errors)[:, None]

    # 定时抖动：频域相位斜坡
    jitter_std = max(0.0, float(timing_jitter_std_ns))
    if jitter_std > 0.0 and np.any(echo):
        jitter_sec = rng.normal(0.0, jitter_std * 1e-9, Na)
        frequency = np.fft.fftfreq(Nr, d=1.0 / fs)
        echo = np.fft.ifft(
            np.fft.fft(echo, axis=1)
            * np.exp(-1j * 2.0 * np.pi * jitter_sec[:, None] * frequency[None, :]),
            axis=1,
        )
        error_meta["timing_jitter_rms_ns"] = float(np.sqrt(np.mean(jitter_sec**2)) * 1e9)

    # 复高斯热噪声：n ~ CN(0, σ²)，实部虚部独立同分布；
    # σ 由给定"压缩前 SNR"相对回波平均功率标定 → 成像后形成真实的噪底。
    if noise_snr_db is not None and np.any(echo):
        signal_power = float(np.mean(np.abs(echo) ** 2))
        n_std = np.sqrt(signal_power) * 10 ** (-float(noise_snr_db) / 20)
        if correct_complex_noise_power:
            n_std /= np.sqrt(2.0)
        noise = n_std * (
            rng.standard_normal(echo.shape) + 1j * rng.standard_normal(echo.shape)
        )
        echo = echo + noise
        noise_power = float(np.mean(np.abs(noise) ** 2))
        error_meta["actual_snr_db"] = float(10.0 * np.log10(max(signal_power, 1e-30) / max(noise_power, 1e-30)))

    # 距离压缩 + 走动校正 → RWC
    fr = np.repeat(np.fft.fftshift(np.arange(-Nr / 2, Nr / 2) * fs / Nr)[None, :], Na, axis=0)
    ta2 = (np.arange(-Na / 2, Na / 2) * PRI)[:, None]
    fDC = 2 * vx * np.sin(thetaSQ) / lam
    Hsl = np.exp(-1j * 4 * np.pi / c * lam * fDC / 2 * ta2 * (f0 + fr))
    RWC = np.fft.ifft(np.fft.fft(echo, axis=1) * Hsl, axis=1)

    fa = np.fft.fftshift((np.arange(-Na / 2, Na / 2) / (Na * PRI))[:, None])
    fa = np.repeat(fa, Nr, axis=1)
    fr2 = np.repeat(np.fft.fftshift(np.arange(-Nr / 2, Nr / 2) * fs / Nr)[None, :], Na, axis=0)
    R0 = np.repeat((np.arange(-Nr / 2, Nr / 2) / fs * c / 2 + Rc)[None, :], Na, axis=0)

    RCMF = np.sqrt(np.maximum(1 - lam**2 * fa**2 / (4 * vxa**2), 0.0))
    Ksrc = np.full_like(fa, np.inf)
    nz = fa != 0
    Ksrc[nz] = 2 * vxa**2 * f0**3 * RCMF[nz] ** 3 / (c * Rc * fa[nz] ** 2)

    Hr = np.exp(1j * np.pi * fr2**2 / kr)
    Hsrc = np.exp(-1j * np.pi * fr2**2 / Ksrc)
    spec = np.fft.fft2(RWC) * Hsrc * Hr
    # 距离/方位加窗：脉冲压缩用矩形谱时旁瓣约 -13.2 dB（sinc 旁瓣），
    # 点目标会呈现"十字形"旁瓣、分布式目标出现周期条纹。乘 Hamming/
    # Blackman/Kaiser 窗以旁瓣抑制换取主瓣展宽（约 1.3~1.6 倍）。
    selected_window = str(window_type or ("hamming" if window else "none")).lower()
    if selected_window not in {"none", "hamming", "blackman", "kaiser"}:
        raise ValueError("window_type must be none/hamming/blackman/kaiser")
    if selected_window == "hamming":
        win_r, win_a = np.hamming(Nr), np.hamming(Na)
    elif selected_window == "blackman":
        win_r, win_a = np.blackman(Nr), np.blackman(Na)
    elif selected_window == "kaiser":
        win_r, win_a = np.kaiser(Nr, float(kaiser_beta)), np.kaiser(Na, float(kaiser_beta))
    else:
        win_r, win_a = None, None
    if win_r is not None and correct_window_frequency_order:
        # fft2 的 DC 在下标 0，而对称窗峰值在中心，需 ifftshift 对齐后乘谱
        win_r = np.fft.ifftshift(win_r)
        win_a = np.fft.ifftshift(win_a)
    if win_r is not None:
        spec = spec * win_r[None, :]
    if apply_rcmc_to_image:
        Hrcmc = np.exp(1j * 4 * np.pi * fr2 * (Rc / RCMF - Rc) / c)
        focus_spec = spec * Hrcmc
    else:
        focus_spec = spec
    IFFT_r = np.fft.ifft(focus_spec, axis=1)
    if win_a is not None:
        IFFT_r = IFFT_r * win_a[:, None]
    Ha = np.exp(1j * 4 * np.pi / lam * R0 * RCMF)
    Im = np.fft.ifft(IFFT_r * Ha, axis=0)

    error_meta["window_type"] = selected_window
    if return_metadata:
        return echo, RWC, Im, error_meta
    return echo, RWC, Im


# =========================================================================
# 6. 归一化与显示辅助
#
# 【理论速览】SAR 幅值动态范围极大（强散射点可比噪底高 60dB 以上）：
#   线性 min-max 归一化会让最亮的单个散射点"压黑"整幅图；
#   dB 显示（20·log10(|Im|/max)）配上有限动态范围（默认 40dB）才符合
#   人眼观察真实 SAR 图像的习惯——这也是部署系统的默认显示方式。
# =========================================================================


def normalize_im(im: np.ndarray) -> np.ndarray:
    """复 SAR 图像幅值线性 min-max 归一化到 [0,1]。"""
    mag = np.abs(im)
    mn = float(np.min(mag))
    mx = float(np.max(mag))
    if mx <= mn:
        return np.zeros_like(mag, dtype=np.float64)
    return (mag - mn) / (mx - mn)


def normalize_im_db(im: np.ndarray, dynamic_range_db: float = 40.0) -> np.ndarray:
    """|Im| 按 dB 归一化到 [0,1]，下限 -dynamic_range_db（真实 SAR 显示方式）。"""
    mag = np.abs(im)
    peak = float(np.max(mag))
    if peak <= 0:
        return np.zeros_like(mag, dtype=np.float64)
    db = 20.0 * np.log10(mag / peak + 1e-12)
    return np.clip((db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0)


def remove_vertical_stripes(img01: np.ndarray, strength: float = 0.85, window_size: int = 31) -> np.ndarray:
    """显示层竖条纹抑制（列中值背景估计扣除；只作用于显示图，不改复数 Im）。"""
    img = np.asarray(img01, dtype=np.float64)
    strength = min(max(strength, 0.0), 1.0)
    if strength <= 0 or img.ndim != 2:
        return img
    w = max(3, int(window_size))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    profile = np.median(img, axis=0)
    profile = np.convolve(profile, kernel, mode="same")
    clean = img - strength * profile[None, :]
    return np.clip(clean, 0.0, None)


def _png_bytes(image01: np.ndarray) -> bytes:
    """[0,1] 灰度图编码为 PNG 字节流（与部署系统相同路径）。"""
    from io import BytesIO

    from PIL import Image

    arr = (np.clip(np.asarray(image01), 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# =========================================================================
# 7. SAR 成像封装与顶层接口
# =========================================================================


def simulate_sar(
    P: np.ndarray,
    parameters: np.ndarray,
    realism: RealismConfig | None = None,
) -> dict[str, Any]:
    """散射点云 → SAR 复图像 + 归一化显示（与部署系统 simulate_sar 一致）。

    开启海杂波时先生成杂波点并与目标点合并；speckle/window/noise/误差注入
    任一开启走 rda_numpy_enhanced，否则走确定性 rda_numpy。
    """
    config = realism or realism_preset("clean")
    target = np.asarray(P, dtype=np.float64)
    clutter, clutter_meta = generate_sea_clutter(target, parameters, config)
    all_points = np.vstack((target, clutter)) if clutter.size else target
    enhanced = any((
        config.speckle,
        config.window,
        config.noise_snr_db is not None,
        float(config.phase_noise_std_deg) > 0.0,
        float(config.timing_jitter_std_ns) > 0.0,
        float(config.motion_position_std_m) > 0.0,
        float(config.velocity_drift_mps) != 0.0,
    ))
    error_meta: dict[str, Any] = {
        "window_type": "none", "actual_snr_db": float("inf"),
        "phase_noise_rms_deg": 0.0, "timing_jitter_rms_ns": 0.0,
        "motion_error_rms_m": 0.0,
    }
    if enhanced:
        echo, RWC, image, error_meta = rda_numpy_enhanced(
            all_points,
            parameters,
            input_layout="xy",
            speckle=config.speckle,
            window=config.window,
            noise_snr_db=config.noise_snr_db,
            seed=config.seed,
            apply_rcmc_to_image=True,
            window_type=config.window_type,
            kaiser_beta=config.kaiser_beta,
            speckle_phase_strength=config.speckle_phase_strength,
            phase_noise_std_deg=config.phase_noise_std_deg,
            phase_noise_correlation=config.phase_noise_correlation,
            timing_jitter_std_ns=config.timing_jitter_std_ns,
            motion_position_std_m=config.motion_position_std_m,
            velocity_drift_mps=config.velocity_drift_mps,
            correct_complex_noise_power=True,
            correct_window_frequency_order=True,
            return_metadata=True,
        )
        intermediates = {"echo": echo, "RWC": RWC}
    else:
        echo, RWC, RGC, RMC, image = rda_numpy(
            all_points, parameters, input_layout="xy", apply_rcmc_to_image=True
        )
        intermediates = {"echo": echo, "RWC": RWC, "RGC": RGC, "RMC": RMC}
    linear = normalize_im(image)
    display = normalize_im_db(image, config.display_db) if config.display_db else linear
    return {
        "image": image,
        "linear": linear,
        "display": display,
        "points": all_points,
        "target_points": target,
        "clutter_points": clutter,
        "clutter_meta": clutter_meta,
        "error_meta": error_meta,
        **intermediates,
    }


def _jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组递归转成可 JSON 序列化的 Python 对象。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def optical_to_sar(
    image: Any,
    *,
    mask: Any = None,
    mask_polarity: str = "bright",
    parameters: np.ndarray | list[float] | None = None,
    segmentation: str = "auto",
    threshold: float | None = None,
    polarity: str = "bright",
    largest_component: bool = True,
    roi: tuple[int, int, int, int] | None = None,
    step: int | None = None,
    pixel_size_x: float = 1.0,
    pixel_size_y: float = 1.0,
    realism: str | RealismConfig = "realistic",
    realism_options: dict[str, Any] | None = None,
    seed: int = 7,
    display_db: float | None = None,
    destripe: bool = False,
    include_clean_reference: bool = False,
    return_complex: bool = False,
    use_yolo: bool = False,
    yolo_model: Any = None,
    yolo_weights: str | Path | None = None,
    yolo_imgsz: int = 1280,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    yolo_use_obb: bool = True,
    yolo_max_dets: int = 1,
    yolo_fallback: str = "auto",
) -> dict[str, Any]:
    """光学图像 → SAR 反演主入口（对应双输入页面的"光学 + SAR"组合）。

    参数
    ----
    image : 路径 / np.ndarray / PIL.Image
        输入光学图像（灰度或彩色）。
    mask : np.ndarray / str / Path，可选
        ★ 外部目标掩码入口（例如自行训练的 YOLO 舰船检测输出）。
        bool 数组或非 0 即目标的数值数组（3 通道取第 1 通道），
        也可以是掩码图像文件路径；尺寸需与输入图像一致。
        提供后跳过内置分割，直接从掩码进入点云与成像链路。
    mask_polarity : "bright" | "dark"
        外部掩码模式下散射点幅度的取法：bright=随光学亮度增大（默认），
        dark=舰船比海面暗时先反转。
    use_yolo : 服务端 YOLO 舰船检测开关（CLI/HTTP 层默认开启，函数默认关闭）。
    yolo_weights : YOLO 权重路径；None 时按旧版约定解析
        （默认 <脚本上级>/optical2sar/weights/optyolo/best.pt 或环境变量
        OPTICAL2SAR_YOLO_WEIGHTS），直接替换 best.pt 即可换模型。
    yolo_imgsz / yolo_conf / yolo_iou / yolo_use_obb / yolo_max_dets :
        YOLO 推理参数，与旧版一致（1280 / 0.25 / 0.45 / True / 1）。
    yolo_fallback : "auto"=权重缺失或未检出时回退内置分割；
        "error"=直接抛错。
    parameters : 11 维雷达参数（None=默认值，见 DEFAULT_PARAMETERS）。
    segmentation : "auto"（Otsu 亮/暗双极性+连通域评分）| "manual"（需 threshold）。
    threshold / polarity / largest_component / roi : manual 分割的选项。
    step : 抽稀步长（None=自动，点数上限 30000）。
    pixel_size_x/y : 每像素对应的地面尺寸（米）。
    realism : "clean" | "realistic" | "rough-sea" | "custom" 或 RealismConfig。
    realism_options : dict，覆盖预设的任意 RealismConfig 字段。
    seed : 随机种子（杂波 rng=seed+101，成像 rng=seed）。
    display_db : 显示动态范围 dB；None=沿用真实感预设值（realistic=40），
        传 0 或负数切换为线性归一化显示。
    destripe : 显示层竖条纹抑制（只影响输出 PNG/显示图）。
    include_clean_reference : 额外用 clean 预设再成像一次作无杂波对照。
    return_complex : 返回结果中附带复数 Im / echo / RWC 数组。

    返回
    ----
    dict，关键字段：
        sar          [0,1] SAR 显示图（水平=方位向，竖直=地距向，与输入同朝向）
        linear       线性归一化幅值图（同朝向）
        sar_png / linear_png   对应 PNG 字节流
        clean / clean_png      clean 对照图（include_clean_reference=True 时）
        mask         实际使用的目标掩码（bool H×W）
        points       全部散射点（目标+杂波）N×4
        target_points 目标散射点 N×4
        metadata     元信息（尺寸/掩码诊断/雷达参数与派生量/告警/耗时）
        im_complex   复数 SAR 图像（return_complex=True 时，原生 RDA 朝向）
    """
    started = time.perf_counter()
    params = validate_parameters(parameters)
    image = _to_array(image)

    # ---- 1. 目标掩码解析：显式 mask= > 服务端 YOLO 检测 > 内置分割 ----
    yolo_info: dict[str, Any] = {}
    if mask is None and (use_yolo or yolo_model is not None):
        try:
            weights_used = resolve_yolo_weights(yolo_weights)
            model = yolo_model if yolo_model is not None else _load_yolo_model(weights_used)
            ymask = _yolo_ship_mask(
                image, model,
                imgsz=int(yolo_imgsz), conf=float(yolo_conf), iou=float(yolo_iou),
                use_obb=bool(yolo_use_obb), max_dets=int(yolo_max_dets),
            )
            if ymask is not None and bool(ymask.any()):
                mask = ymask
                yolo_info = {
                    "weights": str(weights_used),
                    "imgsz": int(yolo_imgsz), "conf": float(yolo_conf),
                    "iou": float(yolo_iou), "use_obb": bool(yolo_use_obb),
                    "max_dets": int(yolo_max_dets),
                }
            else:
                yolo_info["warning"] = "YOLO 未检出舰船目标"
        except Exception as exc:
            yolo_info["warning"] = f"YOLO 加载/推理失败: {type(exc).__name__}: {exc}"
        if mask is None and yolo_info.get("warning") and str(yolo_fallback).lower() == "error":
            raise RuntimeError(yolo_info["warning"])

    if mask is not None:
        seg = load_external_mask(mask, image, polarity=mask_polarity)
        seg.method = "yolo" if yolo_info.get("weights") else "external_mask"
        seg.diagnostics = mask_diagnostics(seg.mask, seg.score)
        if yolo_info:
            seg.diagnostics["yolo"] = dict(yolo_info)
    else:
        method = str(segmentation).lower()
        if method == "auto":
            seg = segment_auto(image, roi=roi)
        elif method == "manual":
            if threshold is None:
                raise ValueError("manual segmentation requires threshold")
            seg = segment_manual(
                image, threshold=threshold, polarity=polarity,
                largest_component=largest_component, roi=roi,
            )
        else:
            raise ValueError("segmentation method must be auto/manual (or pass mask=...)")

    # ---- 2. 掩码 → 点云 → 波束中心放置 ----
    used_step = _auto_step(seg.mask, step)
    points_local = mask_to_points(
        seg,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        step=used_step,
        center=True,
    )
    if points_local.size == 0:
        raise RuntimeError("optical segmentation produced no scatter points")
    P = place_scene_at_beam(points_local, params)

    # ---- 3. 真实感配置解析 ----
    if isinstance(realism, RealismConfig):
        if realism_options:
            raise ValueError("realism_options cannot be combined with a RealismConfig")
        realism_config = realism
    else:
        realism_config = realism_from_dict(realism_options, preset=str(realism), seed=seed)
    effective_seed = int(realism_config.seed)

    # ---- 4. SAR 成像 ----
    sar = simulate_sar(P, params, realism=realism_config)
    full_display = sar["display"]
    linear = sar["linear"]
    effective_display_db = float(display_db) if display_db else realism_config.display_db
    if effective_display_db:
        full_display = normalize_im_db(sar["image"], effective_display_db)
    else:
        full_display = linear
    if destripe:
        # 与批量转换页一致：条纹抑制作用于 RDA 原生竖直（距离列）方向
        full_display = remove_vertical_stripes(full_display)
    # RDA 原生矩阵行=方位（对应光学列）；转置显示使结果与输入同朝向
    primary = full_display.T
    secondary = linear.T

    extras: dict[str, np.ndarray] = {}
    if include_clean_reference and realism_config.preset != "clean":
        clean_sar = simulate_sar(P, params, realism=realism_preset("clean", seed=effective_seed))
        clean_display = clean_sar["display"] if realism_preset("clean").display_db else clean_sar["linear"]
        extras["clean"] = clean_display.T

    # ---- 5. 元数据 ----
    elapsed = time.perf_counter() - started
    wavelength = C_LIGHT / float(params[8])
    range_resolution = C_LIGHT / (2.0 * float(params[9]))
    azimuth_resolution = float(params[5]) / 2.0
    az_beamwidth = 0.886 * wavelength / float(params[5])
    rc = float(params[0]) / np.cos(np.deg2rad(float(params[2])))
    synthetic_aperture = rc * az_beamwidth
    doppler_bandwidth = 2.0 * float(params[3]) * az_beamwidth / wavelength
    warnings: list[str] = []
    if seg.warning:
        warnings.append(seg.warning)
    if yolo_info.get("warning"):
        warnings.append(str(yolo_info["warning"]))
    if float(params[6]) < doppler_bandwidth:
        warnings.append("PRF低于估算多普勒带宽，方位向可能发生混叠")
    elif float(params[6]) < 1.15 * doppler_bandwidth:
        warnings.append("PRF接近估算多普勒带宽下限，建议保留采样裕量")

    metadata = {
        "source_type": "optical",
        "source": "ndarray" if not isinstance(image, (str, Path)) else str(image),
        "input_size": [int(image.shape[1]), int(image.shape[0])],
        "segmentation": seg.method,
        "segmentation_requested": (
            "yolo" if yolo_info.get("weights")
            else "external_mask" if mask is not None
            else str(segmentation)
        ),
        "polarity": seg.polarity,
        "threshold": seg.threshold,
        "mask_pixels": int(seg.mask.sum()),
        "segmentation_score": seg.score,
        "segmentation_diagnostics": seg.diagnostics,
        "yolo": dict(yolo_info) if yolo_info else None,
        "step": int(used_step),
        "interpretation": "optical template-driven approximation",
        "output_size": [int(primary.shape[1]), int(primary.shape[0])],
        "full_output_size": [int(full_display.shape[1]), int(full_display.shape[0])],
        "display_axes": "horizontal=azimuth, vertical=ground_range",
        "secondary_view": "linear-magnitude SAR reconstruction",
        "display_db": float(effective_display_db) if effective_display_db else None,
        "clutter": sar["clutter_meta"],
        "errors": sar["error_meta"],
        "imaging_mode": "sar",
        "realism_preset": realism_config.preset,
        "seed": effective_seed,
        "target_points": int(P.shape[0]),
        "total_points": int(sar["points"].shape[0]),
        "radar_parameters": [float(x) for x in params],
        "radar_derived": {
            "wavelength_m": wavelength,
            "range_resolution_m": range_resolution,
            "azimuth_resolution_m": azimuth_resolution,
            "azimuth_beamwidth_deg": float(np.rad2deg(az_beamwidth)),
            "synthetic_aperture_m": synthetic_aperture,
            "estimated_doppler_bandwidth_hz": doppler_bandwidth,
        },
        "warnings": warnings,
        "extra_views": list(extras),
        "elapsed_sec": round(elapsed, 3),
        "disclaimer": (
            "Optical mode is template-driven simulation; it does not recover a "
            "real coherent SAR observation."
        ),
    }

    result: dict[str, Any] = {
        "sar": primary,
        "linear": secondary,
        "sar_png": _png_bytes(primary),
        "linear_png": _png_bytes(secondary),
        "mask": seg.mask,
        "points": sar["points"],
        "target_points": P,
        "metadata": metadata,
    }
    for name, arr in extras.items():
        result[name] = arr
        result[f"{name}_png"] = _png_bytes(arr)
    if return_complex:
        result["im_complex"] = sar["image"]
        result["echo"] = sar["echo"]
        result["RWC"] = sar["RWC"]
    return result


# =========================================================================
# 光学图像 → SAR 新算法入口（沿用旧函数名 optical_to_isar_v2_yolo，
# 输入输出同型：uint8 单通道图像；YOLO 权重调用方式与旧版一致）
# =========================================================================

def optical_to_isar_v2_yolo(
    optical_img,
    *,
    # ---- 旧版参数：保留占位兼容旧调用，新算法不再使用 ----
    saliency_threshold=None, saliency_feather=None, white_ratio=None,
    randomness=None, brightness=None,
    # ---- 新算法参数 ----
    parameters=None,                   # 11 维雷达参数（列表或 {名称:值} 覆盖字典）
    segmentation="auto",               # auto=Otsu自动 / manual=人工阈值
    threshold=None,                    # manual 分割阈值 (0-255)
    polarity="bright",                 # bright=亮目标 / dark=暗目标
    largest_component=True,
    roi=None,                          # [x0,y0,x1,y1]
    step=None,                         # 抽稀步长（None=自动 ≤30000 点）
    pixel_size_x=1.0, pixel_size_y=1.0,
    preset="realistic",                # clean / realistic / rough-sea
    realism_options=None,
    yolo_model=None, yolo_weights=None,
    yolo_imgsz=1280, yolo_conf=0.25, yolo_iou=0.45,
    yolo_use_obb=True, yolo_max_dets=1, yolo_fallback="auto",
    mask_polarity="bright",
    display_db=None, destripe=False,
    seed=7,
):
    """旧接口函数名的新算法实现：光学图像 → SAR（RDA 成像）。"""
    if seed is None:
        seed = 7
    if isinstance(parameters, dict):
        params = DEFAULT_PARAMETERS.copy()
        for key, value in parameters.items():
            if key not in PARAMETER_FIELDS:
                raise ValueError(f"未知雷达参数名: {key}（可用: {', '.join(PARAMETER_FIELDS)}）")
            params[PARAMETER_FIELDS[key]] = float(value)
        parameters = params
    result = optical_to_sar(
        optical_img,
        parameters=parameters,
        segmentation=segmentation,
        threshold=threshold,
        polarity=polarity,
        largest_component=largest_component,
        roi=roi,
        step=step,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        realism=preset,
        realism_options=realism_options,
        seed=seed,
        display_db=display_db,
        destripe=destripe,
        use_yolo=(yolo_model is not None) or (yolo_weights is not None),
        yolo_model=yolo_model,
        yolo_weights=yolo_weights,
        yolo_imgsz=yolo_imgsz,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        yolo_use_obb=yolo_use_obb,
        yolo_max_dets=yolo_max_dets,
        yolo_fallback=yolo_fallback,
        mask_polarity=mask_polarity,
    )
    return (np.clip(result["sar"], 0.0, 1.0) * 255.0).astype(np.uint8)


# ========== 输入/输出 ==========

# 默认输入：相对于本脚本目录（motai）的一键处理样例
DEFAULT_INPUT_IMAGES = [
    "./2312090039.jpg",
    "./P3461.bmp",
]


def generate_output_filename(input_path: str, suffix: str = "_isar_v2_yolo", output_ext: str = ".png", use_timestamp: bool = True) -> str:
    p = Path(input_path)
    if use_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = p.parent / f"{p.stem}{suffix}_{ts}{output_ext}"
    else:
        out = p.parent / f"{p.stem}{suffix}{output_ext}"
        c = 1
        while out.exists():
            out = p.parent / f"{p.stem}{suffix}_{c}{output_ext}"
            c += 1
    return str(out)


def _imread_cn(path: str) -> Optional[np.ndarray]:
    """
    优先使用 cv2.imdecode 兼容中文路径，失败则 PIL 兜底。
    """
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image

        pil = Image.open(path)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# 新代码20260316（接口不变：POST /  JSON input_path/output_path → {success, resultPath}）
@optical2sar_bp.route('/', methods=['POST'])
def convert_image():
    """
    光学图像转SAR图像API接口（协议与旧版一致，核心算法已替换为 RDA 新算法）
    输入:
        - JSON格式请求体，包含input_path和output_path
        - input_path: 输入光学图像的存储路径
        - output_path: 输出SAR图像的存储路径（可选，缺省 <原名>_vis<原后缀>）
        - options: 可选参数 {preset, seed, method, threshold, polarity,
                    parameters, yolo_weights, no_yolo}
    输出:
        - JSON格式响应，包含success状态和resultPath路径
    """
    try:
        data = request.get_json(silent=True) or {}
        input_raw = data.get('input_path')
        if not input_raw:
            return jsonify({
                'success': False,
                'message': '缺少 input_path'
            }), 400
        input_path = Path(input_raw)
        if not input_path.is_absolute():
            input_path = base / input_path
        if not input_path.exists():
            return jsonify({
                'success': False,
                'message': f'文件 {input_path} 不存在'
            }), 404
        options = data.get('options') or {}

        # 输出路径：显式指定优先；缺省沿用旧版规则 <原名>_vis<原后缀>
        out_raw = options.get('output_path') or data.get('output_path')
        if out_raw:
            out_path = Path(out_raw)
            if not out_path.is_absolute():
                out_path = base / out_path
        else:
            out_path = input_path.parent / f"{input_path.stem}_vis{input_path.suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # YOLO 权重调用方式与旧版一致：默认 optical2sar/weights/optyolo/best.pt，
        # 替换 best.pt 文件即换模型；失败/未检出时自动回退内置分割
        yolo_model = None
        if not options.get('no_yolo', False):
            try:
                yolo_model = _load_yolo_model(resolve_yolo_weights(options.get('yolo_weights')))
                print(f"YOLO 权重: {resolve_yolo_weights(options.get('yolo_weights'))}")
            except Exception as e:
                print(f"警告: YOLO 初始化失败，回退内置分割：{e}")
                yolo_model = None

        img = _imread_cn(str(input_path))
        if img is None:
            return jsonify({
                'success': False,
                'message': f'无法读取图像 {input_path}'
            }), 400

        result_img = optical_to_isar_v2_yolo(
            img,
            seed=options.get('seed', 7),
            preset=str(options.get('preset', 'realistic')),
            segmentation=str(options.get('method', 'auto')),
            threshold=options.get('threshold'),
            polarity=str(options.get('polarity', 'bright')),
            parameters=options.get('parameters'),
            yolo_model=yolo_model,
        )

        # 保存
        try:
            from PIL import Image

            Image.fromarray(result_img, mode="L").save(out_path)
        except Exception:
            cv2.imwrite(str(out_path), result_img)

        return jsonify({
            'success': True,
            'resultPath': str(out_path)
        })

    except Exception as e:
        # 捕获并返回异常信息
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


def main():
    parser = argparse.ArgumentParser(description="光学图像 -> SAR（RDA 新算法；接口与旧版 optical_to_isar_v2_yolo 一致）")

    parser.add_argument("--input", "-i", type=str, default=None, help="输入图像路径（不指定则处理默认列表）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径（单张时有效）")

    # 旧版参数：保留占位兼容旧命令行（新算法不再使用）
    parser.add_argument("--saliency-threshold", type=float, default=0.01, help="（兼容占位，新算法不使用）")
    parser.add_argument("--feather", type=int, default=10, help="（兼容占位，新算法不使用）")
    parser.add_argument("--white-ratio", type=float, default=0.9, help="（兼容占位，新算法不使用）")
    parser.add_argument("--randomness", type=float, default=0.9, help="（兼容占位，新算法不使用）")
    parser.add_argument("--brightness", type=float, default=0.9, help="（兼容占位，新算法不使用）")

    # 新算法参数
    parser.add_argument("--seed", type=int, default=None, help="随机种子，可复现")
    parser.add_argument("--preset", type=str, default="realistic", choices=["clean", "realistic", "rough-sea"], help="真实感预设")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "manual"], help="内置分割方法")
    parser.add_argument("--threshold", type=float, default=None, help="manual 分割阈值 (0-255)")
    parser.add_argument("--polarity", type=str, default="bright", choices=["bright", "dark"], help="目标明暗")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="覆盖雷达参数，如 Hc=3000（KEY: Hc/thetaSQ/thetaSL/vx/D_rg/D_az/PRF/T/f0/Br/fs）")

    # YOLO 参数（与旧版同名同义）
    parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO 掩码（退回内置分割）")
    parser.add_argument("--yolo-weights", type=str, default=None, help="YOLO 权重路径（默认 opyolo/best.pt 约定路径）")
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="YOLO 推理尺寸 imgsz")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU 阈值")
    parser.add_argument("--yolo-use-obb", action="store_true", default=True, help="优先使用 OBB（如果模型支持）")
    parser.add_argument("--yolo-max-dets", type=int, default=1, help="最多使用前 N 个目标掩码（建议 1）")

    args = parser.parse_args()

    yolo_model = None
    if not args.no_yolo:
        try:
            yolo_model = _load_yolo_model(resolve_yolo_weights(args.yolo_weights))
            print(f"YOLO 权重: {resolve_yolo_weights(args.yolo_weights)}")
        except Exception as e:
            print(f"警告: YOLO 初始化失败，回退内置分割：{e}")
            yolo_model = None

    params = None
    if args.param:
        params = DEFAULT_PARAMETERS.copy()
        for item in args.param:
            key, raw = (part.strip() for part in item.split("=", 1))
            if key not in PARAMETER_FIELDS:
                raise ValueError(f"未知雷达参数名: {key}（可用: {', '.join(PARAMETER_FIELDS)}）")
            params[PARAMETER_FIELDS[key]] = float(raw)

    if args.input:
        input_list = [args.input]
        use_timestamp = args.output is None
    else:
        # 默认按"脚本目录相对路径"拼接，直接运行脚本即可处理这两张图
        input_list = [str((Path(__file__).resolve().parent / name).resolve()) for name in DEFAULT_INPUT_IMAGES]
        use_timestamp = True

    for inp in input_list:
        if not os.path.exists(inp):
            print(f"跳过（不存在）: {inp}")
            continue

        img = _imread_cn(inp)
        if img is None:
            print(f"错误: 无法读取图像 {inp}")
            continue

        out_path = args.output if args.output and len(input_list) == 1 else generate_output_filename(inp, use_timestamp=use_timestamp)

        result_img = optical_to_isar_v2_yolo(
            img,
            seed=args.seed,
            preset=args.preset,
            segmentation=args.method,
            threshold=args.threshold,
            polarity=args.polarity,
            parameters=params,
            yolo_model=yolo_model,
            yolo_imgsz=args.yolo_imgsz,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_use_obb=bool(args.yolo_use_obb),
            yolo_max_dets=args.yolo_max_dets,
        )

        # 保存
        try:
            from PIL import Image

            Image.fromarray(result_img, mode="L").save(out_path)
        except Exception:
            cv2.imwrite(out_path, result_img)

        print(f"SAR(新算法) 完成: {out_path}")


if __name__ == "__main__":
    main()
