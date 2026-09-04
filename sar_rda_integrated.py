from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


# ============================================================
# 1. 雷达参数：与原 MATLAB 的 parameter 向量顺序一致
# ============================================================

@dataclass(frozen=True)
class RadarParams:
    # [Hc, thetaSQ, thetaSL, v, D_rg, D_az, PRF, T, f0, Br, fs]
    Hc: float = 5000.0
    thetaSQ_deg: float = 0.0
    thetaSL_deg: float = 45.0
    v: float = 100.0
    D_rg: float = 2.0
    D_az: float = 8.0
    PRF: float = 30.0
    T: float = 5.0e-6
    f0: float = 1.0e9
    Br: float = 8.0e7
    fs: float = 1.0e8

    @classmethod
    def from_vector(cls, values: Iterable[float]) -> "RadarParams":
        values = list(values)
        if len(values) != 11:
            raise ValueError("Radar parameter vector must contain 11 values.")
        return cls(*map(float, values))


@dataclass
class RDAResult:
    echo: np.ndarray
    RWC: np.ndarray
    RGC: np.ndarray
    RMC: np.ndarray
    Im: np.ndarray


@dataclass
class ScatterExtractionResult:
    gray: np.ndarray
    mask: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rcs: np.ndarray
    scatter_points: np.ndarray
    threshold: float


DEFAULT_PARAMS = RadarParams()


# ============================================================
# 2. 基础工具：MATLAB round、.mat 读取、图像保存
# ============================================================

def matlab_round(x: np.ndarray | float) -> np.ndarray | int:
    """MATLAB round: half away from zero."""
    arr = np.asarray(x)
    out = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    if np.isscalar(x):
        return int(out)
    return out.astype(np.int64)


def even_int(value: float | int) -> int:
    out = int(matlab_round(float(value)))
    return out if out % 2 == 0 else out + 1


def normalize_scatter_points(points: np.ndarray) -> np.ndarray:
    """兼容 N×4 或 4×N 散射点矩阵，输出 N×4: [x, y, z, rcs]."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("scatter points must be a 2-D array.")
    if p.shape[1] == 4:
        return p.copy()
    if p.shape[0] == 4:
        return p.T.copy()
    raise ValueError(f"Expected N×4 or 4×N scatter points, got {p.shape}.")


def load_scatter_mat(path: str | Path, variable: str | None = "P") -> np.ndarray:
    """读取 MATLAB .mat 散射点。兼容普通 .mat 和 v7.3 HDF5 .mat。"""
    path = Path(path)
    try:
        data = sio.loadmat(path)
        candidates = {k: v for k, v in data.items() if not k.startswith("__")}
        if variable and variable in candidates:
            return normalize_scatter_points(candidates[variable])
        for v in candidates.values():
            a = np.asarray(v)
            if a.ndim == 2 and (a.shape[0] == 4 or a.shape[1] == 4):
                return normalize_scatter_points(a)
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as f:
        if variable and variable in f:
            return normalize_scatter_points(f[variable][()])
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                a = obj[()]
                if a.ndim == 2 and (a.shape[0] == 4 or a.shape[1] == 4):
                    return normalize_scatter_points(a)
    raise ValueError(f"No N×4 or 4×N scatter-point variable found in {path}.")


def save_sar_png(
    image: np.ndarray,
    output_path: str | Path,
    *,
    aspect: str = "auto",
    gamma: float = 1.0,
    dpi: int = 200,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> None:
    """保存 SAR 幅度图。

    关键修正：默认使用 aspect='auto'，接近 MATLAB imagesc 的显示效果，
    避免直接按矩阵像素比例保存导致目标横向拉长。

    aspect='equal' 或 'raw' 可用于查看真实矩阵像素比例，方便调试。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.abs(image).astype(np.float64)
    if arr.size == 0:
        raise ValueError("Cannot save an empty image.")
    m = float(arr.max())
    if m > 0:
        arr /= m
    if gamma != 1.0:
        arr = arr ** gamma

    if aspect == "raw":
        # 原始矩阵比例，调试用：如果 Im 是 130×990，图片也会很宽。
        plt.imsave(output_path, arr, cmap="gray")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap="gray", aspect=aspect)
    ax.axis("off")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_scatter_png(result: ScatterExtractionResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(result.x, result.y, s=8, c=result.rcs)
    fig.colorbar(sc, ax=ax, label="RCS")
    ax.set_title("Scattering Points Distribution")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Azimuth (m)")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def default_image_paths() -> list[Path]:
    data_dir = Path("data")
    candidates = [
        data_dir / "123.jpg",
        data_dir / "100000460.bmp",
        data_dir / "100000467.bmp",
        data_dir / "林肯号航母.png",
    ]
    return [path for path in candidates if path.exists()]


# ============================================================
# 3. 光学图像 -> 散射点：对应 main.m 前半部分
# ============================================================

def read_gray01(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        gray = arr.astype(np.float64)
    elif arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        # MATLAB rgb2gray 近似权重
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    max_value = float(gray.max())
    if max_value <= 0:
        raise ValueError("Input image is all zero.")
    return gray / max_value


def image_to_scatter_points(
    image_path: str | Path,
    *,
    step: int = 4,
    Lx: float = 100.0,
    Ly: float = 40.0,
    min_area: int = 50,
    rcs_mode: str = "square",
    range_scale: float = 20.0,
    azimuth_scale: float = 20.0,
    range_offset: float = 5000.0,
) -> ScatterExtractionResult:
    """从光学图像提取散射点。

    默认参数保持原 main.m：step=4, Lx=100, Ly=40,
    rcs=intensity^2, scatter=[x*20+5000, y*20, 0, rcs]。
    """
    if step <= 0:
        raise ValueError("step must be positive.")

    gray = read_gray01(image_path)
    level = float(threshold_otsu(gray, nbins=256))
    bw = gray > level

    # MATLAB bwareaopen(..., 50) + imfill(..., 'holes')
    structure = ndi.generate_binary_structure(2, 2)
    labels, n_labels = ndi.label(bw, structure=structure)
    if n_labels > 0:
        counts = np.bincount(labels.ravel())
        keep = counts >= min_area
        keep[0] = False
        bw = keep[labels]
    else:
        bw = np.zeros_like(bw, dtype=bool)
    bw = ndi.binary_fill_holes(bw)

    # MATLAB find() 是列优先顺序；这里保持一致，否则 step 抽样点会不同。
    flat_idx = np.flatnonzero(bw.ravel(order="F"))
    row0, col0 = np.unravel_index(flat_idx, bw.shape, order="F")
    row0 = row0[::step]
    col0 = col0[::step]

    ny, nx = gray.shape
    dx = Lx / nx
    dy = Ly / ny
    row1 = row0.astype(np.float64) + 1.0
    col1 = col0.astype(np.float64) + 1.0
    x = (col1 - nx / 2.0) * dx
    y = (row1 - ny / 2.0) * dy

    intensity = gray[row0, col0]
    if rcs_mode == "linear":
        rcs = intensity
    elif rcs_mode == "square":
        rcs = intensity ** 2
    elif rcs_mode == "log":
        rcs = 10.0 * np.log10(intensity + 1e-6)
    else:
        raise ValueError("rcs_mode must be linear, square, or log.")

    scatter = np.column_stack((
        x * range_scale + range_offset,
        y * azimuth_scale,
        np.zeros_like(x),
        rcs,
    )).astype(np.float64)

    return ScatterExtractionResult(gray=gray, mask=bw, x=x, y=y, rcs=rcs,
                                   scatter_points=scatter, threshold=level)


# ============================================================
# 4. RDA 核心：对应 RDA.m。默认保留 MATLAB 的前两列坐标交换
# ============================================================

def visible_mask_for_pulse(P: np.ndarray, q: np.ndarray, qp: np.ndarray,
                           theta_az: float, theta_rg: float) -> np.ndarray:
    qq = qp - q
    qq_norm_sq = float(np.dot(qq, qq))
    if qq_norm_sq <= 0.0:
        return np.zeros(P.shape[1], dtype=bool)

    qpx = np.vstack((
        P[0, :] - q[0],
        np.full(P.shape[1], qp[1] - q[1]),
        P[2, :] - q[2],
    ))
    qpy = np.vstack((
        np.full(P.shape[1], qp[0] - q[0]),
        P[1, :] - q[1],
        P[2, :] - q[2],
    ))

    dot_x = qq @ qpx
    dot_y = qq @ qpy
    norm_x = np.sqrt(qq_norm_sq * np.sum(qpx * qpx, axis=0))
    norm_y = np.sqrt(qq_norm_sq * np.sum(qpy * qpy, axis=0))
    cos_x = np.clip(dot_x / norm_x, -1.0, 1.0)
    cos_y = np.clip(dot_y / norm_y, -1.0, 1.0)
    return (np.arccos(cos_x) <= theta_az / 2.0) & (np.arccos(cos_y) <= theta_rg / 2.0)


def rda(parameters: RadarParams | Iterable[float], scatter_points: np.ndarray,
        *, progress: bool = True, swap_xy: bool = True) -> RDAResult:
    if not isinstance(parameters, RadarParams):
        parameters = RadarParams.from_vector(parameters)

    P_in = normalize_scatter_points(scatter_points)
    if swap_xy:
        # 原 MATLAB RDA.m 开头有 tmp=P(:,1); P(:,1)=P(:,2); P(:,2)=tmp;
        P_in = P_in.copy()
        P_in[:, [0, 1]] = P_in[:, [1, 0]]
    P = P_in.T  # 4×N
    num = P.shape[1]
    if num == 0:
        raise ValueError("No scatter points were provided.")

    Hc = parameters.Hc
    thetaSQ = np.deg2rad(parameters.thetaSQ_deg)
    thetaSL = np.deg2rad(parameters.thetaSL_deg)
    vx0 = parameters.v
    D_rg = parameters.D_rg
    D_az = parameters.D_az
    PRF = parameters.PRF
    T = parameters.T
    f0 = parameters.f0
    Br = parameters.Br
    fs = parameters.fs

    c = 2.9979e8
    kr = Br / T
    PRI = 1.0 / PRF
    Rc = Hc / (np.cos(thetaSQ) * np.cos(thetaSL))
    wavelength = c / f0
    theta_az = 0.886 * wavelength / D_az
    theta_rg = 0.886 * wavelength / D_rg
    LSAR = Rc * theta_az / np.cos(thetaSQ)

    Na1 = even_int((np.min(P[0, :]) - 0.6 * LSAR) / vx0 / PRI)
    Na2 = even_int((np.max(P[0, :]) + 0.6 * LSAR) / vx0 / PRI)
    Na = Na2 - Na1
    if Na <= 0:
        raise ValueError(f"Invalid azimuth sample count Na={Na}.")

    ta = np.arange(Na1, Na2, dtype=np.float64) * PRI
    Q = np.array([[0.0], [0.0], [Hc]]) @ np.ones((1, Na)) + np.array([[vx0], [0.0], [0.0]]) @ ta[None, :]
    Qp = np.vstack((
        Q[0, :] + Q[2, :] / np.cos(thetaSL) * np.tan(thetaSQ),
        Q[1, :] + Q[2, :] * np.tan(thetaSL),
        np.zeros(Na),
    ))

    N_LFM = even_int(T * fs)
    tLFM = np.arange(-N_LFM / 2, N_LFM / 2, dtype=np.float64) / fs

    diff = Q[:, None, :] - P[:3, :, None]
    R_all = np.sqrt(np.sum(diff * diff, axis=0))  # Ntarget × Na

    visible_masks: list[np.ndarray] = []
    Rmax = np.zeros((num, Na), dtype=np.float64)
    for i in range(Na):
        mask = visible_mask_for_pulse(P, Q[:, i], Qp[:, i], theta_az, theta_rg)
        visible_masks.append(mask)
        Rmax[mask, i] = R_all[mask, i]

    tRmin = 2.0 * np.min(R_all) / c
    max_visible_range = float(np.max(Rmax))
    if max_visible_range <= 0:
        raise RuntimeError("No scatter point is inside the antenna beam footprint.")
    tRmax = 2.0 * max_visible_range / c
    Nr = even_int((tRmax - tRmin) * fs) + N_LFM
    if Nr <= N_LFM:
        raise ValueError(f"Invalid range sample count Nr={Nr}.")

    echo = np.zeros((Na, Nr), dtype=np.complex128)
    lfm = np.exp(1j * np.pi * kr * tLFM ** 2)
    for i in range(Na):
        idx_targets = np.flatnonzero(visible_masks[i])
        if idx_targets.size:
            ranges = R_all[idx_targets, i]
            delays = matlab_round((2.0 * ranges / c - tRmin) * fs)
            phases = np.exp(-1j * 4.0 * np.pi * ranges / wavelength)
            amps = P[3, idx_targets]
            for delay, amp, phase in zip(delays, amps, phases):
                start = int(delay)
                stop = start + N_LFM
                if 0 <= start and stop <= Nr:
                    echo[i, start:stop] += amp * phase * lfm
        if progress and (i == 0 or i + 1 == Na or (i + 1) % max(1, Na // 10) == 0):
            print(f"Echo generating: {i + 1}/{Na}")

    # Equivalent side-looking / RWC
    fr_vec = np.arange(-Nr / 2, Nr / 2, dtype=np.float64) * fs / Nr
    fr = np.fft.fftshift(np.tile(fr_vec, (Na, 1)))
    ta_vec = np.arange(-Na / 2, Na / 2, dtype=np.float64) * PRI
    ta_grid = np.tile(ta_vec[:, None], (1, Nr))
    fDC = 2.0 * vx0 * np.sin(thetaSQ) / wavelength
    Hsl = np.exp(-1j * 4.0 * np.pi / c * wavelength * fDC / 2.0 * ta_grid * (f0 + fr))
    RWC = np.fft.ifft(np.fft.fft(echo, axis=1) * Hsl, axis=1)
    vx = vx0 * np.cos(thetaSQ)

    # Range compression + SRC
    fa_vec = np.arange(-Na / 2, Na / 2, dtype=np.float64) / (Na * PRI)
    fa = np.fft.fftshift(np.tile(fa_vec[:, None], (1, Nr)))
    R0 = np.tile(np.arange(-Nr / 2, Nr / 2, dtype=np.float64) / fs * c / 2.0 + Rc, (Na, 1))
    Rref = Rc
    with np.errstate(divide="ignore", invalid="ignore"):
        RCMF = np.sqrt(1.0 - wavelength ** 2 * fa ** 2 / (4.0 * vx ** 2))
        Ksrc = 2.0 * vx ** 2 * f0 ** 3 * RCMF ** 3 / (c * Rref * fa ** 2)
        Hr = np.exp(1j * np.pi * fr ** 2 / kr)
        Hsrc = np.exp(-1j * np.pi * fr ** 2 / Ksrc)
    Hsrc = Hsrc.astype(np.complex128, copy=False)
    bad = ~np.isfinite(Hsrc.real) | ~np.isfinite(Hsrc.imag)
    Hsrc[bad] = 1.0 + 0.0j

    spec = np.fft.fft2(RWC) * Hsrc * Hr
    RGC = np.fft.ifft2(spec)

    # RCMC
    Hrcmc = np.exp(1j * 4.0 * np.pi * fr * (Rref / RCMF - Rc) / c)
    spec = spec * Hrcmc
    RMC = np.fft.ifft2(spec)

    # Azimuth compression
    Ha = np.exp(1j * 4.0 * np.pi / wavelength * R0 * RCMF)
    Im = np.fft.ifft(np.fft.ifft(spec, axis=1) * Ha, axis=0)
    return RDAResult(echo=echo, RWC=RWC, RGC=RGC, RMC=RMC, Im=Im)


# ============================================================
# 5. 一键运行入口
# ============================================================

def run_from_image(args: argparse.Namespace) -> None:
    images = list(args.image)
    if not images:
        raise ValueError("No input images were provided.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        image_path = Path(image_path)
        out = out_root if len(images) == 1 else out_root / image_path.stem
        out.mkdir(parents=True, exist_ok=True)

        extracted = image_to_scatter_points(
            image_path,
            step=args.step,
            Lx=args.Lx,
            Ly=args.Ly,
            min_area=args.min_area,
            rcs_mode=args.rcs_mode,
            range_scale=args.range_scale,
            azimuth_scale=args.azimuth_scale,
            range_offset=args.range_offset,
        )
        np.save(out / "scatter_points.npy", extracted.scatter_points)
        sio.savemat(out / "scatter_points.mat", {"P": extracted.scatter_points})
        save_scatter_png(extracted, out / "scatter_points.png")

        print(f"[{image_path.name}] Otsu threshold: {extracted.threshold:.6f}")
        print(f"[{image_path.name}] Scatter points: {extracted.scatter_points.shape[0]}")

        result = rda(DEFAULT_PARAMS, extracted.scatter_points, progress=not args.no_progress, swap_xy=not args.no_rda_swap)
        save_sar_png(result.Im, out / "sar_image_abs.png", aspect=args.aspect, gamma=args.gamma)
        save_sar_png(result.Im, out / "sar_image_abs_raw_pixel.png", aspect="raw", gamma=args.gamma)

        if args.save_npz:
            np.savez_compressed(out / "rda_outputs.npz", echo=result.echo, RWC=result.RWC,
                                RGC=result.RGC, RMC=result.RMC, Im=result.Im)

        print(f"[{image_path.name}] Im shape: {result.Im.shape}  (Na, Nr)")
        print(f"[{image_path.name}] Saved: {out.resolve()}")


def run_from_mat(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scatter = load_scatter_mat(args.mat, variable=args.var)
    print(f"Scatter points: {scatter.shape}")

    result = rda(DEFAULT_PARAMS, scatter, progress=not args.no_progress, swap_xy=not args.no_rda_swap)
    save_sar_png(result.Im, out / "sar_image_abs.png", aspect=args.aspect, gamma=args.gamma)
    save_sar_png(result.Im, out / "sar_image_abs_raw_pixel.png", aspect="raw", gamma=args.gamma)

    if args.save_npz:
        np.savez_compressed(out / "rda_outputs.npz", echo=result.echo, RWC=result.RWC,
                            RGC=result.RGC, RMC=result.RMC, Im=result.Im)

    print(f"Im shape: {result.Im.shape}  (Na, Nr)")
    print(f"Saved: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Integrated optical-image/scatter-points to SAR RDA imaging.")
    p.add_argument("--mode", choices=["image", "mat"], default="image", help="image: 光学图像成像；mat: 散射点 .mat 成像")
    p.add_argument("--image", type=Path, nargs="+", default=default_image_paths(), help="输入光学图像，可传一个或多个；默认读取 data 目录下可用图片")
    p.add_argument("--mat", type=Path, default=Path("data/P_20250313T093915.mat"), help="输入散射点 .mat 文件")
    p.add_argument("--var", type=str, default="P", help=".mat 中的变量名")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="输出目录")

    p.add_argument("--step", type=int, default=4, help="散射点降采样步长，越大越快，默认保持 main.m 的 4")
    p.add_argument("--Lx", type=float, default=100.0, help="图像横向物理尺寸，默认 100 m")
    p.add_argument("--Ly", type=float, default=40.0, help="图像纵向物理尺寸，默认 40 m")
    p.add_argument("--min-area", type=int, default=50, help="去除小区域阈值，默认 50")
    p.add_argument("--rcs-mode", choices=["linear", "square", "log"], default="square", help="RCS 灰度映射方式")
    p.add_argument("--range-scale", type=float, default=20.0, help="对应 main.m 的 x*20")
    p.add_argument("--azimuth-scale", type=float, default=20.0, help="对应 main.m 的 y*20")
    p.add_argument("--range-offset", type=float, default=5000.0, help="对应 main.m 的 +5000")

    p.add_argument("--aspect", choices=["auto", "equal"], default="auto", help="保存图像显示比例；auto 接近 MATLAB imagesc，避免横向拉长")
    p.add_argument("--gamma", type=float, default=1.0, help="显示 gamma；例如 0.5 可增强暗部")
    p.add_argument("--save-npz", action="store_true", help="保存 echo/RWC/RGC/RMC/Im，中间结果很大，默认不保存")
    p.add_argument("--no-progress", action="store_true", help="关闭进度输出")
    p.add_argument("--no-rda-swap", action="store_true", help="关闭 RDA.m 开头的 x/y 坐标交换，仅调试用")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "image":
        run_from_image(args)
    else:
        run_from_mat(args)


if __name__ == "__main__":
    main()
