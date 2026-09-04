# SAR RDA 集成版 Python 代码

这个版本把原来分散的 Python 包合并成一个主文件：`sar_rda_integrated.py`。

核心修改：

1. 默认保存 SAR 图像时使用 `aspect="auto"`，接近 MATLAB `imagesc(abs(Im)); axis off` 的显示方式，避免直接按矩阵像素比例保存造成目标横向拉长。
2. 保留原 MATLAB `RDA.m` 中的坐标交换逻辑，保证算法链路尽量不变。
3. 默认不保存巨大的 `rda_outputs.npz`，只输出最终 SAR 图、原始像素比例调试图和散射点，工程更轻。
4. 所有核心功能集中在一个 Python 文件里，方便你后续修改和调试。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 从光学图像生成 SAR

```bash
python sar_rda_integrated.py --mode image --image data/123.jpg --out outputs/from_image
```

输出：

- `sar_image_abs.png`：修正显示比例后的 SAR 图，默认推荐看这个。
- `sar_image_abs_raw_pixel.png`：按矩阵原始像素比例保存的 SAR 图，用于检查为什么会横向变长。
- `scatter_points.npy` / `scatter_points.mat`：从图像提取的散射点。
- `scatter_points.png`：散射点分布图。

## 从 MATLAB 散射点文件生成 SAR

```bash
python sar_rda_integrated.py --mode mat --mat data/P_20250313T093915.mat --out outputs/from_mat
```

## 常用参数

```bash
# 增大 step 可以加快运行，但散射点会更少
python sar_rda_integrated.py --mode image --step 6

# 保存中间结果 echo/RWC/RGC/RMC/Im，文件会比较大
python sar_rda_integrated.py --mode image --save-npz

# 查看真实矩阵显示效果，可对比 raw_pixel 图
python sar_rda_integrated.py --mode image --aspect equal

# 仅调试坐标交换，不建议作为默认结果
python sar_rda_integrated.py --mode image --no-rda-swap
```

## 横向拉长的原因

RDA 输出矩阵通常是 `(Na, Nr)`，当前图像可能出现 `Nr` 远大于 `Na`。如果用 `plt.imsave()` 按原始矩阵像素保存，就会生成很宽的图片。原 MATLAB 代码用的是 `imagesc(abs(Im))`，并且 `axis equal` 是注释状态，所以显示时会自动填充坐标轴。本版本默认改成 `imshow(..., aspect="auto")` 后，视觉上更接近 MATLAB 显示。


## 相关分支

- [`optical-sim-v2.2.2`](https://github.com/heng526/optical_to_sar/tree/optical-sim-v2.2.2)：光学模板驱动的 SAR 仿真纯 Python 版（Otsu/YOLO 分割 → 散射点云 → K 分布海杂波 → RDA → 40dB 显示），附[算法原理详解文档](https://github.com/heng526/optical_to_sar/blob/optical-sim-v2.2.2/docs/算法原理详解.md)与理论级注释，适合学习 RDA 全链路。
