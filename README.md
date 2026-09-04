# optical_to_sar — 光学图像 → SAR 反演（纯 Python 实现）

把一张普通**光学图像**（海面舰船照片）转换为具有真实 **SAR（合成孔径雷达）** 观感的图像：相干斑颗粒、点目标十字旁瓣、K 分布海杂波噪底、40dB 动态范围。

- **算法**：光学模板 → 目标分割(Otsu/YOLO) → 散射点云 → K 分布海杂波 → RDA 成像（回波正演 + 距离压缩 + SRC + RCMC + 方位压缩）→ dB 显示
- **依赖**：仅 `numpy + pillow`（scipy 可选；ultralytics/flask 按需）——无 torch、无 onnx、无 opencv
- **来源**：从"双输入 SAR/ISAR 舰船仿真演示系统 v2.2.2"源码工程忠实转写，与部署系统**逐位一致**（同参数同 numpy 版本可复现）
- **免责声明**：输出为模板驱动的仿真近似，不能当作真实相干 SAR 观测数据

## 文件说明

| 文件 | 说明 |
|---|---|
| [`optical_to_sar.py`](optical_to_sar.py) | ★ 主文件：算法库 + 命令行 + HTTP 服务（兼容旧协议），带理论级中文注释 |
| [`optical_to_isar_v2_yolo.py`](optical_to_isar_v2_yolo.py) | 旧接口**直接替换版**：文件名/Blueprint/函数名/权重调用方式与旧版一致，仅核心算法换新 |
| [`docs/算法原理详解.md`](docs/算法原理详解.md) | ★ 算法思路、逐步公式推导、理论知识、代码导航——新人从这里开始 |

## 快速开始

```bash
pip install numpy pillow
```

### 命令行

```bash
# 最简：Otsu 自动分割 + realistic 真实感预设（海杂波/斑点/加窗/30dB噪声）
python optical_to_sar.py 舰船照片.jpg -o sar结果.png

# 用自训练 YOLO 权重做舰船检测（默认路径 optical2sar/weights/optyolo/best.pt，替换文件即换模型）
python optical_to_sar.py 照片.jpg -o sar.png --yolo-weights best.pt

# 完全确定性（无杂波/噪声，可逐位复现）
python optical_to_sar.py 照片.bmp -o sar.png --preset clean --seed 7

# 覆盖雷达参数（例如距离分辨率提高到 0.375m：Br=400MHz）
python optical_to_sar.py 照片.jpg -o sar.png --param Br=400e6 --param PRF=60

# 查看全部 11 个雷达参数
python optical_to_sar.py --list-params
```

### 在代码中调用

```python
from optical_to_sar import optical_to_sar

out = optical_to_sar("ship.jpg")          # 默认：auto 分割 + realistic 预设
out["sar"]        # [0,1] SAR 显示图（水平=方位向，竖直=地距向，与输入同朝向）
out["linear"]     # 线性归一化幅值图
out["mask"]       # 实际使用的目标掩码
out["points"]     # 散射点云 N×4 [方位x, 地距y, z, 幅度]
out["metadata"]   # 分辨率/点数/告警/耗时等元信息
```

### HTTP 服务（兼容旧协议）

```bash
python optical_to_sar.py --serve --host 127.0.0.1 --port 5001
# 或集成进现有 Flask 应用（两行，与旧版完全一致）：
#   from optical_to_sar import optical2sar_bp
#   app.register_blueprint(optical2sar_bp)
```

```bash
curl -X POST http://127.0.0.1:5001/ -H "Content-Type: application/json" \
     -d '{"input_path": "D:/img/船.jpg", "output_path": "D:/out/sar.png"}'
# → {"success": true, "resultPath": "D:\\out\\sar.png"}
```

### YOLO 权重对接（新训练权重直接替换）

- 默认权重路径：`<脚本上级目录>/optical2sar/weights/optyolo/best.pt`，把新训练的 `best.pt` 覆盖到该路径即完成换模型（运行中替换自动重载）；
- 推理参数与旧版一致：`imgsz=1280, conf=0.25, iou=0.45, OBB 优先, 取置信度最高的 1 个目标`；
- 支持三类权重输出：旋转框(OBB) / 实例分割掩码 / 水平框(HBB)；
- 权重缺失或未检出时自动回退内置 Otsu 分割（`yolo_fallback="error"` 可改为报错）。

## 雷达参数速查（11 维，默认值）

| 参数 | 默认 | 含义 | 派生分辨率 |
|---|---|---|---|
| Hc | 4999 m | 平台高度 | — |
| thetaSQ / thetaSL | 0° / 45° | 斜视角 / 侧视角 | — |
| vx | 100 m/s | 平台速度 | — |
| D_rg / D_az | 2 m / 8 m | 天线孔径 | 方位分辨率 = D_az/2 = 4 m |
| PRF | 30 Hz | 脉冲重复频率 | — |
| T | 5 µs | 脉宽 | — |
| f0 | 1 GHz | 载频（L 波段） | λ ≈ 0.3 m |
| Br | 80 MHz | 发射带宽 | 距离分辨率 = c/2Br ≈ 1.87 m |
| fs | 100 MHz | 采样率 | — |

## 学习路线（给下一位接手的同学）

1. 读 [`docs/算法原理详解.md`](docs/算法原理详解.md) §1–§3 建立大图景；
2. 对照 §7 的公式逐步读 `optical_to_sar.py` 第 5 节（RDA 核心，每步都有公式注释）；
3. 用 `--preset clean --seed 7` 跑一张样图，再改 `--param` 观察参数影响；
4. 需要改接口/对接时看第 8 节（HTTP）与 README 的 YOLO 部分。

## 依赖

```
numpy, pillow                  # 必需
scipy                          # 可选：连通域评分增强（缺失自动降级）
ultralytics                    # 可选：服务端 YOLO 舰船检测
flask                          # 可选：HTTP 服务
```

## License / 免责声明

仅供仿真演示与算法研究使用。输出为模板驱动的 SAR 观感近似，不代表真实雷达观测数据，请勿用于真实情报判读场景。
