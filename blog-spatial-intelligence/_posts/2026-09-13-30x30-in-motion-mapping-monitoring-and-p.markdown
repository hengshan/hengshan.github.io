---
layout: post-wide
title: "从卫星到三维:用星载激光雷达重建森林冠层做生物多样性监测"
date: 2026-09-13 08:03:09 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://medium.com/google-earth/30x30-in-motion-mapping-monitoring-and-protecting-latin-american-biodiversity-aaef9305be76?source=rss----a747a9e16c1c---4
generated_by: Claude Code CLI
---

## 一句话总结

要保护哥伦比亚瓦哈棕榈(Quindío wax palm)这样的森林生态系统，光靠卫星拍"照片"是不够的——真正有用的是把森林变成一个三维模型:知道每一片林冠有多高、哪里的树被砍了、变化发生在什么时候。这篇文章讲的是这套三维重建的技术底层：如何从星载激光雷达(GEDI)和数字高程模型(DEM)中提取森林冠层高度，并做大范围变化监测。

## 为什么这个问题重要?

Google Earth 团队在 2026 年 7 月报道的哥伦比亚保护区案例，是"30x30"全球生物多样性目标(到 2030 年保护 30% 的陆地和海洋)的一个缩影。这类保护区划定、监测、执法，本质上都依赖同一套空间智能基础设施:

- **划定边界**需要知道地形、植被覆盖类型的三维结构，而不只是二维颜色分类
- **持续监测**需要检测林冠高度的变化(选择性采伐往往不会让整片林子消失，而是逐步"掏空"树冠)
- **执法响应**需要在变化发生后尽快(而不是等到卫星影像肉眼可辨认出秃斑时)发出预警

传统方法主要依赖光学影像的二维语义分割(森林 vs 非森林)，这类方法对**渐进式退化**(forest degradation，不同于突然的 forest loss)非常不敏感——因为颜色和纹理变化很小，但三维结构已经发生了显著变化。这正是三维视觉方法的价值所在。

## 背景知识

在遥感和地球观测领域，"三维"的表示方式和我们熟悉的室内 3D 视觉(点云、体素、NeRF)有相似之处，但尺度和数据获取方式完全不同：

| 表示方式 | 数据来源 | 分辨率/覆盖 | 特点 |
|---------|---------|------------|------|
| 数字表面模型 DSM | 立体摄影测量、机载 LiDAR | 米级，局部 | 包含地表所有物体高度(含树冠) |
| 数字地形模型 DTM | 机载 LiDAR 穿透植被 | 米级，局部 | 只有裸地地形 |
| 冠层高度模型 CHM | DSM - DTM | 米级 | 树冠高度，本文核心 |
| 星载全波形激光雷达 | NASA GEDI(装在国际空间站上) | 25m 足迹，全球稀疏采样 | 单点精度高但覆盖稀疏 |
| 光学多光谱影像 | Sentinel-2, Landsat | 10-30m，全球稠密 | 二维，无直接高度信息 |

关键前置知识：GEDI(Global Ecosystem Dynamics Investigation)是一台安装在国际空间站上的全波形激光雷达，它向地面发射激光脉冲，记录反射回来的完整波形——波形里最早的强反射峰通常来自树冠顶端，最后一个峰来自地面，二者的时间差换算成高度就是冠层高度。这和自动驾驶用的 LiDAR 原理相同，只是尺度从米级变成了全球尺度。

## 核心方法

### 直觉解释

想象一束激光垂直射向森林：如果森林很密，大部分光子会被树冠反射，只有很少能穿透到地面；如果森林稀疏或已被砍伐，大部分光子会直接打到地面。GEDI 记录的"全波形"，本质上就是这束光在不同高度上被反射的能量分布直方图。我们要做的，就是从这个直方图里找出"树冠顶端"和"地面"两个关键位置。

### 数学细节

给定 GEDI 波形能量函数 $E(h)$（$h$ 为相对高度），地面峰值高度为 $h_{ground}$，则相对高度指标 RH（Relative Height）定义为：

$$
RH_p = h_p - h_{ground}, \quad \text{其中} \sum_{h=h_{ground}}^{h_p} E(h) = p \times \sum_{h=h_{ground}}^{h_{top}} E(h)
$$

即 $RH_p$ 是累积能量达到 $p\%$ 时对应的高度。业界最常用的冠层高度指标是 $RH_{98}$（累积能量达到 98% 处的高度），因为 $RH_{100}$ 容易受噪声影响。

栅格化的 CHM 更简单：

$$
CHM(x, y) = DSM(x, y) - DTM(x, y)
$$

### Pipeline 概览

```
GEDI 原始波形 → 去噪/基线校正 → 地面峰检测 → RH 高度计算 → 稀疏点云
Sentinel-2 影像 → 云掩膜 → NDVI/时序特征 → 变化检测 → 变化候选区
   ↓
稀疏 GEDI 高度点 + 稠密 Sentinel-2 特征 → 插值/回归 → 全覆盖冠层高度图
```

## 实现

### 环境配置

```bash
pip install rasterio numpy matplotlib scipy
# 如需访问 Google Earth Engine 做行星尺度分析
pip install earthengine-api
```

### 核心代码：GEDI 全波形提取冠层高度

```python
import numpy as np
from scipy.signal import find_peaks

def extract_canopy_height(waveform, height_axis, ground_search_range=5.0):
    """从单条 GEDI 全波形中提取 RH98 冠层高度
    waveform: 归一化能量数组
    height_axis: 与 waveform 对应的高度值(米)，地面附近为0
    """
    # 平滑降噪，避免噪声被误判为峰
    smoothed = np.convolve(waveform, np.ones(5) / 5, mode="same")

    # 地面峰通常是波形末端(高度最低处)附近能量最强的峰
    peaks, properties = find_peaks(smoothed, height=0.02)
    near_ground = peaks[height_axis[peaks] < ground_search_range]
    ground_idx = near_ground[np.argmax(smoothed[near_ground])] if len(near_ground) else np.argmin(height_axis)
    h_ground = height_axis[ground_idx]

    # 从地面峰以上的能量做累积分布，求 RH98
    above_ground = smoothed[ground_idx:]
    cumulative = np.cumsum(above_ground) / np.sum(above_ground)
    rh98_idx = np.searchsorted(cumulative, 0.98)
    h_top = height_axis[ground_idx + rh98_idx]

    return h_top - h_ground  # 冠层高度

# 合成一条示例波形：地面尖峰 + 冠层宽峰
h = np.linspace(0, 40, 200)
canopy_signal = 0.6 * np.exp(-((h - 25) ** 2) / (2 * 6 ** 2))
ground_signal = 1.0 * np.exp(-((h - 2) ** 2) / (2 * 0.8 ** 2))
waveform = canopy_signal + ground_signal + np.random.normal(0, 0.01, size=h.shape)

canopy_height = extract_canopy_height(waveform, h)
print(f"估计冠层高度: {canopy_height:.1f} 米")
```

### 核心代码：栅格化 CHM

```python
import rasterio
import numpy as np

def compute_chm(dsm_path, dtm_path, out_path):
    """DSM 减 DTM 得到冠层高度模型，两者需已配准到同一网格"""
    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        dsm = dsm_src.read(1).astype(np.float32)
        dtm = dtm_src.read(1).astype(np.float32)
        profile = dsm_src.profile

    chm = dsm - dtm
    chm[chm < 0] = 0        # 配准误差可能导致负值，裁剪到合理范围
    chm[chm > 80] = np.nan  # 超过80米大概率是异常值(如云、建筑物)

    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(chm, 1)
    return chm
```

### 3D 可视化

```python
import matplotlib.pyplot as plt

def visualize_chm_3d(chm):
    """把冠层高度栅格渲染成三维地形，直观展示林冠起伏"""
    ys, xs = np.mgrid[0:chm.shape[0], 0:chm.shape[1]]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xs, ys, np.nan_to_num(chm), cmap="YlGn", linewidth=0)
    ax.set_zlabel("冠层高度 (m)")
    ax.set_title("森林冠层三维重建")
    plt.savefig("chm_3d.png", dpi=150)
```

## 实验

### 数据集说明

- **GEDI L2A/L2B**：NASA 提供的全波形激光雷达产品，覆盖南北纬 51.6° 之间，2019 年至今，通过 [NASA Earthdata](https://earthdata.nasa.gov) 免费下载，但足迹稀疏(约每 60m 一个采样点，轨道间距上百米)
- **Copernicus DEM GLO-30**：全球 30m 分辨率地形数据，可作为 DTM 参考，欧空局免费提供
- **Sentinel-2 L2A**：10-20m 分辨率光学影像，5 天重访周期，用于云掩膜后的 NDVI 时序变化检测
- 数据获取难度：GEDI/Sentinel-2 都是开放数据，但**云覆盖**在热带雨林地区是最大障碍，往往需要数月的影像堆叠才能拼出一张"干净"的合成图

### 定量评估

| 方法 | 冠层高度 RMSE | 空间覆盖 | 更新频率 | 单次成本 |
|-----|-------------|---------|---------|---------|
| 星载 GEDI (RH98) | ~3-4 m | 稀疏点(全球) | 持续采样但稀疏 | 免费 |
| 机载 LiDAR | ~0.5-1 m | 局部飞行区域 | 按需(通常年度) | 高(需包机) |
| 立体摄影测量(SfM) | ~2-5 m | 依赖影像覆盖 | 灵活 | 中等 |
| GEDI + Sentinel-2 融合回归 | ~4-6 m | 全球连续 | 可周期更新 | 免费(计算成本) |

### 定性结果

在实际的哥伦比亚保护区场景中，融合方法的典型表现是：能清晰识别出选择性采伐造成的"局部塌陷"(冠层高度在小范围内骤降但周边仍是完整林冠)，而单纯的 NDVI 变化检测在这种情况下几乎没有信号——这正是三维方法相比传统二维遥感的核心优势。失败案例通常出现在云雾常年覆盖的山地云雾林，光学影像长期缺失有效观测。

## 工程实践

### 实际部署考虑

- **规模问题**：单个保护区做本地处理没问题，但"30x30"目标覆盖的是大陆尺度，本地下载/处理 PB 级影像不现实。Google Earth Engine 这类云端地理空间计算平台的价值就在于把计算下推到数据所在地，避免数据搬运
- **GEDI 稀疏性**：单条轨道的高度点无法直接生成连续地图，实践中通常训练一个回归模型(如随机森林/梯度提升)，用 GEDI 高度点作为标签、Sentinel-2/雷达纹理特征作为输入，插值出全覆盖的冠层高度图
- **没有实时性要求**：这类监测通常是天到周级别的批处理任务，不需要考虑帧率和推理延迟，但需要考虑批处理集群的调度和存储成本

### 数据采集建议

- GEDI 采样有季节和轨道限制，需要跨年累积足够的足迹密度
- DSM/DTM 融合时务必检查坐标系和空间分辨率是否严格配准，否则 CHM 会出现大量假阳性

### 常见坑

1. **DSM/DTM 时间不一致导致假变化** → 确保两期数据获取时间接近，或至少排除季节性因素(落叶林冬夏差异巨大)
2. **云/云影被误判为地表变化** → 光学影像处理前必须做严格的云掩膜(如 Sentinel-2 的 SCL 波段)，否则变化检测全是噪声
3. **密林中 DTM 存在"空洞"** → 激光穿透率不足会导致地面点缺失，需要用周边地形插值填补，不能直接用 NaN 参与计算

## 什么时候用/不用?

| 适用场景 | 不适用场景 |
|---------|-----------|
| 大范围、低频率的生态监测 | 需要厘米级精度的单木测量 |
| 有稳定卫星重访覆盖的开阔区域 | 常年云雾覆盖且缺乏机载 LiDAR 补充的区域 |
| 检测渐进式森林退化 | 需要实时(分钟级)响应的场景 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| GEDI 星载 LiDAR | 全球覆盖、免费、精度较高 | 足迹稀疏，需插值 | 全球/区域尺度冠层高度基线 |
| 机载 LiDAR | 精度最高，覆盖连续 | 成本高，覆盖范围有限 | 局部精细监测、科研样地 |
| 光学摄影测量(SfM/多视立体) | 成本低、易获取 | 无法穿透植被得到真实地面 | 无人机小范围测绘 |
| NeRF / 3D Gaussian Splatting | 高保真新视角合成 | 依赖密集多视角影像，无法用于卫星尺度稀疏观测 | 室内/小场景重建，不适用于本文场景 |

值得说明的是，NeRF/3DGS 这类方法在这个问题上基本不适用——它们假设的是密集多视角光学观测，而卫星遥感的观测几何(近乎正下方俯视、稀疏时间采样)完全不满足这个前提。

## 我的观点

生物多样性监测正在从"人工判读卫星图"转向"自动化的三维结构监测"，这个趋势会持续加速，因为渐进式森林退化只有在三维层面才能被早期发现。但离真正的实时执法闭环还有距离——目前的瓶颈不在算法，而在于**地面真值的获取成本**(需要实地样地测量校验模型精度)和**跨传感器融合的一致性**(不同卫星、不同轨道周期的数据如何拼接成时间一致的产品)。一个值得关注的开放问题是：能否用类似隐式神经表示的思路，对稀疏、异构、多时相的遥感观测做统一建模，而不是像现在这样为每个任务单独训练回归模型——这可能是未来几年"地球尺度空间智能"的一个重要方向。