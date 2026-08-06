---
layout: post-wide
title: "时序 InSAR 地表形变分析：从干涉图网络到毫米级位移时间序列"
date: 2026-08-06 12:02:42 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.04915v1
generated_by: Claude Code CLI
---

## 一句话总结

时序 InSAR（TS-InSAR）通过堆叠多幅 SAR 干涉图、反演地表形变时间序列，精度可达毫米量级——但大气噪声、相位解缠错误和网络构建策略会让结果一塌糊涂。DefoEye 把 GMTSAR 的复杂流程包装成 Python 接口，加入干涉图网络剪枝和锚点校正，是当前开源 TS-InSAR 工具链中少见的端到端实现。

---

## 背景：为什么 TS-InSAR 这么难用？

InSAR（合成孔径雷达干涉测量）的基本思想很简单：两幅 SAR 图像的相位差包含了地表高程和形变信息。但实际使用中有三个令人头疼的问题：

**问题 1：相位是缠绕的（wrapped）**
SAR 测量的是微波相位，范围是 $[0, 2\pi)$。真实形变可能跨越多个波长，需要"解缠"才能还原真实位移。解缠算法在低相干区域（植被、水体）极易出错，而且错误会蔓延。

**问题 2：大气延迟淹没信号**
水蒸气分布不均导致的大气相位延迟，可以轻松伪造数厘米的"形变"。Sentinel-1 的 C 波段（5.6 cm 波长）对大气尤其敏感。

**问题 3：单幅干涉图太脆弱**
任何一幅干涉图都可能因为时间去相干、大气异常或轨道误差而报废。时序方法（TS-InSAR）的本质是**用统计冗余抗噪**：构建干涉图网络，联合反演时间序列。

现有工具（StaMPS、MintPy、GMTSAR）各有局限：商业软件贵、GMTSAR 需要手动 C-shell 命令、MintPy 缺乏完整的网络剪枝。DefoEye 的定位是：GMTSAR 的 Python 封装 + 网络剪枝 + 解缠锚点校正，补上了关键缺口。

---

## 算法原理

### 直觉解释

把每次 Sentinel-1 过境想成一次"拍照"。两次拍照之间，地面如果下沉了，相位就会变化。我们有 $N$ 个时间点、$M$ 幅干涉图，形成一个网络：

```
t1 --- t2 --- t3 --- t4
  \   / \   / 
   t2   t3  
```

短基线干涉图相干性好，但只有短基线覆盖整个时序会漏掉长期趋势。网络剪枝的任务是：去掉低相干的"烂"干涉图，保留能约束时序的好图。

### 数学推导

第 $k$ 幅干涉图（连接时间点 $i$ 和 $j$）的观测相位：

$$
\phi_k = \frac{4\pi}{\lambda}(d_j - d_i) + \phi_k^{atm} + \phi_k^{noise}
$$

其中 $d_j, d_i$ 是视线方向（LOS）形变，$\lambda$ 是波长（Sentinel-1 约 5.6 cm）。

SBAS（Small Baseline Subset）方法把问题写成线性系统：

$$
\mathbf{A} \mathbf{v} = \boldsymbol{\phi}
$$

- $\mathbf{A}$ 是 $M \times (N-1)$ 的设计矩阵，$A_{ki} = \Delta t_i$（时间间隔）
- $\mathbf{v}$ 是各时间段的平均形变速率
- $\boldsymbol{\phi}$ 是解缠后的干涉图相位观测

当网络连通时用最小二乘，网络有子集时用奇异值分解（SVD）或 L1 正则化。

### 与其他方法的关系

| 方法 | 核心思想 | 优点 | 局限 |
|------|---------|------|------|
| PS-InSAR | 识别稳定散射体 | 城市精度高 | 农村/植被区稀疏 |
| SBAS | 小基线网络反演 | 时空覆盖均匀 | 解缠错误传播 |
| DefoEye | SBAS + 网络剪枝 + 锚点 | 端到端自动化 | 依赖 GMTSAR |

---

## 实现

### 最小可运行版本：干涉图网络构建与剪枝

```python
import numpy as np
import networkx as nx
from itertools import combinations
from dataclasses import dataclass

@dataclass
class Acquisition:
    date: str          # YYYYMMDD
    b_perp: float      # 垂直基线（相对参考景，米）

def build_interferogram_network(
    acquisitions: list[Acquisition],
    max_temporal_days: int = 180,
    max_spatial_baseline: float = 150.0,  # 米
    min_coherence: float = 0.3,
    coherence_map: dict = None,           # {(i,j): mean_coherence}
) -> nx.Graph:
    """构建小基线干涉图网络并剪枝"""
    G = nx.Graph()
    
    for i, acq in enumerate(acquisitions):
        G.add_node(i, date=acq.date, b_perp=acq.b_perp)
    
    for (i, a1), (j, a2) in combinations(enumerate(acquisitions), 2):
        dt = abs(int(a2.date) - int(a1.date))  # 简化的日期差计算
        db = abs(a2.b_perp - a1.b_perp)
        
        if dt > max_temporal_days * 10000 or db > max_spatial_baseline:
            continue  # 超出基线阈值，跳过
        
        coh = coherence_map.get((i, j), 0.5) if coherence_map else 0.5
        
        if coh >= min_coherence:
            G.add_edge(i, j, dt=dt, db=db, coherence=coh)
    
    # 确保网络连通：若有孤立节点，强制连接最近邻
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for comp in components[1:]:
            node = min(comp)  # 取孤立子图的一个节点
            # 找主图中时间最近的节点
            main_nodes = list(components[0])
            nearest = min(main_nodes, key=lambda n: abs(n - node))
            G.add_edge(node, nearest, dt=999, db=0, coherence=0.3)
    
    return G
```

### 核心算法：SBAS 时间序列反演

```python
def sbas_inversion(
    interferograms: list[tuple[int, int]],  # (master_idx, slave_idx)
    unwrapped_phases: np.ndarray,           # shape: (M, pixels)
    acquisition_dates: list[int],           # YYYYMMDD 整数列表
    reg_weight: float = 1e-2,              # 时间平滑正则化强度
) -> np.ndarray:
    """
    SBAS 反演：从 M 幅干涉图 → N-1 个时间段速率 → 累积形变时间序列
    返回 shape: (N, pixels)，单位与输入相同（通常是毫米）
    """
    N = len(acquisition_dates)
    M = len(interferograms)
    P = unwrapped_phases.shape[1]  # 像素数
    
    # 计算时间间隔（年）
    def date_to_year(d):
        d = str(d)
        return int(d[:4]) + (int(d[4:6]) - 1) / 12 + int(d[6:]) / 365
    
    years = [date_to_year(d) for d in acquisition_dates]
    
    # 构建设计矩阵 A（M × N-1）
    A = np.zeros((M, N - 1))
    for k, (i, j) in enumerate(interferograms):
        # 干涉图 (i,j)：phase = v[i]*dt[i] + ... + v[j-1]*dt[j-1]
        for t in range(i, j):
            A[k, t] = years[t + 1] - years[t]  # 该时段的时间长度
    
    # 添加时间平滑正则化（抑制速率突变）
    L = np.zeros((N - 2, N - 1))
    for i in range(N - 2):
        L[i, i] = 1; L[i, i + 1] = -1
    
    A_reg = np.vstack([A, reg_weight * L])
    b_reg = np.vstack([unwrapped_phases, np.zeros((N - 2, P))])
    
    # 最小二乘求解（SVD 处理网络不连通情况）
    v, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)  # shape: (N-1, P)
    
    # 积分得到累积形变
    dt = np.array([years[t + 1] - years[t] for t in range(N - 1)])
    displacement = np.cumsum(v * dt[:, None], axis=0)  # shape: (N-1, P)
    
    # 在第一个时间点之前插入零（参考时刻无形变）
    return np.vstack([np.zeros((1, P)), displacement])
```

### 解缠锚点校正

解缠后的相位可能含有整数 $2\pi$ 倍的偏差，锚点校正是关键 trick：

```python
def anchor_unwrapped_phase(
    unwrapped: np.ndarray,   # shape: (M, rows, cols)
    anchor_mask: np.ndarray, # shape: (rows, cols)，稳定参考区域为 True
) -> np.ndarray:
    """
    将每幅干涉图的解缠相位对齐到参考区域均值为 0
    anchor_mask 通常选取已知稳定的基岩区或 GNSS 站点周边
    """
    corrected = unwrapped.copy()
    for k in range(len(unwrapped)):
        anchor_values = unwrapped[k][anchor_mask]
        if anchor_values.size == 0:
            continue
        # 去除异常值后取均值
        q25, q75 = np.percentile(anchor_values, [25, 75])
        valid = anchor_values[(anchor_values >= q25) & (anchor_values <= q75)]
        corrected[k] -= valid.mean()
    return corrected
```

### 关键 Trick（没有就跑不起来）

**1. 大气延迟估计与去除**

```python
def remove_atmospheric_ramp(phase_2d: np.ndarray) -> np.ndarray:
    """最简单的大气校正：拟合线性倾斜面并去除"""
    rows, cols = phase_2d.shape
    y, x = np.mgrid[0:rows, 0:cols]
    valid = ~np.isnan(phase_2d)
    
    # 线性最小二乘：phase = a*x + b*y + c
    A = np.column_stack([x[valid], y[valid], np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_2d[valid], rcond=None)
    
    ramp = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    return phase_2d - ramp
```

注意：线性倾斜面只是近似，真实大气是非线性的。ERA5 气象再分析数据校正效果更好，但需要额外下载。

**2. 相干性加权**

SBAS 反演时应该给高相干像素更高权重，否则噪声像素会污染结果：

```python
# 在 sbas_inversion 中加入相干性加权
weights = coherence_stack.reshape(M, -1)  # (M, P)
A_weighted = A * weights.mean(axis=1, keepdims=True)  # 简化版
```

**3. 参考点选择**

- 不要选在形变区域内
- 避免植被覆盖区域（时间去相干）
- 最好有附近的 GNSS 站可以验证
- 多个参考点取中位数，比单点鲁棒

---

## 实验与验证

### 与 GNSS 对比

```python
import matplotlib.pyplot as plt

def validate_against_gnss(
    insar_ts: np.ndarray,    # (N,) 某像素的时间序列，单位 mm
    dates: list,             # datetime 列表
    gnss_dates: list,
    gnss_los: np.ndarray,    # GNSS 投影到 LOS 方向的形变，mm
) -> dict:
    """计算 InSAR 与 GNSS 的 RMSE 和 Pearson 相关系数"""
    from scipy import interpolate, stats
    
    # 插值 InSAR 到 GNSS 时间点
    t_insar = np.array([(d - dates[0]).days for d in dates])
    t_gnss = np.array([(d - dates[0]).days for d in gnss_dates])
    
    f = interpolate.interp1d(t_insar, insar_ts, bounds_error=False, fill_value=np.nan)
    insar_at_gnss = f(t_gnss)
    
    valid = ~np.isnan(insar_at_gnss) & ~np.isnan(gnss_los)
    rmse = np.sqrt(np.mean((insar_at_gnss[valid] - gnss_los[valid]) ** 2))
    r, _ = stats.pearsonr(insar_at_gnss[valid], gnss_los[valid])
    
    return {"rmse_mm": rmse, "pearson_r": r}

# DefoEye 论文报告的验证结果（参考值）
results = {
    "Bologna, Italy": {"rmse": 4.3, "r": 0.95},
    "Gotland, Sweden": {"rmse": 8.7, "r": 0.78},
    "Houston, USA":   {"rmse": 11.9, "r": 0.63},
    "Karaj, Iran":    {"rmse": 4.8, "r": 0.98},  # 与其他工具对比
}
```

Houston 的 RMSE 偏高（11.9 mm）且相关系数偏低（0.63），值得注意——可能是石油开采导致的快速非线性沉降，或者大气噪声较强。论文没有详细分析这一点，是个遗憾。

### 与其他工具对比

| 工具 | 开源 | 端到端 | 网络剪枝 | 并行 | 易用性 |
|------|------|--------|---------|------|--------|
| GMTSAR | ✓ | ✗ | ✗ | 部分 | 低 |
| MintPy | ✓ | 部分 | ✓ | ✓ | 中 |
| StaMPS | 部分 | ✗ | ✓ | ✗ | 低 |
| DefoEye | ✓ | ✓ | ✓ | ✓ | 高 |

---

## 调试指南

### 常见问题

**1. 时间序列完全不动（所有像素趋势为零）**

先检查解缠是否成功：
- 输出相位是否全是 0 或 NaN？
- 相干性图是否合理（城区应该 > 0.7）？
- 参考点选在了形变中心？（改到稳定区域）

**2. 时间序列呈阶梯状跳变**

典型的相位解缠错误。解决方案：
- 提高相干性阈值，排除低质量干涉图
- 检查跳变发生的日期对应哪幅干涉图
- 对该干涉图单独可视化，定位解缠失败区域

**3. 所有像素出现同步季节性振荡**

几乎一定是大气延迟没去干净。检查：
- 振荡幅度 > 10 mm 且周期约 1 年 → 季节性大气
- 用 ERA5 或 GACOS 做大气校正
- 检查参考点是否在高海拔区（大气效应更强）

**4. 边缘区域突然形变异常**

解缠的相位连续性在图像边缘常常断裂。用 `anchor_mask` 排除边缘像素。

### 超参数调优

| 参数 | 推荐值 | 敏感度 | 建议 |
|------|--------|--------|------|
| 最大时间基线 | 120-180 天 | 高 | 先用 180 天，若相干性差缩短到 60 天 |
| 最大空间基线 | 100-200 m | 中 | Sentinel-1 临界基线约 5 km，200 m 很保守 |
| 最小相干阈值 | 0.25-0.4 | 高 | 植被区用 0.25，城区可用 0.4 |
| 时间平滑权重 | 1e-3 到 1e-1 | 中 | 从 1e-2 开始，太大会抹掉真实信号 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 地面沉降监测（矿区、城市） | 快速形变（同震位移 > 1 m，相位缠绕严重） |
| 滑坡早期预警（缓慢蠕变） | 热带雨林区（时间去相干严重） |
| 地热/水文储层监测 | 需要米级精度的工程测量 |
| 火山形变研究 | 高海拔积雪区（季节性去相干） |

---

## 我的观点

DefoEye 填补了一个真实的工程缺口：GMTSAR 的处理质量不错，但手动流程让大量用户望而却步。把它包装成 Python + 自动化流水线，是正确的工程决策。

但几个值得警惕的地方：

**1. Houston 验证结果需要解释**。RMSE 11.9 mm、相关系数 0.63 在 TS-InSAR 社区里属于偏弱的结果，论文没有深入分析原因。是算法局限还是 GNSS 本身有问题？

**2. 大气校正是软肋**。论文没有明确说使用了哪种大气校正方法。这是 TS-InSAR 结果质量最关键的因素之一，缺乏说明让结果可重复性存疑。

**3. 工具链锁定**。依赖 GMTSAR 意味着 DefoEye 继承了 GMTSAR 的所有限制。如果未来 SAR 社区迁移到其他处理引擎（如 ISCE3），DefoEye 的适用性会受限。

值不值得用？如果你的需求是 Sentinel-1 的标准沉降监测、GMTSAR 本地环境已搭好，DefoEye 能帮你省去大量手动步骤。如果你需要更精细的大气校正或者 PS-InSAR 分析，MintPy 的生态更成熟。

InSAR 这条路没有捷径：大气、解缠、参考点，每一关都要认真对待，工具再自动化也代替不了对数据的理解。