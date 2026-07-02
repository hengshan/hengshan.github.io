---
layout: post-wide
title: "5G/6G 一体化感知通信（ISAC）：Rician 信道下的波束成形性能极限"
date: 2026-07-02 12:05:17 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.00912v1
generated_by: Claude Code CLI
---

## 一句话总结

ISAC（Integrated Sensing and Communication）让基站用同一套天线和波形**同时**完成雷达探测与无线通信，本文从信息论角度推导了 Rician 衰落信道下感知与通信的中断概率极限，揭示了视距（LoS）分量如何打破传统 Rayleigh 分析框架。

## 为什么这个问题重要？

### 频谱资源的零和博弈正在终结

传统系统中，雷达占雷达频段，通信占通信频段，井水不犯河水。但 6G 时代的目标是让每个基站既是通信节点，又是环境感知节点——探测无人机、追踪车辆、室内定位。

**当前瓶颈**：

- 独立部署的感知和通信系统：频谱浪费、设备冗余
- 简单频分复用：牺牲各自峰值性能
- ISAC：**共享波形**同时携带数据和雷达信号，理论上"1+1 > 2"

### Rician 信道为什么值得单独分析？

大多数 ISAC 理论假设 Rayleigh 衰落（完全散射）。但现实中——

- 毫米波基站与用户之间通常有直射路径
- 无人机、车辆等目标有强反射主瓣
- 城市微基站：LoS 主导信道

Rician 的确定性 LoS 分量引入了**角度相关项**，让感知和通信出现非单调的 K 因子效应——这是 Rayleigh 分析完全看不到的现象。

## 背景知识

### MIMO 波束成形的空间直觉

均匀线阵（ULA）的导向矢量是 ISAC 的基础。$M$ 根天线、半波长间距：

$$\mathbf{a}(\theta) = \frac{1}{\sqrt{M}} \left[1,\ e^{j\pi\sin\theta},\ \ldots,\ e^{j(M-1)\pi\sin\theta}\right]^T$$

**几何意义**：$\mathbf{a}(\theta)$ 是方向 $\theta$ 的"空间频率"。当 $M$ 足够大时，不同角度的导向矢量近似正交——这是空间分辨率和 ISAC 可行性的根基。

### Rician 信道模型

用户信道 $\mathbf{h} \in \mathbb{C}^M$：

$$\mathbf{h} = \underbrace{\sqrt{\frac{K}{K+1}} \mathbf{a}(\theta_u)}_{\text{确定性 LoS 分量}} + \underbrace{\sqrt{\frac{1}{K+1}} \tilde{\mathbf{h}}}_{\text{随机散射分量}}$$

- $K$：Rician K 因子，$K \to 0$ 退化为 Rayleigh，$K \to \infty$ 变为纯 LoS
- $\tilde{\mathbf{h}} \sim \mathcal{CN}(\mathbf{0}, \mathbf{I}_M)$：散射分量

### 两种波束成形策略

**SJB（子空间联合波束成形）**：波束向量在用户和目标导向矢量张成的子空间内搜索最优解：

$$\mathbf{w}_{\text{SJB}} = \alpha\, \mathbf{a}(\theta_u) + \beta\, \mathbf{a}(\theta_t)$$

**LB（线性波束成形）**：发射信号分为独立的通信流和感知流：

$$\mathbf{x} = \mathbf{w}_c x_c + \mathbf{w}_s x_s, \quad x_c \perp x_s$$

LB 更灵活，但引入了**雷达自干扰**（sensing self-interference）问题。

### 感知性能度量：CRB

克拉美罗界给出目标角度 $\theta_t$ 估计方差的下界：

$$\text{FIM}(\theta_t) = \frac{2P}{\sigma^2} \left| \dot{\mathbf{a}}(\theta_t)^H \mathbf{w} \right|^2, \quad \text{CRB}(\theta_t) = \frac{1}{\text{FIM}(\theta_t)}$$

**感知中断概率**：CRB 超过精度门限 $\epsilon$ 时视为"感知中断"：

$$P_{out}^{sens} = P\!\left(\text{CRB}(\theta_t) > \epsilon\right)$$

## 核心方法

### 直觉：感知与通信的天然矛盾

通信希望波束对准用户（最大化 SNR）；雷达希望波束对准目标（最大化 Fisher 信息）。当方向相近时两者共赢，方向差 90° 时冲突最大：

```
角度差 ≈  0°: 通信 ★★★  感知 ★★★  (天赐良机)
角度差 ≈ 45°: 通信 ★★   感知 ★★   (折中区间)
角度差 ≈ 90°: 通信 ★★★  感知 ★    (纯通信代价)
```

SJB 通过调整 $\alpha, \beta$ 寻找 Pareto 最优点；LB 直接两路独立波束，代价是引入干扰。

### 高功率下的关键差异

对 LB 无 DPC 方案，用户接收的 SINR 为：

$$\text{SINR}_{\text{LB}} = \frac{P_c \left|\mathbf{h}^H \mathbf{w}_c\right|^2}{P_s \left|\mathbf{h}^H \mathbf{w}_s\right|^2 + \sigma_n^2}$$

分母中的 $P_s|\cdot|^2$ 与 $P_c$ 同比例增长，SINR **不再随功率无限提升**——这就是 LB 无 DPC 的干扰受限问题。

### Pipeline 概览

```
输入: M, K因子, θ_u, θ_t, 发射功率P
  │
  ├─ 生成 Rician 信道 h（Monte Carlo）
  │
  ├─ SJB: 最优 α,β 调整（子空间内优化）
  │    └─ 计算 SINR + CRB
  │
  ├─ LB: w_c=MRT, w_s=a(θ_t)（独立设计）
  │    └─ 计算 SINR（含自干扰）+ CRB
  │
  └─ 统计中断概率: #{事件} / N_trials
```

## 实现

### 信道模型与 CRB 核心实现

```python
import numpy as np

def steering_vector(theta, M, d=0.5):
    """ULA 导向矢量，半波长间距"""
    k = np.arange(M)
    return np.exp(1j * 2 * np.pi * d * k * np.sin(theta)) / np.sqrt(M)

def steering_vector_deriv(theta, M, d=0.5):
    """导向矢量对角度的偏导，用于 FIM 计算"""
    k = np.arange(M)
    return 1j * 2 * np.pi * d * k * np.cos(theta) * steering_vector(theta, M, d)

def generate_rician_channel(M, K, theta_u):
    """生成 Rician 信道向量"""
    a_los = steering_vector(theta_u, M)
    h_scatter = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
    return (np.sqrt(K / (K + 1)) * a_los +
            np.sqrt(1 / (K + 1)) * h_scatter)

def compute_crb(w, theta_t, M, P, sigma2=1.0):
    """目标角度估计的 CRB（单目标远场）"""
    a_dot = steering_vector_deriv(theta_t, M)
    fim = 2 * P / sigma2 * np.abs(w.conj() @ a_dot) ** 2
    return 1.0 / (fim + 1e-15)

def design_lb_beamformers(a_los, theta_t, M, P_c, P_s):
    """LB：MRT 通信波束 + 导向矢量感知波束"""
    w_c = a_los.conj() / (np.linalg.norm(a_los) + 1e-15)
    w_s = steering_vector(theta_t, M)
    return np.sqrt(P_c) * w_c, np.sqrt(P_s) * w_s
```

### 中断概率 Monte Carlo 仿真

```python
def simulate_outage(M, K, theta_u, theta_t,
                    P_total, gamma_th, epsilon_crb,
                    power_split=0.7, N=20000):
    """同时仿真 LB 和 SJB 的通信/感知中断概率"""
    P_c, P_s = power_split * P_total, (1 - power_split) * P_total
    a_los = steering_vector(theta_u, M)
    a_t   = steering_vector(theta_t, M)
    w_c, w_s = design_lb_beamformers(a_los, theta_t, M, P_c, P_s)

    # SJB：简化版等权子空间合并
    w_sjb = np.sqrt(P_total) * (a_los.conj() + a_t) / np.linalg.norm(a_los + a_t)

    counts = {"lb_comm": 0, "lb_sens": 0, "sjb_comm": 0, "sjb_sens": 0}

    for _ in range(N):
        h = generate_rician_channel(M, K, theta_u)

        # LB 评估
        sinr_lb = np.abs(h.conj() @ w_c) ** 2 / (np.abs(h.conj() @ w_s) ** 2 + 1.0)
        crb_lb  = compute_crb(w_c + w_s, theta_t, M, P_total)
        if sinr_lb < gamma_th:    counts["lb_comm"] += 1
        if crb_lb  > epsilon_crb: counts["lb_sens"] += 1

        # SJB 评估
        sinr_sjb = np.abs(h.conj() @ w_sjb) ** 2
        crb_sjb  = compute_crb(w_sjb, theta_t, M, P_total)
        if sinr_sjb < gamma_th:    counts["sjb_comm"] += 1
        if crb_sjb  > epsilon_crb: counts["sjb_sens"] += 1

    return {k: v / N for k, v in counts.items()}
```

### K 因子效应分析

```python
import matplotlib.pyplot as plt

M, theta_u, theta_t = 16, np.deg2rad(30), np.deg2rad(60)
gamma_th, epsilon_crb, P_total = 5.0, 1e-3, 10.0

K_values = [0.01, 0.1, 1, 5, 10, 50, 100]
results = [simulate_outage(M, K, theta_u, theta_t,
                           P_total, gamma_th, epsilon_crb)
           for K in K_values]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for ax, keys, title in [
    (ax1, ("lb_comm", "sjb_comm"), "通信中断概率 vs Rician K 因子"),
    (ax2, ("lb_sens", "sjb_sens"), "感知中断概率 vs Rician K 因子"),
]:
    for label, key in zip(["LB（无DPC）", "SJB"], keys):
        ax.semilogx(K_values, [r[key] for r in results], marker='o', label=label)
    ax.set(xlabel="K 因子", ylabel="中断概率", title=title)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("isac_outage_vs_K.png", dpi=150)
```

**预期结果**：通信中断曲线随 K 增大持续下降（LoS 增强稳定性）；感知中断曲线对 K 变化不敏感，这正是论文的核心实证——**K 因子对通信的影响远大于感知**。

## 实验

### 高功率缩放定律（工程核心结论）

| 方案 | 高功率行为 | 根本原因 |
|------|-----------|---------|
| SJB | 中断概率 → 0 | 无自干扰，功率直接转化为 SNR |
| LB + DPC | 中断概率 → 0 | DPC 预消除已知雷达干扰 |
| **LB（无 DPC）** | **趋于定值（干扰受限）** | 分母干扰项与功率同比增长 |

这个结论直接影响工程选型：**没有 DPC 时，增加发射功率无法持续改善 LB 的通信质量**。

### K 因子的非单调行为

```
K 极小（近 Rayleigh）: 通信依赖分集增益，感知靠波束增益
K 增大（LoS 增强）:   通信 SINR 直接提升，感知 CRB 变化平缓
K 极大（近纯 LoS）:   信道趋于确定，干扰可预测，DPC 效果最佳
```

论文揭示：在某些 K 值区间通信中断概率出现非单调下降再回升的现象，这是 Rayleigh 框架下永远看不到的。

## 工程实践

### 实际部署考虑

- **SJB 优化复杂度**：最优 $\alpha, \beta$ 需要求解 SINR-CRB 权衡的二阶锥规划（SOCP），延迟约 5~20 ms
- **LB 实现简单**：两路波束独立设计，适合实时场景，但高功率下有性能上限
- **DPC 的强假设**：需要精确的干扰信道 CSI，在高移动性场景中获取困难

### 常见坑

**坑 1：未归一化导向矢量导致功率计算错误**

```python
# 错误：直接用 exp() 未归一化
w = np.exp(1j * np.pi * np.arange(M) * np.sin(theta))  # 功率 = M，不是 1

# 正确：除以 sqrt(M)
w = np.exp(1j * np.pi * np.arange(M) * np.sin(theta)) / np.sqrt(M)
```

**坑 2：LB 高功率场景误用 SJB 的性能上限分析**

```python
# 验证是否处于干扰受限区间
def check_interference_limited(P_range, lb_comm_outages):
    delta = np.diff(lb_comm_outages)
    # 若高功率区间 delta ≈ 0，说明已进入干扰受限区
    return np.all(np.abs(delta[-3:]) < 0.01)
```

**坑 3：目标角度先验信息假设过强**

论文假设基站知道目标角度 $\theta_t$ 以计算 CRB。实际需要先做粗扫描：

```python
def coarse_angle_scan(received_signal, angles_grid, M):
    """用导向矢量内积扫描获取初始角度估计"""
    powers = [np.abs(received_signal.conj() @ steering_vector(a, M))**2
              for a in angles_grid]
    return angles_grid[np.argmax(powers)]
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 毫米波 LoS 主导（K > 1） | 富散射室内（K ≈ 0，Rayleigh 分析足够） |
| 静态或慢速目标 | 高速移动目标（多普勒未建模） |
| 单用户单目标（理论分析） | 多用户多目标（干扰矩阵指数增长） |
| 有数字基带 DPC 能力 | 纯模拟波束成形架构 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 独立雷达+通信 | 各自最优 | 双倍频谱/硬件 | 频谱充裕，预算充足 |
| OFDM-ISAC | 兼容 5G NR 标准 | PAPR 高，感知精度受限 | 现有系统升级 |
| **SJB（本文）** | 理论最优，无自干扰 | 优化复杂，需精确 CSI | 理论分析基准 |
| **LB + DPC（本文）** | 强 LoS 下通信最优 | DPC 实现依赖精确信道 | 毫米波固定接入 |
| **LB 无 DPC（本文）** | 实现最简单 | 高功率干扰受限 | 低功率低复杂度场景 |

## 我的观点

这篇论文做了一件有意义的事：把 Rician ISAC 分析从"推广已知结论"推进到了"需要全新分析框架"的层次。LoS 分量打破了 Rayleigh 下的对称性，产生了感知与通信对 K 因子截然不同的敏感度——这对 6G 毫米波部署有直接指导价值。

**离实用的三个主要差距**：

1. 单用户单目标假设过于理想——实际 6G 是多用户多目标，波束管理复杂度指数上升
2. DPC 的 CSI 假设太强——高移动性场景中获取精确干扰信道本身就是未解问题
3. 感知与通信的 QoS 联合优化框架尚未给出——论文分别分析两个中断概率，未提供统一的权衡设计方法

**值得关注的方向**：随机几何（Stochastic Geometry）与 ISAC 的结合，以及大规模 MIMO（$M > 100$）下的渐近分析。当 $M \to \infty$ 时，导向矢量完全正交，感知与通信的空间冲突理论上可以消失——这可能是 ISAC 从理论走向工程实用的关键跨越。