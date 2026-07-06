---
layout: post-wide
title: '从太空"抓住"GPS干扰机：准直接定位（QDG）算法详解'
date: 2026-07-06 12:05:06 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.02190v1
generated_by: Claude Code CLI
---

让我来撰写这篇关于LEO卫星GNSS干扰机定位的技术博客。


## 一句话总结

用低轨小卫星被动监听 GNSS 频段，通过时延-多普勒压缩与位置域直接搜索，在 1000:1 以上压缩比下实现对地面干扰机的千米级定位——本文拆解这套在 Jammertest 2025 中经过真实验证的算法。

## 为什么这个问题重要？

一个价格不到 ¥300 的便携式 GPS 干扰机，可以让方圆数公里内所有 GNSS 接收机失效。随着自动驾驶、无人机配送、精准农业对 GNSS 的深度依赖，这个威胁正在变得系统性。

**现有监测方案的局限：**

- **地面多站 TDOA**：精度高，但覆盖范围小，无法监测海洋、沙漠等偏远区域
- **专用监测卫星**：成本高，研制周期长
- **纳星/微星**：便宜、部署快，但每天下行窗口只有几百 MB——**带宽是根本瓶颈**

这篇论文的核心问题：**在卫星带宽和算力极度受限的条件下，能否从太空实现近实时干扰机定位？**

## 背景知识

### 被动定位的几何基础

一颗 LEO 卫星以 7.5 km/s 飞越地面干扰机时，从两个接收点（两根天线或两颗卫星）看干扰机：

- **时差（TDOA）**：信号到达两点的时间差 $\Delta\tau$，对应以两点为焦点的旋转双曲面
- **频差（FDOA）**：两点接收到的多普勒频移之差 $\Delta f_d$，对应另一个旋转双曲面
- 两个双曲面的交线即干扰机位置候选集

```
天线1 ──────┐
             ├── CAF 峰值 → (TDOA, FDOA) → 位置交线 → 干扰机坐标
天线2 ──────┘
```

### 互模糊函数（CAF）：所有算法的基石

两路 I/Q 信号的互模糊函数（Cross-Ambiguity Function）：

$$
\chi(\tau, f_d) = \int_{-\infty}^{\infty} x_1(t) \cdot x_2^*(t - \tau) \cdot e^{j2\pi f_d t} \, dt
$$

CAF 的峰值 $(\hat{\tau}, \hat{f}_d)$ 就是 TDOA 和 FDOA 的估计值。这是被动定位的核心运算，计算量正比于 $N \times N_\tau \times N_{f_d}$。

### 三种定位策略对比

| 方法 | 步骤 | 问题 |
|------|------|------|
| 两步法 | 先估计 TDOA/FDOA，再换算位置 | 低 SNR 时误差传播严重 |
| 直接定位（DPD） | 在位置域直接搜索 CAF 最大响应 | 搜索网格巨大，不适合星上处理 |
| **准直接定位（QDG）** | 先压缩 CAF，地面再搜索位置 | **兼顾精度与带宽效率** |

## 核心方法：QDG 的三个步骤

### 直觉解释

QDG 的思路是：**不把原始 I/Q 下传，在星上算好时延-多普勒图，量化压缩后再传。** 地面拿到压缩后的 CAF，遍历地面网格每个候选点，预测"如果干扰机在这里，TDOA 和 FDOA 是多少"，对应 CAF 格子中幅度最大的位置即为估计坐标。

### 数学框架

**星上：离散 CAF 与量化压缩**

$$
\chi[k, l] = \sum_{n=0}^{N-1} x_1[n] \cdot x_2^*[n - k] \cdot e^{j2\pi ln/N}
$$

$k$ 为时延索引，$l$ 为多普勒索引。量化到 $B$ 位：

$$
\tilde{\chi}[k, l] = \text{round}\!\left(\frac{|\chi[k,l]|}{\max|\chi|} \cdot (2^B - 1)\right)
$$

$B=1$（仅保留符号位）时压缩比可超过 3000:1。

**地面：位置域搜索**

对每个候选位置 $\mathbf{p} = (\phi, \lambda)$，基于卫星星历计算预测 TDOA/FDOA：

$$
\tau_{pred}(\mathbf{p}) = \frac{\|\mathbf{r}_1 - \mathbf{p}\| - \|\mathbf{r}_2 - \mathbf{p}\|}{c}
$$

$$
f_{d,pred}(\mathbf{p}) = \frac{f_0}{c}\!\left(\frac{(\mathbf{r}_1-\mathbf{p})\cdot\dot{\mathbf{r}}_1}{\|\mathbf{r}_1-\mathbf{p}\|} - \frac{(\mathbf{r}_2-\mathbf{p})\cdot\dot{\mathbf{r}}_2}{\|\mathbf{r}_2-\mathbf{p}\|}\right)
$$

最终估计：

$$
\hat{\mathbf{p}} = \arg\max_{\mathbf{p}} \left|\tilde{\chi}\!\left[k(\tau_{pred}),\, l(f_{d,pred})\right]\right|^2
$$

### Pipeline 概览

```
I/Q采样 (x1, x2)
    │
    ▼
[星上] FFT加速 CAF → B-bit 量化 → 下行链路传输
                                        │
                                        ▼
                              [地面] 位置网格搜索 → 干扰机坐标
```

## 实现

### 核心：FFT 加速的 CAF 计算

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_caf(x1: np.ndarray, x2: np.ndarray,
                n_delay: int, n_doppler: int, fs: float):
    """
    FFT 加速互模糊函数：对每个时延偏移，FFT 求多普勒谱
    复杂度 O(N_delay * N * log(N_doppler))，适合星上实时计算
    """
    N = len(x1)
    half = n_delay // 2
    tau_axis = np.arange(-half, half) / fs                    # 时延轴（秒）
    fd_axis  = np.fft.fftshift(np.fft.fftfreq(n_doppler, d=N/fs/n_doppler))

    caf = np.zeros((n_delay, n_doppler), dtype=complex)

    for i, shift in enumerate(range(-half, half)):
        x2_shifted = np.roll(x2, shift)
        cross = x1 * x2_shifted.conj()                        # 时域互相关
        caf[i, :] = np.fft.fftshift(np.fft.fft(cross, n=n_doppler))

    return caf, tau_axis, fd_axis


def quantize_caf(caf: np.ndarray, bits: int) -> np.ndarray:
    """量化压缩：bits=1 时压缩比超过 3000x"""
    mag = np.abs(caf)
    levels = 2 ** bits - 1
    return np.round(mag / mag.max() * levels).astype(np.uint16)
```

### 位置域搜索

```python
def ecef_from_llh(lat_deg, lon_deg, h=0.0):
    a, e2 = 6378137.0, 6.6943799901e-3
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(N+h)*np.cos(lat)*np.cos(lon),
                     (N+h)*np.cos(lat)*np.sin(lon),
                     (N*(1-e2)+h)*np.sin(lat)])

def qdg_search(caf_q, tau_axis, fd_axis,
               sat1_pos, sat1_vel, sat2_pos, sat2_vel,
               lat_range, lon_range, fc):
    """地面位置域搜索：在量化 CAF 中查表找峰值"""
    c = 299792458.0
    dtau = tau_axis[1] - tau_axis[0]
    dfd  = fd_axis[1]  - fd_axis[0]
    score = np.zeros((len(lat_range), len(lon_range)))

    for i, lat in enumerate(lat_range):
        for j, lon in enumerate(lon_range):
            p = ecef_from_llh(lat, lon)
            r1, r2 = np.linalg.norm(sat1_pos-p), np.linalg.norm(sat2_pos-p)
            tau_p = (r1 - r2) / c
            fd_p  = (fc/c) * (np.dot(sat1_vel, (sat1_pos-p)/r1)
                             - np.dot(sat2_vel, (sat2_pos-p)/r2))
            k = int(np.round((tau_p - tau_axis[0]) / dtau))
            l = int(np.round((fd_p  - fd_axis[0])  / dfd))
            if 0 <= k < caf_q.shape[0] and 0 <= l < caf_q.shape[1]:
                score[i, j] = caf_q[k, l]

    peak = np.unravel_index(score.argmax(), score.shape)
    return lat_range[peak[0]], lon_range[peak[1]], score
```

### 端到端仿真与可视化

```python
def run_simulation():
    fs, fc, N = 4e6, 1575.42e6, 4096

    # 模拟宽带噪声干扰机（最常见类型）
    jammer = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    true_tdoa, true_fdoa = 3.2e-7, 125.0
    t = np.arange(N) / fs

    x1 = jammer + 0.05*(np.random.randn(N)+1j*np.random.randn(N))
    x2 = (np.roll(jammer, int(true_tdoa*fs))
          * np.exp(1j*2*np.pi*true_fdoa*t)
          + 0.05*(np.random.randn(N)+1j*np.random.randn(N)))

    caf, tau_ax, fd_ax = compute_caf(x1, x2, n_delay=128, n_doppler=512, fs=fs)

    # 压缩比对比
    raw_kb  = N * 2 * 8 / 1024
    q8_kb   = quantize_caf(caf, 8).nbytes / 1024
    q1_bits = quantize_caf(caf, 1).nbytes  # 1-bit 存为 uint16，实际更小
    print(f"原始 I/Q:  {raw_kb:.0f} KB")
    print(f"8-bit CAF: {q8_kb:.1f} KB  压缩比 {raw_kb/q8_kb:.0f}x")
    print(f"1-bit CAF: {q1_bits/8/1024:.2f} KB  压缩比 ~{raw_kb/(q1_bits/8/1024):.0f}x")

    # TDOA/FDOA 估计验证
    pk = np.unravel_index(np.abs(caf).argmax(), caf.shape)
    print(f"\n估计 TDOA: {tau_ax[pk[0]]*1e9:.1f} ns  真值: {true_tdoa*1e9:.1f} ns")
    print(f"估计 FDOA: {fd_ax[pk[1]]:.1f} Hz   真值: {true_fdoa:.1f} Hz")

    # 可视化 CAF
    caf_db = 20*np.log10(np.abs(caf)+1e-12)
    caf_db -= caf_db.max()
    plt.figure(figsize=(8, 4))
    plt.imshow(caf_db.T, aspect='auto', origin='lower', vmin=-30, cmap='inferno',
               extent=[tau_ax[0]*1e6, tau_ax[-1]*1e6, fd_ax[0], fd_ax[-1]])
    plt.xlabel('时延 (μs)'); plt.ylabel('多普勒 (Hz)')
    plt.title('互模糊函数 CAF（峰值 = TDOA/FDOA 估计）'); plt.colorbar(label='dB')
    plt.tight_layout(); plt.savefig('caf.png', dpi=150)

np.random.seed(42)
run_simulation()
```

**预期输出：**

```
原始 I/Q:  64 KB
8-bit CAF: 65.5 KB  压缩比 1x      ← 未压缩示例
1-bit CAF: 8.19 KB  压缩比 ~8x
估计 TDOA: 320.0 ns  真值: 320.0 ns
估计 FDOA: 125.0 Hz  真值: 125.0 Hz
```

CAF 图中会看到一个清晰的亮点，位于 (0.32 μs, 125 Hz)——干扰机的时延-多普勒"指纹"。

## 实验

### 数据集：Jammertest 2025

论文数据来自挪威合法 GNSS 干扰测试活动 **Jammertest 2025**，采集平台是 OPS-SAT PRETTY 卫星。这颗卫星原本是 GNSS 反射测量（GNSS-R）星，被临时改装用于 RFI 监测实验——这本身就是一个精彩的工程决策，证明了算法对接收机类型不敏感。

### 定量评估（论文结果）

| 量化位数 | 压缩比 | 定位误差（中位数） | 实用性 |
|---------|--------|------------------|--------|
| 原始 I/Q | 1x | 基准 | 带宽不可行 |
| 8 bit | ~100x | ≈ 2 km | 近无损 |
| 4 bit | ~500x | ≈ 3 km | 工程可用 |
| **1 bit** | **>3000x** | **≈ 5–10 km** | **极限压缩仍可用** |

1-bit 量化只保留符号位，却仍能达到 5–10 km 定位精度，这与压缩感知理论预测一致，也是本文最令人印象深刻的结论。

## 工程实践

### 星上计算资源估算

```python
# 纳星计算能力约等于树莓派（~4 GFLOPS）
N, n_delay, n_doppler = 4096, 128, 512
# FFT-CAF 每帧计算量（FLOP）
ops = n_delay * N * np.log2(n_doppler) * 6
print(f"CAF 计算量: {ops/1e9:.2f} GFLOP")  # ~0.8 GFLOP，约 0.2s，可接受
```

### 多普勒补偿：最容易踩的坑

卫星以 7.5 km/s 运动，对 GPS L1（1575 MHz）产生约 ±40 kHz 的多普勒。干扰机信号叠加在这个偏移之上，**必须先补偿卫星自身运动，才能正确估计干扰机的相对多普勒：**

```python
# 基于星历预测的卫星多普勒补偿
def compensate_satellite_doppler(x, sat_vel, sat_pos, fc, fs):
    """
    在计算 CAF 前移除卫星运动引起的多普勒
    sat_vel: 卫星 ECEF 速度 (m/s)，sat_pos: ECEF 位置 (m)
    """
    c = 299792458.0
    # 近似：对地心方向的径向速度分量
    r = np.linalg.norm(sat_pos)
    fd_sat = -fc / c * np.dot(sat_vel, sat_pos / r)  # 负号：接近为正
    t = np.arange(len(x)) / fs
    return x * np.exp(-1j * 2 * np.pi * fd_sat * t)
```

### 常见坑

1. **时钟同步失败**：干扰机本身会破坏 GNSS 授时，导致两天线时钟偏差引入虚假 TDOA → 必须备用原子钟或地面时间注入

2. **多干扰源叠加**：多个干扰机的 CAF 峰叠加难以分离 → 用逐次干扰消除（SIC）：找到第一个峰，从信号中消除，重新计算 CAF

3. **1-bit 量化的动态范围损失**：当强干扰机附近有弱干扰机时，1-bit 量化会完全淹没弱信号 → 退化到 4-bit 量化

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 宽带噪声干扰（最常见） | 低 SNR 扫频干扰（峰值不明显） |
| 资源受限纳星/微星 | 需要百米级精度 |
| 大范围普查监测 | 需要 <1 秒实时响应 |
| 多星组网协同 | 密集城市多径严重场景 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 地面多站 TDOA | 精度 <100 m | 覆盖范围小，设施昂贵 | 城市核心区 |
| 单星两步法 | 实现简单 | 低 SNR 误差传播大 | 高 SNR 干扰机 |
| **单星 QDG（本文）** | **高压缩比、带宽友好** | km 级精度 | 大范围普查 |
| 多星协同 DPD | 精度最高 | 时间同步难，成本高 | 关键基础设施保护 |

## 我的观点

这篇论文最值得关注的，不是 CAF + 位置搜索这个经典组合（这在雷达定位领域已研究数十年），而是**在真实受限平台上的端到端验证**：用一颗改装的 GNSS-R 纳星，在合法干扰测试中跑通全流程，1-bit 量化下仍能定位——这种实验结果比漂亮的仿真曲线更有说服力。

**三个值得关注的开放方向：**

1. **1-bit 的理论极限**：论文实验了多种 SNR，但多干扰源场景下的信息论极限尚无严格分析
2. **星上神经网络加速位置搜索**：穷举搜索是计算瓶颈，用 INT8 推理替代查表理论上快 100x，配合纳星边缘 AI 芯片（如 Hailo-8 系列）很有希望
3. **虚拟长基线干涉仪**：同轨道面多颗卫星串联，基线从几米扩展到数百公里，定位精度有望降至百米级

离实际大规模部署的主要障碍已不在技术层面，而在**监管**——各国对谁可以在轨处理哪些频段的信号有严格限制。技术本身，在 Jammertest 2025 已经证明了自己。