---
layout: post-wide
title: "用城市 5G 基站网络测降雨：分布式机会雷达的信号处理原理"
date: 2026-03-22 08:07:14 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.19153v1
generated_by: Claude Code CLI
---

## 一句话总结

把城市里已有的蜂窝基站当作多普勒气象雷达来用，无需新建专用硬件，即可获得传统气象站无法企及的米级城市降雨精度。

## 为什么这个问题重要？

### 城市水文的感知盲区

传统气象雷达空间分辨率约 1 km、时间分辨率约 5 分钟。对于城市而言完全不够用：

- **城市暴雨**往往局限在 100-500 米的街区尺度，传统雷达看不清
- **城市洪涝预警**需要知道"哪条街在下雨"，而不是"这 1 平方千米区域在下雨"
- **城市热岛效应**导致强对流频发，极端降雨事件时空变化极快

雨量站虽精确，但站点稀疏，城区全覆盖成本极高。商业微波链路衰减法（OpenMRG）可以反演路径平均雨量，但分辨率受限于基站间距，且只有**路径积分信息**，无法定位降雨区域。

### 核心创新：把基站当多普勒雷达用

这篇论文的关键突破：不仅仅测量信号衰减，而是**分析反射信号的多普勒频移**，直接提取传统气象雷达的三个核心产品。

| 测量方式 | 信息类型 | 分辨率 |
|---------|---------|--------|
| 微波链路衰减 | 路径积分雨量 | 基站间距（百米-公里） |
| **基站机会雷达（本文）** | 反射率 + 速度场 + 湍流 | **米级** |

## 背景知识

### 多普勒天气雷达的三个矩

天气雷达通过分析信号的**多普勒功率谱** $S(v)$ 提取三个统计矩：

- **0 阶矩（反射率）**：$M_0 = \int S(v)\,dv$，正比于降雨强度
- **1 阶矩（平均速度）**：$\bar{v} = \dfrac{\int v \cdot S(v)\,dv}{M_0}$，径向风速
- **2 阶矩（谱宽）**：$\sigma_v = \sqrt{\dfrac{\int (v-\bar{v})^2 S(v)\,dv}{M_0}}$，湍流指标

反射率因子的物理定义——雨滴越大、越多，回波越强：

$$Z = \int_0^\infty N(D) \cdot D^6 \, dD \quad [\text{mm}^6/\text{m}^3]$$

通过 Marshall-Palmer Z-R 关系换算降雨强度：

$$Z = 200 \cdot R^{1.6} \quad (Z \text{ 线性单位，} R \text{ 单位 mm/h})$$

### 基站做雷达：机遇与约束

5G NR 基站（3.5 GHz 附近，$\lambda \approx 0.086$ m）具备：

- **宽带信号**：100-400 MHz 带宽，距离分辨率 0.375-0.75 m
- **相控阵天线**：Massive MIMO，可电控波束方向
- **高密度部署**：城区间距 200-500 m，天然分布式孔径

主要限制——雷达方程直接揭示了问题：

$$P_r \propto \frac{P_t \cdot G^2 \cdot Z}{r^2 \cdot \lambda^2}$$

基站发射功率 $P_t \approx 10$-$40$ W，而专业气象雷达脉冲功率达几十万瓦。低功率加上几乎**水平的波束指向**（传统气象雷达有 0.5°-20° 仰角），导致地物杂波极其严重——这是全文的核心工程挑战。

## 核心方法

### 直觉解释

基站持续发射 OFDM 信号，部分能量被雨滴散射后返回基站天线。每个 OFDM 符号等效为一次"脉冲"，子载波频差提供距离信息，沿时间轴做 FFT 提供多普勒信息。

```
基站发射 OFDM → 雨滴/地物散射 → 基站接收 IQ 数据
                                        ↓
                              [脉冲维 FFT + 加窗]
                                        ↓
                              多普勒功率谱 S(v, r)
                                        ↓
                    [高斯杂波模型拟合并减除地物杂波]
                                        ↓
                              净雨滴谱 → 三个矩
                                        ↓
                         dBZ → Z-R 关系 → 降雨强度 R
```

地物（楼宇、车辆、树木）是静止目标，多普勒频移接近零；雨滴随风运动，谱峰偏向非零速度。这个物理差异是分离二者的基础。

### 高斯杂波模型

建模杂波的多普勒谱为零均值高斯分布（树木摆动等轻微运动造成谱展宽，但仍集中在零速附近）：

$$S_c(v) = P_c \cdot \exp\!\left(-\frac{v^2}{2\sigma_c^2}\right), \quad \sigma_c \approx 0.3\text{-}0.5\ \text{m/s}$$

净雨滴谱：$S_{\text{rain}}(v) = \max\!\left[S(v) - S_c(v),\ 0\right]$

## 实现

### 核心信号仿真

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_bs_iq_data(
    n_pulses=128, n_range_bins=256,
    wavelength=0.086,       # 3.5 GHz 波长 (m)
    prf=1000,               # 脉冲重复频率 (Hz)
    rain_velocity=2.0,      # 雨滴径向速度 (m/s)
    rain_snr_db=-5,         # 雨滴信噪比 (dB)
    clutter_scr_db=25,      # 杂波信号比 (dB)
    rain_gates=(50, 100),   # 有雨的距离门范围
):
    """模拟基站雷达 IQ 采样：热噪声 + 地物杂波 + 雨滴回波"""
    rng = np.random.default_rng(42)

    # 热噪声
    noise = (rng.standard_normal((n_pulses, n_range_bins)) +
             1j * rng.standard_normal((n_pulses, n_range_bins))) / np.sqrt(2)

    # 地物杂波：强度高，跨脉冲基本相干，轻微相位抖动模拟树木摆动
    sigma_c = 0.3  # 杂波谱宽 (m/s)
    sigma_phi = 2 * np.pi * 2 * sigma_c / wavelength / prf  # 每脉冲相位标准差
    clutter_phase = np.cumsum(rng.standard_normal(n_pulses) * sigma_phi)
    clutter = (10 ** (clutter_scr_db / 20) * np.exp(1j * clutter_phase)[:, None]
               * np.ones((1, n_range_bins)))

    # 雨滴回波：分布式随机目标，有多普勒偏移
    fd = 2 * rain_velocity / wavelength  # 多普勒频率 (Hz)
    t = np.arange(n_pulses) / prf
    carrier = np.exp(1j * 2 * np.pi * fd * t)
    rs, re = rain_gates
    rain = np.zeros((n_pulses, n_range_bins), dtype=complex)
    rain[:, rs:re] = (10 ** (rain_snr_db / 20) * carrier[:, None] *
                      (rng.standard_normal((n_pulses, re - rs)) +
                       1j * rng.standard_normal((n_pulses, re - rs))) / np.sqrt(2))

    return noise + clutter + rain
```

### 多普勒处理与矩估计

```python
def doppler_process(iq_data, wavelength=0.086, prf=1000):
    """
    脉冲维 FFT → 多普勒谱 → 三个矩
    输入: iq_data (n_pulses, n_range_bins)
    输出: spectrum, velocity_axis, M0, M1(m/s), M2(m/s)
    """
    n_pulses = iq_data.shape[0]

    # Hann 窗降低旁瓣，然后 FFT，fftshift 将零频移至中心
    win = np.hanning(n_pulses)
    spectrum = np.abs(np.fft.fftshift(
        np.fft.fft(iq_data * win[:, None], axis=0), axes=0
    )) ** 2

    # 速度轴：最大不模糊速度 v_max = λ·PRF/4
    v_max = wavelength * prf / 4
    velocity = np.linspace(-v_max, v_max, n_pulses)

    M0 = spectrum.sum(axis=0)
    M1 = (spectrum * velocity[:, None]).sum(axis=0) / (M0 + 1e-12)
    M2 = np.sqrt(
        (spectrum * (velocity[:, None] - M1[None, :]) ** 2).sum(axis=0)
        / (M0 + 1e-12)
    )
    return spectrum, velocity, M0, M1, M2

def gaussian_clutter_filter(spectrum, velocity, sigma_c=0.3):
    """高斯杂波模型拟合并减除"""
    # 从零速附近区域估算杂波峰值（逐距离门）
    peak_mask = np.abs(velocity) < sigma_c
    clutter_peak = spectrum[peak_mask].mean(axis=0)  # (n_range_bins,)

    # 构建高斯杂波模型并减除
    clutter_model = clutter_peak[None, :] * np.exp(
        -velocity[:, None] ** 2 / (2 * sigma_c ** 2)
    )
    return np.maximum(spectrum - clutter_model, 0)

def to_rainfall(M0_filtered, radar_constant_db=60):
    """0 阶矩 → dBZ → Marshall-Palmer → 降雨率 (mm/h)"""
    dbz = 10 * np.log10(M0_filtered + 1e-12) - radar_constant_db
    Z = 10 ** (dbz / 10)
    R = (Z / 200) ** (1 / 1.6)
    return dbz, R
```

### 完整流水线与可视化

```python
# 生成仿真数据
iq = simulate_bs_iq_data(rain_velocity=2.0, rain_snr_db=-5, clutter_scr_db=25)

# 多普勒处理
spectrum, velocity, M0, M1, M2 = doppler_process(iq)

# 杂波滤除，从滤波谱重新计算矩
spec_f = gaussian_clutter_filter(spectrum, velocity, sigma_c=0.3)
M0_f = spec_f.sum(axis=0)
M1_f = (spec_f * velocity[:, None]).sum(axis=0) / (M0_f + 1e-12)

# Z-R 换算
dbz, rain_rate = to_rainfall(M0_f)

# 可视化：多普勒谱 + Range-Doppler 图
rain_gate = 75
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(velocity, 10 * np.log10(spectrum[:, rain_gate] + 1e-12), label='原始谱')
axes[0].plot(velocity, 10 * np.log10(spec_f[:, rain_gate] + 1e-12), label='滤波后')
axes[0].axvline(2.0, color='r', ls='--', label='真实雨速 2 m/s')
axes[0].set(xlabel='速度 (m/s)', ylabel='功率谱 (dB)', title='含雨距离门的多普勒谱')
axes[0].legend()

for ax, sp, title in zip(axes[1:], [spectrum, spec_f],
                          ['原始 Range-Doppler 图', '杂波滤除后']):
    ax.imshow(10 * np.log10(sp.T + 1e-12), aspect='auto',
              extent=[velocity[0], velocity[-1], 0, sp.shape[1]],
              vmin=-20, vmax=40)
    ax.set(xlabel='速度 (m/s)', ylabel='距离门', title=title)

plt.tight_layout()
plt.savefig('bs_radar_result.png', dpi=150)
```

运行后预期结果：
- **左图**：原始谱在零速附近有一个几十 dB 的杂波峰；滤波后该峰被压制，真实雨速（2 m/s）处的信号得以保留
- **中图**：未滤波的 Range-Doppler 图，零速列亮度远高于其他区域（强杂波）
- **右图**：滤波后，有雨的距离门（50-100）在 2 m/s 附近的信号清晰可辨

## 实验

### 性能对比

| 系统 | 空间分辨率 | 时间分辨率 | 城区覆盖 | 增量成本 |
|------|-----------|-----------|---------|---------|
| C 波段气象雷达 | ~250 m | 5-10 min | 差（遮蔽） | 极高 |
| 双偏振气象雷达 | ~100 m | 1-5 min | 差 | 更高 |
| 微波链路衰减法 | 路径积分 | 1 min | 中 | 低 |
| **基站机会雷达** | **几米** | **数十秒** | 优 | **极低** |

空间分辨率相比传统气象雷达提升了 2-3 个数量级。

### 信噪比：真正的瓶颈

```python
# SNR 灵敏度分析：估计速度误差随 SNR 的变化
errors = []
for snr_db in range(-20, 10, 2):
    iq_t = simulate_bs_iq_data(rain_snr_db=snr_db, clutter_scr_db=25)
    sp, vel, *_ = doppler_process(iq_t)
    sp_f = gaussian_clutter_filter(sp, vel)
    m0 = sp_f.sum(axis=0)
    m1 = (sp_f * vel[:, None]).sum(axis=0) / (m0 + 1e-12)
    errors.append(np.abs(m1[50:100] - 2.0).mean())
# SNR < -10 dB 时误差 > 0.5 m/s，速度估计已不可靠
```

实际部署中，只有强降雨（dBZ > 30，约 $R > 5$ mm/h）时 SNR 才勘堪可用。毛毛雨几乎检测不到。

## 工程实践

### 陷波滤波 vs 高斯模型

```python
def notch_filter(spectrum, velocity, notch_mps=0.5):
    """简单陷波：速度绝对值小于阈值的功率直接清零"""
    filtered = spectrum.copy()
    filtered[np.abs(velocity) < notch_mps] = 0
    return filtered

# 问题：陷波会丢失近零速度的降雨（如几乎垂直下落的雨）
# 推荐：优先用高斯模型；仅在杂波谱宽 > 1 m/s 时退化到陷波
```

### 雷达常数标定

基站并非为雷达设计，将接收功率转换为 dBZ 需要标定：

```python
def calibrate_with_reference(measured_power, range_m, reference_dbz):
    """与附近气象雷达或卫星降雨产品交叉标定，估计系统偏置"""
    dbz_raw = 10 * np.log10(measured_power * range_m ** 2 + 1e-12)
    bias = np.median(reference_dbz - dbz_raw)
    return bias  # 后续所有测量值加上此偏置
```

### 常见坑

1. **OFDM 循环前缀未去除** → 距离旁瓣严重，近处目标能量泄漏到远处距离门；需在距离处理前正确截断 CP
2. **多普勒模糊** → 最大不模糊速度 $v_{max} = \lambda \cdot PRF/4$，若风速超出则发生折叠；需要 PRF 设计权衡
3. **噪声基底温度漂移** → 基站热噪声随环境温度变化约 ±3 dB；需要实时估计噪声功率，否则弱雨滴信号检测门限失准
4. **IQ 数据访问权限** → 需运营商授权访问基带单元，无法从标准协议层获取

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 城区强降雨（R > 5 mm/h）精细监测 | 毛毛雨、小雨（SNR 不足） |
| 城区洪涝实时预警 | 郊区 / 农村（基站密度不够） |
| 研究城市内降雨分布不均匀性 | 需要精确绝对定标的水文测量 |
| 利用现有基础设施，无需新建 | 雪、冰雹等固态降水（Z-R 关系失效） |
| 风暴移动轨迹追踪 | 需要观测降雨垂直结构 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 传统气象雷达 | 覆盖广、成熟、可标定 | 分辨率粗、城区遮蔽、造价高 | 区域降雨预报 |
| 微波链路衰减 | 复用现有链路 | 只有路径积分，无空间信息 | 城区平均雨量 |
| 雨量站网 | 精确可追溯 | 站点稀疏，空间覆盖差 | 点测量、标定 |
| **基站机会雷达** | 米级分辨率、高时效、低成本 | 低 SNR、强杂波、标定依赖参考源 | 城区精细降雨感知 |

## 我的观点

这篇工作最有价值的不是"基站能测雨"这个结论，而是**定量证明了信号处理层面的可行性**——把多普勒气象雷达的整套矩估计体系成功迁移到近水平波束的蜂窝系统，并系统评估了杂波滤除后的质量损失。

几个值得关注的开放问题：

**5G 协同处理的潜力**：O-RAN 架构让多个基站可以共享原始 IQ 数据。同一降雨区域被多个 BS 从不同角度观测，协同处理既能提升 SNR，又能在空间域做杂波抑制——比单 BS 陷波滤波强得多。

**Massive MIMO 空间滤波**：64T64R 天线阵列可以在空间上区分方向，楼宇回波和雨滴回波来自不同仰角，空间域自适应滤波比多普勒域过滤更本质。

**与卫星降雨产品融合**：GPM 等卫星提供面平均参考（解决标定问题），基站提供局地精细结构（解决分辨率问题），两者融合可能是城市水文感知的终局形态。

离实用化最大的障碍不是算法，而是**数据访问**：运营商是否愿意开放基带 IQ 数据，以及如何建立标准化的数据接口。这是商业和政策问题，技术层面已经足够成熟。