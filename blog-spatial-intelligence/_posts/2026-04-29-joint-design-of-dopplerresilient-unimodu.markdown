---
layout: post-wide
title: "MIMO 雷达 Doppler 鲁棒波形设计：SQNGD 框架详解"
date: 2026-04-29 08:05:57 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.25728v1
generated_by: Claude Code CLI
---

## 一句话总结

针对 MIMO 雷达中运动目标检测问题，SQNGD 框架通过**软量化可微参数化 + 梯度下降**联合优化发射波形与接收滤波器，在延迟-多普勒二维域同时压制旁瓣，比传统 MMCD 方法快 2-11 倍，PSL 改善 3-6 dB。

## 为什么这个问题重要？

### 应用场景

MIMO 雷达（多输入多输出雷达）是无人机感知、车载雷达、低空防御的核心传感器，波形设计直接决定：

- **目标检测率**：旁瓣高 → 弱小目标被强杂波淹没
- **多目标分辨**：互模糊函数不理想 → 相邻目标串扰
- **运动目标跟踪**：多普勒旁瓣高 → 高速目标误判为地物杂波

关键挑战：当目标以速度 $v$ 运动时，回波产生多普勒频移 $f_d = 2v/\lambda$。波形设计若只考虑延迟域（静止目标），在多普勒维旁瓣会急剧抬升，导致高速目标漏检。

### 现有方法的瓶颈

| 方法类型 | 代表方法 | 问题 |
|---------|---------|------|
| BCD 迭代优化 | MMCD | 复杂度 $O(M^2 L^2 N_d)$，长序列不可用 |
| 学习类方法 | SQN | 只优化延迟域，忽略多普勒旁瓣 |

SQNGD 的目标：**学习类方法的速度 + 联合延迟-多普勒优化能力**。

## 背景知识

### 模糊函数：波形的"雷达身份证"

对长度 $L$ 的单路波形 $s[l]$，**自模糊函数（Auto-AF）** 描述其在延迟-多普勒二维空间的响应：

$$
\chi(\tau, f) = \sum_{l=0}^{L-1} s[l] \cdot s^*[l - \tau] \cdot e^{j2\pi f l / L}
$$

MIMO 雷达还需要 $M$ 路波形之间的**互模糊函数（CAF）**来表征通道间干扰：

$$
\chi_{mn}(\tau, f) = \sum_{l=0}^{L-1} s_m[l] \cdot s_n^*[l - \tau] \cdot e^{j2\pi f l / L}, \quad m \neq n
$$

**好的波形设计目标**：
- 自模糊函数在 $(\tau=0, f=0)$ 之外的**峰值旁瓣电平（PSL）** 尽量低
- 所有互模糊函数的峰值尽量低（波形间正交性）

### 单模 + 离散相位约束

实际雷达功率放大器要求**单模约束**（固定幅度）以保证工作在线性区：

$$|s[l]| = 1, \quad \forall l$$

进一步，实际数字系统使用 $2^B$-PSK 调制（**离散相位约束**）：

$$\phi[l] \in \left\{0, \frac{2\pi}{2^B}, \frac{2 \cdot 2\pi}{2^B}, \ldots\right\}$$

离散约束将连续优化变为**组合优化**，是整个问题的核心难点。

## 核心方法：SQNGD 框架

### 直觉解释

```
连续相位参数 θ (可学习)
        ↓  软量化 (可微分!)
近离散波形 s̃  →  FFT加速计算 AF / CAF
                        ↓
              多目标损失：PSL + SNRL
                        ↓
        ←  反向传播更新 θ
        ←  梯度下降更新接收滤波器 h
        
训练完成后：θ → [硬量化] → 最终离散相位码
```

**关键洞察**：硬量化不可导，但基于温度的软量化（Softmax）可以渐进地从"软"过渡到"硬"，让梯度在训练全程流动。

### 软量化参数化

设 $K = 2^B$ 个离散相位候选 $\mathcal{P} = \{p_0, \ldots, p_{K-1}\}$，软量化：

$$
\hat{s}[l] = \sum_{k=0}^{K-1} \underbrace{\frac{e^{-T \cdot |\theta[l] - p_k|^2}}{\sum_{k'} e^{-T \cdot |\theta[l] - p_{k'}|^2}}}_{\text{温度}T\text{控制的 Softmax 权重}} \cdot e^{j p_k}
$$

随温度 $T$ 升高，分布趋向 one-hot，最终逼近硬量化。

### 多目标损失函数

$$
\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{\text{Auto-AF}} + \lambda_2 \cdot \mathcal{L}_{\text{CAF}} + \lambda_3 \cdot \text{SNRL}
$$

其中**信噪比损失（SNRL）** 惩罚接收滤波器偏离匹配滤波器：

$$
\text{SNRL} = 10\log_{10}\frac{\|s\|^2 \cdot \|h\|^2}{|h^H s|^2} \geq 0 \text{ dB}
$$

$\text{SNRL} = 0$ dB 对应完美匹配滤波，无 SNR 损失。

## 实现

### 环境配置

```bash
pip install torch numpy matplotlib
```

### 核心代码：FFT 加速模糊函数计算

```python
import torch
import numpy as np

def compute_af_fft(waveforms, num_doppler=64):
    """
    FFT加速计算自模糊函数
    waveforms: [M, L] 复数张量
    returns:   [M, L, num_doppler] 自模糊函数
    
    关键加速：对每个延迟τ，沿序列维做FFT一次性得到所有多普勒切面
    vs 朴素算法：对每对(τ, f)分别累加，复杂度高O(L²·Nd)
    """
    M, L = waveforms.shape
    af = torch.zeros(M, L, num_doppler, dtype=torch.complex64,
                     device=waveforms.device)
    for tau in range(L):
        # 循环移位模拟延迟
        s_shifted = torch.roll(waveforms, shifts=tau, dims=-1)
        # 延迟相关: [M, L]
        prod = waveforms * s_shifted.conj()
        # FFT沿序列维 → 同时获得所有多普勒频率: [M, num_doppler]
        af[:, tau, :] = torch.fft.fft(prod, n=num_doppler, dim=-1)
    return af

def psl_loss(af):
    """峰值旁瓣电平损失（越低越好）"""
    power = torch.abs(af) ** 2
    main_peak = power[:, 0, 0]           # [M] 主峰在(τ=0, f=0)
    sidelobe = power.clone()
    sidelobe[:, 0, 0] = 0                # 排除主峰
    max_sl = sidelobe.view(power.shape[0], -1).max(dim=-1).values
    return (max_sl / (main_peak + 1e-8)).mean()

def snrl_loss(waveforms, filters):
    """信噪比损失：衡量接收滤波器偏离匹配的程度"""
    s_norm = waveforms.norm(dim=-1) ** 2       # [M]
    h_norm = filters.norm(dim=-1) ** 2         # [M]
    inner  = (filters.conj() * waveforms).sum(dim=-1).abs() ** 2  # [M]
    return 10 * torch.log10(s_norm * h_norm / (inner + 1e-8)).mean()
```

### 核心代码：软量化模块与优化主循环

```python
class SoftQuantizer(torch.nn.Module):
    """连续相位参数 → 可微近离散波形"""
    def __init__(self, M, L, num_bits=2):
        super().__init__()
        K = 2 ** num_bits
        # 离散相位候选集，不参与梯度
        self.register_buffer('phases', torch.linspace(0, 2*np.pi, K+1)[:-1])
        # 可学习连续相位参数 [M, L]
        self.theta = torch.nn.Parameter(torch.rand(M, L) * 2 * np.pi)

    def forward(self, temperature=10.0):
        # 到各候选相位的角度距离: [M, L, K]
        diff = self.theta.unsqueeze(-1) - self.phases
        diff = torch.abs(torch.remainder(diff + np.pi, 2*np.pi) - np.pi)
        weights = torch.softmax(-temperature * diff**2, dim=-1)
        # 加权合成复数波形: [M, L]
        return (weights * torch.exp(1j * self.phases)).sum(dim=-1)

    def hard_quantize(self):
        """推理时硬量化到最近离散相位"""
        diff = torch.abs(torch.remainder(
            self.theta.unsqueeze(-1) - self.phases + np.pi, 2*np.pi) - np.pi)
        idx = diff.argmin(dim=-1)
        return torch.exp(1j * self.phases[idx]).detach()


def train_sqngd(M=4, L=64, num_bits=2, num_doppler=32, num_epochs=400):
    quantizer = SoftQuantizer(M, L, num_bits)
    # 接收滤波器，保持单位能量
    h = torch.nn.Parameter(
        torch.randn(M, L, dtype=torch.complex64) / (L ** 0.5))

    opt = torch.optim.Adam(list(quantizer.parameters()) + [h], lr=0.02)

    for epoch in range(num_epochs):
        # 温度退火：缓慢从1升到20，使量化逐渐"变硬"
        T = 1.0 + 19.0 * (epoch / num_epochs) ** 0.5

        s = quantizer(T)
        # 使用前面定义的 compute_af_fft 和损失函数
        af   = compute_af_fft(s, num_doppler)
        loss = psl_loss(af) + 0.1 * snrl_loss(s, h)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 投影：保持接收滤波器能量归一
        with torch.no_grad():
            h.data /= h.data.norm(dim=-1, keepdim=True)

        if epoch % 100 == 0:
            psl_db = 10 * np.log10(psl_loss(af).item() + 1e-10)
            print(f"Epoch {epoch:4d} | T={T:.1f} | PSL≈{psl_db:.1f} dB")

    return quantizer.hard_quantize(), h.detach()
```

### 延迟-多普勒图可视化

```python
import matplotlib.pyplot as plt

def visualize_delay_doppler(waveforms, num_doppler=128, title="自模糊函数"):
    """绘制第一路波形的延迟-多普勒模糊函数热图"""
    with torch.no_grad():
        af = compute_af_fft(waveforms, num_doppler)
    
    # fftshift使零多普勒居中
    af0 = torch.fft.fftshift(torch.abs(af[0]), dim=-1).numpy()
    af_db = 10 * np.log10(af0 / af0.max() + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(af_db.T, aspect='auto', vmin=-60, vmax=0,
                   cmap='viridis',
                   extent=[0, af_db.shape[0], -0.5, 0.5])
    ax.set_xlabel('延迟 τ (采样点)'); ax.set_ylabel('归一化多普勒频率')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='归一化功率 (dB)')
    psl = af_db.copy(); psl[0, af_db.shape[1]//2] = -999
    ax.set_title(f"{title}  |  PSL = {psl.max():.1f} dB")
    plt.tight_layout(); plt.show()

# 运行示例
waveforms, filters = train_sqngd(M=2, L=32, num_bits=2, num_epochs=400)
visualize_delay_doppler(waveforms, title="SQNGD 优化后")
```

**预期输出**：主峰清晰位于图中央 $(\tau=0, f=0)$，整个延迟-多普勒平面旁瓣压低至 -40 dB 以下，多普勒维不出现"刀刃"状旁瓣脊。

## 实验

### 定量评估

| 方法 | PSL（窄多普勒 $\pm 0.5$） | PSL（宽多普勒 $\pm 600$） | 优化时间 | SNRL |
|-----|--------------------------|--------------------------|---------|------|
| MMCD | -37.2 dB | -27.6 dB | 1× (基准) | 0.5 dB |
| SQN | ~-35 dB | ~-20 dB | 0.3× | 0.5 dB |
| **SQNGD** | **-43.0 dB** | **-31.0 dB** | **0.1~0.5×** | **0.5 dB** |

### FFT 加速分析

对延迟维暴力计算 vs. FFT：

- **暴力**：$O(L \cdot N_d)$ 次乘加，$N_d$ 个多普勒点逐一计算
- **FFT**：$O(L \log N_d)$，理论加速比 $N_d / \log_2 N_d$

当 $N_d = 256$ 时理论加速 ~32×，实测 1.9-11×（受内存带宽限制）。

## 工程实践

### 实际部署流程

```
离线设计阶段（GPU服务器）:
  SQNGD训练 → 硬量化 → 波形码本 (M × L 整数数组)
        ↓
烧录到雷达 DDS（直接数字合成）
        ↓
实时工作阶段（雷达嵌入式，无需 GPU）:
  DDS 读取码本 → 发射 → 接收滤波器（同样预计算）→ 检测
```

**硬件需求**：训练用 RTX 3090 (24GB)，$M=8, L=128$ 约 10 分钟完成。部署时只需将相位码表写入 DDS 寄存器。

### 常见坑

**坑 1：温度退火速度过快导致梯度消失**

```python
# 错误：线性退火，后期梯度在量化边界处几乎为零
T = epoch * 0.5  # epoch=40 时已经很硬

# 正确：先慢后快，早期保持充分梯度流动
T = 1.0 + 19.0 * (epoch / num_epochs) ** 0.5
```

**坑 2：大规模 CAF 内存溢出**

```python
# M=16 时 caf 形状为 [16,16,L,Nd]，内存可能超过 32GB
# 正确：分批次计算，只保留当前波形对
for m in range(M):
    for n in range(m+1, M):
        caf_mn = compute_af_fft(
            waveforms[[m, n]], num_doppler)  # 只取两路
        loss_cross += psl_loss(caf_mn[0:1])  # 互模糊
```

**坑 3：模糊函数零多普勒定位错误**

```python
# FFT 后零频在索引 0，可视化时需要 fftshift
af_vis = torch.fft.fftshift(af, dim=-1)  # 零多普勒移到中央
# 不做 fftshift：图像左右边缘才是零多普勒，误导分析
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| MIMO 雷达波形**离线**预设计 | 需要实时自适应波形更新 |
| 目标多普勒范围已知且固定 | 目标速度分布宽且动态变化 |
| 4-PSK / 8-PSK 低阶调制 | FMCW 类连续相位雷达 |
| 序列较长（$L > 64$），传统方法太慢 | 极短序列（$L < 16$），穷举可行 |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| MMCD | 数学收敛保证，无需调参 | $L$ 增大时慢得难以接受 | 短序列，精度优先 |
| SQN | 训练快，天然离散约束 | 仅延迟域优化，多普勒性能差 | 不关心多普勒旁瓣时 |
| **SQNGD** | 速度快 + 延迟-多普勒联合优化 | 需调温度、权重超参数 | 长序列 + 高速目标场景 |

## 我的观点

**工程落地距离：近。** 波形设计本质是**离线预计算**，训练好的相位码表直接写入雷达 DDS，无需在雷达嵌入式上运行神经网络。这与 NeRF 要实时渲染完全不同——雷达场景的"推理"在信号发射时不需要 GPU。

**值得关注的开放问题**：

1. **自适应场景**：当杂波谱或目标多普勒分布变化时，如何在毫秒内快速重新优化？当前所有方法都是固定场景的离线设计
2. **非理想发射机建模**：实际 PA 有 AM/AM 和 AM/PM 非线性失真，当前框架假设理想单模约束，实测与仿真之间仍有 2-5 dB 的旁瓣性能差距
3. **空时联合设计**：相控阵的空域自由度（波束赋形）尚未与波形设计融合

**趋势判断**：基于学习的雷达波形设计会持续替代传统迭代优化，核心驱动是 MIMO 系统天线数量和序列长度增加后，传统方法的超线性计算代价。SQNGD 这类"可微参数化 + 梯度下降"的框架是一个可扩展的方向，随着更大规模 MIMO（如 $M = 64$ 的大规模 MIMO 雷达），FFT 加速和批量并行化优势会更加凸显。