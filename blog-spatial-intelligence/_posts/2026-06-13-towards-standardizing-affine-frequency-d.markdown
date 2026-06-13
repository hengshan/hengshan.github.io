---
layout: post-wide
title: 'AFDM：用"啁啾"子载波征服双选择性信道'
date: 2026-06-13 08:05:17 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.13416v1
generated_by: Claude Code CLI
---

## 一句话总结

AFDM（仿射频分复用）将数据调制在线性调频（chirp）子载波上，通过离散仿射傅里叶变换在高速移动信道中实现全分集增益，同时天然兼容雷达感知，是 5G-Advanced 及 6G 的强力候选波形。

---

## 为什么这个问题重要？

高铁（350 km/h）、低轨卫星（LEO）、水下声学通信——这些场景的信道同时受到两个破坏性因素的夹击：

- **多径延迟**（建筑/海底反射）：信道在频域上选择性衰落
- **多普勒频移**（高速运动）：信道在时域上快速变化

两者叠加形成**双选择性信道**（doubly-selective channel），而现有 OFDM 在此场景下存在根本性缺陷。

OFDM 的问题：子载波是固定频率的正弦波，多普勒频移会把能量"泄漏"到相邻子载波——这就是**载间干扰（ICI）**。ICI 导致误码率出现"平台效应"，无论怎么提高发射功率，BER 都降不下去。

---

## 背景知识

### 双选择性信道模型

离散时域信道的数学表达：

$$y[n] = \sum_{p=1}^{P} h_p \cdot x[n - \ell_p] \cdot e^{j2\pi \nu_p n / N} + w[n]$$

其中 $\ell_p$ 是第 $p$ 条路径的延迟（采样点数），$\nu_p$ 是归一化多普勒频移，$P$ 是路径数。

### 三种多载波方案一览

| 方案 | 子载波形状 | 双选信道表现 | 实现复杂度 | ISAC 友好性 |
|-----|-----------|------------|-----------|------------|
| OFDM | 纯正弦 | 差（ICI） | 最低（单次 FFT） | 一般 |
| OTFS | 延迟-多普勒域 | 好 | 中（二维变换） | 好 |
| AFDM | 线性调频（chirp） | 好（全分集） | 低（FFT + 点乘） | 极好 |

---

## 核心方法

### 直觉：为什么 chirp 能抗多普勒？

OFDM 子载波是**单频正弦**，一个 Doppler 频移 $\Delta f$ 会让能量泄漏到所有相邻子载波（ICI 弥散）。

AFDM 的子载波是**线性调频**（频率随时间线性扫描）。一条运动中的散射体对 chirp 信号的影响，在仿射频域中等价于**精确的整数平移**，而不是弥散式泄漏——因此可以精确建模，精确均衡。

```
OFDM 在双选信道：              AFDM 在双选信道：
频域                           仿射频域
 │  ○ ● ● ● ○ ○               │  ○ ○ ● ○ ○ ○
 │  ↑ICI扩散到所有子载波        │    ↑仅平移到固定位置
 └──────────────              └──────────────
```

### 离散仿射傅里叶变换（DAFT）

DAFT 由两个参数定义：$c_1$（预啁啾）和 $c_2$（后啁啾）：

$$X[k] = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi\left(c_1 n^2 + \frac{kn}{N} + c_2 k^2\right)}$$

分解为三个可高效计算的步骤：

$$x[n] \xrightarrow{\times e^{-j2\pi c_1 n^2}} \tilde{x}[n] \xrightarrow{\text{DFT}} \tilde{X}[k] \xrightarrow{\times e^{-j2\pi c_2 k^2}} X[k]$$

**关键参数选取——全分集条件：**

$$c_1 = \frac{2Q+1}{2N}, \quad Q \in \mathbb{Z}^{\geq 0}$$

满足此条件时，任意两条路径 $(\ell_i, \nu_i) \neq (\ell_j, \nu_j)$ 在仿射频域中的投影严格不重叠，系统对 $P$ 条路径实现 **$P$ 阶分集增益**。

### Pipeline

```
发送端
  数据符号 X[k]
    → 后啁啾 × exp(j2π c₂ k²)
    → IDFT
    → 预啁啾 × exp(j2π c₁ n²)
    → 添加循环前缀（CP）
    → 发送

接收端
  去 CP
    → 去预啁啾 × exp(-j2π c₁ n²)
    → DFT
    → 去后啁啾 × exp(-j2π c₂ k²)
    → 信道均衡（利用带状矩阵结构）
    → 恢复 X̂[k]
```

---

## 实现

### DAFT 变换核心

```python
import numpy as np

class AFDMModem:
    """AFDM 调制解调器：核心为 DAFT/IDAFT"""
    
    def __init__(self, N=64, Q=0, c2=0.0, cp_len=16):
        self.N = N
        self.c1 = (2*Q + 1) / (2*N)  # 全分集参数，Q=0 → c1=1/(2N)
        self.c2 = c2                   # 通常设为 0
        self.cp_len = cp_len
        self.n = np.arange(N)
        self.k = np.arange(N)
    
    def modulate(self, X):
        """IDAFT：频域数据符号 → 时域发送信号"""
        # 步骤1：后啁啾（频域）
        X_post = X * np.exp(1j * 2 * np.pi * self.c2 * self.k**2)
        # 步骤2：IDFT（numpy ifft 归一化 1/N，补 sqrt(N) 得 1/sqrt(N)）
        x_idft = np.fft.ifft(X_post) * np.sqrt(self.N)
        # 步骤3：预啁啾（时域）
        x = x_idft * np.exp(1j * 2 * np.pi * self.c1 * self.n**2)
        # 添加循环前缀
        return np.concatenate([x[-self.cp_len:], x])
    
    def demodulate(self, y):
        """DAFT：接收信号 → 仿射频域符号"""
        x = y[self.cp_len:]                        # 去 CP
        # 步骤1：去预啁啾
        x_pre = x * np.exp(-1j * 2 * np.pi * self.c1 * self.n**2)
        # 步骤2：DFT
        X_dft = np.fft.fft(x_pre) / np.sqrt(self.N)
        # 步骤3：去后啁啾
        return X_dft * np.exp(-1j * 2 * np.pi * self.c2 * self.k**2)
```

当 `c1 = 0`，DAFT 退化为标准 DFT，AFDM 退化为 OFDM——这保证了向后兼容性。

### 双选择性信道仿真

```python
def doubly_selective_channel(x, paths, N, snr_db=20):
    """
    仿真延迟-多普勒信道
    paths: [(增益h, 延迟ℓ/采样点, 多普勒ν/归一化bins)]
    """
    y = np.zeros(len(x), dtype=complex)
    t = np.arange(len(x))
    
    for h, l, nu in paths:
        x_delayed = np.roll(x, l)   # 延迟：循环移位
        x_delayed[:l] = 0           # 消除循环部分
        # 多普勒：时变相位旋转
        y += h * x_delayed * np.exp(1j * 2 * np.pi * nu * t / N)
    
    # AWGN 噪声
    sigma = np.sqrt(10**(-snr_db/10) / 2)
    return y + sigma * (np.randn(len(x)) + 1j * np.randn(len(x)))
```

### 信道矩阵结构对比（核心洞察）

理解 AFDM 优势的最直观方式——比较 OFDM 与 AFDM 的有效信道矩阵：

```python
import matplotlib.pyplot as plt

def channel_matrix_afdm(paths, N, c1):
    """AFDM 仿射频域信道矩阵：理论上近似带状"""
    H = np.zeros((N, N), dtype=complex)
    for h, l, nu in paths:
        # 每条路径在仿射频域中贡献到特定的移位对角线
        effective_shift = int(round(2 * c1 * l * N + nu)) % N
        H += h * np.roll(np.eye(N), effective_shift, axis=1)
    return H

def channel_matrix_ofdm(paths, N):
    """OFDM 频域信道矩阵：多普勒导致严重的非对角扩散（ICI）"""
    H = np.zeros((N, N), dtype=complex)
    n = np.arange(N)
    for h, l, nu in paths:
        for k in range(N):
            for m in range(N):
                # 多普勒使能量从子载波 m 泄漏到 k
                H[k, m] += h * np.exp(-1j*2*np.pi*m*l/N) * \
                            np.exp(1j*np.pi*(nu-(k-m))) * \
                            np.sinc(nu - (k - m))
    return H / N

# 测试信道：3条路径，有延迟和多普勒
paths = [(1.0, 0, 0), (0.6, 3, 2), (0.4, 7, -3)]
N = 32
c1 = 1 / (2 * N)  # Q=0

H_afdm = channel_matrix_afdm(paths, N, c1)
H_ofdm = channel_matrix_ofdm(paths, N)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(np.abs(H_ofdm), cmap='hot', aspect='auto')
axes[0].set_title('OFDM 信道矩阵\n（ICI导致全矩阵非零）')
axes[1].imshow(np.abs(H_afdm), cmap='hot', aspect='auto')
axes[1].set_title('AFDM 信道矩阵\n（仿射频域：稀疏带状结构）')
plt.tight_layout()
plt.show()
```

**预期输出**：OFDM 矩阵几乎全部非零（能量弥散到每个子载波对），而 AFDM 矩阵只有少数非零对角条纹——每条路径对应一条移位的对角线。带状结构直接决定了均衡器的复杂度和分集性能。

---

## ISAC：通感一体化

AFDM 的 chirp 子载波与 FMCW 雷达波形天然相似，同一帧数据可以同时传递信息和探测目标：

```python
def afdm_range_doppler_map(tx_frames, rx_frames, N, cp_len):
    """
    利用 AFDM 通信帧进行距离-多普勒估计
    tx_frames, rx_frames: shape (n_symbols, N+cp_len)
    """
    n_sym = len(tx_frames)
    rd_matrix = np.zeros((n_sym, N), dtype=complex)
    
    for i, (tx, rx) in enumerate(zip(tx_frames, rx_frames)):
        tx_body = tx[cp_len:]  # 去CP
        rx_body = rx[cp_len:]
        # 频域相关：消除通信调制，保留信道响应
        Tx = np.fft.fft(tx_body)
        Rx = np.fft.fft(rx_body)
        rd_matrix[i] = Rx * np.conj(Tx) / (np.abs(Tx)**2 + 1e-10)
    
    # 跨符号 FFT → 多普勒维度
    rd_map = np.abs(np.fft.fftshift(np.fft.fft2(rd_matrix), axes=0))
    return rd_map  # 行=多普勒，列=距离（频率延迟）
```

无需额外硬件——通信链路同步输出目标的距离-速度估计图。

---

## 工程实践

### PAPR 问题

AFDM 的 PAPR 与 OFDM 相近（8–12 dB），因为时域样本仍是多路 chirp 叠加：

```python
def papr_db(signal):
    """峰均功率比（越低越好，功放友好）"""
    return 10 * np.log10(np.max(np.abs(signal)**2) / np.mean(np.abs(signal)**2))

# AFDM 的 PAPR 削减：可沿用 OFDM 的 SLM/PTS 方法，
# 区别在于相位旋转作用在频域符号 X[k] 上，而非 OFDM 子载波
```

PAPR 是 AFDM 标准化必须解决的难题——与 OFDM 相当意味着功放效率瓶颈没有改善。

### 常见坑

**坑1：c₁ 选错，分集增益消失**
```python
# 危险：c1 必须满足 (2Q+1)/(2N) 形式
c1_bad  = 1.0 / N        # 偶数分子，不满足条件
c1_good = 1.0 / (2 * N)  # Q=0，满足全分集条件
```

**坑2：CP 长度只考虑延迟，忘了多普勒引起的等效"延伸"**
```python
# OFDM：cp_len >= max_delay
# AFDM：cp_len 需覆盖延迟 + 多普勒引起的符号间串扰范围
# 保守估计：
cp_len = max_delay_bins + 2 * max_doppler_bins
```

**坑3：低轨卫星多普勒超出整数假设**

LEO 卫星的多普勒可达 ±200 kHz（子载波间隔 15 kHz 时对应 ±13 个 bin），必须先做粗频偏补偿，再用 AFDM 处理残余整数 Doppler。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 高铁/高速公路 V2X（高多普勒） | 室内 WiFi（低速，OFDM 已够用） |
| 低轨卫星通信（LEO） | 固定宽带接入（纯频率选择信道） |
| 水下声学通信（UWA） | 现有 OFDM 系统平滑升级（生态成本高） |
| ISAC 联合感知通信 | 对 PAPR 极度敏感的应用 |
| 应急通信（极端移动性） | 静态大规模 MIMO 场景 |

---

## 与其他方案对比

| 方案 | 双选信道 | 复杂度 | ISAC | 标准兼容 |
|-----|---------|-------|------|---------|
| OFDM | 差（ICI 平台） | $O(N\log N)$ | 一般 | 现行标准 |
| OTFS | 好 | $O(N\log N)$（两次） | 好 | 研究阶段 |
| **AFDM** | **好（全分集）** | **$O(N\log N)$（接近 OFDM）** | **极好** | **研究阶段** |
| FMCW | N/A | 低 | 极好 | 雷达标准 |

AFDM 相比 OTFS 的核心竞争力：DAFT = 一次 FFT + 两次向量点乘，接收机架构改动极小，可在 OFDM 芯片上以软件升级方式实现。

---

## 我的看法

论文的标准化路径分析做得很扎实——与 4G/5G numerology（$c_1=0$ 退化为 OFDM）、LoRa（chirp 天然兼容）和 FMCW 的兼容性都有理论论证。但以下挑战仍是拦路虎：

**近期可落地**：水下声学和低轨卫星通信。这两个领域 Doppler 最极端、现有 OFDM 方案已明显破防，且生态包袱最轻，是 AFDM 最可能首先商用的切入口。

**中期挑战**：大规模 MIMO 下的 DAFT 预编码设计、实际信道估计的导频开销、CSI 反馈码本重设计——这些都需要大量标准化工作。

**长期问题**：OFDM 积累了 30 年的芯片 IP 和算法生态，替换成本不容低估。AFDM 更可能以"可选波形"或"高速增强模式"的形式进入标准，而非全面替代。

值得持续关注的开放问题：AFDM + 稀疏信道估计的联合设计、非整数 Doppler 下的性能分析、以及与 AI 辅助均衡的结合。