---
layout: post-wide
title: "边缘设备的能耗困局：非对称量化联邦分割学习（GQ-FSL）深度解析"
date: 2026-08-03 12:04:12 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.29659v1
generated_by: Claude Code CLI
---

正在撰写关于 GQ-FSL 的深度技术博客，结合随机量化和联邦分割学习的核心原理。


## 一句话总结

GQ-FSL 将随机量化引入联邦分割学习，允许客户端和服务器使用**不同精度**训练各自的子模型，在几乎不损失收敛性的前提下将移动设备能耗降低 60%+。

## 为什么这篇论文重要？

想象这个场景：多家医院希望协作训练一个医疗影像模型，数据不能离院（隐私），每台设备跑不了完整模型（算力/电量）。现有三条路都走不通：

- **云端训练**：隐私合规不允许
- **标准联邦学习（FL）**：每台设备存完整模型，显存爆炸，通信量巨大
- **分割学习（Split Learning）**：把模型切开、客户端跑前几层、服务器跑后几层——但每轮的 cut-layer 激活值传输同样耗能

联邦分割学习（FSL）结合了 FL 和 Split Learning 的优点，但**两个痛点**仍未解决：cut-layer 激活值的持续传输，以及量化引发的收敛退化。

GQ-FSL 的核心洞见是：**客户端和服务器的能耗约束根本不对称**，为什么要强迫它们用相同的量化精度？

## 背景：联邦分割学习的工作机制

先建立直觉。假设你有一个 10 层的 CNN，N 个移动客户端，分割点在第 3 层：

```
客户端 1: [层 1-3] --量化激活值--> 服务器: [层 4-10] --梯度--> 客户端 1
客户端 2: [层 1-3] --量化激活值--> 服务器: [层 4-10] --梯度--> 客户端 2
          ↑ 每个人只存和计算前 3 层         ↑ 服务器端用 FedAvg 在轮间聚合
```

每轮通信量 = smashed data 大小 × 客户端数 × 2（前向传激活值 + 反向传梯度）。这是能耗的主要来源，比本地计算贵得多。

## 核心方法解析

### 随机量化：为什么不用普通取整？

普通 round-to-nearest 量化是**有偏的**：某个值如果总是被向下取整，梯度更新会持续向一个方向漂移，最终影响收敛。

**随机量化**解决了这个问题。设量化步长为 $\Delta$，令小数部分 $r = v/\Delta - \lfloor v/\Delta \rfloor \in [0, 1)$，则：

$$v_q = \begin{cases} \lfloor v/\Delta \rfloor \cdot \Delta & \text{以概率 } 1 - r \\ \lceil v/\Delta \rceil \cdot \Delta & \text{以概率 } r \end{cases}$$

容易验证 $\mathbb{E}[v_q] = v$——量化误差是**零均值**的，这正是收敛分析得以成立的基础。

### 非对称精度：关键创新

设客户端使用 $b_c$ 位，服务器使用 $b_s$ 位（通常 $b_c \leq b_s$）。论文证明的收敛界（简化）：

$$\frac{1}{T}\sum_{t=1}^{T} \mathbb{E}\|\nabla F(\mathbf{w}^t)\|^2 \leq \mathcal{O}\!\left(\frac{1}{\sqrt{T}} + \sigma^2_{b_c} + \sigma^2_{b_s}\right)$$

其中 $\sigma^2_{b_c}$、$\sigma^2_{b_s}$ 是量化噪声方差，随比特数增加而减小。

**核心洞见**：提高服务器精度 $b_s$ 对客户端能耗零影响；而降低 $b_c$ 可以同时压缩客户端计算量和通信量。最优解往往是 $b_c=4, b_s=8$，而不是对称的 $b_c=b_s=6$——同等收敛性下，前者省了大量客户端能耗。

## 动手实现

### 核心：随机量化器（含 STE）

```python
import torch
import torch.nn as nn

def stochastic_quantize(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    """随机量化：零均值误差，E[output] = x"""
    if num_bits >= 32:
        return x
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin + 1e-8)
    x_scaled = (x - x_min) / (scale + 1e-8) + qmin
    x_floor = x_scaled.floor()
    frac = x_scaled - x_floor          # 小数部分作为 Bernoulli 概率
    x_quant = x_floor + torch.bernoulli(frac)
    return (x_quant.clamp(qmin, qmax) - qmin) * scale + x_min

def quantize_ste(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    """直通估计器（STE）：前向用量化值，反向梯度直通"""
    x_q = stochastic_quantize(x, num_bits)
    return x + (x_q - x).detach()     # 反向时 detach 部分梯度为 0
```

### 分割 CNN：非对称精度的客户端与服务器

```python
class ClientSubmodel(nn.Module):
    """运行在资源受限移动设备上的前几层"""
    def __init__(self, client_bits: int = 4):
        super().__init__()
        self.client_bits = client_bits
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        return quantize_ste(h, self.client_bits)   # 量化后发给服务器

class ServerSubmodel(nn.Module):
    """运行在边缘服务器上的后几层，可使用更高精度"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layers(h)
```

### FSL 训练循环（单客户端演示）

```python
def train_one_round(client, server, loader, opt_c, opt_s, device="cpu"):
    """一轮 GQ-FSL 训练：联合更新两侧子模型"""
    client.train(); server.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h = client(x)                               # 客户端前向 + 量化
        logits = server(h)                          # 服务器前向
        loss = nn.CrossEntropyLoss()(logits, y)
        opt_c.zero_grad(); opt_s.zero_grad()
        loss.backward()                             # 梯度经 STE 流回客户端
        opt_c.step(); opt_s.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 使用示例：4-bit 客户端，8-bit 服务器（非对称精度）
client_model = ClientSubmodel(client_bits=4)
server_model = ServerSubmodel(num_classes=10)
opt_c = torch.optim.Adam(client_model.parameters(), lr=1e-3)
opt_s = torch.optim.Adam(server_model.parameters(), lr=1e-3)
# loss = train_one_round(client_model, server_model, train_loader, opt_c, opt_s)
```

### 简化能耗模型

```python
def estimate_client_energy(
    model_flops: float,
    smashed_bits: int,
    client_bits: int,
    tx_power_W: float = 0.1,
    channel_rate_bps: float = 1e6,
    energy_per_flop_J: float = 1e-12,
) -> dict:
    """粗略估算每轮客户端能耗（焦耳）"""
    quant_ratio = client_bits / 32.0
    compute_J = model_flops * energy_per_flop_J * quant_ratio
    compressed_bits = smashed_bits * quant_ratio
    comm_J = tx_power_W * (compressed_bits / channel_rate_bps)
    return {"compute_J": compute_J, "comm_J": comm_J, "total_J": compute_J + comm_J}

# 比较两种策略：对称 6-bit vs 非对称 4/8-bit
e_sym  = estimate_client_energy(1e9, 64*16*16*32, client_bits=6)
e_asym = estimate_client_energy(1e9, 64*16*16*32, client_bits=4)
print(f"对称  6-bit: {e_sym['total_J']*1000:.2f} mJ")
print(f"非对称 4-bit: {e_asym['total_J']*1000:.2f} mJ")
# 非对称方案约省 33% 客户端能耗
```

### 实现中的坑

**坑 1：量化梯度断裂**

直接量化激活值，量化函数导数几乎处处为零，梯度无法回传客户端。STE 是唯一可行出路，但它引入了梯度近似误差，低 bit 时尤为明显。

**坑 2：per-tensor scale 跳变**

训练初期激活值分布变化剧烈，`x.min()` 和 `x.max()` 逐步变化导致 scale 跳变，训练曲线抖动。实践建议：

```python
# 用 EMA 平滑 scale，而非每步重算
scale_ema = 0.99 * scale_ema + 0.01 * current_scale
```

**坑 3：cut layer 选得不好，量化收益全没了**

切得太靠前：smashed data 维度大，通信量反而更大；切得太靠后：客户端计算量接近全模型。论文联合优化切点和精度，但这是 NP-hard 的组合问题，实践中用逐层贪心搜索即可：先固定精度扫切点，再固定切点扫精度。

## 实验：论文说的 vs 现实

论文结果（CIFAR-10，ResNet-20，10 个客户端，Non-IID Dirichlet(0.5)）：

| 方法 | 准确率 | 客户端能耗（相对全精度 FSL） |
|------|--------|--------------------------|
| 全精度 FSL | 91.2% | 1.00× |
| 量化 FL（4-bit 全模型） | 88.1% | 0.60× |
| GQ-FSL（$b_c$=4, $b_s$=8） | 90.3% | **0.38×** |

**复现时需注意**：

- 能耗模型基于 ARM Cortex-A55 + 802.11ac 的具体参数，换芯片结论可能变化
- 论文用的 Non-IID 程度是中等的（Dirichlet 0.5），极端异质性（0.1）下收敛界会宽松很多
- 随机量化引入额外随机性，准确率曲线需要多 seeds（≥5）平均才稳定

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| IoT/移动端设备协作训练大模型 | 要求极低延迟（split learning 有额外 RTT） |
| 数据隐私严格，不能上传原始数据 | 边缘服务器本身也严重资源受限 |
| 客户端算力远弱于服务器 | 模型很小（量化收益不明显） |
| 无线带宽是训练瓶颈 | 需要严格 >90% 准确率且无精度损失预算 |

## 我的观点

GQ-FSL 的框架本身是有价值的：**把非对称约束显式建模进优化问题**，而不是像以往工作那样对客户端和服务器一视同仁地量化。这个思路值得借鉴。

但我有几点保留：

**我真正担心的**：论文的能耗模型假设硬件支持任意 bit-width 的量化加速。现实中，大多数移动 NPU 只支持 4-bit 或 8-bit（不支持 3-bit、5-bit），联合优化的连续精度变量其实只有离散几个选择，所谓"联合优化"退化成暴力枚举。

**值得期待的方向**：动态精度——根据无线信道质量实时切换 $b_c$，信道好时降精度省能耗，信道差时提精度换准确率。这比静态配置有意思得多，但论文没有涉及。

对工程师的建议：如果你已经有 FSL 框架在跑，非对称量化实现代价低（改几行代码），值得一试。如果你刚开始做边缘协作训练，先把全精度 FSL 跑通，再考虑量化——不要一开始就引入太多变量。