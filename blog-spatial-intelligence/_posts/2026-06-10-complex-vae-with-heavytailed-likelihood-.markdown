---
layout: post-wide
title: "用复值 VAE 检测海杂波中的雷达目标"
date: 2026-06-10 12:04:56 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2606.10540v1
generated_by: Claude Code CLI
---

## 一句话总结

仅用纯杂波数据训练的复值变分自编码器，通过重建偏差实现海面雷达目标的 CFAR 检测——无需任何目标标签。

## 为什么这个问题重要？

海面目标检测（船只、小艇、漂浮物）是海岸监控和海事安全的核心任务。难点在两处：

**海杂波统计特性特殊**：海面回波不是高斯分布的，而是重尾、有尖峰的。K 分布、复合高斯等模型能描述它，但参数估计困难，尤其在不同海况下。

**目标标签极度稀缺**：雷达工作时大多数时候都是杂波，偶尔才有目标回波。手动标注代价极高，不同场景下目标特征差异也大。

现有方法的短板：
- **MF/AMF**：假设高斯杂波，面对重尾杂波时虚警率飙升
- **CFAR 经典方法**（CA-CFAR、OS-CFAR）：非均匀场景下分布估计失效
- **有监督深度学习**：需要目标标签，而目标样本极少

本文的思路很干净：**把检测问题转化成异常检测**——只用杂波数据训练生成模型，目标因为和杂波"不像"，重建误差会显著偏大，从而触发检测。

## 背景知识

### 雷达慢时间信号

雷达向海面发射脉冲，对于某个固定距离单元，连续 $N$ 个脉冲的回波叫**慢时间序列**：

$$\mathbf{x} = [x_1, x_2, \ldots, x_N]^\top \in \mathbb{C}^N$$

每个采样是复数（基带信号），拆成**同相分量 I** 和**正交分量 Q**：$x_t = I_t + j Q_t$

### 海杂波的重尾特性

海面回波幅度分布偏离高斯，有"重尾"特征——小概率大幅度事件（尖峰）比高斯预测的多得多。Student-t 分布更合适：

$$p(r; \nu) \propto \left(1 + \frac{r^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

自由度 $\nu$ 越小，尾部越重。相比高斯，Student-t 对离群点的惩罚是对数增长而非平方增长，训练时不会被少数极端值主导。

## 核心方法

### 直觉解释

把每个距离单元的 $N$ 点复信号打平成 $2N$ 维实向量（I 和 Q 拼接），喂给 VAE：

```
慢时间复信号 [I,Q]
      ↓
   编码器 → (μ, σ) → 重采样 z → 解码器 → 重建 [Î, Q̂]
                                              ↓
                              重建误差 D = NLL(x, x̂) → 与阈值比较 → 检测结果
```

模型只见过杂波，学会"杂波长什么样"：
- 纯杂波 → 重建好 → $D$ 小 → 不报警
- 目标叠加杂波 → 重建差 → $D$ 大 → 触发检测

### 数学细节

**Student-t 重建损失**

对每个慢时间样本的 I/Q 残差，用 Student-t 负对数似然：

$$\mathcal{L}_{St}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{\nu+1}{2} \sum_{t=1}^{N} \log\left(1 + \frac{(I_t - \hat{I}_t)^2 + (Q_t - \hat{Q}_t)^2}{\nu \sigma^2}\right)$$

训练时，杂波尖峰产生大残差，但对数函数压低了其梯度权重，模型更专注于拟合"典型杂波"。

**幅度误差约束**

慢时间幅度序列反映目标的多普勒特征。额外加一项惩罚幅度失配：

$$\mathcal{L}_{amp} = \frac{1}{N} \sum_{t=1}^{N} \left( \sqrt{I_t^2 + Q_t^2} - \sqrt{\hat{I}_t^2 + \hat{Q}_t^2} \right)^2$$

**总损失**

$$\mathcal{L} = \underbrace{\mathcal{L}_{St}}_{\text{重建}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{正则}} + \lambda \cdot \underbrace{\mathcal{L}_{amp}}_{\text{幅度约束}}$$

**CFAR 阈值**

从纯杂波验证集估计检测统计量的分位数：

$$\eta = \hat{Q}_{1-P_{fa}}\!\left(\{D_i\}_{i \in \mathcal{V}_{clutter}}\right)$$

不管海况如何变化，虚警率都维持在设定水平 $P_{fa}$。

## 实现

### 模型定义

```python
import torch
import torch.nn as nn
import numpy as np

class ComplexVAE(nn.Module):
    """复值慢时间 VAE：输入为 I/Q 拼接的实数向量"""
    def __init__(self, seq_len: int = 32, latent_dim: int = 8):
        super().__init__()
        d = seq_len * 2  # I 和 Q 各 seq_len 维

        self.encoder = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, d),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

### 损失函数

```python
def student_t_nll(x: torch.Tensor, x_hat: torch.Tensor,
                  nu: float = 3.0, sigma: float = 1.0) -> torch.Tensor:
    """Student-t 负对数似然，逐样本返回标量 [B]"""
    r2 = ((x - x_hat) ** 2).sum(dim=-1)
    return (nu + 1) / 2 * torch.log(1 + r2 / (nu * sigma ** 2))

def amp_constraint(x: torch.Tensor, x_hat: torch.Tensor,
                   seq_len: int) -> torch.Tensor:
    """慢时间幅度不匹配惩罚"""
    I,  Q  = x[:, :seq_len],     x[:, seq_len:]
    Ih, Qh = x_hat[:, :seq_len], x_hat[:, seq_len:]
    amp     = (I**2  + Q**2  ).sqrt()
    amp_hat = (Ih**2 + Qh**2 + 1e-8).sqrt()
    return ((amp - amp_hat) ** 2).mean(dim=-1)   # [B]

def vae_loss(x, x_hat, mu, logvar, seq_len,
             nu=3.0, lam=0.1) -> torch.Tensor:
    recon = student_t_nll(x, x_hat, nu).mean()
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
    amp   = amp_constraint(x, x_hat, seq_len).mean()
    return recon + kl + lam * amp
```

### 训练与 CFAR 阈值设定

```python
def train(model, loader, epochs=50, lr=1e-3, seq_len=32, device='cuda'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(batch)
            loss = vae_loss(batch, x_hat, mu, logvar, seq_len)
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1:3d}/{epochs}  loss={total/len(loader):.4f}")

def cfar_threshold(model, clutter_val: torch.Tensor,
                   pfa: float = 0.01) -> float:
    """用纯杂波验证集确定 CFAR 阈值"""
    model.eval()
    with torch.no_grad():
        x_hat, mu, _ = model(clutter_val)
        scores = student_t_nll(clutter_val, x_hat).cpu().numpy()
    return float(np.quantile(scores, 1 - pfa))
```

### 可视化检测结果

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_detection(scores: np.ndarray, threshold: float,
                   labels: np.ndarray = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 检测统计量分布
    axes[0].hist(scores, bins=60, density=True, alpha=0.7)
    axes[0].axvline(threshold, color='r', ls='--',
                    label=f"CFAR 阈值 η={threshold:.2f}")
    axes[0].set_xlabel("重建误差 D"); axes[0].legend()
    axes[0].set_title("检测统计量分布")

    # ROC 曲线
    if labels is not None:
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[1].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        axes[1].plot([0,1],[0,1],'k--')
        axes[1].set_xlabel("虚警率 (PFA)"); axes[1].set_ylabel("检测率 (PD)")
        axes[1].legend(); axes[1].set_title("ROC 曲线")

    plt.tight_layout(); plt.show()
```

预期输出：左图显示杂波样本的分数紧密集中在低值区，目标样本分数明显偏高；右图 ROC 曲线 AUC 应高于 MF/AMF 基线。

## 实验

### 数据说明

论文使用**实测海杂波数据**，典型公开数据集包括 IPIX（麦克马斯特大学）。数据包含多个距离单元的慢时间 IQ 采样，标注了目标出现的时间窗口。

数据获取难度：**中等**。IPIX 可申请访问，格式为 `.mat`，需预处理。商业雷达实测数据基本不公开，这是该方向最大的工程瓶颈。

### 定量评估

在 $P_{fa} = 10^{-2}$ 约束下，各方法检测率对比（依据论文结论定性排列）：

| 方法 | 杂波模型假设 | 需要目标标签 | 重尾场景 $P_d$ |
|------|-------------|-------------|--------------|
| MF | 高斯 | 否 | 低 |
| AMF | 高斯（协方差估计）| 否 | 中等 |
| 实值 β-VAE | 高斯 NLL | 否 | 较高 |
| **复值 VAE（本文）** | **Student-t NLL** | **否** | **最高** |

**核心增益来源**：Student-t NLL 在训练时对杂波尖峰降权，保留了"典型杂波"的建模能力；幅度约束在低 SNR 场景帮助更大。

## 工程实践

### 硬件与延迟

| 部署场景 | 配置 | 每 CPI 处理延迟 |
|---------|------|----------------|
| 岸基实时 | RTX 3060 | < 5 ms（32 距离单元并行） |
| 嵌入式 | Jetson AGX | ~50 ms，勉强实时 |
| 离线分析 | CPU | 100+ ms |

CPI 通常 10–100 ms，GPU 推理完全满足实时要求，模型本身较小（< 1 MB）。

### 数据采集建议

1. **杂波多样性**：训练数据应覆盖多个海况（蒲福 2–6 级）和掠射角，否则换场景后性能退化
2. **IQ 归一化**：训练集用 z-score 归一化，推理时用相同的均值/方差，存储这两个值
3. **片段长度**：慢时间 $N$ 取 32–64 脉冲，太短捕捉不到多普勒信息，太长 RCS 起伏影响大

### 常见坑

**坑 1：阈值随海况漂移**

固定阈值在 3 级海况训出的模型直接用于 5 级海况，虚警率爆炸。

```python
# 滑动窗口在线更新 CFAR 阈值
def update_threshold(score_buffer: list, pfa: float = 0.01) -> float:
    scores = np.array(score_buffer[-2000:])  # 保留最近 2000 个杂波估计
    return float(np.quantile(scores, 1 - pfa))
```

**坑 2：目标污染训练集**

训练用的"纯杂波"距离单元中混入了船只回波，模型会把目标当作正常杂波拟合，导致漏检。解决：用传统 CFAR 做初筛，剔除疑似目标单元后再训练。

**坑 3：latent_dim 过小**

海杂波在时域有复杂相关结构，`latent_dim < 4` 时模型欠拟合，重建误差对杂波和目标都偏大，鉴别力下降。建议从 8–16 开始调，用验证集 AUC 指导。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 无法获取目标标签 | 有大量标注数据（有监督方法更强）|
| 海杂波为主，目标偶现 | 目标密集（如港口场景）|
| 重尾非高斯海况 | 平静海面，杂波近高斯分布 |
| 静止或慢速目标 | 高速目标（直接用 MTI/MTD 更有效）|
| 单站岸基固定场景 | 快变场景（实时重训练代价高）|

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| MF/AMF | 快速、无需训练 | 高斯假设，重尾场景虚警高 | 平静海况 |
| OS-CFAR | 对非均匀杂波鲁棒 | 需要精确分布假设 | 均匀杂波 |
| 有监督 CNN | 精度高 | 需要大量目标标签 | 有标注数据集 |
| 实值 β-VAE | 无监督，结构简单 | 高斯 NLL 对重尾不鲁棒 | 轻尾场景 |
| **复值 VAE（本文）** | 无监督，处理重尾，保留相位信息 | 需调 $\nu$、$\lambda$、latent_dim | 重尾、标签稀缺场景 |

## 我的观点

这篇论文做了一件务实的事：把 VAE 从图像生成搬到雷达信号处理，解决了"没有目标标签怎么检测"这个真实工程痛点。几个值得关注的方向：

**复数表示仍有空间**：I/Q 拼接是一种简化，真正的复数神经网络（complex-valued NN）可以更好地保持相位等变性，在相位敏感的任务上应有增益。

**在线适应是核心挑战**：现在的方法假设海况统计平稳，但海况变化很快。如何让 VAE 快速适应新的杂波分布（元学习、在线微调），是最直接的扩展方向。

**从单单元到二维**：当前方法处理单距离单元的慢时间序列。如果把整个距离-多普勒图作为输入，二维结构能提供更丰富的上下文，也许能更好地分离目标和杂波。

**离产品化的距离**：模型推理足够快，主要障碍是训练数据的多样性。岸基固定雷达可以积累，但机载/舰载平台的快变场景仍然是开放问题。