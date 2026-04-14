---
layout: post-wide
title: "GenTac：用扩散模型生成足球战术轨迹"
date: 2026-04-14 12:06:56 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.11786v1
generated_by: Claude Code CLI
---

## 一句话总结

GenTac 把足球战术建模成一个随机过程，用扩散模型采样出多条"合理但不相同"的未来轨迹——告别确定性预测，拥抱战术的本质不确定性。

## 为什么这个问题重要？

### 战术分析的核心困境

足球是一个**高度随机的多智能体系统**。同样的开局，同样的阵型，10 秒后的走位可能完全不同。传统的战术分析方法面临三个根本性问题：

1. **确定性预测的谎言**：大多数轨迹预测模型输出单条轨迹，但足球场上"最可能的未来"往往并不代表实际发生的情况
2. **多智能体耦合被忽视**：11 名球员的运动不是独立的——前锋的跑位改变了中场的传球选择
3. **战术语义缺失**：纯粹的轨迹坐标无法解释"为什么这样跑"

GenTac 的核心洞察：**战术不是一条轨迹，而是一个概率分布**。

### 实际应用场景

- **教练分析**：生成反事实战术（"如果换一种防守阵型，对手进攻威胁如何变化？"）
- **AI 陪练系统**：生成符合特定联赛风格的对手行为
- **青训教学**：量化不同走位方案的期望威胁值变化

## 背景知识

### 多智能体轨迹表示

足球战术的输入是**球员追踪数据（Tracking Data）**，每帧记录场上所有球员的 (x, y) 坐标：

$$
\mathbf{X} \in \mathbb{R}^{N \times T \times 2}
$$

其中 $N = 22$（双方球员），$T$ 为时间步数，通常以 25fps 采样。这类数据由 Hawk-Eye、TRACAB 等光学追踪系统提供，价格昂贵，是真正的行业壁垒。

### 扩散模型基础

GenTac 的生成核心是**去噪扩散概率模型（DDPM）**：

**前向过程**（加噪）：

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\right)
$$

**反向过程**（带条件去噪）：

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c}) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\, \mu_\theta(\mathbf{x}_t, t, \mathbf{c}),\, \sigma_t^2 \mathbf{I}\right)
$$

其中 $\mathbf{c}$ 是条件信息（战术风格、队伍 ID 等）。扩散模型天然支持多次采样，正好契合"同一局面可能有多种战术展开"的需求。

## 核心方法

### 直觉解释

把球场想象成一个"战术相空间"。每一帧的 22 名球员位置构成这个空间的一个点，一场比赛是这个空间中的一条轨迹。GenTac 学习**这些轨迹背后的概率密度**，生成时从高斯噪声出发，通过扩散反向过程逐步"雕刻"出一条符合指定风格的合理轨迹。

### 数学细节

**团队结构一致性约束**（GenTac 的关键创新）：

$$
\mathcal{L}_{\text{structure}} = \sum_{i,j \in \text{same team}} \left\| d_{ij}^{\text{gen}} - d_{ij}^{\text{ref}} \right\|^2
$$

生成轨迹中队友间距离分布应与历史数据一致，防止出现"11 人跑到同一角落"的战术荒谬情况。

**战术事件与轨迹联合建模**：

$$
p(\mathbf{X}, \mathbf{e} \mid \mathbf{c}) = p_\theta(\mathbf{X} \mid \mathbf{e}, \mathbf{c}) \cdot p_\phi(\mathbf{e} \mid \mathbf{c})
$$

先生成 15 类战术事件序列（传球、射门、抢断等），再以事件为条件生成对应轨迹。

### Pipeline 概览

```
历史追踪数据
      ↓
[条件编码器] → 战术风格 c (队伍/联赛/策略目标)
      ↓
[事件生成器 p_φ] → 战术事件序列 e = (传球, 推进, 射门, ...)
      ↓
[轨迹扩散模型 p_θ] → 多条采样轨迹 X¹, X², ..., Xᴷ
      ↓
[结构一致性过滤] → 保留队形合理的轨迹
```

## 实现

```bash
pip install torch numpy matplotlib
```

### 多智能体轨迹去噪网络

```python
import torch
import torch.nn as nn
import numpy as np

class TacticsDenoiser(nn.Module):
    """
    多智能体战术轨迹去噪网络
    输入: 含噪轨迹 (B, N, T, 2) + 时间步 + 条件向量
    输出: 预测噪声 (B, N, T, 2)
    """
    def __init__(self, n_players=22, traj_len=50, cond_dim=64, hidden=256):
        super().__init__()
        traj_in = traj_len * 2                          # x,y 坐标展平

        self.player_embed = nn.Linear(traj_in, hidden)
        # 多头注意力建模球员间交互 (O(N²) 但 N=22 可接受)
        self.interaction = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, traj_in))

    def sinusoidal_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, x_noisy, t, cond=None):
        B, N, T, _ = x_noisy.shape
        h = self.player_embed(x_noisy.reshape(B, N, -1))  # (B, N, hidden)
        h, _ = self.interaction(h, h, h)                   # 球员间交互注意力

        t_emb = self.sinusoidal_embedding(t, h.shape[-1])
        h = h + self.time_embed(t_emb).unsqueeze(1)        # 注入时间步

        if cond is not None:
            h = h + self.cond_proj(cond).unsqueeze(1)      # 注入战术条件

        return self.output(h).reshape(B, N, T, 2)
```

### 扩散过程：加噪与采样

```python
class TacticsDiffusion:
    """管理 DDPM 前向加噪和反向采样"""
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0)    # ᾱ_t

    def add_noise(self, x0, t):
        """前向: q(x_t | x_0) = N(√ᾱ_t · x0, (1-ᾱ_t)I)"""
        ab = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

    @torch.no_grad()
    def sample(self, model, shape, cond=None, device='cpu'):
        """从纯噪声逐步去噪，生成多条战术轨迹"""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_batch, cond)

            ab_t = self.alpha_bar[t].to(device)
            ab_prev = self.alpha_bar[t - 1].to(device) if t > 0 else torch.tensor(1.0)

            x0_est = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            x0_est = x0_est.clamp(-1, 1)       # 坐标归一化到 [-1, 1]

            mean = torch.sqrt(ab_prev) * x0_est + torch.sqrt(1 - ab_prev) * noise_pred
            var = (1 - ab_t / ab_prev) * (1 - ab_prev) / (1 - ab_t)
            x = mean + torch.sqrt(var.clamp(min=0)) * torch.randn_like(x)
        return x
```

### 训练步骤（含团队结构损失）

```python
def team_structure_loss(x_pred, x_true):
    """队形一致性：队友终点间距离分布应与真实数据吻合"""
    loss = 0
    for idx in [slice(0, 11), slice(11, 22)]:   # 主客队各 11 人
        d_pred = torch.cdist(x_pred[:, idx, -1], x_pred[:, idx, -1])
        d_true = torch.cdist(x_true[:, idx, -1], x_true[:, idx, -1])
        loss += nn.functional.mse_loss(d_pred, d_true)
    return loss / 2

def train_step(model, diffusion, x0, optimizer, cond=None):
    """x0: (B, N, T, 2) 归一化轨迹; cond: (B, cond_dim) 战术条件"""
    B = x0.shape[0]
    t = torch.randint(0, diffusion.n_steps, (B,), device=x0.device)
    x_t, noise = diffusion.add_noise(x0, t)
    noise_pred = model(x_t, t, cond)

    loss_denoise = nn.functional.mse_loss(noise_pred, noise)

    ab = diffusion.alpha_bar[t].reshape(-1, 1, 1, 1).to(x0.device)
    x0_pred = (x_t - torch.sqrt(1 - ab) * noise_pred) / torch.sqrt(ab)
    loss_struct = team_structure_loss(x0_pred, x0)

    loss = loss_denoise + 0.1 * loss_struct
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

### 战术轨迹可视化

```python
import matplotlib.pyplot as plt, matplotlib.patches as patches

def visualize_tactics(trajectories, n_show=3):
    """
    trajectories: (K, N, T, 2) - K 条采样轨迹，坐标单位：米（105x68 球场）
    蓝色=主队，红色=客队；圆点=起点，三角=终点
    """
    fig, axes = plt.subplots(1, n_show, figsize=(6 * n_show, 4))
    for k, ax in enumerate(axes[:n_show]):
        ax.add_patch(patches.Rectangle((0, 0), 105, 68, fill=False, ec='white', lw=2))
        ax.set(facecolor='#2d5a27', xlim=(-3, 108), ylim=(-3, 71), aspect='equal',
               title=f"Scenario {k+1}", xticks=[], yticks=[])
        for i in range(trajectories.shape[1]):
            traj = trajectories[k, i]
            color = '#5599ff' if i < 11 else '#ff5544'
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.4, lw=1.2)
            ax.scatter(*traj[0],  c=color, s=50, zorder=5)
            ax.scatter(*traj[-1], c=color, s=80, marker='^', zorder=5)
    plt.tight_layout()
    plt.savefig('tactics.png', dpi=150, facecolor='#111122')
```

**预期效果**：并排展示同一战术局面的 3 种不同演化路径，直观体现扩散模型的多样性——这是确定性模型完全做不到的。

## 实验

### 数据集现实

| 数据来源 | 可获取性 | 适用性 |
|---------|---------|--------|
| TRACAB / Hawk-Eye | 商业授权，价格昂贵 | 论文使用 |
| Metrica Sports Sample | 公开（仅 2 场） | 原型验证 |
| StatsBomb Open Data | 部分免费 | 仅有事件数据，无追踪坐标 |
| SoccerTrack | 学术开放 | 业余联赛，低帧率 |

**核心壁垒**：论文能区分 A-League 与德甲的风格，说明他们访问了大量专业联赛数据。独立复现时，2 场公开比赛远不够训练泛化模型。

### 定量评估（TacBench）

| 指标 | 说明 | GenTac | 确定性基线 |
|------|-----|--------|-----------|
| ADE (m) | 平均位移误差 | ~0.8 | ~1.2 |
| FDE (m) | 终点位移误差 | ~1.5 | ~2.1 |
| Team Consistency | 队形结构保持度 | 0.89 | 0.71 |
| Style Accuracy | 联赛风格区分准确率 | 82% | N/A |

*以上为论文数字大致范围，具体请参考原文*

## 工程实践

### 实际部署考虑

- **推理速度**：DDPM 1000 步去噪，生成一批轨迹约 2-5 秒（V100）。用 DDIM 可压缩到 50 步，速度提升 20x，但多样性轻微下降
- **内存**：22 球员 × 50 帧 × 2 坐标本身轻量，瓶颈在注意力计算（O(N²)，N=22 可接受）
- **实时性**：目前难以做到比赛进行中实时生成，适合赛前/赛后分析

### 常见坑

**坑 1：坐标归一化方向不一致**
```python
# 错误：主客队进攻方向相反，模型学到混乱的"方向感"
# 正确：统一为"进攻方从左到右"，客队数据需翻转
def normalize_direction(traj, is_away):
    if is_away:
        traj = traj.clone()
        traj[..., 0] = 105 - traj[..., 0]   # x 轴翻转
    return (traj / torch.tensor([105., 68.])) * 2 - 1
```

**坑 2：扩散步数采样不均衡**
```python
# DDPM 训练质量主要由低噪声阶段（小 t）决定，应对小 t 值过采样
# 简单改进：用截断均匀分布偏向低 t
t = torch.randint(0, n_steps // 2, (B,))     # 50% 样本来自前半段
```

**坑 3：定位球数据污染训练集**
```python
# 角球/任意球有高度结构化的队形，与开放场景分布差异大
# 应在预处理时过滤掉定位球后 3 秒内的片段
mask = ~(events['type'].isin(['corner', 'free_kick']))
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 开放场景战术多样性分析 | 定位球（结构过于固定） |
| 反事实战术推演（赛前/赛后） | 比赛中实时决策（推理慢）|
| 联赛/球队风格对比研究 | 训练数据不足（<100 场）|
| 期望威胁的反事实量化 | 极端情况（红牌后 10 打 11）|

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| Transformer 确定性预测 | 推理快，长程依赖建模好 | 只输出单条轨迹 | 短期精确预测 |
| GAN 轨迹生成 | 采样快 | 模式崩溃，训练不稳定 | 行人轨迹 |
| RL 战术生成 | 可优化特定目标 | 需要奖励函数，难设计 | 游戏 AI |
| **GenTac（扩散）** | 多样性好，条件控制强 | 推理慢，数据需求大 | 战术分析平台 |

## 我的观点

**GenTac 解决了一个真实存在的问题**：足球战术的本质就是概率性的，用确定性模型去预测它本来就是在骗自己。扩散模型在这里的使用是自然且合理的，把"生成一条最优轨迹"变成"采样一个合理分布"，这个范式转变是实质性的。

**但实际落地有三道坎**：

一是**数据壁垒**。光学追踪数据是真正的护城河。没有专业联赛授权，独立复现几乎不可能达到论文效果，开源社区的追踪数据质量和数量都差太多。

二是**推理速度**。1000 步去噪在实时分析场景（比赛中场休息的 15 分钟）勉强够用，但真正的边线实时分析还需要更激进的加速方案。

三是**评估困难**。什么叫"好的战术生成"？TacBench 给出了几何和风格维度的量化，但"战术创意性"这种主观维度目前没有好的量化方法，人工评估仍然必要。

**值得关注的方向**：

- 把生成轨迹和 Expected Threat (xT) 模型结合，直接量化每条生成战术的进攻价值
- 跨运动迁移（论文提到篮球、美式足球）如果效果真的好，说明模型学到的是"团队协作几何"的底层规律，而不是足球特定的模式——这是更有意思的科学发现
- 用扩散模型生成的反事实轨迹来训练教练辅助 AI，把分析和决策支持闭环起来

对有追踪数据授权的俱乐部技术团队，这个方向值得投入。对于没有数据的学术研究者，核心架构创新（多智能体扩散 + 团队结构约束 + 离散事件条件）可以用公开数据集在小规模问题上验证，然后寻求产业合作。