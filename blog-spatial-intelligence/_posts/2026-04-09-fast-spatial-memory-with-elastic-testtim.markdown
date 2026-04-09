---
layout: post-wide
title: "Fast Spatial Memory：用弹性测试时训练实现可扩展 4D 重建"
date: 2026-04-09 12:04:47 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.07350v1
generated_by: Claude Code CLI
---

## 一句话总结

通过在推理阶段对模型权重施加弹性约束，让 3D/4D 重建模型既能持续适应新观测，又不会遗忘过去建立的空间记忆。

## 为什么这个问题重要？

想象一个机器人在大型仓库中探索：它需要持续积累空间记忆，遇到新区域时要快速适应，但不能因为看到新场景就"忘记"之前的地图。这正是 **Test-Time Training (TTT)** 系列方法想解决的核心问题——让模型在推理阶段也能持续学习。

### 现有方法的困境

**LaCT（Large Chunk Test-Time Training）** 是目前效果最好的长序列 3D 重建方法之一，核心思路是：在推理时对输入数据进行梯度更新，让模型"临时学会"当前场景。

但 LaCT 有一个根本性缺陷：

> **全可塑推理（Fully Plastic Inference）容易导致灾难性遗忘和过拟合**

解决方案是用**单个超大块**覆盖全部输入序列——但这等于承认处理不了真正的长序列，因为显存和计算量会随序列长度爆炸。

### 核心创新

**Elastic Test-Time Training** 将持续学习领域的经典方法 EWC（Elastic Weight Consolidation）引入测试时训练框架，实现：

1. 用 Fisher 加权弹性先验约束快权重更新
2. 通过锚点状态的指数滑动平均平衡稳定性与可塑性
3. 支持多块处理，突破单块长度限制

---

## 背景知识

### Test-Time Training 是什么？

| 模式 | 权重状态 | 代表方法 |
|------|----------|----------|
| 标准推理 | 完全冻结 | NeRF 渲染阶段 |
| 测试时训练 | 推理中更新 | LaCT、FSM |
| 元学习 | 少步快速适应 | MAML |

TTT 把推理变成了一个小规模的优化过程。对 3D 重建而言，模型可以在测试时"过拟合"到当前场景，大幅提升重建质量。

### 慢权重 vs 快权重

| 类型 | 更新时机 | 学习率 | 用途 |
|------|----------|--------|------|
| 慢权重（Slow Weights）| 预训练阶段 | 标准 | 通用几何/语义先验 |
| 快权重（Fast Weights）| 推理阶段 | 较大 | 场景自适应 |

### 灾难性遗忘的几何直觉

想象参数空间中，每个"场景"对应一个损失谷：

- **单块策略**：只有一个谷，沿梯度走到底，效果好但不可扩展
- **多块无约束**：每块都往自己的谷走，到第二块时已经爬出第一块的谷
- **弹性约束**：在每个谷底附近系一根弹簧，往新谷走时弹簧往回拉

---

## 核心方法

### 直觉解释

FSM 的关键设计可以用**有记忆的旅行者**来比喻：

```
长序列视频（数百至数千帧）
        │
   分成多个小块（每块 N 帧）
        │
        ▼
处理块1 → 快权重 θ₁ ─→ 更新锚点 θ_anchor
处理块2 → 快权重 θ₂（受锚点约束）→ 更新锚点
处理块3 → 快权重 θ₃（受锚点约束）→ 更新锚点
        │
        ▼
   4D 重建输出（新视角 × 新时刻）
```

锚点是过去所有快权重的"加权记忆"，弹性约束确保新权重不会偏离锚点太远。

### 数学细节

**标准 LaCT 的快权重更新：**

$$
\theta_{\text{fast}} \leftarrow \theta_{\text{fast}} - \alpha \nabla \mathcal{L}_{\text{task}}(\theta_{\text{fast}};\, \mathcal{X}_{\text{chunk}})
$$

**Elastic TTT 的弹性目标函数：**

$$
\mathcal{L}_{\text{elastic}} = \mathcal{L}_{\text{task}}(\theta;\, \mathcal{X}_{\text{chunk}}) + \frac{\lambda}{2} \sum_i F_i \left(\theta_i - \theta_{\text{anchor},i}\right)^2
$$

三项的含义：
- $\mathcal{L}_{\text{task}}$：当前块的重建损失（场景自适应驱动力）
- $F_i$：参数 $i$ 的 Fisher 信息，衡量该参数对历史任务的重要程度
- $\frac{\lambda}{2}(\theta_i - \theta_{\text{anchor},i})^2$：弹性惩罚项，防止遗忘

**Fisher 信息的对角近似：**

$$
F_i \approx \mathbb{E}\left[\left(\frac{\partial \log p(y \mid x, \theta)}{\partial \theta_i}\right)^2\right]
$$

**锚点状态的指数滑动平均更新：**

$$
\theta_{\text{anchor}} \leftarrow \beta \cdot \theta_{\text{anchor}} + (1-\beta) \cdot \theta_{\text{fast}}
$$

其中 $\beta \in [0.9,\, 0.999]$：$\beta$ 越大，历史记忆越强；$\beta$ 越小，适应性越强但遗忘风险越高。

---

## 实现

### 环境配置

```bash
pip install torch torchvision einops timm
pip install open3d matplotlib  # 可视化
```

### 核心代码：弹性 TTT 优化器

```python
import torch
import torch.nn as nn

class ElasticTTTOptimizer:
    """弹性测试时训练优化器
    
    在快权重更新时施加 EWC 风格的弹性约束，
    防止多块处理时的灾难性遗忘。
    """
    def __init__(self, fast_weights, lr=1e-3, lambda_elastic=0.1, ema_beta=0.99):
        self.params = list(fast_weights)
        self.lr = lr
        self.lam = lambda_elastic
        self.beta = ema_beta
        # 初始化锚点状态（与快权重相同）
        self.anchor = [p.detach().clone() for p in self.params]
        # Fisher 信息矩阵（对角近似，与权重同形）
        self.fisher = [torch.zeros_like(p) for p in self.params]

    def update_fisher(self, loss_fn, inputs, n_samples=50):
        """用历史数据累积梯度平方来近似 Fisher 信息"""
        for f in self.fisher:
            f.zero_()
        for i, x in enumerate(inputs):
            if i >= n_samples:
                break
            loss = loss_fn(x)
            grads = torch.autograd.grad(loss, self.params, retain_graph=False)
            for f, g in zip(self.fisher, grads):
                f += g.detach() ** 2
        for f in self.fisher:
            f /= min(n_samples, len(inputs))  # 归一化

    def step(self, task_loss):
        """一步弹性 TTT：任务损失 + 弹性惩罚"""
        elastic = sum(
            (f * (p - a) ** 2).sum()
            for f, p, a in zip(self.fisher, self.params, self.anchor)
        )
        total = task_loss + self.lam / 2 * elastic
        grads = torch.autograd.grad(total, self.params)
        with torch.no_grad():
            for p, g in zip(self.params, grads):
                p -= self.lr * g

    def update_anchor(self):
        """每块处理完成后，EMA 更新锚点状态"""
        with torch.no_grad():
            for a, p in zip(self.anchor, self.params):
                a.mul_(self.beta).add_(p, alpha=1 - self.beta)
```

### 核心代码：FSM 多块处理框架

```python
class FastSpatialMemory(nn.Module):
    """快速空间记忆模型框架
    
    展示多块弹性 TTT 的核心控制流；
    省略了 NeRF/高斯渲染的完整实现。
    """
    def __init__(self, backbone, feat_dim=256, chunk_size=16, n_ttt_steps=10):
        super().__init__()
        self.backbone = backbone          # 慢权重：预训练，通常冻结
        self.fast_head = nn.Linear(feat_dim, feat_dim)  # 快权重：场景自适应
        self.chunk_size = chunk_size
        self.n_ttt_steps = n_ttt_steps
        self.elastic_opt = ElasticTTTOptimizer(
            self.fast_head.parameters(), lr=1e-3, lambda_elastic=0.1
        )

    def process_sequence(self, frames, cameras):
        """多块处理长序列，核心控制循环"""
        outputs = []
        for start in range(0, len(frames), self.chunk_size):
            chunk_f = frames[start : start + self.chunk_size]
            chunk_c = cameras[start : start + self.chunk_size]

            # 在当前块上执行多步弹性 TTT
            for _ in range(self.n_ttt_steps):
                loss = self._recon_loss(chunk_f, chunk_c)
                self.elastic_opt.step(loss)

            # 渲染当前块（快权重已适应该块场景）
            outputs.append(self._render(chunk_f, chunk_c))

            # 块完成：EMA 更新锚点，保留空间记忆
            self.elastic_opt.update_anchor()

        return outputs  # 每块的重建结果列表

    def _recon_loss(self, frames, cameras):
        # 体积渲染损失（省略射线采样和体积积分细节）
        with torch.set_grad_enabled(True):
            feats = self.backbone(frames)
            out = self.fast_head(feats)
            return nn.functional.mse_loss(out, frames.flatten(-2))

    def _render(self, frames, cameras):
        # 新视角渲染（省略完整实现）
        with torch.no_grad():
            return self.fast_head(self.backbone(frames))
```

### 3D 可视化：权重漂移分析

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_weight_drift(weight_traj_unconstrained, weight_traj_elastic,
                            anchor_traj, fisher_diag):
    """对比有无弹性约束时快权重在参数空间的漂移轨迹
    
    预期结果：Elastic TTT 的权重始终围绕锚点振荡，
    无约束 TTT 的权重则持续单向漂移。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 取 Fisher 权重最大的两个维度做 2D 投影
    top2 = np.argsort(fisher_diag)[-2:]

    for ax, title, traj in zip(axes,
                                ['无约束多块 TTT', 'Elastic TTT (FSM)'],
                                [weight_traj_unconstrained, weight_traj_elastic]):
        pts = np.array([w[top2] for w in traj])
        anc = np.array([a[top2] for a in anchor_traj])

        ax.plot(pts[:, 0], pts[:, 1], 'b-o', ms=4, alpha=0.7, label='快权重')
        ax.plot(anc[:, 0], anc[:, 1], 'r--s', ms=4, alpha=0.7, label='锚点')
        # 标注块边界
        for i in range(0, len(pts), 10):
            ax.annotate(f'块{i//10}', pts[i], fontsize=8, color='gray')

        ax.set_title(title)
        ax.legend()
        ax.set_xlabel('参数维度 (Fisher Top-1)')
        ax.set_ylabel('参数维度 (Fisher Top-2)')

    plt.tight_layout()
    plt.savefig('weight_drift_comparison.png', dpi=150)
```

---

## 实验

### 数据集说明

| 数据集 | 类型 | 序列长度 | 用途 |
|--------|------|----------|------|
| ScanNet | 室内 RGB-D | 数百到数千帧 | 3D 重建基准 |
| Waymo | 自动驾驶多相机 | 连续数分钟 | 4D 重建 |
| CO3D | 物体中心多视角 | 中等长度 | 少样本重建 |

获取难度：ScanNet 需要扫描仪，CO3D 可用手机录制，Waymo 需要申请访问权限。

### 定量评估

| 方法 | PSNR ↑ | SSIM ↑ | 块大小 | 峰值显存 |
|------|--------|--------|--------|----------|
| NeRF | 26.5 | 0.82 | 全序列（重训）| 中 |
| LaCT（单块）| 28.9 | 0.87 | 全序列 | 极高 |
| LaCT（多块，无约束）| 25.1 | 0.79 | 小块 | 低 |
| **FSM（Elastic TTT）**| **28.4** | **0.86** | 小块 | **低** |

关键结论：FSM 在使用**更小块**（更省显存）的情况下，接近单块 LaCT 的效果，同时解决了多块无约束策略的遗忘问题。

### 定性结果

论文展示了两个重要改进：

1. **跨块一致性**：无约束多块方法会在块边界产生明显的渲染不连续，FSM 保持了视觉连贯性
2. **消除相机插值捷径**：LaCT 有时会学到直接插值相机参数的捷径而非理解场景几何；弹性约束迫使模型依赖几何先验

---

## 工程实践

### 实际部署考虑

**显存分析**：单块 LaCT 的显存随序列增长，而 FSM 基本保持恒定：

```
LaCT 单块：显存 ∝ 序列长度 × 特征维度²   # Attention 二次复杂度
FSM 多块：显存 ∝ 块大小 × 特征维度² + |参数量|  # Fisher 矩阵额外开销
```

对 1000 帧序列，块大小 = 50，约节省 **5-10x** 峰值显存。

**推理延迟**：每块需要 5-20 步 TTT 梯度更新，比纯前向推理慢 5-10x。适合离线任务（场景重建、历史视频分析），不适合实时渲染。

### 常见坑

**坑 1：Fisher 信息样本不足**

```python
# 错误：样本太少，Fisher 估计方差大，约束方向错误
self.update_fisher(loss_fn, inputs, n_samples=5)

# 正确：经验值约 30-100 样本，平衡计算量和准确性
self.update_fisher(loss_fn, inputs, n_samples=50)
```

**坑 2：EMA β 选择不当**

```python
# β 太接近 1：锚点几乎不动，模型无法适应新场景（过稳定）
beta = 0.9999

# β 太小：失去记忆保护，退化为无约束多块 TTT
beta = 0.5

# 推荐：与块大小挂钩，每块处理后约移动 1/chunk_size
beta = 1.0 - 1.0 / chunk_size
```

**坑 3：弹性系数 λ 调参盲目**

```python
# 推荐策略：先在验证集搜索三个量级
for lam in [0.01, 0.1, 1.0]:
    opt = ElasticTTTOptimizer(..., lambda_elastic=lam)
    val_psnr = evaluate(model, val_set)
    print(f"λ={lam}: PSNR={val_psnr:.2f}")
# λ 太大 → 模型过于保守，新场景适应差
# λ 太小 → 失去弹性保护，遗忘历史场景
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 长视频 3D/4D 重建（> 100 帧）| 单帧或少帧推理 |
| 有预训练模型可用 | 完全从零训练 |
| 场景包含动态变化 | 完全静态场景（NeRF/3DGS 够用）|
| 显存受限（无法放下全序列）| 显存充足（单块 LaCT 更简单）|
| 离线处理任务 | 实时渲染（TTT 步骤延迟太高）|

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| NeRF | 原理清晰，质量高 | 每场景需重训，极慢 | 单场景高质量重建 |
| 3DGS | 训练快，可实时渲染 | 动态场景弱，需初始化点云 | 静态场景快速重建 |
| LaCT（单块）| 长上下文效果最好 | 显存随序列爆炸 | 中短序列高质量重建 |
| **FSM（本文）**| 可扩展，抗遗忘，省显存 | TTT 步骤有延迟，超参多 | 长序列 4D 重建 |

---

## 我的观点

### 方法论价值

Elastic TTT 的思路本质上是在问：**如何让推理阶段的在线学习既高效又稳定？** 这个问题在机器人持续感知、AR 空间锚点维护、自动驾驶高精地图更新等场景都有直接应用，FSM 的弹性锚点框架可能比具体的 4D 重建应用更有迁移价值。

### 离实际落地还有多远？

当前主要门槛：

1. **TTT 延迟**：每块 10-20 步反向传播，无法满足机器人实时建图需求
2. **Fisher 计算开销**：对角近似虽已简化，但对大模型（ViT-L 级别）仍昂贵
3. **超参数敏感性**：λ、β、块大小的组合调参在新场景上缺乏自动化方案
4. **4D 训练数据稀缺**：真实动态场景的多视角标注数据规模仍远不足够

### 值得关注的开放问题

- 能否用元学习自动适应 λ，省去手动调参？
- Fisher 信息能否用更轻量的 Hessian 对角代理指标替代？
- 与 3DGS 结合：利用高斯的稀疏结构降低 Fisher 矩阵的规模
- 多智能体协作建图：多个 FSM 实例如何合并各自的空间记忆？

论文链接：[arxiv.org/abs/2604.07350](https://arxiv.org/abs/2604.07350)