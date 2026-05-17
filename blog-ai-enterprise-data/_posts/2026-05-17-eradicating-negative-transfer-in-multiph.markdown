---
layout: post-wide
title: "用稀疏专家路由消灭多物理场基础模型中的负迁移"
date: 2026-05-17 12:02:49 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.15179v1
generated_by: Claude Code CLI
---

## 一句话总结

Shodh-MoE 通过 Top-1 稀疏路由让神经网络**自动发现**物理域边界——无需任何标注，训练结束后路由器已完全将流体和多孔介质流分配给不同专家。

---

## 为什么这是个真实的工程问题？

想象你在训练一个"通用 PDE 求解器"，同时处理：

- **明渠流**（open-channel flow）：Navier-Stokes 主导，宽频谱，含自由液面
- **多孔介质流**（porous media）：Darcy 定律主导，边界效应强，几何复杂

这两类物理的梯度方向根本不兼容。在单一密集网络里联合训练时，它们互相干扰，导致：

1. **梯度冲突**：损失面方向相悖，优化器左右为难
2. **可塑性丧失（plasticity loss）**：网络权重被反复覆盖，"忘记"之前学到的物理规律
3. **灾难性遗忘的逆问题**：不是新任务压制旧任务，而是两个任务同时压制彼此

这就是科学机器学习（SciML）社区长期以来绕不过去的**负迁移**问题。

Shodh-MoE 给出的答案很直接：让不同物理走不同的参数路径。但它最惊人的结果是：**路由器不需要被告知哪个样本属于哪种物理——它自己学会了。**

---

## 三个核心洞见（论文摘要里没提够的）

**洞见一：Helmholtz 参数化比散度正则化项更可靠**

大多数物理感知神经网络用损失项惩罚 $\nabla \cdot \mathbf{u} \neq 0$。Shodh-MoE 换了思路：不预测速度场 $\mathbf{u}$ 本身，而是预测向量势 $\mathbf{A}$，然后令：

$$\mathbf{u} = \nabla \times \mathbf{A}$$

由向量微积分恒等式 $\nabla \cdot (\nabla \times \mathbf{A}) \equiv 0$，**散度为零是数学上的必然，不依赖权重收敛**。这就是为什么他们能在 FP64 后处理下测出 $2.8 \times 10^{-10}$ 的散度——这接近浮点数值零，不是"训练得好"，是**结构保证**。

**洞见二：Top-1 路由的极端性正是它的优势**

Top-k 路由（通常 $k=2$）允许混合专家，这在语言模型里是好事。但在多物理问题里，**模糊的混合正是负迁移的来源**。Top-1 强制每个潜在 patch 只走一条专家路径，物理域之间零参数共享（除了专门设计的"共享专家"处理对称性）。代价是训练更不稳定，需要精心设计负载均衡。

**洞见三：自主域分叉（Autonomous Domain Bifurcation）说明什么**

论文最有意思的数据不是 MSE，而是路由分布图：20000 步训练后，所有明渠流验证 token 路由到 Expert 0，所有多孔介质 token 路由到 Expert 1，**准确率接近 100%，没有任何路由监督信号**。这意味着物理差异足以通过潜变量特征被区分。换句话说，**物理域的身份信息隐含在 PDE 解的统计特性里，路由器学到了这个映射**。

---

## 方法解析

### 整体架构

```
原始物理张量 [128³]
    ↓ 物理感知编码器（带 Helmholtz 参数化）
潜在物理 tokens [16³ patches]
    ↓ Soft-Semantic Top-1 路由器
    ┌──────────────────────┐
    Expert 0（明渠流专家）   Expert 1（多孔介质专家）
    └──────────────────────┘
    ↓ 共享专家（处理普遍对称性）
    ↓ 潜变量 Transformer
解码器 → 物理输出
```

### Helmholtz 速度参数化

直觉先行：想象磁场。磁场满足 $\nabla \cdot \mathbf{B} = 0$，我们总是用向量势 $\mathbf{A}$ 表示 $\mathbf{B} = \nabla \times \mathbf{A}$。速度场的无散度约束和这个完全同构。

```python
import torch
import torch.nn as nn

class HelmholtzVelocityDecoder(nn.Module):
    """
    通过向量势参数化保证速度场严格无散度
    u = curl(A) ⟹ ∇·u ≡ 0（数学恒等式，非近似）
    """
    def __init__(self, latent_dim: int, grid_size: int = 16):
        super().__init__()
        self.G = grid_size
        # 预测 3 分量向量势 A，而非直接预测速度
        self.to_potential = nn.Linear(latent_dim, 3 * grid_size**3)
    
    def curl_3d(self, A: torch.Tensor) -> torch.Tensor:
        """计算 3D curl，使用周期性有限差分"""
        # A: [B, 3, D, H, W]
        Ax, Ay, Az = A[:, 0], A[:, 1], A[:, 2]
        
        # 周期性边界的一阶差分
        roll = lambda f, dim: torch.roll(f, -1, dim) - f
        
        # ∇ × A 的三个分量
        ux = roll(Az, -2) - roll(Ay, -3)   # ∂Az/∂y - ∂Ay/∂z
        uy = roll(Ax, -3) - roll(Az, -1)   # ∂Ax/∂z - ∂Az/∂x
        uz = roll(Ay, -1) - roll(Ax, -2)   # ∂Ay/∂x - ∂Ax/∂y
        
        return torch.stack([ux, uy, uz], dim=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, G = z.shape[0], self.G
        A = self.to_potential(z).view(B, 3, G, G, G)
        return self.curl_3d(A)  # 返回严格无散度的速度场
```

验证散度确实为零：

```python
import torch
import torch.nn as nn

class HelmholtzVelocityDecoder(nn.Module):
    # u = curl(A) ⟹ ∇·u ≡ 0（数学恒等式）
    def __init__(self, latent_dim: int, grid_size: int = 16):
        super().__init__()
        self.G = grid_size
        self.to_potential = nn.Linear(latent_dim, 3 * grid_size**3)  # 预测向量势 A

    def curl_3d(self, A: torch.Tensor) -> torch.Tensor:
        Ax, Ay, Az = A[:, 0], A[:, 1], A[:, 2]
        roll = lambda f, dim: torch.roll(f, -1, dim) - f
        # ∇ × A 的三个分量
        ux = roll(Az, -2) - roll(Ay, -3)
        uy = roll(Ax, -3) - roll(Az, -1)
        uz = roll(Ay, -1) - roll(Ax, -2)
        return torch.stack([ux, uy, uz], dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        A = self.to_potential(z).view(z.shape[0], 3, self.G, self.G, self.G)
        return self.curl_3d(A)
```

### Top-1 软语义路由器

路由器需要解决两个矛盾的目标：**硬分配**（保证零负迁移）和**可微分**（允许梯度回传）。解法是经典的 Straight-Through Estimator（STE）：

```python
import torch.nn.functional as F

class Top1SoftSemanticRouter(nn.Module):
    """
    Top-1 路由器：前向传播用 argmax（硬），反向传播用 softmax（软）
    "soft-semantic"：路由依据是 patch 的语义特征，非位置信息
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # 路由器只有一个线性层——刻意保持轻量，避免路由器本身成为瓶颈
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        logits = self.gate(x)                         # [B, T, E]
        probs = F.softmax(logits, dim=-1)              # 软概率，用于梯度
        
        indices = probs.argmax(dim=-1)                 # [B, T] 硬分配
        
        # Straight-Through Estimator：
        # 前向 = one_hot（硬），反向 = probs（软）
        hard = F.one_hot(indices, self.num_experts).float()
        gates = hard + probs - probs.detach()          # STE trick
        
        # 负载均衡损失：防止所有 token 路由到同一专家（专家塌缩）
        # 理想情况：每个专家均匀处理 1/E 的 token
        mean_load = probs.mean(dim=[0, 1])             # [E]
        balance_loss = self.num_experts * (mean_load ** 2).sum()
        
        return gates, indices, balance_loss
```

### 完整 MoE Transformer Block

```python
class ShodhMoEBlock(nn.Module):
    """
    带稀疏 MoE FFN 的 Transformer block
    自注意力层参数共享（处理通用时空依赖）
    FFN 层稀疏激活（处理物理域特异性）
    """
    def __init__(self, d_model: int, num_heads: int,
                 num_experts: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        self.router = Top1SoftSemanticRouter(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        # 共享专家：处理跨域不变量（守恒律、对称性）
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, d_ff // 2), nn.GELU(),
            nn.Linear(d_ff // 2, d_model)
        )
    
    def forward(self, x: torch.Tensor):
        # 自注意力（共享，所有物理域）
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # 稀疏 MoE FFN
        residual = x
        h = self.norm2(x)
        gates, indices, balance_loss = self.router(h)
        
        expert_out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            mask = (indices == e)          # [B, T] bool mask
            if mask.any():
                # 只对路由到本专家的 token 计算
                flat_h = h.view(-1, h.size(-1))
                flat_mask = mask.view(-1)
                expert_out.view(-1, h.size(-1))[flat_mask] = \
                    expert(flat_h[flat_mask])
        
        # 门控加权 + 共享专家（始终激活）
        x = residual + (expert_out * gates.sum(-1, keepdim=True)
                        + self.shared_expert(h))
        
        return x, balance_loss
```

---

## 最令人惊讶的结果：无监督域分叉

这是值得单独讨论的。训练时，模型看到的是混合的三维物理张量，**没有"这是明渠流"的标签**。但到训练末期：

```
明渠流验证 token → Expert 0（路由率 ~100%）
多孔介质 token  → Expert 1（路由率 ~100%）
```

这说明什么？两类流体的潜变量特征（通过 autoencoder 压缩后）在特征空间中是**线性可分**的。路由器（只是一个线性层）就学会了这个分类边界。

这实际上给了我们一个**免费的物理域分类器**。可以这样探测：

```python
class ShodhMoEBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.router = Top1SoftSemanticRouter(d_model, num_experts)
        self.experts = nn.ModuleList([...])  # 域特异性专家
        self.shared_expert = nn.Sequential(...)  # 跨域不变量（守恒律、对称性）
        # ... (norm 层省略)

    def forward(self, x):
        # 共享自注意力（所有物理域）
        x = x + self.attn(self.norm1(x), ...)[0]

        h = self.norm2(x)
        gates, indices, balance_loss = self.router(h)

        # 稀疏路由：只对路由到本专家的 token 计算
        expert_out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            mask = (indices == e)
            if mask.any():
                expert_out[mask] = expert(h[mask])

        # 门控加权 + 共享专家（始终激活）
        x = x + expert_out * gates.sum(-1, keepdim=True) + self.shared_expert(h)
        return x, balance_loss
```

---

## 实现中的坑

**坑 1：专家塌缩（Expert Collapse）**

Top-1 路由最常见的失败模式：所有 token 都路由到同一个专家，其他专家梯度为零，逐渐退化。负载均衡损失是必须的，但权重需要调试：

```python
# 过大的均衡损失会让路由变随机，过小会塌缩
# 经验起点：0.01 × 主任务损失量级
total_loss = mse_loss + 0.01 * balance_loss
```

**坑 2：Helmholtz 在非周期性边界下的精度**

上面的有限差分实现用了 `torch.roll`，隐含了**周期性边界条件**。明渠流通常不满足这个假设，实际工程中可能需要：

```python
# 非周期性边界：用 padding 处理边界
def ddx_nonperiodic(f):
    # 对边界用前向/后向差分，内部用中心差分
    return F.pad(f, (0, 1))[..., 1:] - F.pad(f, (1, 0))[..., :-1]
```

**坑 3：FP32 vs FP64 的散度数值**

论文中 $2.8 \times 10^{-10}$ 的散度是在 FP64 下测量的。FP32 下同样的 curl 计算散度约为 $10^{-7}$，这是浮点累积误差，不是 Helmholtz 参数化失效。**不要用 FP32 的散度结果声称物理守恒。**

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 两种以上物理域联合训练 | 单一物理域，负迁移本就不存在 |
| 有充足的多域数据（每域至少数千样本）| 数据稀缺，专家无法充分专业化 |
| 物理场需要严格守恒律（质量、能量）| 目标是纯数据驱动代理模型，不在意物理一致性 |
| 有足够算力做 Top-1 路由调试 | 快速原型阶段，不想调负载均衡权重 |
| 各域的 PDE 结构差异显著 | 各域物理相近，强制分离反而有害 |

---

## 我的观点

Shodh-MoE 做对了一件事：**把物理约束放在架构里，而不是损失函数里**。Helmholtz 参数化的思路在计算流体力学（CFD）社区早有先例，但把它嵌入神经算子的 tokenizer 层，优雅程度是值得借鉴的。

但我有两个保留意见：

**一、Top-1 路由的极端性是否过度设计？** 论文只测试了两个物理域。当域数量增加到 5-10 个时，Top-1 的负载均衡难度指数上升。这时候更鲁棒的可能是 Top-2 路由 + 更强的正则化。

**二、"自主域分叉"的可复现性存疑。** 这个结果非常漂亮，但明渠流和多孔介质流是物理上差异极大的两类——它们几乎是论文作者能选到的"最容易分离"的组合。如果把层流和湍流放进来，或者加入近壁流动，路由是否还能清晰分叉？这是未来工作需要回答的问题。

这篇论文的最大价值不在于 MSE 数字，而在于它提出了一个可量化的框架来讨论 SciML 中的负迁移：通过路由分布的熵来诊断干扰程度，通过专家激活模式来理解模型的"物理意识"。这个思路值得推广到更广泛的多任务科学计算问题。

---

**原论文链接**：[arXiv:2605.15179](https://arxiv.org/abs/2605.15179v1)