---
layout: post-wide
title: "流式3D重建的长程注意力：HorizonStream 架构解析"
date: 2026-05-25 12:06:04 +0800
category: Tools
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.23889v1
generated_by: Claude Code CLI
---

## 一句话总结

通过将几何证据传播分解为多时间尺度线性注意力 + 局部精确匹配，HorizonStream 用 **恒定内存** 处理超过 10,000 帧序列，根治了现有方法在长序列上的漂移和崩溃问题。

---

## 问题根源：流式重建为什么这么难？

在线3D重建必须满足两个苛刻约束：

- **因果性**：只能依赖过去帧，不能用未来信息
- **有界内存**：不能随序列增长无限堆积历史

现有三类方案各有致命缺陷：

**滑动窗口（Sliding Window）**：窗口边界产生硬截断。全局尺度信息（防止漂移的锚点）一旦滑出窗口就彻底消失。

**无门控循环（Ungated Recurrence）**：$h_t = h_{t-1} + f(x_t)$ 没有遗忘机制，历史信息无限叠加导致**缓存饱和**——新帧的影响趋近于零。

**因果 Softmax Attention**：归一化机制在长序列下产生**注意力汇聚（attention sink）**，权重极度集中在序列起始 token，后续帧贡献趋近于零。

这三类问题被作者统一抽象为**证据影响核（Evidence Influence Kernel）**的失配。

---

## 核心抽象：证据影响核

定义几何证据的时间影响：

$$K(t, s) = \text{帧 } s \text{ 的几何证据对帧 } t \text{ 的影响强度}$$

好的 kernel 需要同时满足：
- 短程精确：支持 2D/3D 特征精确匹配（短时间距，高精度）
- 长程持续：传播全局尺度和累积位姿（长时间距，低频信号）
- 有界：不随序列长度发散

HorizonStream 的核心创新是**显式分解**这个 kernel：

$$K(t, s) = K_{\text{long}}(t, s) \cdot K_{\text{short}}(t, s)$$

- $K_{\text{long}}$：Geometric Linear Attention，多时间尺度指数衰减
- $K_{\text{short}}$：Geometric Local Attention + SpatioTemporal RoPE

---

## 组件一：Geometric Linear Attention

### 从线性注意力到多尺度衰减

标准线性注意力把 Softmax 替换为核函数近似，得到 O(1) 内存的循环形式：

$$h_t = h_{t-1} + K_t^\top V_t, \quad o_t = Q_t h_t$$

这等价于无限累加历史 KV 对——即缓存饱和问题。

改进：每个通道学习独立衰减率 $\lambda_c \in (0,1)$：

$$h_t = \operatorname{diag}(\lambda) \cdot h_{t-1} + K_t^\top V_t$$

衰减率小的通道遗忘快（捕捉短程对应），衰减率大的通道遗忘慢（传播全局尺度）——这就是多时间尺度的关键机制。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GeometricLinearAttention(nn.Module):
    """
    带通道级衰减的线性注意力
    内存复杂度: O(H * d^2), 与序列长度无关
    """
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # 跨越多个时间尺度初始化衰减率
        # linspace(-3, -0.1) 经 sigmoid → [0.05, 0.48]: 快速到慢速遗忘
        self.log_decay = nn.Parameter(
            torch.linspace(-3, -0.1, dim_head).unsqueeze(0).repeat(heads, 1)
        )  # (H, d)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # ELU+1 保证 Q, K > 0，是线性注意力近似成立的条件
        q, k = F.elu(q) + 1, F.elu(k) + 1

        decay = torch.sigmoid(self.log_decay)  # (H, d), 每通道独立衰减率
        if state is None:
            state = torch.zeros(B, self.heads, self.dim_head, self.dim_head,
                                device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            kt, vt, qt = k[:,:,t], v[:,:,t], q[:,:,t]
            # 衰减旧状态 + 积累新 KV 外积
            state = decay.unsqueeze(0).unsqueeze(-1) * state + \
                    torch.einsum('bhd,bhe->bhde', kt, vt)
            outputs.append(torch.einsum('bhd,bhde->bhe', qt, state))

        out = rearrange(torch.stack(outputs, dim=2), 'b h n d -> b n (h d)')
        return self.to_out(out), state
```

> **性能注意**：推理时 for 循环是真正的 O(1) 内存；训练时应替换为并行前缀扫描（参考 Mamba 的 `selective_scan` CUDA kernel），否则反向传播中 Python 循环会成为瓶颈。

---

## 组件二：Geometric Local Attention + SpatioTemporal RoPE

长程注意力处理全局尺度，短程注意力负责精确的 3D 特征匹配。后者需要知道特征点在**空间**和**时间**上的精确位置。

### SpatioTemporal RoPE

标准 RoPE 对 1D 序列位置做旋转编码。3D 重建中每个特征有时空坐标 $(t, x, y, z)$，需要 4D 扩展。

做法：把 `dim_head` 均分为 4 段，分别对 $t, x, y, z$ 应用独立的旋转编码。

```python
class SpatioTemporalRoPE(nn.Module):
    """4D 时空旋转位置编码：t, x, y, z 各占 dim_head//4 通道"""
    def __init__(self, dim_head, base=10000):
        super().__init__()
        half = dim_head // 8
        freqs = 1.0 / (base ** (torch.arange(0, half*2, 2).float() / (half*2)))
        self.register_buffer('freqs', freqs)

    def _rotate(self, x, pos):
        """x: (..., d), pos: (...) → 旋转后的 x"""
        angles = pos.unsqueeze(-1) * self.freqs   # (..., half)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * angles.cos() - x2 * angles.sin(),
                          x1 * angles.sin() + x2 * angles.cos()], dim=-1)

    def forward(self, q, k, t_pos, xyz_pos):
        """q, k: (B,H,N,D); t_pos: (B,N); xyz_pos: (B,N,3)"""
        d = q.shape[-1] // 4
        positions = [t_pos, xyz_pos[...,0], xyz_pos[...,1], xyz_pos[...,2]]
        q_parts = list(q.split(d, dim=-1))
        k_parts = list(k.split(d, dim=-1))
        for i, pos in enumerate(positions):
            # 广播到 (B, 1, N) 以匹配 (B, H, N, d/4)
            p = pos.unsqueeze(1)
            q_parts[i] = self._rotate(q_parts[i], p)
            k_parts[i] = self._rotate(k_parts[i], p)
        return torch.cat(q_parts, -1), torch.cat(k_parts, -1)
```

```python
class GeometricLocalAttention(nn.Module):
    """窗口局部注意力 + 时空 RoPE，避免 attention sink"""
    def __init__(self, dim, heads=8, dim_head=64, window_size=16):
        super().__init__()
        self.heads, self.dim_head, self.W = heads, dim_head, window_size
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.rope = SpatioTemporalRoPE(dim_head)
        self.scale = dim_head ** -0.5

    def forward(self, x, t_pos, xyz_pos):
        B, T, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q, k = self.rope(q, k, t_pos, xyz_pos)  # 注入时空位置

        outputs = []
        for t in range(T):
            s = max(0, t - self.W + 1)   # 因果窗口起点（无 padding，无伪影响）
            attn = torch.einsum('bhd,bhjd->bhj', q[:,:,t], k[:,:,s:t+1]) * self.scale
            out = torch.einsum('bhj,bhjd->bhd', attn.softmax(-1), v[:,:,s:t+1])
            outputs.append(out)

        out = rearrange(torch.stack(outputs, dim=2), 'b h n d -> b n (h d)')
        return self.to_out(out)
```

---

## 组件三：Metric Readout Token

位姿估计需要**绝对尺度**（metric scale），而不是相对尺度。方案：引入可学习的特殊 token，通过 cross-attention 从线性注意力的持久状态中读取全局几何信息。

```python
class MetricReadoutToken(nn.Module):
    """
    从持久几何状态恢复绝对位姿和尺度
    设计逻辑：线性注意力状态 h 编码了全局场景结构，readout token 提取之
    """
    def __init__(self, dim, num_tokens=4, heads=4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim//2), nn.GELU(),
            nn.Linear(dim//2, 7)   # quaternion(4) + translation(3)
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Softplus()  # 保证正值
        )

    def forward(self, frame_feat, state_summary):
        """state_summary: 将线性注意力状态 h 投影到序列形式 (B, N_s, D)"""
        B = frame_feat.shape[0]
        tokens = self.tokens.expand(B, -1, -1)
        context = torch.cat([frame_feat, state_summary], dim=1)
        readout, _ = self.cross_attn(tokens, context, context)
        pooled = readout.mean(dim=1)   # (B, D)
        return self.pose_head(pooled), self.scale_head(pooled)
```

---

## 组装：完整 Block

```python
class HorizonStreamBlock(nn.Module):
    """= 长程线性注意力 + 短程局部注意力 + FFN"""
    def __init__(self, dim, heads=8, dim_head=64, window_size=16):
        super().__init__()
        self.long_range  = GeometricLinearAttention(dim, heads, dim_head)
        self.short_range = GeometricLocalAttention(dim, heads, dim_head, window_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)
        )

    def forward(self, x, t_pos, xyz_pos, state=None):
        long_out, new_state = self.long_range(x, state)   # O(1) 内存
        x = x + long_out
        x = x + self.short_range(x, t_pos, xyz_pos)      # O(W) 内存
        x = x + self.ffn(x)
        return x, new_state
```

---

## 性能对比（论文 benchmark）

| 方法 | 推理内存 | 时间复杂度 | 长序列稳定性 | 绝对尺度 |
|------|---------|-----------|------------|---------|
| 滑动窗口 Transformer | O(W·d) | O(T·W) | 全局信息截断，漂移 | 差 |
| 无门控循环 | O(d²) | O(T) | 缓存饱和后崩溃 | 差 |
| 因果 Softmax Attn | O(T·d) | O(T²) | attention sink | 中 |
| **HorizonStream** | **O(d²)** | **O(T)** | **10,000帧稳定** | **好** |

关键结论：仅用 **48帧训练片段** 泛化到 **10,000+ 帧推理**，内存消耗恒定不增长。这是滑动窗口方法从结构上无法做到的。

---

## 常见陷阱

### 陷阱一：衰减率初始化

```python
# 错误：相同初始值 → 单时间尺度，无法同时处理短程匹配和长程传播
self.log_decay = nn.Parameter(torch.zeros(heads, dim_head))  # ❌

# 正确：覆盖多个时间尺度
self.log_decay = nn.Parameter(
    torch.linspace(-3, -0.1, dim_head).unsqueeze(0).repeat(heads, 1)
)  # ✅
```

### 陷阱二：Local Attention 的 padding

```python
# 错误：用 0 填充使窗口对齐，导致边界帧受到虚假的"空帧"影响
k_padded = F.pad(k, (0,0,W-1,0))  # ❌

# 正确：用 max(0, t-W+1) 自然限制有效范围，见上方代码实现  ✅
```

### 陷阱三：训练时的 for 循环瓶颈

线性循环 $h_t = \lambda h_{t-1} + x_t$ 在训练时应用并行前缀扫描，否则反向传播的内存占用是 O(T)（需要存储所有中间状态）。

```python
# 推理：sequential（真 O(1) 内存）
# 训练：parallel scan（参考 Mamba selective_scan 的 CUDA 实现）
# 两者数学等价，只是计算图展开方式不同
```

---

## 适用边界

| 适用场景 | 不适用场景 |
|---------|-----------|
| 机器人导航（序列无限增长） | 静态场景一次重建（NeRF/3DGS 更合适） |
| AR/VR 实时追踪定位 | 离线处理（可用双向 Attention，效果更好） |
| 自动驾驶连续地图构建 | 纹理极稀少场景（特征匹配本身失效） |
| 需要绝对尺度输出的系统 | 嵌入式端侧部署（cross-attn 有额外参数） |

---

## 延伸阅读

- **Mamba / S4**：线性循环的高效 CUDA 实现，HorizonStream 的线性注意力训练加速可直接借鉴其 `selective_scan` kernel
- **RetNet**（arxiv 2307.08621）：语言模型中类似的多尺度指数衰减思路，可对比两个领域的不同应用
- **DROID-SLAM**：经典流式3D重建 baseline，理解 HorizonStream 改进的参考起点
- 官方项目页面（含代码和预训练模型）：[https://3dagentworld.github.io/horizonstream/](https://3dagentworld.github.io/horizonstream/)