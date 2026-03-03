---
layout: post-wide
title: '深度序列模型的概率化改造：让 Transformer 学会说"我不确定"'
date: 2026-03-03 08:02:59 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.00888v1
generated_by: Claude Code CLI
---

## 一句话总结

这篇论文挖掘了 Transformer 注意力机制与稀疏高斯过程（Sparse GP）之间的深层对应关系，为深度序列模型提供了一条理论严谨、可落地的不确定性量化路径。

---

## 为什么这篇论文重要？

深度序列模型（DSM）的"过自信"问题一直是工业落地的隐患。一个医疗诊断系统、一个金融预测模型，给出 99.8% 的置信度时，你得相信它吗？

现有方法面临两个痛点：

**痛点一：先验设计困难。** 贝叶斯方法需要指定先验，但神经网络的权重空间几乎没有自然的先验选择，通常只能用各向同性高斯凑合。

**痛点二：近似质量差。** 变分推断在大规模深度模型上的近似质量难以保证，尤其是后验分布高度非高斯时。

这篇论文的核心洞见是：**不要把神经网络和概率模型当作两套独立系统——Transformer 的注意力机制本身就是稀疏高斯过程的一种近似实现。** 承认这一点，就可以直接借用 GP 理论来设计先验和后验近似，而不是凭空猜测。

---

## 核心方法一：注意力机制即稀疏 GP

### 直觉建立

稀疏高斯过程的核心操作是：用一组"诱导点"（inducing points）$\mathbf{Z}$ 来压缩表示整个函数。对于新的测试点 $x^*$，预测均值为：

$$\mu(x^*) = k(x^*, \mathbf{Z}) \, K(\mathbf{Z}, \mathbf{Z})^{-1} \mathbf{m}$$

其中 $k$ 是核函数，$\mathbf{m}$ 是诱导点处的函数值。

现在看注意力机制：

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

对比一下，二者的对应关系几乎是逐项的：

| 稀疏 GP | Transformer 注意力 |
|--------|------------------|
| 测试点 $x^*$（查询位置）| Query $Q$ |
| 诱导点位置 $\mathbf{Z}$（参考位置）| Key $K$ |
| 诱导点函数值 $\mathbf{m}$ | Value $V$ |
| 核函数 $k(x^*, \mathbf{Z})$ | $\exp(QK^\top / \sqrt{d_k})$（归一化前）|
| $K(\mathbf{Z},\mathbf{Z})^{-1}$ 的归一化 | softmax 归一化 |

这不是表面的类比。Softmax 归一化对应 RBF 核在诱导点上的归一化：

$$\text{softmax}(QK^\top)_{ij} \approx \frac{k(q_i, k_j)}{\sum_{l} k(q_i, k_l)}$$

### 从类比到贝叶斯推断

一旦建立了这种对应，就可以利用稀疏 GP 的方差公式来估计注意力输出的不确定性：

$$\sigma^2(x^*) = k(x^*, x^*) - k(x^*, \mathbf{Z})\, K(\mathbf{Z},\mathbf{Z})^{-1}\, k(\mathbf{Z}, x^*)$$

这给了我们一个有数学支撑的方式来计算"注意力对这个 token 有多确定"。

### 最小实现示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianAttention(nn.Module):
    """
    基于稀疏 GP 类比的贝叶斯注意力层
    输出预测均值和不确定性（方差）
    """
    def __init__(self, d_model, n_heads, noise_var=1e-3):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.noise_var = noise_var
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_k
        
        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)  # (B, H, T, dk)
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)
        
        # 注意力权重：对应 GP 核的归一化
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        attn_weights = F.softmax(scores, dim=-1)           # (B, H, T, T)
        
        # 预测均值：标准注意力输出
        mean_out = torch.matmul(attn_weights, V)           # (B, H, T, dk)
        
        # 不确定性估计：基于 GP 后验方差
        # k(x*, Z)K(Z,Z)^{-1}k(Z, x*) 的近似
        # 对角核值近似为 1（归一化 RBF），减去注意力权重的平方和
        # var ≈ 1 - sum_j w_j^2，w_j 越集中（越确定），方差越小
        uncertainty = 1.0 - (attn_weights ** 2).sum(dim=-1, keepdim=True)
        uncertainty = uncertainty.expand_as(mean_out) * self.noise_var
        
        # 还原形状
        mean_out = mean_out.transpose(1, 2).contiguous().view(B, T, D)
        uncertainty = uncertainty.transpose(1, 2).contiguous().view(B, T, D)
        
        return mean_out, uncertainty


# 演示：区分"有把握的预测"和"不确定的预测"
def demo_uncertainty():
    model = BayesianAttention(d_model=64, n_heads=4)
    
    # 场景 1：重复模式，注意力应该集中 → 低不确定性
    x_structured = torch.zeros(1, 10, 64)
    x_structured[:, ::2, :] = 1.0   # 每隔一个 token 相同
    
    # 场景 2：随机序列，注意力分散 → 高不确定性
    x_random = torch.randn(1, 10, 64)
    
    with torch.no_grad():
        _, unc_structured = model(x_structured)
        _, unc_random = model(x_random)
    
    print(f"结构化序列平均不确定性: {unc_structured.mean():.4f}")
    print(f"随机序列平均不确定性:   {unc_random.mean():.4f}")

demo_uncertainty()
```

这里的方差估计 $1 - \sum_j w_j^2$ 有清晰的直觉：当注意力权重集中在少数几个 token 上（权重的 $\ell_2$ 范数大）时，模型是"确定"的；当权重均匀分散时，不确定性高。

---

## 核心方法二：HiPPO 诱导点与在线学习

### 在线 GP 的挑战

标准稀疏 GP 有一个隐含假设：诱导点位置 $\mathbf{Z}$ 是固定的，或者通过全局优化确定。这在流式数据场景（时间序列在线预测）中完全失效——历史在不断增长，你的诱导点需要"记住过去"。

HiPPO（High-order Polynomial Projection Operators）提供了一个优雅的解：将历史信号投影到正交多项式基上，通过递推公式实时维护历史的压缩表示：

$$\frac{dc(t)}{dt} = A\, c(t) + B\, x(t)$$

其中矩阵 $A, B$ 由多项式阶数决定，$c(t) \in \mathbb{R}^N$ 是一个固定维度的"记忆向量"。

这篇论文将 HiPPO 状态直接用作 GP 的**域间诱导点**（interdomain inducing points）：$c(t)$ 既是历史的压缩摘要，也是 GP 推断所需的诱导表示。

```python
import numpy as np

def make_hippo_matrix(N):
    """构造 HiPPO-LegT（Legendre Translated）的 A, B 矩阵"""
    A = np.zeros((N, N))
    B = np.zeros(N)
    for n in range(N):
        B[n] = (2*n + 1) ** 0.5
        for k in range(n):
            A[n, k] = -(2*n + 1) ** 0.5 * (2*k + 1) ** 0.5
        A[n, n] = -(n + 1)
    return A, B


class HiPPOState:
    """在线维护历史的 HiPPO 状态（离散化版本）"""
    def __init__(self, N=64, dt=0.01):
        A, B = make_hippo_matrix(N)
        # 前向 Euler 离散化：c_{t+1} = (I + dt*A) c_t + dt*B*x_t
        self.A_disc = np.eye(N) + dt * A
        self.B_disc = dt * B
        self.c = np.zeros(N)
    
    def update(self, x_t: float):
        """输入新的观测值，更新记忆状态"""
        self.c = self.A_disc @ self.c + self.B_disc * x_t
        return self.c.copy()   # 返回当前的"历史摘要"


# 演示：HiPPO 状态追踪一个正弦波
state = HiPPOState(N=32, dt=0.01)
t_series = np.linspace(0, 10, 1000)
signal = np.sin(t_series) + 0.1 * np.random.randn(1000)

memory_norms = []
for x_t in signal:
    c = state.update(x_t)
    memory_norms.append(np.linalg.norm(c))

print(f"记忆状态范数范围: [{min(memory_norms):.3f}, {max(memory_norms):.3f}]")
print(f"状态维度 32 压缩了 {1000} 步历史")
```

### 实现中的坑

使用前向 Euler 离散化时步长 `dt` 过大会导致 $A$ 矩阵的特征值稳定性问题。实践中推荐使用双线性变换（bilinear/Tustin 方法）：

```python
# 双线性离散化（更稳定）
import numpy as np
I = np.eye(A.shape[0])
A_disc_stable = np.linalg.solve(I - dt/2 * A, I + dt/2 * A)
B_disc_stable = np.linalg.solve(I - dt/2 * A, dt * B)
```

---

## 核心方法三：自监督序列潜变量

扩散模型的成功秘密之一：前向加噪过程为每一步的潜变量提供了**显式的自监督信号**——模型知道 $z_t$ 应该是什么，所以可以直接用 MSE 监督。

这篇论文把这个思路推广到其他序列生成模型（如分层 VAE）：给每个潜变量层设计自监督目标，而不只是依赖整体的 ELBO。核心思想是**用信息逐步解耦的结构约束潜变量序列**：

```python
class SelfSupervisedLatentLayer(nn.Module):
    """带自监督信号的潜变量层（简化演示）"""
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.encoder = nn.Linear(x_dim, z_dim * 2)   # 均值 + 对数方差
        # 自监督头：从 z 预测某种已知的结构特征（如位置编码、频率）
        self.self_sup_head = nn.Linear(z_dim, x_dim)
    
    def forward(self, x, self_sup_target):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        
        # 重参数化采样
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        
        # 标准 ELBO 的 KL 项
        kl = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(-1)
        
        # 自监督损失：z 应该能预测某个已知目标（e.g., 扩散的噪声量）
        self_sup_pred = self.self_sup_head(z)
        self_sup_loss = F.mse_loss(self_sup_pred, self_sup_target)
        
        return z, kl, self_sup_loss
```

---

## 论文说的 vs 现实

| 方面 | 论文结论 | 实践注意事项 |
|------|---------|------------|
| 贝叶斯 Transformer 不确定性 | 优于 MC Dropout 基线 | 需要重新训练，不能直接插入现有模型 |
| HiPPO GP 在线学习 | 有效记忆长期依赖 | 离散化步长敏感，需要仔细调参 |
| 自监督潜变量 | 改善生成质量 | 自监督目标设计依赖任务，无通用方案 |

复现时的一个现实问题：注意力-GP 对应的精确性依赖于"键值对服从某种高斯过程先验"，而实际训练的 Transformer 对此没有任何保证。文中的方差公式更多是**启发式近似**，而非严格推导。

---

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 安全关键任务（医疗、金融），需要置信区间 | 追求最高吞吐量的推理服务 |
| 在线/流式时间序列预测 | 序列长度超过 10k（GP 开销仍大）|
| 研究：深度模型的可解释性和不确定性 | 仅关心最终精度的工业任务 |
| 数据量少、需要正则化的场景 | 数据充足且过拟合不是问题 |

---

## 我的观点

这篇论文最有价值的地方不是某个具体算法，而是**提供了一个视角转换的框架**：Transformer 不只是一个黑盒，它的注意力机制本身就隐含了概率结构，只是被工程实践掩盖了。

不确定的地方在于 HiPPO GP 部分：HiPPO 的离散化在长序列上确实比 LSTM 稳定，但将其直接对应到 GP 诱导点的数学严格性仍有争议，论文中的"域间诱导点"框架依赖于一些不容易验证的假设。

对于工程师，最直接可用的是方差估计那个公式：$1 - \sum_j w_j^2$。它计算近乎无额外开销，却能给出注意力是否"聚焦"的信号——哪怕你不接受完整的 GP 理论，这个指标本身在 debug 和分析模型行为时就很有用。

深度学习和概率机器学习的融合是一个长期课题，这类工作的贡献更多是搭桥——让两个社区的工具箱能互相借用。这条路还很长。