---
layout: post-wide
title: "QGPINNs：用物理驱动神经网络求解量子图上的偏微分方程"
date: 2026-08-31 08:05:34 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.28589v1
generated_by: Claude Code CLI
---

正在撰写关于 QGPINNs 的深度技术博客，覆盖量子图 PDE、多网络联合训练和关键 trick。


## 一句话总结

QGPINNs 将 PINN 的思路搬到**量子图**（Quantum Graph）上——每条边分配一个神经网络，通过包含节点耦合条件的图感知损失函数联合优化，可以求解图结构上的分数阶偏微分方程，包括正问题和参数识别逆问题。

## 背景：量子图是什么，为什么它难？

### 量子图 = 图结构 + 1D PDE

"量子图"听起来高深，实际上很直白：把一张图（节点 + 边）上的每条边看作一段 1D 区间，在每条边上求解微分方程，同时在节点处满足耦合条件。

一个直觉类比：城市供水管网。每段管道是一条边，交叉口是节点。水流在每段管道里遵从压力方程，在交叉口满足守恒律（流入 = 流出）。量子图就是这类网络物理问题的数学框架。

节点耦合条件有两类：
- **连续性条件**：相邻边在节点处函数值相等
- **Kirchhoff-Neumann 条件**：节点处各边法向导数之和为零（类比电路中的 KCL）

### 传统方法的局限

有限元法（FEM）处理图结构尚可，但遇到**非局部算子**（如分数阶拉普拉斯 $(-\Delta)^\alpha$，$\alpha \in (0,1)$）时，刚度矩阵变成稠密矩阵，计算代价爆炸。分数阶方程的解还可能在节点处出现 $x^\beta$ 型**弱奇异性**，对 FEM 网格要求极高。

PINN 理论上对计算域形状无限制，但直接套用到图结构需要解决三个问题：多网络联合训练的梯度均衡、节点条件的软/硬约束施加、分数阶积分的数值稳定计算。QGPINNs 是针对这三个问题的系统性解决方案。

## 算法原理

### 核心思想

设图有边集 $\mathcal{E}$、节点集 $\mathcal{V}$。对每条边 $e$ 分配网络 $\hat{u}_e(x; \theta_e)$，整体参数 $\Theta = \{\theta_e\}_{e \in \mathcal{E}}$。

训练目标：

$$\mathcal{L}(\Theta) = \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{vertex}} \mathcal{L}_{\text{vertex}}$$

### 节点耦合损失（关键）

对每个内部节点 $v$，设与它相连的边集为 $\mathcal{E}_v$，$n_{e,v} \in \{+1, -1\}$ 为边 $e$ 在节点 $v$ 处的外法向符号：

$$\mathcal{L}_{\text{vertex}} = \sum_{v \in \mathcal{V}_{\text{int}}} \left[ \underbrace{\sum_{\substack{e,e' \in \mathcal{E}_v \\ e \neq e'}} \!\!\!\!(\hat{u}_e(v) - \hat{u}_{e'}(v))^2}_{\text{连续性}} + \underbrace{\left(\sum_{e \in \mathcal{E}_v} n_{e,v}\, \hat{u}'_e(v)\right)^2}_{\text{Kirchhoff-Neumann}} \right]$$

这一项是 QGPINNs 区别于普通 PINN 的核心，通过自动微分计算各边在节点处的导数，然后强制 KCL 条件。

### 分数阶算子的处理

对分数阶椭圆问题 $(-\Delta)^\alpha u_e = f_e$，分数阶拉普拉斯在有界域上近似为：

$$(-\Delta)^\alpha u(x) \approx c_\alpha \text{ P.V.} \int_0^{L_e} \frac{u(x) - u(y)}{|x-y|^{1+2\alpha}} dy$$

实现时用 Gauss-Jacobi 求积离散积分节点，可以天然处理积分端点的奇异性，每个积分值通过网络前向传播获得，整体仍可自动微分。

## 实现

### 最小可运行版本：星形图上的 Poisson 方程

先从整数阶问题入手，验证框架正确性：**3 条边的星形图**，求解 $-u'' = 1$，外端 $u(0) = 0$，内节点满足连续性和 Kirchhoff 条件。解析解为 $u_e(x) = \frac{1}{2}x(1-x) + \frac{1}{6}x$（对称图上三边完全相同）。

```python
import torch
import torch.nn as nn

class EdgeNet(nn.Module):
    def __init__(self, hidden=32, n_layers=3):
        super().__init__()
        seq = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
    
    def forward(self, x):
        return self.net(x)

def grad(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]

n_edges = 3
nets = [EdgeNet() for _ in range(n_edges)]
optimizer = torch.optim.Adam(
    [p for net in nets for p in net.parameters()], lr=1e-3
)

for step in range(8000):
    optimizer.zero_grad()
    x_col = torch.rand(100, 1, requires_grad=True)   # 内部配点
    x_bc  = torch.zeros(1, 1)                         # 外端 Dirichlet
    x_v   = torch.ones(1, 1, requires_grad=True)      # 内节点

    loss_pde = loss_bc = 0.
    u_v_list, du_v_list = [], []

    for net in nets:
        u   = net(x_col)
        du  = grad(u, x_col)
        d2u = grad(du, x_col)
        loss_pde += ((d2u + 1.0) ** 2).mean()   # -u'' = 1
        loss_bc  += (net(x_bc) ** 2).mean()      # u(0) = 0

        uv  = net(x_v)
        duv = grad(uv, x_v)
        u_v_list.append(uv)
        du_v_list.append(duv)

    # 连续性条件
    loss_cont = sum((u_v_list[i] - u_v_list[0]) ** 2 for i in range(1, n_edges))
    # Kirchhoff-Neumann（内端方向 n=+1）
    loss_kn = (sum(du_v_list) ** 2).mean()

    loss = loss_pde + 10 * loss_bc + 100 * loss_cont + 100 * loss_kn
    loss.backward()
    optimizer.step()

    if step % 2000 == 0:
        print(f"step {step:5d}  loss={loss.item():.4e}  "
              f"KN={loss_kn.item():.4e}  cont={loss_cont.item():.4e}")
```

运行后 `KN` 和 `cont` 应在 5000 步内降至 1e-5 量级。

### 完整实现：带动态权重的 QGPINN

下面在前面定义的 `EdgeNet` 和 `grad` 基础上，加入动态权重和梯度裁剪：

```python
class QGPINN:
    def __init__(self, n_edges, hidden=64, n_layers=4, lr=1e-3):
        self.nets = [EdgeNet(hidden, n_layers) for _ in range(n_edges)]
        # 可学习的对数权重（SoftAdapt 风格）
        self.log_w = nn.Parameter(torch.zeros(3))
        all_params = (
            [p for net in self.nets for p in net.parameters()]
            + [self.log_w]
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def train_step(self, x_col_list, x_bc_list, vertex_x):
        self.optimizer.zero_grad()
        w = torch.exp(self.log_w)   # [w_pde, w_bc, w_vert]

        l_pde = l_bc = 0.
        u_v, du_v = [], []

        for net, xc, xb in zip(self.nets, x_col_list, x_bc_list):
            xc = xc.requires_grad_(True)
            u = net(xc)
            du_ = grad(u, xc)
            l_pde += ((grad(du_, xc) + 1.0) ** 2).mean()
            l_bc  += (net(xb) ** 2).mean()

            xv = vertex_x.requires_grad_(True)
            uv_ = net(xv)
            u_v.append(uv_)
            du_v.append(grad(uv_, xv))

        l_vert = (
            sum((u_v[i] - u_v[0]) ** 2 for i in range(1, len(u_v)))
            + (sum(du_v) ** 2).mean()
        )
        loss = w[0] * l_pde + w[1] * l_bc + w[2] * l_vert

        loss.backward()
        all_params = [p for net in self.nets for p in net.parameters()]
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self.optimizer.step()
        return loss.item(), l_pde.item(), l_vert.item()
```

动态权重 `log_w` 参与梯度下降，当某类损失过大时其权重自动增加，减少手动调 $\lambda$ 的工作量。

## 关键 Trick

论文提到了几个在实践中缺一不可的技巧，但描述不总是充分：

**1. Fourier Feature Embedding**

普通 MLP 有 spectral bias，对高频解收敛慢。把输入先做 Fourier 映射：

```python
class FourierEdgeNet(nn.Module):
    def __init__(self, n_freq=16, sigma=5.0, hidden=64, n_layers=3):
        super().__init__()
        B = torch.randn(1, n_freq) * sigma
        self.register_buffer('B', B)
        seq = [nn.Linear(2 * n_freq, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)
    
    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat)
```

**2. 奇异性捕获特征**

分数阶方程的解在节点处可能有 $d^\beta$ 型弱奇异性（$d$ 为到节点的距离）。在网络输入中加入显式奇异基，让神经网络专注拟合剩余光滑部分：

```python
# 对距离内节点为 d = 1 - x 的区域
def augment_singular(x, beta=0.6):
    d = 1.0 - x          # 到内节点的距离
    return torch.cat([x, d.abs() ** beta], dim=-1)
```

**3. 配点策略**

节点附近的梯度信息最重要，均匀采样会浪费计算：

```python
# 内点均匀 + 节点附近加密
x_uniform = torch.rand(80, 1)
x_near_vertex = torch.rand(20, 1) * 0.1 + 0.9   # 靠近 x=1
x_col = torch.cat([x_uniform, x_near_vertex])
```

## 调试指南

### 常见问题

1. **Kirchhoff 损失降不下去**：通常是 $\lambda_{\text{vertex}}$ 太小。从 100 开始，每次翻倍试。如果调到 10000 还不行，检查各边的导数符号是否一致（内/外端方向定义混乱是常见 bug）。

2. **不同边的网络精度差异大**：各边长度相差大时，输入归一化是必须的。把每条边的坐标统一映射到 $[0, 1]$，在损失中乘以相应的 Jacobi 因子。

3. **分数阶积分数值爆炸**：Riemann-Liouville 积分靠近端点有奇异性，不要用均匀积分节点。改用 Gauss-Jacobi 或 Gauss-Legendre 求积，并在端点附近加权。

4. **总损失下降但物理解不对**：分开打印各项损失。如果 `l_bc` 在 1e-6 而 `l_vert` 在 1e-1，说明权重分配严重失衡，需要重新调 $\lambda$。

### 量化指标

| 检查项 | 正常状态 | 问题信号 |
|--------|----------|----------|
| 连续性误差 | < 1e-4 | > 1e-2，$\lambda$ 太小 |
| Kirchhoff 误差 | < 1e-4 | 持续不降，检查符号定义 |
| 各边 PDE 残差 | 均匀下降 | 某边卡住，检查初始化 |
| 动态权重 $e^{w_i}$ | 稳定在合理范围 | 某项指数爆炸，加学习率预热 |

### 超参数调优

| 参数 | 推荐范围 | 敏感度 | 建议 |
|------|---------|--------|------|
| `lr` | 1e-4 ~ 1e-3 | 高 | 先试 3e-4 |
| $\lambda_{\text{vertex}}$ | 50 ~ 1000 | 极高 | 100 起步，倍增调 |
| 隐藏层宽度 | 32 ~ 128 | 中 | 64 通常够用 |
| 配点数/边 | 100 ~ 500 | 中 | 先用 100 验证再增加 |
| Fourier 频率数 | 8 ~ 32 | 中 | 16 起步 |

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 图结构上的分数阶 PDE | 大规模图（> 500 条边），训练代价爆炸 |
| 几何复杂，网格生成困难 | 需要高精度（> 4 位有效数字），FEM 更可靠 |
| 逆问题：从含噪数据识别 $\alpha$ | 时间敏感的实时应用 |
| 物理参数不确定需要数据融合 | 整数阶方程，成熟 FEM/谱方法已够用 |
| 电力网、排水网等真实拓扑 | 超参数调试资源有限的项目 |

## 我的观点

QGPINNs 确实填补了"PINN + 图结构 + 分数阶算子"这个交叉领域的空白，但有几点需要清醒认识：

**逆问题是真正的亮点**。从带噪声的观测数据中识别分数阶次 $\alpha$ 或物理参数，是传统数值方法很难做到的。这类场景下 QGPINNs 有实际价值。

**精度问题存疑**。分数阶问题的参考解本身就难获得，论文中的误差比较可信度有限。对于整数阶椭圆方程，FEM 在精度和收敛保证上远比 PINN 成熟。

**扩展性天花板明显**。每条边一个网络，边数线性增加参数量。工业级电力网络（数千条边）基本不可行，除非改用参数共享或图神经网络加速。

总结：如果你处理的是**中小规模图（< 100 条边）上的分数阶逆问题**，值得投入精力去跑 QGPINNs。如果只是在图上解普通 Poisson 方程，FEM + Kirchhoff 边界条件的现有实现成熟得多，别自找麻烦。