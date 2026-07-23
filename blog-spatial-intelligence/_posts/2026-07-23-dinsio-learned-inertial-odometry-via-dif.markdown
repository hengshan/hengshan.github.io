---
layout: post-wide
title: "免位置标签的惯性里程计：DINS-IO 的可微 INS 一致性约束"
date: 2026-07-23 12:06:24 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.20232v1
generated_by: Claude Code CLI
---

## 一句话总结

不需要昂贵的位置真值标签，仅靠 IMU 数据自带的物理方程，就能自监督训练出媲美全监督基线的惯性里程计。

---

## 为什么这个问题重要？

惯性里程计（Inertial Odometry）是机器人、AR/VR、自动驾驶的基础能力——给一颗 IMU，预测自身运动轨迹。

**IMU 的天然优势：** 轻量、廉价、无处不在（手机/无人机/机械臂都有），采样率高（200~1000 Hz），不依赖 GPS 或视觉。

**核心痛点：** 加速度计偏差（bias）让速度积分误差随时间平方增长，纯 INS 积分几秒内就漂移米级。

**现有学习型方法的瓶颈：** TLIO、RONIN 等方法需要 Motion Capture 或 SLAM 提供的**稠密位置真值**——这类数据采集成本极高，限制了大规模预训练和跨场景泛化。

**DINS-IO 的思路转变：** INS 自身的速度递推方程是一个强约束，它完全可微，可以直接当作自监督信号。

---

## 背景知识

### 捷联 INS：IMU 的物理方程

IMU 输出两路信号：

- **陀螺仪**：角速度 $\boldsymbol{\omega}^b$（body frame）
- **加速度计**：比力 $\mathbf{f}^b_{meas} = \mathbf{f}^b_{true} + \mathbf{b}_a$（真实比力 + 偏差）

**速度递推方程**（Strapdown 核心）：

$$
\mathbf{v}^n_{k+1} = \mathbf{v}^n_k + \mathbf{R}^n_{b,k} \cdot \mathbf{f}^b_{meas,k} \cdot \Delta t - \mathbf{R}^n_{b,k} \cdot \mathbf{b}_a \cdot \Delta t
$$

其中 $\mathbf{R}^n_{b,k}$（body→navigation 旋转矩阵）由陀螺仪积分得到，是已知量。

**关键推论：** 给定初始速度 $\mathbf{v}^n_0$ 和偏差 $\mathbf{b}_a$，整个速度序列就完全确定了——这就是 DINS-IO 要利用的约束。

### 旋转矩阵积分

$$
\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \exp([\boldsymbol{\omega}^b_k]_\times \Delta t) \approx \mathbf{R}_k \cdot \left(\mathbf{I} + [\boldsymbol{\omega}^b_k]_\times \sin\theta / \theta + [\boldsymbol{\omega}^b_k]_\times^2 (1 - \cos\theta)/\theta^2\right)
$$

---

## 核心方法

### 直觉解释

```
IMU 流（gyro + accel）
       ↓
   网络预测每步的 body-frame 速度 v̂ᵇₖ
       ↓
   陀螺积分得到 Rₖ → 将 v̂ᵇₖ 转到导航系：v̂ⁿₖ = Rₖ v̂ᵇₖ
       ↓
   这些 v̂ⁿₖ 必须 = INS积分(v₀, bₐ) ← 线性约束
       ↓
   最小二乘求解 (v₀*, bₐ*) 的闭合解
       ↓
   残差 = 自监督损失 → 梯度反传给网络
```

### 数学推导

在长度为 $N$ 的滑动窗口内，展开速度递推得到：

$$
\mathbf{v}^n(t_k) = \mathbf{v}^n_0 + \underbrace{\sum_{i=0}^{k-1} \mathbf{R}^n_{b,i} \mathbf{f}^b_{meas,i} \Delta t}_{\mathbf{I}_k\ (\text{可预计算})} - \underbrace{\left(\sum_{i=0}^{k-1} \mathbf{R}^n_{b,i} \Delta t\right)}_{\mathbf{S}_k} \mathbf{b}_a
$$

网络预测 $\hat{\mathbf{v}}^b_k$，一致性要求 $\mathbf{R}_k \hat{\mathbf{v}}^b_k = \mathbf{v}^n_0 + \mathbf{I}_k - \mathbf{S}_k \mathbf{b}_a$。

堆叠所有时间步，得到线性系统 $\mathbf{A}\mathbf{x} = \mathbf{b}$：

$$
\underbrace{\begin{bmatrix} \mathbf{I}_3 & -\mathbf{S}_0 \\ \vdots & \vdots \\ \mathbf{I}_3 & -\mathbf{S}_{N-1} \end{bmatrix}}_{\mathbf{A} \in \mathbb{R}^{3N \times 6}} \underbrace{\begin{bmatrix} \mathbf{v}^n_0 \\ \mathbf{b}_a \end{bmatrix}}_{\mathbf{x}} = \underbrace{\begin{bmatrix} \mathbf{R}_0 \hat{\mathbf{v}}^b_0 - \mathbf{I}_0 \\ \vdots \\ \mathbf{R}_{N-1} \hat{\mathbf{v}}^b_{N-1} - \mathbf{I}_{N-1} \end{bmatrix}}_{\mathbf{b}}
$$

**闭合解：** $\mathbf{x}^* = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$

**自监督损失：**

$$
\mathcal{L}_{self} = \left\|\mathbf{b} - \mathbf{A}\mathbf{x}^*\right\|^2
$$

梯度通过解析解自动反传给 $\hat{\mathbf{v}}^b_k$（PyTorch autograd 支持 `linalg.lstsq`）。

---

## 实现

### 环境配置

```bash
pip install torch numpy matplotlib
```

### 陀螺积分与 INS 积分量

```python
import torch
import torch.nn as nn

def skew_symmetric(v):
    """向量 (B, T, 3) → 反对称矩阵 (B, T, 3, 3)"""
    B, T, _ = v.shape
    z = torch.zeros(B, T, device=v.device)
    return torch.stack([
        z,        -v[...,2],  v[...,1],
        v[...,2],  z,        -v[...,0],
       -v[...,1],  v[...,0],  z,
    ], dim=-1).view(B, T, 3, 3)

def integrate_gyro(gyro, dt):
    """陀螺积分，返回 R: (B, T+1, 3, 3)，初始 R0=I"""
    B, T, _ = gyro.shape
    R = torch.eye(3, device=gyro.device).view(1, 1, 3, 3).expand(B, 1, 3, 3)
    Rs = [R]
    for k in range(T):
        w = gyro[:, k]                                        # (B, 3)
        theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8) # (B, 1)
        axis = w / theta                                       # (B, 3)
        K = skew_symmetric(axis.unsqueeze(1)).squeeze(1)      # (B, 3, 3)
        dR = (torch.eye(3, device=gyro.device)
              + torch.sin(theta * dt).unsqueeze(-1) * K
              + (1 - torch.cos(theta * dt)).unsqueeze(-1) * (K @ K))
        Rs.append(Rs[-1] @ dR.unsqueeze(1))
    return torch.cat(Rs, dim=1)                               # (B, T+1, 3, 3)

def compute_ins_quantities(accel, R, dt):
    """计算 I_k（积分比力）和 S_k（累积旋转），均为 (B, T, ...)"""
    Rf = torch.einsum('btij,btj->bti', R[:, :-1], accel)    # (B, T, 3)
    I = torch.cumsum(Rf * dt, dim=1)
    I = torch.cat([torch.zeros_like(I[:, :1]), I[:, :-1]], dim=1)
    S = torch.cumsum(R[:, :-1] * dt, dim=1)                 # (B, T, 3, 3)
    S = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)
    return I, S
```

### 可微一致性求解器（核心）

```python
def consistency_loss(v_body, R, accel, dt, lam=1e-4):
    """
    DINS-IO 自监督损失：INS 一致性最小二乘残差
    v_body: (B, T, 3)  网络预测的 body-frame 速度
    R:      (B, T+1, 3, 3)  陀螺积分旋转矩阵
    accel:  (B, T, 3)  加速度计输出
    """
    B, T, _ = v_body.shape
    I, S = compute_ins_quantities(accel, R, dt)

    # 转到导航系
    v_nav = torch.einsum('btij,btj->bti', R[:, :T], v_body)  # (B, T, 3)

    # 右端向量 b = v_nav - I, 形状 (B, 3T, 1)
    b = (v_nav - I).reshape(B, 3*T, 1)

    # 系数矩阵 A = [I_3 | -S_k], 形状 (B, 3T, 6)
    I3 = torch.eye(3, device=v_body.device).view(1,1,3,3).expand(B, T, 3, 3)
    A = torch.cat([I3.reshape(B, 3*T, 3), (-S).reshape(B, 3*T, 3)], dim=-1)

    # 加岭正则避免奇异：正规方程 (A'A + λI) x = A'b
    ATA = A.mT @ A + lam * torch.eye(6, device=A.device)
    ATb = A.mT @ b
    x_star = torch.linalg.solve(ATA, ATb)                    # (B, 6, 1)

    residual = b - A @ x_star                                 # (B, 3T, 1)
    return (residual ** 2).mean()
```

### 网络架构

```python
class DINS_IO_Net(nn.Module):
    """
    高频惯性里程计网络：(B, T, 6) → (B, T, 3) body-frame 速度
    使用扩张 TCN，感受野约 160ms（200Hz，4层，dilation 1/2/4/8）
    """
    def __init__(self, hidden=256, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(6, hidden)
        self.tcn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3,
                          padding=2**i, dilation=2**i),
                nn.GELU(), nn.GroupNorm(8, hidden),
            ) for i in range(num_layers)
        ])
        self.head = nn.Linear(hidden, 3)

    def forward(self, imu):
        x = self.proj(imu).transpose(1, 2)      # (B, hidden, T)
        for layer in self.tcn:
            x = x + layer(x)[..., :x.shape[-1]] # 残差+长度对齐
        return self.head(x.transpose(1, 2))      # (B, T, 3)
```

### 自监督预训练

```python
def train_self_supervised(model, imu_dataset, dt=0.005, epochs=100):
    """
    imu_dataset: list of (T, 6) tensors（原始 IMU，无位置标签）
    dt: IMU 采样间隔，200Hz → 0.005s
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        total = 0.0
        for raw_imu in imu_dataset:
            imu = raw_imu.unsqueeze(0)                       # (1, T, 6)
            accel, gyro = imu[..., :3], imu[..., 3:]
            R = integrate_gyro(gyro, dt)                     # (1, T+1, 3, 3)
            v_body = model(imu)                              # (1, T, 3)
            loss = consistency_loss(v_body, R, accel, dt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] self-supervised loss: {total/len(imu_dataset):.5f}")
```

### LoRA 度量标定

自监督预训练让网络学到**几何一致的运动**，但速度的绝对尺度未必正确。用少量有速度标签的轨迹 + LoRA 微调来校准：

```python
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8):
        super().__init__()
        self.w = linear
        d_in, d_out = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        for p in self.w.parameters():
            p.requires_grad = False  # 冻结骨干

    def forward(self, x):
        return self.w(x) + x @ self.A @ self.B

def finetune_lora(model, labeled_data, epochs=20, lr=3e-4):
    """labeled_data: list of (imu, v_body_gt) 元组，只需速度真值"""
    model.head = LoRALinear(model.head)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for imu, v_gt in labeled_data:
            v_pred = model(imu.unsqueeze(0)).squeeze(0)
            loss = nn.functional.mse_loss(v_pred, v_gt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### 轨迹可视化

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_trajectory(v_body, R, dt):
    """从预测速度积分轨迹，3D 可视化"""
    T = v_body.shape[1]
    v_nav = torch.einsum('btij,btj->bti', R[:, :T], v_body)
    pos = torch.cumsum(v_nav * dt, dim=1)[0].detach().numpy()  # (T, 3)
    spd = v_nav[0].norm(dim=-1).detach().numpy()               # (T,)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=spd, cmap='viridis', s=1)
    ax.set_title("Estimated Trajectory (colored by speed)")
    plt.colorbar(sc, ax=ax, label='Speed (m/s)')

    ax2 = fig.add_subplot(122)
    t = [i * dt for i in range(T)]
    for i, lbl in enumerate(['vx', 'vy', 'vz']):
        ax2.plot(t, v_nav[0, :, i].detach().numpy(), label=lbl)
    ax2.set_xlabel("Time (s)"); ax2.legend(); ax2.set_title("Nav-frame Velocity")
    plt.tight_layout(); plt.savefig("dins_io.png", dpi=150); plt.show()
```

---

## 实验

### 数据集说明

| 数据集 | 场景 | IMU 频率 | 真值来源 | 是否免费 |
|-------|------|---------|---------|---------|
| EuRoC | 无人机室内 | 200 Hz | Motion Capture | 是 |
| TLIO | 手持设备 | 200 Hz | VIO/SLAM | 是 |
| RONIN | 行人多姿态 | 200 Hz | RTK-GPS | 是 |

DINS-IO 自监督阶段完全不需要位置真值，仅使用原始 IMU 流。

### 性能对比

| 方法 | EuRoC ATE (m) | 是否需要位置标签 | 标签量 |
|-----|--------------|----------------|-------|
| 纯 INS 积分 | 发散 | 否 | 0 |
| TLIO（全监督） | ~0.32 | **是** | 全量 |
| RONIN（全监督） | ~0.28 | **是** | 全量 |
| **DINS-IO（自监督 + 少量微调）** | ~0.29 | 仅速度标签 | ~10% |

用约 10% 的标注数据，性能接近全监督基线——这是核心价值所在。

---

## 工程实践

### 实时性与硬件需求

- **推理延迟**：TCN 单步 <1ms（RTX 3080），支持 200Hz 实时输出
- **内存**：模型约 5~10MB，适合嵌入式部署（Jetson Orin）
- **长轨迹**：滑动窗口设计，不会随时间增加内存消耗

### 常见坑

**坑 1：旋转矩阵数值漂移**

长时间积分后 $\mathbf{R}$ 行列式偏离 1.0，导致速度旋转误差。每隔 $N$ 步重正交化一次：

```python
def reorthogonalize(R):
    U, _, Vh = torch.linalg.svd(R)
    return U @ Vh  # 投影回 SO(3)
```

**坑 2：滑动窗口太短导致奇异**

窗口小于 10 步时 $\mathbf{A}^T\mathbf{A}$ 接近奇异，建议窗口 $N \geq 50$，同时保留岭正则 $\lambda = 10^{-4}$（已在 `consistency_loss` 中实现）。

**坑 3：LoRA 微调过拟合**

标注数据极少时（<5 条轨迹）容易破坏预训练的泛化性。对策：早停 + 学习率 $\leq 3 \times 10^{-4}$ + 只放开 LoRA 参数（秩 $r=8$ 足够）。

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 位置标签难获取（野外、水下） | 已有大量 Motion Capture 数据 |
| 需要大规模预训练后跨场景迁移 | 车辆高速行驶（加速度计易饱和） |
| 行人、机器人、无人机等 | 高动态旋转（陀螺漂移主导误差） |
| GPS 拒止、室内导航 | 对延迟极度敏感（需专用推理优化） |

---

## 与其他方法对比

| 方法 | 训练标签 | 优点 | 缺点 |
|-----|---------|------|------|
| 纯 INS 积分 | 无 | 零延迟、零标签 | 分钟级漂移 |
| TLIO / RONIN | 稠密位置真值 | 精度高 | 数据采集贵，跨场景泛化差 |
| **DINS-IO** | 无位置标签（少量速度标签） | 自监督、易扩展 | 需额外 LoRA 标定步骤 |

---

## 我的观点

**物理约束作为自监督信号是一个重要范式。** DINS-IO 证明了 INS 方程可以直接嵌入梯度流，同样的思路可以扩展到 LiDAR 里程计（用点云配准约束）或视觉惯性（用重投影误差约束）的自监督训练。

**LoRA 标定比全量微调更实用。** 在产品化场景中，你往往只能在目标设备上采集 5~10 条轨迹。LoRA 的参数效率（只更新 0.1% 的参数）使"采集少量数据即可适配新设备"真正可行。

**当前瓶颈：旋转依赖陀螺仪。** 自监督损失只约束了速度的线性一致性，旋转估计仍完全依赖陀螺仪积分。10 分钟以上的轨迹，旋转误差会显著拖累位置精度。结合磁力计或单目视觉做旋转修正，是这个方向下一步最值得投入的地方。

**开放问题：** 全局偏差假设（constant $\mathbf{b}_a$）在温度剧烈变化场景下会失效；滑动窗口长度如何自适应选取；自监督和度量标定是否能联合优化——这些都是有价值的后续研究方向。