---
layout: post-wide
title: 'UQ-Loc：为 LiDAR 定位加上"置信度"——不确定性感知的场景坐标回归'
date: 2026-08-07 12:02:49 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.06307v1
generated_by: Claude Code CLI
---

## 一句话总结

UQ-Loc 在 LiDAR 场景坐标回归（SCR）的每个体素预测结果上附加一个各向异性高斯协方差矩阵，让定位系统不仅知道"我在哪里"，还知道"我对这个答案有多大把握"。

---

## 为什么这个问题重要？

自动驾驶、机器人导航和工业巡检都依赖 LiDAR 定位来获取精确的 6-DoF 位姿（3 个平移 + 3 个旋转自由度）。现有的确定性方法只输出一个坐标预测值，没有任何质量信号——在遮挡、动态障碍物或点云退化的情况下，系统无从判断当前预测是否可信。

**现有 SCR 方法的三个痛点：**

1. **盲目信任**：所有体素的预测被同等对待，无法过滤低质量的对应关系
2. **RANSAC 低效**：传统 RANSAC 随机采样种子点，大量时间浪费在离群点上
3. **无法进行下游融合**：卡尔曼滤波等状态估计方法依赖协方差输入，确定性输出无法直接使用

UQ-Loc 的核心创新是：**在 SCR 管线中引入逐体素的各向异性不确定性估计**，并将不确定性系统性地融入 RANSAC 求解器，形成从感知到决策的完整不确定性传播链。

---

## 背景知识

### LiDAR 场景坐标回归（SCR）

SCR 方法将点云直接映射到预先建好的地图坐标系中，跳过了传统的描述子匹配和地图检索步骤。

```
原始 LiDAR 点云 → 体素化 → 3D 稀疏 CNN → 场景坐标预测 → PnP 求解 → 6-DoF 位姿
```

核心假设：训练时见过的场景，网络能"记住"每个位置对应的全局坐标。UQ-Loc 的基础架构是 **LightLoc**，它使用 MinkowskiEngine 风格的稀疏卷积在体素特征上进行场景坐标预测。

### 协方差矩阵的几何含义

一个 3D 各向异性高斯分布用 3×3 正定协方差矩阵 Σ 描述不确定性的**方向和幅度**：

- 球形协方差（各向同性）：各方向不确定性相同，适合简单场景
- 椭球协方差（各向异性）：沿某些方向（如 LiDAR 深度方向）不确定性更大

为保证 Σ 正定，使用 Cholesky 分解 $\Sigma = L L^\top$，网络只需预测下三角矩阵 L 的 6 个参数（3×3 下三角有 6 个独立元素）。

---

## 核心方法

### 直觉解释

想象每个体素对应的场景坐标预测不是一个点，而是一个 3D 椭球。椭球越小，预测越确定；椭球越扁，说明在某个方向上信息不足（如 LiDAR 在水平方向精度远高于垂直方向）。

后续的 RANSAC 求解器优先选择"椭球最小"的体素作为种子点，并用 Mahalanobis 距离而非欧氏距离判断内点——这正是 UQ-Loc 能提升定位精度的关键。

### 各向异性协方差头

```python
import torch
import torch.nn as nn

class AnisoGaussianHead(nn.Module):
    """在 LightLoc 特征之上添加的协方差预测头"""
    
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.coord_head = nn.Linear(in_channels, 3)          # 预测场景坐标均值
        self.cov_head   = nn.Linear(in_channels, 6)          # 预测 Cholesky 因子的 6 个参数
    
    def _build_cholesky(self, params: torch.Tensor) -> torch.Tensor:
        """将 6 个参数组装成下三角矩阵 L，保证对角线正数"""
        N = params.shape[0]
        L = torch.zeros(N, 3, 3, device=params.device)
        # 对角线用 exp 保证正数，非对角线自由
        L[:, 0, 0] = torch.exp(params[:, 0])
        L[:, 1, 0] = params[:, 1]
        L[:, 1, 1] = torch.exp(params[:, 2])
        L[:, 2, 0] = params[:, 3]
        L[:, 2, 1] = params[:, 4]
        L[:, 2, 2] = torch.exp(params[:, 5])
        return L  # Σ = L @ L^T 是正定矩阵
    
    def forward(self, features: torch.Tensor):
        coords = self.coord_head(features)                    # (N, 3)
        L      = self._build_cholesky(self.cov_head(features))  # (N, 3, 3)
        return coords, L
```

### 训练损失

**NLL 损失**：对多变量高斯分布取负对数似然

$$
\mathcal{L}_\text{NLL} = \frac{1}{N} \sum_i \left[ \mathbf{d}_i^\top \Sigma_i^{-1} \mathbf{d}_i + \log |\Sigma_i| \right]
$$

其中 $\mathbf{d}_i = \hat{\mathbf{c}}_i - \mathbf{c}_i^*$ 是预测坐标与真值的偏差。利用 Cholesky 因子计算更稳定：

```python
def nll_loss(pred_coords: torch.Tensor,
             gt_coords:   torch.Tensor,
             L:           torch.Tensor) -> torch.Tensor:
    """
    多变量高斯 NLL，通过 Cholesky 因子 L 计算（数值稳定）。
    Σ^{-1} d 等价于求解 L x = d 后的 ||x||^2
    """
    d = (pred_coords - gt_coords).unsqueeze(-1)               # (N, 3, 1)
    # 前向代入求解 L·x = d，避免显式求逆
    Linv_d = torch.linalg.solve_triangular(L, d, upper=False) # (N, 3, 1)
    mahal   = (Linv_d ** 2).sum(dim=(-2, -1))                 # (N,)  马氏距离平方
    # log|Σ| = 2·Σ log(L_ii)
    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return (mahal + log_det).mean()
```

**kNN 空间平滑正则**：相邻体素的不确定性应该相似，避免噪声导致协方差剧烈抖动

$$
\mathcal{L}_\text{smooth} = \frac{1}{NK} \sum_i \sum_{j \in \mathcal{N}_k(i)} \|\Sigma_i - \Sigma_j\|_F^2
$$

```python
def knn_smoothness_loss(L: torch.Tensor, voxel_coords: torch.Tensor, k: int = 8) -> torch.Tensor:
    """鼓励 k 近邻体素拥有相近的协方差矩阵"""
    # 计算体素间欧氏距离，找 k 近邻
    dist = torch.cdist(voxel_coords.float(), voxel_coords.float())
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    idx = idx[:, 1:]                                          # 排除自身

    cov   = L @ L.transpose(-1, -2)                          # (N, 3, 3)
    neigh = cov[idx.reshape(-1)].reshape(-1, k, 3, 3)        # (N, k, 3, 3)
    diff  = cov.unsqueeze(1) - neigh                         # (N, k, 3, 3)
    return (diff ** 2).sum(dim=(-2, -1)).mean()
```

总损失：$\mathcal{L} = \mathcal{L}_\text{NLL} + \lambda \mathcal{L}_\text{smooth}$，$\lambda = 0.1$

### 不确定性加权 SC2-PCR 求解器

SC2-PCR 是一个基于图匹配的精确求解器。UQ-Loc 对其做了两处修改：

**1. 不确定性加权种子评分**

传统方法随机选种子点，UQ-Loc 优先选择协方差行列式最小的对应关系（置信度最高）：

```python
def uncertainty_seed_score(L: torch.Tensor) -> torch.Tensor:
    """
    用协方差体积（行列式）给候选对应关系打分，越小越优先
    log|Σ| = 2·Σ log(L_ii)，避免数值溢出
    """
    log_vol = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (N,)
    score   = -log_vol                                           # 负号：体积越小分数越高
    return score
```

**2. Mahalanobis 距离内点检测**

```python
def mahalanobis_inlier_test(residual:    torch.Tensor,
                             L:          torch.Tensor,
                             chi2_thresh: float = 7.815) -> torch.Tensor:
    """
    内点判断：d^T Σ^{-1} d < χ²(df=3, p=0.95) = 7.815
    替代传统的固定欧氏距离阈值（如 0.5m）
    """
    r = residual.unsqueeze(-1)                                   # (N, 3, 1)
    Linv_r  = torch.linalg.solve_triangular(L, r, upper=False)
    maha_sq = (Linv_r ** 2).sum(dim=(-2, -1))                   # (N,)
    return maha_sq < chi2_thresh
```

---

## 实现

### 完整训练循环（核心骨架）

```python
import torch
import torch.optim as optim

def train_epoch(model, cov_head, dataloader, optimizer, lambda_smooth=0.1):
    model.train(); cov_head.train()
    total_loss = 0.0

    for batch in dataloader:
        voxel_feats  = batch["features"].cuda()   # 稀疏体素特征
        voxel_coords = batch["coords"].cuda()      # 体素 3D 坐标
        gt_scene_coord= batch["scene_coord"].cuda()# 真值场景坐标

        # 前向传播
        base_feats         = model(voxel_feats)
        pred_coords, L     = cov_head(base_feats)

        # 计算损失
        loss_nll    = nll_loss(pred_coords, gt_scene_coord, L)
        loss_smooth = knn_smoothness_loss(L, voxel_coords, k=8)
        loss        = loss_nll + lambda_smooth * loss_smooth

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### 不确定性椭球可视化

```python
import numpy as np
import open3d as o3d

def draw_uncertainty_ellipsoids(scene_coords: np.ndarray,
                                covariances:  np.ndarray,
                                scale: float = 3.0):
    """在 Open3D 中渲染每个预测点的不确定性椭球"""
    geometries = []
    for mu, cov in zip(scene_coords, covariances):
        vals, vecs = np.linalg.eigh(cov)           # 特征分解
        radii = scale * np.sqrt(np.abs(vals))       # 椭球半轴 = 3σ

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        # 将单位球变形为椭球
        T = np.eye(4)
        T[:3, :3] = vecs @ np.diag(radii)
        T[:3,  3] = mu
        sphere.transform(T)

        # 不确定性越大颜色越红
        uncertainty = np.trace(cov)
        sphere.paint_uniform_color([min(uncertainty / 5.0, 1.0), 0.2, 0.2])
        geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries)
    # 预期输出：彩色椭球点云，红色=高不确定区域（遮挡/边界），
    #           蓝色=低不确定区域（开阔平坦的建筑立面）

# ... （点云加载与场景坐标解码代码省略）
```

### 不确定性校准评估（ECE）

Expected Calibration Error 衡量"预测的置信区间是否覆盖真实误差的对应比例"：

```python
def compute_ece(pred_std: np.ndarray, actual_error: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = Σ_b (|B_b|/N) · |emp_coverage(b) - nominal_coverage(b)|
    pred_std: 预测的标准差（协方差矩阵迹的平方根）
    actual_error: 实际预测误差（欧氏距离）
    """
    z = actual_error / (pred_std + 1e-8)                  # 标准化误差
    thresholds = np.linspace(0.1, 3.0, n_bins)            # 对应 68%→99.7% 置信区间
    ece = 0.0
    for t in thresholds:
        nominal_cov  = 2 * (scipy.stats.norm.cdf(t) - 0.5) # 标称覆盖率
        empirical_cov= (z < t).mean()                       # 实际覆盖率
        ece += abs(empirical_cov - nominal_cov)
    return ece / n_bins
```

---

## 实验

### 数据集与硬件

| 数据集 | 特点 | 场景规模 |
|---|---|---|
| Oxford RobotCar | 城市场景，多时段重复 | 10km 路段 |
| MulRan | 韩国城市，季节变化 | 大规模户外 |
| NCLT | 校园，长时间跨度 | 室内外混合 |

LiDAR SCR 训练需要 RTX 3090/A100 级别 GPU，推理可在 RTX 2080 Ti 上实时运行（>10 FPS）。

### 定量评估

| 方法 | 平移误差 (m) | 旋转误差 (°) | 召回率@1m | ECE |
|---|---|---|---|---|
| LightLoc（基线）| 0.48 | 1.32 | 78.3% | — |
| LightLoc + 各向同性不确定性 | 0.43 | 1.21 | 80.1% | 0.124 |
| **UQ-Loc** | **0.39** | **1.15** | **83.7%** | **0.067** |

ECE 越低代表不确定性越校准（0 为完美）。

---

## 工程实践

### 硬件与实时性

- 训练：至少 24GB VRAM，批大小 4-8 个扫描帧
- 推理：协方差头仅增加约 5% 延迟，整体流程 < 100ms/frame（RTX 3080）
- 大场景：体素化分辨率从 0.1m 调大到 0.3m 可降低约 60% 内存占用

### 常见坑

**坑 1：协方差矩阵数值爆炸**

对角线参数初始化为 0 时，`exp(0)=1`，初始椭球很大，梯度不稳定。

```python
# 修复：将对角线偏置初始化为负数，让初始不确定性接近真实噪声水平
nn.init.constant_(cov_head.bias[:3], -2.0)  # exp(-2) ≈ 0.13m 初始标准差
```

**坑 2：kNN 在推理时开销过大**

训练时用 `torch.cdist` 计算全局 kNN，N 个体素时复杂度为 $O(N^2)$。

```python
# 修复：推理时跳过平滑损失，只在训练时计算 kNN
if self.training:
    loss_smooth = knn_smoothness_loss(L, voxel_coords)
```

**坑 3：Mahalanobis 阈值选错**

直接用欧氏距离阈值（如 0.5m）会因各方向尺度不同导致内点率崩溃。

```python
# 修复：用卡方分布临界值（df=3，置信度 95%）
chi2_thresh = 7.815   # scipy.stats.chi2.ppf(0.95, df=3)
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---|---|
| 需要位姿置信度用于后续决策（自动驾驶 planning） | 点云极稀疏（< 5000 点/帧） |
| 与卡尔曼滤波/粒子滤波融合 | 纯新场景（SCR 依赖预训练地图） |
| 遮挡多、动态障碍物频繁的城市环境 | 对延迟极其敏感（< 20ms） |
| 需要检测定位失败（协方差大 = 预警信号） | 训练数据极少（< 1km 路段） |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 传统描述子匹配 | 无需训练，可迁移 | 无不确定性，慢 | 离线建图 |
| NeRF-based 定位 | 渲染质量高 | 极慢，无 LiDAR 优化 | RGB 室内场景 |
| LightLoc（确定性 SCR） | 快速，精度好 | 无置信度输出 | 已知场景在线定位 |
| **UQ-Loc** | 有校准不确定性，精度更高 | 依赖预先建图 | 安全关键的自动驾驶定位 |

---

## 我的观点

UQ-Loc 代表了 LiDAR 定位领域的一个重要趋势：**将感知不确定性系统性地传播到决策链**。这不是噱头，在自动驾驶安全领域，知道"我不确定"和知道"我在哪里"同样重要。

**值得关注的开放问题：**

1. **时序不确定性**：当前方法逐帧独立预测，缺乏跨帧的不确定性传播（与 IMU 积分的融合）
2. **地图退化检测**：协方差突然增大是否能可靠预测地图过期（如施工后场景变化）？
3. **训练数据需求**：NLL 损失需要精确的 GT 场景坐标，标注成本仍然较高

离实际产品化还有 1-2 年距离，主要障碍是**大规模地图更新**和**跨域泛化**。但作为融合感知-决策的不确定性框架，UQ-Loc 的设计思路值得借鉴。

---

*论文链接：https://arxiv.org/abs/2608.06307v1*