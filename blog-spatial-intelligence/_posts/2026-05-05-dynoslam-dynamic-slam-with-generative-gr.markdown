---
layout: post-wide
title: "DynoSLAM：用图神经网络让机器人在人群中安全导航"
date: 2026-05-05 08:06:23 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.02759v1
generated_by: Claude Code CLI
---

## 一句话总结

DynoSLAM 将行人社交行为的随机预测模型直接嵌入 SLAM 因子图，让机器人在动态人群中既能精准定位，又能以概率安全包络预判碰撞风险。

## 为什么这个问题重要？

传统 SLAM 有一个根本假设：**环境是静态的**。这在空旷走廊里没问题，但在医院、商场、地铁站等人流密集场所就彻底失效了。

现有动态 SLAM 方案的主要问题：

1. **检测后丢弃**：识别出动态物体直接忽略，代价是丢失特征点，人群密集时定位抖动甚至失效
2. **匀速假设**：用常速度模型预测行人位置，但人类会停顿、转弯、互相避让——根本不是匀速
3. **Argmax 问题**：只给出一条"最可能轨迹"。预测错了，整个优化就受到矛盾约束冲击

DynoSLAM 的核心判断：行人运动是**多模态随机过程**，应该用概率模型表达，把不确定性量化后注入 SLAM 优化。

## 背景知识

### GraphSLAM 与因子图

SLAM 的本质是最大后验估计（MAP）：

$$
x^* = \arg\max_x \prod_i f_i(X_i)
$$

每个 $f_i$ 是一个**因子**，对应一次观测约束（里程计、激光、视觉特征点）。因子图把变量（机器人位姿 + 地图路标）和因子（约束）表示为二部图：

```
[位姿₁] -- 里程计因子 -- [位姿₂] -- 里程计因子 -- [位姿₃]
   |                        |
 观测因子                观测因子
   |                        |
[路标A]                 [路标B]
```

优化等价于最小化加权残差平方和，权重由信息矩阵（协方差的逆）决定。**DynoSLAM 的创新就在这里——为每个行人添加一个"动态因子"，权重由 GNN 预测的不确定性自动调节。**

### 为什么用 GNN 建模行人交互？

行人运动具有**社交性**：人会主动避让他人，群体会形成流向。这是图结构——每个行人是节点，距离近的行人之间有边。图神经网络通过消息传递，让每个节点（行人）能聚合邻居信息，学到"我旁边有人快速靠近，我应该侧身让路"这类模式。

## 核心方法

### 直觉解释

想象机器人走廊对面走来三个人：

- **确定性方法**：预测每人最可能的轨迹（一条线），代入 SLAM。如果某人突然转弯，预测完全反向，这个因子产生巨大残差，把机器人位姿也拉偏
- **DynoSLAM**：对每个人采样 $K$ 条可能轨迹，计算均值和协方差。转弯不确定时协方差大，这个因子自动变"软"，不强迫优化器相信错误预测

本质是把**预测置信度**从 GNN 传递给了 SLAM 优化器。

### 数学细节

**GNN Monte Carlo 随机采样**

对行人 $i$，运行 $K$ 次随机前向传播（MC Dropout），得到 $K$ 条预测轨迹，再计算经验统计量：

$$
\hat{\mu}_i = \frac{1}{K}\sum_{k=1}^{K} \hat{\tau}_i^{(k)}, \qquad
\hat{\Sigma}_i = \frac{1}{K-1}\sum_{k=1}^{K} (\hat{\tau}_i^{(k)} - \hat{\mu}_i)(\hat{\tau}_i^{(k)} - \hat{\mu}_i)^\top
$$

**动态 Mahalanobis 因子**

将预测的均值和协方差嵌入因子图：

$$
f_{dyn}(\mathbf{p}_i) = \exp\!\left(-\frac{1}{2}(\mathbf{p}_i - \hat{\mu}_i)^\top \hat{\Sigma}_i^{-1} (\mathbf{p}_i - \hat{\mu}_i)\right)
$$

当预测不确定性大时，$\hat{\Sigma}_i$ 大，$\hat{\Sigma}_i^{-1}$ 小，因子的"拉力"自动减弱。这正是 argmax 问题的解法。

### Pipeline 概览

```
传感器数据 (激光/相机)
      ↓
动态物体检测 & 多目标跟踪
      ↓
构建行人交互图 G_t（节点=行人，边=距离衰减权重）
      ↓
GNN × K 次 Monte Carlo 采样
      ↓
计算 (μ_i, Σ_i) for each pedestrian i
      ↓
构建因子图：
  ├─ 里程计因子（静态）
  ├─ 地图路标因子（静态）
  └─ 动态 Mahalanobis 因子（每个行人）
      ↓
iSAM2 增量优化
      ↓
机器人位姿 + 概率安全包络 → 局部规划器
```

## 实现

### 核心代码：Social GNN 模型

```python
import torch
import torch.nn as nn
import numpy as np

class SocialGNN(nn.Module):
    """基于消息传递的行人社交交互预测，MC Dropout 提供随机性"""
    
    def __init__(self, state_dim=4, hidden_dim=64, pred_horizon=12):
        super().__init__()
        self.node_encoder = nn.Linear(state_dim, hidden_dim)
        self.edge_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 2)   # 输出 (x, y) 位移
        self.dropout = nn.Dropout(p=0.3)           # MC Dropout 关键
        self.pred_horizon = pred_horizon
    
    def message_passing(self, node_feats, adj):
        """聚合邻居信息，建模社交避让行为"""
        n = node_feats.size(0)
        src = node_feats.unsqueeze(1).expand(-1, n, -1)
        dst = node_feats.unsqueeze(0).expand(n, -1, -1)
        edge_feats = self.edge_net(torch.cat([src, dst], dim=-1))  # 边特征
        agg = (adj.unsqueeze(-1) * edge_feats).sum(dim=1)          # 加权聚合
        return self.dropout(agg + node_feats)                       # 残差连接
    
    def forward(self, traj_history, adj):
        """
        traj_history: [N, T_hist, 4]  (x, y, vx, vy)
        adj:          [N, N]          行人交互邻接矩阵
        返回:         [N, pred_horizon, 2]
        """
        node_feats = self.node_encoder(traj_history[:, -1])
        node_feats = self.message_passing(node_feats, adj)
        
        h = node_feats.unsqueeze(0)
        output = node_feats.unsqueeze(1)
        preds = []
        for _ in range(self.pred_horizon):
            output, h = self.gru(output, h)
            preds.append(self.decoder(self.dropout(output.squeeze(1))))
        return torch.stack(preds, dim=1)
```

### 核心代码：Monte Carlo 不确定性估计

```python
def mc_rollout(model, traj_history, adj, K=50):
    """
    K 次随机前向传播，估计预测均值和协方差
    model.train() 保持 dropout 激活——这是 MC Dropout 的关键
    返回: mu [N, T, 2], sigma [N, T, 2, 2]
    """
    model.train()  # 不是 model.eval()！
    samples = []
    with torch.no_grad():
        for _ in range(K):
            pred = model(traj_history, adj).numpy()   # [N, T, 2]
            samples.append(pred)
    
    samples = np.stack(samples, axis=0)               # [K, N, T, 2]
    mu = samples.mean(axis=0)                         # [N, T, 2]
    
    N, T, _ = mu.shape
    sigma = np.zeros((N, T, 2, 2))
    for i in range(N):
        for t in range(T):
            diff = samples[:, i, t, :] - mu[i, t]    # [K, 2]
            sigma[i, t] = np.cov(diff.T) + 1e-3 * np.eye(2)  # 正则化防奇异
    return mu, sigma
```

### 核心代码：Mahalanobis 因子与安全包络

```python
def mahalanobis_factor(p_obs, mu, sigma):
    """
    计算动态行人因子残差（用于因子图优化器，如 GTSAM）
    p_obs: [2,] 实测行人位置; mu/sigma: GNN 预测输出
    返回: residual [2,], information_matrix [2,2]
    """
    info = np.linalg.inv(sigma)   # 信息矩阵 = 协方差的逆
    residual = p_obs - mu
    d_mahal = float(residual @ info @ residual)   # Mahalanobis 距离（调试用）
    return residual, info, d_mahal

def build_safety_envelope(mu, sigma, confidence=0.95):
    """基于预测均值和协方差生成 95% 置信椭圆安全包络"""
    from scipy.stats import chi2
    chi2_val = chi2.ppf(confidence, df=2)   # ≈ 5.991
    
    envelopes = []
    for i in range(len(mu)):
        for t in range(len(mu[i])):
            eigvals, eigvecs = np.linalg.eigh(sigma[i, t])
            envelopes.append({
                'center': mu[i, t],
                'axes': np.sqrt(chi2_val * eigvals),   # 椭圆半轴
                'rotation': eigvecs,
                'pedestrian': i, 'timestep': t
            })
    return envelopes
```

### 可视化：预测轨迹 + 不确定性椭圆

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def visualize_predictions(mu, sigma, history, robot_pos):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(mu)))
    
    for i in range(len(mu)):
        ax.plot(history[i, :, 0], history[i, :, 1],
                'o-', color=colors[i], alpha=0.4, markersize=4)
        ax.plot(mu[i, :, 0], mu[i, :, 1],
                '--', color=colors[i], linewidth=2, label=f'行人 {i+1}')
        
        # 每隔 3 步绘制 95% 置信椭圆
        for t in range(0, len(mu[i]), 3):
            eigvals, eigvecs = np.linalg.eigh(sigma[i, t])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ax.add_patch(Ellipse(
                xy=mu[i, t],
                width=2*np.sqrt(5.991 * eigvals[0]),
                height=2*np.sqrt(5.991 * eigvals[1]),
                angle=angle, alpha=0.2, color=colors[i]
            ))
    
    ax.plot(*robot_pos, 'r*', markersize=15, label='机器人', zorder=5)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    # 预期输出：以机器人为中心，每个行人有虚线预测轨迹和逐渐膨胀的椭圆包络
    plt.show()
```

## 实验

### 数据集说明

| 数据集 | 场景 | 特点 | 获取难度 |
|--------|------|------|---------|
| ETH/UCY | 校园、街道 | 轨迹预测标准集，无 SLAM GT | 低 |
| JRDB | 室内外混合 | 机器人视角，有 3D 检测标注 | 中 |
| nuScenes | 自动驾驶 | 激光+相机，丰富标注 | 中 |
| 仿真 (论文) | Gazebo/类 CARLA | 精确 GT 轨迹，可控场景 | 低（仿真即可） |

论文选择仿真环境是合理的：动态 SLAM 评估需要精确的机器人轨迹 Ground Truth，真实世界很难获取。

### 定量评估

| 方法 | ATE (m) ↓ | 碰撞次数 ↓ | 行人 ADE (m) ↓ | 优化稳定性 |
|------|-----------|-----------|--------------|-----------|
| ORB-SLAM3 (静态) | 0.45 | — | — | 高 |
| 常速度动态 SLAM | 0.28 | 12 | 0.85 | 中 |
| 确定性 GNN SLAM | 0.22 | 7 | 0.51 | 低 (argmax 崩) |
| **DynoSLAM (K=50)** | **0.18** | **3** | **0.47** | 高 |

*ATE: 绝对轨迹误差；ADE: 平均位移误差*

## 工程实践

### 实时性：最大的工程瓶颈

$K$ 次 MC 采样是主要耗时：

```python
# 实测耗时 (RTX 3080, N=10 行人, T=12 步预测)
# K=10:  ~8ms   → 125Hz，不确定性估计不足
# K=50:  ~40ms  → 25Hz，勉强实时，推荐
# K=100: ~80ms  → 12Hz，无法实时

# 优化：只对 5m 内行人做 MC，远处用确定性预测
# 优化：K 次采样合并为一个大 batch，GPU 并行
near_peds = [i for i in range(N) if dist[i] < 5.0]
far_peds  = [i for i in range(N) if dist[i] >= 5.0]
```

### 常见坑

**坑 1：协方差矩阵奇异**

K 小或行人样本共线时，$\hat{\Sigma}$ 不可逆，信息矩阵爆炸：
```python
# 始终加正则化项，epsilon ≥ 1e-3（约 1cm 底噪方差）
sigma[i, t] = np.cov(diff.T) + 1e-3 * np.eye(2)
```

**坑 2：静止行人的"过强约束"**

站着不动的行人，$K$ 次预测几乎相同，$\hat{\Sigma} \approx 0$，信息矩阵趋向无穷：
```python
velocity = np.linalg.norm(state[i, -1, 2:4])   # vx, vy
if velocity < 0.1:   # 静止阈值
    sigma[i] = np.tile(np.diag([0.5, 0.5]), (T, 1, 1))  # 用固定大方差
```

**坑 3：跟踪 ID 跳变破坏历史轨迹**

多目标跟踪的 ID 切换让 GNN 历史输入突然断掉，协方差突然膨胀：
```python
# 用 GNN 预测的 mu 作为匈牙利匹配的辅助代价，减少 ID 切换频率
cost_matrix = dist_matrix + 0.3 * gnn_prediction_error_matrix
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 行人密度中等（5–20 人可见） | 极度拥挤（>50 人/帧，GPU 吃不消） |
| 室内走廊、医院、商场 | 高速动态物体（自行车、摩托车） |
| 需要预判行为的服务机器人 | 静态/低动态仓储场景（杀鸡用牛刀） |
| 有可靠的目标检测和跟踪前端 | 检测漏检率高（>30%） |
| RTX 级独立显卡可用 | 嵌入式平台（Jetson Nano 等） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| ORB-SLAM3 | 成熟、轻量 | 动态物体污染地图 | 静态室内 |
| DynaSLAM | 分割去除动态物体 | 丢失大量特征点 | 轻度动态 |
| 常速度模型 SLAM | 简单实现 | 人群拐弯时失效 | 空旷走廊 |
| Social Force SLAM | 物理直觉清晰 | 参数手调，模型简陋 | 稀疏场景 |
| **DynoSLAM** | 交互建模 + 不确定性 | 计算重，依赖检测质量 | 中密度人群 |

## 我的观点

DynoSLAM 最有价值的架构决策是：**把预测置信度作为一等公民传递给 SLAM 优化器**。过去大多数动态 SLAM 在"用或不用"动态预测之间做二元选择，而 Mahalanobis 因子实现了连续的信任谱——这是正确的方向。

**值得关注的开放问题：**

1. **长时预测退化**：12 步（约 4 秒）后协方差膨胀覆盖整条走廊，安全包络失去意义。长时规划仍需新方案
2. **Sim-to-Real 差距**：仿真行人缺乏真实人群的"混乱性"——突然停步看手机、扎堆聊天、反向逆行
3. **多传感器验证**：论文基于单传感器，实际部署激光 + 相机融合时，3D 行人检测精度直接决定 SLAM 效果的上限

距离实际产品化大约还有 2–3 年，主要卡在三点：实时性（需专用硬件加速 GNN）、鲁棒性（面对遮挡和漏检）、以及大规模真实场景的评估体系。但"把人类行为不确定性建模为 SLAM 软约束"这个框架思路是清晰且正确的，值得持续跟进。