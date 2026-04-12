---
layout: post-wide
title: "ETCH-X：从着装人体点云到 SMPL-X 的鲁棒拟合"
date: 2026-04-12 08:04:17 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.08548v1
generated_by: Claude Code CLI
---

## 一句话总结

给定穿着衣物的人体点云，ETCH-X 通过"脱衣"和"稠密对应"两阶段流程，稳健地恢复精细的 SMPL-X 人体参数（含手指和面部表情）。

---

## 为什么这个问题重要？

人体拟合（Body Fitting）是数字人制作、游戏动画、AR 试衣、机器人感知的基础步骤。典型生产流程：

```
3D 扫描仪 → 点云 → 人体参数估计 → 骨骼/姿态 → 动画/渲染
```

**现有方法的痛点**：
- 稀疏 landmark 方法：局部遮挡（手部、侧脸）或宽松衣物时失效
- 基于 NeRF/3DGS 的方法：可以渲染，但不给出语义参数（关节角度、体型）
- 单纯优化方法：容易陷入局部最优，对点云噪声敏感

**ETCH-X 的核心创新**：
1. **感知衣物松紧（tightness-aware）**：显式建模衣物厚度，而不是忽略它
2. **从 SMPL 升级到 SMPL-X**：包含 51 个手指关节 + 面部表情参数
3. **稠密对应代替稀疏 landmark**：每个输入点都有对应体表位置，对残缺点云更鲁棒

---

## 背景知识

### SMPL-X 参数模型

SMPL-X 是人体的"可控 3D 模板"，将人体表示为：

$$
M(\beta, \theta, \psi) = W(T(\beta, \theta, \psi),\ J(\beta),\ \theta,\ \mathcal{W})
$$

其中：
- $\beta \in \mathbb{R}^{10}$：体型参数（高矮胖瘦）
- $\theta \in \mathbb{R}^{3K}$：关节旋转（K=55 个关节，含手指）
- $\psi \in \mathbb{R}^{10}$：面部表情
- $W$：LBS（线性蒙皮）函数，将 T-pose 变换到当前姿态

输出是固定拓扑的人体网格：$\mathbf{V} \in \mathbb{R}^{10475 \times 3}$

### 着装人体的建模挑战

着装点云 $\mathcal{P}$ 和裸体表面 $\mathbf{V}$ 之间的关系：

$$
\mathcal{P} \approx \mathbf{V} + \Delta_{\text{cloth}}
$$

$\Delta_{\text{cloth}}$ 是衣物偏移量，受衣物松紧度和当前姿态共同影响：紧身裤偏移近似为零，羽绒服偏移可达 10cm。

### 稠密对应 vs 稀疏 Landmark

| 方法 | 关键点数量 | 对残缺输入的鲁棒性 | 计算量 |
|------|-----------|-------------------|--------|
| 稀疏 landmark | ~20–50 | 差（缺一个就失败） | 低 |
| 稠密对应 | 所有输入点 | 好（可聚合多个估计） | 高 |

---

## 核心方法

### 直觉解释

想象你在扫描一个穿着厚棉袄的人。你看到的是衣物表面，但你想估计身体姿态。ETCH-X 分两步走：

```
着装点云
  │
  ▼
[Undress 网络]  → 预测每点衣物偏移量，"投影"到裸体表面
  │
  ▼
裸体点云估计
  │
  ▼
[Dense Fit 网络] → 建立每点与 SMPL-X 体表的对应关系
  │
  ▼
[梯度优化]       → 最小化对应点距离，求解 β、θ、ψ
  │
  ▼
SMPL-X 参数（含手指/面部）
```

### 数学细节

**Undress 阶段**：

对于输入点云 $\mathcal{P} = \{p_i\}$，预测每点的衣物法向偏移：

$$
\hat{p}_i^{\text{body}} = p_i - \hat{d}_i \cdot \hat{n}_i
$$

其中 $\hat{d}_i$ 是预测的偏移距离，$\hat{n}_i$ 是预测的法向方向。松紧度权重 $w_i \in [0,1]$ 控制区域特异性（手部 $w \approx 1$，宽松外套区域 $w < 1$）。

**Dense Fit 阶段**：

对每个去衣物后的点，预测其在 SMPL-X 规范空间中的对应位置：

$$
c_i = f_\phi\!\left(\hat{p}_i^{\text{body}},\ \mathcal{F}_{\text{global}}\right)
$$

拟合的优化目标：

$$
\mathcal{L}_{\text{fit}} = \sum_i \left\| M(\beta, \theta, \psi)[c_i] - \hat{p}_i^{\text{body}} \right\|_2 + \lambda_{\text{reg}} \left(\|\beta\|^2 + \|\theta\|^2\right)
$$

---

## 实现

### 环境配置

```bash
pip install torch torchvision open3d smplx trimesh
# SMPL-X 模型文件需在官网注册后下载
# https://smpl-x.is.tue.mpg.de/
```

### SMPL-X 基础操作

```python
import torch
import smplx

def create_smplx_model(model_path: str, device='cuda'):
    """创建 SMPL-X 模型实例"""
    return smplx.create(
        model_path,
        model_type='smplx',
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=True,
        num_pca_comps=12,       # 每只手 12 个 PCA 分量
        flat_hand_mean=True,    # PCA=0 对应平摊手掌（避免"爪子手"）
    ).to(device)

def forward_smplx(model, betas, body_pose, hand_pose, expression, transl):
    """前向推理，返回网格顶点"""
    output = model(
        betas=betas,                          # [B, 10]
        body_pose=body_pose,                  # [B, 63]  (21 关节 × 3)
        left_hand_pose=hand_pose[:, :12],     # [B, 12]
        right_hand_pose=hand_pose[:, 12:],    # [B, 12]
        expression=expression,                # [B, 10]
        transl=transl,                        # [B, 3]
        return_verts=True,
    )
    return output.vertices, output.joints     # [B, 10475, 3], [B, 144, 3]
```

### 稠密对应预测网络（简化版）

```python
import torch.nn as nn

class DenseCorrespondenceNet(nn.Module):
    """
    输入:  点云 [B, N, 3]
    输出:  每点对 SMPL-X 顶点的对应概率 log_softmax [B, N, 10475]
    实际论文使用更复杂的特征聚合（KPConv / 注意力机制）
    """
    def __init__(self, num_smplx_verts=10475):
        super().__init__()
        # 点级特征（类 PointNet）
        self.point_enc = nn.Sequential(
            nn.Linear(3, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256),
        )
        # 全局特征
        self.global_enc = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256),
        )
        # 对应预测头
        self.corr_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, num_smplx_verts),
        )

    def forward(self, points):               # [B, N, 3]
        feat   = self.point_enc(points)      # [B, N, 256]
        g_feat = self.global_enc(feat.max(dim=1)[0])   # [B, 256]
        g_exp  = g_feat.unsqueeze(1).expand_as(feat)
        logits = self.corr_head(torch.cat([feat, g_exp], dim=-1))
        return torch.log_softmax(logits, dim=-1)        # [B, N, 10475]
```

### 基于稠密对应的 SMPL-X 拟合优化

```python
import torch.optim as optim

def fit_smplx_from_correspondences(
    point_cloud,     # [N, 3] 去衣物偏移后的点云
    corr_indices,    # [N,]   每点对应的 SMPL-X 顶点索引
    smplx_model,
    num_iters=200,
    device='cuda',
):
    """梯度下降优化 SMPL-X 参数"""
    B = 1
    betas      = torch.zeros(B, 10,  requires_grad=True, device=device)
    body_pose  = torch.zeros(B, 63,  requires_grad=True, device=device)
    hand_pose  = torch.zeros(B, 24,  requires_grad=True, device=device)
    expression = torch.zeros(B, 10,  requires_grad=True, device=device)
    transl     = torch.zeros(B, 3,   requires_grad=True, device=device)

    optimizer = optim.Adam(
        [betas, body_pose, hand_pose, expression, transl], lr=1e-2
    )
    pcd = point_cloud.to(device)

    for i in range(num_iters):
        optimizer.zero_grad()
        verts, _ = forward_smplx(smplx_model, betas, body_pose,
                                  hand_pose, expression, transl)
        target  = verts[0, corr_indices]            # [N, 3]
        loss    = ((target - pcd)**2).mean()
        loss   += 0.01*(betas**2).mean() + 0.001*(body_pose**2).mean()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"Iter {i:3d} | loss={loss.item():.4f}")

    return betas.detach(), body_pose.detach(), hand_pose.detach()
```

### 3D 可视化

```python
import open3d as o3d

def visualize_fitting(input_pcd_np, smplx_verts_np, smplx_faces_np):
    """对比输入点云（蓝）和拟合网格（红）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(input_pcd_np)
    pcd.paint_uniform_color([0.2, 0.4, 0.8])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(smplx_verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(smplx_faces_np)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.3, 0.2])

    o3d.visualization.draw_geometries(
        [pcd, mesh], window_name="ETCH-X Result", mesh_show_wireframe=True
    )
    # 期望输出：蓝色点云包裹红色人体网格，手指区域对齐尤为关键
```

---

## 实验

### 数据集说明

ETCH-X 的一大设计亮点是**模块可独立训练**，不同阶段用不同数据源：

| 数据集 | 类型 | 用于哪个模块 |
|--------|------|-------------|
| CLOTH3D | 仿真着装序列 | Undress（衣物偏移学习） |
| AMASS | 动捕裸体动作 | Dense Fit（对应关系学习） |
| InterHand2.6M | 手部动作 | 手部姿态精细化 |
| 4D-Dress | 动态扫描 | 评估（已见数据） |
| BEDLAM 2.0 | 合成渲染 | 评估（未见数据泛化） |

**关键洞察**：无需采集"着装 + 精确 SMPL-X 标注"的配对数据——这类数据极难获取。两个模块解耦，各自使用已有的大规模数据。

### 定量评估

| 评估集 | 指标 | ETCH（原版） | ETCH-X | 提升 |
|--------|------|-------------|--------|------|
| 4D-Dress | MPJPE-All ↓ | baseline | — | **33.0%** |
| CAPE | V2V-Hands ↓ | baseline | — | **35.8%** |
| BEDLAM 2.0（未见）| MPJPE-All ↓ | baseline | — | **80.8%** |
| BEDLAM 2.0（未见）| V2V-All ↓ | baseline | — | **80.5%** |

BEDLAM 2.0 未见数据上的大幅提升，说明"可组合数据集"设计对泛化能力帮助显著。

---

## 工程实践

### 实际部署考虑

- **实时性**：单帧推理约 100–500ms（视点云密度），不适合实时感知
- **硬件需求**：训练需要 A100 级 GPU；推理 RTX 3090 可接受
- **内存占用**：SMPL-X + 对应网络约 3–5GB 显存

### 常见坑

**坑 1：点云稀疏导致稠密对应退化**

```python
# 不要用 FPS 降采样！应该从原始扫描重采样
def check_density(pcd_np, min_pts=2000):
    if len(pcd_np) < min_pts:
        raise ValueError(f"点数 {len(pcd_np)} 过少，稠密对应会退化")
```

**坑 2：手部 PCA 初始化产生"爪子手"**

```python
# ❌ 错误：默认 flat_hand_mean=False，PCA=0 ≠ 自然手型
model = smplx.create(..., use_pca=True, flat_hand_mean=False)
# ✓ 正确：让 PCA=0 对应平摊手掌，优化更稳定
model = smplx.create(..., use_pca=True, flat_hand_mean=True)
```

**坑 3：宽松衣物区域 Undress 方向歧义**

裙子、羽绒服等宽松区域偏移方向不唯一，需在后处理加关节角度约束：

```python
# 给 body_pose 加关节角度约束防止物理违反
joint_limit_loss = torch.clamp(body_pose.abs() - max_angle, min=0).mean()
total_loss = fit_loss + 0.1 * joint_limit_loss
```

### 数据采集建议

- 确保全身覆盖率 > 80%（手部覆盖率最关键）
- 避免强逆光或镜面材质（产生大量外点）
- 宽松衣物场景建议多角度扫描而非单方向

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 静态或低速扫描 | 高速运动实时估计 |
| 常见服装（T恤、外套、裤子） | 极度宽松服装（婚纱拖尾、汉服） |
| 需要精细手部参数 | 只需粗略躯干姿态 |
| 离线处理（动画资产制作） | 边缘设备实时推理 |
| 点云输入流程 | 仅有 RGB 图像的场景 |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| SMPLify | 无需训练数据 | 初始化敏感，无手部 | 裸体/紧身服 |
| PARE / CLIFF | 快速，图像输入 | 2D→3D 深度歧义 | 视频分析 |
| ETCH（原版） | 点云鲁棒 | 无手部精细化 | 着装躯干 |
| **ETCH-X（本文）** | 全身含手部，耐厚衣 | 推理慢，需点云 | 3D 扫描后处理 |
| 4D Gaussian 人体 | 时序一致性好 | 无语义参数 | 渲染/生成 |

---

## 我的观点

**这个工作解决了一个真实痛点**：工业级 3D 扫描流程里，手部和面部参数历来是体型估计的薄弱环节。ETCH-X 用可组合数据集设计巧妙地规避了"全身精细扫描数据稀缺"的核心困难——把一个难以端到端解决的问题，拆解成两个可以分别用现有数据集训练的子问题。

**离实际应用还有多远？**

- **动画资产制作**：已经相当接近，离线处理完全可行
- **AR 实时试衣**：还有 10–20 倍的速度差距，需要模型蒸馏或专用推理硬件

**值得关注的开放问题**：
1. 极端姿态（瑜伽、杂技）下 SMPL-X 关节角度约束的合理建模
2. 衣物动态与体型估计的联合建模（布料仿真 ↔ 体型参数互相约束）
3. 从单目 RGB 视频直接估计 SMPL-X，摆脱对 3D 扫描仪的依赖

官方代码（待发布）：https://xiaobenli00.github.io/ETCH-X/