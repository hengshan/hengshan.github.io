---
layout: post-wide
title: "GAVIS：用球谐函数量化 3DGS 可见性，实现实时不确定性建图"
date: 2026-05-30 12:06:58 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.30342v1
generated_by: Claude Code CLI
---

## 一句话总结

GAVIS 为每个 3D 高斯粒子附加一个球谐函数编码的"各向异性可见性场"，让渲染器知道哪些像素是在训练盲区里猜出来的，并以此驱动主动建图——以 200 FPS 的速度实时量化不确定性。

---

## 为什么这个问题重要

### 3DGS 的阿喀琉斯之踵

3D Gaussian Splatting（3DGS）渲染极快，但有一个根本性的弱点：**对于训练时未覆盖的视角，渲染结果不可靠，而模型对此毫无自知之明**。

这在以下场景是致命的：

- **SLAM 系统**：需要知道哪里探索不够，决定下一步去哪
- **机器人导航**：盲区 = 危险区域
- **AR 场景扫描**：用户走到新角度时突然出现浮影

传统方案要么用 NeRF 系的 Bayesian 方法（慢），要么干脆忽略不确定性（糙）。GAVIS 的答案是：**3DGS 已经用球谐函数（SH）编码颜色，用同样的结构顺带记录可见性，几乎零额外成本**。

---

## 背景知识

### 3DGS 渲染回顾

场景用 $N$ 个 3D 高斯粒子表示，每个粒子有位置 $\mu_k$、协方差 $\Sigma_k$、不透明度 $o_k$ 和颜色 SH 系数。像素颜色通过 alpha 合成：

$$
C = \sum_{i=1}^{N} c_i \, \alpha_i \prod_{j < i} (1 - \alpha_j)
$$

其中 $\alpha_i = o_i \cdot \exp\!\left(-\frac{1}{2} \delta_i^\top \Sigma_i^{-1} \delta_i\right)$，$c_i(\mathbf{d})$ 是视角相关颜色（SH 编码）。

**关键问题**：如果训练视角从没从方向 $\mathbf{d}$ 看过粒子 $i$，对应的 SH 系数是外推预测，**不可信**。

### 球谐函数：方向信息的紧凑编码

SH 函数 $Y_l^m(\mathbf{d})$ 是单位球面上的正交基：

$$
f(\mathbf{d}) \approx \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_l^m(\mathbf{d})
$$

3DGS 用 $L=3$（16 系数）编码颜色。GAVIS 用 $L=2$（9 系数）编码可见性，两者结构完全一致，可以并行实现。

---

## 核心方法

### 直觉解释

想象每个高斯粒子上插着一个"方向探测器"，训练时记录从哪些方向被摄像机观测到、贡献了多少。

新视角渲染时，对每个参与像素合成的粒子，查询"这个方向见过吗？"：
- 见过多 → 低不确定性
- 没见过 → 高不确定性

整个过程在 rasterization 时顺带完成，不需要额外的网络推理。

### 各向异性可见性场

对于第 $k$ 个高斯粒子，其可见性场为：

$$
V_k(\mathbf{d}) = \sigma\!\left( \sum_{l,m} v_k^{lm} \cdot Y_l^m(\mathbf{d}) \right)
$$

$v_k^{lm}$ 是可见性 SH 系数，从训练过程中按粒子贡献权重累积更新。

### 不确定性传播

渲染新视角时，像素不确定性与颜色合成并行计算：

$$
U(\mathbf{r}) = \sum_{i} \bigl(1 - V_i(\mathbf{d}_{\mathbf{r}})\bigr) \cdot w_i, \quad w_i = \alpha_i \prod_{j<i}(1-\alpha_j)
$$

### 主动建图目标

有了逐像素不确定性图，Next-Best-View 选择退化为：

$$
\xi^* = \arg\max_{\xi \in \mathcal{C}} \sum_{p} U_p(\xi)
$$

从候选位姿集合 $\mathcal{C}$ 中选能最大化信息增益的下一个观测位置。

### Pipeline 概览

```
训练视角 → 3DGS 标准训练 ─────────────────────→ 高斯粒子参数
                          ↘ 同步累积 SH 可见性系数
                                    ↓
新视角查询 → 标准 Rasterizer ──→ 颜色图 (C)
             Bayesian 通道  ──→ 不确定性图 (U)
                                    ↓
候选位姿 → IG 评分 → 选最优位姿 → 执行观测 → 更新可见性
```

---

## 实现

### SH 基函数与可见性场

```python
import torch
import torch.nn as nn

def eval_sh_basis(dirs: torch.Tensor) -> torch.Tensor:
    """
    计算 l=0,1,2 阶 SH 基函数（9 个系数）
    dirs: (N, 3) 单位方向向量 → (N, 9)
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    basis = torch.zeros(*dirs.shape[:-1], 9, device=dirs.device)
    basis[..., 0] = 0.2821                             # l=0
    basis[..., 1] = 0.4886 * y                         # l=1
    basis[..., 2] = 0.4886 * z
    basis[..., 3] = 0.4886 * x
    basis[..., 4] = 1.0925 * x * y                    # l=2
    basis[..., 5] = 1.0925 * y * z
    basis[..., 6] = 0.3154 * (2*z**2 - x**2 - y**2)
    basis[..., 7] = 1.0925 * x * z
    basis[..., 8] = 0.5463 * (x**2 - y**2)
    return basis


class GaussianVisibilityField(nn.Module):
    """每个高斯粒子的各向异性可见性场（SH 编码）"""

    def __init__(self, n_gaussians: int, sh_degree: int = 2):
        super().__init__()
        n_coeffs = (sh_degree + 1) ** 2
        self.vis_coeffs = nn.Parameter(
            torch.zeros(n_gaussians, n_coeffs), requires_grad=False
        )
        self.obs_count = torch.zeros(n_gaussians)

    def update(self, gids: torch.Tensor,
               view_dirs: torch.Tensor, weights: torch.Tensor):
        """
        gids: (M,) 参与渲染的粒子索引
        view_dirs: (M, 3) 射线方向（归一化）
        weights: (M,) alpha 合成权重
        """
        sh_vals = eval_sh_basis(view_dirs)              # (M, 9)
        self.vis_coeffs.data.index_add_(0, gids, weights.unsqueeze(-1) * sh_vals)
        self.obs_count.index_add_(0, gids, torch.ones(len(gids)))

    def query(self, gids: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """返回指定粒子在指定方向的可见性 ∈ [0, 1]"""
        sh_vals = eval_sh_basis(dirs)                   # (M, 9)
        raw = (self.vis_coeffs[gids] * sh_vals).sum(-1) # (M,)
        return torch.sigmoid(raw)
```

### 不确定性感知 Rasterizer

```python
def uncertainty_rasterize(
    pixel_gaussian_ids: list,    # 每个像素关联的高斯粒子列表（按深度排序）
    opacities: torch.Tensor,     # (N,) 各粒子不透明度
    colors: torch.Tensor,        # (N, 3) 各粒子颜色
    ray_dirs: torch.Tensor,      # (H*W, 3) 射线方向
    vis_field: GaussianVisibilityField,
) -> tuple:
    """
    并行计算颜色图和不确定性图
    返回: color_map (H*W, 3), uncertainty_map (H*W,)
    """
    n_pixels = ray_dirs.shape[0]
    color_out = torch.zeros(n_pixels, 3)
    unc_out = torch.zeros(n_pixels)

    for px in range(n_pixels):
        gids = torch.tensor(pixel_gaussian_ids[px])
        T = 1.0  # 透射率

        for i, gid in enumerate(gids):
            alpha = opacities[gid].item()  # 简化：实际需投影 2D 高斯
            w = alpha * T

            # 颜色合成
            color_out[px] += w * colors[gid]

            # 不确定性：查询该粒子在当前射线方向的可见性
            vis = vis_field.query(gid.unsqueeze(0), ray_dirs[px].unsqueeze(0))
            unc_out[px] += w * (1.0 - vis.item())

            T *= (1.0 - alpha)
            if T < 1e-4:
                break

    return color_out, unc_out
```

### 主动建图：Next-Best-View 选择

```python
import numpy as np

def select_next_view(
    render_fn,            # callable(pose) -> (color, uncertainty_map)
    candidate_poses: list,
    use_top_k: int = 5,   # 先低分辨率粗筛，再精评
) -> int:
    """
    两阶段 NBV 选择：粗筛 + 精评，平衡速度与质量
    返回信息增益最大的候选位姿索引
    """
    # 第一阶段：低分辨率快速评估所有候选
    gains = []
    for pose in candidate_poses:
        _, unc = render_fn(pose, resolution_scale=0.25)  # 1/4 分辨率
        gains.append(unc.mean().item())

    # 第二阶段：全分辨率精评 Top-K
    top_k_idx = np.argsort(gains)[-use_top_k:]
    best_idx, best_gain = top_k_idx[0], -np.inf

    for idx in top_k_idx:
        _, unc = render_fn(candidate_poses[idx], resolution_scale=1.0)
        gain = unc.mean().item()
        if gain > best_gain:
            best_gain, best_idx = gain, idx

    return int(best_idx)
```

### 可视化：球面可见性场

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_visibility_sphere(vis_coeffs_np: np.ndarray):
    """
    将单个粒子的可见性场可视化为彩色极坐标球面
    预期输出：训练视角方向凸出，未覆盖方向收缩
    """
    phi, theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:100j]
    dirs = np.stack([
        np.sin(phi)*np.cos(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(phi)
    ], axis=-1).reshape(-1, 3)

    dirs_t = torch.tensor(dirs, dtype=torch.float32)
    sh_vals = eval_sh_basis(dirs_t).numpy()
    vis = 1 / (1 + np.exp(-(sh_vals @ vis_coeffs_np)))  # sigmoid
    vis = vis.reshape(50, 100)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        np.sin(phi)*np.cos(theta) * vis,
        np.sin(phi)*np.sin(theta) * vis,
        np.cos(phi) * vis,
        facecolors=plt.cm.plasma(vis), alpha=0.9
    )
    ax.set_title("各向异性可见性场（半径=可见性大小）")
    plt.tight_layout(); plt.show()
```

---

## 实验

### 数据集说明

| 数据集 | 场景类型 | 获取难度 | 用途 |
|--------|---------|---------|------|
| Replica | 室内合成 | 简单（官方下载） | 精度 benchmark |
| ScanNet | 室内真实 | 中等（需注册） | 泛化性测试 |
| Matterport3D | 大场景室内 | 中等 | 可扩展性验证 |

### 定量评估

| 方法 | Coverage ↑ | PSNR ↑ | SSIM ↑ | 速度 (FPS) |
|------|-----------|--------|--------|-----------|
| 随机探索 | 72.3 | 24.1 | 0.81 | — |
| Frontier Exploration | 78.6 | 25.3 | 0.83 | — |
| NeRF-based NBV | 85.2 | 27.8 | 0.87 | 1.2 |
| **GAVIS（本文）** | **91.4** | **29.6** | **0.91** | **200** |

速度差异（200 vs 1.2 FPS）是量级差距，不是调参能弥补的，这是方法本质决定的。

---

## 工程实践

### Post-hoc 集成（最重要的工程特性）

GAVIS 可以对已有 3DGS 模型做**后处理**补充可见性信息，无需重新训练：

```python
def build_visibility_posthoc(
    gaussians: dict,           # 已训练的 3DGS 模型参数
    training_frames: list,     # [(camera_pose, alpha_weights, gids, dirs), ...]
    vis_field: GaussianVisibilityField,
):
    """回放训练帧，仅更新可见性 SH 系数，不修改高斯参数"""
    with torch.no_grad():
        for pose, weights, gids, dirs in training_frames:
            vis_field.update(gids, dirs, weights)
    print(f"可见性场构建完成，覆盖 {vis_field.obs_count.gt(0).sum()} 个粒子")
```

### 硬件需求

- **实时推理（200 FPS）**：RTX 3090 及以上，内存额外占用约 +10%（每百万粒子多 36 MB SH 系数）
- **大场景处理**：按空间分块加载，仅对当前视锥内粒子激活 SH 查询

### 常见坑

1. **SH 系数未归一化 → 可见性饱和为 0 或 1**  
   `update()` 中改为 EMA：`coeffs = 0.9 * coeffs + 0.1 * new`，避免累积值爆炸

2. **候选位姿太密 → NBV 选择延迟超过传感器帧率**  
   两阶段评估（见上方 `select_next_view`）：1/4 分辨率粗筛，再对 Top-5 精评

3. **动态物体（人/车）污染可见性场**  
   结合光流或语义分割过滤动态粒子，GAVIS 本身假设静态场景

4. **训练视角覆盖不均匀 → 可见性场偏差大**  
   `obs_count` 低于阈值的粒子标记为"高先验不确定性"，不信任其 SH 系数

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| SLAM + 主动探索 | 动态场景（人、车、植被） |
| 机器人室内 3D 建图 | 光照剧烈变化（SH 颜色本就不稳定） |
| AR 扫描引导（哪里没扫到） | 纯离线渲染任务（不需要不确定性） |
| 对已有 3DGS 补充不确定性 | 需要物理准确光照（改用 NeRF-W 体系） |

---

## 与其他方法对比

| 方法 | 不确定性来源 | 速度 | 主动建图 | 备注 |
|------|------------|------|---------|------|
| BayesRays | 权重扰动 | ~5 FPS | 间接 | NeRF 系，训练慢 |
| ActiveNeRF | 信息熵 | 2–10 FPS | 原生 | 精度高，实时难 |
| CF-3DGS | Depth 不一致性 | 60 FPS | 有限 | 不区分方向不确定性 |
| **GAVIS** | SH 可见性场 | **200 FPS** | 原生 | Post-hoc 可集成 |

---

## 我的观点

GAVIS 的聪明之处在于**找到了一个几乎免费的表示空间**：3DGS 已经用 SH 编码颜色，加一套平行的 SH 编码可见性，数学上优雅，工程上廉价。200 FPS 的实时不确定性量化确实打开了 3DGS 在机器人领域的大门。

但有几点需要保持清醒：

**静态场景假设没有突破**。真实机器人环境里动态物体会污染可见性场，需要配合动态点过滤。这部分工作留给下游系统，GAVIS 本身不解决。

**Post-hoc 是亮点，也是限制**。它意味着方法本身无法在训练中利用不确定性来主动改善重建质量，这与 ActiveNeRF 类端到端优化在理念上是不同的——GAVIS 是"先建好，再补信息"，而非"边建边引导"。

**候选位姿的质量上限了 NBV 的质量**。文中用均匀采样或路径规划提供候选，真实机器人还需要运动学约束和碰撞检测介入，这是从 demo 到产品必须解决的工程问题。

总体而言，GAVIS 是目前将不确定性引入 3DGS 最务实的工作之一，后处理集成能力降低了使用门槛。对于需要主动建图的机器人和 AR 系统，值得认真评估。原论文链接：https://arxiv.org/abs/2605.30342v1