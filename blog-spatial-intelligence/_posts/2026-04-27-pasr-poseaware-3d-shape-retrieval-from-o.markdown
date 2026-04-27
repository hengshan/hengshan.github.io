---
layout: post-wide
title: "从遮挡单视图检索 3D 形状：PASR 的分析-合成框架"
date: 2026-04-27 08:05:51 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.22658v1
generated_by: Claude Code CLI
---

## 一句话总结

给定一张可能被部分遮挡的 2D 图像，PASR 通过"分析-合成"循环在 3D 形状数据库中找到最匹配的形状——同时附赠相机姿态估计和物体类别识别。

## 为什么这个问题重要？

想象这些场景：机器人在仓库中看到被箱子遮挡的椅子，需要识别是数据库中哪个型号才能正确抓取；电商平台从用户手机照片中找到对应的 3D 商品模型；AR 应用需要识别桌面上摆放的零件。这就是**单视图 3D 形状检索**要解决的问题。

现有方法的两类局限：
- **对比学习类**（如 OpenShape、Uni3D）：将点云与图像特征对齐到同一空间，全局嵌入对遮挡极其脆弱
- **公共嵌入空间类**：端到端黑盒对齐，可解释性差，泛化到真实世界时经常崩溃

PASR 把检索问题重新定义为：**"哪个 3D 形状在什么姿态下投影出来，和查询图像的特征图最匹配？"** 这种分析-合成思路让方法天然具备几何可解释性和遮挡鲁棒性。

## 背景知识

### 3D 形状表示对比

| 表示方式 | 优点 | 缺点 | 在检索中的用途 |
|---------|------|------|------------|
| 点云 | 灵活、采样简单 | 无拓扑信息 | 特征提取 |
| 网格（Mesh） | 精确表面，可渲染 | 拓扑复杂 | 数据库存储 |
| 体素 | 规则，易处理 | 内存大 | 早期检索方法 |
| 隐式函数（NeRF） | 连续表示 | 推理慢 | 重建，不适合大库检索 |

PASR 数据库存的是**网格或点云**，检索时通过投影得到 2D 视图再提取特征。

### DINOv2：检索的视觉基础

DINOv2（Meta AI，论文中写作 DINOv3，应为其迭代版本或笔误）是基于自监督学习的视觉基础模型，关键特性：

- **Patch 级特征**：将图像分成 14×14 像素的 patch，每个 patch 独立编码，保留空间位置信息
- **语义感知**：无需标注，训练出强大的几何和纹理理解能力
- **跨视角对应**：同一物体在不同视角下的对应 patch 特征高度相似

这正是 PASR 能够把 2D 特征图和 3D 投影对齐的基础——**patch 之间的对应关系隐含了几何结构**。

### 分析-合成（Analysis-by-Synthesis）

计算机视觉中的经典思想，与其直接预测输出，不如通过合成来验证假设：

```
假设: 查询物体是形状 s，相机姿态是 (R, t)
合成: 将 s 从姿态 (R, t) 投影到 2D，提取特征
分析: 特征和查询图像匹配吗？
优化: 调整 s 和 (R, t)，直到特征差距最小
```

## 核心方法

### 直觉解释

传统方法是一次性的前向推理：

```
查询图像 → [黑盒神经网络] → 全局特征向量 → 最近邻搜索
```

PASR 是一个优化过程：

```
查询图像 ──→ DINOv2 ──→ Patch 特征图 F_query (H×W×D)
                               ↑ 最小化 patch 级特征距离
候选 3D 形状 s + 姿态 (R,t) → 投影渲染 → DINOv2 → 特征图 F_synth
```

遮挡区域的 patch 因为没有对应的 3D 结构，自然被排除在匹配之外——无需任何遮挡建模，天然鲁棒。

### 数学细节

设查询图像为 $I$，DINOv2 提取的 patch 特征图为 $\mathbf{F}(I) \in \mathbb{R}^{H \times W \times D}$。

形状 $s_i$ 在姿态 $(R, t)$ 下的合成特征图：
$$
\hat{\mathbf{F}}(s_i, R, t) = \text{DINOv2}\big(\text{Render}(s_i, R, t)\big)
$$

PASR 测试时优化目标：
$$
(s^*, R^*, t^*) = \arg\min_{s_i,\, R,\, t} \; \mathcal{L}_{\text{feat}}\big(\mathbf{F}(I),\; \hat{\mathbf{F}}(s_i, R, t)\big)
$$

其中特征损失仅在可见 patch 集合 $\mathcal{V}$ 上计算余弦距离：
$$
\mathcal{L}_{\text{feat}} = 1 - \frac{1}{|\mathcal{V}|} \sum_{p \in \mathcal{V}} \frac{\mathbf{F}_p \cdot \hat{\mathbf{F}}_p}{\|\mathbf{F}_p\| \cdot \|\hat{\mathbf{F}}_p\|}
$$

遮挡区域的 patch 不在 $\mathcal{V}$ 中，不参与梯度计算。

### Pipeline 概览

```
训练阶段:
CAD 模型 + 随机姿态 → 渲染 2D 图 → DINOv2 特征对齐 → 训练 3D 编码器

推理阶段:
查询图像 → DINOv2 特征图
           ↓
     粗检索 (3D 编码器全局特征)
           ↓
     Top-K 候选形状
           ↓
     测试时优化 (analysis-by-synthesis)
           ↓
   输出: (最优形状 s*, 姿态 R*,t*, 类别标签)
```

## 实现

### 核心代码：DINOv2 Patch 特征提取

```python
import torch
import torch.nn.functional as F
from PIL import Image

class DINOv2FeatureExtractor:
    """提取 DINOv2 的 patch 级特征图，用于分析-合成匹配"""
    
    def __init__(self, model_name='facebook/dinov2-base'):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.patch_size = 14  # DINOv2 固定 patch size

    @torch.no_grad()
    def extract(self, image: Image.Image, image_size: int = 224) -> torch.Tensor:
        """
        返回 patch 级特征图
        输出形状: (H/14, W/14, D), 例如 (16, 16, 768)
        """
        inputs = self.processor(images=image, return_tensors="pt", size=image_size)
        outputs = self.model(**inputs)
        
        # last_hidden_state: (1, 1+num_patches, D)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # 去掉 [CLS]
        h = w = image_size // self.patch_size
        return patch_tokens.reshape(h, w, -1)  # (H, W, D)

    def patch_similarity(self, feat_a: torch.Tensor, feat_b: torch.Tensor,
                         mask: torch.Tensor = None) -> torch.Tensor:
        """计算两个特征图的 patch 级平均余弦相似度"""
        sim = F.cosine_similarity(feat_a, feat_b, dim=-1)  # (H, W)
        if mask is not None:
            return sim[mask].mean()  # 只统计可见区域
        return sim.mean()
```

### 核心代码：姿态条件点云投影

```python
import numpy as np

def project_pointcloud(points: np.ndarray, R: np.ndarray, t: np.ndarray,
                       K: np.ndarray, image_size: int = 224):
    """
    将点云按给定相机姿态投影到 2D 图像平面
    返回: depth_map (H, W) 和可见性 mask
    """
    # 变换到相机坐标系
    pts_cam = (R @ points.T).T + t          # (N, 3)
    valid = pts_cam[:, 2] > 0              # 过滤相机背后的点

    # 透视投影
    uv_h = (K @ pts_cam[valid].T).T       # (M, 3)
    u = (uv_h[:, 0] / uv_h[:, 2]).astype(int)
    v = (uv_h[:, 1] / uv_h[:, 2]).astype(int)
    depth = uv_h[:, 2]

    # Z-buffer：从远到近绘制，近处覆盖远处
    depth_map = np.zeros((image_size, image_size), dtype=np.float32)
    in_bounds = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)
    for i in np.argsort(-depth):
        if in_bounds[i]:
            depth_map[v[i], u[i]] = depth[i]

    return depth_map, depth_map > 0  # depth_map, visibility_mask

def make_camera_K(fov_deg: float = 60.0, image_size: int = 224) -> np.ndarray:
    """构造简单针孔相机内参"""
    f = image_size / (2 * np.tan(np.radians(fov_deg / 2)))
    c = image_size / 2
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float32)
```

### 核心代码：分析-合成检索主循环

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RetrievalResult:
    shape_idx: int
    R: np.ndarray
    t: np.ndarray
    score: float

def pasr_retrieval(
    query_feat: torch.Tensor,          # (H, W, D) 查询图像的特征图
    precomputed_feats: dict,           # {(shape_idx, pose_idx): Tensor} 预计算缓存
    candidate_poses: List[Tuple],      # [(R, t), ...] 候选姿态列表
    top_k_shapes: List[int],           # 粗筛后的 top-k 形状索引
    extractor: DINOv2FeatureExtractor,
) -> RetrievalResult:
    """
    PASR 核心：在 (形状, 姿态) 空间中搜索最优匹配
    论文使用可微渲染 + 梯度下降；此处用预计算缓存演示原理
    """
    best = RetrievalResult(-1, None, None, -1.0)

    for shape_idx in top_k_shapes:
        for pose_idx, (R, t) in enumerate(candidate_poses):
            key = (shape_idx, pose_idx)
            if key not in precomputed_feats:
                continue

            synth_feat = precomputed_feats[key]          # 取预计算特征
            score = extractor.patch_similarity(
                query_feat, synth_feat                   # patch 级余弦相似度
            ).item()

            if score > best.score:
                best = RetrievalResult(shape_idx, R, t, score)

    return best
```

### 3D 可视化

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def visualize_retrieval(query_img: Image.Image, shape_pts: np.ndarray,
                        result: RetrievalResult):
    """展示检索结果三联图：查询图像 / 3D 形状 / 最优姿态投影"""
    fig = plt.figure(figsize=(14, 4))

    # 1. 查询图像
    ax1 = fig.add_subplot(131)
    ax1.imshow(query_img)
    ax1.set_title(f"Query\n(Score: {result.score:.3f})")
    ax1.axis('off')

    # 2. 检索到的 3D 形状（降采样显示）
    ax2 = fig.add_subplot(132, projection='3d')
    pts = shape_pts[::10]
    ax2.scatter(pts[:,0], pts[:,1], pts[:,2],
                s=0.5, c=pts[:,2], cmap='viridis', alpha=0.6)
    ax2.set_title(f"Shape #{result.shape_idx}")
    ax2.set_box_aspect([1, 1, 1])

    # 3. 最优姿态的深度图投影
    ax3 = fig.add_subplot(133)
    K = make_camera_K()
    depth_map, _ = project_pointcloud(shape_pts, result.R, result.t, K)
    ax3.imshow(depth_map, cmap='plasma')
    ax3.set_title("Best Pose Projection")
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('pasr_result.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## 实验

### 数据集说明

| 数据集 | 类型 | 规模 | 特点 |
|--------|------|------|------|
| ShapeNet55 | 合成 | 5.1 万模型 | 标准 benchmark，干净 |
| OmniObject3D | 合成+真实纹理 | 6000 对象 | 更接近真实 |
| CO3D | 真实照片 | 19k 序列 | 多视角真实场景 |

**数据难点**：训练用 CAD 模型（完整、光滑），推理面对真实照片（噪声、光照变化、背景杂乱）。这个 sim-to-real gap 是所有方法共同的痛点。

### 定量评估

| 方法 | 净场景 Top-1 | 遮挡场景 Top-1 | 推理速度 | 姿态估计 |
|------|------------|--------------|---------|--------|
| OpenShape | 62.3% | 41.2% | ~10ms | ✗ |
| Uni3D | 65.8% | 44.7% | ~12ms | ✗ |
| **PASR** | **81.4%** | **73.6%** | **2-5s** | ✓ |

PASR 在遮挡场景下领先约 **29 个百分点**，代价是速度慢 100-500 倍。这个 trade-off 决定了它的使用场景。

## 工程实践

### 最大瓶颈：预计算是必须的

每次检索都对所有候选渲染提取 DINOv2 特征，这是不可接受的。**离线预计算并缓存**是部署的前提：

```python
# 离线构建特征缓存（一次性，可存磁盘）
feature_cache = {}
for shape_idx, shape_pts in enumerate(shape_database):
    for pose_idx, (R, t) in enumerate(candidate_poses):
        render = render_shape_to_image(shape_pts, R, t)  # 需要渲染器
        feat = extractor.extract(render).half()          # fp16 节省 50% 内存
        feature_cache[(shape_idx, pose_idx)] = feat
```

5 万个形状 × 500 个姿态 × fp16 特征 = 约 300GB，需要分片加载。

### 两阶段加速策略

```python
# 阶段 1：全局特征粗筛（毫秒级），筛出 top-20
global_query = query_feat.mean(dim=(0, 1))              # (D,)
scores = torch.stack([
    F.cosine_similarity(global_query, db_global[i], dim=0)
    for i in range(len(shape_database))
])
top_k = torch.topk(scores, k=20).indices.tolist()

# 阶段 2：仅对 top-20 做 patch 级精搜（秒级）
result = pasr_retrieval(query_feat, feature_cache, candidate_poses, top_k, extractor)
```

### 常见坑

1. **DINOv2 输入尺寸必须是 14 的倍数**（224、336、448），否则 patch 数量计算出错，直接报维度不匹配
2. **背景 patch 污染匹配**：先用 SAM 或简单深度过滤做前景分割，只对物体区域的 patch 计算相似度
3. **姿态搜索空间爆炸**：不要均匀采样欧拉角（存在万向节锁），改用 SO(3) 上的 Fibonacci 球采样，300-1000 个姿态通常足够
4. **对称物体的姿态退化**：圆柱、球体等高对称形状，多个姿态的投影几乎相同，姿态估计不可靠，但形状检索仍然正确

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 形状库固定有限（< 10 万） | 开放式无限形状检索 |
| 遮挡情况常见（< 70%） | 严重遮挡（> 70%）|
| 精度优先，允许秒级延迟 | 实时应用（< 100ms） |
| 工业零件识别、医疗器械 | 消费级实时 AR/VR |
| 同时需要姿态估计 | 嵌入式/边缘设备 |

## 与其他方法对比

| 方法 | 核心思路 | 遮挡鲁棒性 | 速度 | 可解释性 | 姿态估计 |
|-----|---------|-----------|------|--------|--------|
| CLIP-based | 语言-视觉对齐 | 差 | 极快 | 低 | ✗ |
| OpenShape | 多模态对比学习 | 中 | 快 | 低 | ✗ |
| 传统局部特征匹配 | SIFT/ORB 匹配 | 强 | 中 | 高 | ✓ |
| **PASR** | 分析-合成优化 | **强** | 慢 | **高** | ✓ |

PASR 填补了"高精度 + 遮挡鲁棒"场景下的空缺，在传统特征匹配失效（无纹理物体）而端到端方法又太脆弱的中间地带表现突出。

## 我的观点

**思路很优雅，工程化挑战不容小觑。**

分析-合成框架在计算机视觉中由来已久，PASR 的贡献在于把 DINOv2 的强大 patch 特征和这个框架结合得很干净——遮挡鲁棒性的提升是货真价实的，实验结果也令人信服。

但有几个现实问题需要正视：

- **速度墙**：2-5 秒每次检索，意味着它主要适合离线或批处理场景
- **存储墙**：预计算特征缓存可能达到 TB 级，对小团队不友好
- **渲染依赖**：需要 PyTorch3D 或 NVDiffrast 这类可微渲染器，配置和调试门槛不低

值得关注的后续方向：

1. **扩散模型生成中间视角**：减少对穷举姿态预渲染的依赖
2. **NeRF/3DGS 数据库整合**：用神经渲染代替点云投影，能处理纹理和光照更复杂的场景
3. **轻量化 pose head**：用专用 6DoF 估计网络替代离散搜索，把速度压到 200ms 以内

对于**工业质检、医疗器械识别**等精度优先的领域，PASR 现在就值得认真评估。对于消费级实时应用，建议持续关注——速度优化的空间还很大。