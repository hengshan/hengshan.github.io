---
layout: post-wide
title: "不依赖深度学习的医学体积图像无损压缩：三平面上下文树（TCT）解析"
date: 2026-08-17 12:04:08 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.13897v1
generated_by: Claude Code CLI
---

## 一句话总结

用三平面上下文树（TCT）将复杂的3D上下文建模分解为三个正交2D平面，在不依赖任何神经网络的前提下，实现与DNN方法媲美的无损医学体积图像压缩性能。

## 为什么这篇论文重要？

医学图像压缩是个"死亡沼泽"——任何像素级别的损失都可能导致误诊。无损压缩是唯一选择，但CT、MRI等体积数据动辄数百MB，对带宽和存储都是巨大压力。

**现有方法的两难困境：**
- 传统方法（JPEG-LS、FLAC）：快速、无需训练，但压缩率受限于手工设计的预测模型
- DNN方法（基于Hyperprior的神经压缩）：压缩率高，但推理需要大量GPU显存，在边缘医疗设备、嵌入式系统上难以部署

这篇论文的核心洞见是：**"学习"不一定需要神经网络**。它借鉴了上下文树权重（CTW）的经典算法思想，在推理时针对每个输入体积自适应学习一个轻量级模型，同时用三平面分解把3D上下文建模的复杂度压回到2D量级。

**我的判断**：这篇论文的真正贡献不是"打败DNN"，而是证明了测试时学习（test-time learning）在信息论框架下的可行性——MDL优化本质上是一种在线元学习，目标函数是编码长度而非任务损失。

## 核心方法解析

### 上下文预测压缩的基本逻辑

无损压缩的本质是**预测**：如果我能精确预测下一个像素，残差接近0，用很少的比特就能编码它。信息论告诉我们，最优编码长度是：

$$L = -\log_2 P(x \mid \text{context})$$

所以问题转化为：**如何对3D体积中的每个体素，精确估计给定已编码邻域的条件概率？**

直接建模3D邻域的问题：上下文空间是指数级的——如果邻域有27个8bit体素，状态空间是 $256^{27}$，根本无法穷举，也无法做可靠的频率统计。

### 三平面分解：把3D问题降回2D

TCT的解法是**正交分解**：对于体素 $(z, y, x)$，不建模完整3D邻域，而是分别在三个正交平面上提取上下文：

- **XY平面**：当前切片 $z$ 上的因果邻域（上方、左方等已编码像素）——捕捉切片内空间相关性
- **XZ平面**：固定行 $y$，在前序切片的相同行附近采样——捕捉Z轴跨层相关性
- **YZ平面**：固定列 $x$，在前序切片的相同列附近采样——捕捉另一方向的跨层相关性

三者拼接得到一个紧凑的上下文向量，同时保留了切片内和跨层两种关键冗余。

```python
import numpy as np

def extract_triplane_context(volume: np.ndarray, z: int, y: int, x: int) -> np.ndarray:
    D, H, W = volume.shape
    ctx = []

    # XY平面：当前切片内的因果邻域
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (-2, 0), (0, -2)]:
        ny, nx = y + dy, x + dx
        ctx.append(int(volume[z, ny, nx]) if 0 <= ny < H and 0 <= nx < W else 0)

    # XZ/YZ平面：前两层切片，分别沿x/y方向采样跨层相关性
    for dz in [1, 2]:
        nz = z - dz
        for dx in [-1, 0, 1]:
            ctx.append(int(volume[nz, y, x + dx]) if nz >= 0 and 0 <= x + dx < W else 0)
        for dy in [-1, 0, 1]:
            ctx.append(int(volume[nz, y + dy, x]) if nz >= 0 and 0 <= y + dy < H else 0)

    return np.array(ctx, dtype=np.int32)  # shape: (18,)
```

### 上下文树：自适应的上下文索引结构

有了18维的上下文向量，下一步是把它映射到一个**预测器**。TCT用自适应二叉树完成这个映射：

- 每个**内部节点**根据某维上下文特征的值进行二分
- 每个**叶节点**存储：① 一个线性预测器权重向量 ② 该上下文下的残差频率直方图

```python
class TCTNode:
    """上下文树节点"""
    def __init__(self):
        self.histogram: dict = defaultdict(int)  # residual -> count
        self.predictor: Optional[np.ndarray] = None  # 线性预测权重
        self.split_feat: Optional[int] = None
        self.split_thresh: Optional[float] = None
        self.left: Optional['TCTNode'] = None
        self.right: Optional['TCTNode'] = None

    def is_leaf(self) -> bool:
        return self.split_feat is None

    def route(self, context: np.ndarray) -> 'TCTNode':
        """根据上下文特征路由到对应叶节点"""
        node = self
        while not node.is_leaf():
            node = node.left if context[node.split_feat] <= node.split_thresh else node.right
        return node

    def predict(self, context: np.ndarray, fallback: int = 0) -> int:
        if self.predictor is not None:
            return int(np.clip(round(float(self.predictor @ context)), 0, 255))
        return fallback  # 退化为左邻像素预测
```

### MDL 原则：何时分裂、何时剪枝

如何决定树的深度和分裂点？论文使用**最小描述长度（MDL）**原则：

$$\text{MDL} = L_{\text{model}} + L_{\text{data} \mid \text{model}}$$

叶节点的数据编码代价用残差直方图估计（带 Laplace 平滑避免零概率）：

$$L_{\text{data}}(S) = -\sum_{r \in S} \log_2 \hat{P}(r), \quad \hat{P}(r) = \frac{\text{count}(r) + \alpha}{\lvert S \rvert + \alpha \cdot V}$$

其中 $V$ 是残差字母表大小（8bit图像为511），$\alpha$ 是平滑系数。

**分裂决策**：只有当分裂后总 MDL 减小时才执行分裂，否则保持叶节点。

```python
def node_coding_cost(histogram: dict, alpha: float = 0.5, V: int = 511) -> float:
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    # 拉普拉斯平滑后的负对数似然
    return -sum(c * np.log2((c + alpha) / (total + alpha * V) + 1e-12)
                for c in histogram.values())

def find_best_split(contexts, residuals, min_samples=50):
    if len(residuals) < min_samples * 2:
        return None, None, 0.0

    baseline = node_coding_cost(Counter(residuals.astype(int)))
    best_gain, best_feat, best_thresh = 0.0, None, None
    MODEL_OVERHEAD = 8.0  # 节点描述的粗略比特代价

    for feat_idx in range(contexts.shape[1]):
        for thresh in np.unique(contexts[:, feat_idx])[:-1]:
            mask = contexts[:, feat_idx] <= thresh
            if mask.sum() < min_samples or (~mask).sum() < min_samples:
                continue
            # ... (左右子集直方图构建省略)
            gain = baseline - node_coding_cost(left_hist) - node_coding_cost(right_hist) - MODEL_OVERHEAD
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat_idx, float(thresh)

    return best_feat, best_thresh, best_gain
```

### 完整流程：采样学习 + 逐体素编码

```python
def build_tct(volume: np.ndarray, sample_ratio: float = 0.1, max_depth: int = 8) -> TCTNode:
    # 采样：随机选取体素，提取上下文和残差
    contexts, residuals = [], []
    for z, y, x in sampled_voxels(volume, sample_ratio):  # ... (采样遍历代码省略)
        ctx = extract_triplane_context(volume, z, y, x)
        pred = int(volume[z, y, x - 1]) if x > 0 else 128
        contexts.append(ctx)
        residuals.append(int(volume[z, y, x]) - pred)
    contexts, residuals = np.array(contexts), np.array(residuals)

    # 递归构建树
    def build_node(ctx_subset, res_subset, depth) -> TCTNode:
        node = TCTNode()
        node.histogram = Counter(res_subset)  # ... (直方图统计代码省略)
        if depth < max_depth:
            feat, thresh, gain = find_best_split(ctx_subset, res_subset)
            if gain > 0:
                node.split_feat, node.split_thresh = feat, thresh
                mask = ctx_subset[:, feat] <= thresh
                node.left = build_node(ctx_subset[mask], res_subset[mask], depth + 1)
                node.right = build_node(ctx_subset[~mask], res_subset[~mask], depth + 1)
        return node

    return build_node(contexts, residuals, depth=0)
```

## 实现中的坑

**坑1：因果性约束——编解码顺序必须完全一致**

压缩时按 Z→Y→X 扫描，上下文必须只引用**已经编码**的体素。一旦定义不严格，编解码器读到的值不同，直接产生乱码，且无报错。

```python
# 错误：访问了尚未编码的像素
ctx.append(volume[z, y, x + 1])  # x+1 还没到！

# 正确：只用 (nz < z) 或 (nz==z 且 ny<y) 或 (nz==z, ny==y 且 nx<x) 的位置
```

**坑2：MDL 模型传输代价不能简单设为常数**

树本身需要随压缩码流传输给解码器。节点越多，传输代价越高。对小体积（$< 64^3$），模型传输开销可能超过数据编码增益，导致压缩率不升反降。经验值：叶节点样本数少于 50 时停止分裂。

**坑3：CT 图像是 12bit，不是 8bit**

标准 CT 的 Hounsfield 值范围约为 $[-1024, +3071]$，存储为 16bit 有符号整数。直接用 8bit 的残差字母表 $V=511$ 会严重低估平滑代价。需要针对具体模态调整：

```python
# 针对12bit CT调整字母表大小（残差范围约 ±4095）
V_CT = 8191
cost = node_coding_cost(histogram, V=V_CT)
```

## 实验：论文说的 vs 现实

论文报告在多个医学图像数据集上压缩率与近期 DNN 方法相当，编解码速度更快。

**现实中需要注意的限制：**
- **各向异性数据受益有限**：如2mm层厚、0.5mm面内分辨率的CT，XZ/YZ平面的预测价值会明显下降，因为相邻切片的空间距离远大于面内
- **采样率是个隐藏超参数**：`sample_ratio=0.1` 对 512×512×300 的体积意味着约790万样本，学习时间可能达数十秒；过小则树质量下降
- **与DNN基线的对比条件不够透明**：论文未详细说明DNN方法的推理内存和时间，"在同等资源预算下"的对比才有工程意义

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 边缘/嵌入式医疗设备，无GPU | 研究环境、算力充足、追求极限压缩率 |
| 无外部训练数据，需要开箱即用 | 体积极小（$< 32^3$），模型传输开销显著 |
| 各向同性体素（如1mm³ CT） | 高度稀疏的分割掩码（游程编码更合适） |
| 担心训练集分布与新采集协议不匹配 | 对部署工程复杂度不敏感的场景 |

## 我的观点

这篇论文最有意思的地方不是压缩率数字，而是它揭示了一个被忽视的设计空间：**测试时自适应的上下文建模**。

MDL + 自适应树在本质上是一种每个输入都重新学习模型的机制。这与神经压缩中的"测试时微调"思路异曲同工，但计算代价低得多，且无需梯度回传。这个思路在视频压缩中已有苗头，但在医学图像领域还相对少见。

这类方法真正的工程优势不是压缩率，而是**零外部依赖**和**天然的领域鲁棒性**：不需要担心训练集分布和新采集协议不匹配，每个体积自己学自己的模型，部署即用。这在医院IT环境中比任何压缩率提升都更实用。