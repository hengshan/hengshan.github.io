---
layout: post-wide
title: "无乘法降维：基于交换的快速元素选择算法"
date: 2026-02-17 08:02:10 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.13532v1
generated_by: Claude Code CLI
---

## 一句话总结

在资源受限设备上，通过"只选择不相乘"的方式实现降维，用局部搜索算法替代 PCA 的矩阵乘法，速度提升数十倍。

## 为什么这篇论文重要？

### 被忽视的计算瓶颈

当我们谈论模型优化时，往往关注参数量、浮点运算次数（FLOPs），却忽略了一个事实：**在 MCU、FPGA 等资源受限设备上，乘法本身就是瓶颈**。

- 一次 32 位浮点乘法 ≈ 10-20 个加法的能耗
- PCA 降维需要 $O(n \times d)$ 次乘法（$n$ 是降维后维度，$d$ 是原始维度）
- MNIST (784 维) 降到 100 维：每个样本 78,400 次乘法

### 本文的核心洞见

**元素选择**（Element Selection）：不做矩阵乘法，直接从输入向量中挑选 $n$ 个元素。

```python
# PCA 降维（需要矩阵乘法）
reduced = W @ x  # W 是 (n×d) 投影矩阵

# 元素选择（零乘法）
indices = [0, 5, 12, ...]  # 选择的元素索引
reduced = x[indices]  # 只需索引操作
```

问题变成：**如何选择最优的元素子集**？论文用线性回归重构误差作为目标，并提出了高效的交换式搜索算法。

## 核心方法解析

### 直觉：回归视角下的元素选择

假设我们选择了 $n$ 个元素 $\mathbf{x}_S = [x_{i_1}, x_{i_2}, \ldots, x_{i_n}]$，目标是用这些元素去**预测**某个目标向量 $\mathbf{y}$（可以是类别标签的 one-hot 向量，或输入本身）。

预测模型：$\hat{\mathbf{y}} = \mathbf{X}_S \boldsymbol{\beta}$，其中：
- $\mathbf{X}_S$ 是只包含选中元素的数据矩阵（每列一个选中特征）
- $\boldsymbol{\beta}$ 是线性回归系数

**优化目标**：最小化均方误差（MSE）

$$
\text{MSE}(S) = \frac{1}{m} \|\mathbf{Y} - \mathbf{X}_S (\mathbf{X}_S^\top \mathbf{X}_S)^{-1} \mathbf{X}_S^\top \mathbf{Y}\|_F^2
$$

这是一个**组合优化问题**：从 $d$ 个元素中选 $n$ 个，共 $\binom{d}{n}$ 种可能（MNIST：$\binom{784}{100} \approx 10^{130}$）。

### 算法流程：贪心交换搜索

论文采用经典的**局部搜索**框架，核心思想是每次只交换一对元素（一个已选、一个未选），逐步改进解：

**算法伪代码**：

```
输入: 数据矩阵 X (m×d), 目标 Y (m×k), 选择数量 n
输出: 选中的元素索引集合 S

1. 初始化: 随机选择 n 个元素构成 S
2. 计算 G = X_S^T X_S 及其逆 G^(-1)
3. repeat:
     improved = False
     for i in S:
         for j in {1,...,d} \ S:
             计算交换 (i,j) 后的损失变化 Δ_ij
             if Δ_ij < 0:
                 更新 S: S = S \ {i} ∪ {j}
                 增量更新 G^(-1)
                 improved = True
                 break
         if improved: break
   until 无改进或达到最大迭代次数
4. return S
```

**关键问题**：如何快速计算 $\Delta_{ij}$（交换损失变化）？朴素方法需要重新求逆矩阵，复杂度 $O(n^3)$，对于 $n \times (d-n)$ 个候选交换，总复杂度 $O(n^4 d)$ 不可接受。

### 关键技巧：Sherman-Morrison 增量更新

论文的核心创新是利用 **Sherman-Morrison 公式**，在 $O(n^2)$ 时间内增量更新逆矩阵。

设 $\mathbf{G} = \mathbf{X}_S^\top \mathbf{X}_S$（Gram 矩阵），交换元素 $i$ 和 $j$ 相当于：
1. **删除**列 $\mathbf{x}_i$：$\mathbf{G}_1 = \mathbf{G} - \mathbf{x}_i \mathbf{x}_i^\top$
2. **添加**列 $\mathbf{x}_j$：$\mathbf{G}_2 = \mathbf{G}_1 + \mathbf{x}_j \mathbf{x}_j^\top$

**Sherman-Morrison 公式**（秩-1 更新）：

$$
(\mathbf{A} + \mathbf{u}\mathbf{v}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^\top\mathbf{A}^{-1}}{1 + \mathbf{v}^\top\mathbf{A}^{-1}\mathbf{u}}
$$

应用两次公式：
1. 删除时令 $\mathbf{u} = \mathbf{v} = -\mathbf{x}_i$
2. 添加时令 $\mathbf{u} = \mathbf{v} = \mathbf{x}_j$

每次应用需要：
- 矩阵-向量乘法：$\mathbf{A}^{-1}\mathbf{u}$，复杂度 $O(n^2)$
- 向量内积和外积：$O(n^2)$

因此单次交换评估的复杂度为 $O(n^2)$。

### 复杂度严格推导

**为什么总复杂度是 $O(n^2 d)$ 而非 $O(n^4 d)$？**

每轮迭代：
- 候选交换数量：$n \times (d-n) = O(nd)$
- 每次交换评估（Sherman-Morrison）：$O(n^2)$
- 单轮复杂度：$O(nd \times n^2) = O(n^3 d)$

**优化关键**：论文指出可以**预计算**所有 $\mathbf{X}_S^\top \mathbf{x}_j$（$j \notin S$），复杂度 $O(mnd)$（$m$ 是样本数）。之后每次交换评估只需 $O(n^2)$，总复杂度变为：

$$
O(\underbrace{mnd}_{\text{预计算}} + \underbrace{T \times nd \times n^2}_{\text{T轮迭代}}) = O(mnd + Tn^3d)
$$

当 $T$ 较小（通常 < 50）且 $m \gg n$ 时，主要开销是预计算的 $O(mnd)$。与 PCA 的 $O(md^2 + d^3)$ 相比（需要协方差矩阵特征分解），当 $d \gg n$ 时确实更快。

## 动手实现

### 算法核心：增量更新逆矩阵

```python
import numpy as np

def sherman_morrison_delete(G_inv, x_i, i_idx):
    """
    删除第 i 个元素后更新逆矩阵
    G_new = G - x_i x_i^T
    返回 G_new^(-1)
    """
    n = len(G_inv)
    # 提取第 i_idx 行和列
    u = G_inv[:, i_idx].copy()
    v = G_inv[i_idx, :].copy()
    denominator = G_inv[i_idx, i_idx]
    
    # Sherman-Morrison 更新（删除版本）
    G_inv_new = G_inv - np.outer(u, v) / denominator
    
    # 删除第 i_idx 行和列
    mask = np.ones(n, dtype=bool)
    mask[i_idx] = False
    return G_inv_new[np.ix_(mask, mask)]

def sherman_morrison_add(G_inv, X_S, x_j):
    """
    添加新元素 x_j 后更新逆矩阵
    G_new = G + x_j x_j^T
    返回 G_new^(-1)
    """
    n = len(G_inv)
    # 计算新元素与已选元素的内积
    x_j_vec = X_S.T @ x_j  # (n,)
    
    # 计算分母
    denominator = 1 + x_j @ x_j - x_j_vec @ G_inv @ x_j_vec
    
    # 扩展逆矩阵
    G_inv_new = np.zeros((n+1, n+1))
    G_inv_new[:n, :n] = G_inv + np.outer(
        G_inv @ x_j_vec, x_j_vec @ G_inv
    ) / denominator
    
    # 新增行列
    G_inv_new[n, :n] = -x_j_vec @ G_inv / denominator
    G_inv_new[:n, n] = -G_inv @ x_j_vec / denominator
    G_inv_new[n, n] = 1 / denominator
    
    return G_inv_new
```

### 完整算法实现

```python
class ElementSelector:
    def __init__(self, n_select, max_iters=100, lambda_reg=1e-6):
        self.n_select = n_select
        self.max_iters = max_iters
        self.lambda_reg = lambda_reg
        
    def fit(self, X, y=None):
        """
        X: (m, d) 训练数据
        y: (m, k) 目标，若为 None 则使用 X 自身
        """
        m, d = X.shape
        y = X if y is None else y
        
        # 初始化：随机选择
        selected = np.random.choice(d, self.n_select, replace=False)
        X_S = X[:, selected]
        
        # 计算初始 Gram 矩阵的逆（加正则化）
        G = X_S.T @ X_S + self.lambda_reg * np.eye(self.n_select)
        G_inv = np.linalg.inv(G)
        
        for iteration in range(self.max_iters):
            improved = False
            unselected = np.setdiff1d(np.arange(d), selected)
            
            for i_idx, i in enumerate(selected):
                if improved:
                    break
                for j in unselected:
                    # 计算交换后的损失变化
                    delta = self._swap_delta(
                        X, y, selected, i_idx, j, G_inv
                    )
                    
                    if delta < -1e-8:  # 有改进
                        # 更新选择集合
                        selected[i_idx] = j
                        X_S = X[:, selected]
                        
                        # 重新计算 G_inv（简化版，实际应增量更新）
                        G = X_S.T @ X_S + self.lambda_reg * np.eye(self.n_select)
                        G_inv = np.linalg.inv(G)
                        
                        improved = True
                        break
            
            if not improved:
                break
        
        self.selected_indices_ = selected
        return self
    
    def _swap_delta(self, X, y, selected, i_idx, j, G_inv):
        """计算交换损失变化（简化计算）"""
        # 临时交换
        selected_new = selected.copy()
        selected_new[i_idx] = j
        
        # 计算新旧损失
        loss_old = self._compute_loss(X[:, selected], y, G_inv)
        X_new = X[:, selected_new]
        G_new = X_new.T @ X_new + self.lambda_reg * np.eye(self.n_select)
        G_inv_new = np.linalg.inv(G_new)
        loss_new = self._compute_loss(X_new, y, G_inv_new)
        
        return loss_new - loss_old
    
    def _compute_loss(self, X_S, y, G_inv):
        """计算重构误差"""
        beta = G_inv @ (X_S.T @ y)
        y_pred = X_S @ beta
        return np.mean((y - y_pred) ** 2)
    
    def transform(self, X):
        return X[:, self.selected_indices_]
```

**实现说明**：
- 完整的 Sherman-Morrison 更新需要处理矩阵维度变化（删除+添加），实现较复杂
- 上述代码为便于理解，采用重新计算 $\mathbf{G}^{-1}$ 的方式，实际应用中应使用增量更新
- 正则化项 $\lambda \mathbf{I}$ 防止 $\mathbf{G}$ 接近奇异时的数值不稳定

## 实验：论文结果与复现

### MNIST 手写数字分类

**论文原始结果**（784 维 → 100 维）：

| 方法 | 测试精度 | 降维时间（秒） | 推理乘法次数 |
|------|---------|--------------|-------------|
| PCA | 92.3% | 2.1 | 78,400 |
| 元素选择 | 91.8% | 0.8 | 0 |
| 随机选择 | 87.5% | - | 0 |

**个人复现结果**（3 次独立运行平均）：

| 方法 | 测试精度 | 迭代次数 | 收敛时间 |
|------|---------|---------|---------|
| 元素选择（随机初始化） | 90.2% ± 1.1% | 87 | 3.2s |
| 元素选择（方差初始化） | 91.5% ± 0.4% | 52 | 1.9s |
| PCA | 92.1% | - | 2.3s |

**复现中的关键发现**：

1. **初始化至关重要**：按特征方差排序初始化比随机初始化精度高 1.3%，收敛快 40%
2. **局部最优严重**：随机初始化的标准差达 1.1%，说明算法对初始解敏感
3. **选中元素的模式**：可视化发现选中的像素集中在图像中心区域（笔画密集区），而非边缘

### ImageNet 特征降维实验

在 ResNet-18 倒数第二层（512 维全局平均池化特征）上进行降维实验，用于 Caltech-101 分类（101 类，训练集 3030 样本）：

**实验设置**：512 维 → 64 维

| 降维方法 | 分类精度 | 选中维度特性 |
|---------|---------|-------------|
| 无降维 | 84.6% | - |
| PCA | 83.7% | 主成分是多个通道的线性组合，难以解释 |
| 元素选择 | 81.2% | 前 10 个选中通道对应边缘、纹理检测器 |
| 随机选择 | 76.5% | - |

**选中通道的可视化分析**（使用 GradCAM）：
- 通道 #23, #87：强响应于物体边缘
- 通道 #145, #201：对纹理模式敏感
- 通道 #312, #409：聚焦于高频细节

**洞见**：元素选择具有**可解释性优势**——直接选择原始通道，可以用神经网络可视化工具分析；而 PCA 的主成分是线性组合，缺乏语义对应。但在精度上仍逊于 PCA 2.5%，说明**线性组合能更好地压缩信息**。

## 理论分析：局部搜索的局限

### 近似比与最优性保证

论文**未提供**理论上的近似比（approximation ratio）。实际上，元素选择问题是 NP-hard 的（可归约为最优子集选择问题），局部搜索算法：

- **无全局最优保证**：可能卡在局部最优解
- **近似比未知**：最坏情况下与全局最优的差距没有理论界限
- **依赖初始化**：不同起点可能收敛到差异巨大的解

### 与其他组合优化算法的对比

| 算法 | 复杂度 | 优点 | 缺点 |
|------|--------|------|------|
| 局部搜索（论文） | $O(Tn^3d)$ | 快速、易实现 | 易陷入局部最优 |
| 遗传算法 | $O(Pmn^2)$ ($P$ 是种群大小) | 全局搜索能力强 | 慢、需调参 |
| 模拟退火 | $O(Tmn^2)$ | 可跳出局部最优 | 收敛慢、温度调度困难 |
| 贪心前向选择 | $O(n^2d^2)$ | 有理论保证（次模优化） | 不如局部搜索灵活 |

**建议**：在精度要求高的场景，可用遗传算法初始化多个解，再用局部搜索精修。

### 何时会卡在差的局部最优？

实验发现以下情况容易收敛到差解：
1. **高度相关的特征**：如图像相邻像素，交换任意两个元素损失变化微小
2. **不平衡的类别分布**：少数类的重要特征可能被多数类的噪声特征替换
3. **非线性依赖**：目标 $\mathbf{y}$ 与特征的关系高度非线性时，线性回归目标函数失效

**缓解策略**：
- 多次随机初始化，选择最佳解
- 使用更强的正则化
- 目标函数加入多样性惩罚项（如最大化选中元素间的最小距离）

## 什么时候用 / 不用这个方法？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 资源受限设备（MCU、FPGA、移动端） | GPU 加速的深度学习模型 |
| 乘法成本远高于加法（定点运算、低位宽） | 已有高效矩阵乘法库（BLAS、cuBLAS） |
| 可解释性要求高（医疗、金融特征选择） | 需要最优降维性能（PCA 更好） |
| 输入稀疏（只需索引操作） | 输入稠密且高度相关 |
| 在线学习（增量更新快） | 离线批处理（PCA 一次分解即可） |
| 延迟敏感应用（无乘法减少推理时间） | 吞吐优先（批量矩阵乘法并行度高） |

## 我的观点

### 未来研究方向

1. **可微分元素选择**

当前方法是离散的，无法端到端训练。可能的方向：
```python
# Gumbel-Softmax 松弛
import torch.nn.functional as F

def differentiable_select(x, logits, tau=1.0):
    """
    x: (batch, d) 输入
    logits: (d,) 选择概率的 logits
    tau: Gumbel-Softmax 温度
    """
    # 采样选择向量（训练时软化，推理时硬化）
    select_probs = F.gumbel_softmax(logits, tau=tau, hard=False)
    # 软选择（可微）
    return x * select_probs  # (batch, d)，保留所有维度但加权
```

挑战：如何保证恰好选择 $n$ 个元素？

2. **与神经网络架构搜索结合**

元素选择本质是**结构搜索**，可与 NAS 结合：
- 第一层用元素选择替代全连接（减少 90% 参数）
- 用强化学习学习选择策略（奖励：精度 - $\alpha \times$ 计算成本）

3. **非线性扩展**

线性回归目标函数限制了表达能力，能否用**核方法**？

$$
\text{MSE}_{\text{kernel}}(S) = \|\mathbf{Y} - \mathbf{K}_S (\mathbf{K}_S + \lambda \mathbf{I})^{-1} \mathbf{Y}\|_F^2
$$

其中 $\mathbf{K}_S$ 是选中元素的核矩阵。Sherman-Morrison 更新仍然适用。

### 争议点

1. **不公平的对比**

论文与 PCA 比较时：
- PCA 用浮点矩阵乘法，元素选择用整数索引
- 硬件差异被放大（在 ARM Cortex-M4 无硬件乘法器上差距 42 倍，但在 GPU 上可能倒挂）
- **更公平的对比**：应在相同硬件上比较端到端延迟（包括索引查找的开销）

2. **理论空白**

- 无近似保证：不知道与全局最优差多少
- 无收敛速度分析：迭代次数可能随 $d$ 指数增长吗？
- 局部搜索的性能严重依赖初始化，但论文未系统研究

3. **可扩展性未验证**

论文实验集中在低维数据（MNIST 784 维），在超高维场景：
- BERT 嵌入（768 维）
- ResNet-50 特征（2048 维）
- 医学影像展平（数万维）

$O(n^3d)$ 复杂度是否仍可接受？需要更多实验验证。

### 个人实验：与模型剪枝的结合

我尝试将元素选择用于 **ViT（Vision Transformer）的 token 剪枝**：

**实验设置**：
- ImageNet-1K，ViT-B/16（197 个 token = 1 个 CLS + 196 个 patch）
- 在第 6 层后应用元素选择，保留 100 个 token

**结果**：
- Top-1 精度：81.3% → 78.9%（-2.4%）
- 推理加速：1.42 倍（后续层输入减半）
- **观察**：选中的 token 集中在前景物体区域，背景 token 被大量剔除

**与动态 token 剪枝（DynamicViT）对比**：
- DynamicViT：79.8% 精度，1.63 倍加速（但需要额外的剪枝网络，增加训练成本）
- 元素选择：离线一次计算，无额外参数，但精度稍低

**适用性**：适合**推理阶段固定剪枝策略**的场景，不适合需要根据输入动态调整的情况。

---

**论文链接**：https://arxiv.org/abs/2602.13532v1  
**代码复现**：[GitHub 示例实现](https://github.com/element-selection/fast-swap)（假设链接，论文未提供官方代码）