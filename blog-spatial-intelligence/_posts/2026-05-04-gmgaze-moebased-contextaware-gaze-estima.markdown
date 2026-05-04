---
layout: post-wide
title: "注视估计新范式：GMGaze 如何用语义原型、早期融合与稀疏 MoE 突破三大瓶颈"
date: 2026-05-04 08:05:21 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.00799v1
generated_by: Claude Code CLI
---

## 一句话总结

GMGaze 通过语义原型条件化 + 早期多源特征融合 + 稀疏专家混合，在四个主流基准上全面达到 SOTA——在不均匀扩大参数量的前提下，让模型同时感知光照、背景、头姿和外观对注视方向的影响。

---

## 为什么注视估计重要？

注视方向（Gaze Direction）——人眼在三维空间中看向何处——是人机交互的关键信号：

- **驾驶监控**：判断司机是否注意路面，是 L2+ 辅助驾驶的标配能力
- **VR/AR 交互**：眼控 UI、注视渲染（Foveated Rendering）大幅节省 GPU 算力
- **心理与医学**：自闭症诊断、阅读障碍评估、注意力分析
- **广告与用研**：用户页面热力图分析

问题看起来简单（一张人脸图 → 一个方向向量），但三个挑战让精度一直卡在瓶颈：

| 挑战 | 问题描述 |
|------|---------|
| 特征融合太晚 | CNN 特征与 CLIP/Transformer 特征在末层才合并，早期信息已丢失 |
| 缺乏因子感知 | 光照、背景、头姿、外观对注视方向影响不同，但被均匀对待 |
| 容量扩展不实际 | 加大模型能提升精度，但推理代价线性增长，工程上不可接受 |

GMGaze 针对性地解决了这三个问题。论文地址：https://arxiv.org/abs/2605.00799

---

## 背景知识

### 注视方向的数学表示

注视方向通常用**球坐标** (pitch, yaw) 表示，对应三维单位向量：

$$
\mathbf{g} = (\cos\theta\sin\phi,\ \sin\theta,\ \cos\theta\cos\phi)
$$

其中 $\theta$ 为 pitch（俯仰角），$\phi$ 为 yaw（水平偏转角）。评估指标为**平均角度误差（MAE）**：

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} \arccos\!\left(\hat{\mathbf{g}}_i \cdot \mathbf{g}_i^*\right)
$$

人类眼动仪的测量误差约 0.5-1°，模型 MAE < 3° 在多数工业场景已够用。

### CLIP 在注视估计中的角色

CLIP 图像编码器产生两种特征：
- **全局嵌入** $\mathbf{g} \in \mathbb{R}^D$：整图语义，对光照、场景上下文敏感
- **Patch Token 序列** $\{p_i\} \in \mathbb{R}^{N \times D}$：局部细节，捕捉眼部区域

GMGaze 的核心洞察：CLIP 全局嵌入已经对影响注视估计的"外部因子"（光照、背景）非常敏感，但如果直接使用，这些因子的影响是混叠在一起的——需要分解和精细调制。

---

## 核心方法

### 直觉解释

想象你估计一个人的注视方向时，会同时考虑：
- **光照**（强侧光使瞳孔位置看起来偏移）
- **背景**（室内/室外场景参照系不同）
- **头部姿态**（头朝左 30° 时的"看正前方"和头朝正时完全不同）
- **个人外观**（眼距宽窄、虹膜颜色深浅）

GMGaze 学习 4 组"原型字典"，每组对应一个因子，用 CLIP 全局嵌入检索最匹配的原型，生成两个"上下文感知全局 token"——第一层就与局部特征融合，而非最后才合并。

### 语义原型条件化

设 CLIP 全局嵌入为 $\mathbf{g} \in \mathbb{R}^D$，第 $k$ 个因子的原型库为 $\mathbf{P}_k \in \mathbb{R}^{K \times D}$（$K$ 个可学习原型）。通过软注意力检索：

$$
\mathbf{f}_k = \text{softmax}\!\left(\frac{\mathbf{g}\,\mathbf{P}_k^\top}{\sqrt{D}}\right)\mathbf{P}_k
$$

两个互补全局 token 通过拼接后投影生成：

$$
\mathbf{t}_1 = \text{MLP}_1([\mathbf{g};\,\mathbf{f}_1;\,\mathbf{f}_2]), \quad \mathbf{t}_2 = \text{MLP}_2([\mathbf{g};\,\mathbf{f}_3;\,\mathbf{f}_4])
$$

为防止两个 token 学到冗余信息，引入**特征分离损失**（Feature Separation Loss）：

$$
\mathcal{L}_{\text{sep}} = \left\|\frac{\mathbf{t}_1^\top\,\mathbf{t}_2}{\|\mathbf{t}_1\|\cdot\|\mathbf{t}_2\|}\right\|^2
$$

### Pipeline 概览

```
人脸图像
  ├─ CLIP 编码器 ─→ 全局嵌入 g + Patch Tokens
  │                       ↓
  │           ┌─────────────────────────┐
  │           │  语义原型条件化模块      │
  │           │  4个因子原型库 × 软注意力│
  │           │  → 互补全局token t1, t2  │
  │           └─────────────────────────┘
  └─ CNN 主干 ─→ 局部特征 Tokens
  
早期统一融合（第1层）: [t1, t2, CLIP Patch, CNN Token]
              ↓
多尺度 Transformer（每层含稀疏 MoE 替换标准 FFN）
              ↓
          注视方向 (pitch, yaw)
```

---

## 实现

### 语义原型条件化模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticPrototypeConditioner(nn.Module):
    """语义原型条件化：用CLIP全局嵌入检索4个因子原型，生成2个互补全局token"""
    
    def __init__(self, embed_dim=512, num_prototypes=16):
        super().__init__()
        self.embed_dim = embed_dim
        # 4个因子原型库：[光照, 背景, 头部姿态, 外观]
        self.banks = nn.ParameterList([
            nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)
            for _ in range(4)
        ])
        # 两个互补token的投影层：输入为 [g; f_k; f_{k+1}]
        self.proj1 = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.LayerNorm(embed_dim))
        self.proj2 = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.LayerNorm(embed_dim))
    
    def _soft_retrieve(self, query, bank):
        """软注意力检索：query[B,D] × bank[K,D] → context[B,D]"""
        attn = F.softmax(query @ bank.T / self.embed_dim**0.5, dim=-1)
        return attn @ bank
    
    def forward(self, clip_global):
        # clip_global: [B, D]
        f = [self._soft_retrieve(clip_global, bank) for bank in self.banks]
        t1 = self.proj1(torch.cat([clip_global, f[0], f[1]], dim=-1))  # 光照+背景
        t2 = self.proj2(torch.cat([clip_global, f[2], f[3]], dim=-1))  # 头姿+外观
        return t1, t2
    
    @staticmethod
    def separation_loss(t1, t2):
        """特征分离损失：惩罚两个token的余弦相关性，鼓励去相关"""
        cos_sim = F.cosine_similarity(t1, t2, dim=-1)
        return (cos_sim ** 2).mean()
```

### 稀疏 MoE FFN 块

```python
class SparseMoEFFN(nn.Module):
    """稀疏专家混合FFN：每个token只激活top-k专家，实现条件计算"""
    
    def __init__(self, embed_dim=512, num_experts=8, top_k=2, ffn_ratio=4):
        super().__init__()
        self.top_k = top_k
        ffn_dim = embed_dim * ffn_ratio
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim), nn.GELU(),
                nn.Linear(ffn_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        x_flat = x.view(-1, D)                                        # [BN, D]
        router_w = F.softmax(self.router(x_flat), dim=-1)             # [BN, E]
        topk_w, topk_idx = router_w.topk(self.top_k, dim=-1)          # [BN, k]
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)                # 归一化
        
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e_idx in range(len(self.experts)):
                mask = (topk_idx[:, k] == e_idx)
                if mask.any():
                    out[mask] += topk_w[mask, k:k+1] * self.experts[e_idx](x_flat[mask])
        
        return out.view(B, N, D)
    
    @staticmethod
    def load_balance_loss(router_probs):
        """Switch Transformer式负载均衡正则化，防止专家坍塌"""
        # router_probs: [BN, E]
        fraction = router_probs.mean(0)           # 每个专家的平均分配比例
        mean_prob = router_probs.mean(0)
        return (fraction * mean_prob).sum() * router_probs.shape[-1]
```

### GMGaze 主模型骨架

```python
class GMGaze(nn.Module):
    """
    GMGaze简化骨架实现
    完整实现参考论文: https://arxiv.org/abs/2605.00799
    """
    def __init__(self, embed_dim=512, num_layers=6, num_experts=8):
        super().__init__()
        self.clip_proj = nn.Linear(512, embed_dim)   # CLIP全局嵌入对齐
        self.patch_proj = nn.Linear(512, embed_dim)  # CLIP Patch token对齐
        
        self.conditioner = SemanticPrototypeConditioner(embed_dim)
        
        # 多尺度Transformer层，标准FFN替换为稀疏MoE
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn':  nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'moe':   SparseMoEFFN(embed_dim, num_experts),
            }) for _ in range(num_layers)
        ])
        
        self.gaze_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
    
    def forward(self, clip_global, clip_patches):
        # clip_global: [B, D]  clip_patches: [B, N, D]
        g = self.clip_proj(clip_global)             # [B, D]
        patches = self.patch_proj(clip_patches)     # [B, N, D]
        
        # 语义原型条件化 → 两个全局token
        t1, t2 = self.conditioner(g)               # [B, D] each
        
        # 早期统一融合：第1层输入包含全局token + patch token
        tokens = torch.cat([t1.unsqueeze(1), t2.unsqueeze(1), patches], dim=1)
        
        # 多尺度Transformer + MoE逐层处理
        for layer in self.layers:
            res = tokens
            tokens = layer['norm1'](tokens)
            attn_out, _ = layer['attn'](tokens, tokens, tokens)
            tokens = res + attn_out
            tokens = tokens + layer['moe'](layer['norm2'](tokens))
        
        # 用t1对应位置（index=0）的token预测注视方向
        return self.gaze_head(tokens[:, 0])         # [B, 2] → (pitch, yaw)
```

### 注视方向 3D 可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_gaze(gt_angles, pred_angles, n_samples=8):
    """可视化GT与预测注视向量及误差分布"""
    
    def to_vector(pitch, yaw):
        return np.stack([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ], axis=-1)
    
    gt = to_vector(gt_angles[:, 0], gt_angles[:, 1])
    pred = to_vector(pred_angles[:, 0], pred_angles[:, 1])
    angular_err = np.degrees(np.arccos(np.clip((gt * pred).sum(-1), -1, 1)))
    
    fig = plt.figure(figsize=(12, 5))
    
    # 左：3D注视向量对比
    ax = fig.add_subplot(121, projection='3d')
    origin = np.zeros((n_samples, 3))
    ax.quiver(*origin.T, *gt[:n_samples].T, color='royalblue', label='GT')
    ax.quiver(*origin.T, *pred[:n_samples].T, color='tomato',
              linestyle='dashed', label='Pred')
    ax.set(xlim=[-1,1], ylim=[-1,1], zlim=[-1,1], title='注视方向3D对比')
    ax.legend()
    
    # 右：角度误差分布
    ax2 = fig.add_subplot(122)
    ax2.hist(angular_err, bins=30, color='steelblue', edgecolor='white')
    ax2.axvline(angular_err.mean(), color='red', linestyle='--',
                label=f'MAE = {angular_err.mean():.2f}°')
    ax2.set(xlabel='角度误差 (°)', ylabel='样本数', title='误差分布')
    ax2.legend()
    
    plt.tight_layout()
    return angular_err.mean()
```

---

## 实验

### 数据集说明

| 数据集 | 规模 | 特点 | 获取 |
|--------|------|------|------|
| MPIIFaceGaze | ~45K 图 | 真实场景，笔记本摄像头 | 公开申请 |
| EYEDIAP | ~16K 片段 | 头部+眼动双任务 | 公开申请 |
| Gaze360 | ~172K 图 | 360° 全向注视 | 公开 |
| ETH-XGaze | ~1M 图 | 受控实验室，多相机高质量 | 公开申请 |

**自采数据建议**：使用 Tobii 眼动仪提供高精度 GT；覆盖多种光照条件；保证 20+ 受试者防止人脸过拟合。

### 定量评估（MAE，°，越低越好）

| 方法 | MPIIFaceGaze | EYEDIAP | Gaze360 | ETH-XGaze |
|------|:-----------:|:-------:|:-------:|:---------:|
| GazeTR | 4.00 | 4.62 | 10.62 | 2.10 |
| CLIP-Gaze | 3.20 | 4.01 | 11.13 | 1.86 |
| **GMGaze** | **2.49** | **3.22** | **10.16** | **1.44** |

跨域评估（ETH→MPII、MPII→EYEDIAP）GMGaze 均达到 SOTA，体现了原型条件化 + 对抗域适应的泛化优势。

---

## 工程实践

### 实际部署考虑

- **推理速度**：RTX 3080 上约 15-30ms/帧（30+ FPS），但 MoE 稀疏路由在 `batch_size=1` 时因条件分支无法并行，效率折损约 30%——生产部署建议批量推理或 TensorRT 算子融合
- **内存占用**：CLIP ViT-B/16 约 400MB，总推理内存 ~2GB，移动端 GPU 压力较大
- **前处理依赖**：GMGaze 以裁剪好的人脸为输入，实际系统还需 RetinaFace/MediaPipe 先做人脸检测，总端到端延迟需累加

### 常见坑

**1. CLIP 特征和 CNN 特征尺度不匹配**

```python
# 错误：直接拼接不同归一化尺度的特征
tokens = torch.cat([clip_tokens, cnn_tokens], dim=1)

# 正确：各自 LayerNorm 后再融合
tokens = torch.cat([
    F.layer_norm(clip_tokens, [D]),
    F.layer_norm(cnn_tokens, [D])
], dim=1)
```

**2. MoE 训练出现专家坍塌（Expert Collapse）**

```python
# 所有token涌向少数几个专家，其他专家几乎不更新
# 解决方案：在总损失中加入负载均衡正则化
total_loss = gaze_loss + 0.01 * SparseMoEFFN.load_balance_loss(router_probs) \
                       + 0.1  * SemanticPrototypeConditioner.separation_loss(t1, t2)
```

**3. 跨域测试时 BatchNorm 统计量污染**

```python
# 推理时务必切换到 eval 模式，否则 BN 使用当前小批次统计，导致跨域性能不稳定
model.eval()
with torch.no_grad():
    pred_angles = model(clip_global, clip_patches)  # [B, 2]
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 静态或慢速头部运动 | 头部极速摇晃（运动模糊严重） |
| RGB 单目摄像头 | 需要纯深度相机方案 |
| 室内控制场景（驾驶舱、会议室） | 强逆光或阳光直射遮挡眼部 |
| 需要跨设备/跨场景部署 | 资源受限嵌入式端侧（< 1GB RAM） |
| 精度要求 MAE < 3° | 离线科研级精度（需专业眼动仪标定） |

---

## 与其他方法对比

| 方法 | 核心思路 | 优点 | 缺点 |
|-----|---------|------|------|
| CNN+FC | 直接回归 pitch/yaw | 速度快，易部署 | 精度上限低，域迁移差 |
| GazeTR | 纯 Transformer | 全局注意力 | 无因子感知，融合晚 |
| CLIP-Gaze | CLIP 特征提取 | 语义先验强 | 单一全局 token，无条件分支 |
| **GMGaze** | 原型条件化 + 早期融合 + MoE | 多因子感知，跨域强 | 推理更重，MoE 调试复杂 |

---

## 我的观点

GMGaze 最值得借鉴的设计哲学是：**把"什么因素影响注视"显式建模进架构，而非靠参数量蛮力学习**。4 个原型库对应 4 个物理因子，这种归纳偏置（inductive bias）让模型在数据有限时也能有效泛化。这和 ControlNet 向扩散模型注入条件信号的思路一脉相承。

**几个开放问题值得关注：**

1. **时序建模缺失**：视频流中的注视估计需要时序平滑，当前方法每帧独立推理容易跳变——Mamba 状态空间模型是轻量级的自然延伸方向

2. **从角度到 3D 落点**：目前估计的是头部坐标系下的方向角，在 AR/VR 中定位注视在三维场景中的落点，还需要深度信息或双目视差

3. **端侧压缩**：CLIP + MoE 对移动端不友好。将 4 个专家蒸馏到 1 个条件化轻量网络，是实际落地的必经之路

**离产品化的距离**：对于驾驶监控、会议注意力分析，MAE < 3° 已够用。GMGaze 在 MPIIFaceGaze 上 2.49° 非常接近眼动仪的测量误差下界。真正的部署瓶颈不在算法，而在**标注成本**（精确 GT 依赖专业眼动仪，单次采集费用高）和**长尾数据覆盖**（遮挡眼镜、极端光照、非标准摄像头角度的泛化能力仍有缺口）。