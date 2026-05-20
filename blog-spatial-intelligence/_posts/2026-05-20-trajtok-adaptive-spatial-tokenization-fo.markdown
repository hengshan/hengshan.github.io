---
layout: post-wide
title: "TrajTok：GPS 轨迹的自适应六边形 Tokenization 与迁移学习"
date: 2026-05-20 12:05:36 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2605.20134v1
generated_by: Claude Code CLI
---

## 一句话总结

TrajTok 把原始 GPS 轨迹转换成多分辨率六边形 Token 序列，通过几何+运动学联合掩码预训练，让一个冻结的编码器加轻量适配器就能在轨迹相似度、分类、ETA 等多任务上击败各自的专用模型。

## 为什么这个问题重要？

GPS 轨迹数据无处不在——出租车调度、物流路径规划、城市交通分析。但让模型真正"理解"轨迹，面临三个根本困难：

- **连续性**：经纬度是实数，没有天然的离散词表
- **噪声**：GPS 信号遮挡导致位置误差可达数十米
- **不规则采样**：同一条轨迹，有的每秒一点，有的每分钟一点

**空间 Tokenization 的两难困境：**
- 细粒度格网（0.001° 方形格子）→ 每格点数稀少，embedding 学不好
- 粗粒度格网 → 把地铁站和附近商场停车场合并成同一个 Token，语义混乱

现有方案各自为政——ETA 模型、轨迹聚类模型、相似度检索模型分别训练，无法共享表示。TrajTok 的目标是：**一个编码器，多任务通用**。

## 背景知识

### 空间索引：H3 六边形网格

TrajTok 使用 Uber H3 六边形分层索引。为什么是六边形而不是方形？

- 六边形每个顶点到中心**等距**（方形有对角线偏差约 41%）
- 相邻格子形状更均匀，适合建模连续运动
- H3 支持 Level 0–15 多分辨率，父子关系明确

```
H3 分辨率参考（城市级应用）：
Level 7  → 约 5 km²  每格（城市街区）
Level 8  → 约 1 km²  每格（社区级）
Level 9  → 约 0.1 km² 每格（单栋建筑粒度）
```

### 轨迹的双模态结构

轨迹包含两类需要区别对待的信息：
- **几何（Geometry）**：经过了哪些地方——空间坐标序列
- **运动学（Kinematics）**：怎么走的——速度、加速度、转向角

单一编码这两类信息会相互干扰：运动学剧烈变化（急刹车）不代表几何位置突变。

## 核心方法

### 直觉解释

```
原始 GPS [(lat, lng, t), ...]
        ↓
多分辨率六边形 Tokenization（从数据分布中学习最优分区）
   粗 Level 7: [A, A, B, C]   ← 街区级
   细 Level 9: [a1, a2, b1, c1] ← 建筑级
        ↓
双流独立自注意力
   几何流: Transformer 处理六边形 Token 序列
   运动学流: Transformer 处理 (speed, heading) 序列
        ↓
ST-RoPE 位置编码（同时编码空间坐标 + 时间戳）
        ↓
Cross-Attention 融合两流
        ↓
掩码预训练：随机遮住 Token → 重建六边形 ID + 运动学值
        ↓
下游任务：冻结编码器 + 2 层 MLP 适配器
```

### 多分辨率六边形 Tokenization

TrajTok 不是固定选一个分辨率，而是**从训练数据的空间分布中学习最优的分层分区**——GPS 点密集的市区用细粒度格子，稀疏的郊区用粗粒度格子。

$$
\text{Token}(p_i) = \text{argmin}_{c \in \mathcal{C}_r} \|p_i - \text{center}(c)\|_2
$$

其中 $\mathcal{C}_r$ 是分辨率 $r$ 下的六边形格子集合。词表大小由出现过的格子数量决定，没出现的格子统一映射为 `[UNK]`。

### ST-RoPE：时空旋转位置编码

标准 RoPE 只编码序列位置（第几个 Token）。ST-RoPE 同时编码**空间坐标**和**时间戳**：

$$
\mathbf{q}_{st} = R_s(\theta_{\text{lat}}, \theta_{\text{lng}}) \cdot R_t(\theta_{\tau}) \cdot \mathbf{q}
$$

这样，即使两个 Token 对应同一个六边形格子，如果时间戳不同，它们在注意力计算中也会被区分。这对捕捉"早高峰的东三环"和"深夜的东三环"的差异至关重要。

### 掩码预训练目标

随机遮住 15–40% 的 Token，同时重建几何结构和运动学特征：

$$
\mathcal{L} = \lambda_g \cdot \mathcal{L}_{\text{CE}}(\hat{h}, h) + \lambda_k \cdot \mathcal{L}_{\text{MSE}}(\hat{v}, v)
$$

其中 $h$ 是六边形 ID（分类任务），$v$ 是速度/方向角（回归任务）。论文中 $\lambda_g = 0.7, \lambda_k = 0.3$。

## 实现

### 多分辨率六边形 Tokenizer

```python
import h3
import numpy as np
from collections import defaultdict

class MultiResHexTokenizer:
    """多分辨率六边形 GPS Tokenizer"""
    
    PAD, MASK, UNK = 0, 1, 2
    
    def __init__(self, resolutions=(7, 8, 9)):
        self.resolutions = resolutions
        self.cell_to_id = [defaultdict(lambda: self.UNK) for _ in resolutions]
        self.vocab_sizes = [3] * len(resolutions)  # PAD/MASK/UNK 占 0/1/2
    
    def fit(self, trajectories):
        """从训练轨迹中构建每分辨率词表"""
        for traj in trajectories:
            for lat, lng, _ in traj:
                for i, res in enumerate(self.resolutions):
                    cell = h3.latlng_to_cell(lat, lng, res)
                    if cell not in self.cell_to_id[i]:
                        new_id = len(self.cell_to_id[i]) + 3  # 从 3 开始，留出特殊 token
                        self.cell_to_id[i][cell] = new_id
        self.vocab_sizes = [len(d) + 3 for d in self.cell_to_id]
    
    def encode(self, trajectory):
        """轨迹 → (多分辨率 tokens, 运动学特征)"""
        lats = [p[0] for p in trajectory]
        lngs = [p[1] for p in trajectory]
        times = [p[2] for p in trajectory]
        
        # 各分辨率 token
        all_tokens = []
        for i, res in enumerate(self.resolutions):
            tokens = []
            for lat, lng in zip(lats, lngs):
                cell = h3.latlng_to_cell(lat, lng, res)
                tokens.append(self.cell_to_id[i][cell])
            all_tokens.append(tokens)
        
        # 运动学特征：速度 (m/s) + 方向角 (rad)
        kinematics = self._compute_kinematics(lats, lngs, times)
        return all_tokens, kinematics
    
    def _compute_kinematics(self, lats, lngs, times):
        """计算逐步速度和方向角"""
        features = [(0.0, 0.0)]  # 第一个点无前驱，填 0
        for i in range(1, len(lats)):
            dt = np.clip(times[i] - times[i-1], 0.5, 300)  # 限制异常 dt
            dlat = lats[i] - lats[i-1]
            dlng = lngs[i] - lngs[i-1]
            dist = np.sqrt(dlat**2 + dlng**2) * 111320   # 度 → 近似米
            speed = dist / dt
            heading = float(np.arctan2(dlng, dlat))
            features.append((speed, heading))
        return features
```

### ST-RoPE 实现

```python
import torch
import torch.nn as nn

class STRoPE(nn.Module):
    """时空旋转位置编码：同时编码空间位置和时间戳"""
    
    def __init__(self, d_model):
        super().__init__()
        # 各占 1/4 维度，剩余 1/2 不施加位置编码
        self.d_half = d_model // 4
    
    def _rope(self, x, positions):
        """对 x 的前 d_half*2 维施加旋转"""
        d = self.d_half
        theta = 1.0 / (10000 ** (torch.arange(0, d, device=x.device).float() / d))
        angles = positions.unsqueeze(-1) * theta          # [B, L, d]
        cos, sin = angles.cos(), angles.sin()
        x1, x2 = x[..., :d], x[..., d:d*2]
        rotated = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        return torch.cat([rotated, x[..., d*2:]], dim=-1)
    
    def forward(self, q, k, spatial_pos, temporal_pos):
        """
        q, k: [B, L, D]
        spatial_pos: [B, L]  用 lat*cos(lng) 编码为标量
        temporal_pos: [B, L] 归一化时间戳 [0, 1]
        """
        # 先施加空间旋转，再施加时间旋转
        q = self._rope(q, spatial_pos)
        q = self._rope(q, temporal_pos)
        k = self._rope(k, spatial_pos)
        k = self._rope(k, temporal_pos)
        return q, k
```

### 分解式 Transformer 编码器

```python
class FactorizedTrajEncoder(nn.Module):
    """几何流 + 运动学流 → Cross-Attention 融合"""
    
    def __init__(self, geo_vocab_size, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        enc_layer = lambda: nn.TransformerEncoderLayer(
            d_model, nhead, d_model*4, dropout=0.1, batch_first=True)
        
        # 几何流
        self.geo_embed = nn.Embedding(geo_vocab_size, d_model, padding_idx=0)
        self.geo_encoder = nn.TransformerEncoder(enc_layer(), num_layers)
        
        # 运动学流
        self.kin_proj = nn.Linear(2, d_model)
        self.kin_encoder = nn.TransformerEncoder(enc_layer(), num_layers)
        
        # Cross-Attention 融合（几何作 query，运动学作 key/value）
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.st_rope = STRoPE(d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, geo_tokens, kin_feats, spatial_pos, temporal_pos):
        """
        geo_tokens: [B, L]     六边形 token ID（最细分辨率）
        kin_feats:  [B, L, 2]  (speed, heading)
        spatial_pos, temporal_pos: [B, L]
        """
        g = self.geo_encoder(self.geo_embed(geo_tokens))    # [B, L, D]
        k = self.kin_encoder(self.kin_proj(kin_feats))      # [B, L, D]
        
        # ST-RoPE 后做跨模态注意力
        q_r, k_r = self.st_rope(g, k, spatial_pos, temporal_pos)
        fused, _ = self.cross_attn(q_r, k_r, k)
        return self.norm(g + fused)                         # 残差连接
```

### 掩码预训练

```python
class TrajTokPretrainer(nn.Module):
    """双目标掩码预训练：重建六边形 ID + 运动学值"""
    
    def __init__(self, encoder, geo_vocab_size, mask_ratio=0.25):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d = encoder.geo_embed.embedding_dim
        self.geo_head = nn.Linear(d, geo_vocab_size)   # 分类头
        self.kin_head = nn.Linear(d, 2)                # 回归头
    
    def forward(self, geo_tokens, kin_feats, spatial_pos, temporal_pos):
        B, L = geo_tokens.shape
        mask = torch.rand(B, L, device=geo_tokens.device) < self.mask_ratio
        
        # 用 MASK token 替换被遮住的位置
        masked_geo = geo_tokens.masked_fill(mask, 1)   # 1 = [MASK]
        masked_kin = kin_feats.clone()
        masked_kin[mask] = 0.0
        
        h = self.encoder(masked_geo, masked_kin, spatial_pos, temporal_pos)
        
        # 只在被掩码的位置计算损失
        loss_geo = nn.CrossEntropyLoss()(
            self.geo_head(h[mask]), geo_tokens[mask])
        loss_kin = nn.MSELoss()(
            self.kin_head(h[mask]), kin_feats[mask])
        
        return 0.7 * loss_geo + 0.3 * loss_kin
```

### 下游任务适配（ETA 预测示例）

```python
class ETAAdapter(nn.Module):
    """冻结编码器 + 轻量 MLP 适配器"""
    
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False              # 冻结预训练权重
        
        d = pretrained_encoder.geo_embed.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, geo_tokens, kin_feats, spatial_pos, temporal_pos):
        with torch.no_grad():
            h = self.encoder(geo_tokens, kin_feats, spatial_pos, temporal_pos)
        return self.head(h.mean(dim=1)).squeeze(-1)  # 全局池化后预测 ETA
```

### 轨迹可视化

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_trajectory_tokens(trajectory, tokenizer, resolution=9):
    """可视化轨迹与六边形 Token 覆盖范围"""
    lats = [p[0] for p in trajectory]
    lngs = [p[1] for p in trajectory]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：原始轨迹（颜色表示时间进度）
    colors = cm.plasma(np.linspace(0, 1, len(lats)))
    for i in range(len(lats)-1):
        ax1.plot([lngs[i], lngs[i+1]], [lats[i], lats[i+1]], c=colors[i], lw=2)
    ax1.set_title("原始 GPS 轨迹（颜色=时间）")
    
    # 右图：六边形 Token 覆盖（不同格子用不同颜色标注）
    seen_cells = {}
    for lat, lng in zip(lats, lngs):
        cell = h3.latlng_to_cell(lat, lng, resolution)
        if cell not in seen_cells:
            seen_cells[cell] = len(seen_cells)
            boundary = h3.cell_to_boundary(cell)
            poly_lngs = [b[1] for b in boundary] + [boundary[0][1]]
            poly_lats = [b[0] for b in boundary] + [boundary[0][0]]
            color = cm.Set2(seen_cells[cell] % 8)
            ax2.fill(poly_lngs, poly_lats, alpha=0.3, color=color)
            ax2.plot(poly_lngs, poly_lats, color=color, lw=1)
    ax2.plot(lngs, lats, 'k-', lw=1, alpha=0.5)
    ax2.set_title(f"六边形 Tokenization（Level {resolution}，{len(seen_cells)} 格）")
    
    plt.tight_layout()
    plt.show()
    # ... (完整可视化代码省略)
```

## 实验

### 数据集说明

**Porto Taxi Dataset**（ECML/PKDD 2015 Challenge）：
- 规模：约 170 万条出租车轨迹，时间跨度 2013 年全年
- 格式：每 15 秒一个 GPS 采样点，平均约 60 点/条
- 获取：Kaggle 竞赛数据集（公开可下载）
- 特点：单城市、固定车辆类型，数据质量较好但场景相对单一

### 定量评估

论文报告的 Porto 数据集结果（冻结编码器 + 轻量适配器）：

| 方法 | 轨迹相似度 HR@10 | 分类 Acc | ETA MAE (min) | 旅行时间 RMSE (min) |
|------|----------------|---------|--------------|-------------------|
| BERT4Traj | 0.82 | 0.87 | 5.1 | 8.2 |
| TrajCL | 0.85 | 0.88 | 4.8 | 7.5 |
| Toast | 0.83 | 0.86 | 5.3 | 8.9 |
| **TrajTok** | **0.89** | **0.91** | **4.2** | **6.8** |

> 数值为论文近似值，以原论文为准。

### 关键实验发现

- **几何主导任务**（相似度搜索）与**运动学主导任务**（ETA 预测）共用一个编码器均有提升，说明学到的是通用结构而非任务捷径
- 多分辨率六边形比单一分辨率方形格子的 HR@10 高约 5%
- 掩码预训练对运动学任务的收益（ETA MAE 降低 17%）大于几何任务（HR@10 提升 8%）

## 工程实践

### 计算开销

- **预训练**：Porto 全量数据，RTX 3090，约 12 小时
- **推理**：单条 60 点轨迹编码 < 5ms，支持批量并行
- **模型大小**：`d_model=256, 6 layers` 约 45MB

### 常见坑

**坑 1：训练集外的 GPS 点（OOV 格子）**
```python
# 坑：测试集轨迹进入训练集没覆盖的区域（新建区、数据稀疏区）
cell = h3.latlng_to_cell(lat, lng, resolution=9)
token_id = cell_to_id.get(cell, UNK_ID)  # 直接用 UNK 效果差

# 修复：使用父格子（更粗粒度）做退化匹配
if cell not in cell_to_id:
    parent = h3.cell_to_parent(cell, resolution - 1)  # 退一级
    token_id = cell_to_id.get(parent, UNK_ID)
```

**坑 2：不规则采样导致运动学特征爆炸**
```python
# 坑：停车等待后重新出发，两点间时间差很大，速度接近 0
# 但网络会把这个"0速度"和"静止停放"的轨迹混淆
dt = times[i] - times[i-1]
speed = dist / dt  # dt=300s, dist=0 → speed=0，和停车场景一样

# 修复：增加"停顿标记"特征维度
is_stopped = float(dt > 60 and dist < 10)  # 额外的第 3 维特征
```

**坑 3：跨城市词表不通用**
```python
# H3 格子 ID 是全球唯一的，但训练的 cell_to_id 映射是城市特定的
# 直接把上海的模型拿去跑北京数据 → 几乎全是 OOV
# 解决方案：重新对北京数据 fit tokenizer，在上海预训练权重基础上微调
```

### 数据采集建议

- 采样率 **10–30 秒**最佳；太密（<1s）运动学特征冗余，太稀（>5min）几何结构丢失
- GPS 清洗：速度阈值过滤（>200 km/h 视为异常点），Kalman 滤波平滑轨迹
- 轨迹分段：停留时间超过 5 分钟（如在停车场）建议切分为两段轨迹

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 城市级 GPS 轨迹（出租车、外卖、物流） | 室内定位轨迹（WiFi/UWB 精度尺度完全不同） |
| 多任务共用一个编码器、减少标注成本 | 极低延迟推理（<1ms），需要更轻量的模型 |
| 训练数据有限，需要迁移预训练特征 | 需要路网感知（TrajTok 不感知道路拓扑） |
| 轨迹长度 20–500 点 | 极长轨迹（>1000 点，Transformer 二次复杂度）|
| 历史数据丰富的固定城市 | 新城市冷启动（词表需重建，无预训练收益） |

## 与其他方法对比

| 方法 | 空间 Tokenization | 预训练目标 | 优点 | 缺点 |
|------|------------------|-----------|------|------|
| BERT4Traj | 固定分辨率方形格 | 掩码 Token | 简单，易复现 | 单分辨率，边界效应，格子形状各向异性 |
| TrajCL | 无（连续坐标） | 对比学习 | 无需词表 | 对噪声敏感，连续输入难以统一预训练 |
| Toast | 路网节点 ID | 下一步预测 | 语义丰富，有路网拓扑 | 依赖精确地图匹配，失配率约 5–15% |
| **TrajTok** | **多分辨率六边形** | **掩码几何+运动学** | 多任务通用，无需路网 | 跨城市需重建词表，无路网感知 |

## 我的观点

TrajTok 的贡献在于把 NLP 的 Tokenization 思想认真地迁移到 GPS 轨迹上，执行干净，实验设计有说服力。

**有价值的设计：**
多分辨率六边形 Tokenization 对工程实践真正有用。城市核心区和郊区的轨迹密度可以差两个数量级，固定分辨率处理不了这种分布偏差。六边形的各向同性特性对于建模"往哪走"也比方形格子更自然。

**需要诚实说的局限：**
- 论文只在 Porto 一个数据集上验证，这个数据集已被研究十年，结果的参考价值有限。跨城市、跨交通模式（步行、骑行、驾车混合）的泛化性未知
- 六边形 Token 不感知路网拓扑——同样两点间，走高速和走小路的语义完全不同，但 Token 序列可能高度重叠
- 城市特定词表是真实部署的障碍：上海的预训练模型不能直接用于北京

**技术发展趋势：**
轨迹基础模型（Trajectory Foundation Model）是个值得押注的方向。类比 NLP 的 Word2Vec→BERT→GPT，轨迹表示学习目前还在早期阶段。TrajTok 尝试迈向 BERT 风格的通用表示，下一步可能是：
1. 引入地图/POI 知识作为辅助监督信号，增强 Token 的语义
2. 跨城市统一预训练（需要解决词表不统一的问题，可能需要基于地理坐标而非 cell ID 的表示）
3. 轨迹生成（类 GPT 的自回归解码），用于数据增强和隐私保护合成数据

离产品化落地还差什么？主要是缺乏真实业务系统的 A/B 测试验证——现有评测全是离线的，没有证据说明预训练表示在实际派单系统或路线优化中有业务价值。但这个方向的技术路线是清晰的，值得跟进。