---
layout: post-wide
title: 'PhyCo：让视频生成模型真正"懂"物理——可控物理先验的生成运动'
date: 2026-05-03 12:07:16 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.28169v1
generated_by: Claude Code CLI
---

## 一句话总结

PhyCo 将摩擦系数、弹性恢复系数等物理属性编码为**像素对齐的物理属性图**，通过 ControlNet 机制注入预训练视频扩散模型，再借助 VLM 奖励信号进一步对齐，使生成视频在碰撞、弹跳、形变等场景中真正遵守物理规律——推理时无需物理引擎。

---

## 为什么这个问题重要？

视频扩散模型在外观合成上已经相当出色：光影、纹理、运动流畅性都接近真实。但一个篮球碰到地面，弹跳高度对吗？一块橡皮泥被按压，形变方式符合材料特性吗？**物理一致性**是当前视频生成的盲区。

这个问题在以下场景中尤为突出：

- **机器人学习**：用合成视频训练操作策略，接触动力学必须可信
- **影视特效**：艺术家希望精确控制材质物理属性，而不只是"看起来像"
- **自动驾驶仿真**：测试场景中的碰撞响应必须物理正确
- **游戏内容生成**：视觉合成和物理行为需要一致

现有方法的核心矛盾：纯数据驱动的视频模型根本没有"物理"概念，只是在统计上拟合常见运动；用物理引擎渲染虽然准确，但视觉质量差，sim-to-real gap 巨大。

PhyCo 的思路是：**用大规模高质量物理仿真数据训练扩散模型，使模型内化物理先验**，而非在推理时调用仿真器。

---

## 背景知识

### 视频扩散模型的条件控制

视频扩散模型将视频生成建模为去噪过程，模型学习预测噪声：

$$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]
$$

其中 $c$ 是条件信号（文本、图像、深度图等）。**ControlNet** 通过在编码器旁侧接入可训练副本来注入额外条件，关键设计是**零初始化卷积**——训练初期不干扰原始模型。

### 物理属性的量化表示

PhyCo 关注四类物理属性：

| 属性 | 符号 | 物理意义 | 典型范围 |
|------|------|---------|---------|
| 摩擦系数 | $\mu$ | 抵抗相对滑动的力 | 冰面 0.02 → 橡胶 1.5 |
| 恢复系数 | $e$ | 碰撞能量保留比 | 黏土 0 → 弹力球 0.95 |
| 形变刚度 | $k$ | 抵抗变形的能力 | 橡皮泥 0.1 → 金属 1.0 |
| 外力 | $\mathbf{F}$ | 施加在物体上的力向量 | — |

PhyCo 将上述属性表示为**与视频帧像素对齐的属性图**：每个像素都有自己的物理参数，不同区域可以有不同材质。

---

## 核心方法

### 直觉解释

想象你在编辑一张图片，不是修改颜色，而是修改每个像素的"物理性质"——这块区域很滑（低摩擦），那块很弹（高恢复系数）。把这张**物理属性地图**作为额外输入喂给视频生成模型，模型就知道：当物体运动到这个区域时，应该如何响应。

### 三大核心组件

**① 大规模物理仿真数据集**

100K+ 个光真实感仿真视频，系统地变化物理参数，涵盖刚体碰撞、软体形变、布料运动、流体，每个视频附带对应的逐帧逐像素物理属性图。

**② Physics-supervised ControlNet 微调**

$$
\mathcal{L}_{\text{phys}} = \mathbb{E}_{x_0, t, \epsilon, \mathbf{P}} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c_{\text{text}}, \mathbf{P}) \|^2 \right]
$$

其中 $\mathbf{P} \in \mathbb{R}^{T \times H \times W \times 5}$ 是时序物理属性图，通过 ControlNet 分支注入扩散模型。

**③ VLM 奖励优化**

微调后的视觉语言模型作为"物理评判员"，接收生成视频和物理查询，返回一致性得分作为奖励：

$$
R = \text{VLM}(v_{\text{gen}} \mid q_{\text{physics}})
$$

这个奖励信号通过 REINFORCE 反向传播到扩散模型，实现端到端的物理对齐。

### Pipeline 概览

```
输入: 文本描述 + 物理属性图 P [T×H×W×5]
         ↓
  预训练视频扩散 UNet（冻结）
  + PhysicsControlNet（可训练，ControlNet分支）
         ↓
   DDIM 去噪采样（~25步）
         ↓
   VLM 物理一致性评分 → REINFORCE 梯度
         ↓
  输出: 物理一致的生成视频
```

---

## 实现

### 物理属性图的定义与操作

```python
import torch
import matplotlib.pyplot as plt

class PhysicsPropertyMap:
    """
    像素对齐的物理属性图
    5通道: [摩擦系数, 恢复系数, 刚度, 力x分量, 力y分量]
    """
    def __init__(self, H: int, W: int):
        self.map = torch.zeros(5, H, W)
        self.map[0] = 0.5   # 默认摩擦系数
        self.map[1] = 0.5   # 默认恢复系数
        self.map[2] = 1.0   # 默认刚度

    def set_region(self, mask: torch.Tensor,
                   friction=None, restitution=None, stiffness=None):
        """对选定区域设置物理属性"""
        if friction is not None:
            self.map[0][mask] = friction
        if restitution is not None:
            self.map[1][mask] = restitution
        if stiffness is not None:
            self.map[2][mask] = stiffness

    def visualize(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (ax, title, cmap) in enumerate(zip(
            axes, ['摩擦系数 μ', '恢复系数 e', '刚度 k'],
            ['Blues', 'Reds', 'Greens']
        )):
            im = ax.imshow(self.map[i].numpy(), cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title); plt.colorbar(im, ax=ax)
        plt.tight_layout(); return fig

# 示例：左半冰面（低摩擦高弹性），右半橡皮泥（高摩擦低弹性）
H, W = 256, 256
pmap = PhysicsPropertyMap(H, W)

ice = torch.zeros(H, W, dtype=torch.bool); ice[:, :W//2] = True
pmap.set_region(ice, friction=0.02, restitution=0.9)

clay = ~ice
pmap.set_region(clay, friction=1.5, restitution=0.05, stiffness=0.2)

fig = pmap.visualize()  # 三张热力图：左半低摩擦/高弹性，右半反之
```

### PhysicsControlNet 核心实现

```python
import torch.nn as nn

class PhysicsEncoder(nn.Module):
    """将物理属性图编码为特征，作为 ControlNet 条件输入"""
    def __init__(self, physics_channels=5, base_dim=320):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(physics_channels, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.SiLU(),  # 1/2
            nn.Conv2d(32, 96, 3, stride=2, padding=1),  nn.SiLU(),  # 1/4
            nn.Conv2d(96, base_dim, 3, stride=2, padding=1),        # 1/8
        )
        # 零初始化：训练初期对原模型零影响，保持生成质量
        self.zero_conv = nn.Conv2d(base_dim, base_dim, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, physics_map: torch.Tensor) -> torch.Tensor:
        """physics_map: [B,5,H,W] → 条件特征 [B,C,H/8,W/8]"""
        return self.zero_conv(self.input_proj(physics_map))


class PhysicsConditionedDenoiser(nn.Module):
    """在冻结的视频扩散 UNet 上叠加物理条件（简化示意）"""
    def __init__(self, frozen_unet, physics_encoder):
        super().__init__()
        self.unet = frozen_unet
        self.physics_enc = physics_encoder
        for p in self.unet.parameters():  # 冻结预训练权重
            p.requires_grad_(False)

    def forward(self, x_t, t, text_emb, physics_map):
        phys_feat = self.physics_enc(physics_map)
        # 注入到 UNet encoder 各分辨率特征（实际需 hook 各层残差）
        return self.unet(x_t, t, text_emb, additional_residuals=phys_feat)
```

### VLM 物理奖励评估

```python
import torch.nn.functional as F

class PhysicsRewardModel:
    """用 VLM 评估生成视频的物理一致性"""
    QUERIES = {
        'friction':     "Rate 0-1: does the sliding motion match μ={v:.2f}?",
        'restitution':  "Rate 0-1: does the bounce height match e={v:.2f}?",
        'stiffness':    "Rate 0-1: does the deformation match stiffness k={v:.2f}?",
    }

    def __init__(self, vlm):
        self.vlm = vlm

    @torch.no_grad()
    def score(self, video_frames: torch.Tensor, physics_params: dict) -> torch.Tensor:
        """
        video_frames: [T, H, W, 3]
        physics_params: {'friction': 0.3, 'restitution': 0.8, ...}
        → scalar reward in [0, 1]
        """
        key_frames = video_frames[::max(1, len(video_frames)//4)]  # 均匀采样4帧
        scores = []
        for attr, val in physics_params.items():
            if attr not in self.QUERIES:
                continue
            query = self.QUERIES[attr].format(v=val)
            score = self.vlm.score_frames(key_frames, query)  # VLM推理
            scores.append(score)
        return torch.stack(scores).mean()
```

### REINFORCE 奖励优化训练步

```python
def reward_optimization_step(model, reward_model, batch, optimizer):
    """VLM 奖励驱动的强化微调（简化 REINFORCE）"""
    text_emb      = batch['text_emb']
    physics_map   = batch['physics_map']    # [B, 5, H, W]
    physics_params = batch['physics_params'] # 目标物理参数 dict

    # 1. 采样生成视频（保留计算图）
    generated = model.sample(text_emb, physics_map)  # [B, T, H, W, 3]

    # 2. VLM 评估物理奖励
    rewards = torch.stack([
        reward_model.score(generated[i], physics_params[i])
        for i in range(len(generated))
    ])  # [B]

    # 3. REINFORCE：高奖励样本增加其生成概率
    log_probs = model.log_prob(generated, text_emb, physics_map)
    baseline  = rewards.mean().detach()          # 方差缩减 baseline
    loss      = -((rewards - baseline) * log_probs).mean()

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'loss': loss.item(), 'mean_reward': rewards.mean().item()}
```

---

## 实验

### Physics-IQ Benchmark

Physics-IQ 专为评估视频生成的物理合理性设计，涵盖刚体碰撞、软体形变、摩擦动力学、流体行为四大类场景，每类都有对照组（已知物理参数的地面真值视频）。

### 定量评估

| 方法 | Physics-IQ ↑ | 物理控制精度 ↑ | FID ↓ | 推理速度 |
|------|-------------|--------------|-------|---------|
| SVD (基线) | 31.2 | — | 18.4 | ~8 fps |
| CogVideoX (基线) | 34.7 | — | 16.1 | ~6 fps |
| PhyCo (ControlNet only) | 48.3 | 0.61 | 17.2 | ~5 fps |
| **PhyCo (+ VLM reward)** | **56.8** | **0.79** | 16.8 | ~5 fps |

VLM 奖励优化带来 +8.5 点物理一致性提升，代价是 FID 轻微上升（视觉质量略有下降）。

### 失败案例

- **多物体密集碰撞**：3+ 个物体同时接触时，属性图无法精确表达接触面
- **极端参数值**：$e > 0.95$ 时模型倾向生成"普通弹跳"而非超弹性响应
- **快速运动**：高速物体的接触时间极短，视频扩散模型时序分辨率不足

---

## 工程实践

### 推理时物理属性图的生成

训练阶段物理属性图来自仿真器；推理时通常需要手动指定。一种实用方案是**从语义分割结果自动赋予物理属性**：

```python
MATERIAL_LIBRARY = {
    'wood_floor':  {'friction': 0.3,  'restitution': 0.7,  'stiffness': 0.9},
    'marble':      {'friction': 0.02, 'restitution': 0.85, 'stiffness': 1.0},
    'rubber_mat':  {'friction': 1.2,  'restitution': 0.1,  'stiffness': 0.6},
}

def build_physics_map_from_seg(seg_mask, material_library):
    """seg_mask: [H,W] 语义标签 → PhysicsPropertyMap"""
    H, W = seg_mask.shape
    pmap = PhysicsPropertyMap(H, W)
    for label, params in material_library.items():
        mask = (seg_mask == label)
        pmap.set_region(mask, **params)
    return pmap
```

### 常见坑

**坑1：分辨率不对齐**

ControlNet 在 1/8 分辨率特征空间操作，属性图必须提前下采样：

```python
# 错误：直接传 512×512 → ControlNet 特征 64×64，尺寸不匹配
# 正确：
phys_map_64 = F.interpolate(phys_map_512, size=(64, 64), mode='bilinear')
```

**坑2：VLM 奖励信号方差大导致训练震荡**

```python
# 奖励标准化，防止梯度爆炸
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

**坑3：仿真数据和真实视频的域差**

仿真视频颜色分布与真实视频差异明显。数据增强阶段须加入颜色抖动、随机光照变化，以及真实视频的混合微调（比例约 1:5）来缩小 sim-to-real gap。

### 硬件需求

| 阶段 | 显存 | 推荐配置 |
|------|------|---------|
| ControlNet 微调 | 40-80 GB | 4× A100 |
| VLM 奖励优化 | 80+ GB（VLM+扩散模型同时加载） | 8× A100 |
| 推理 | 24 GB | 1× A6000 或 4090 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 精确控制材质物理属性 | 多物体密集接触（堆叠、颗粒流） |
| 静态背景 + 单/双物体运动 | 场景中有大量随机动态（火焰、烟雾） |
| 机器人操作的合成训练数据 | 需要实时生成（当前 ~5 fps） |
| 影视特效的物理可控合成 | 工程级精度仿真（误差要求 < 1%） |
| 物理教育演示视频生成 | 输入场景缺乏准确语义分割 |

---

## 与其他方法对比

| 方法 | 物理控制粒度 | 视觉质量 | 推理速度 | 推理依赖 |
|------|------------|---------|---------|---------|
| Isaac Sim 渲染 | 精确（参数化） | 差（仿真感） | 中 | 物理引擎 |
| NeRF + 物理代理 | 无控制接口 | 好 | 慢 | NeRF 重建 |
| ControlVideo | 轨迹/姿态控制 | 好 | 中 | 无 |
| **PhyCo** | 像素级物理属性 | 好 | 中 | 无 |

PhyCo 的独特价值：**实现了"物理属性 → 物理行为"的直接控制，且推理时不依赖物理引擎**。代价是训练成本极高，VLM 奖励优化阶段是主要瓶颈。

---

## 我的观点

PhyCo 提出了一个清晰且重要的问题：**视频生成模型能否内化物理先验？** 答案是可以，但仍有明显局限。

**真正有价值的技术贡献**：像素对齐的物理属性图 + ControlNet 这套设计是扎实的。物理属性从"全局标量"升级为"逐像素张量"，这才允许一个场景中有多种材质共存，是真正实用的接口设计。

**值得警惕的地方**：VLM 作为物理评判员存在根本性问题——VLM 的"物理理解"本质上也是从视频数据中习得的统计模式，而非真正的物理知识。它判断"这个弹跳看起来像高弹性材质"和"这个弹跳真正满足动量守恒"是两回事。论文中缺乏对 VLM 评判准确性的深入验证。

**离实际应用还有多远？**

- 机器人合成训练数据：**1-2 年内可用**，对精度要求宽松
- 影视特效辅助工具：**2-3 年**，需要更好的交互式属性图编辑界面
- 工程级物理仿真替代：**不适用**，扩散模型是统计近似，不是精确求解器

**开放问题**：
1. 如何将这套框架迁移到 3D 内容生成（如 3D Gaussian Splatting），实现物理可控的 3D 动态场景？
2. 物理属性图的自动生成（从单张图像估计材质物理参数）是解锁实用化的关键
3. 能否设计更严格的物理一致性评估指标，取代依赖 VLM 的主观评分？

这个方向会继续演进。视频生成和物理仿真的融合是空间智能领域最有潜力的交叉方向之一。

论文链接：[PhyCo: Learning Controllable Physical Priors for Generative Motion](https://arxiv.org/abs/2604.28169)