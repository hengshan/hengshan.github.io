---
layout: post-wide
title: "单张贴纸图像个性化生成：SEAL 如何同时克服背景污染与结构僵化"
date: 2026-04-30 08:04:27 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.26883v1
generated_by: Claude Code CLI
---

## 一句话总结

给一张参考图，SEAL 在测试时微调（TTF）过程中通过三个协同约束，同时解决"背景污染概念"和"生成姿态固化"两个核心难题，无需修改 U-Net 骨干网络。

## 为什么这个问题重要？

贴纸个性化生成的需求非常具体：给我一张猫咪图片，生成"这只猫做出各种表情、动作"的贴纸系列。

**单图 TTF 方法会产生两类致命问题：**

**视觉纠缠（Visual Entanglement）**：模型在微调时把背景一起记住了。参考图里猫在绿色草坪上，微调后生成的猫在任何背景下都带着草坪的绿色光晕。

**结构僵化（Structural Rigidity）**：模型记住了参考图的具体空间布局。参考图里猫是"正面坐姿"，之后不管 prompt 写什么（"猫在奔跑"），生成结果都还是原来那个坐姿。

SEAL 的核心创新：**在 embedding 微调阶段加入三个协同约束，即插即用，无需修改 U-Net 骨干**。

## 背景知识

### 扩散模型个性化的技术路线

| 方法 | 代表工作 | 微调对象 | 图片需求 | 主要问题 |
|------|---------|---------|---------|---------|
| Textual Inversion | TI | 文本 token embedding | 3-10 张 | 身份保持弱 |
| DreamBooth | DB | 整个 U-Net | 3-10 张 | 算力需求高 |
| IP-Adapter | IPA | 图像 encoder | 0（即插即用）| 可控性差 |
| **单图 TTF** | SEAL 目标场景 | embedding | **1 张** | 过拟合问题 |

### U-Net 中的两类注意力

扩散模型 U-Net 包含两种注意力机制，理解这一点是理解 SEAL 的关键：

- **自注意力（Self-Attention）**：图像 token 之间的关系，控制**空间布局和结构**
- **交叉注意力（Cross-Attention）**：图像 token 与文本 token 之间的关系，控制**语义与视觉的对应**

视觉纠缠源于交叉注意力失控；结构僵化源于自注意力被过度修改。SEAL 的三个组件分别针对这两种失效模式。

## 核心方法

### 直觉解释

```
参考图（单张）
    │
    ├── 语义分割 ──→ 前景 mask M
    │
    ▼
TTF 微调过程（SEAL 的三个约束）
    │
    ├── [1] 语义引导注意力损失：用 M 监督交叉注意力图
    │       让概念 token 只关注前景，不关注背景
    │
    ├── [2] 分裂-合并 Token 策略：v* → [v1, v2, v3]
    │       分裂后独立优化，合并后防止单 token 过拟合
    │
    └── [3] 结构感知层限制：只微调交叉注意力层
            冻结自注意力，保留姿态/布局的可控性
    │
    ▼
微调完成 → 用 prompt 生成多样化贴纸
```

### 数学细节

**组件一：语义引导空间注意力损失**

设 $\mathbf{A} \in \mathbb{R}^{B \times H \times W \times N}$ 为某层交叉注意力图，$\mathbf{M} \in \{0,1\}^{B \times H \times W}$ 为前景 mask（前景=1）。

$$\mathcal{L}_{attn} = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \left\| \mathbf{A}_i \odot (1 - \mathbf{M}) \right\|_F^2$$

$\mathcal{S}$ 是概念相关 token 的索引集，$(1-\mathbf{M})$ 是背景区域。这个损失**直接惩罚概念 token 关注背景**。

**组件二：分裂-合并 Token 策略**

单个概念 token $\mathbf{v}^*$ 同时承载外观和结构，容易过拟合。将其分裂为 $K$ 个子 token，各自独立优化后合并：

$$\mathbf{v}^* = \text{Merge}(\mathbf{v}_1, \ldots, \mathbf{v}_K) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{v}_k$$

**组件三：结构感知层限制**

$$\mathcal{L}_{tune} = \left\{ l \;\middle|\; l \text{ 是交叉注意力层} \right\}$$

自注意力层（控制空间布局）全部冻结，确保 prompt 中的"动作/姿态"描述仍然有效。

**总损失**：

$$\mathcal{L}_{total} = \mathcal{L}_{LDM} + \lambda_{attn} \cdot \mathcal{L}_{attn}$$

## 实现

### 分裂-合并 Token 策略

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitMergeTokens(nn.Module):
    def __init__(self, embed_dim: int = 768, num_splits: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_splits = num_splits
        # 每个子 token 独立优化，从小随机值初始化
        self.sub_tokens = nn.Parameter(
            torch.randn(num_splits, embed_dim) * 0.01
        )
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self) -> torch.Tensor:
        """训练时：带 dropout 的合并，起正则化作用"""
        dropped = self.dropout(self.sub_tokens)  # 随机屏蔽子 token
        return dropped.mean(dim=0)               # [embed_dim]
    
    def get_merged_token(self) -> torch.Tensor:
        """推理时：确定性合并，不加 dropout"""
        return self.sub_tokens.mean(dim=0).detach()
```

### 语义引导注意力损失

```python
def semantic_guided_attention_loss(
    attn_maps: torch.Tensor,       # [B, H, W, N_tokens]
    semantic_mask: torch.Tensor,    # [B, 1, H_orig, W_orig]，前景=1
    concept_token_ids: list[int],   # 概念相关 token 的索引
) -> torch.Tensor:
    H, W = attn_maps.shape[1], attn_maps.shape[2]
    
    # 将语义 mask 下采样到注意力图分辨率
    mask = F.interpolate(semantic_mask.float(), size=(H, W), mode='nearest')
    background_mask = 1.0 - mask.squeeze(1)  # [B, H, W]
    
    loss = 0.0
    for token_id in concept_token_ids:
        attn = attn_maps[..., token_id]             # [B, H, W]
        background_attn = attn * background_mask     # 背景区域的注意力值
        loss += background_attn.pow(2).mean()        # 惩罚概念关注背景
    
    return loss / max(len(concept_token_ids), 1)
```

### 结构感知层限制

```python
def apply_layer_restriction(unet: nn.Module) -> None:
    """
    只解冻交叉注意力层（attn2），冻结其余所有层
    不同扩散模型框架中命名约定不同，先 print 确认层名
    """
    # 先检查：print([n for n, _ in unet.named_parameters() if "attn" in n.lower()])
    cross_attn_keys = ["attn2", "cross_attn", "encoder_attn"]
    
    trainable, frozen = 0, 0
    for name, param in unet.named_parameters():
        is_cross_attn = any(k in name for k in cross_attn_keys)
        param.requires_grad_(is_cross_attn)
        if is_cross_attn:
            trainable += param.numel()
        else:
            frozen += param.numel()
    
    ratio = trainable / (trainable + frozen) * 100
    print(f"可训练: {trainable:,} ({ratio:.1f}%) | 冻结: {frozen:,} ({100-ratio:.1f}%)")
```

### SEAL 微调主循环

```python
def seal_finetune(
    unet, noise_scheduler, ref_latent: torch.Tensor,
    sem_mask: torch.Tensor, concept_token_ids: list[int],
    n_steps: int = 300, lr: float = 1e-3, lambda_attn: float = 0.1,
) -> torch.Tensor:
    """
    ref_latent: [1, 4, H/8, W/8] 参考图的 VAE 编码
    sem_mask:   [1, 1, H, W] 语义前景 mask
    """
    split_tokens = SplitMergeTokens(embed_dim=768, num_splits=3)
    apply_layer_restriction(unet)
    
    # 只优化 split_tokens，U-Net 交叉注意力层通过 requires_grad 控制
    optimizer = torch.optim.Adam(split_tokens.parameters(), lr=lr)
    attn_cache = {}  # 用 forward hook 捕获注意力图（此处省略 hook 注册代码）

    for step in range(n_steps):
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (1,))
        noise = torch.randn_like(ref_latent)
        noisy = noise_scheduler.add_noise(ref_latent, noise, t)
        
        concept_emb = split_tokens()  # 当前合并的概念 token embedding
        # ... 构建 text embedding，替换 placeholder 位置为 concept_emb ...
        
        pred_noise = unet(noisy, t, encoder_hidden_states=concept_emb).sample
        loss_ldm = F.mse_loss(pred_noise, noise)
        
        # 从 hook 获取交叉注意力图计算语义损失
        # loss_attn = semantic_guided_attention_loss(attn_cache["maps"], sem_mask, concept_token_ids)
        # loss = loss_ldm + lambda_attn * loss_attn
        
        loss_ldm.backward()
        optimizer.step()
    
    return split_tokens.get_merged_token()  # 推理时使用此 token
```

## StickerBench 数据集

这是 SEAL 的另一重要贡献——一个结构化大规模贴纸数据集，用于系统评估个性化生成。

### 六维正交属性体系

```
StickerBench 标注结构
├── Appearance（外观）：颜色、纹理、物种特征
├── Emotion（情绪）：开心、悲伤、愤怒、惊讶...
├── Action（动作）：奔跑、跳跃、挥手...
├── Camera Composition（镜头构图）：全身、半身、特写
├── Style（风格）：Q版、写实、扁平化、线描...
└── Background（背景）：透明、纯色、场景
```

六个维度**相互正交**，可以固定五个维度只变一个，精确评估模型对单一属性的可控性。这比通用的 FID 分数更能反映贴纸个性化的实际需求。

### 评估框架

```python
def evaluate_seal(model, ref_image, attribute_prompts: dict) -> dict:
    """
    对每个属性维度，固定 Identity，只变该属性，
    同时评估：身份保持（DINO 相似度）和属性可控性（CLIP 对齐）
    """
    results = {}
    for attr, prompts in attribute_prompts.items():
        # prompts 示例：{"Emotion": ["happy", "sad", "surprised"]}
        generated = [model.generate(ref_image, f"<sticker> {p}") for p in prompts]
        
        identity_scores = [clip_image_sim(ref_image, g) for g in generated]
        attr_scores     = [clip_text_sim(p, g) for p, g in zip(prompts, generated)]
        
        results[attr] = {
            "identity": sum(identity_scores) / len(identity_scores),
            "controllability": sum(attr_scores) / len(attr_scores),
        }
    return results
    # ... (可视化代码省略)
```

## 工程实践

### 实际部署考虑

- **速度**：每张新参考图约需 300 步微调（RTX 3090 约 1-3 分钟），不适合在线实时场景
- **显存**：10-16 GB（取决于基础扩散模型规模），SD v1.5 约 10 GB
- **边际成本低**：微调一次后可生成无限多贴纸，适合**离线批量预生成**

### 常见坑

**坑一：语义 mask 质量差导致注意力损失失效**

```python
# 坏：简单阈值分割
mask = (image.mean(0) > 128).float()  # 背景复杂时完全错误

# 好：用 SAM 或 U2-Net 获取高质量前景 mask
# mask = sam_predictor.predict(image, point_coords=[[cx, cy]])[0]
```

**坑二：子 token 数 K 设置不当**

```python
# K=1 等价于普通 TTF，无分裂效果
# K>5 训练不稳定，loss 震荡
# 经验值：K=3 在大多数场景下最稳定
split_tokens = SplitMergeTokens(num_splits=3)  # 推荐
```

**坑三：不同框架的注意力层命名不同**

```python
# 微调前必须先确认层名，否则 restriction 完全失效
for name, _ in unet.named_parameters():
    if "attn" in name.lower():
        print(name)  # 确认后再设置 cross_attn_keys
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 贴纸/表情包批量生成 | 实时个性化（需要 <1s 响应） |
| 卡通/插画风格角色 | 复杂遮挡或多人场景 |
| 背景简单的参考图 | 参考图背景杂乱无章 |
| 可接受离线预计算的产品 | 用户上传即时响应的场景 |

## 与其他方法对比

| 方法 | 参考图数量 | 身份保持 | 属性可控 | 推理延迟 |
|-----|----------|---------|--------|---------|
| Textual Inversion | 3-10 张 | 中 | 高 | 快 |
| DreamBooth | 3-10 张 | 高 | 高 | 快 |
| IP-Adapter | 0 | 中 | 中 | 很快 |
| 普通单图 TTF | 1 张 | 高但有污染 | 低（结构僵化）| 慢 |
| **SEAL** | **1 张** | **高且干净** | **高** | 慢 |

## 我的观点

SEAL 的三组件设计有明显的工程美感：**每个组件对应一个具体的失败模式，目标清晰，可以独立消融验证**。这种"精确诊断 + 针对性修复"的思路值得借鉴，优于"堆数据/堆参数"的暴力路线。

**对 TTF 范式本身的顾虑**：每张新图都要 300 步微调，分钟级延迟在产品中是硬伤。更有前景的方向可能是：

1. **离线预计算**：品牌吉祥物/固定 IP 提前微调，在线推理
2. **与 IP-Adapter 融合**：把 SEAL 的空间解耦思路移植到无 TTF 的方案中
3. **一致性模型加速**：把微调步数从 300 压缩到 10-20 步

**StickerBench 的价值可能超过方法本身**：六维正交标注体系提供了一个清晰的评估范式——在贴纸/表情包这个垂直场景中，这可能成为社区标准基准。

论文链接：https://arxiv.org/abs/2604.26883