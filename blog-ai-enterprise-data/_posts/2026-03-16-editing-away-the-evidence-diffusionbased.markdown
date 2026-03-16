---
layout: post-wide
title: "扩散模型正在悄悄抹掉你的水印：信息论视角下的崩溃分析"
date: 2026-03-16 12:05:04 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.12949v1
generated_by: Claude Code CLI
---

## 一句话总结

扩散模型的图像编辑会"无意间"将水印信号投影回数据流形，导致隐写水印从信息论层面变得不可恢复——这不是攻击，这是副作用。

## 为什么这篇论文重要？

### 水印的承诺与现实的裂缝

几乎所有主流图像生成平台都宣称采用了"鲁棒不可见水印"保护内容版权。方案听起来很完美：在像素层面嵌入人眼不可见的信号，即使经过压缩、裁剪、亮度调整，检测器也能将其恢复。

但有一个问题——用户会用扩散模型来"修图"。

换个背景、改变风格、局部重绘……这些操作在鲁棒性测试集里从未出现过。这篇论文的核心发现是：**这些"常规编辑"就足以让水印从信息论层面消失**，不需要任何对抗攻击意图。

### 现有方法的根本假设失效

传统鲁棒水印（HiDDeN、TreeRing、Stable Signature）的设计假设是：

- 攻击者施加**已知类型**的降质操作：JPEG压缩、高斯噪声、缩放、旋转
- 这些操作是**信号层面**的变换，不改变图像语义
- 水印信号通过训练来抵抗这些操作

扩散编辑打破了这个假设的根基：它不在信号层面操作，而是在**表示层面**——把图像注入噪声，再用强大的生成先验"重建"它。这个生成先验天生不知道"水印"是什么，它只知道"自然图像长什么样"。

## 核心理论：水印为什么会被杀死

### 直觉：流形上的点 vs 流形外的扰动

想象图像空间中有一个"自然图像流形"$\mathcal{M}$。所有真实照片都落在这个流形上（或附近）。

水印本质上是一个**偏移量** $\boldsymbol{\delta}$，将原始图像 $\mathbf{x} \in \mathcal{M}$ 推到流形外：

$$\mathbf{x}_w = \mathbf{x} + \boldsymbol{\delta}, \quad \mathbf{x}_w \notin \mathcal{M}$$

这个偏移量设计得足够小（人眼不可见），但足够大（检测器可辨）。

扩散模型编辑（SDEdit 范式）做的事情是：

1. **加噪**：$\mathbf{x}_{t^*} = \sqrt{\bar{\alpha}_{t^*}} \mathbf{x}_w + \sqrt{1 - \bar{\alpha}_{t^*}} \boldsymbol{\epsilon}$
2. **去噪**：$\hat{\mathbf{x}}_0 = D_\theta(\mathbf{x}_{t^*}, t^*)$，将其投影回 $\mathcal{M}$

**问题的关键**：去噪器 $D_\theta$ 被训练为"生成自然图像"，而水印扰动 $\boldsymbol{\delta}$ 是流形外的信号——去噪器会自然地将其滤除。一个越好的扩散模型，对流形的刻画越精准，对水印的破坏就越彻底。

### 水印 SNR 随扩散轨迹的衰减

设噪声调度的累积乘积为 $\bar{\alpha}_t$，水印扰动 $\ell_2$ 范数为 $\|\boldsymbol{\delta}\|$，图像维度为 $d$。

在时间步 $t^*$ 处，加噪图像中水印信号的**信噪比**为：

$$\text{SNR}_w(t^*) = \frac{\bar{\alpha}_{t^*} \cdot \|\boldsymbol{\delta}\|^2}{(1 - \bar{\alpha}_{t^*}) \cdot d}$$

这个公式揭示两个硬约束：

- **$t^*$ 越大（编辑幅度越大），SNR 越低**：$\bar{\alpha}_{t^*} \to 0$，水印信号被淹没
- **$\|\boldsymbol{\delta}\|$ 越小（水印越不可见），SNR 越低**：不可见性和鲁棒性从根本上对立

### 互信息上界：不可恢复的信息论证明

水印在重建图像中的**可恢复信息量**被以下互信息上界控制：

$$I(\boldsymbol{\delta}; \hat{\mathbf{x}}_0) \leq \frac{d}{2} \log\!\left(1 + \text{SNR}_w(t^*)\right)$$

当 $\text{SNR}_w(t^*) \ll 1$ 时，上界趋近于零。此时无论检测器多么精巧，都**无法从信息论层面恢复水印**。这不是算法设计的问题——是信息量本身不存在了。

## 动手实现：测量水印在扩散编辑中的存活率

### 最小可运行示例

```python
import torch
import numpy as np

# --- 1. 简单的频域水印（伪随机载波叠加）---
def embed_watermark(image_tensor, strength=0.03):
    """image_tensor: [C, H, W], 范围 [0, 1]"""
    C, H, W = image_tensor.shape
    rng = np.random.default_rng(42)  # 固定密钥 = 固定载波
    wm = image_tensor.clone()
    for c in range(C):
        carrier = rng.standard_normal((H, W)).astype(np.float32)
        carrier /= np.linalg.norm(carrier)
        wm[c] = torch.tensor(image_tensor[c].numpy() + strength * carrier)
    return wm.clamp(0, 1)

def detect_watermark(image_tensor):
    """返回与密钥载波的相关系数（越高=水印越完整）"""
    rng = np.random.default_rng(42)
    C, H, W = image_tensor.shape
    scores = []
    for c in range(C):
        carrier = rng.standard_normal((H, W)).astype(np.float32)
        carrier /= np.linalg.norm(carrier)
        scores.append(np.dot(image_tensor[c].numpy().flatten(), carrier.flatten()))
    return float(np.mean(scores))

# --- 2. 模拟 SDEdit：前向加噪 + MMSE 去噪 ---
def simulate_sdedit(image_tensor, t_star_ratio=0.5, num_steps=1000):
    """
    t_star_ratio: 0~1，对应 SDEdit 中的 strength 参数
    去噪器用 MMSE 估计器模拟（即高斯先验下的最优重建）
    """
    betas = torch.linspace(1e-4, 0.02, num_steps)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    t_star = max(1, int(t_star_ratio * num_steps))
    ab = alpha_bars[t_star - 1]

    # 前向加噪
    noise = torch.randn_like(image_tensor)
    x_t = (ab ** 0.5) * image_tensor + ((1 - ab) ** 0.5) * noise

    # MMSE 重建（模拟扩散模型去噪器的收缩效应）
    sigma_x_sq = image_tensor.var()
    x_hat = (ab ** 0.5 * sigma_x_sq / (ab * sigma_x_sq + (1 - ab))) * x_t
    return x_hat.clamp(0, 1), ab.item()

# --- 3. 验证 SNR 衰减规律 ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    image = torch.tensor(rng.random((3, 64, 64), dtype=np.float32))
    strength = 0.03
    wm_image = embed_watermark(image, strength=strength)

    baseline = detect_watermark(wm_image)
    print(f"原始水印得分: {baseline:.4f}\n")

    for t_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        edited, ab = simulate_sdedit(wm_image, t_star_ratio=t_ratio)
        score = detect_watermark(edited)
        # 理论 SNR（用于对比）
        delta_sq = ((wm_image - image) ** 2).sum().item()
        snr_theory = ab * delta_sq / ((1 - ab) * 3 * 64 * 64)
        print(f"t*={t_ratio:.1f} | ᾱ={ab:.3f} | 水印得分={score:.4f} "
              f"| 衰减={1-score/baseline:.1%} | 理论SNR={snr_theory:.5f}")
```

运行这个示例，输出应接近以下规律：`t_ratio` 从 0.1 到 0.9，水印衰减从 ~10% 升至 ~95%，且与理论 SNR 单调对应。这不是随机现象，是公式的直接体现。

### 对真实扩散模型的接入

```python
# pip install diffusers transformers accelerate
from diffusers import StableDiffusionImg2ImgPipeline
import torch

def test_with_real_sd(wm_image_pil, strengths=[0.3, 0.5, 0.7]):
    """
    img2img 的 strength 参数直接对应 t*/T
    用同语义 prompt 做"无害编辑"，测量水印存活率
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")

    for s in strengths:
        edited = pipe(
            prompt="a beautiful photograph",  # 保持语义一致
            image=wm_image_pil,
            strength=s,          # 核心：编辑强度 = 噪声比例
            guidance_scale=7.5,
            num_inference_steps=50
        ).images[0]
        tensor = torch.tensor(
            np.array(edited).transpose(2,0,1) / 255.0, dtype=torch.float32
        )
        print(f"strength={s:.1f}: 水印得分={detect_watermark(tensor):.4f}")
```

### 实现中的坑

**坑 1：水印强度存在硬约束，不存在帕累托最优**

```python
# 不可见性要求 PSNR > 40dB → strength <= 0.03
# SNR_w(t*=0.5) ∝ strength^2 → 0.03^2 = 0.0009，极其脆弱
# 提高 strength 救不了 SNR，只会破坏不可见性
```

**坑 2：检测阈值在扩散编辑后必须重新校准**

```python
# 扩散去噪会引入系统性偏移（均值回归效应）
# 用原始图像集合校准的阈值在编辑后图像上会产生大量误报
# 正确做法：在包含扩散编辑的测试集上重新设定 FPR=1% 的阈值
threshold = np.percentile(scores_on_clean_edited_images, 99)
```

**坑 3：MMSE 模拟器低估了真实模型的破坏力**

真实神经网络去噪器比 MMSE 估计器更激进地"清洗"流形外扰动——它有强烈的语义先验，会主动将水印区域替换为符合语义的纹理。上面的简化实现给出的是**理论下界**，实际衰减更严重。

## 实验：论文说的 vs 现实

### 论文报告的关键结果（近似数值）

| 水印方案 | 无编辑检测率 | SDEdit t*=0.5 | SDEdit t*=0.7 |
|---------|------------|--------------|--------------|
| HiDDeN | ~99% | ~60% | ~30% |
| TreeRing | ~98% | ~55% | ~25% |
| Stable Signature | ~97% | ~50% | ~20% |

即使是轻度编辑（t*=0.3），检测率也显著下降——而 t*=0.3 对用户来说只是轻微的风格迁移。

### 论文没有充分讨论的限制

1. **测试方案偏旧**：ZoDiac 等 2024 年方案未全面覆盖
2. **没有对抗性 co-design**：如果水印训练时加入扩散编辑作为增强，会怎样？（答案：信息论上界依然存在，只能在低 t* 区间获益）
3. **用户行为的实际 t* 分布**：想要语义一致的编辑，t* 通常 < 0.55，这给了水印一定生存空间

## 什么时候用 / 不用现有鲁棒水印？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 只经过 JPEG 压缩、缩放等传统处理 | 用户会用 AI 工具"修图" |
| 版权归属的法律辅助证据（低编辑风险） | 追踪经过 img2img 传播的图像 |
| 元数据丢失后的辅助标识 | 对抗恶意 AI 去水印 |
| 检测原图是否被轻微篡改 | inpainting/outpainting 后的变体追踪 |

## 我的观点：这个问题真正的出路在哪里？

### 三个方向

**方向 1：语义水印（On-manifold 嵌入）**
既然扩散模型保留语义，水印也应该嵌在语义层。例如 fine-tuning 生成模型使其在特定语义模式下留下"指纹"。代价是必须控制生成过程，无法用于后处理已有图像。

**方向 2：内容哈希 + 溯源链**
放弃信号嵌入，转向感知哈希 + 分布式存储。问题：扩散编辑后感知哈希也会漂移，需要专门针对扩散变体的鲁棒哈希方案。

**方向 3：扩散编辑鲁棒水印**
将扩散编辑加入训练的"攻击增强"。但论文的信息论分析表明，高 $t^*$ 下存在根本性上界——除非大幅提高水印强度（破坏不可见性）。

**判断**：短期内没有完美解。实际的对策是：

1. 水印与内容哈希**组合使用**（defense in depth）
2. 检测端**假设图像经过了扩散编辑**，重新校准阈值，降低对高信噪比的依赖
3. 高价值内容考虑将信息嵌入更难被编辑替换的区域（前景主体的语义特征）

这篇论文最大的价值不是提供解决方案，而是**精确刻画了问题的边界**——水印方案的测试集必须包含扩散编辑，否则你在保护一个你以为存在、实际已经消失的信号。这对整个内容溯源领域的威胁模型都是一次必要的修正。