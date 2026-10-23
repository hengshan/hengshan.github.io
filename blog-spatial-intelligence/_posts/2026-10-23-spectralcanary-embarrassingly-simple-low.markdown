---
layout: post-wide
title: 'SpectralCanary：用低频水印审计文生图模型是否"偷"了你的数据集'
date: 2026-10-23 12:01:58 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2610.31659v1
generated_by: Claude Code CLI
---

## 一句话总结

SpectralCanary 把一张固定的"金丝雀"图像，通过低通滤波柔和地混入数据集里的每一张图片，之后只需看文生图（T2I）模型生成的图像里是否残留这个低频信号，就能判断这个模型是否在你的数据集上训练过——而且几乎不影响原图的视觉质量。

## 为什么这个问题重要？

文生图模型的训练数据来源越来越不透明。数据集所有者（摄影师、医学影像机构、卫星图像公司）想知道："我的图被某个模型拿去训练了吗？"但这个问题比想象中难：

- 模型可能**没有记住**具体图片，只是吸收了"风格""色调""构图"这类集合级特征，传统的 membership inference（判断单张图是否在训练集里）对这种情况基本失效。
- 被动审计（事后去猜）天然滞后；**主动审计**（proactive auditing）在数据发布前就埋入信号，事后检测，思路更可靠，但以往方法要在四个维度上做权衡：
  - **可验证性**：信号够不够强，能不能统计显著地证明"是"或"不是"
  - **保真度**：加了信号后图片还好看吗，不会被人眼或自动筛选发现异常
  - **鲁棒性**：图片被裁剪、压缩、重新编码之后信号还在吗
  - **部署成本**：是否需要额外生成模型、是否要对每张图做昂贵的优化

SpectralCanary 的核心创新是：**只用一次固定的低通滤波操作**，不需要为每张图单独优化、不需要辅助生成网络，却同时拿到了高保真（SSIM 0.989）和高检测率（TPR@1%FPR 91%+）。

## 背景知识

### 频域视角下的图像

任何一张图像都可以通过二维傅里叶变换（FFT）分解为不同频率的分量：

$$
F(u,v) = \sum_{x=0}^{H-1}\sum_{y=0}^{W-1} I(x,y) \, e^{-i2\pi(\frac{ux}{H}+\frac{vy}{W})}
$$

- **低频分量**（靠近频谱中心）对应图像的整体亮度、大块颜色、全局结构——人眼对低频失真相对不敏感，但这部分往往是图像"全局统计特征"最集中的地方。
- **高频分量**（频谱边缘）对应边缘、纹理细节——人眼对高频失真非常敏感，而且 JPEG 压缩、resize 等常见操作会优先损失高频。

这就是为什么把水印藏在**低频**里是个好选择：既不容易被肉眼察觉（低频扰动表现为柔和的色调偏移），又比高频更能抗住常见的图像处理（压缩、resize、重新编码），因为这些操作通常"杀高频留低频"。

### 和传统水印/会员推断的区别

传统数字水印关心"这张图是不是被改过"，而这里关心的是"模型的*生成分布*是否携带了数据集的统计特征"。模型训练相当于一个巨大的、有损的信息传递管道——数据集里稳定、一致、低频的信号更容易穿过这个管道，被模型学到并在生成时复现。

## 核心方法

### 直觉解释

想象给整个数据集"统一加一层很淡的滤镜"：

1. 准备一张固定的金丝雀图像 $C$（随机噪声图或任意图案都可以）。
2. 对 $C$ 做低通滤波，只保留它的低频部分 $C_{low}$。
3. 把 $C_{low}$ 以很小的强度 $\alpha$ 混合进数据集里**每一张**图片的低频分量，caption 完全不改。
4. 发布数据集，等待模型被训练。
5. 之后对任意可疑模型生成一批图，提取这些图的低频分量，和 $C_{low}$ 做相关性检测——如果相关性显著高于随机基线，说明模型"学会了"这个全局一致的低频偏置，等价于它见过这批数据。

关键在于：因为**每一张**训练图都带有同一个低频偏置，这个偏置会成为数据集层面"异常稳定"的统计特征，比任何单张图的细节都更容易被模型学习和泄露出来。

### 数学细节

设原图 $I$，金丝雀 $C$，低通掩码（频域里的平滑圆形窗）为 $M_{cutoff}$。频域嵌入：

$$
\hat{I} = \mathcal{F}^{-1}\Big[\mathcal{F}(I) + \alpha \cdot M_{cutoff} \odot \mathcal{F}(C)\Big]
$$

其中 $\mathcal{F}$ 是 FFT，$M_{cutoff}$ 是以频谱中心为圆心、半径为 $cutoff$ 的平滑（非硬截断）高斯型窗口，避免振铃伪影（ringing artifact）。$\alpha$ 控制嵌入强度，决定保真度与可检测性的权衡。

检测时，对可疑生成图 $G$ 计算其低频分量与金丝雀低频分量的归一化相关系数：

$$
s(G) = \frac{\big\langle M_{cutoff}\odot\mathcal{F}(G),\; M_{cutoff}\odot\mathcal{F}(C)\big\rangle}{\|M_{cutoff}\odot\mathcal{F}(G)\|\;\|M_{cutoff}\odot\mathcal{F}(C)\|}
$$

把一批（如 30 张）生成图的得分汇总（取均值或做 t 检验），与"干净模型"（未见过带水印数据）的得分分布比较，用阈值做模型级判决。阈值通过校准集标定，使假阳性率（FPR）控制在 1% 左右，论文报告在该阈值下单次生成的真阳性率（TPR）平均达到 91.1%。

### Pipeline 概览

```
数据集发布前:
  原始图像集 → [FFT + 低通混合金丝雀] → 发布的"带水印"数据集 (caption 不变)

审计阶段:
  可疑模型 → 生成 N 张图 → [FFT 提取低频] → 与金丝雀低频模板做相关性 → 阈值判决
```

## 实现

下面给出核心算法的最小可运行实现（原论文未公开代码，以下为我们自己复现的教学版本）。

### 环境配置

```bash
pip install numpy scipy pillow matplotlib scikit-image
```

### 嵌入模块：低通混合金丝雀

```python
import numpy as np

def make_lowpass_mask(h, w, cutoff_ratio=0.1):
    """生成以频谱中心为圆心的平滑(高斯型)低通掩码"""
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    sigma = cutoff_ratio * min(h, w)
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))  # 高斯软窗，避免振铃

def embed_canary(image, canary, alpha=0.05, cutoff_ratio=0.1):
    """把 canary 的低频成分以强度 alpha 混入 image（逐通道处理）"""
    h, w = image.shape[:2]
    mask = make_lowpass_mask(h, w, cutoff_ratio)
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        F_img = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
        F_can = np.fft.fftshift(np.fft.fft2(canary[:, :, c]))
        F_mix = F_img + alpha * mask * F_can
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(F_mix)))
    return np.clip(out, 0, 255).astype(np.uint8)
```

### 检测模块：低频相关性评分

```python
def lowfreq_score(image, canary, cutoff_ratio=0.1):
    """计算待测图像与 canary 在低频段的归一化相关系数"""
    h, w = image.shape[:2]
    mask = make_lowpass_mask(h, w, cutoff_ratio)
    scores = []
    for c in range(image.shape[2]):
        F_img = mask * np.fft.fftshift(np.fft.fft2(image[:, :, c]))
        F_can = mask * np.fft.fftshift(np.fft.fft2(canary[:, :, c]))
        num = np.real(np.sum(F_img * np.conj(F_can)))
        denom = np.linalg.norm(F_img) * np.linalg.norm(F_can) + 1e-8
        scores.append(num / denom)
    return float(np.mean(scores))

def calibrate_threshold(clean_scores, target_fpr=0.01):
    """用干净(未见过水印)模型的得分分布标定判决阈值"""
    return float(np.quantile(clean_scores, 1 - target_fpr))
```

使用方式：先用 `embed_canary` 处理整个数据集并发布；拿到可疑模型的若干张生成图后，对每张图算 `lowfreq_score`，和用一批"已知未训练过该数据"的基线模型得分算出的 `calibrate_threshold` 比较，超过阈值即判定该模型用过这个数据集。

### 3D/频域可视化

```python
import matplotlib.pyplot as plt

def visualize_spectrum(image, canary, cutoff_ratio=0.1, channel=0):
    mask = make_lowpass_mask(*image.shape[:2], cutoff_ratio)
    spec_img = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image[:, :, channel]))))
    spec_can = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(canary[:, :, channel]))))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(spec_img, cmap="magma"); axes[0].set_title("原图频谱")
    axes[1].imshow(spec_can, cmap="magma"); axes[1].set_title("Canary 频谱")
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title(f"低通掩码 (cutoff={cutoff_ratio})")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
```

这个可视化能直观看到：低通掩码只保留频谱中心一小圈区域，金丝雀的能量被限制在这个圈内，混入后几乎不改变图像在高频区的"纹理指纹"——这正是保真度高的原因。

## 实验

### 数据集说明

论文主实验用 **Pokemon 数据集**（T2I 微调的标准 benchmark，图片风格统一、数量可控，适合快速验证检测方法），并扩展到艺术画作、医学影像（ROCOv2）、卫星图像、时尚图片五个领域，覆盖 Stable Diffusion 3.5 等不同架构。这类数据集获取门槛不高（公开可下载），但医学/卫星领域要注意授权和隐私合规。

### 定量评估（摘自论文核心结果）

| 方法 | 单次生成 TPR@1%FPR | 30 次生成模型级准确率 | SSIM | ΔFID |
|-----|------|-----|------|------|
| SpectralCanary | 91.1% | 94.8% | 0.989 | 0.05 |
| 对比方法（高频嵌入） | 更低 | 更低 | 相近或更差 | 更高 |

关键的**匹配能量消融实验**（matched-energy ablation）：在相同嵌入能量预算下，把信号放在低频比放在中/高频段迁移效果更好——这是支撑"低频是更好载体"这一设计选择的核心证据，而不是靠更大的扰动强度取胜。

### 定性结果

- 好案例：数据集风格统一（如 Pokemon 卡通风），低频色调偏移稳定地传递到生成图里，肉眼基本看不出原图被改动。
- 失败/边界案例：论文测试了"自适应攻击"（攻击者知道有水印，主动去除）——最有效的移除手段需要对生成结果做激进的后处理，代价是 ΔFID 上升超过 25 个点，也就是说**去水印的成本高到会明显损害模型本身的生成质量**，这是一个很好的"攻防不对称性"设计。

## 工程实践（重要！）

### 实际部署考虑

- **计算成本**：嵌入阶段只需一次 FFT + 逐图混合，复杂度 $O(HW\log HW)$ per image，CPU 上对几千张图也是分钟级，不需要 GPU。
- **检测阶段**：生成图数量越多（论文用 30 张）统计越稳健，单张图检测率已经不低（91%），但模型级判决建议多采样几十张再汇总，降低方差。
- **内存**：FFT 对大图（>2K）内存占用会上升，建议检测前统一 resize 到固定分辨率（如 512×512），同时也要保证嵌入和检测用的是同一分辨率假设，否则低频掩码的"半径"语义会变。

### 数据采集/发布建议

- 金丝雀图案本身不需要特殊设计，论文用固定随机图案即可，但**不要用太规则的图案**（如纯色块），否则可能被简单的频域异常检测识别出"数据集整体有一个异常一致的低频分量"。
- $\alpha$（嵌入强度）要根据目标保真度调：论文选取的 $\alpha$ 使 SSIM 保持在 0.989 这种"人眼基本无感"的水平，建议先在小样本上跑一遍 SSIM/LPIPS 再批量处理全量数据集。

### 常见坑

1. **问题**：用硬截断的圆形掩码做低通滤波，会在空间域产生振铃伪影（图像边缘出现波纹）。
   **解决**：用高斯型软掩码（如上面 `make_lowpass_mask` 的实现），而不是 0/1 的硬边界。

```python
# 错误示范：硬截断掩码会导致 ringing
hard_mask = (dist < cutoff).astype(float)  # 不要这样做
# 正确：用高斯软窗
soft_mask = np.exp(-(dist ** 2) / (2 * sigma ** 2))
```

2. **问题**：检测时直接用原始图像对比，没有做分辨率/中心对齐，导致 FFT 频谱错位，相关系数虚低。
   **解决**：检测前统一 resize + 居中裁剪，保证生成图和金丝雀模板处于同一频域坐标系。

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 数据集所有者在发布前能控制图像处理流程 | 数据已经发布、无法重新嵌入水印 |
| 需要证明"模型是否用过我的数据集"（集合级） | 需要证明"某一张具体图片"被训练过（单图级会员推断） |
| caption 不能改动（如配对好的图文数据） | 可以容忍修改 caption 的场景（有更多自由度的方案可能更强） |
| 对抗的是"正常训练流程"的模型 | 对抗的是专门针对本方法设计的自适应去水印攻击者（虽然论文显示代价很高，但不是不可能） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 被动 Membership Inference | 不需要提前干预数据 | 对"学到风格而非记住图片"的情况失效 | 怀疑模型逐字记忆训练图的场景 |
| 高频/像素级水印 | 嵌入能量小、不改变全局统计 | 容易被压缩/resize 破坏，鲁棒性差 | 不涉及大规模图像处理流程的场景 |
| 辅助生成式水印（需要额外模型/优化） | 可能嵌入更隐蔽 | 部署成本高，每张图或每批数据需要额外计算 | 数据量较小、可承受优化成本的场景 |
| SpectralCanary | 部署简单、保真度高、鲁棒性强、理论支持清晰 | 对自适应攻击不是绝对免疫（只是代价高） | 大规模数据集发布前的主动审计 |

## 我的观点

这篇论文最有价值的部分不是"又发明了一种水印"，而是用**匹配能量消融实验**严谨地证明了"低频是更好的载体"这个设计选择——这给后续工作提供了一个可复用的结论：如果你要在图像里藏一个需要"穿过神经网络训练"这个有损信道的信号，优先考虑频谱中心而不是边缘。

离实际落地还有几个开放问题值得关注：

1. **跨数据集水印冲突**：如果多个数据集所有者都用类似的低频金丝雀方案，检测时如何避免"假阳性串台"？论文目前主要验证单一水印场景。
2. **多模态/视频扩展**：T2I 模型已经在往 T2V、3D 生成演进，低频思路能否直接迁移到视频的时间维度（比如时序上的低频一致性）还没有定论。
3. **法律效力**：技术上能检测到相关性，不等于在版权纠纷中能作为"证据"——统计阈值和举证标准之间还有距离。

总体而言，这是一个"工程成本低、理论干净、攻防不对称性设计得很聪明"的方案，适合数据集所有者作为发布前的低成本保险措施，但不应被当作能对抗专业对手的终极方案。