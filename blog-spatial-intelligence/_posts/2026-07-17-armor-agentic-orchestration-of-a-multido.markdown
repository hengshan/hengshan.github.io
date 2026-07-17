---
layout: post-wide
title: "深度伪造检测器的对抗迁移攻击：ARMOR++ 多智能体框架解析"
date: 2026-07-17 12:04:24 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2607.15246v1
generated_by: Claude Code CLI
---

我将写一篇关于 ARMOR++ 的深度技术博客，聚焦对抗迁移攻击原理与防御启示，包含可运行代码示例。


## 一句话总结

ARMOR++ 通过 VLM + LLM 双智能体协调五种攻击原语，在无查询黑盒条件下生成高迁移性对抗扰动，揭示了现有深度伪造检测器的真实可靠性缺口——理解这些攻击，是构建更鲁棒防御体系的前提。

---

## 为什么这个问题重要？

深度伪造检测器在学术基准上动辄 99% 准确率，但真实部署中屡屡被绕过。问题的根源在于：**大多数检测器依赖的是架构相关的伪造指纹**——GAN 生成图像在高频域留下的特定叠加痕迹、扩散模型的特征分布偏移。这些特征脆弱、不可泛化。

更严峻的是**黑盒迁移攻击**的威胁：攻击者在本地的替代模型（surrogate）上生成对抗扰动，直接迁移到目标检测器，全程无需访问目标模型。

ARMOR++ 在 AADD-2025 基准上的结果表明，当前最优检测器在盲目标攻击成功率（Blind-target ASR）上仍有显著缺口，且这一缺口在对抗性防御配置下依然存在。

---

## 背景知识

### 迁移攻击的核心挑战

在 CNN 替代模型上生成的扰动，往往对 Transformer 目标模型失效。根本原因是**归纳偏置不同**：

- **CNN**：局部感受野，依赖低级纹理和高频特征
- **ViT / Transformer**：全局注意力机制，依赖语义块级特征

在 CNN 上过度拟合的扰动，换到 ViT 上大概率失效。提升跨架构迁移性的关键是**同时攻击多个层次的特征**。

### 对抗扰动约束

标准形式化：给定伪造图像 $x$，目标是找到扰动 $\delta$ 使检测器 $f$ 误判：

$$
\arg\max_\delta \mathcal{L}(f(x + \delta), y_{\text{fake}}) \quad \text{s.t.} \quad \|\delta\|_\infty \leq \epsilon
$$

实践中 $\epsilon$ 通常取 $4/255$ 到 $16/255$，保持扰动对人眼不可见。

---

## ARMOR++ 框架核心

### 整体 Pipeline

```
伪造图像输入
    ↓
[Qwen2.5-VL] → 空间语义先验（显著区域、边缘、频域特征分布）
    ↓
[Qwen3 LLM] → 智能编排（原语选择 + 超参数重参数化）
    ↓
五种攻击原语并行执行
    ↓
熵正则化扰动混合 (Entropy-Regularized Mixing)
    ↓
对抗图像输出 → 绕过深度伪造检测器
```

### 五种攻击原语

| 原语类型 | 目标 | 针对的归纳偏置 |
|---------|------|--------------|
| 密集优化（Dense Opt）| 像素级梯度对齐 | CNN 纹理特征 |
| 显著性攻击（Saliency）| 攻击语义显著区域 | Transformer 注意力区域 |
| 空间变换（Spatial）| 几何扭曲绕过纹理检测 | 局部感受野依赖 |
| 频域扰动（Frequency）| 修改 DCT/FFT 系数 | 频域指纹检测 |
| 块结构修改（Block）| 局部块级扰动 | Patch-based 特征 |

多原语覆盖异构目标的不同特征层次，这是 ARMOR++ 迁移性提升的核心逻辑。

---

## 技术深度：关键组件实现

### 1. 密集优化原语：MI-FGSM

动量迭代 FGSM 是迁移攻击的基础模块，动量项减少了过拟合到单一替代模型的问题：

$$
g_{t+1} = \mu \cdot g_t + \frac{\nabla_x \mathcal{L}(x_t, y)}{\|\nabla_x \mathcal{L}(x_t, y)\|_1}
$$

$$
x_{t+1} = \text{Clip}\left(x_t + \alpha \cdot \text{sign}(g_{t+1}),\; x \pm \epsilon\right)
$$

```python
import torch
import torch.nn.functional as F

def mi_fgsm(model, x, y, epsilon=8/255, alpha=2/255, steps=10, mu=1.0):
    """MI-FGSM：动量迭代 FGSM，提升黑盒迁移性"""
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()

        grad = x_adv.grad.data
        # L1 归一化梯度后累积动量
        g = mu * g + grad / (grad.abs().sum(dim=[1,2,3], keepdim=True) + 1e-8)

        x_adv = x_adv.detach() + alpha * g.sign()
        # 投影回 L∞ 球并限制到合法像素范围
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon).clamp(0, 1)

    return x_adv
```

### 2. 频域扰动原语

深度伪造检测器常依赖 DCT 域的特定伪影，频域扰动直接在 FFT 系数上施加扰动，消除或混淆这些指纹：

$$
x_{\text{adv}} = \mathcal{F}^{-1}\!\left(\mathcal{F}(x) + \delta_{\text{freq}} \odot M_{\text{band}}\right)
$$

其中 $M_{\text{band}}$ 是频段掩码，集中扰动在检测器敏感的高频区域。

```python
import torch
import numpy as np

def frequency_perturbation(x, budget=0.05, band="high"):
    """频域扰动：定向修改特定频段，破坏伪造指纹"""
    B, C, H, W = x.shape
    x_np = x.detach().cpu().numpy()
    x_freq = np.fft.fft2(x_np, axes=(-2, -1))
    x_freq = np.fft.fftshift(x_freq, axes=(-2, -1))

    # 构建频段掩码（仅扰动高频区域）
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - H//2)**2 + (X - W//2)**2)
    mask = (dist > H / 4).astype(float) if band == "high" else (dist <= H / 4).astype(float)

    noise = np.random.randn(*x_freq.shape) * budget
    x_freq += noise * mask[np.newaxis, np.newaxis]

    x_freq = np.fft.ifftshift(x_freq, axes=(-2, -1))
    x_out = np.real(np.fft.ifft2(x_freq, axes=(-2, -1)))
    return torch.tensor(np.clip(x_out, 0, 1), dtype=x.dtype, device=x.device)
```

### 3. 熵正则化混合

ARMOR++ 的编排核心：LLM 为各原语分配权重 $w_i$，同时强制满足最小熵约束，防止系统退化为单一策略。

$$
\delta^* = \sum_{i=1}^{5} w_i \cdot \delta_i \qquad \text{s.t.} \quad -\sum_i w_i \log w_i \geq H_{\min}
$$

```python
def entropy_regularized_mixing(perturbations, weights=None, min_entropy=1.0):
    """
    熵正则化混合：保证多原语均有贡献
    perturbations: List[(B,C,H,W)] 各原语输出
    weights: LLM 编排器给出的初始权重
    """
    n = len(perturbations)
    if weights is None:
        w = torch.ones(n) / n
    else:
        w = F.softmax(torch.tensor(weights, dtype=torch.float32), dim=0)

    entropy = -(w * (w + 1e-8).log()).sum().item()

    if entropy < min_entropy:
        # 向均匀分布拉近直到满足熵约束
        alpha = min(1.0, max(0.0, (min_entropy - entropy) / (np.log(n) - entropy + 1e-8)))
        w = (1 - alpha) * w + alpha * torch.ones(n) / n
        w = w / w.sum()

    stacked = torch.stack(perturbations, dim=0)          # (n, B, C, H, W)
    mixed = (w.view(n, 1, 1, 1, 1) * stacked).sum(dim=0)
    return mixed, w
```

$H_{\min}$ 建议从 $\ln(n)/2 \approx 0.8$ 开始调参——设太高退化为均等混合，设太低又变回单原语。

### 4. VLM 语义先验：LLM 编排器的输入

Qwen2.5-VL 的作用是把"图像中哪里重要"翻译成攻击策略的先验信息，再由 Qwen3 LLM 据此分配原语权重：

```python
# 概念示意（实际调用 Qwen2.5-VL API）
def get_semantic_priors_and_weights(image, vlm_client, llm_client):
    vlm_prompt = """
    分析这张图像，输出 JSON：
    1. 视觉显著区域坐标（人脸关键点、边缘、纹理复杂区域）
    2. 高频/低频特征的空间分布描述
    """
    visual_analysis = vlm_client.analyze(image=image, prompt=vlm_prompt)

    llm_prompt = f"""
    基于以下图像分析：{visual_analysis}
    为五种攻击原语分配权重（dense_opt, saliency, spatial, frequency, block）：
    - 若边缘/纹理丰富 → 提高 frequency 权重
    - 若人脸显著区域明确 → 提高 saliency 权重
    输出 JSON: {{"weights": [w1, w2, w3, w4, w5]}}
    """
    weights = llm_client.generate(llm_prompt)
    return weights  # ... (JSON 解析省略)
```

---

## 防御视角：检测器如何变得更鲁棒？

理解攻击是为了构建更好的防御。ARMOR++ 的成功揭示三个防御方向：

### 1. 多域特征融合（打破单一依赖）

```python
class RobustDeepfakeDetector(torch.nn.Module):
    """融合三个域的特征，避免单一频域/纹理依赖"""
    def __init__(self, backbone):
        super().__init__()
        self.spatial_enc = backbone        # 像素域
        self.freq_enc = backbone           # 频域（输入 FFT 幅度谱）
        self.fusion = torch.nn.Linear(768 * 2, 2)

    def forward(self, x):
        f_spatial = self.spatial_enc(x).mean(dim=[2, 3])
        # 转换为频域幅度谱再编码
        x_freq = torch.fft.fft2(x).abs().log1p()
        x_freq = (x_freq - x_freq.mean()) / (x_freq.std() + 1e-8)
        f_freq = self.freq_enc(x_freq).mean(dim=[2, 3])
        return self.fusion(torch.cat([f_spatial, f_freq], dim=-1))
```

### 2. 对抗训练（主动学习鲁棒特征）

```python
def adversarial_training_step(model, x_real, x_fake, optimizer, epsilon=4/255):
    """在每个 batch 内生成对抗样本，增强模型鲁棒性"""
    y_real = torch.ones(len(x_real), dtype=torch.long, device=x_real.device)
    y_fake = torch.zeros(len(x_fake), dtype=torch.long, device=x_fake.device)

    x_real_adv = mi_fgsm(model, x_real, y_real, epsilon=epsilon, steps=5)
    x_fake_adv = mi_fgsm(model, x_fake, y_fake, epsilon=epsilon, steps=5)

    x_all = torch.cat([x_real, x_real_adv, x_fake, x_fake_adv])
    y_all = torch.cat([y_real, y_real, y_fake, y_fake])

    optimizer.zero_grad()
    loss = F.cross_entropy(model(x_all), y_all)
    loss.backward()
    optimizer.step()
    return loss.item()
```

### 3. 随机平滑推理（破坏精确扰动依赖）

```python
def randomized_smoothing_inference(model, x, sigma=0.05, n_samples=50):
    """多次加噪推理取多数票，有效应对确定性对抗扰动"""
    votes = torch.zeros(x.shape[0], 2, device=x.device)
    for _ in range(n_samples):
        noise = torch.randn_like(x) * sigma
        with torch.no_grad():
            pred = model((x + noise).clamp(0, 1)).argmax(dim=-1)
        votes[torch.arange(x.shape[0]), pred] += 1
    return votes.argmax(dim=-1)
```

---

## 工程实践

### 正确评估检测器鲁棒性

只报告干净准确率是常见的误导性评估。正确做法必须同时报告对抗准确率：

```python
def evaluate_robustness(model, dataloader, epsilon=8/255):
    clean_correct = adv_correct = total = 0
    for x, y in dataloader:
        with torch.no_grad():
            clean_correct += (model(x).argmax(1) == y).sum().item()
        x_adv = mi_fgsm(model, x, y, epsilon=epsilon)
        with torch.no_grad():
            adv_correct += (model(x_adv).argmax(1) == y).sum().item()
        total += len(y)
    print(f"Clean Acc:  {100*clean_correct/total:.1f}%")
    print(f"Robust Acc: {100*adv_correct/total:.1f}%")
    print(f"Gap:        {100*(clean_correct-adv_correct)/total:.1f}%")
```

### 常见坑

1. **JPEG 压缩消除扰动但性能下降**  
   频域攻击的扰动部分在 JPEG 压缩后仍存在。正确做法：在压缩后图像上重新评估，并把 JPEG 增强纳入检测器训练数据。

2. **替代模型选择单一导致迁移性差**  
   方案：使用 CNN + ViT 异构集成作为替代模型，梯度融合生成扰动，覆盖不同归纳偏置。

3. **熵约束 $H_{\min}$ 设置不当**  
   太高 → 权重均等，失去 LLM 编排的意义；太低 → 退化为单一原语。建议初始值 $\ln(5)/2 \approx 0.8$，根据目标迁移性调整。

---

## 什么时候需要关注这类研究？

| 适合的应用场景 | 不适合的场景 |
|--------------|------------|
| 检测器安全红队评估（授权范围内）| 任何针对具体人物的定向操控 |
| 构建对抗训练数据提升检测鲁棒性 | 无授权绕过平台内容安全系统 |
| 学术研究：对抗迁移机制分析 | 与其他恶意技术结合的实际部署 |
| 评估采购的检测方案真实能力 | |

---

## 与其他迁移攻击方法对比

| 方法 | 查询需求 | 跨架构迁移性 | 语义感知 | 多原语协调 |
|-----|---------|------------|---------|-----------|
| FGSM / PGD | 白盒 | 低 | 无 | 无 |
| MI-FGSM / DI-MI-FGSM | 白盒 | 中-高 | 无 | 无 |
| 传统智能体攻击 | 黑盒（需查询）| 中 | 有限 | 少 |
| **ARMOR++** | **零查询** | **高** | **VLM 空间先验** | **五种原语** |

---

## 我的观点

ARMOR++ 最有价值的贡献不是"提供了更强的攻击工具"，而是**量化了当前部署的检测器有多脆弱**。在 AADD-2025 基准上，即使在对抗防御配置下，攻击成功率仍然可观，这迫使业界重新审视"高准确率 = 可靠安全"的假设。

几个值得关注的开放问题：

1. **VLM 的增益到底来自哪里？** 目前消融实验是理解 VLM 语义先验究竟增加了多少真实收益（相对于随机原语选择）的关键——这部分尚未被完整披露。

2. **攻守对称性**：VLM 帮助攻击方定位显著区域，同样可以帮助防御方在这些区域施加更严格的一致性校验。这是一个真正对称的博弈，防御方不应被动等待。

3. **纵深防御的必要性**：单纯的检测器防线已经不够。C2PA 内容认证标准、溯源水印和多模态一致性校验需要与检测器协同工作——任何单一防线在有充分动机的攻击者面前都是脆弱的。

这类研究提醒我们：在 AI 安全领域，"在学术基准上工作"和"在真实对抗环境中可靠"之间的距离，比我们通常愿意承认的要远得多。