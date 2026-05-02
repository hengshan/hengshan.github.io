---
layout: post-wide
title: "用 Fréchet 距离训练生成模型：FD-loss 原理与实现"
date: 2026-05-02 08:06:54 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2604.28190v1
generated_by: Claude Code CLI
---

## 一句话总结

FD-loss 将评估指标 FID 转化为可微训练目标——通过解耦"分布估计所需的大规模样本"与"梯度计算所需的小 batch"，在无需对抗训练或 teacher 网络的情况下，将多步生成器压缩为单步生成器，实现 ImageNet 256×256 上 0.72 FID。

## 为什么这个问题重要？

生成模型领域长期存在一个隐秘的矛盾：**我们用 FID 排名，却从不训练 FID**。

常见训练目标和 FID 之间存在语义鸿沟：
- 扩散模型优化 noise prediction 的 MSE
- GAN 优化判别器/生成器的对抗损失
- Flow Matching 优化速度场匹配

这些损失下降了，FID 不一定变好；FID 变好了，训练目标也未必能感知。

**为什么不直接优化 FID？**

准确估计两个分布的统计量需要约 50,000 个样本，但 GPU 内存限制下 batch 通常只有 256-1024 个。用 1024 个样本估计 50k 样本才能稳定的协方差矩阵，噪声淹没梯度信号，训练发散。

FD-loss 的思路干净：**估计统计量用 50k，计算梯度用 1k——二者解耦**。

## 背景知识

### Fréchet 距离与 FID

FID（Fréchet Inception Distance）= 在 InceptionV3 特征空间中，计算真实图像分布和生成图像分布之间的 Fréchet 距离（FD）：

$$
\text{FD}(\mathcal{P}, \mathcal{Q}) = \|\mu_{\mathcal{P}} - \mu_{\mathcal{Q}}\|^2 + \text{Tr}\!\left(\Sigma_{\mathcal{P}} + \Sigma_{\mathcal{Q}} - 2\sqrt{\Sigma_{\mathcal{P}} \Sigma_{\mathcal{Q}}}\right)
$$

**第一项**：惩罚分布中心偏移。**第二项**：惩罚分布形状（协方差）差异。

假设两个分布均为高斯，Fréchet 距离是封闭形式——但只在统计量估计准确的前提下有意义。

### 为什么需要大量样本？

特征维度 $d=2048$ 时，协方差矩阵有 $2048 \times 2048$ 个元素。可靠估计需要 $N \gg d$，经验上 $50k \gg 2048$，而 $1k < 2048$，协方差矩阵直接奇异。

## 核心方法

### 直觉解释

核心结构是一个**滚动特征队列**，类似 MoCo 的动量对比队列：

```
每一训练步:
  Real Buffer  [r₁ ... r_{50k}]  ← 持续更新（no grad）
  Fake Buffer  [f₁ ... f_{50k}]  ← 持续更新（detach 存储）

  ┌─────────────────────────────────────────┐
  │  统计估计：用 50k 样本 → 准确           │
  │  梯度计算：流经当前 1k batch → 可行     │
  └─────────────────────────────────────────┘
```

### 梯度推导

FD 对生成器参数 $\theta$ 的梯度（以均值项为例）：

$$
\frac{\partial \text{FD}}{\partial \theta} = \underbrace{2(\mu_g - \mu_r)}_{\text{用大缓冲区准确估计}} \cdot \underbrace{\frac{\partial \mu_g}{\partial \theta}}_{\approx \frac{1}{B}\sum_{\text{batch}} \nabla_\theta}
$$

关键：$2(\mu_g - \mu_r)$ 是梯度**方向**，用 50k 样本精准估计；$\nabla_\theta$ 是梯度**大小**，通过当前 batch 提供。这是一个有偏但方向正确的随机梯度估计。

### Pipeline 概览

```
真实图像 → 编码器(no_grad) ──→ 更新 Real Buffer
生成图像 → 编码器(with_grad) ─→ 更新 Fake Buffer (detach)
                              ↓
                    从缓冲区计算准确统计量
                              ↓
                    构造代理损失（梯度流经当前 batch）
                              ↓
                    更新生成器参数
```

## 实现

### Fréchet 距离计算

```python
import torch
import torch.nn.functional as F

def matrix_sqrt_via_eigh(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """通过特征值分解计算正定矩阵平方根: A = VΛV^T → √A = V√ΛV^T"""
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = eigenvalues.clamp(min=0)  # 截断数值负值，保证稳定性
    sqrt_diag = torch.diag(eigenvalues.sqrt())
    return eigenvectors @ sqrt_diag @ eigenvectors.mT


def frechet_distance(
    mu1: torch.Tensor, sigma1: torch.Tensor,
    mu2: torch.Tensor, sigma2: torch.Tensor,
) -> torch.Tensor:
    """
    计算两个高斯分布的 Fréchet 距离（纯评估用，无梯度）
    FD = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2√(Σ1·Σ2))
    """
    diff = mu1 - mu2
    mean_term = diff @ diff  # 标量

    sqrt_product = matrix_sqrt_via_eigh(sigma1 @ sigma2)
    if torch.is_complex(sqrt_product):
        sqrt_product = sqrt_product.real  # 处理极小数值虚部

    trace_term = (
        torch.trace(sigma1) + torch.trace(sigma2)
        - 2.0 * torch.trace(sqrt_product)
    )
    return mean_term + trace_term
```

### 滚动特征缓冲区

```python
class RollingFeatureBuffer:
    """
    环形特征队列：维护大规模特征以支持稳定的分布统计估计
    所有存储均为 detach 特征，不占用梯度图内存
    """
    def __init__(self, capacity: int, dim: int, device: str = "cuda"):
        self.capacity = capacity
        self.buffer = torch.zeros(capacity, dim, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def push(self, features: torch.Tensor):
        n = features.shape[0]
        end = (self.ptr + n) % self.capacity
        if end > self.ptr:
            self.buffer[self.ptr:end] = features.detach()
        else:  # 环绕写入
            split = self.capacity - self.ptr
            self.buffer[self.ptr:] = features[:split].detach()
            self.buffer[:end] = features[split:].detach()
        self.ptr = end
        self.filled = min(self.filled + n, self.capacity)

    @torch.no_grad()
    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.buffer[:self.filled]
        mu = data.mean(0)
        sigma = torch.cov(data.T)
        return mu, sigma

    @property
    def ready(self) -> bool:
        """至少 10k 样本才认为统计量可靠"""
        return self.filled >= 10000
```

### FD-Loss 核心

```python
class FDLoss(torch.nn.Module):
    """
    Fréchet 距离训练损失
    大缓冲区提供准确梯度方向，当前 batch 提供随机梯度
    注意：这是 FD 梯度的近似实现，均值项为主，协方差项为辅
    """
    def __init__(self, encoder, buffer_size: int = 50000, feature_dim: int = 2048):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)  # 编码器必须冻结！
        self.real_buf = RollingFeatureBuffer(buffer_size, feature_dim)
        self.fake_buf = RollingFeatureBuffer(buffer_size, feature_dim)

    def forward(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        # 真实特征：更新缓冲区，不需要梯度
        with torch.no_grad():
            self.real_buf.push(self.encoder(real_images))

        # 生成特征：保留梯度图，同时 detach 副本入缓冲区
        fake_feat = self.encoder(fake_images)
        self.fake_buf.push(fake_feat)

        if not (self.real_buf.ready and self.fake_buf.ready):
            return fake_feat.mean() * 0  # 预热期：零损失，但保留梯度图

        # 从大缓冲区获取准确统计量（no grad）
        mu_r, sigma_r = self.real_buf.get_stats()
        mu_f, sigma_f = self.fake_buf.get_stats()

        # === 均值项代理损失（主要梯度来源）===
        # ∂||μ_r - μ_f||² / ∂θ 的无偏估计：方向由缓冲区提供，量由 batch 提供
        grad_direction = 2.0 * (mu_f - mu_r).detach()
        mean_loss = (grad_direction * fake_feat.mean(0)).sum()

        # === 协方差项代理损失（辅助）===
        # 最小化当前 batch 协方差与真实协方差的距离
        centered = fake_feat - mu_f.detach()
        local_sigma = (centered.T @ centered) / (fake_feat.shape[0] - 1)
        cov_loss = F.mse_loss(local_sigma, sigma_r.detach())

        return mean_loss + 0.1 * cov_loss
```

### 训练循环

```python
# 初始化（以扩散模型单步 consistency 微调为例）
encoder = load_inception_v3(pretrained=True).eval().cuda()
fd_loss_fn = FDLoss(encoder, buffer_size=50000, feature_dim=2048).cuda()
optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-5)

for step, (real_batch, noise) in enumerate(dataloader):
    fake_batch = generator(noise.cuda())

    # FD-loss 作为主要后训练目标
    loss = fd_loss_fn(real_batch.cuda(), fake_batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
    optimizer.step()

    if step % 1000 == 0:
        # 独立评估：用标准 50k 样本计算真实 FID
        fid = compute_fid_score(generator, real_dataset, n_samples=50000)
        print(f"Step {step}: proxy_loss={loss.item():.4f}, real_FID={fid:.2f}")
```

## FDr^k：多表示空间指标

论文第二个贡献：**FID 会误判视觉质量**。

InceptionV3 是 2014 年为 ImageNet 分类训练的网络。现代生成模型（DiT、Flux 等）在人眼看来更逼真，但 InceptionV3 认为它们更差。

**FDr^k** 的解法：同时在 $k$ 种现代表示空间（CLIP、DINOv2 等）计算 FD，取聚合结果。

```python
class MultiRepFD:
    """
    FDr^k：多表示空间 Fréchet 距离，提供更可靠的生成质量评估
    """
    def __init__(self, device="cuda"):
        self.device = device
        # 不同时代、不同任务导向的编码器
        self.encoders = {
            "inception_v3": load_inception_v3().to(device).eval(),
            "clip_vit_b32": load_clip_encoder("ViT-B/32").to(device).eval(),
            "dinov2_vitb14": torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14"
            ).to(device).eval(),
        }

    @torch.no_grad()
    def evaluate(self, real_loader, fake_loader, n_samples=10000) -> dict[str, float]:
        results = {}
        for name, encoder in self.encoders.items():
            real_feats = extract_features(encoder, real_loader, n_samples, self.device)
            fake_feats = extract_features(encoder, fake_loader, n_samples, self.device)
            mu_r, sigma_r = real_feats.mean(0), torch.cov(real_feats.T)
            mu_f, sigma_f = fake_feats.mean(0), torch.cov(fake_feats.T)
            results[f"FD_{name}"] = frechet_distance(mu_r, sigma_r, mu_f, sigma_f).item()
        return results
    # ... extract_features 省略
```

## 实验结果

### 论文关键数字（ImageNet 256×256）

| 方法 | FID | 步数 | 是否需要 Teacher |
|------|-----|------|----------------|
| 多步扩散（EDM） | ~2.0 | 250步 | - |
| Consistency Distillation | ~3.5 | 1步 | 需要 |
| Adversarial Distillation（ADD） | ~1.5 | 1步 | 需要 |
| **FD-loss 后训练（本文）** | **0.72** | 1步 | 不需要 |

0.72 FID 的单步生成器，是无 teacher 方法中的 SOTA 级结果。

### FID 误判案例

同一个生成器，用不同表示空间计算 FD：

| 表示空间 | FD 值 | 与人类感知一致性 |
|---------|-------|---------------|
| InceptionV3（传统 FID） | 较高（显示变差） | 低 |
| CLIP ViT-B/32 | 较低（显示变好） | 高 |
| DINOv2 | 较低（显示变好） | 高 |

现代表示空间与人类感知更对齐，这也是 FDr^k 的动机。

## 工程实践

### 内存分析

| 组件 | 显存占用 |
|------|---------|
| Real buffer（50k × 2048 float32）| ~400 MB |
| Fake buffer（50k × 2048 float32）| ~400 MB |
| 协方差矩阵（2048 × 2048）| ~16 MB |
| **额外总开销** | **~820 MB** |

对于 80GB A100 可接受；buffer_size 降至 20k 可节省至 ~320 MB。

### 常见坑

**坑 1：忘记冻结编码器**
```python
# 错误：梯度意外流入 encoder
fake_feat = encoder(fake_images)  # 如果 encoder 未冻结，参数被更新

# 修复：初始化时明确冻结
encoder.requires_grad_(False)
```

**坑 2：缓冲区预热期梯度爆炸**
```python
# 错误：缓冲区未就绪时统计量不可靠，方向乱，梯度爆炸
if self.filled < 1000:
    return fd_loss_value  # 方向不可信

# 修复：预热期返回零损失，但保留梯度图结构
return fake_feat.mean() * 0
```

**坑 3：协方差矩阵奇异**
```python
# 修复：加正则化
sigma = torch.cov(features.T)
sigma += 1e-6 * torch.eye(sigma.shape[0], device=sigma.device)
```

**坑 4：编码器输出未归一化导致量纲不匹配**
```python
# CLIP 等编码器输出通常需要 L2 归一化
feat = encoder(images)
feat = F.normalize(feat, dim=-1)  # 统一量纲，协方差估计更稳定
```

### 硬件需求

- **最低**：单张 A100 40GB，buffer_size=20k，batch=256
- **推荐**：单张 A100 80GB，buffer_size=50k，batch=512
- **后训练场景**：通常 20k-100k 步收敛，比从头训练便宜 10× 以上

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 后训练已有生成器以提升质量 | 从零开始训练生成模型 |
| 将多步生成器压缩为单步 | 真实数据集小于 50k 样本 |
| 无法获取 teacher 的蒸馏场景 | 显存极其紧张（< 24GB） |
| Class-conditional 生成（ImageNet 等） | Text-conditional 生成（需额外改造） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| GAN 对抗训练 | 历史验证充分 | 训练不稳定，需判别器 | 专用架构 |
| 知识蒸馏 | 保留 teacher 特性 | 需要高质量 teacher | 有成熟 teacher 时 |
| Consistency Distillation | 理论优雅 | 需要 teacher，ODE solver | 扩散模型加速 |
| **FD-loss（本文）** | 直接优化评估指标，无需 teacher | 显存开销大，实现有技巧 | 后训练、单步生成 |

## 我的观点

**这个工作解决了一个"明明重要但被忽视"的问题**：评估指标和训练目标的对齐。方法思路简洁——解耦估计规模和 batch 大小——但需要工程才能落地，协方差矩阵的数值稳定性不是小事。

**0.72 FID 是真实进步**。此前单步生成通常依赖复杂的 consistency distillation 或 adversarial training，FD-loss 提供了更干净的路径，且不需要 teacher 网络（意味着更低的运维复杂度）。

**两个开放问题值得关注：**

1. **Text-conditional 扩展**：论文主要验证 class-conditional ImageNet。文生图场景下真实分布是 text-conditioned 的，如何维护 conditional FD 的大规模队列，还未有清晰答案。

2. **FDr^k 的权重设计**：多表示空间聚合时怎么加权？人脸、风景、艺术画最适合的表示空间可能不同，这需要任务感知的权重学习。

**离产品级落地**：额外 ~800MB 显存和精心的数值实现是主要门槛。对于有 A100 的团队，这已经是**可以直接在现有生成器上尝试的后训练方案**，值得工程投入。

---

*论文链接：[Representation Fréchet Loss for Visual Generation](https://arxiv.org/abs/2604.28190v1)*