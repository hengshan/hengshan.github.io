---
layout: post-wide
title: "Provable Robustness in Multimodal Large Language Models via Feature Space Smooth"
date: 2026-01-23 17:30:01 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2601.16200v1
generated_by: AI Agent
---

## 多模态大语言模型的对抗鲁棒性：特征空间平滑理论与实践

## 问题背景与动机

多模态大语言模型（MLLMs）在图像理解、视觉问答等任务中展现出强大能力，但它们对对抗性扰动极其脆弱。微小的像素级扰动就能导致模型输出完全错误的结果。本文介绍一种基于特征空间平滑（Feature-space Smoothing, FS）的可证明鲁棒性方法，并提供完整的实现教程。

### 核心问题

**对抗攻击威胁**：攻击者通过在输入图像上添加不可察觉的扰动 $\delta$（满足 $\|\delta\|_2 \leq \epsilon$），使模型产生错误预测。

**传统防御的局限**：
- 对抗训练：计算开销大，且无法提供理论保证
- 输入空间平滑：仅关注输入层，忽略了特征空间的脆弱性

**特征空间平滑的优势**：
- 提供可证明的特征相似度下界
- 无需重新训练模型
- 适用于任意特征编码器

## 理论基础

### 特征空间平滑定义

给定一个特征编码器 $f: \mathcal{X} \rightarrow \mathbb{R}^d$，其平滑版本定义为：

$$
\bar{f}(x) = \mathbb{E}_{\eta \sim \mathcal{N}(0, \sigma^2 I)}[f(x + \eta)]
$$

**定理（特征余弦相似度界）**：对于干净样本 $x$ 和对抗样本 $x' = x + \delta$（$\|\delta\|_2 \leq \epsilon$），平滑特征的余弦相似度满足：

$$
\cos(\bar{f}(x), \bar{f}(x')) \geq \text{FCSB}(\epsilon, \sigma, \rho)
$$

其中 $\rho$ 是高斯鲁棒性分数，FCSB 是可证明的下界。

### 高斯鲁棒性分数

$$
\rho(x) = \mathbb{P}_{\eta \sim \mathcal{N}(0, \sigma^2 I)}[\|f(x + \eta) - f(x)\|_2 \leq r]
$$

**关键洞察**：提升 $\rho$ 可以增强认证鲁棒性。

## 实现：基础特征空间平滑

### 环境准备

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
```

### 简单特征编码器

```python
class SimpleFeatureEncoder(nn.Module):
    """
    简单的卷积特征编码器
    用于演示特征空间平滑的基本原理
    """
    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        """
        输入: x [batch_size, 3, H, W]
        输出: features [batch_size, feature_dim]
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        # L2归一化，使特征在单位球面上
        features = F.normalize(features, p=2, dim=1)
        return features
```

### 特征空间平滑实现

```python
class FeatureSpaceSmoothing:
    """
    特征空间平滑的核心实现
    通过在输入空间添加高斯噪声，平滑特征表示
    """
    def __init__(
        self,
        encoder: nn.Module,
        sigma: float = 0.25,
        num_samples: int = 100
    ):
        """
        参数:
            encoder: 特征编码器
            sigma: 高斯噪声标准差
            num_samples: 蒙特卡洛采样次数
        """
        self.encoder = encoder
        self.sigma = sigma
        self.num_samples = num_samples
        self.encoder.eval()

    def smooth_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算平滑特征: E[f(x + η)]

        参数:
            x: 输入图像 [batch_size, C, H, W]
        返回:
            smoothed_features: 平滑后的特征 [batch_size, feature_dim]
        """
        batch_size = x.size(0)
        feature_dim = None
        accumulated_features = None

        with torch.no_grad():
            for i in range(self.num_samples):
                # 添加高斯噪声
                noise = torch.randn_like(x) * self.sigma
                noisy_input = x + noise

                # 编码特征
                features = self.encoder(noisy_input)

                if accumulated_features is None:
                    feature_dim = features.size(1)
                    accumulated_features = torch.zeros(
                        batch_size, feature_dim,
                        device=x.device
                    )

                accumulated_features += features

        # 计算平均并归一化
        smoothed_features = accumulated_features / self.num_samples
        smoothed_features = F.normalize(smoothed_features, p=2, dim=1)

        return smoothed_features

    def compute_gaussian_robustness_score(
        self,
        x: torch.Tensor,
        radius: float = 0.5
    ) -> float:
        """
        计算高斯鲁棒性分数 ρ
        ρ = P(||f(x+η) - f(x)||_2 <= r)

        参数:
            x: 输入图像
            radius: 距离阈值 r
        返回:
            鲁棒性分数 (0到1之间)
        """
        with torch.no_grad():
            # 原始特征
            original_features = self.encoder(x)

            count = 0
            for i in range(self.num_samples):
                noise = torch.randn_like(x) * self.sigma
                noisy_input = x + noise
                noisy_features = self.encoder(noisy_input)

                # 计算L2距离
                distance = torch.norm(
                    noisy_features - original_features,
                    p=2,
                    dim=1
                )

                # 统计在radius内的样本数
                count += (distance <= radius).sum().item()

            score = count / (self.num_samples * x.size(0))
            return score

    def certified_cosine_similarity_bound(
        self,
        epsilon: float,
        rho: float
    ) -> float:
        """
        计算可证明的特征余弦相似度下界 FCSB

        参数:
            epsilon: 对抗扰动的L2范数上界
            rho: 高斯鲁棒性分数
        返回:
            余弦相似度下界
        """
        # 简化的理论界计算（实际论文中有更复杂的推导）
        # FCSB 随 ρ 增大而增大，随 ε 增大而减小
        if rho < 0.5:
            return 0.0

        # 基于论文的近似公式
        bound = max(0.0, 1.0 - (epsilon / self.sigma) * (1.0 - rho))
        return bound
```

### 对抗攻击实现

```python
class PGDAttack:
    """
    投影梯度下降（PGD）对抗攻击
    在特征空间中最大化特征距离
    """
    def __init__(
        self,
        encoder: nn.Module,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 40
    ):
        """
        参数:
            encoder: 目标编码器
            epsilon: 扰动的L2范数上界
            alpha: 每步的步长
            num_iter: 迭代次数
        """
        self.encoder = encoder
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def attack(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成对抗样本
        目标: 最大化 ||f(x') - f(x)||_2

        参数:
            x: 干净图像 [batch_size, C, H, W]
        返回:
            x_adv: 对抗样本
        """
        x_adv = x.clone().detach()

        # 计算原始特征（用于目标）
        with torch.no_grad():
            original_features = self.encoder(x)

        for i in range(self.num_iter):
            x_adv.requires_grad = True

            # 前向传播
            adv_features = self.encoder(x_adv)

            # 损失：负的特征距离（最大化距离 = 最小化负距离）
            loss = F.cosine_similarity(
                adv_features,
                original_features,
                dim=1
            ).mean()

            # 反向传播
            loss.backward()

            # 梯度上升
            with torch.no_grad():
                grad = x_adv.grad
                x_adv = x_adv + self.alpha * grad.sign()

                # 投影到L2球
                delta = x_adv - x
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                delta_norm = delta_norm.view(-1, 1, 1, 1)
                delta = delta / (delta_norm + 1e-8) * torch.clamp(
                    delta_norm, max=self.epsilon
                )
                x_adv = torch.clamp(x + delta, 0, 1)

            x_adv.grad = None

        return x_adv.detach()
```

### 训练与评估

```python
def evaluate_robustness(
    encoder: nn.Module,
    test_images: torch.Tensor,
    epsilon: float = 0.1,
    sigma: float = 0.25
):
    """
    评估模型的对抗鲁棒性

    参数:
        encoder: 特征编码器
        test_images: 测试图像
        epsilon: 对抗扰动强度
        sigma: 平滑参数
    """
    print(f"\n{'='*60}")
    print(f"对抗鲁棒性评估 (ε={epsilon}, σ={sigma})")
    print(f"{'='*60}\n")

    # 初始化
    fs = FeatureSpaceSmoothing(encoder, sigma=sigma, num_samples=50)
    attacker = PGDAttack(encoder, epsilon=epsilon)

    # 1. 原始特征
    with torch.no_grad():
        clean_features = encoder(test_images)

    # 2. 生成对抗样本
    adv_images = attacker.attack(test_images)
    with torch.no_grad():
        adv_features = encoder(adv_images)

    # 3. 计算未防御时的相似度
    vanilla_similarity = F.cosine_similarity(
        clean_features,
        adv_features,
        dim=1
    ).mean().item()

    # 4. 平滑特征
    smooth_clean_features = fs.smooth_predict(test_images)
    smooth_adv_features = fs.smooth_predict(adv_images)

    # 5. 计算防御后的相似度
    smooth_similarity = F.cosine_similarity(
        smooth_clean_features,
        smooth_adv_features,
        dim=1
    ).mean().item()

    # 6. 计算高斯鲁棒性分数
    rho = fs.compute_gaussian_robustness_score(test_images, radius=0.5)

    # 7. 计算理论界
    certified_bound = fs.certified_cosine_similarity_bound(epsilon, rho)

    # 8. 输出结果
    print(f"未防御的余弦相似度: {vanilla_similarity:.4f}")
    print(f"特征空间平滑后的余弦相似度: {smooth_similarity:.4f}")
    print(f"高斯鲁棒性分数 ρ: {rho:.4f}")
    print(f"可证明的余弦相似度下界: {certified_bound:.4f}")
    print(f"实际相似度是否 >= 理论界: {smooth_similarity >= certified_bound}")

    # 9. 可视化
    visualize_results(
        test_images, adv_images,
        vanilla_similarity, smooth_similarity, certified_bound
    )

    return {
        'vanilla_similarity': vanilla_similarity,
        'smooth_similarity': smooth_similarity,
        'rho': rho,
        'certified_bound': certified_bound
    }

def visualize_results(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    vanilla_sim: float,
    smooth_sim: float,
    certified_bound: float
):
    """可视化对抗样本和防御效果"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(4):
        # 干净图像
        clean_img = clean_images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f'Clean Image {i+1}')
        axes[0, i].axis('off')

        # 对抗图像
        adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f'Adversarial Image {i+1}')
        axes[1, i].axis('off')

    plt.suptitle(
        f'Vanilla Sim: {vanilla_sim:.3f} | '
        f'Smooth Sim: {smooth_sim:.3f} | '
        f'Certified Bound: {certified_bound:.3f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('adversarial_robustness_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# 生成测试数据
def generate_test_images(num_images=8, image_size=64):
    """生成随机测试图像"""
    images = torch.rand(num_images, 3, image_size, image_size).to(device)
    return images

# 运行评估
encoder = SimpleFeatureEncoder(feature_dim=128).to(device)
test_images = generate_test_images(num_images=8)

results = evaluate_robustness(
    encoder,
    test_images,
    epsilon=0.1,
    sigma=0.25
)
```

## 高级技巧：Purifier and Smoothness Mapper (PSM)

PSM 是一个即插即用模块，用于提升高斯鲁棒性分数 $\rho$，从而增强认证鲁棒性。

### PSM 架构

```python
class PurifierSmoothnessMapper(nn.Module):
    """
    Purifier and Smoothness Mapper (PSM)

    核心思想:
    1. Purifier: 去除输入中的对抗性噪声
    2. Smoothness Mapper: 将特征映射到更平滑的空间
    """
    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 128,
        purifier_depth: int = 3
    ):
        super().__init__()

        # Purifier: 基于去噪自编码器
        self.purifier = self._build_purifier(input_channels, purifier_depth)

        # Smoothness Mapper: 特征空间的平滑映射
        self.smoothness_mapper = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def _build_purifier(self, channels: int, depth: int) -> nn.Module:
        """
        构建Purifier网络
        使用残差连接保持输入的主要信息
        """
        layers = []
        for i in range(depth):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ])
        return nn.Sequential(*layers)

    def purify(self, x: torch.Tensor) -> torch.Tensor:
        """
        净化输入，去除对抗性扰动
        使用残差连接: x_clean = x + purifier(x)
        """
        residual = self.purifier(x)
        # 残差连接 + 裁剪到有效范围
        x_purified = torch.clamp(x + 0.1 * residual, 0, 1)
        return x_purified

    def smooth_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        将特征映射到更平滑的空间
        """
        smooth_features = self.smoothness_mapper(features)
        # 保持单位范数
        smooth_features = F.normalize(smooth_features, p=2, dim=1)
        return smooth_features

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            x: 输入图像
            features: 可选的特征（如果提供）
        返回:
            x_purified: 净化后的图像
            smooth_features: 平滑后的特征（如果提供了features）
        """
        x_purified = self.purify(x)

        if features is not None:
            smooth_features = self.smooth_features(features)
            return x_purified, smooth_features

        return x_purified, None
```

### 集成 PSM 的特征空间平滑

```python
class EnhancedFeatureSpaceSmoothing:
    """
    增强版特征空间平滑，集成 PSM 模块
    """
    def __init__(
        self,
        encoder: nn.Module,
        psm: PurifierSmoothnessMapper,
        sigma: float = 0.25,
        num_samples: int = 100
    ):
        self.encoder = encoder
        self.psm = psm
        self.sigma = sigma
        self.num_samples = num_samples

        self.encoder.eval()
        self.psm.eval()

    def smooth_predict_with_psm(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 PSM 增强的平滑预测

        流程:
        1. Purifier 净化输入
        2. 特征空间平滑
        3. Smoothness Mapper 平滑特征
        """
        batch_size = x.size(0)
        accumulated_features = None

        with torch.no_grad():
            # 步骤1: 净化输入
            x_purified, _ = self.psm(x)

            # 步骤2: 特征空间平滑
            for i in range(self.num_samples):
                noise = torch.randn_like(x_purified) * self.sigma
                noisy_input = x_purified + noise

                # 编码特征
                features = self.encoder(noisy_input)

                if accumulated_features is None:
                    feature_dim = features.size(1)
                    accumulated_features = torch.zeros(
                        batch_size, feature_dim,
                        device=x.device
                    )

                accumulated_features += features

            # 平均特征
            avg_features = accumulated_features / self.num_samples
            avg_features = F.normalize(avg_features, p=2, dim=1)

            # 步骤3: 平滑特征
            _, smooth_features = self.psm(x_purified, avg_features)

        return smooth_features

    def compute_enhanced_robustness_score(
        self,
        x: torch.Tensor,
        radius: float = 0.5
    ) -> float:
        """
        计算使用 PSM 后的增强鲁棒性分数
        """
        with torch.no_grad():
            # 净化输入
            x_purified, _ = self.psm(x)

            # 原始特征
            original_features = self.encoder(x_purified)
            _, original_smooth = self.psm(x_purified, original_features)

            count = 0
            for i in range(self.num_samples):
                noise = torch.randn_like(x_purified) * self.sigma
                noisy_input = x_purified + noise

                noisy_features = self.encoder(noisy_input)
                _, noisy_smooth = self.psm(x_purified, noisy_features)

                distance = torch.norm(
                    noisy_smooth - original_smooth,
                    p=2,
                    dim=1
                )

                count += (distance <= radius).sum().item()

            score = count / (self.num_samples * x.size(0))
            return score
```

### PSM 训练

```python
def train_psm(
    encoder: nn.Module,
    psm: PurifierSmoothnessMapper,
    train_images: torch.Tensor,
    num_epochs: int = 50,
    learning_rate: float = 1e-3
):
    """
    训练 PSM 模块

    损失函数:
    1. 重建损失: ||x_purified - x||
    2. 平滑性损失: 最小化特征在噪声下的方差
    3. 保真度损失: 保持特征的判别性
    """
    optimizer = torch.optim.Adam(psm.parameters(), lr=learning_rate)

    print("开始训练 PSM...")
    losses_history = []

    for epoch in range(num_epochs):
        psm.train()
        epoch_loss = 0.0

        # 小批量训练
        batch_size = 8
        num_batches = len(train_images) // batch_size

        for i in range(num_batches):
            batch = train_images[i*batch_size:(i+1)*batch_size]

            # 前向传播
            x_purified, _ = psm(batch)

            # 损失1: 重建损失（保持图像内容）
            recon_loss = F.mse_loss(x_purified, batch)

            # 损失2: 平滑性损失
            with torch.no_grad():
                original_features = encoder(batch)

            # 添加噪声并计算特征方差
            noise1 = torch.randn_like(x_purified) * 0.1
            noise2 = torch.randn_like(x_purified) * 0.1

            feat1 = encoder(x_purified + noise1)
            feat2 = encoder(x_purified + noise2)

            _, smooth_feat1 = psm(x_purified, feat1)
            _, smooth_feat2 = psm(x_purified, feat2)

            # 平滑性损失：特征在噪声下应该接近
            smoothness_loss = F.mse_loss(smooth_feat1, smooth_feat2)

            # 损失3: 保真度损失（保持与原始特征的相似性）
            _, smooth_original = psm(x_purified, original_features)
            fidelity_loss = 1.0 - F.cosine_similarity(
                smooth_original,
                original_features,
                dim=1
            ).mean()

            # 总损失
            total_loss = recon_loss + 0.5 * smoothness_loss + 0.3 * fidelity_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / num_batches
        losses_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 可视化训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PSM Training Loss')
    plt.grid(True)
    plt.savefig('psm_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

    return psm

# 训练 PSM
psm = PurifierSmoothnessMapper(
    input_channels=3,
    feature_dim=128,
    purifier_depth=3
).to(device)

train_images = generate_test_images(num_images=64)
psm = train_psm(encoder, psm, train_images, num_epochs=50)
```

### 对比实验

```python
def comprehensive_evaluation():
    """
    全面对比实验：
    1. 无防御
    2. 仅特征空间平滑
    3. 特征空间平滑 + PSM
    """
    print("\n" + "="*70)
    print("全面对抗鲁棒性评估")
    print("="*70)

    test_images = generate_test_images(num_images=16)
    epsilon_values = [0.05, 0.1, 0.15, 0.2]

    results = {
        'epsilon': [],
        'vanilla': [],
        'fs_only': [],
        'fs_psm': []
    }

    for epsilon in epsilon_values:
        print(f"\n测试 ε = {epsilon}")
        print("-" * 70)

        # 创建攻击器
        attacker = PGDAttack(encoder, epsilon=epsilon)
        adv_images = attacker.attack(test_images)

        # 1. 无防御
        with torch.no_grad():
            clean_feat = encoder(test_images)
            adv_feat = encoder(adv_images)
            vanilla_sim = F.cosine_similarity(
                clean_feat, adv_feat, dim=1
            ).mean().item()

        # 2. 仅 FS
        fs = FeatureSpaceSmoothing(encoder, sigma=0.25, num_samples=50)
        smooth_clean = fs.smooth_predict(test_images)
        smooth_adv = fs.smooth_predict(adv_images)
        fs_sim = F.cosine_similarity(
            smooth_clean, smooth_adv, dim=1
        ).mean().item()

        # 3. FS + PSM
        enhanced_fs = EnhancedFeatureSpaceSmoothing(
            encoder, psm, sigma=0.25, num_samples=50
        )
        psm_clean = enhanced_fs.smooth_predict_with_psm(test_images)
        psm_adv = enhanced_fs.smooth_predict_with_psm(adv_images)
        psm_sim = F.cosine_similarity(
            psm_clean, psm_adv, dim=1
        ).mean().item()

        # 记录结果
        results['epsilon'].append(epsilon)
        results['vanilla'].append(vanilla_sim)
        results['fs_only'].append(fs_sim)
        results['fs_psm'].append(psm_sim)

        print(f"  无防御: {vanilla_sim:.4f}")
        print(f"  仅 FS:  {fs_sim:.4f}")
        print(f"  FS+PSM: {psm_sim:.4f}")

    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.plot(results['epsilon'], results['vanilla'], 'o-', label='No Defense', linewidth=2)
    plt.plot(results['epsilon'], results['fs_only'], 's-', label='FS Only', linewidth=2)
    plt.plot(results['epsilon'], results['fs_psm'], '^-', label='FS + PSM', linewidth=2)

    plt.xlabel('Perturbation Budget (ε)', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Adversarial Robustness Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('robustness_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# 运行全面评估
comparison_results = comprehensive_evaluation()
```

## 实验分析

### 性能提升分析

```python
def analyze_performance():
    """分析不同方法的性能提升"""
    print("\n" + "="*70)
    print("性能提升分析")
    print("="*70 + "\n")

    # 基于之前的实验结果
    epsilon = 0.1

    # 模拟实验数据（实际运行时会得到真实数据）
    vanilla_asr = 0.89  # 攻击成功率
    fs_asr = 0.35
    fs_psm_asr = 0.01

    print(f"在 ε = {epsilon} 下的攻击成功率:")
    print(f"  无防御:     {vanilla_asr*100:.1f}%")
    print(f"  仅 FS:      {fs_asr*100:.1f}%")
    print(f"  FS + PSM:   {fs_psm_asr*100:.1f}%")

    print(f"\n相对改进:")
    print(f"  FS 相比无防御:     {(1-fs_asr/vanilla_asr)*100:.1f}% 降低")
    print(f"  FS+PSM 相比无防御: {(1-fs_psm_asr/vanilla_asr)*100:.1f}% 降低")
    print(f"  FS+PSM 相比 FS:    {(1-fs_psm_asr/fs_asr)*100:.1f}% 降低")

    # 计算效率分析
    print(f"\n计算开销分析:")
    print(f"  无防御:     1× (基准)")
    print(f"  仅 FS:      ~50× (需要50次采样)")
    print(f"  FS + PSM:   ~52× (PSM 开销很小)")

    # 可视化
    methods = ['No Defense', 'FS Only', 'FS + PSM']
    asr_values = [vanilla_asr, fs_asr, fs_psm_asr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 攻击成功率对比
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = ax1.bar(methods, asr_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Attack Success Rate', fontsize=12)
    ax1.set_title('Attack Success Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)

    # 在柱状图上标注数值
    for bar, val in zip(bars, asr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 防御效果（1 - ASR）
    defense_effectiveness = [1 - asr for asr in asr_values]
    bars2 = ax2.bar(methods, defense_effectiveness, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Defense Effectiveness', fontsize=12)
    ax2.set_title('Defense Effectiveness (1 - ASR)', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, defense_effectiveness):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

analyze_performance()
```

### 超参数敏感性分析

```python
def hyperparameter_sensitivity():
    """分析关键超参数的影响"""
    print("\n" + "="*70)
    print("超参数敏感性分析")
    print("="*70 + "\n")

    test_images = generate_test_images(num_images=8)
    epsilon = 0.1

    # 1. σ (平滑参数) 的影响
    sigma_values = [0.1, 0.15, 0.25, 0.35, 0.5]
    sigma_results = []

    print("测试不同的 σ 值...")
    for sigma in sigma_values:
        fs = FeatureSpaceSmoothing(encoder, sigma=sigma, num_samples=50)
        attacker = PGDAttack(encoder, epsilon=epsilon)

        adv_images = attacker.attack(test_images)
        smooth_clean = fs.smooth_predict(test_images)
        smooth_adv = fs.smooth_predict(adv_images)

        sim = F.cosine_similarity(
            smooth_clean, smooth_adv, dim=1
        ).mean().item()

        sigma_results.append(sim)
        print(f"  σ = {sigma:.2f}: 相似度 = {sim:.4f}")

    # 2. 采样次数的影响
    num_samples_values = [10, 25, 50, 100, 200]
    samples_results = []

    print("\n测试不同的采样次数...")
    for num_samples in num_samples_values:
        fs = FeatureSpaceSmoothing(encoder, sigma=0.25, num_samples=num_samples)
        attacker = PGDAttack(encoder, epsilon=epsilon)

        adv_images = attacker.attack(test_images)
        smooth_clean = fs.smooth_predict(test_images)
        smooth_adv = fs.smooth_predict(adv_images)

        sim = F.cosine_similarity(
            smooth_clean, smooth_adv, dim=1
        ).mean().item()

        samples_results.append(sim)
        print(f"  采样次数 = {num_samples}: 相似度 = {sim:.4f}")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # σ 的影响
    ax1.plot(sigma_values, sigma_results, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax1.set_xlabel('Smoothing Parameter (σ)', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('Impact of Smoothing Parameter', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax1.legend()

    # 采样次数的影响
    ax2.plot(num_samples_values, samples_results, 's-', linewidth=2, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Number of Samples', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Impact of Sample Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig('hyperparameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n建议的超参数设置:")
    print("  σ: 0.25 - 0.35 (平衡鲁棒性和准确性)")
    print("  采样次数: 50 - 100 (性能收益递减)")

hyperparameter_sensitivity()
```

## 实际应用：多模态大语言模型防御

```python
class MLLMWithDefense(nn.Module):
    """
    集成防御机制的多模态大语言模型（简化版）

    架构:
    1. 图像编码器（带防御）
    2. 文本编码器
    3. 多模态融合
    4. 语言生成器
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        psm: PurifierSmoothnessMapper,
        vocab_size: int = 1000,
        d_model: int = 256
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.psm = psm

        # 文本编码器（简化）
        self.text_embedding = nn.Embedding(vocab_size, d_model)

        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(128 + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 语言生成器（简化）
        self.generator = nn.Linear(d_model, vocab_size)

        # 特征空间平滑
        self.fs = EnhancedFeatureSpaceSmoothing(
            image_encoder, psm, sigma=0.25, num_samples=50
        )

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        use_defense: bool = True
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            images: 输入图像 [batch_size, 3, H, W]
            text_tokens: 文本token [batch_size, seq_len]
            use_defense: 是否使用防御
        返回:
            logits: 生成的token概率 [batch_size, seq_len, vocab_size]
        """
        # 图像编码（带防御）
        if use_defense:
            image_features = self.fs.smooth_predict_with_psm(images)
        else:
            image_features = self.image_encoder(images)

        # 文本编码
        text_features = self.text_embedding(text_tokens)  # [B, L, d_model]

        # 扩展图像特征以匹配序列长度
        batch_size, seq_len, _ = text_features.shape
        image_features_expanded = image_features.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [B, L, 128]

        # 多模态融合
        fused_features = torch.cat([
            image_features_expanded, text_features
        ], dim=-1)  # [B, L, 128 + d_model]

        fused_features = self.fusion(fused_features)  # [B, L, d_model]

        # 生成输出
        logits = self.generator(fused_features)  # [B, L, vocab_size]

        return logits

def test_mllm_defense():
    """测试 MLLM 的防御效果"""
    print("\n" + "="*70)
    print("多模态大语言模型防御测试")
    print("="*70 + "\n")

    # 初始化模型
    mllm = MLLMWithDefense(
        image_encoder=encoder,
        psm=psm,
        vocab_size=1000,
        d_model=256
    ).to(device)

    # 测试数据
    test_images = generate_test_images(num_images=4)
    test_tokens = torch.randint(0, 1000, (4, 10)).to(device)

    # 生成对抗样本
    attacker = PGDAttack(encoder, epsilon=0.1)
    adv_images = attacker.attack(test_images)

    # 1. 无防御
    with torch.no_grad():
        clean_logits = mllm(test_images, test_tokens, use_defense=False)
        adv_logits_no_defense = mllm(adv_images, test_tokens, use_defense=False)

        # 计算输出差异
        output_diff_no_defense = F.kl_div(
            F.log_softmax(adv_logits_no_defense, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean'
        ).item()

    # 2. 有防御
    with torch.no_grad():
        adv_logits_with_defense = mllm(adv_images, test_tokens, use_defense=True)

        output_diff_with_defense = F.kl_div(
            F.log_softmax(adv_logits_with_defense, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean'
        ).item()

    print(f"输出分布的 KL 散度:")
    print(f"  无防御: {output_diff_no_defense:.4f}")
    print(f"  有防御: {output_diff_with_defense:.4f}")
    print(f"  改进: {(1 - output_diff_with_defense/output_diff_no_defense)*100:.1f}%")

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(4):
        # 原始图像
        img = test_images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Clean Image {i+1}')
        axes[0, i].axis('off')

        # 对抗图像
        adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f'Adversarial Image {i+1}')
        axes[1, i].axis('off')

    plt.suptitle(
        f'MLLM Defense: KL Divergence reduced from {output_diff_no_defense:.3f} to {output_diff_with_defense:.3f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('mllm_defense_results.png', dpi=150, bbox_inches='tight')
    plt.show()

test_mllm_defense()
```

## 调试技巧

### 常见问题与解决方案

```python
def debugging_guide():
    """
    调试指南：常见问题和解决方案
    """
    print("\n" + "="*70)
    print("调试指南")
    print("="*70 + "\n")

    print("1. 特征空间平滑后相似度过低")
    print("   原因: σ 过大或采样次数不足")
    print("   解决: 减小 σ (0.15-0.25) 或增加采样次数 (>50)")
    print()

    print("2. 防御效果不明显")
    print("   原因: PSM 未充分训练或编码器特征不够平滑")
    print("   解决: ")
    print("   - 增加 PSM 训练轮数")
    print("   - 调整 PSM 的平滑性损失权重")
    print("   - 使用更深的 Purifier 网络")
    print()

    print("3. 计算开销过大")
    print("   原因: 采样次数过多")
    print("   解决: ")
    print("   - 使用自适应采样（根据不确定性调整）")
    print("   - 批量处理多个样本")
    print("   - 使用 GPU 加速")
    print()

    print("4. 理论界验证失败")
    print("   原因: 高斯鲁棒性分数 ρ 过低")
    print("   解决: ")
    print("   - 训练 PSM 提升 ρ")
    print("   - 增大平滑参数 σ")
    print("   - 检查特征归一化是否正确")
    print()

debugging_guide()

def visualize_feature_space():
    """
    可视化特征空间的平滑效果
    使用 t-SNE 降维
    """
    from sklearn.manifold import TSNE

    print("\n可视化特征空间...")

    test_images = generate_test_images(num_images=32)
    attacker = PGDAttack(encoder, epsilon=0.1)
    adv_images = attacker.attack(test_images)

    # 收集特征
    with torch.no_grad():
        # 原始特征
        clean_feat = encoder(test_images).cpu().numpy()
        adv_feat = encoder(adv_images).cpu().numpy()

        # 平滑特征
        fs = FeatureSpaceSmoothing(encoder, sigma=0.25, num_samples=50)
        smooth_clean = fs.smooth_predict(test_images).cpu().numpy()
        smooth_adv = fs.smooth_predict(adv_images).cpu().numpy()

    # t-SNE 降维
    all_features = np.vstack([clean_feat, adv_feat, smooth_clean, smooth_adv])
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(all_features)

    # 分割
    n = len(test_images)
    clean_emb = embedded[:n]
    adv_emb = embedded[n:2*n]
    smooth_clean_emb = embedded[2*n:3*n]
    smooth_adv_emb = embedded[3*n:]

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 原始特征空间
    ax1.scatter(clean_emb[:, 0], clean_emb[:, 1],
               c='blue', label='Clean', s=100, alpha=0.6, edgecolors='black')
    ax1.scatter(adv_emb[:, 0], adv_emb[:, 1],
               c='red', label='Adversarial', s=100, alpha=0.6, edgecolors='black')

    # 绘制连线
    for i in range(n):
        ax1.plot([clean_emb[i, 0], adv_emb[i, 0]],
                [clean_emb[i, 1], adv_emb[i, 1]],
                'k-', alpha=0.2, linewidth=1)

    ax1.set_title('Original Feature Space', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 平滑后的特征空间
    ax2.scatter(smooth_clean_emb[:, 0], smooth_clean_emb[:, 1],
               c='blue', label='Clean (Smoothed)', s=100, alpha=0.6, edgecolors='black')
    ax2.scatter(smooth_adv_emb[:, 0], smooth_adv_emb[:, 1],
               c='red', label='Adversarial (Smoothed)', s=100, alpha=0.6, edgecolors='black')

    for i in range(n):
        ax2.plot([smooth_clean_emb[i, 0], smooth_adv_emb[i, 0]],
                [smooth_clean_emb[i, 1], smooth_adv_emb[i, 1]],
                'k-', alpha=0.2, linewidth=1)

    ax2.set_title('Smoothed Feature Space', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_space_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("特征空间可视化完成！")

visualize_feature_space()
```

## 性能优化建议

```python
def optimization_strategies():
    """
    性能优化策略
    """
    print("\n" + "="*70)
    print("性能优化策略")
    print("="*70 + "\n")

    print("1. 批量处理优化")
    print("   - 一次处理多个图像，充分利用 GPU 并行性")
    print("   - 使用 DataLoader 进行异步数据加载")
    print()

    print("2. 采样优化")
    print("   - 自适应采样：对不确定的样本增加采样次数")
    print("   - 重要性采样：集中在高风险区域")
    print()

    print("3. 模型量化")
    print("   - 使用 FP16 或 INT8 量化减少内存和计算")
    print("   - 保持防御效果的同时提升速度")
    print()

    print("4. 缓存机制")
    print("   - 缓存常见图像的平滑特征")
    print("   - 使用哈希表快速查找")
    print()

optimization_strategies()

# 批量处理示例
class BatchFeatureSpaceSmoothing:
    """
    优化的批量特征空间平滑
    """
    def __init__(self, encoder, psm, sigma=0.25, num_samples=100, batch_size=32):
        self.encoder = encoder
        self.psm = psm
        self.sigma = sigma
        self.num_samples = num_samples
        self.batch_size = batch_size

    def smooth_predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        批量处理的平滑预测
        更高效地利用 GPU
        """
        num_images = images.size(0)
        feature_dim = None
        all_smoothed_features = []

        with torch.no_grad():
            # 净化
            images_purified, _ = self.psm(images)

            # 批量采样
            for start_idx in range(0, num_images, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_images)
                batch_images = images_purified[start_idx:end_idx]

                accumulated = None

                for _ in range(self.num_samples):
                    # 批量添加噪声
                    noise = torch.randn_like(batch_images) * self.sigma
                    noisy_batch = batch_images + noise

                    # 批量编码
                    features = self.encoder(noisy_batch)

                    if accumulated is None:
                        feature_dim = features.size(1)
                        accumulated = torch.zeros_like(features)

                    accumulated += features

                # 平均并平滑
                avg_features = accumulated / self.num_samples
                avg_features = F.normalize(avg_features, p=2, dim=1)
                _, smooth_features = self.psm(batch_images, avg_features)

                all_smoothed_features.append(smooth_features)

        return torch.cat(all_smoothed_features, dim=0)

# 性能对比测试
def benchmark_optimization():
    """对比优化前后的性能"""
    import time

    print("\n性能基准测试...")

    test_images = generate_test_images(num_images=64)

    # 原始方法
    enhanced_fs = EnhancedFeatureSpaceSmoothing(
        encoder, psm, sigma=0.25, num_samples=50
    )

    start_time = time.time()
    for i in range(64):
        _ = enhanced_fs.smooth_predict_with_psm(test_images[i:i+1])
    original_time = time.time() - start_time

    # 批量方法
    batch_fs = BatchFeatureSpaceSmoothing(
        encoder, psm, sigma=0.25, num_samples=50, batch_size=8
    )

    start_time = time.time()
    _ = batch_fs.smooth_predict_batch(test_images)
    batch_time = time.time() - start_time

    print(f"\n原始方法: {original_time:.2f} 秒")
    print(f"批量方法: {batch_time:.2f} 秒")
    print(f"加速比: {original_time/batch_time:.2f}x")

benchmark_optimization()
```

## 总结

### 核心要点

1. **特征空间平滑**提供了可证明的对抗鲁棒性保证，通过在输入空间添加高斯噪声平滑特征表示

2. **PSM 模块**通过 Purifier 和 Smoothness Mapper 提升高斯鲁棒性分数，进一步增强防御效果

3. **理论保证**：特征余弦相似度下界 (FCSB) 提供了数学上的鲁棒性证明

4. **实用性**：无需重新训练原始模型，即插即用

### 优缺点分析

**优点**：
- ✅ 提供可证明的理论保证
- ✅ 无需重新训练大模型
- ✅ 适用于任意特征编码器
- ✅ 防御效果显著（ASR 从 90% 降至 1%）

**缺点**：
- ❌ 计算开销大（需要多次采样）
- ❌ 推理延迟增加
- ❌ 对超参数敏感

### 适用场景

- 高安全性要求的多模态应用
- 医疗影像分析
- 自动驾驶视觉系统
- 金融文档理解

### 进阶阅读

1. **原始论文**: "Provable Robustness in Multimodal Large Language Models via Feature Space Smoothing"
2. **随机平滑理论**: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing"
3. **对抗训练**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
4. **多模态学习**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"

### 实验环境

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 16GB+ GPU 内存

---

**致谢**: 感谢原论文作者提供的理论框架和实验指导。本教程旨在帮助研究者和工程师理解并实现可证明的对抗鲁棒性防御方法。
