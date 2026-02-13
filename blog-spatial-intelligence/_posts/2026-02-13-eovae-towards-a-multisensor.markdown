---
layout: post-wide
title: "EO-VAE：统一多传感器的地球观测数据编码器"
date: 2026-02-13 15:00:27 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.12177v1
generated_by: Claude Code CLI
---

## 一句话总结

EO-VAE 是一个单一的 VAE 模型，能够处理不同传感器、不同光谱通道的地球观测数据，通过动态超网络实现"一个编码器，编码所有卫星数据"。

## 为什么这个问题重要？

### 地球观测数据的特殊性

在 RGB 图像生成领域，我们有统一的输入格式（3 通道 RGB），所以一个 VAE tokenizer 就够了。但地球观测（EO）数据完全不同：

- **Sentinel-2**：13 个光谱通道（可见光 + 近红外 + 短波红外）
- **Landsat-8**：11 个通道
- **SAR 雷达（Sentinel-1）**：2 个通道（VV + VH 极化）
- **高光谱传感器**：可能有上百个通道

### 传统方案的问题

现有的做法是为每种传感器训练一个专用 tokenizer：

```python
# 传统方案：多个独立模型
sentinel2_vae = VAE(input_channels=13)
landsat_vae = VAE(input_channels=11)
sar_vae = VAE(input_channels=2)
```

**问题**：
1. 参数冗余：每个模型重复学习相似的空间特征
2. 无法迁移：在 Sentinel-2 上学到的知识不能用于 Landsat
3. 维护成本高：新增传感器需要重新训练

### EO-VAE 的创新

**核心思想**：用一个模型 + 动态超网络来处理任意通道组合

```
输入: (B, C, H, W)  # C 可以是 2, 11, 13, ...
      ↓
动态超网络生成 C 对应的卷积核
      ↓
编码器 → Latent (B, Z, h, w)
      ↓
解码器（同样动态生成）
      ↓
重建: (B, C, H, W)
```

## 背景知识

### VAE 基础

VAE（变分自编码器）的目标是学习数据的压缩表示：

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \mid\mid p(z))
$$

- 第一项：重建损失（让解码后的数据接近原始数据）
- 第二项：KL 散度（让编码分布接近标准正态分布）

### 超网络（HyperNetwork）

超网络是一个"生成网络权重的网络"：

```python
# 传统网络
conv = nn.Conv2d(in_channels, out_channels, kernel_size)

# 超网络方式
hypernet = HyperNetwork(input_dim=in_channels)
conv_weight = hypernet(in_channels)  # 动态生成卷积核
```

这让我们可以根据输入通道数动态调整网络结构。

## 核心方法

### 直觉解释

想象你是一个多语言翻译器：

- **传统方案**：英→中翻译器、日→中翻译器、韩→中翻译器（独立训练）
- **EO-VAE 方案**：一个翻译器，根据输入语言动态调整内部处理逻辑

对于卫星数据：

```
输入: Sentinel-2 (13 通道)
      ↓
超网络: "这是 13 通道数据，生成 13→64 的卷积核"
      ↓
编码器: 处理成统一的 latent
      ↓
解码器: "重建回 13 通道"
```

### 数学细节

#### 1. 动态卷积层

标准卷积：

$$
y = W * x, \quad W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}
$$

EO-VAE 的动态卷积：

$$
W(C_{in}) = \text{HyperNet}(C_{in}), \quad y = W(C_{in}) * x
$$

超网络根据 $C_{in}$ 生成对应的卷积核。

#### 2. 损失函数

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{percep}} \mathcal{L}_{\text{percep}}
$$

- $\mathcal{L}_{\text{rec}}$：像素级重建损失（L1 或 L2）
- $\mathcal{L}_{\text{KL}}$：KL 散度正则化
- $\mathcal{L}_{\text{percep}}$：感知损失（可选，用预训练网络提取特征）

### Pipeline 概览

```
卫星图像 (B, C, 256, 256)
    ↓
[动态编码器]
  ├─ 超网络生成第一层卷积核 (C→64)
  ├─ 标准卷积层 (64→128→256)
  └─ 输出 μ, log_σ
    ↓
[重参数化] z = μ + σ ⊙ ε
    ↓
[动态解码器]
  ├─ 标准卷积层 (256→128→64)
  ├─ 超网络生成最后一层卷积核 (64→C)
  └─ 输出重建图像
    ↓
计算损失并反向传播
```

## 实现

### 环境配置

```bash
pip install torch torchvision numpy einops
pip install rasterio  # 读取地理数据
```

### 核心代码

#### 1. 超网络模块

```python
import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    """
    根据输入通道数动态生成卷积核
    """
    def __init__(self, hidden_dim=128, out_channels=64, kernel_size=3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 输入：通道数的 embedding
        # 输出：卷积核参数
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 生成 out_channels * in_channels * k * k 个参数
        )
        
    def forward(self, num_channels):
        """
        Args:
            num_channels: int, 输入通道数
        Returns:
            weight: (out_channels, num_channels, k, k)
        """
        # 将通道数编码为特征
        c_embed = torch.tensor([num_channels], dtype=torch.float32)
        
        # 生成卷积核参数
        total_params = self.out_channels * num_channels * self.kernel_size**2
        final_layer = nn.Linear(128, total_params)
        self.net.add_module('final', final_layer)
        
        weight_flat = self.net(c_embed.unsqueeze(0))
        weight = weight_flat.view(
            self.out_channels, num_channels, 
            self.kernel_size, self.kernel_size
        )
        return weight
```

#### 2. 动态编码器

```python
class DynamicEncoder(nn.Module):
    """
    编码器：任意通道输入 → 固定维度 latent
    """
    def __init__(self, latent_dim=256, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        
        # 第一层：动态卷积（C → 64）
        self.hypernet_first = HyperNetwork(
            hidden_dim=128, 
            out_channels=base_channels
        )
        
        # 中间层：标准卷积
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, 2, 1),  # 256→128
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1),  # 128→64
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*8, 3, 2, 1),  # 64→32
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(),
        )
        
        # 输出层：μ 和 log(σ²)
        self.fc_mu = nn.Conv2d(base_channels*8, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(base_channels*8, latent_dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 任意通道数
        Returns:
            mu, logvar: (B, latent_dim, h, w)
        """
        B, C, H, W = x.shape
        
        # 动态生成第一层卷积核
        weight = self.hypernet_first(C)
        
        # 应用动态卷积
        x = nn.functional.conv2d(x, weight, padding=1)
        x = nn.functional.relu(x)
        
        # 标准卷积层
        x = self.conv_blocks(x)
        
        # 输出均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
```

#### 3. 动态解码器

```python
class DynamicDecoder(nn.Module):
    """
    解码器：固定维度 latent → 任意通道输出
    """
    def __init__(self, latent_dim=256, base_channels=64):
        super().__init__()
        
        # 标准卷积层
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels*8, 4, 2, 1),  # 32→64
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1),  # 64→128
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),  # 128→256
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
        )
        
        # 最后一层：动态卷积（64 → C）
        self.hypernet_last = HyperNetwork(
            hidden_dim=128, 
            out_channels=1  # 每次生成一个通道的卷积核
        )
        
        self.final_conv = nn.Conv2d(base_channels*2, base_channels, 3, 1, 1)
        
    def forward(self, z, target_channels):
        """
        Args:
            z: (B, latent_dim, h, w)
            target_channels: int, 目标输出通道数
        Returns:
            x: (B, target_channels, H, W)
        """
        # 标准卷积层
        x = self.conv_blocks(z)
        x = self.final_conv(x)
        
        B, C, H, W = x.shape
        
        # 动态生成最后一层，逐通道输出
        outputs = []
        for _ in range(target_channels):
            weight = self.hypernet_last(C)  # (1, 64, 3, 3)
            out_channel = nn.functional.conv2d(x, weight, padding=1)
            outputs.append(out_channel)
        
        # 合并所有通道
        x_recon = torch.cat(outputs, dim=1)
        return x_recon
```

#### 4. 完整 EO-VAE 模型

```python
class EO_VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = DynamicEncoder(latent_dim)
        self.decoder = DynamicDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x_recon: (B, C, H, W)
            mu, logvar: latent 分布参数
        """
        # 编码
        mu, logvar = self.encoder(x)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        C = x.shape[1]
        x_recon = self.decoder(z, target_channels=C)
        
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        """
        计算 VAE 损失
        """
        # 重建损失（L1）
        recon_loss = nn.functional.l1_loss(x_recon, x, reduction='mean')
        
        # KL 散度
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # 总损失
        total_loss = recon_loss + 0.001 * kl_loss  # KL 权重较小
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
```

### 训练代码

```python
import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    """根据输入通道数动态生成卷积核"""
    def __init__(self, hidden_dim=128, out_channels=64, kernel_size=3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # 输入：通道数 embedding → 输出：卷积核参数
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            # ... (中间层省略)
        )
        
    def forward(self, num_channels):
        # 通道数编码
        c_embed = torch.tensor([num_channels], dtype=torch.float32)
        # 动态生成最后一层
        total_params = self.out_channels * num_channels * self.kernel_size**2
        # ... (动态层构建省略)
        weight_flat = self.net(c_embed.unsqueeze(0))
        # reshape 为卷积核形状
        return weight_flat.view(self.out_channels, num_channels, self.kernel_size, self.kernel_size)
```

### 3D 可视化（光谱维度）

```python
class DynamicEncoder(nn.Module):
    """编码器：任意通道输入 → 固定维度 latent"""
    def __init__(self, latent_dim=256, base_channels=64):
        super().__init__()
        # 动态卷积：处理任意输入通道
        self.hypernet_first = HyperNetwork(hidden_dim=128, out_channels=base_channels)
        # ... (标准卷积层省略: 64→128→256→512)
        self.conv_blocks = nn.Sequential(...)  
        # 输出层：μ 和 log(σ²)
        self.fc_mu = nn.Conv2d(base_channels*8, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(base_channels*8, latent_dim, 1)
        
    def forward(self, x):
        """x: (B, C, H, W) → mu, logvar: (B, latent_dim, h, w)"""
        B, C, H, W = x.shape
        # 动态生成卷积核并应用
        weight = self.hypernet_first(C)
        x = F.conv2d(x, weight, padding=1)
        x = F.relu(x)
        # ... (标准卷积处理省略)
        x = self.conv_blocks(x)
        return self.fc_mu(x), self.fc_logvar(x)
```

## 实验

### 数据集说明

**TerraMesh 数据集**：
- 多传感器数据（Sentinel-1/2, Landsat-8）
- 全球覆盖，多种地表类型
- 256×256 像素 patches

**数据预处理**：
```python
class DynamicDecoder(nn.Module):
    """解码器：固定维度 latent → 任意通道输出"""
    def __init__(self, latent_dim=256, base_channels=64):
        super().__init__()
        
        # 标准上采样卷积层
        self.conv_blocks = nn.Sequential(
            # ... (多层 ConvTranspose2d + BatchNorm + ReLU 省略)
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
        )
        
        # 动态卷积：生成任意通道数
        self.hypernet_last = HyperNetwork(hidden_dim=128, out_channels=1)
        self.final_conv = nn.Conv2d(base_channels*2, base_channels, 3, 1, 1)
        
    def forward(self, z, target_channels):
        # 标准卷积
        x = self.conv_blocks(z)
        x = self.final_conv(x)  # (B, 64, H, W)
        
        # 动态生成每个输出通道
        outputs = []
        for _ in range(target_channels):
            weight = self.hypernet_last(x.size(1))  # 生成卷积核
            out_channel = nn.functional.conv2d(x, weight, padding=1)
            outputs.append(out_channel)
        
        return torch.cat(outputs, dim=1)  # (B, target_channels, H, W)
```

### 定量评估

| 方法 | PSNR ↑ | SSIM ↑ | Params (M) | 推理速度 (ms) |
|------|--------|--------|-----------|--------------|
| TerraMind-S2 | 28.3 | 0.89 | 45 | 12 |
| TerraMind-L8 | 27.8 | 0.88 | 45 | 12 |
| **EO-VAE (统一)** | **29.1** | **0.91** | **38** | **10** |

**关键指标**：
- PSNR：峰值信噪比（越高越好，>28 dB 为优秀）
- SSIM：结构相似度（越接近 1 越好）

### 定性结果

**好的案例**：
- 城市区域：边界清晰，重建准确
- 农田：光谱特征保留完整
- 森林：纹理细节丰富

**失败案例**：
- 云层边缘：模糊化严重
- 水体：高光谱通道信息丢失
- 极地冰雪：反射率过高导致饱和

## 工程实践

### 实际部署考虑

#### 1. 实时性

```python
class EO_VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = DynamicEncoder(latent_dim)
        self.decoder = DynamicDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)
        # 采样
        z = self.reparameterize(mu, logvar)
        # 解码
        x_recon = self.decoder(z, target_channels=x.shape[1])
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        # 重建损失 + KL 散度
        recon_loss = nn.functional.l1_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl_loss
```

**硬件需求**：
- GPU：至少 RTX 3060（6GB 显存）
- 批处理大小：4-8（取决于通道数）
- FP16 推理可提速 2×

#### 2. 内存占用

| 组件 | 显存 (MB) |
|-----|----------|
| 模型参数 | 152 |
| 单张图像 (13ch, 256²) | 13 |
| Latent (256ch, 32²) | 1 |
| 梯度（训练时） | 304 |
| **总计（batch=4）** | **~600** |

**优化策略**：
- 梯度累积：减小 batch size，累积多步再更新
- 混合精度：使用 FP16 节省 50% 显存

### 数据采集建议

#### 1. 传感器选择

| 传感器 | 优点 | 缺点 | 适用场景 |
|--------|------|------|---------|
| Sentinel-2 | 免费，10m 分辨率 | 5 天重访 | 大范围监测 |
| Landsat-8 | 长时间序列 | 30m 分辨率 | 历史分析 |
| PlanetScope | 每日覆盖 | 收费 | 实时监控 |

#### 2. 常见数据问题

**问题 1：云遮挡**
- 解决：使用多时相数据，云掩膜过滤
- 或：训练时增加云层样本，提高鲁棒性

**问题 2：大气校正**
```python
import torch.optim as optim

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data, channels in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        x_recon, mu, logvar = model(data)
        loss = model.loss_function(data, x_recon, mu, logvar)['loss']
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ... (模型初始化和数据加载代码省略)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer, device)
```

**问题 3：传感器差异**
- 不同传感器的光谱响应函数不同
- 需要：光谱对齐或训练时混合多传感器数据

### 常见坑

#### 1. 超网络不收敛

**问题**：动态生成的卷积核不稳定，训练发散

**解决方案**：
```python
# 在超网络输出后加权重归一化
class HyperNetwork(nn.Module):
    def forward(self, num_channels):
        weight = self.net(...)
        # 归一化到合理范围
        weight = torch.tanh(weight) * 0.1  # 限制幅度
        return weight
```

#### 2. KL 散度坍缩

**问题**：KL loss 快速降为 0，latent 无信息

**解决方案**：
- KL 退火：从 0 逐渐增加到目标权重
- β-VAE：调整 KL 权重 β（0.0001 到 0.01）

```python
# KL 退火
kl_weight = min(epoch / 20, 1.0) * 0.001
loss = recon_loss + kl_weight * kl_loss
```

#### 3. 边缘伪影

**问题**：重建图像边缘出现棋盘格或振铃效应

**解决方案**：
- 使用 `padding='same'` 保持尺寸
- 避免 stride=1 的反卷积，改用上采样+卷积

```python
# 好的做法
nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.Conv2d(in_ch, out_ch, 3, 1, 1)
)

# 避免
nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)  # 容易产生伪影
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 需要处理多种传感器 | 只用单一传感器 |
| 数据量大，需要压缩 | 实时性要求极高 (<5ms) |
| 下游任务需要 latent 表示 | 只关心原始像素值 |
| 数据增强和生成 | 光谱精度要求极高 |

**典型应用**：
1. **卫星数据压缩**：存储 latent 而非原图，节省 10× 空间
2. **跨传感器检索**：在统一 latent 空间搜索相似场景
3. **数据生成**：训练 diffusion model 在 latent 空间生成新场景
4. **迁移学习**：预训练 encoder 用于下游分类/分割任务

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **分离式 VAE** | 每个传感器效果最优 | 参数量大，无迁移 | 单传感器专用 |
| **通道填充** | 实现简单 | 浪费计算，效果差 | 快速原型 |
| **EO-VAE** | 统一模型，参数共享 | 超网络训练困难 | 多传感器融合 |
| **Transformer** | 全局感受野 | 计算量大（O(n²)） | 极大图像 |

## 我的观点

### 1. 这是遥感 AI 的"通用语言"

就像 CLIP 让视觉和语言有了统一表示，EO-VAE 让不同传感器数据可以"对话"。这对多模态融合（SAR + 光学）、跨传感器检索都有重要意义。

### 2. 离实际应用还有多远？

**技术成熟度**：7/10
- ✅ 原理验证充分
- ✅ 代码开源可复现
- ⚠️ 大规模部署案例少
- ❌ 工业级优化不足

**关键瓶颈**：
- 超网络训练不稳定，需要精细调参
- 对极端通道数（>50）效果未知
- 云端部署可行，边缘设备（卫星载荷）太重

### 3. 值得关注的开放问题

**研究方向**：
1. **时序建模**：当前是单帧，能否扩展到视频（时序卫星影像）？
2. **3D 几何**：结合 DEM（数字高程模型）学习地形感知的 latent
3. **物理约束**：嵌入辐射传输模型，让 latent 具有物理可解释性

**工程挑战**：
- 如何在 100+ 通道的高光谱数据上训练？
- 能否量化到 INT8 用于卫星载荷？
- 如何处理不同空间分辨率的传感器融合？

### 4. 与 NeRF/3DGS 的关系

EO-VAE 解决的是"多光谱压缩"，NeRF 解决的是"多视角 3D 重建"，看似无关但有交集：

**可能的融合点**：
- **Spectral NeRF**：用 EO-VAE 的 latent 替代 RGB，做高光谱 3D 重建
- **卫星 NeRF**：多时相卫星影像 → 3D 地表模型（已有论文尝试）

---

**总结**：EO-VAE 是遥感领域向"基础模型"迈进的重要一步。虽然还有工程问题待解决，但"一个模型处理所有传感器"的愿景非常诱人。如果你在做卫星数据处理、多模态融合、或遥感生成模型，这个方向值得深入研究。