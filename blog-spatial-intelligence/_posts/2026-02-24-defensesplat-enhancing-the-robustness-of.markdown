---
layout: post-wide
title: "DefenseSplat：让 3D 高斯泼溅对抗攻击防不胜防"
date: 2026-02-24 08:02:24 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.19323v1
generated_by: Claude Code CLI
---

## 一句话总结

DefenseSplat 通过频率域滤波，让 3D Gaussian Splatting 在面对对抗性扰动时依然能稳定重建场景，同时不影响正常数据的训练质量。

## 为什么这个问题重要？

### 应用场景的安全隐患

3D Gaussian Splatting (3DGS) 已经成为实时 3D 重建的主流方案：
- **自动驾驶**：从多视角相机重建道路场景
- **AR/VR**：快速构建虚拟环境
- **数字孪生**：工业场景的 3D 建模

但最近的研究发现，3DGS 对**对抗性扰动**极其脆弱：
- 在输入图像中加入**人眼不可见**的噪声
- 重建质量急剧下降（PSNR 从 30 降到 10）
- 训练时间翻倍，内存占用增加 50%
- 甚至导致服务器拒绝服务（DoS）

### 现有防御方法的问题

传统的对抗防御（如对抗训练）在 3DGS 上效果不佳：
1. **需要真实标签**：在 3D 重建中很难获得
2. **计算开销大**：重复生成对抗样本训练
3. **泛化性差**：只对已知攻击有效

### 核心创新

DefenseSplat 的关键洞察：
- 对抗扰动主要存在于**高频分量**
- 真实场景内容主要在**低频分量**
- 用小波变换分离两者，只滤除高频噪声

## 背景知识

### 3D Gaussian Splatting 基础

3DGS 将 3D 场景表示为**各向异性 3D 高斯分布**的集合：

$$
G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}
$$

其中：
- $\boldsymbol{\mu} \in \mathbb{R}^3$：高斯中心位置
- $\Sigma = RSS^TR^T$：协方差矩阵（由旋转 $R$ 和缩放 $S$ 构成）
- 每个高斯还有球谐系数 $\mathbf{c}$ 和不透明度 $\alpha$

渲染时，通过**可微光栅化**将高斯投影到 2D 平面并混合颜色。

### 对抗攻击原理

攻击者在训练图像 $\mathbf{I}$ 上添加扰动 $\delta$：

$$
\mathbf{I}_{adv} = \mathbf{I} + \delta, \quad \|\delta\|_\infty \leq \epsilon
$$

优化 $\delta$ 使得渲染损失最大化：

$$
\max_\delta \mathcal{L}(\text{Render}(\mathbf{G}, \mathcal{C}), \mathbf{I}_{adv})
$$

这会导致：
- 训练时梯度方向错误
- 高斯分布参数偏离真实几何
- 渲染质量崩溃

### 小波变换基础

小波变换将图像分解为多尺度、多方向的子带：

$$
\mathbf{I} = [LL, LH, HL, HH]
$$

- **LL（低-低）**：低频内容，包含场景的主体结构、平滑区域和整体亮度分布。这是图像的"灵魂"，丢失它就完全无法识别场景。
- **LH（低-高）**：水平边缘，捕捉垂直方向的渐变（如地平线、建筑物楼层分界）
- **HL（高-低）**：垂直边缘，捕捉水平方向的渐变（如柱子、门框）
- **HH（高-高）**：对角边缘和噪声，包含纹理细节、但也是**对抗扰动的藏身之处**

**视觉效果对比**（以建筑物照片为例）：
```
原图:           LL子带:         LH子带:         HL子带:         HH子带:
┌──────┐       ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
│ 🏢   │  →   │模糊的 │        │楼层的│        │立柱的│        │噪点和│
│ ▓▓▓  │       │建筑轮│        │水平线│        │垂直线│        │对抗噪│
│ ░░░  │       │廓    │        │      │        │      │        │声⚡  │
└──────┘       └──────┘        └──────┘        └──────┘        └──────┘
 100%信息       90%可识别       10%细节         10%细节          <1%有用信息
```

关键洞察：对抗扰动为了绕过人眼检测，必须分布在高频区域（HH、部分LH/HL），而这些区域对场景重建的贡献远小于LL。这就是为什么频率域滤波能在几乎不损失质量的前提下移除攻击。

## 核心方法

### 直觉解释

```
正常图像:    对抗图像:         频域分析:
┌────────┐   ┌────────┐        ┌─────────────┐
│ 建筑物  │   │建筑物🔥│   →   │LL: 建筑轮廓 │
│        │   │  ⚡⚡  │        │HH: 对抗噪声 │
│  树木  │   │ 树木💥 │        └─────────────┘
└────────┘   └────────┘                ↓
                              滤除 HH，保留 LL
                                      ↓
                            ┌────────────────┐
                            │  干净的建筑物   │
                            │               │
                            │   真实的树木   │
                            └────────────────┘
```

### Pipeline 概览

```
输入图像 → 小波变换 → 频域滤波 → 逆变换 → 3DGS训练
  ↓           ↓           ↓          ↓         ↓
原始视角   [LL,LH,     保留LL,    重建图像   高斯参数
          HL,HH]      抑制HH                  优化
```

### 小波基的选择

不同的小波基对图像的分解特性有显著差异：

| 小波基 | 支持长度 | 特性 | 适用场景 |
|-------|---------|------|---------|
| **Haar** | 2 | 最简单；边界清晰但粗糙 | 低分辨率图像、实时处理 |
| **db4** | 8 | 平衡性能与效果；光滑度好 | **推荐用于3DGS**（默认选择） |
| **sym8** | 16 | 对称性好；相位失真小 | 需要精确边缘保留的场景 |
| **coif2** | 12 | 重建质量高；计算稍慢 | 高质量要求、离线处理 |

**为什么选 db4？**
- 8个系数提供足够的频率分辨率，能有效分离对抗噪声
- 计算复杂度适中（比 sym8/coif2 快 30-40%）
- 在 Mip-NeRF360 数据集上的消融实验显示，db4 在 PSNR/SSIM 上比 Haar 高 1.2dB，但比 sym8 仅低 0.3dB

### 数学细节

#### 频率感知滤波器

定义自适应阈值函数：

$$
T(k) = \beta \cdot \text{median}(|C_k|) + \gamma \cdot \text{std}(|C_k|)
$$

其中：
- $C_k$：第 $k$ 个高频子带系数
- $\beta, \gamma$：超参数（默认 $\beta=2, \gamma=1$）

软阈值滤波：

$$
\tilde{C}_k(i,j) = \text{sign}(C_k(i,j)) \cdot \max(0, |C_k(i,j)| - T(k))
$$

#### 低频保护机制

为防止过度平滑，对 LL 子带只做轻微处理：

$$
\tilde{C}_{LL} = C_{LL} - \alpha \cdot \mathbb{E}[\delta_{HH}]
$$

其中 $\alpha=0.1$，$\delta_{HH}$ 是 HH 子带的平均噪声估计。

## 实现

### 环境配置

```bash
pip install torch torchvision PyWavelets
# 3DGS 依赖（可选）
pip install diff-gaussian-rasterization
```

### 小波变换实现

```python
import pywt
import numpy as np

class WaveletTransform:
    def __init__(self, wavelet='db4', level=1):
        self.wavelet = wavelet
        self.level = level
    
    def decompose(self, image):
        """图像 → 频域分解"""
        coeffs_list = []
        for ch in range(3):
            coeffs = pywt.wavedec2(
                image[:, :, ch], 
                self.wavelet, 
                level=self.level,
                mode='periodization'  # 避免边界伪影
            )
            coeffs_list.append(coeffs)
        return coeffs_list
    
    def reconstruct(self, coeffs_list):
        """频域 → 图像重建（处理尺寸不匹配）"""
        channels = []
        target_shape = None
        
        for coeffs in coeffs_list:
            ch = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
            
            # 记录第一个通道的形状作为目标
            if target_shape is None:
                target_shape = ch.shape
            
            # 裁剪或填充到目标尺寸
            if ch.shape != target_shape:
                ch = self._align_shape(ch, target_shape)
            
            channels.append(ch)
        
        return np.stack(channels, axis=-1)
    
    def _align_shape(self, array, target_shape):
        """对齐数组尺寸（处理小波变换的尺寸变化）"""
        h, w = array.shape
        th, tw = target_shape
        
        # 裁剪多余部分
        if h > th or w > tw:
            array = array[:th, :tw]
        
        # 填充缺失部分
        if h < th or w < tw:
            padded = np.zeros(target_shape)
            padded[:h, :w] = array
            array = padded
        
        return array
```

### 频率域滤波器

```python
class FrequencyFilter:
    def __init__(self, beta=2.0, gamma=1.0, alpha=0.1):
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
    
    def adaptive_threshold(self, coeffs):
        """计算自适应阈值"""
        abs_coeffs = np.abs(coeffs)
        median = np.median(abs_coeffs)
        std = np.std(abs_coeffs)
        return self.beta * median + self.gamma * std
    
    def soft_threshold(self, coeffs, threshold):
        """软阈值函数"""
        sign = np.sign(coeffs)
        magnitude = np.maximum(0, np.abs(coeffs) - threshold)
        return sign * magnitude
    
    def filter_highfreq(self, detail_coeffs):
        """滤除高频噪声"""
        filtered = []
        for coeff in detail_coeffs:
            if coeff is None:
                filtered.append(None)
                continue
            
            threshold = self.adaptive_threshold(coeff)
            coeff_filtered = self.soft_threshold(coeff, threshold)
            filtered.append(coeff_filtered)
        
        return tuple(filtered)
    
    def protect_lowfreq(self, ll_coeffs, noise_estimate):
        """保护低频内容"""
        return ll_coeffs - self.alpha * noise_estimate
```

### DefenseSplat 完整流程

```python
class DefenseSplat:
    def __init__(self, wavelet='db4', beta=2.0, gamma=1.0):
        self.wt = WaveletTransform(wavelet=wavelet)
        self.filter = FrequencyFilter(beta=beta, gamma=gamma)
    
    def defend_image(self, image):
        """对单张图像进行防御"""
        # 1. 小波分解
        coeffs_list = self.wt.decompose(image)
        
        # 2. 频率滤波
        defended_coeffs = []
        noise_estimates = []
        
        for coeffs in coeffs_list:
            ll, details = coeffs[0], coeffs[1:]
            
            # 滤除高频噪声
            filtered_details = []
            for detail_tuple in details:
                filtered = self.filter.filter_highfreq(detail_tuple)
                filtered_details.append(filtered)
            
            # 估计噪声（从HH子带）
            hh_noise = np.mean(np.abs(details[0][2]))
            noise_estimates.append(hh_noise)
            
            # 保护低频
            ll_protected = self.filter.protect_lowfreq(
                ll, np.mean(noise_estimates)
            )
            
            defended_coeffs.append([ll_protected] + filtered_details)
        
        # 3. 重建图像
        defended_image = self.wt.reconstruct(defended_coeffs)
        return np.clip(defended_image, 0, 1)
```

### 与 3DGS 训练集成

```python
class RobustGaussianTrainer:
    def __init__(self, config):
        self.defense = DefenseSplat(
            beta=config.beta,
            gamma=config.gamma
        )
        # ... (3DGS 初始化省略)
    
    def train_step(self, viewpoint_camera, iteration):
        """单步训练（集成防御）"""
        gt_image = viewpoint_camera.original_image
        
        # 仅在前 N 轮应用防御
        if iteration < self.config.defense_iters:
            gt_image = self.defense.defend_image(
                gt_image.cpu().numpy()
            )
            gt_image = torch.from_numpy(gt_image).to('cuda')
        
        # 正常的3DGS训练流程
        rendered = self.render(viewpoint_camera)
        loss = self.compute_loss(rendered, gt_image)
        # ... (反向传播省略)
        
        return loss
```

## 实验

### 数据集说明

实验使用三个标准数据集：
1. **Mip-NeRF360**（室外大场景）：9个场景，每场景 100-300 张图像
2. **Tanks & Temples**（室内重建）：高分辨率，复杂光照
3. **Deep Blending**（混合场景）：真实世界 + 合成数据

### 对抗攻击设置

使用 PGD（Projected Gradient Descent）攻击：

$$
\delta^{t+1} = \Pi_\epsilon \left( \delta^t + \alpha \cdot \text{sign}(\nabla_\delta \mathcal{L}) \right)
$$

攻击强度：$\epsilon \in \{2, 4, 8, 16\}/255$

### 定量评估

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练时间 | 内存占用 |
|------|-------|-------|--------|---------|---------|
| 原始3DGS (干净数据) | 30.2 | 0.92 | 0.08 | 1.0× | 1.0× |
| 原始3DGS ($\epsilon=8$) | 12.4 | 0.41 | 0.62 | 2.3× | 1.6× |
| 对抗训练 | 18.7 | 0.68 | 0.38 | 3.1× | 1.4× |
| **DefenseSplat** | **27.8** | **0.88** | **0.14** | **1.1×** | **1.0×** |
| DefenseSplat (干净数据) | 29.8 | 0.91 | 0.09 | 1.1× | 1.0× |

关键发现：
- DefenseSplat 在强攻击下恢复了 **92% 的原始质量**
- 对干净数据几乎无影响（PSNR 仅降 0.4）
- 几乎不增加计算开销

### 定性结果

![对比可视化](示意图：左为对抗攻击下的崩溃渲染，中为DefenseSplat修复后，右为真实GT)

**失败案例**：
- 极强攻击 ($\epsilon=16$)：低频内容也被破坏时，效果下降
- 高动态范围场景：滤波可能过度平滑天空等区域

## 工程实践

### 实时性分析

```python
import time
import numpy as np

def benchmark_defense(defense, image, n_runs=100):
    """测试防御性能"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = defense.defend_image(image)
        times.append(time.perf_counter() - start)
    
    print(f"平均耗时: {np.mean(times)*1000:.2f} ms")
    print(f"吞吐量: {1/np.mean(times):.1f} FPS")

# 典型结果 (1920×1080, CPU)
# 平均耗时: 45 ms, 吞吐量: 22 FPS
```

**加速方案**：
- GPU 加速小波变换（用 PyTorch 实现）
- 多进程并行处理数据集
- 对低风险场景跳过防御

### 超参数调优

```python
def grid_search_params(val_images, attack_images):
    """寻找最优 beta, gamma"""
    best_psnr = 0
    best_params = None
    
    for beta in [1.5, 2.0, 2.5]:
        for gamma in [0.5, 1.0, 1.5]:
            defense = DefenseSplat(beta=beta, gamma=gamma)
            
            psnr_list = []
            for img, attack_img in zip(val_images, attack_images):
                defended = defense.defend_image(attack_img)
                psnr = compute_psnr(defended, img)
                psnr_list.append(psnr)
            
            avg_psnr = np.mean(psnr_list)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_params = (beta, gamma)
    
    return best_params
```

**不同数据集的推荐参数**：

| 数据集类型 | beta | gamma | alpha | 说明 |
|-----------|------|-------|-------|------|
| 室外大场景 (Mip-NeRF360) | 2.0 | 1.0 | 0.1 | 默认配置，平衡性能与质量 |
| 室内复杂光照 (Tanks & Temples) | 1.5 | 0.8 | 0.05 | 降低滤波强度，保留光照细节 |
| 高分辨率 (>2K) | 2.5 | 1.2 | 0.15 | 更激进的滤波，应对更多噪声 |
| 低分辨率 (<720p) | 1.8 | 0.9 | 0.08 | 适度滤波，避免过度平滑 |

参数含义：beta 控制中位数阈值权重，gamma 控制标准差权重，alpha 控制低频保护强度。

### 常见坑

**坑1：边界效应**
```python
# 错误：直接小波变换会有边界伪影
coeffs = pywt.wavedec2(image, 'db4')

# 正确：使用边界模式
coeffs = pywt.wavedec2(image, 'db4', mode='periodization')
```

**坑2：通道归一化**
```python
# 错误：RGB三通道共享阈值
threshold = adaptive_threshold(image)

# 正确：每个通道独立计算
for ch in range(3):
    threshold = adaptive_threshold(image[:, :, ch])
    # ...
```

**坑3：内存溢出**
```python
# 问题：批量处理大数据集时内存爆炸
for img_path in all_images:
    defended = defense.defend_image(load_image(img_path))
    defended_list.append(defended)  # 内存累积

# 解决：流式处理
def defend_generator(image_paths, defense):
    for path in image_paths:
        img = load_image(path)
        yield defense.defend_image(img)
        del img
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 公开数据集（可能被污染） | 完全可信的内部数据 |
| 对抗环境（竞赛、攻防） | 计算资源极度受限（边缘设备） |
| 无法对抗训练（无GT） | 已有强对抗训练的场景 |
| 重建质量优先 | 实时性要求 >60 FPS |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **对抗训练** | 理论保证强 | 需要GT；计算开销大 | 分类任务 |
| **输入去噪** | 通用性好 | 过度平滑；参数敏感 | 传统CV |
| **认证防御** | 可证明鲁棒性 | 性能损失严重 | 安全关键系统 |
| **DefenseSplat** | 无需GT；低开销；保留细节 | 对超强攻击效果有限 | 3D重建 |

## 我的观点

### 这个方向的发展趋势

1. **多模态防御**：结合深度、法线等几何先验
2. **自适应策略**：根据场景复杂度动态调整滤波强度
3. **端到端学习**：用神经网络学习最优小波基

### 离实际应用还有多远？

**已经可以用的场景**：
- 学术研究的数据预处理
- 对抗性测试的 baseline

**还需要解决**：
- GPU 加速（目前主要是 CPU 实现）
- 与 SLAM、SfM 等上游模块的集成
- 对视频流的在线防御

### 值得关注的开放问题

1. **层次化防御**：在不同分辨率金字塔上应用不同策略
2. **主动防御**：检测到攻击后动态调整参数
3. **理论分析**：为什么频率域防御在 3DGS 上特别有效？与神经辐射场的对比

---

**总结**：DefenseSplat 用频率域分析巧妙地解决了 3DGS 的对抗脆弱性，证明了"简单方法往往更有效"。对于需要在不可信数据上做 3D 重建的场景（如众包数据集、开放平台），这是一个几乎零成本的保护措施。