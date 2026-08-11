---
layout: post-wide
title: '用夜间灯光卫星图像检测"暗船"：双分支 YOLO11 实战指南'
date: 2026-08-11 08:04:13 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.09360v1
generated_by: Claude Code CLI
---

## 一句话总结

通过融合 SDGSAT-1 卫星的全色（PAN）和多光谱（RGB）夜间灯光图像，双分支 YOLO11 能检测出不发送 AIS 信号的"暗船"，精度达 0.99、召回率 0.93——但部署到新海域前，泛化能力值得存疑。

## 背景：77% 的渔船不发信号

现实海洋监控有个严峻事实：船只通过 **AIS（船舶自动识别系统）**广播位置，但大量渔船刻意关闭 AIS 以规避监管——或者根本没有安装 AIS 的义务。

在印度西海岸的实验中，检测到的 31525 艘船只里，只有 22.7%（7146 艘）有 AIS 记录，剩下 77.3% 是"暗船"。

传统 SAR 雷达卫星能发现金属目标，但数据成本高、时间分辨率低。**夜间灯光卫星图像（NTL）**的核心优势在于：渔船夜间作业用强光吸引鱼群，这种光辐射在卫星图像中清晰可见，且成本远低于 SAR。

问题是：NTL 图像里的渔船极小——典型目标只有几个像素，对传统检测器极具挑战。

## 技术挑战

### 小目标检测有多难？

SDGSAT-1 卫星提供两种互补数据：
- **全色（PAN）图像**：10m 分辨率，单通道，空间细节丰富
- **RGB 图像**：40m 分辨率，三通道，包含光谱信息

一艘 20 米长的小渔船在 10m 分辨率图像中大约占 2×1 像素。在 40m 分辨率的 RGB 图像里，它可能不足一个像素。

标准 YOLO 在这种尺度上表现糟糕，根本原因有三：
1. 特征图经过多次下采样，微小目标的梯度信号消失
2. 单模态输入丢失了互补的光谱信息
3. 夜间背景复杂（海浪反光、月光、石油平台等固定光源干扰）

### 核心 insight

论文的思路直接：**两个 backbone 并联，分别提取 PAN 和 RGB 特征，在特征空间拼接后统一检测**。PAN 分支贡献空间精度，RGB 分支贡献光谱区分性。

## 双分支 YOLO11 架构

### 直觉理解

```
PAN图像 (1ch, 10m) ──→ [Backbone A] ──→ PAN特征图 ──┐
                                                      concat ──→ [Neck+Head] ──→ 检测框
RGB图像 (3ch, 40m) ──→ [Backbone B] ──→ RGB特征图 ──┘
                  (先上采样对齐到PAN分辨率)
```

### 最小可运行版本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNA(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU()
        )
    def forward(self, x): return self.net(x)


class DualBranchFusion(nn.Module):
    """双分支特征融合：PAN高分辨率 + RGB光谱信息"""
    def __init__(self):
        super().__init__()
        self.pan_branch = nn.Sequential(
            ConvBNA(1, 32, s=2),   # 1/2
            ConvBNA(32, 64, s=2),  # 1/4
            ConvBNA(64, 128, s=2), # 1/8
        )
        self.rgb_branch = nn.Sequential(
            ConvBNA(3, 32, s=2),
            ConvBNA(32, 64, s=2),
            ConvBNA(64, 128, s=2),
        )
        self.fusion = ConvBNA(256, 128, k=1, p=0)  # 拼接后降维

    def forward(self, pan, rgb):
        pan_feat = self.pan_branch(pan)
        # RGB分辨率低4倍，需上采样对齐PAN分辨率后再提取特征
        rgb_up = F.interpolate(rgb, size=pan.shape[-2:], mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_branch(rgb_up)
        fused = torch.cat([pan_feat, rgb_feat], dim=1)  # [B, 256, H/8, W/8]
        return self.fusion(fused)
```

### 完整检测模型与训练循环

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DualInputDataset(Dataset):
    def __init__(self, pan_paths, rgb_paths, labels, img_size=1280):
        self.pan_paths = pan_paths
        self.rgb_paths = rgb_paths
        self.labels = labels  # 每个元素: [[x_c, y_c, w, h], ...]（归一化坐标）
        self.img_size = img_size

    def __len__(self): return len(self.pan_paths)

    def __getitem__(self, idx):
        pan = self._load_pan(self.pan_paths[idx])
        rgb = self._load_rgb(self.rgb_paths[idx])
        boxes = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pan, rgb, boxes

    def _load_pan(self, path):
        # 用 rasterio 读取真实GeoTIFF文件，这里用随机值占位
        arr = np.random.rand(1, self.img_size, self.img_size).astype(np.float32)
        return torch.from_numpy(normalize_ntl(arr))

    def _load_rgb(self, path):
        # RGB分辨率低4倍
        arr = np.random.rand(3, self.img_size // 4, self.img_size // 4).astype(np.float32)
        return torch.from_numpy(arr.clip(0, 1))


class DualBranchDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.fusion = DualBranchFusion()
        # 检测头：实际部署时替换为YOLO11的anchor-free head
        self.detect_head = nn.Sequential(
            ConvBNA(128, 256),
            ConvBNA(256, 256),
            nn.Conv2d(256, (num_classes + 4 + 1), 1)  # cls + xywh + conf
        )

    def forward(self, pan, rgb):
        feat = self.fusion(pan, rgb)
        return self.detect_head(feat)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    for pan, rgb, boxes in loader:
        pan, rgb = pan.to(device), rgb.to(device)
        optimizer.zero_grad()
        pred = model(pan, rgb)
        loss = detection_loss(pred, boxes.to(device))  # 需实现CIoU + focal loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
```

### 用 Ultralytics 快速集成

不想从头造轮子？可以用加法融合改造 YOLO11 的输入层：

```python
from ultralytics import YOLO
import torch.nn as nn, torch

class DualInputYOLO(nn.Module):
    """
    轻量改造：RGB经小网络压缩后与PAN相加，其余复用标准YOLO11
    加法融合比拼接融合侵入性更小，但表达能力稍弱
    """
    def __init__(self, yolo_weights='yolo11s.pt'):
        super().__init__()
        self.yolo = YOLO(yolo_weights).model
        # RGB → 单通道特征图，与PAN相加
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 1, 1),       # 投影到单通道
        )

    def forward(self, pan, rgb=None):
        if rgb is not None:
            rgb_aligned = torch.nn.functional.interpolate(rgb, size=pan.shape[-2:])
            pan = pan + self.rgb_adapter(rgb_aligned)
        return self.yolo(pan)
```

## 小目标检测关键 Trick

### Trick 1：不要让目标被下采样淹没

普通 YOLO 的最大步长是 32，640 图像最终特征图是 20×20。2 像素的渔船经过多次下采样后几乎消失。

解决方案：在 YOLO11 的 yaml 配置里增加 P2 检测头（stride=4），专门检测小目标；同时把输入分辨率提高到 1280：

```yaml
# 在 yolo11s.yaml 的 head 部分添加浅层输出
# 默认: [P3/8, P4/16, P5/32] → 修改为: [P2/4, P3/8, P4/16]
head:
  - [4, 1, Conv, [256, 1, 1]]   # 引用P2层特征（更浅，保留小目标细节）
  ...
```

### Trick 2：NTL 图像特化归一化

夜间灯光图像的像素值分布极度偏斜——渔船的亮点可能是 65535，背景是 0-100。ImageNet 的归一化策略完全不适用：

```python
def normalize_ntl(img: np.ndarray, percentile_high: float = 99.9) -> np.ndarray:
    """截断高亮异常值后归一化，保留亮度相对大小（关键信号）"""
    p_high = np.percentile(img, percentile_high)
    img_clipped = np.clip(img, 0, p_high)
    return img_clipped / (p_high + 1e-8)
    # 绝对不能用均值归一化：会把渔船的亮度信号压成0
```

### Trick 3：谨慎使用亮度增强

渔船本质上是点光源，随机亮度扰动会破坏最重要的特征信号：

```python
# 在 Ultralytics 训练配置中
train_args = {
    'imgsz': 1280,
    'hsv_h': 0.0,   # 关闭色调变换
    'hsv_s': 0.1,   # 极小饱和度变化
    'hsv_v': 0.05,  # 亮度几乎不动——亮度IS the signal
    'mosaic': 0.5,  # 马赛克增强对小目标有帮助
    'mixup':  0.0,  # 关闭mixup（会混淆亮度绝对值）
    'flipud': 0.5,
    'fliplr': 0.5,
}
```

## 实验结果

### 单分支 vs 双分支对比

| 模型 | Precision | Recall | F1 | mAP@50 |
|------|-----------|--------|----|--------|
| YOLOv5s (仅PAN) | 0.91 | 0.82 | 0.86 | 0.88 |
| YOLOv8s (仅PAN) | 0.93 | 0.85 | 0.89 | 0.91 |
| YOLO11s (仅PAN) | 0.95 | 0.87 | 0.91 | 0.93 |
| **YOLO11 双分支** | **0.99** | **0.93** | **0.96** | **0.96** |

双分支的提升是真实的，但 **0.99 的精度要留心**——需要确认测试集与训练集是否来自同一时间段和同一海域，跨域评估的数字通常会低得多。

### 时空分布的附加价值

模型的检测结果揭示了可供后处理利用的规律：
- 峰值活动期：1-4 月（大陆架季节性高产期）
- 主活动廊道：距岸 50-100 km
- 活动高峰时间：夜间 20:00-23:00

这些先验可以用来过滤假阳性：静止不动的检测框（石油平台、养殖浮标）连续帧对比即可排除。

## 调试指南

### 常见问题

**1. mAP 在 0.3 左右卡死**

最可能原因：IoU 阈值对小目标太严格。NTL 图像的目标框有时只有 3×3 像素，1 像素的定位偏差就会让 IoU 跌到 0。

```python
# 评估和推理时都降低IoU阈值
results = model.val(iou=0.3)        # 默认0.7 → 降到0.3
preds = model.predict(img, iou=0.5) # NMS阈值也相应放宽
```

**2. 精度高但召回率低（大量漏检）**

密集渔场（多艘船紧密排列）里，NMS 会把真实目标当重叠框删掉。调高 NMS 的 IoU 阈值：

```python
preds = model.predict(img, iou=0.6, conf=0.2)
```

**3. 石油平台、养殖浮标被大量误检**

固定光源是 NTL 小目标检测的天敌，两种解法：
- 训练集中标注负样本（把固定设施标为背景）
- 后处理：用已知固定设施坐标库做空间过滤

### 超参数敏感度

| 参数 | 推荐值 | 敏感度 | 说明 |
|------|--------|--------|------|
| `imgsz` | 1280 | 极高 | 别用 640，小目标直接消失 |
| `lr0` | 0.001 | 高 | 不稳定再降一个数量级 |
| `iou` (训练匹配) | 0.4 | 中 | 太高会让小目标匹配不上正样本 |
| `conf` (推理) | 0.2 | 高 | NTL 信噪比低，别设太高 |
| `batch` | 8-16 | 低 | 显存允许就往大调 |

### 判断模型是否真的学到了渔船

```python
# 检查预测框的尺寸分布——渔船应该极小
preds = model.predict(test_img, conf=0.2)
widths = [box.xywh[0][2].item() for box in preds[0].boxes]
print(f"预测框平均宽度: {np.mean(widths):.1f} px")
# 若平均宽度 > 50px，模型在检测海岛或云层，而非渔船
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 夜间作业渔船大范围普查 | 白天监控（无灯光信号） |
| AIS 数据的补充核查 | 精确船只类型分类 |
| 热带/亚热带低云量海域 | 多云季节（云层遮蔽 NTL） |
| 配合时序分析区分移动/固定目标 | 密集港口（固定灯光干扰过多） |

## 我的观点

这篇论文做了一件正确的事：把多模态融合应用在**真实有需求的问题**上，而不是为了用多模态而用。双分支融合本身不是新思路，但用在 NTL 小目标检测这个场景是合理的工程选择。

不过有两点存疑。第一，0.99 精度在夜间图像检测上并不意外——渔船本质上就是黑色背景上的亮点，这个任务固有难度不像陆地目标检测那么高，数字本身可能高估了模型的实际能力。

第二，**"77% 都是暗船"**这个结论需要谨慎解读。IMO 规定 300GT 以上的船只才强制装 AIS，大量小型渔船从法规上就不需要 AIS，所以这个比例更多反映 AIS 制度的覆盖范围，不完全等于"刻意规避监管"。用这套系统做执法依据时要注意这个区别。

工程上最值得借鉴的是：**双分支特征融合 + P2 浅层检测头 + NTL 特化归一化**这三件事组合，在其他小目标检测场景（无人机图像、航拍交通监控）里同样适用，可以直接复用。