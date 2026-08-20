---
layout: post-wide
title: "农业自动拖拉机：LiDAR 导航 + 非对称损失除草检测的系统设计"
date: 2026-08-20 12:02:37 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2608.19004v1
generated_by: Claude Code CLI
---

## 一句话总结

AgriNav 解决了水稻田自动除草的三个核心痛点：树冠遮挡下 GNSS 失效、水稻与杂草的实时视觉区分、以及"把水稻当杂草喷药"这一不可逆错误的代价最小化。

---

## 为什么这个问题难？

精准农业中的选择性除草（Site-Specific Weed Management）理论上可以减少 70-90% 的除草剂用量，但三个工程挑战长期阻碍落地：

1. **GNSS 在作物行间失效**：水稻拔节期后，叶片遮挡严重，多路径效应导致定位误差从 <0.5m 跳升至 >3m
2. **水稻 vs 杂草的形态相似性**：稗草（*Echinochloa crus-galli*）早期与水稻几乎一致，视觉分类困难
3. **误分类代价不对称**：把杂草漏掉（FN）可以下次补喷；把水稻当杂草（FP）喷了农药就死了——两类错误代价完全不同

AgriNav 的思路是：不用更昂贵的传感器，而是在**算法层面**显式建模这种不对称性。

---

## 系统架构

```
传感器层        融合层              决策层
┌─────────┐
│  相机   │──→ WeedDet CNN-FPN ─┐
└─────────┘                     ├─→ 反逻辑置信门 ──→ 喷药执行
┌─────────┐   LiDAR-Camera      │
│  LiDAR  │──→ ROI 约束桥 ──────┘
└─────────┘          │
┌─────────┐          ↓
│  IMU    │──→ 6-状态 EKF ──────→ 路径跟踪
│  GNSS   │         ↑
│  轮编码 │─────────┘
└─────────┘
```

四个 ROS 模块通过消息队列解耦，导航 EKF 和视觉检测可独立运行，是这个设计最重要的工程决策。

---

## 核心一：非对称损失 + 反逻辑置信门

### 直觉

传统分类：问"这是杂草吗？" → 置信度 > 0.5 就喷药

AgriNav 的逻辑反转：问"这**不是**水稻吗？" → 只有当模型对"这是水稻"的置信度**低于阈值**时，才允许喷药。水稻保护优先级硬编码。

### 非对称焦点损失

$$
\mathcal{L}_{asym} = -\sum_{c} w_c \cdot (1 - p_c)^\gamma \cdot \log(p_c)
$$

其中 $w_{rice} \gg w_{weed}$，强迫模型在水稻上有更高的召回率。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    """
    非对称焦点损失：对水稻类给予更高惩罚权重
    rice_weight: 水稻误分类的额外惩罚倍数（论文中隐含约 3-5x）
    gamma: 焦点参数，降低简单样本权重
    """
    def __init__(self, num_classes=3, rice_class_idx=0,
                 rice_weight=5.0, gamma=2.0):
        super().__init__()
        self.rice_idx = rice_class_idx
        self.gamma = gamma
        # 类别权重：水稻 >> 杂草 >> 背景
        weights = torch.ones(num_classes)
        weights[rice_class_idx] = rice_weight
        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # 取出目标类别的概率
        targets_one_hot = F.one_hot(targets, logits.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        p_t = (probs * targets_one_hot).sum(dim=1)        # [B, H, W]
        focal = (1 - p_t) ** self.gamma

        # 逐像素乘以类别权重
        w_t = (self.weights[targets])                      # [B, H, W]
        loss = -w_t * focal * (log_probs * targets_one_hot).sum(dim=1)
        return loss.mean()
```

### 反逻辑置信门（硬编码水稻保护）

```python
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, num_classes=3, rice_class_idx=0, rice_weight=5.0, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        weights = torch.ones(num_classes)
        weights[rice_class_idx] = rice_weight  # w_rice >> w_weed >> w_bg
        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        # logits: [B,C,H,W], targets: [B,H,W]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, logits.size(1)).permute(0,3,1,2).float()

        p_t = (probs * targets_one_hot).sum(dim=1)          # 目标类概率
        focal = (1 - p_t) ** self.gamma                     # 焦点调制
        w_t = self.weights[targets]                          # 非对称类别权重

        loss = -w_t * focal * (F.log_softmax(logits, dim=1) * targets_one_hot).sum(dim=1)
        return loss.mean()
```

这个"硬编码置信门"的设计哲学值得注意：它把安全约束放到了**算法控制流**里而不是损失函数里。即使模型输出有误，逻辑门也会在最后一刻拦住高置信水稻区域。

---

## 核心二：6-状态 CVTR-EKF 导航融合

### 状态模型

状态向量 $\mathbf{x} = [x, y, \theta, v, \omega, a]^T$：位置、航向、线速度、角速度、纵向加速度。

常速转弯率（CTRV）预测模型：

$$
\begin{bmatrix} x' \\ y' \\ \theta' \\ v' \\ \omega' \\ a' \end{bmatrix} = \begin{bmatrix} x + \frac{v}{\omega}\bigl(\sin(\theta + \omega \Delta t) - \sin\theta\bigr) \\ y + \frac{v}{\omega}\bigl(-\cos(\theta + \omega \Delta t) + \cos\theta\bigr) \\ \theta + \omega \Delta t \\ v + a \Delta t \\ \omega \\ a \end{bmatrix}
$$

```python
import numpy as np

class CVTR_EKF:
    """6-状态 CVTR 扩展卡尔曼滤波器，用于 GNSS/IMU/轮编码融合"""

    def __init__(self, dt=0.05):
        self.dt = dt
        self.x = np.zeros(6)          # [x, y, θ, v, ω, a]
        self.P = np.eye(6) * 0.1
        # 过程噪声（通过田间标定确定）
        self.Q = np.diag([0.01, 0.01, 0.005, 0.1, 0.01, 0.5])

    def predict(self):
        x, y, th, v, w, a = self.x
        dt = self.dt
        eps = 1e-6

        if abs(w) < eps:              # 直线运动退化情况
            x_n = x + v * np.cos(th) * dt
            y_n = y + v * np.sin(th) * dt
        else:
            x_n = x + (v / w) * (np.sin(th + w * dt) - np.sin(th))
            y_n = y + (v / w) * (-np.cos(th + w * dt) + np.cos(th))

        self.x = np.array([x_n, y + (v/w)*(-np.cos(th+w*dt)+np.cos(th))
                           if abs(w) >= eps else y_n,
                           th + w * dt, v + a * dt, w, a])
        # 雅可比矩阵 F（线性化用）
        F = self._jacobian_F()
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, z_gnss, R_gnss):
        """GNSS 更新：观测 [x, y]"""
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        self._ekf_update(z_gnss, H, R_gnss)

    def update_imu(self, z_imu, R_imu):
        """IMU 更新：观测 [ω, a]"""
        H = np.zeros((2, 6))
        H[0, 4] = H[1, 5] = 1.0
        self._ekf_update(z_imu, H, R_imu)

    def _ekf_update(self, z, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    def _jacobian_F(self):
        # ... (雅可比矩阵数值微分省略)
        return np.eye(6)
```

论文的三级 GNSS 中断桥接策略：

| 中断时长 | 策略 |
|---------|------|
| 0-5 秒 | 纯 IMU + 轮编码推算 |
| 5-15 秒 | EKF 预测步骤继续，不做 GNSS 更新 |
| >15 秒 | 切换到 LiDAR 作物行特征定位 |

---

## 核心三：LiDAR-Camera 融合桥

四个机制以**零额外硬件成本**实现融合：

```python
import numpy as np

class CVTR_EKF:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.x = np.zeros(6)   # [x, y, θ, v, ω, a]
        self.P = np.eye(6) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.005, 0.1, 0.01, 0.5])

    def predict(self):
        x, y, th, v, w, a = self.x
        dt, eps = self.dt, 1e-6
        if abs(w) < eps:  # 直线运动退化
            x_n = x + v * np.cos(th) * dt
            y_n = y + v * np.sin(th) * dt
        else:
            x_n = x + (v / w) * (np.sin(th + w * dt) - np.sin(th))
            y_n = y + (v / w) * (-np.cos(th + w * dt) + np.cos(th))
        self.x = np.array([x_n, y_n, th + w * dt, v + a * dt, w, a])
        F = self._jacobian_F()  # ... (解析雅可比省略)
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, z, R):
        H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1.0
        self._ekf_update(z, H, R)

    def update_imu(self, z, R):
        H = np.zeros((2, 6)); H[0, 4] = H[1, 5] = 1.0
        self._ekf_update(z, H, R)

    def _ekf_update(self, z, H, R):
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.x += K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    def _jacobian_F(self):
        # ... (数值微分省略)
        return np.eye(6)
```

ROI 约束的效果直观：LiDAR 先"扫描"作物行位置，相机只在有植被的区域做密集推理，背景区域直接跳过。

---

## 工程实践

### 非对称阈值的田间标定

```python
def lidar_camera_fusion_bridge(lidar_pts_cam, img_h, img_w, K, detection_model, ground_z_threshold=0.1):
    # 机制1: 地面过滤
    pts = lidar_pts_cam[lidar_pts_cam[:, 2] > ground_z_threshold]

    # 机制2: 投影到图像平面
    pts = pts[pts[:, 2] > 0]
    uv = (K @ pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    # 机制3: 生成 ROI mask（约束推理区域，减少 30-50% 计算量）
    roi_mask = np.zeros((img_h, img_w), dtype=bool)
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    valid = (0 <= u) & (u < img_w) & (0 <= v) & (v < img_h)
    roi_mask[v[valid], u[valid]] = True
    roi_mask = cv2.dilate(roi_mask.astype(np.uint8), np.ones((15, 15)), iterations=2).astype(bool)

    # 机制4: 双向置信融合（模型只在 ROI 内推理）
    return detection_model(roi_mask), roi_mask
```

### 常见坑

**坑 1：LiDAR-Camera 外参标定误差**
LiDAR 到相机的外参 $T_{LC}$ 误差 >1cm 就会导致 ROI 偏移，作物行边缘的植株落在 ROI 外被漏检。用棋盘格+反射板联合标定，RMS 重投影误差应 <0.5 像素。

**坑 2：轮编码器打滑**
水田泥泞条件下轮编码器滑移率可达 15-20%，直接接入 EKF 会累积漂移。需要对轮速度乘以滑移补偿系数，或降低轮编码观测噪声矩阵 $R$ 的可信度权重。

**坑 3：非对称损失训练不稳定**
`rice_weight=5.0` 在训练初期容易导致梯度爆炸。建议前 10 个 epoch 用标准 CE 预热，再逐渐引入非对称权重（从 1.0 线性增加到目标值）。

---

## 定量结果

| 指标 | 数值 | 备注 |
|-----|------|------|
| GNSS 中断桥接 | 20 秒连续追踪 | 仿真环境 |
| 作物行检测置信度 | >0.9 | 全程 |
| 水稻检测置信度区间 | 0.32–0.95 | 跨水稻田/航拍/洪水后图像 |
| LiDAR ROI 推理区域缩减 | 30–50% | 相比全图推理 |
| 模型参数量 | 1.68M | CNN-FPN 轻量变体 |

---

## 适用边界

| 适用场景 | 不适用场景 |
|---------|-----------|
| 水稻拔节期前（叶片稀疏，LiDAR 可穿透） | 拔节后期密闭冠层（LiDAR 点云全被遮挡） |
| 单一作物行结构规整的田块 | 不规则种植或套种场景 |
| 光照稳定的白天作业 | 阴雨/逆光/晨雾 |
| GNSS 短暂中断（<20s） | 长时间完全无 GNSS 环境 |
| 静态杂草检测 | 大风导致叶片抖动（FP 率上升） |

---

## 与同类方法对比

| 方法 | 定位方案 | 检测方式 | 安全机制 | 硬件成本 |
|-----|---------|---------|---------|---------|
| 纯 GNSS RTK | RTK，厘米级 | 无 | 无 | 高 |
| 纯视觉 SLAM | VO，累积漂移 | 独立推理 | 无 | 中 |
| **AgriNav** | GNSS+IMU+轮编码 EKF | LiDAR ROI 约束 | 反逻辑置信门 | 中（复用导航 LiDAR） |
| 商业方案（如 Blue River）| RTK | 独立相机 | 阈值调优 | 高 |

AgriNav 的核心价值在于**复用已有的导航 LiDAR**做视觉加速，没有引入额外传感器成本。

---

## 我的判断

**值得借鉴的思路：**
- 反逻辑置信门把安全约束硬编码进控制流而非仅依赖损失函数，这在农业机器人这类高风险场景中是正确的工程思路。类似的设计可以迁移到任何"误报代价不对称"的场景（如医疗影像中的假阳性手术）。
- LiDAR-Camera 零成本融合是实际部署中常被忽视的优化方向——很多系统的 LiDAR 和 Camera 各自独立工作，白白浪费了几何先验。

**离产品化还差什么：**
- 论文全部实验在**仿真环境**中完成，真实田间数据集（泥泞、雨水、复杂光照）的鲁棒性未验证
- 1.68M 参数模型在嵌入式 GPU（如 Jetson Orin NX）上的实时性需要实测——理论上可行，但热功耗管理在封闭机舱里是个实际问题
- 杂草种类泛化性未测试：从稗草到阔叶草，视觉特征差异大，当前数据集覆盖度存疑

**开放问题：** 随着水稻生长，背景纹理变化大，模型是否需要在线自适应？这在实际季节性部署中是必须解决的问题，但论文没有涉及。