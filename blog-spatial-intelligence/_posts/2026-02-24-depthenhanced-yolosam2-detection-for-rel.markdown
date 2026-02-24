---
layout: post-wide
title: '深度增强的铁路道床检测：YOLO-SAM2 如何从"看得见"到"看得准"'
date: 2026-02-24 08:07:34 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.18961v1
generated_by: Claude Code CLI
---

## 一句话总结

结合深度校正和 YOLO-SAM2 分割，将铁路道床缺陷检测的召回率从 49% 提升到 80%，解决了纯 RGB 视觉在安全关键场景下"过度自信"的问题。

## 为什么这个问题重要？

### 应用场景
铁路道床（ballast）是轨枕下的碎石层，起支撑、排水、缓冲作用。道床不足会导致：
- 轨道沉降，影响行车安全
- 排水不畅，加速设备老化
- 维护成本增加

传统检测依赖人工巡检，效率低且易漏检。计算机视觉自动化检测是刚需。

### 现有方法的问题
**YOLOv8 纯 RGB 检测**：
- 精度高（0.99），但召回率低（0.49）
- 倾向于把"道床不足"误判为"道床充足"
- 在安全场景下，漏检比误报更危险

**根本原因**：
- RGB 图像受光照、阴影、角度影响大
- 道床石块的纹理和颜色相似，难以仅凭 2D 特征区分

### 核心创新
1. **深度校正 Pipeline**：补偿 RealSense D435 的空间失真
2. **YOLO-SAM2 融合**：精确分割轨枕和道床区域
3. **几何分析**：基于 3D 轮廓的物理规则判断（不是纯数据驱动）

---

## 背景知识

### 3D 表示方式
本文使用 **RGB-D（深度图）**：
- RGB 图像：$(W, H, 3)$，提供纹理和语义
- 深度图：$(W, H, 1)$，提供每个像素的距离 $Z$

与点云对比：
| 表示 | 优点 | 缺点 |
|-----|------|------|
| 点云 | 直接 3D 坐标 | 稀疏、处理慢 |
| 深度图 | 密集、像素对齐 | 需要相机内参 |

### 相机模型
**针孔相机模型**：
$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X/Z \\ Y/Z \\ 1 \end{bmatrix}, \quad K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

深度图转 3D 点云：
$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}
$$

### RealSense 深度失真问题
Intel RealSense D435 使用**双目立体视觉**，深度测量误差随距离增加：
- 近距离（1-2m）：误差 $\pm 2\%$
- 远距离（3-5m）：误差 $\pm 5\%$
- 边缘区域、反光表面误差更大

**本文解决方案**：多项式建模 + RANSAC 平面拟合 + 时序平滑

---

## 核心方法

### 直觉解释

```
RGB-D 输入
    ↓
YOLOv8 检测轨枕边界框
    ↓
深度校正（补偿 RealSense 失真）
    ↓
SAM2 分割轨枕和道床
    ↓
提取 3D 轮廓 → 计算高度差
    ↓
几何规则判断：充足 / 不足
```

**关键思想**：
1. YOLO 负责"在哪里"（定位）
2. SAM2 负责"是什么"（分割精化）
3. 深度负责"有多高"（物理测量）

### 数学细节

#### 1. 深度校正多项式模型
观测到深度 $Z_{\text{raw}}$ 与真实深度 $Z_{\text{true}}$ 的关系：
$$
Z_{\text{true}} = Z_{\text{raw}} + \sum_{i=1}^{n} a_i \cdot Z_{\text{raw}}^i
$$

使用 RANSAC 拟合平面：
$$
ax + by + cz + d = 0
$$
剔除离群点后，用最小二乘法估计多项式系数 $\{a_i\}$。

#### 2. 时序平滑
对连续帧的深度图：
$$
Z_t = \alpha Z_t^{\text{raw}} + (1-\alpha) Z_{t-1}
$$
其中 $\alpha = 0.3$（经验值），平衡响应速度和稳定性。

#### 3. 几何判断规则
定义轨枕上表面高度 $H_{\text{sleeper}}$，道床高度 $H_{\text{ballast}}$：
$$
\Delta H = H_{\text{sleeper}} - H_{\text{ballast}}
$$

判断标准（论文使用两种）：
- **严格标准**：$\Delta H > 50$ mm
- **宽松标准**：$\Delta H > 30$ mm

---

## 实现

### 环境配置

```bash
# 安装依赖
pip install torch torchvision ultralytics
pip install git+https://github.com/facebookresearch/segment-anything-2
pip install opencv-python open3d pyrealsense2

# 下载预训练模型
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
# SAM2 模型从官方仓库下载
```

### 核心代码

#### 1. 深度校正工具类

```python
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

class DepthCorrector:
    """RealSense 深度校正"""
    def __init__(self, degree=3, alpha=0.3):
        self.degree = degree  # 多项式阶数
        self.alpha = alpha    # 时序平滑系数
        self.model = None
        self.prev_depth = None
    
    def fit(self, depth_map, reference_plane_mask):
        """
        使用参考平面（如地面）拟合校正模型
        
        Args:
            depth_map: (H, W) 原始深度
            reference_plane_mask: (H, W) bool，参考平面区域
        """
        Z_raw = depth_map[reference_plane_mask].reshape(-1, 1)
        
        # RANSAC 拟合平面，估计真实深度
        # 简化版：假设参考平面是水平的
        Z_median = np.median(Z_raw)
        Z_true = np.full_like(Z_raw, Z_median)
        
        # 多项式回归
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(Z_raw)
        
        ransac = RANSACRegressor(random_state=0)
        ransac.fit(X, Z_true)
        
        self.model = (poly, ransac)
    
    def correct(self, depth_map):
        """应用校正"""
        if self.model is None:
            return depth_map
        
        poly, ransac = self.model
        shape = depth_map.shape
        Z_raw = depth_map.flatten().reshape(-1, 1)
        X = poly.transform(Z_raw)
        Z_corrected = ransac.predict(X).reshape(shape)
        
        # 时序平滑
        if self.prev_depth is not None:
            Z_corrected = (self.alpha * Z_corrected + 
                          (1 - self.alpha) * self.prev_depth)
        self.prev_depth = Z_corrected
        
        return Z_corrected
```

**关键点**：
- `RANSACRegressor` 剔除离群点（如道床石块的突起）
- 多项式特征捕捉非线性失真
- 时序平滑避免帧间跳变

#### 2. YOLO-SAM2 检测流程

```python
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class BallastDetector:
    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")
        sam2_checkpoint = "sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.sam2 = SAM2ImagePredictor(
            build_sam2(model_cfg, sam2_checkpoint)
        )
        self.depth_corrector = DepthCorrector()
    
    def detect(self, rgb_image, depth_map):
        """
        Args:
            rgb_image: (H, W, 3) uint8
            depth_map: (H, W) float32, 单位 mm
        
        Returns:
            results: List[dict] 每个检测包含 bbox, mask, label
        """
        # 1. YOLO 检测轨枕
        yolo_results = self.yolo(rgb_image)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()  # (N, 4)
        
        # 2. SAM2 精确分割
        self.sam2.set_image(rgb_image)
        masks, scores, _ = self.sam2.predict(
            box=boxes,  # 用 YOLO 框作为 prompt
            multimask_output=False
        )
        
        # 3. 深度校正
        depth_corrected = self.depth_corrector.correct(depth_map)
        
        # 4. 几何分析
        results = []
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            result = self._analyze_region(
                box, mask, rgb_image, depth_corrected
            )
            results.append(result)
        
        return results
    
    def _analyze_region(self, box, mask, rgb, depth):
        """几何分析单个区域"""
        # 提取轨枕和道床的深度轮廓
        sleeper_mask = mask[0]  # SAM2 输出 (1, H, W)
        
        # 计算轨枕上表面高度（深度最小值）
        sleeper_depths = depth[sleeper_mask]
        H_sleeper = np.percentile(sleeper_depths, 5)  # 5th 百分位
        
        # 计算道床高度（轨枕周围区域）
        x1, y1, x2, y2 = box.astype(int)
        ballast_region = depth[y1:y2, x1:x2]
        ballast_region = ballast_region[~sleeper_mask[y1:y2, x1:x2]]
        H_ballast = np.percentile(ballast_region, 95)  # 95th 百分位
        
        # 判断
        delta_H = H_sleeper - H_ballast
        is_sufficient = delta_H < 50  # mm
        
        return {
            'bbox': box,
            'mask': sleeper_mask,
            'delta_H': delta_H,
            'label': 'sufficient' if is_sufficient else 'insufficient'
        }
```

**工程技巧**：
- 用百分位数代替平均值，抗噪
- SAM2 的 `multimask_output=False` 只输出最佳分割
- 道床区域用"轨枕框内非轨枕像素"定义

#### 3. 3D 可视化

```python
import open3d as o3d

def visualize_detection(rgb, depth, results):
    """3D 可视化检测结果"""
    # 深度图转点云
    H, W = depth.shape
    fx, fy = 600, 600  # RealSense D435 内参（示例）
    cx, cy = W/2, H/2
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 标记不足区域（红色边界框）
    geometries = [pcd]
    for res in results:
        if res['label'] == 'insufficient':
            # ... (添加边界框代码省略)
            pass
    
    o3d.visualization.draw_geometries(geometries)
```

---

## 实验

### 数据集说明
**自建数据集**（论文未开源）：
- **采集方式**：RealSense D435 固定在轨道上方 1.5-2m
- **场景**：英国某铁路线，晴天/阴天，不同道床状态
- **标注**：人工标注轨枕边界框 + 道床充足/不足标签
- **规模**：约 500 张 RGB-D 图像对

**数据预处理**：
1. 深度图插值（RealSense 有孔洞）
2. RGB 图像归一化
3. 剔除运动模糊帧

### 定量评估

| 配置 | Precision | Recall | F1-Score |
|-----|-----------|--------|----------|
| YOLOv8 (RGB only) | 0.99 | 0.49 | 0.66 |
| +Depth (AABB, 严格) | 0.85 | 0.72 | 0.78 |
| +Depth (RBB, 宽松) | 0.82 | 0.80 | 0.81 |

**说明**：
- **AABB**：轴对齐边界框（Axis-Aligned Bounding Box）
- **RBB**：旋转边界框（Rotated Bounding Box），更贴合轨枕形状
- **严格/宽松**：$\Delta H$ 阈值 50mm vs 30mm

**结论**：深度增强后，召回率提升 31-63%，F1 提升 15 个点。

### 定性结果

**成功案例**：
- 光照不均匀的场景（阴影导致 RGB 误判）
- 道床石块颜色与轨枕相近
- 倾斜视角（深度几何更鲁棒）

**失败案例**：
- 极端反光（如雨后积水）→ 深度失效
- 轨枕严重损坏 → SAM2 分割错误
- 快速运动 → 时序平滑引入延迟

---

## 工程实践

### 实际部署考虑

**1. 实时性**
- **YOLOv8n**：30 FPS（RTX 3060）
- **SAM2**：5 FPS（Hiera-Large 模型）
- **瓶颈**：SAM2 的 Transformer 推理

**优化方案**：
```python
# 仅在 YOLO 置信度低时调用 SAM2
if yolo_conf < 0.8:
    mask = sam2.predict(box)
else:
    mask = box_to_mask(box)  # 简单膨胀
```

**2. 硬件需求**
- **GPU**：至少 RTX 3060（6GB 显存）
- **深度相机**：RealSense D435 或 D455
- **存储**：1TB SSD（原始 RGB-D 数据量大）

**3. 内存占用**
- 单帧 RGB-D：$(1920 \times 1080 \times 4) \times 4 = 32$ MB
- 批处理 10 帧：需要 320 MB
- 深度校正历史缓存：50 MB

### 数据采集建议

**1. 相机安装**
- 高度：1.5-2m（太高深度噪声大，太低视野窄）
- 角度：俯仰 20-30°（避免镜面反射）
- 固定方式：防震云台（火车振动影响深度）

**2. 采集时机**
- 避开雨天（积水反光）
- 清晨/傍晚（光照柔和）
- 低速巡检车（避免运动模糊）

### 常见坑

**1. 深度图空洞**
```python
# 问题：RealSense 在纹理不足区域无法计算深度
# 解决：用 OpenCV 的 inpaint 插值
import cv2
mask = (depth == 0)
depth_filled = cv2.inpaint(
    depth.astype(np.float32), 
    mask.astype(np.uint8), 
    inpaintRadius=3, 
    flags=cv2.INPAINT_TELEA
)
```

**2. SAM2 过分割**
```python
# 问题：SAM2 把轨枕的纹理分成多个区域
# 解决：后处理合并小区域
from scipy.ndimage import label
labeled, num = label(mask)
areas = [np.sum(labeled == i) for i in range(1, num+1)]
main_region = np.argmax(areas) + 1
mask = (labeled == main_region)
```

**3. 时序平滑延迟**
```python
# 问题：快速变化场景（如火车加速）会有拖影
# 解决：自适应 alpha
velocity = estimate_camera_velocity()  # 基于视觉里程计
alpha = 0.7 if velocity > 5 else 0.3
```

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 固定相机位置（如巡检车） | 手持相机（深度噪声大） |
| 光照变化大的环境 | 完全黑暗（RGB-D 都失效） |
| 需要精确测量（如合规检查） | 只需粗略判断 |
| 有 GPU 资源 | 嵌入式设备（SAM2 太重） |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **纯 RGB + CNN** | 简单，数据易获取 | 光照敏感，召回率低 | 初步筛选 |
| **激光扫描** | 精度极高（mm 级） | 设备贵，速度慢 | 实验室测试 |
| **本文 RGB-D** | 平衡精度和成本 | 需要深度校正 | 实际巡检 |
| **纯几何规则** | 可解释性强 | 需要完美分割 | 结构化场景 |

---

## 我的观点

### 技术亮点
1. **深度校正是关键**：没有校正，深度的优势体现不出来
2. **YOLO+SAM2 是趋势**：检测+分割的两阶段范式适合工业场景
3. **几何先验很重要**：不是所有问题都要端到端学习

### 工程化难点
- **数据标注成本**：每张图标注轨枕+道床需要 5-10 分钟
- **模型部署**：SAM2 太大，需要蒸馏或替换为轻量分割模型
- **长期维护**：轨道环境变化（如季节、维护后），需要持续更新

### 离实际应用还有多远？
**现状**：
- 实验室条件下效果好（F1 = 0.81）
- 但论文没有测试夜间、雨雪、复杂光照

**还需要**：
1. **大规模数据集**：覆盖各种天气、轨道类型
2. **实时优化**：SAM2 → MobileSAM 或 FastSAM
3. **主动学习**：用少量标注持续改进

**时间线估计**：
- **1 年内**：特定线路的试点应用
- **3 年内**：多线路推广
- **5 年+**：完全自动化巡检

### 值得关注的开放问题
1. **多模态融合**：RGB-D + 振动传感器 + 声学？
2. **弱监督学习**：能否只标注"有问题"的区域？
3. **3D 重建**：从巡检视频重建轨道数字孪生

---

**总结**：这篇论文展示了**深度信息 + 精确分割 + 几何先验**的组合威力。在安全关键的场景下，单纯追求高精度是不够的，召回率（别漏掉缺陷）更重要。深度增强虽然增加了系统复杂度，但换来的是可靠性的大幅提升——这正是从实验室走向实际应用的必经之路。