---
layout: post-wide
title: "接触锚点里程计：让四足机器人不靠视觉也能精准定位"
date: 2026-02-20 09:02:36 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.17393v1
generated_by: Claude Code CLI
---

## 一句话总结

通过把每条接触腿视为"运动学锚点"，仅用 IMU 和电机编码器就能抑制长期漂移，在 700m 闭合回路中实现 1.1% 误差的纯本体感知里程计。

---

## 为什么这个问题重要？

四足机器人在矿洞、地下管道、灾后废墟这些场景中工作时，摄像头被粉尘遮挡，LiDAR 被烟雾散射——此时唯一可靠的传感器只有 IMU 和关节编码器。

现有方法的三大痛点：

- **IMU 积分漂移**：加速度计二次积分后，位置误差以 $t^2$ 速度增长，100 秒后误差轻松超过 50m
- **编码器量化噪声**：低精度编码器在低速运动时脚端速度估计误差大，传统 ZUPT 对此束手无策
- **地形不平导致高度漂移**：机器人在台阶、斜坡上行走后，z 轴估计持续累积误差，仅靠 IMU 根本无法区分"爬坡"和"高度漂移"

这篇论文的核心创新：**把接触腿当作间歇性的世界坐标系位置锚点**，而不只是速度观测。速度约束（ZUPT）告诉滤波器"现在不动"，而位置锚点告诉滤波器"现在在哪"——后者对漂移的抑制能力强得多。

---

## 背景知识

### 里程计方法对比

| 方法 | 传感器 | 优点 | 缺点 |
|------|--------|------|------|
| 视觉里程计 (VO) | 相机 | 精度高 | 光照敏感、烟尘失效 |
| 激光里程计 | LiDAR | 鲁棒 | 成本高、重量大 |
| IMU 积分 | IMU | 无外部依赖 | 漂移快（$O(t^2)$） |
| 腿式里程计（传统） | IMU + 编码器 | 轻量 | 仅速度约束，漂移慢但不消除 |
| **接触锚点法（本文）** | IMU + 编码器 + 力矩 | 主动抑制位置漂移 | 需要力矩传感或估计 |

### 运动学基础

四足机器人的正向运动学将关节角度 $\mathbf{q} \in \mathbb{R}^{12}$ 映射到脚端位置：

$$\mathbf{p}_{foot}^{body} = f_{FK}(\mathbf{q})$$

当脚接触地面时，脚端世界速度为零（零速度约束，ZUPT）：

$$\mathbf{v}_{foot}^{world} = \mathbf{v}_{body}^{world} + \mathbf{R}_{body} \cdot (\boldsymbol{\omega} \times \mathbf{p}_{foot}^{body} + \dot{\mathbf{p}}_{foot}^{body}) = \mathbf{0}$$

这是传统腿式里程计的速度观测。本文更进一步——**记录脚落地的绝对位置**作为位置约束。

---

## 核心方法

### 直觉解释

想象你在黑暗中走路，每踩一步就在地上钉一个钉子。只要你记得每个钉子相对于你身体的位置，就能反推出你走了多远——而且每次踩下新钉子，都会纠正之前的积累误差。这正是"接触锚点"的本质。

系统架构分为四个串联模块：

```
IMU 数据  ─────────────────────────────────────► [IMU 预积分]
                                                      │ 预测位置/速度/姿态
关节角度 → 正向运动学 → 脚端位置                        │
关节力矩 → 接触力估计 → 接触检测 ─► [IK-CKF 足速滤波] → [EKF 量测更新]
                                  ↓                      ↑
                             [锚点管理]  高度聚类修正 ─────┘
                             (记录接触位置 & 置信度)

                         多腿同时接触 → [偏航几何一致性修正]
```

关键在于：**IK-CKF 和 IMU 预积分是分工明确的两步**。IMU 做预测（propagation），足端速度和锚点位置作为观测量做更新（measurement update），二者不能混同。

### 接触检测：基于力矩的足端力估计

不用 F/T 传感器，通过关节力矩反推接触力：

$$\mathbf{F}_{contact} \approx \mathbf{J}^{-T}(\mathbf{q}) \cdot (\boldsymbol{\tau}_{measured} - \boldsymbol{\tau}_{gravity})$$

当 $\|\mathbf{F}_{contact}\| > F_{threshold}$ 时判定为接触。这一简化忽略了动力学项，在低速场景（<1.5m/s）误差可控，高速奔跑时需要加入完整动力学模型。

### 锚点管理与高度聚类

锚点的核心价值在于**高度修正**。水平误差可以靠速度约束缓慢积累，但 z 轴在 IMU 中几乎无法自矫正——IMU 无法区分"重力方向测量误差"和"真实高度变化"。

解决方案是高度聚类（height clustering）：如果最近若干锚点的 z 坐标都接近某一值，新落脚点大概率也在同一支撑面，应向该值修正。引入时间衰减权重，让近期锚点贡献更大：

```python
def _height_cluster_correction(self, new_z: float, current_time: float) -> float:
    """时间衰减高度聚类：识别当前支撑面，抑制 z 轴漂移"""
    weights, heights = [], []
    for anchor in self.anchors[-20:]:  # 只看最近20个锚点
        age = current_time - anchor.timestamp
        if abs(anchor.position[2] - new_z) < self.height_threshold:  # 同一支撑面（5cm）
            w = np.exp(-age / self.time_decay) * anchor.confidence
            weights.append(w)
            heights.append(anchor.position[2])
    
    if not weights:
        return new_z  # 无近似高度 → 新台阶或斜坡，保持原始估计
    return np.average(heights, weights=weights)  # 加权贴合已知支撑面
```

当机器人跨越台阶时，新 z 值远离所有历史锚点，修正权重为零，系统自动识别地形变化——这是一个简洁的自适应设计。

### 容积卡尔曼滤波器（CKF）的角色

CKF 的引入是为了解决**编码器量化噪声对低速足端速度估计的污染**。在 0.1 rad/s 以下的关节速度时，1 LSB 的量化误差可导致脚端速度估计抖动 ±5cm/s。

CKF 用 $2n$ 个容积点代替线性化，无需雅可比矩阵，更稳定地传播非线性运动学：

```python
class IK_CubatureFilter:
    """
    专门用于足端速度估计的 CKF——这是观测量滤波器，不是主状态估计器。
    主状态（位置/速度/姿态）由 IMU 预积分传播，本滤波器输出用于 EKF 量测更新。
    """
    def predict(self, dt: float, imu_gyro: np.ndarray, imu_acc: np.ndarray):
        """预测步：用 IMU 数据传播状态（旋转 → 速度 → 位置）"""
        sigma_pts = self._cubature_points(self.x, self.P)
        propagated = np.array([self._imu_dynamics(pt, imu_gyro, imu_acc, dt)
                                for pt in sigma_pts])
        self.x = propagated.mean(axis=0)
        dev = propagated - self.x
        self.P = (dev.T @ dev) / (2 * self.n) + self.Q * dt

    def update_foot_velocity(self, q: np.ndarray, dq: np.ndarray):
        """量测更新步：用 IK 计算的足端速度作为观测量修正状态"""
        v_foot_measured = self._jacobian(q) @ dq   # IK 足端速度（含量化噪声）
        H = self._observation_jacobian()
        S = H @ self.P @ H.T + self.R_foot_vel     # R_foot_vel 对应编码器噪声
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ (v_foot_measured - H @ self.x)
        self.P = (np.eye(self.n) - K @ H) @ self.P
```

**设计意图**：预测步用 IMU（高频、低噪声短期），更新步用关节编码器（低频、含量化噪声但无积分漂移）。两者互补，比直接积分或直接用编码器都好。

### 偏航角漂移抑制

偏航角（yaw）是里程计最难约束的自由度——IMU 在水平面内无重力参考，积分误差无法自校正。

本文利用**多接触点刚体几何约束**：当多条腿同时接触地面时，每个接触点的世界坐标都满足刚体变换：

$$\hat{\psi} = \arg\min_\psi \sum_{i \in \text{contacts}} \|\mathbf{p}_{foot_i}^{world} - \mathbf{p}_{body}^{world} - \mathbf{R}(\psi) \cdot \mathbf{p}_{foot_i}^{body}\|^2$$

当至少 2 条腿同时有可靠锚点时触发此修正。四足机器人在站立相通常有 2-3 条腿同时接触，这提供了比双足机器人更频繁的偏航约束机会。

---

## 实验结果

### 测试平台

四个不同的四足机器人，覆盖点足和轮腿两种构型：

| 平台 | 类型 | 水平回路 | 闭合误差 | 误差率 |
|------|------|---------|---------|--------|
| Astrall-A | 点足机器人 | ~200m | 0.1638m | **0.082%** |
| Astrall-B | 轮腿机器人 | ~200m | 0.2264m | 0.113% |
| Astrall-C | 轮腿机器人 | ~700m | 7.68m | 1.097% |
| Unitree Go2 | 点足机器人 | ~120m | 2.2138m | 1.845% |

垂直回路（爬坡/台阶）上，z 轴误差控制在 0.1~0.54m 内，高度聚类修正功不可没。

### 各模块贡献分析（消融）

| 配置 | 200m 回路误差 | 关键原因 |
|------|-------------|---------|
| 纯 IMU 积分 | >50m | 加速度二次积分，误差 $\propto t^2$ |
| + 速度 ZUPT | ~2m | 速度约束抑制了速度漂移，但无法纠正已有的位置偏差 |
| + 接触锚点位置约束 | ~0.4m | 每次落脚都"拉"一次位置，漂移变为分段有界 |
| + 高度聚类修正 | ~0.2m | z 轴不再自由漂移，三维误差均衡 |
| + IK-CKF 足速滤波 | **0.16m** | 减少了量化噪声注入量测的污染 |

**为什么 ZUPT 之后还有 2m 误差？** 零速度约束约束的是速度，但速度归零之前积累的位置误差无法消除。打个比方：汽车刹车（速度→0）并不能让已经走偏的轨迹复位。接触锚点的位置约束才等同于"重新对准路标"。

**为什么从 0.4m 到 0.2m 需要高度聚类？** 没有高度修正时，z 轴误差会以稳定速率累积（平坦地面行走约 0.5mm/m），在 200m 后达到 10cm 量级，被水平误差掩盖。加入高度聚类后 z 轴误差近似为零均值随机噪声，总体误差显著降低。

### 点足 vs 轮腿：为什么误差差距这么大？

Astrall-A（点足，200m，0.082%）与 Astrall-C（轮腿，700m，1.097%）的对比揭示了方法的内在局限：

1. **锚点假设的成立条件不同**：点足接触面积小、穿透性强，接触位置确定；轮腿的轮子在接触面上会有滚动，接触点位置本身存在不确定性。
2. **ZUPT 对轮腿失效**：轮子滚动时不能假设足端速度为零，需要根据轮子半径和转速做差分里程计，引入额外误差源。
3. **距离效应**：700m vs 200m 本身就有 3.5 倍的行程差，误差累积本就更多。分离这两个因素，真正由构型造成的误差差距约在 3-5 倍之间。

这说明接触锚点方法对**点足机器人天然更友好**，轮腿机器人需要额外的轮式里程计融合模块才能达到同等精度。

---

## 工程实践

### 完整估计器核心逻辑

```python
class ContactAnchoredOdometry:
    def update(self, imu_data: dict, joint_data: dict, timestamp: float):
        dt = timestamp - self.prev_time

        # 步骤1：IMU 预积分（预测步）——高频传播，误差随时间积累
        self._imu_propagate(imu_data['acc'], imu_data['gyro'], dt)

        # 步骤2：接触检测——基于力矩范数估计接触力
        contact_mask = self._detect_contacts(joint_data['tau'], joint_data['q'])

        # 步骤3：IK-CKF 足端速度观测 + ZUPT 量测更新
        #        足端速度 v_foot = J(q)·dq 作为观测量，修正体速度
        if contact_mask.any():
            v_foot_obs = self._ik_ckf.update_foot_velocity(
                joint_data['q'], joint_data['dq'], contact_mask)
            self._velocity_update(v_foot_obs, contact_mask)  # EKF 量测更新

        # 步骤4：接触锚点位置约束（核心漂移抑制）
        #        落脚时记录世界坐标，后续量测更新拉回位置漂移
        self._update_anchor_constraints(contact_mask, joint_data['q'], timestamp)

        # 步骤5：多接触点偏航几何一致性修正（≥2腿同时接触时触发）
        if contact_mask.sum() >= 2:
            self._yaw_multicontact_correction(contact_mask, joint_data['q'])

        return self.get_state()

    def _update_anchor_constraints(self, contact_mask, q, timestamp):
        """核心：落脚时将锚点位置作为观测量，通过 EKF 量测更新纠正位置漂移"""
        for leg_id in np.where(contact_mask)[0]:
            p_foot_body  = self._forward_kinematics(q, leg_id)
            p_foot_world = self.pos + self.rot.apply(p_foot_body)
            anchor = self.anchor_mgr.update_footfall(
                leg_id, p_foot_world, contact_force=..., timestamp=timestamp)

            # H 矩阵：位置观测量只与状态中的位置分量相关
            H = np.zeros((3, 15)); H[:, :3] = np.eye(3)
            S = H @ self.P @ H.T + np.eye(3) * 0.02   # 锚点位置噪声 ≈ 2cm
            K = self.P @ H.T @ np.linalg.inv(S)
            innovation = anchor.position - (self.pos + self.rot.apply(p_foot_body))
            self.pos += (K @ innovation)[:3]
            self.P   = (np.eye(15) - K @ H) @ self.P
```

### 实时性与硬件需求

整个估计器运行于 1kHz，各模块耗时（i7-1185G7 测试）：

| 模块 | 耗时 |
|------|------|
| IMU 预积分 | <0.1ms |
| 接触检测（力矩计算） | ~0.3ms |
| IK-CKF 更新 | ~0.5ms |
| 锚点管理 | ~0.2ms |
| **总计** | **<1ms** |

嵌入式 ARM Cortex-A72 即可运行，不需要 GPU——这是在资源受限机器人平台上部署的关键优势。

### 常见坑

1. **腿抬起时的锚点残留**：只在腿刚落地的前 50ms 创建锚点，同一腿同一支撑周期不重复更新，防止"锚点被滑动的脚污染"

2. **滑脚导致锚点污染**：用力矩二阶导数检测突变——滑脚时力矩会突然减小再增大，检测到后丢弃可疑锚点

3. **轮腿机器人的特殊处理**：轮子滚动时 ZUPT 失效，需根据轮子半径和转速做差分里程计，与腿式模式切换融合

4. **编码器量化低速影响**：0.1 rad/s 以下的关节速度，1 LSB 量化误差导致脚端速度估计抖动 ±5cm/s，这正是 IK-CKF 滤波要处理的核心问题

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 无视觉/激光传感器 | 需要亚厘米级精度 |
| 室内结构化环境 | 高速奔跑（>3m/s，接触时间极短） |
| 地下矿洞、管道 | 极软地形（泥泞，脚端下陷破坏高度假设） |
| 短中程导航（<1km） | 长距离自主导航（>1km，误差积累） |
| 嵌入式计算平台 | 冰面（零摩擦，锚点假设失效） |

---

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| MIT Cheetah 腿式里程计 | 成熟、开源 | 只用速度约束，无位置锚点 | 一般四足导航 |
| VILENS（视觉+腿式） | 精度高 | 需要相机，光照依赖 | 正常环境导航 |
| LIO-SAM（激光+IMU） | 精度极高 | 成本高、重量大 | 大规模建图 |
| **本文** | 零额外传感器，漂移有界 | 长距离误差积累，轮腿精度下降 | 恶劣感知条件 |

---

## 我的观点

这篇工作解决了一个很实际的问题：**当感知失败时，机器人还能走多远？** 700m 回路 1.1% 的误差已经足够完成很多救援或巡检任务，而且完全不依赖外部感知。

几个值得深入思考的方向：

**高度聚类的隐含假设**是机器人行走在硬质平面上。在沙地、草地等弹性地形，脚端下陷量随负载变化，高度聚类会把"弹性形变"误认为"测量噪声"加以平滑，反而引入系统性误差。如何把地形柔性建模进来是开放问题——一种思路是用接触力的大小来估计下陷量，再修正锚点的 z 坐标。

**点足 vs 轮腿的精度鸿沟**揭示了一个设计取舍：轮腿机器人在平地速度更快，但锚点假设（足端接触位置确定）在轮腿上天然成立度更低。对轮腿机器人，应当设计专用的轮式里程计模块与锚点系统并联融合，而非直接套用点足方案。

**往双足机器人泛化**的挑战在于双足在单脚支撑相几乎没有多接触约束，偏航角修正机会极少。论文声称可以泛化到双足，但实验数据只有四足，需要后续验证——单支撑相时偏航漂移如何约束是关键技术问题。

**与粗粒度定位的融合**是实用化的关键。本体感知里程计天然与楼层地图、Wi-Fi 指纹定位互补：里程计提供短期高频精确估计，粗粒度地图提供长期绝对修正。两者的融合能把"1km 外漂移 10m"的问题变成"每次经过 Wi-Fi 热点修正一次"的分段有界问题。

离产品化还有距离的主要是**鲁棒性**而非精度。1.1% 对点到点导航够用，但精确停靠（如连接充电桩）需要另行解决最后几厘米的精度问题。与视觉里程计的自动切换互备——"有光用视觉，无光用锚点"——是值得工程化的下一步。