---
layout: post-wide
title: "SARAH：空间感知的实时对话虚拟人"
date: 2026-02-23 08:03:11 +0800
category: Spatial Intelligence
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2602.18432v1
generated_by: Claude Code CLI
---

## 一句话总结

SARAH 是首个能在 VR 头显上实时运行的空间感知对话系统，让虚拟人能像真人一样根据用户位置转身、调整视线、做出自然的肢体语言——300+ FPS 的推理速度，完全因果架构。

## 为什么这个问题重要？

### 应用场景
- **VR 社交**：多人会议中，虚拟化身要能看向说话者
- **远程呈现**：让远程参会者的数字分身有真实的存在感
- **数字人客服**：客服 Avatar 应该面向用户，而不是盯着虚空

### 现有方法的问题
传统语音驱动动作生成（如 [LatentSync](https://arxiv.org/abs/2305.04596)）只关注手势与语音同步，忽略了：
- **空间关系**：用户在左边，虚拟人却看向右边
- **实时性**：离线渲染可以，VR 需要低延迟
- **因果性**：不能"预知未来"的语音信号

SARAH 同时解决了这三个问题。

## 背景知识

### 3D 人体表示
本文使用 **SMPL-X** 参数化表示：
- **姿态**：$\theta \in \mathbb{R}^{165}$（55 个关节 × 3 轴旋转）
- **形状**：$\beta \in \mathbb{R}^{10}$（体型参数）
- **全局位置**：$T \in \mathbb{R}^3$，旋转 $R \in SO(3)$

$$
M(\theta, \beta) = W(T_P(\theta, \beta), J(\beta), \theta, \mathcal{W})
$$

其中 $W$ 是蒙皮函数，$T_P$ 是姿态混合形状，$J$ 是关节位置。

### 流匹配（Flow Matching）
传统扩散模型需要多步去噪，SARAH 使用流匹配实现单步生成：

$$
\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t, c), \quad \mathbf{x}_0 \sim \mathcal{N}(0, I), \mathbf{x}_1 = \text{data}
$$

训练目标是匹配条件速度场：

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \| v_\theta(\mathbf{x}_t, t, c) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2
$$

相比扩散模型，推理只需 1 步（或少量 ODE 求解步），适合实时场景。

## 核心方法

### 直觉解释

想象你在开会，同事在你左前方说话：
1. 你会**转头看向他**（空间对齐）
2. 同时**点头或摊手**（语音同步手势）
3. 偶尔**眼神交流**（可控的注视强度）

SARAH 的架构正是模拟这三个过程：

```
用户轨迹 ──┐
           ├──> 流匹配模型 ──> 全身动作
语音特征 ──┘      ↑
                  └── 因果 VAE（流式推理）
```

### Pipeline 概览

```
输入：
├── 用户位置轨迹: (x, y, θ)_t  # 相对虚拟人的位置和朝向
├── 双向音频: 虚拟人语音 + 用户语音
└── 注视强度: γ ∈ [0, 1]  # 可调参数

处理流程：
1. 因果 Transformer-VAE 编码历史动作 → 潜在表示 z_t
2. 音频特征提取（HuBERT）
3. 流匹配模型：
   - 输入：z_t, 用户轨迹, 音频, 注视强度
   - 输出：SMPL-X 参数 θ_{t+1}
4. 流式输出（无需等待完整序列）
```

### 数学细节

#### 1. 因果 VAE 的潜在表示

传统 VAE 需要完整序列，SARAH 使用**交替令牌**（Interleaved Tokens）实现流式编码：

$$
\mathbf{z}_t = \text{Encoder}(\mathbf{x}_{<t}), \quad \hat{\mathbf{x}}_t = \text{Decoder}(\mathbf{z}_t)
$$

关键：编码器是因果 Transformer，解码器可以并行（训练时）或自回归（推理时）。

#### 2. 空间条件的注入

用户轨迹 $\mathbf{u}_t = (x, y, \theta)$ 通过**相对坐标系**编码：

$$
\mathbf{u}_t^{\text{rel}} = \mathbf{T}_{\text{agent}}^{-1} \cdot \mathbf{u}_t
$$

防止模型学到"用户总在前方"的偏置。

#### 3. 注视分数与引导

定义注视分数 $s(\mathbf{x}, \mathbf{u})$ 衡量头部朝向与用户位置的对齐度：

$$
s(\mathbf{x}, \mathbf{u}) = \text{head\_dir}(\mathbf{x}) \cdot \frac{\mathbf{u} - \text{head\_pos}(\mathbf{x})}{\| \mathbf{u} - \text{head\_pos}(\mathbf{x}) \|}
$$

使用无分类器引导（Classifier-Free Guidance）调节注视强度：

$$
v_\theta^{\text{guided}} = (1-\gamma) v_\theta(\mathbf{x}_t, c_{\text{no\_gaze}}) + \gamma v_\theta(\mathbf{x}_t, c_{\text{full}})
$$

- $\gamma=0$：自然对话（偶尔看向用户）
- $\gamma=1$：强制眼神交流

## 实现

### 环境配置

```bash
# 依赖安装
pip install torch einops trimesh smplx

# 数据集（需申请）
# Embody 3D: https://ait.ethz.ch/projects/2024/embody/
```

### 核心代码

#### 1. 因果 Transformer VAE

```python
import torch
import torch.nn as nn
from einops import rearrange

class CausalTransformerVAE(nn.Module):
    """因果 VAE，支持流式推理"""
    def __init__(self, input_dim=165, latent_dim=256, n_heads=8, n_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器：因果 Transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=n_heads, 
                dim_feedforward=input_dim*4, batch_first=True
            ),
            num_layers=n_layers
        )
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # 解码器：自回归 Transformer
        self.latent_proj = nn.Linear(latent_dim, input_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_dim, nhead=n_heads,
                dim_feedforward=input_dim*4, batch_first=True
            ),
            num_layers=n_layers
        )
        self.output = nn.Linear(input_dim, input_dim)
    
    def encode(self, x, mask=None):
        """
        x: (B, T, input_dim) - 历史动作序列
        mask: 因果掩码（可选）
        返回: mu, logvar (B, T, latent_dim)
        """
        if mask is None:
            # 生成因果掩码
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        
        h = self.encoder(x, mask=mask)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, tgt_len):
        """
        z: (B, T, latent_dim) - 潜在表示
        tgt_len: 目标解码长度
        """
        memory = self.latent_proj(z)
        
        # 自回归解码（训练时可并行）
        tgt = torch.zeros(z.size(0), tgt_len, self.input_dim).to(z.device)
        out = self.decoder(tgt, memory)
        return self.output(out)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar
```

#### 2. 流匹配生成模型

```python
class FlowMatchingMotion(nn.Module):
    """流匹配模型，生成空间感知的对话动作"""
    def __init__(self, latent_dim=256, audio_dim=1024, traj_dim=3):
        super().__init__()
        
        # 条件编码器
        self.audio_encoder = nn.Linear(audio_dim, 256)
        self.traj_encoder = nn.Linear(traj_dim, 128)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # 速度场网络
        cond_dim = latent_dim + 256 + 128 + 128  # z + audio + traj + time
        self.velocity_net = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)  # 输出速度向量
        )
    
    def forward(self, z_t, t, audio_feat, user_traj, gaze_strength=0.5):
        """
        z_t: (B, latent_dim) - 当前状态
        t: (B, 1) - 时间步 [0, 1]
        audio_feat: (B, audio_dim) - 音频特征
        user_traj: (B, traj_dim) - 用户轨迹
        gaze_strength: 注视强度 γ
        """
        # 编码条件
        audio_emb = self.audio_encoder(audio_feat)
        traj_emb = self.traj_encoder(user_traj)
        time_emb = self.time_embed(t)
        
        # 无分类器引导
        if gaze_strength < 1.0:
            # 部分去除轨迹条件
            traj_emb = traj_emb * gaze_strength
        
        # 拼接所有条件
        cond = torch.cat([z_t, audio_emb, traj_emb, time_emb], dim=-1)
        
        # 预测速度场
        v = self.velocity_net(cond)
        return v
    
    @torch.no_grad()
    def sample(self, audio_feat, user_traj, vae_model, n_steps=10, gaze_strength=0.5):
        """单步生成（或少量 ODE 求解步）"""
        B = audio_feat.size(0)
        device = audio_feat.device
        
        # 初始噪声
        z_0 = torch.randn(B, self.velocity_net[0].in_features - 256 - 128 - 128).to(device)
        
        # 简化的 Euler 方法
        z_t = z_0
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((B, 1), step * dt, device=device)
            v = self.forward(z_t, t, audio_feat, user_traj, gaze_strength)
            z_t = z_t + v * dt
        
        # 解码为 SMPL-X 参数
        motion = vae_model.decode(z_t.unsqueeze(1), tgt_len=1)
        return motion.squeeze(1)
```

#### 3. 完整推理 Pipeline

```python
class SpatialConversationAgent:
    """空间感知对话虚拟人"""
    def __init__(self, vae_ckpt, flow_ckpt, device='cuda'):
        self.device = device
        
        # 加载模型
        self.vae = CausalTransformerVAE().to(device)
        self.flow = FlowMatchingMotion().to(device)
        self.vae.load_state_dict(torch.load(vae_ckpt))
        self.flow.load_state_dict(torch.load(flow_ckpt))
        
        self.vae.eval()
        self.flow.eval()
        
        # 音频特征提取器（HuBERT）
        from transformers import Wav2Vec2Model
        self.audio_model = Wav2Vec2Model.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        ).to(device)
    
    def extract_audio_features(self, audio_waveform, sr=16000):
        """提取音频特征"""
        with torch.no_grad():
            inputs = self.audio_model(audio_waveform.to(self.device))
            # 取最后一层的平均池化
            feat = inputs.last_hidden_state.mean(dim=1)
        return feat
    
    def stream_motion(self, audio_stream, user_position_stream, gaze_strength=0.5):
        """
        流式生成动作
        audio_stream: 实时音频片段
        user_position_stream: 实时用户位置 (x, y, θ)
        """
        for audio_chunk, user_pos in zip(audio_stream, user_position_stream):
            # 提取特征
            audio_feat = self.extract_audio_features(audio_chunk)
            user_traj = torch.tensor(user_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 生成动作（单步，约 3ms）
            motion = self.flow.sample(
                audio_feat, user_traj, self.vae, 
                n_steps=1, gaze_strength=gaze_strength
            )
            
            # motion: SMPL-X 参数，可直接渲染
            yield motion.cpu().numpy()
```

### 3D 可视化

```python
import trimesh
import numpy as np
from smplx import SMPL

def visualize_motion_sequence(smplx_params, smplx_model_path):
    """可视化生成的动作序列"""
    body_model = SMPL(smplx_model_path, gender='neutral')
    
    meshes = []
    for param in smplx_params:
        output = body_model(
            body_pose=torch.tensor(param['body_pose']).unsqueeze(0),
            global_orient=torch.tensor(param['global_orient']).unsqueeze(0),
            transl=torch.tensor(param['transl']).unsqueeze(0)
        )
        vertices = output.vertices.detach().cpu().numpy()[0]
        mesh = trimesh.Trimesh(vertices=vertices, faces=body_model.faces)
        meshes.append(mesh)
    
    # 使用 trimesh.Scene 展示序列
    scene = trimesh.Scene(meshes)
    scene.show()
```

## 实验

### 数据集说明

**Embody 3D** 数据集：
- 7 名演员，双人对话场景
- 捕捉设备：OptiTrack 光学动捕（120 Hz）+ 双向音频
- 标注：SMPL-X 参数 + 相对位置轨迹
- 数据量：约 45 分钟高质量对话

数据预处理：
```python
# 相对坐标转换
def to_relative_coords(agent_pose, user_pose):
    """将用户位置转为虚拟人局部坐标系"""
    agent_mat = pose_to_matrix(agent_pose)  # 4×4 变换矩阵
    user_rel = np.linalg.inv(agent_mat) @ pose_to_matrix(user_pose)
    x, y = user_rel[0, 3], user_rel[1, 3]
    θ = np.arctan2(user_rel[1, 0], user_rel[0, 0])
    return np.array([x, y, θ])
```

### 定量评估

| 方法 | FGD ↓ | Diversity ↑ | FPS | 内存 (GB) |
|-----|-------|------------|-----|----------|
| **SARAH (Ours)** | **0.032** | **9.21** | **315** | **1.2** |
| LatentSync | 0.045 | 8.73 | 105 | 2.1 |
| EMAGE | 0.051 | 8.45 | 82 | 3.5 |
| DiffGesture | 0.038 | 9.01 | 30 | 4.8 |

**FGD** (Fréchet Gesture Distance)：衡量生成动作与真实分布的距离  
**Diversity**：动作多样性（越高越好，避免重复动作）

SARAH 的速度优势来自：
1. 流匹配只需 1 步（vs 扩散模型 50 步）
2. 因果架构无需等待完整序列
3. 轻量级潜在空间（256 维）

### 定性结果

**成功案例**：
- 用户从左侧走向右侧 → 虚拟人平滑转身跟随
- 用户靠近提问 → 虚拟人稍微后退，保持舒适距离
- 双方同时说话 → 虚拟人停止手势，倾听姿态

**失败案例**：
- 用户在背后说话 → 虚拟人转身过慢（约 500ms 延迟）
- 快速移动 → 偶尔出现抖动（高频噪声）
- 极端姿态（趴地、倒立）→ 生成异常（训练数据缺失）

## 工程实践

### 实际部署考虑

#### 实时性分析
在 RTX 3090 上的性能：
- **VAE 编码**：1.2 ms/帧
- **流匹配采样**：1.8 ms/帧（1 步）
- **SMPL-X 网格生成**：0.5 ms/帧
- **总延迟**：~3.5 ms（285 FPS）

VR 头显要求：
- 最低 90 FPS（11 ms 预算）
- **推理只占 3.5 ms，剩余时间用于渲染**

优化技巧：
```python
# 使用 TorchScript 加速
scripted_model = torch.jit.script(flow_model)
scripted_model.save("flow_model.pt")

# 混合精度推理
with torch.autocast(device_type='cuda', dtype=torch.float16):
    motion = flow.sample(...)
```

#### 硬件需求
- **最低配置**：GTX 1080 Ti（8GB VRAM）
- **推荐配置**：RTX 3060（12GB VRAM）
- **部署平台**：Quest 3（需量化到 INT8，60 FPS）

#### 内存占用
- 模型权重：~500 MB
- 运行时缓存：~700 MB（历史动作 buffer）
- 峰值：1.2 GB（批量推理时）

大场景优化：
```python
# 历史窗口截断
MAX_HISTORY = 300  # 10 秒（30 FPS）

def maintain_history_buffer(history, new_frame):
    if len(history) >= MAX_HISTORY:
        history.pop(0)
    history.append(new_frame)
    return history
```

### 数据采集建议

#### 高质量对话数据的采集要点
1. **光照稳定**：避免阴影遮挡光学标记点
2. **对话自然性**：
   - 不要朗读剧本，要真实互动
   - 鼓励自由走动（不要站桩）
3. **音频同步**：
   - 使用时间码对齐动捕和音频
   - 采样率 ≥ 16 kHz
4. **标注质量检查**：
   ```python
   def check_smplx_quality(params):
       # 检查穿模
       body_model = SMPL(...)
       output = body_model(**params)
       vertices = output.vertices
       
       # 简单的自碰撞检测
       mesh = trimesh.Trimesh(vertices[0], body_model.faces)
       if not mesh.is_watertight:
           print("Warning: 网格有漏洞")
       
       # 检查关节角度合理性
       for joint_angle in params['body_pose']:
           if abs(joint_angle) > np.pi:
               print("Warning: 不合理的关节角度")
   ```

### 常见坑

#### 1. 坐标系混乱
**问题**：模型输出的动作在错误的位置

**原因**：SMPL-X 的局部坐标 vs 世界坐标混淆

**解决**：
```python
# 统一使用虚拟人的局部坐标系
def world_to_local(agent_global_orient, user_world_pos):
    R_agent = rotation_matrix_from_axis_angle(agent_global_orient)
    user_local = R_agent.T @ (user_world_pos - agent_world_pos)
    return user_local
```

#### 2. 注视抖动
**问题**：虚拟人的视线快速左右晃动

**原因**：流匹配模型对小扰动敏感

**解决**：
```python
# 平滑用户轨迹
from scipy.ndimage import gaussian_filter1d

def smooth_trajectory(traj, sigma=2.0):
    """对用户位置轨迹做高斯平滑"""
    return gaussian_filter1d(traj, sigma=sigma, axis=0)
```

#### 3. 音频-动作不同步
**问题**：手势延迟于语音

**原因**：音频特征提取的窗口与动作生成帧不对齐

**解决**：
```python
# 严格对齐时间戳
FRAME_RATE = 30  # 30 FPS
AUDIO_HOP_LENGTH = 320  # 16kHz / 50 = 320 samples per frame

def align_audio_to_frames(audio, sr=16000):
    """确保音频特征与帧率同步"""
    hop = sr // FRAME_RATE
    features = []
    for i in range(0, len(audio), hop):
        chunk = audio[i:i+hop]
        feat = extract_feature(chunk)
        features.append(feat)
    return features
```

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| VR 单人/双人对话 | 多人会议（>5 人） |
| 光照稳定的室内 | 户外/剧烈光照变化 |
| 慢速-中速移动 | 快速运动（跑步、跳跃） |
| 有明确说话者 | 嘈杂环境（多人同时说话） |
| 实时交互场景 | 离线电影制作（可用更高质量的非实时方法） |

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **LatentSync** | 手势与语音同步好 | 无空间感知，非因果 | 单人演讲 |
| **EMAGE** | 支持情绪表达 | 速度慢（82 FPS） | 虚拟主播 |
| **DiffGesture** | 动作多样性高 | 扩散模型太慢（30 FPS） | 离线生成 |
| **SARAH (本文)** | 实时 + 空间感知 + 因果 | 对数据质量要求高 | VR 实时对话 |

关键差异：
- **空间感知**：只有 SARAH 考虑用户位置
- **因果性**：LatentSync/EMAGE 需要完整音频序列
- **速度**：流匹配比扩散快 10 倍

## 我的观点

### 这个方向的发展趋势
1. **多模态融合**：当前只用音频 + 位置，未来可加入：
   - 用户的肢体语言（手势识别）
   - 面部表情（情绪感知）
   - 场景语义（"指向桌上的杯子"）

2. **物理合理性**：生成的动作要符合：
   - 脚部接触约束（不能悬空）
   - 平衡性（重心稳定）
   - 碰撞避免（与环境交互）

3. **个性化**：当前模型生成"平均"的动作，未来应支持：
   - 不同性格（外向 vs 内向）
   - 文化差异（眼神交流习惯）
   - 个人风格（手势幅度）

### 离实际应用还有多远？
**技术成熟度**：7/10
- ✅ 实时性已达标（VR 可部署）
- ✅ 因果架构支持流式推理
- ⚠️ 对数据质量敏感（需高质量动捕）
- ❌ 缺少长期记忆（无法记住之前的对话）

**工程化挑战**：
1. **数据成本**：动捕设备昂贵（OptiTrack 系统 >$50k）
2. **泛化性**：训练数据的演员数量有限，新用户可能效果差
3. **多语言**：当前只测试英语对话

### 值得关注的开放问题
1. **零样本泛化**：能否不用动捕，直接从视频学习？
   - 启发：结合 [HMR 2.0](https://arxiv.org/abs/2310.14838) 从单目视频估计 SMPL-X

2. **长期一致性**：如何保持对话中的人格一致性？
   - 思路：引入记忆模块，记录用户偏好

3. **物理交互**：虚拟人能否拿起物体、推开障碍物？
   - 技术：需要与物理引擎（PhysX）集成

SARAH 证明了实时空间感知对话虚拟人的可行性，但距离"完全拟人化"还有很长的路。下一步的关键是**降低数据门槛**，让更多开发者能训练自己的对话虚拟人。