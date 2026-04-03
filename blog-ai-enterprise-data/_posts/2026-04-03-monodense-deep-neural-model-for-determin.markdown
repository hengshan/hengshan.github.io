---
layout: post-wide
title: "用 Monodense 神经网络估算商品价格弹性：因果推断遇上单调性约束"
date: 2026-04-03 08:05:14 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2603.29261v1
generated_by: Claude Code CLI
---

## 一句话总结

传统 ML 模型直接拟合价格→销量会学到虚假相关性，Monodense-DL 通过 Double Machine Learning 去除混淆变量、用单调性约束嵌入经济学先验，在无 A/B 测试的观测数据上也能得到可信的价格弹性估算。

---

## 背景：价格弹性很简单，估算它很难

价格弹性（PED）的定义干净利落：

$$\epsilon = \frac{\partial \log Q}{\partial \log P}$$

价格涨 1%，销量变化多少百分比。$\epsilon = -1.5$ 意味着价格涨 10%，销量跌 15%。

知道这个数字有什么用？
- **定价**：弹性低的商品可以提价而不大损销量
- **促销**：弹性高的商品降价效果更明显
- **收入优化**：在弹性绝对值 > 1 的区域，降价反而可以增收

**那为什么不直接跑个回归？**

问题在于，价格不是随机设置的。超市打折是因为预期销量会上升，节假日前涨价是因为需求弹性低。价格和销量同时受到季节、促销、竞品、库存等因素驱动——这叫**内生性（Endogeneity）**。

直接用价格预测销量，模型学到的是相关性而非因果。打折日既有低价又有高销量，模型可能得出"降价→销量下降"的荒谬结论。

---

## 核心挑战：内生性与因果识别

用图来看问题：

```
季节性 / 促销活动
    ↓           ↓
  价格    →    销量
```

价格和销量都被混淆变量（confounder）影响。OLS 弹性估计：

$$\hat{\epsilon}_{OLS} = \epsilon_{\text{真实}} + \underbrace{\text{混淆偏差}}_{\text{这部分可能很大}}$$

消除混淆的金标准是 A/B 测试：随机给不同用户展示不同价格。但现实中零售商不可能对每个 SKU 都做价格实验。**Double Machine Learning（DML）** 是处理观测数据内生性的现代工具。

---

## Double Machine Learning：两步残差化

DML 的核心思想来自 Robinson（1988）和 Chernozhukov 等（2018），设部分线性模型：

$$\log Q_i = \theta \cdot \log P_i + g(\mathbf{X}_i) + \epsilon_i$$
$$\log P_i = m(\mathbf{X}_i) + \eta_i$$

其中 $\mathbf{X}_i$ 是混淆变量，$\theta$ 就是我们要的弹性。

**三步走：**
1. 用 ML（LGBM）从混淆变量预测价格，得到价格残差 $\tilde{P}_i = \log P_i - \hat{m}(\mathbf{X}_i)$
2. 用 ML 从混淆变量预测销量，得到销量残差 $\tilde{Q}_i = \log Q_i - \hat{g}(\mathbf{X}_i)$
3. 对残差做回归：$\tilde{Q}_i = \theta \cdot \tilde{P}_i + \text{noise}$

残差 $\tilde{P}_i$ 代表"不能被混淆变量解释的价格变动"——这才是近似随机化的价格变化，对它的弹性估计才有因果意义。

---

## Monodense 层：把经济学先验编进网络

DML 给了我们"去偏"的残差对。用什么模型来估算第三步？

普通神经网络可以拟合任意函数，包括"价格越高销量越高"这种违反经济学常识的曲线。**Monodense 层**的想法是：在网络结构层面强制单调性约束。

对于弹性估算，我们几乎总是知道：

$$\frac{\partial \hat{Q}}{\partial P} \leq 0$$

实现单调递减的关键：让连接"价格"的权重永远为负数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonodenseLayer(nn.Module):
    """
    单调密集层
    前 n_monotone 个输入维度保持单调性，其余自由学习
    """
    def __init__(self, in_features, out_features, n_monotone, direction='decreasing'):
        super().__init__()
        self.n_monotone = n_monotone
        self.direction = direction
        # 单调部分：softplus 保证非负，乘符号控制方向
        self.W_mono = nn.Parameter(torch.randn(out_features, n_monotone) * 0.1)
        # 非单调部分：无约束
        self.W_free = nn.Parameter(torch.randn(out_features, in_features - n_monotone) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x_mono = x[..., :self.n_monotone]
        x_free = x[..., self.n_monotone:]
        sign = -1.0 if self.direction == 'decreasing' else 1.0
        W_constrained = sign * F.softplus(self.W_mono)
        return (F.linear(x_mono, W_constrained) +
                F.linear(x_free, self.W_free) +
                self.bias)
```

**为什么用 `softplus` 而不是 `abs` 或 `relu`？**
- `abs`：梯度在 0 处不连续
- `relu`：零梯度区域太大，权重容易卡死
- `softplus`：处处可微，0 附近有非零梯度，训练更稳定

---

## 完整 Monodense-DL 架构

论文架构把三种组件串联：Embedding 层处理商品 ID 等离散特征，Dense 层学习非线性表示，Monodense 层联合特征和价格残差并强制单调性。

```python
class MonodenseDL(nn.Module):
    """
    输入：商品 ID、品类 ID、数值特征 + 价格残差
    输出：销量残差预测（θ 从这里恢复）
    """
    def __init__(self, num_items, num_cats, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.cat_embed = nn.Embedding(num_cats, embed_dim // 2)

        feat_dim = embed_dim + embed_dim // 2 + 4  # 4 个数值特征
        self.dense = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        # 价格残差是第一个输入维度（对应 n_monotone=1）
        mono_in = 1 + hidden_dim // 2
        self.monodense = MonodenseLayer(mono_in, 32, n_monotone=1, direction='decreasing')
        self.out = nn.Linear(32, 1)

    def forward(self, item_id, cat_id, num_feats, price_resid):
        feats = torch.cat([self.item_embed(item_id),
                           self.cat_embed(cat_id),
                           num_feats], dim=-1)
        dense_out = self.dense(feats)
        # 价格残差放最前面，匹配 n_monotone=1
        mono_in = torch.cat([price_resid.unsqueeze(-1), dense_out], dim=-1)
        return self.out(F.relu(self.monodense(mono_in))).squeeze(-1)
```

---

## 完整训练流程

```python
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

def dml_first_stage(X_conf, log_price, log_qty, n_splits=5):
    """DML 第一阶段：交叉验证残差化，避免过拟合偏差"""
    price_resid = np.zeros_like(log_price)
    qty_resid = np.zeros_like(log_qty)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(X_conf):
        X_tr, X_val = X_conf[tr_idx], X_conf[val_idx]
        m = LGBMRegressor(n_estimators=200, verbose=-1)
        m.fit(X_tr, log_price[tr_idx])
        price_resid[val_idx] = log_price[val_idx] - m.predict(X_val)
        g = LGBMRegressor(n_estimators=200, verbose=-1)
        g.fit(X_tr, log_qty[tr_idx])
        qty_resid[val_idx] = log_qty[val_idx] - g.predict(X_val)
    return price_resid, qty_resid


def train_monodense(model, price_resid, qty_resid, item_ids, cat_ids, num_feats,
                    n_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pr = torch.FloatTensor(price_resid)
    qr = torch.FloatTensor(qty_resid)
    ii, ci = torch.LongTensor(item_ids), torch.LongTensor(cat_ids)
    nf = torch.FloatTensor(num_feats)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(ii, ci, nf, pr)
        loss = F.mse_loss(pred, qr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪很重要
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}")
    return model
```

---

## 合成数据验证

在真实数据之前，先在已知弹性的合成数据上验证是否能正确恢复：

```python
def make_synthetic(n=5000, true_elast=-1.5, seed=42):
    rng = np.random.default_rng(seed)
    season = rng.standard_normal(n)
    log_price = 0.3 * season + 0.1 * rng.standard_normal(n)   # 价格受季节影响
    log_qty = true_elast * log_price + 0.5 * season + rng.standard_normal(n)
    return log_price, log_qty, season.reshape(-1, 1)

log_p, log_q, X = make_synthetic(true_elast=-1.5)

# OLS（有偏）
from numpy.linalg import lstsq
ols_e = lstsq(log_p.reshape(-1, 1), log_q, rcond=None)[0][0]
print(f"OLS 弹性:  {ols_e:.3f}   (真实: -1.5)")   # 预期偏高，约 -0.8~-1.1

# DML 残差化后
pr, qr = dml_first_stage(X, log_p, log_q)
dml_e = lstsq(pr.reshape(-1, 1), qr, rcond=None)[0][0]
print(f"DML 弹性:  {dml_e:.3f}   (真实: -1.5)")   # 预期接近 -1.5
```

OLS 因为季节效应的正向混淆，估计值会明显偏高（绝对值偏小）。DML 通过残差化把混淆洗掉，能更接近真实弹性。

---

## 调试指南

### 弹性估算值明显不对

**症状**：弹性是正数，或绝对值远大于 5

**原因 1：第一阶段欠拟合**。混淆变量没充分去除。增加 LGBM 的 `n_estimators`，补充更多混淆特征（节假日、竞品价格、历史趋势）。

**原因 2：价格变动太少**。该 SKU 价格几乎不变，残差接近 0，噪声主导估算。过滤价格变动标准差极小的商品，或用品类级别弹性代替。

**原因 3：时间跨度太短**。至少需要覆盖完整季节周期（1 年以上）。

### DML 第一阶段必须用交叉验证

不用交叉验证，直接在全量数据拟合再预测残差，会导致残差被人为压缩，第二阶段弹性估计偏向 0。`KFold` 是必须的，不是可选的。

### Monodense 层不收敛

如果初始化标准差太小（`0.01`），softplus 输出集中在接近 0 的区域，梯度极小。改用 `0.1` 初始化可以让训练起步更快。

### 超参数参考

| 参数 | 推荐值 | 敏感度 | 说明 |
|------|--------|--------|------|
| LGBM n_estimators | 100–300 | 中 | 太高过拟合，太低欠拟合 |
| DML k_folds | 5 | 低 | 5 折足够 |
| embed_dim | 8–32 | 低 | 商品数量越大可以大一点 |
| hidden_dim | 32–128 | 中 | 从 64 开始 |
| lr | 1e-3 | 高 | 先试这个值 |
| 梯度裁剪 | 1.0 | 中 | 一定要加，防止不稳定 |

---

## 什么时候用 / 不用？

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有 1 年以上历史价格-销量数据 | 刚上架的新品（历史数据稀少） |
| 价格有自然波动（促销、季节调整） | 价格极少变动的商品 |
| 无法做 A/B 测试的实际业务 | 能做随机价格实验（实验更可信） |
| 需要商品级别弹性 | 只需品类级粗略估算（简单 DML 够用）|
| 要编码价格单调性先验 | 纯预测销量（不需要弹性的经济语义）|

---

## 我的看法

这篇论文解决的问题是真实的——大多数零售公司没有价格 A/B 测试，却需要做定价决策。DML 在计量经济学里已经是标准工具，把它和能编码先验的神经网络结合是合理方向。

但有几点值得注意：

**Monodense 对强内生性的帮助有限**。单调性约束只是归纳偏置，不能替代正确的因果识别。第一阶段的 LGBM 如果没把混淆变量洗干净，加了 Monodense 也没用。

**嵌入层需要足够数据**。长尾商品（几十条记录）的嵌入向量学不好，弹性估计方差会很大。这时应该 fallback 到品类级别弹性，或者用更强的正则化。

**在你的数据上做消融实验再下结论**。论文说 Monodense-DL 优于 LGBM 和 DML，但这个结论是在他们的多品类零售数据上的。你的数据分布不同，结论可能不一样。在替换现有方案之前，跑一跑你自己数据上的对比实验。

调好第一阶段往往比优化神经网络结构更关键。如果 DML 残差化做得不好，再好的网络也救不了有偏的弹性估计。