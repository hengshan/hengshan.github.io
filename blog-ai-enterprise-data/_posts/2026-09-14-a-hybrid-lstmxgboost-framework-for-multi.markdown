---
layout: post-wide
title: "LSTM + XGBoost 混合模型预测股票收益：当深度学习遇上梯度提升树"
date: 2026-09-14 08:04:17 +0800
category: AI
author: Hank Li
use_math: true
source_url: https://arxiv.org/abs/2609.13125v1
generated_by: Claude Code CLI
---

这个话题其实是监督学习/时间序列回归（LSTM+XGBoost预测股票收益），并非强化学习，所以我把写作框架里的PPO/SAC/Bellman方程等RL专属内容替换成了对应的时序预测语境，但保留了"承认脆弱性、调试经验优先、基线为王、诚实报告"的精神。以下是完整博客：


## 一句话总结

用LSTM把过去60天的市场数据压缩成一个64维的"时序摘要"（embedding），再把这个摘要和手工技术指标拼接起来，喂给XGBoost做最终回归——本质上是"深度学习做特征工程，树模型做决策"的stacking套路，用来缓解金融时间序列信噪比低、非平稳的老大难问题。

## 背景：为什么需要这个混合架构？

股票收益预测有三个众所周知的坑：

- **非平稳**：训练集学到的分布，测试集可能完全变了（牛市变熊市）。
- **低信噪比**：日频收益里真正的可预测信号可能只占方差的几个百分点，剩下全是噪声。
- **样本量悖论**：K线数据看起来很多，但真正独立的样本（不同的市场周期）其实很少。

纯LSTM端到端预测在这种数据上很容易出问题：要么梯度学不动（模型退化成预测历史均值），要么在训练集上过拟合出一堆虚假模式，测试集直接崩。纯XGBoost这类树模型倒是不容易过拟合，但它天然不理解"序列"——你喂给它的每一行数据都是独立的，想让它感知时序依赖，只能靠人工构造滞后特征（lag features），费时费力还不一定构造得好。

这篇论文（[arXiv:2609.13125](https://arxiv.org/abs/2609.13125v1)）的核心insight很朴素：**别让LSTM直接做最终预测，让它只负责"读懂序列、压缩成向量"，最终决策交给XGBoost**。具体做法是用两层LSTM（64个隐藏单元）处理60天滑动窗口的5个市场特征，取出64维的隐藏状态作为embedding，再和14个手工技术指标拼成78维特征向量，喂给一个网格搜索调过参的XGBoost回归器。

诚实地说，这不是一个新算法，而是一个特征工程/stacking流水线。论文本身也很诚实地报告了一个关键事实：**在大多数股票上，混合模型只是"勉强持平或略微超过"纯XGBoost基线**，真正的提升主要体现在相对纯LSTM基线上（30天窗口RMSE 0.0949，约为纯LSTM的三分之一）。这说明LSTM单独做端到端预测确实很挣扎，但混合模型是否比一个调好的树模型有本质提升，证据并不算特别有力。

论文还报告了一个容易让人误读的数字：365天窗口方向准确率高达97.6%。但论文自己也指出，这很大程度上只是因为样本期内长期正收益的"基础命中率"本来就很高——如果你无脑预测"永远上涨"，准确率也差不多。这是本文重点要拆解的一个陷阱。

## 算法原理

### 直觉解释

把LSTM想象成一个"阅读理解器"：它读完过去60天的走势，写出一份64个数字的"读后感摘要"。这份摘要和传统的技术指标（均线、RSI、MACD等）被并排交给XGBoost这个"决策者"，让它综合两类信息做出最终判断。LSTM提供的是"序列里隐藏的动态模式"，技术指标提供的是"人类几十年总结出的经验规则"，两者互补。

### 数学推导

给定第 $t$ 天的滑动窗口特征矩阵：

$$
X_t = [x_{t-59}, x_{t-58}, \dots, x_t] \in \mathbb{R}^{60 \times 5}
$$

LSTM编码器输出最后一层隐藏状态作为embedding：

$$
h_t = \text{LSTM}(X_t) \in \mathbb{R}^{64}
$$

拼接14维手工技术指标 $z_t$，得到78维混合特征：

$$
v_t = [h_t \, ; \, z_t] \in \mathbb{R}^{78}
$$

XGBoost回归器输出最终预测：

$$
\hat{y}_t = f_{\text{XGB}}(v_t)
$$

LSTM本身需要先训练出有意义的隐藏状态，论文的做法是让LSTM也附带一个"辅助任务"——直接用一个线性层从embedding预测下一步收益，训练目标是均方误差：

$$
\mathcal{L}_{\text{LSTM}} = \frac{1}{N}\sum_{t} (y_t - \text{Head}(h_t))^2
$$

训练完成后，Head层被丢弃，只保留 $h_t$ 作为特征，交给XGBoost二次训练。

### 与其他方法的关系

这本质上是机器学习竞赛（比如Kaggle）里常见的"神经网络做表征学习 + 树模型做最终决策"套路的一个应用，和纯端到端深度学习（如Temporal Fusion Transformer直接输出预测）、纯树模型（手工特征+XGBoost）相比：

- 比纯LSTM更稳：树模型对噪声、离群点更鲁棒，不容易被几个极端交易日带偏。
- 比纯XGBoost理论上多了序列建模能力，但论文的实证结果说明这个增量在很多股票上并不明显。

## 实现

以下代码基于论文的结构复现核心流程，帮助理解算法骨架。论文摘要中没有提到公开代码仓库，也没有说明14只股票的具体代码和确切的技术指标定义，所以下面用几何布朗运动生成的合成数据代替真实行情，方便直接运行，不依赖网络。真实场景下把数据源换成你自己的行情数据即可。

### 最小可运行版本

```python
import numpy as np, torch, torch.nn as nn
import xgboost as xgb

# 1. 造数据：单只股票，60天窗口预测下一天收益
np.random.seed(0)
prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 1000)))
returns = np.diff(prices) / prices[:-1]
window = 60
X, y = [], []
for t in range(window, len(returns)):
    X.append(returns[t - window:t])
    y.append(returns[t])
X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, 60, 1)
y = torch.tensor(np.array(y), dtype=torch.float32)

# 2. LSTM提取embedding（这里用一维特征简化，论文用5维原始特征）
lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
for _ in range(20):
    opt.zero_grad()
    _, (h_n, _) = lstm(X)
    pred = h_n[-1].mean(dim=1)  # 简化：直接用embedding均值当预测
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    opt.step()

# 3. 拿embedding喂给XGBoost
with torch.no_grad():
    _, (h_n, _) = lstm(X)
    embeddings = h_n[-1].numpy()  # (N, 64)

reg = xgb.XGBRegressor(max_depth=3, n_estimators=200)
reg.fit(embeddings[:800], y.numpy()[:800])
pred = reg.predict(embeddings[800:])
print("RMSE:", np.sqrt(np.mean((pred - y.numpy()[800:]) ** 2)))
```

### 完整实现

数据与特征工程（多股票面板 + 技术指标）：

```python
import numpy as np
import pandas as pd

def generate_synthetic_panel(n_stocks=14, n_days=1500, seed=42):
    """用几何布朗运动模拟多只股票，附带简单成交量"""
    rng = np.random.default_rng(seed)
    sectors = ["Tech", "Finance", "Energy", "Health", "Consumer", "Industrial"]
    panel = {}
    for i in range(n_stocks):
        mu = rng.uniform(0.0002, 0.0006)
        sigma = rng.uniform(0.01, 0.03)
        rets = rng.normal(mu, sigma, n_days)
        price = 100 * np.exp(np.cumsum(rets))
        volume = rng.integers(1_000_000, 5_000_000, n_days)
        name = f"STOCK_{i}_{sectors[i % len(sectors)]}"
        panel[name] = pd.DataFrame({"close": price, "volume": volume})
    return panel

def add_technical_indicators(df):
    df = df.copy()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["volatility20"] = df["close"].pct_change().rolling(20).std()
    df["momentum10"] = df["close"] / df["close"].shift(10) - 1
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    ema12, ema26 = df["close"].ewm(span=12).mean(), df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    return df.dropna().reset_index(drop=True)
```

LSTM特征提取器：

```python
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, n_features=5, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, 1)  # 辅助任务：预测下一步收益

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        embedding = h_n[-1]  # 最后一层隐藏状态，64维
        pred = self.head(embedding)
        return pred.squeeze(-1), embedding

def train_lstm(model, X_train, y_train, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(X_train)
        loss = nn.functional.mse_loss(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
        opt.step()
        # ... (验证集loss监控、早停逻辑省略)
    return model
```

混合特征构建与评估（注意时序数据的划分方式）：

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

def build_hybrid_features(model, X_seq, tech_indicators):
    model.eval()
    with torch.no_grad():
        _, embeddings = model(X_seq)
    return np.concatenate([embeddings.numpy(), tech_indicators], axis=1)  # 78维

def chronological_split(X, y, train_ratio=0.7):
    cut = int(len(X) * train_ratio)
    return X[:cut], X[cut:], y[:cut], y[cut:]

# 关键：用TimeSeriesSplit而不是普通KFold，避免用未来数据调参
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {"max_depth": [3, 5], "learning_rate": [0.01, 0.05], "n_estimators": [200, 400]}
grid = GridSearchCV(xgb.XGBRegressor(objective="reg:squarederror"),
                     param_grid, cv=tscv, scoring="neg_mean_squared_error")

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    naive_baseline = np.mean(y_true > 0)  # "永远预测上涨"基线
    return rmse, dir_acc, naive_baseline
```

### 关键Trick

论文里容易被忽略、但没有就跑不起来（或者结果会虚高）的细节：

- **per-stock独立MinMax缩放**：不同股票价格量级差异巨大，如果全局归一化会让高价股主导梯度更新；必须每只股票单独fit scaler，且scaler只能在训练区间fit，测试区间只transform，否则就是标准的look-ahead泄露。
- **严格按时间划分，不做随机shuffle**：无论是train/test切分还是交叉验证，都要按时间顺序，`TimeSeriesSplit`而不是`KFold`。论文提到用3折交叉验证做网格搜索——如果这个3折是随机KFold而不是时序感知的，调参结果会系统性偏乐观，这是复现时最容易踩的坑。
- **梯度裁剪**：LSTM在金融序列上很容易出现梯度爆炸（价格突然跳变），`clip_grad_norm_`几乎是必需品。
- **辅助任务权重**：如果LSTM的预测头loss权重设得太高，模型会过拟合辅助任务本身，embedding反而丧失泛化性——实践中可以只训练少量epoch，或者对辅助任务加L2正则。

## 实验

### 数据与股票选择

论文选了14只覆盖6个行业的美股做面板训练（pooled训练，而不是每只股票单独训练一个模型），这个设计是合理的：金融数据单只股票的有效样本量太小，跨股票共享参数相当于变相扩充了样本量，也能让模型学到更"通用"的模式而不是记住某只股票的特例。

### 学习曲线（模拟）

用上面的合成数据跑一遍，你大概会观察到：LSTM的训练loss在前几个epoch快速下降，然后在10-20个epoch左右趋于平缓甚至轻微回升（过拟合信号）；这是金融序列训练的典型现象——不像图像分类那样loss能一路平滑下降，你需要提前设好早停。

### 与Baseline对比

论文摘要中明确给出的数字：30天窗口下，Hybrid模型测试RMSE为0.0949，约为纯LSTM基线的三分之一；但在大多数股票上，Hybrid只是"勉强持平或略微超过"纯XGBoost基线。摘要没有给出90/252/365天窗口的具体RMSE数字，所以下表只列出可确认的部分：

| 模型 | 30天RMSE | 相对纯LSTM | 相对纯XGBoost |
|------|---------|-----------|--------------|
| 纯LSTM | 约3倍于Hybrid | 基准 | 通常更差 |
| 纯XGBoost | 未单独给出 | 更优 | 基准 |
| Hybrid（LSTM+XGBoost） | 0.0949 | 明显更优 | 多数股票上持平或略优 |

方向准确率在365天窗口达到97.6%，但论文自己也拿"永远预测上涨"的naive基线做了对比——因为样本期内长期正收益本身占比就很高，所以真正有信息量的不是97.6%这个绝对数字，而是它相对naive基线的**超额部分**，且这个超额部分在短窗口（如30天）反而更有说服力，因为短期市场没有那么明显的单边趋势。

### 消融实验

论文的结构天然支持两个消融方向：

- 去掉LSTM embedding，只用技术指标——等价于纯XGBoost基线，用来验证embedding到底加了多少信息量。
- 去掉技术指标，只用LSTM embedding——测试LSTM单独提取的特征是否已经隐含了技术指标里的信息（比如动量、波动率这类模式LSTM理论上应该能从原始价格里学到）。

如果这两个消融的差距很小，说明78维特征里大部分预测力其实来自14维手工指标，LSTM的贡献有限——这也是我认为读者复现时最值得亲自验证的一点，而不是全盘接受论文的结论。

## 调试指南

### 常见问题

1. **LSTM训练loss不下降**：大概率是学习率太高导致震荡，或者输入没有做per-stock归一化导致数值尺度失衡。先把学习率降到1e-4试试。
2. **训练集RMSE很低，测试集RMSE暴涨**：典型的look-ahead泄露或过拟合。检查scaler是否用了全量数据fit、检查交叉验证是不是不小心用了随机KFold。
3. **方向准确率看起来很高但没意义**：先算一下naive基线（历史正收益比例），如果你的模型准确率只是勉强超过甚至低于naive基线，说明模型没有学到真正的alpha，只是学到了市场的整体涨跌偏置。
4. **长窗口（如365天）预测退化成预测均值**：模型对长期不确定性的建模能力有限时会倾向于输出保守的均值预测，这在长horizon上很常见，不代表实现有bug，但也说明长horizon的RMSE数字本身参考价值有限。

### 如何判断模型真的在学习

不要只看RMSE，同时看两个东西：

- 方向准确率相对naive基线（永远预测上涨/下跌）的**超额准确率**，这个超额部分越大越说明模型学到了真实信号。
- 在不同市场阶段（牛市段 vs 震荡段）分别评估，一个只在单边牛市里表现好的模型大概率只是学会了"顺势而为"，不是真正的预测能力。

### 超参数调优表

| 参数 | 推荐起点 | 敏感度 | 备注 |
|------|---------|-------|------|
| LSTM hidden size | 64 | 中 | 太大容易过拟合，金融数据没必要堆大模型 |
| LSTM层数 | 2 | 低 | 超过2-3层收益递减明显 |
| 窗口长度 | 60天 | 高 | 太短捕捉不到中期趋势，太长引入更多噪声 |
| XGBoost max_depth | 3-5 | 高 | 深了极易过拟合金融数据的噪声 |
| XGBoost learning_rate | 0.01-0.05 | 中 | 配合n_estimators一起调 |
| 交叉验证折数 | 3-5折时序切分 | 高 | 必须用TimeSeriesSplit，普通KFold会泄露未来信息 |

## 什么时候用 / 不用

| 适用场景 | 不适用场景 |
|---------|-----------|
| 有多只相关标的可以pooled训练，样本量足够支撑LSTM | 只有单一标的、数据量很小 |
| 需要短中期（30-90天）相对排序信号，用于组合打分/筛选 | 需要精确点预测用于高频/日内交易 |
| 已经有一套技术指标体系，想看深度学习能否带来增量 | 追求端到端极简架构、不愿意维护两阶段pipeline |
| 研究/组合评分等对可解释性要求不极端高的场景 | 监管或业务方要求模型完全可解释（树+LSTM组合更难解释） |

## 我的观点

这类"LSTM做特征提取 + XGBoost做决策"的混合架构，作为特征工程手段有一定价值，但我不会把它当成量化预测的"银弹"。论文自己报告的两个事实最值得读者留意：一是混合模型相对纯XGBoost基线的提升在多数股票上很有限，真正的大幅提升只体现在相对纯LSTM基线；二是长窗口下97.6%的方向准确率很大程度上是市场长期上涨的"基础命中率"，不是模型的真本事。这两点论文本身写得很诚实，也是我觉得这篇工作比很多"吹爆深度学习"的量化论文更值得参考的地方。

如果你在做组合排序或者因子挖掘，这套pipeline值得一试，成本也不高。但如果目标是实盘交易信号，我建议至少再加两样东西：一是用夏普比率、最大回撤这类风险调整后指标而不是单纯RMSE来评估；二是做walk-forward滚动回测并计入交易成本，因为静态的chronological split评估往往会高估策略在真实交易中的表现。原文摘要没有提到这两点，也没有给出官方代码仓库，所以我在文中用合成数据复现的是算法骨架，具体数值请在自己的真实数据上验证，不要直接套用论文数字做决策依据。