---
layout: post
title:  "Learning Time Series Analysis: (5) The Kalman Filter"
date:   2022-01-21 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

This post is based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"} chapter 5. In the previous post, we learned exponential smoothing and ARIMA. These methods mainly deal with `univariate time series`.

> A *univariate time series* is a sequence of measurements of the same variable collected over time. Most often, the measurements are made at regular time intervals.

Now we are learning how to analyze the time series with a series of states. In a previous [post]({% post_url 2022-01-06-two-approaches-spatial-modeling%}), we learned `state space models` and the basic concept of filtering. 

State space models attempt to describe a phenomenon that has two characteristics:

1. There is an underlying system that has a time-varying dynamical relationship, so that the “state” of the system at time $t$ is related to the state of the system at time $t−1$ . If we know the state of the system at time $t−1$ , then we have everything we need to know in order to make inferences or forcasts about the state at time t.

2. We cannot observe the true underlying state of the system, but rather we observe a noisy version of it.

The state equation describes how the system evolves from one time point to the next, and the observation equation describes how the underlying state is transformed into something that we directly measure. You may wonder why this matters. Well, sometimes we have observed time series but actually we are more interested in the unobserved states hidden behind the time series. Thus, we need to figure out a method to use observed time series to infer the state or to infer the parameters of the model that generate observations from states. A famous method is the `Kalman Filter`. 

### The Basic Idea of Kalman Filter
The timeseriesbook introduces the Kalman Filter using a very simple way. I like it. It starts with a very simple state equation of: 

\[
x_t = \theta x_{t-1} + w_t
\]

and an observation equation of:
\[
y_t = x_t +v_t
\]

where we assume $w_t \sim N(0, \tau^2)$ and $v_t \sim N(0, \sigma^2)$.

In the previous post, we learned ARIMA, and the above state euqation seems like a AR(1) model. We can simply understand the it as a AR(1) model, but note that the state is not observed. In ARIMA model we only deal with time series of observations.

Anyway, let's focus on the two equations. The basic one-dimensional Kalman filter is as follows. First, based on an initial state $x_0^0$ and variance $P_0^0$, we compute:

\begin{align}
x_1^0 &= \theta x_0^0 \\\\ 
P_1^0 &= \theta^2P_0^0 +\tau^2
\end{align}

These are our best guesses before we get the new observation $y_1$. After we get the new observation $y_1$, we can then update our guess:

\begin{align}
x_1^1 &= x_1^0 + K_1(y_1 - x_1^0) \\\\ 
P_1^1 &= (1-K_1)P_1^0
\end{align}

where $K_1 = \frac{P_1^0}{P_1^0 + \sigma^2}$. 

For the general case, we want to estimate $x_t$ based on the current state and variance. Step 1:

\begin{align}
x_t^{t-1} &= \theta x_{t-1}^{t-1} \\\\ 
P_t^t &= \theta^2P_{t-1}^{t-1} +\tau^2
\end{align}

Step 2:
\begin{align}
x_t^t &= x_t^{t-1} + K_t(y_t - x_t^{t-1}) \\\\ 
P_t^t &= (1-K_t)P_t^{t-1}
\end{align}

where $K_t = \frac{P_t^{t-1}}{P_t^{t-1} + \sigma^2}$. 

Base on the above formula, it is very clear that the general idea is:

\begin{align}
\mathrm{if} \ \sigma^2 \ \mathrm{is \ large} &\Rightarrow \mathrm{Trust\ the\ system} \\\\ 
\mathrm{if \ } \tau^2 \mathrm{is\ large} &\Rightarrow \mathrm{Trust\ the\ data}
\end{align}

### Deriving the One-dimensional Case
The timeseriesbook derives the Kalman filter from a statistian perspective. We use the same two equations:
\[
x_t = \theta x_{t-1} + w_t
\]

\[
y_t = x_t +v_t
\]

where $w_t \sim N(0, \tau^2)$ and $v_t \sim N(0, \sigma^2)$.

At t =1, we only have initial state $x_0 \sim N(x_0^0, P_0^0)$, which is unobserved, but we know the mean and variance of the initial state. For example, one task is to measure the building height. Initially, one can estimate the building height simply by looking at it.

The estimated building height is: $x_0^0=60m$. Now we shall initialize the estimate uncertainty. A human’s estimation error (standard deviation) is about 15 meters: $\sigma=15m$. Consequently the variance is 225:$p_0^0=225$.

OK, at time t=1, there is no $y_0$, so we cannot condition on any observed information yet. We can compute the marginal distribution of $x_1$, i.e., $p(x_1)$ as:

\begin{align}
p(x_1) &= \int p(x_1 |x_0)p(x_0)dx_0 \\\\ 
&= \int N(\theta x_0, \tau^2) \times N(x_0^0,P_0^0)dx_0 \\\\ 
&= N(\theta x_0^0, \theta^2 P_0^0 + \tau^2) \\\\ 
&=N(x_1^0, P_1^0)
\end{align}

Note that we have defined $x_1^0 \overset{\Delta}{=} \theta x_0^0$ and $P_1^0 \overset{\Delta}{=} \theta^2 P_0^0 + \tau^2$. $x_1^0$ is the best prediction we can make based on our knowledge of the system and no data.

Given the new observation $y_1$ we want to use this information to update the estimation of $x_1$. For that, we need the conditional distribution $p(x_1\|y_1)$, which is so-callewd *filter density*. Based on Bayes' rule:

\begin{align}
p(x_1|y_1) &\propto p(y_1|x_1)p(x_1) \\\\ 
&=\varphi (y_1|x_1,\sigma^2) \times \varphi(x_1|x_1^0,P_1^0) \\\\ 
&=N(x_1^0+K_1(y_1 - x_1^0),(1-K_1)P_1^0) \\\\ 
\end{align}

where $K_1 = \frac{P_1^0}{P_1^0+\sigma^2}$ is the *Kalman gain coefficient*.

For t = 2, we will have a new observation $y_2$. To compute the new filter density for $x_2$,

\begin{align}
p(x_2|y_1,y_2) &\propto p(y_2|x_2)p(x_2|y_1) 
\end{align}

Implicit in the statement above is that $y_2$ does not depend on $y_1$ conditional on the value of $x_2$. The new filter density is a product of the observation density $p(y_2\|x_2)$ and the *forcast density* $p(x_2\|y_1)$. 

The observation density is simply $N(x_2,\sigma^2)$. The forcast density is:

\begin{align}
p(x_2|y_1) &= \int p(x_2,x_1|y_1)dx_1 \\\\ 
&\propto \int p(x_2|x_1)p(x_1|y_1)dx_1 \\\\ 
&= \int \varphi(x_2|\theta x_1, \tau^2) \varphi(x_1|x_1^1,P_1^1)dx_1 \\\\ 
&= \varphi(x_2 |\theta x_1^1, \theta^2 P_1^1 + \tau^2) \\\\ 
& = \varphi(x_2 |x_2^1, P_2^1)
\end{align}

we get
\begin{align}
p(x_2|y_1,y_2) &\propto \varphi(y_2|x_2,\sigma^2)\varphi(x_2|x_2^1,P_2^1) \\\\ 
&= N(x_2^1 + K_2(y_2-x_2^1),(1-K_2)P_2^1)
\end{align}

where 
\begin{align}
K_2 = \frac{P_2^1}{P_2^1+\sigma^2}
\end{align}

### General Kalman Filter

The more general formulation of the state space model is:
\begin{align}
x_t &= \Theta x_{t-1} + W_t \\\\ 
y_t &= A_tx_t + V_t
\end{align}
where $V_t \sim N(0, S)$ and $W_t \sim N(0, R)$.

Given an initial state $x_0^0$ and $P_0^0$, the prediction equation are:

\begin{align}
x_1^0 &= \Theta x_0^0 \\\\ 
P_1^0 &= \Theta P_0^0 \Theta' + R
\end{align}

Given a new observation $y_1$, the updating equations are:
\begin{align}
x_1^1 &= x_1^0 + K_1(y_1 - A_1x_1^0) \\\\ 
P_1^0 &=(I-K_1A_1)P_1^0 
\end{align}

where
\begin{align}
K_1 = P_1^0A_1'(A_1P_1^0A_1' + S)^{-1}
\end{align}

In general, given the current state $x_{t-1}^{t-1}$ and $P_{t-1}^{t-1}$ and a new observation $y_t$, we have

\begin{align}
x_t^{t-1} &= \Theta x_{t-1}^{t-1} \\\\ 
P_t^{t-1} &= \Theta P_{t-1}^{t-1} \Theta' + R
\end{align}

With a new observation $y_t$,

\begin{align}
x_t^t &= x_t^{t-1} + K_t(y_t - A_tx_t^{t-1}) \\\\ 
P_t^t &=(I-K_tA_t)P_t^{t-1} 
\end{align}

where
\begin{align}
K_t = P_t^{t-1}A_t'(A_tP_t^{t-1}A_t' + S)^{-1}
\end{align}

Notably, if $y_t$ is missing, we can revise the update procedure to simply be:
\begin{align}
x_t^t &= x_t^{t-1} \\\\ 
P_t^t &= P_t^{t-1}
\end{align}
