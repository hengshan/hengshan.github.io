---
layout: post-wide
title:  "Time Series Analysis: (3) Time Series Regression Modeling"
date:   2022-01-17 12:30:32 +0800
category: Spatial Modeling
author: Hank Li
use_math: true
---

In a previous [time scale analysis post]({% post_url 2022-01-11-time-series-2%}), we introduced how to anlalyze the time series of observed outcome $y_t$. In this post, we learn how to analyze the effect of an indepentent variable $x_t$ (with $t=1, \dots, n$) on the observed outcome $y_t$.

In traditional regression models such as a linear regression model, we generally think of relationships as being concurrent in timing. For example, when we study the linear relationship between heights and weights for a group of students (indepentent measures), we generally ignore the effect of time. However, with time series data, we can see if an effect is propagated across time. That is, the effect of a change in $x_t$ is “distributed” across multiple days in the future. This model is often called `distributed lag models`. 

There are two ways of thinking the objectives of this kind of problem.
1. we observe $x_t$ and we know $β_j$, and we want to know how the series $x_t$ is affected by convolving $x_t$ with the series $β_j$. Then the $y_t$ series represents the filtered version of $x_t$. 
2. we observe $x_t$ and $y_t$ and want to estimate the value $\beta_j$, which is often called `impulse-response function`.

### Filtering Time Series
Note that filtering time series is different from the objective of filtering in time series analysis, as the latter involves a latent variable.
Here, the collection of ${\beta_j}$ is called a *linear filter* in the model:
\[
y_t = \sum_{j=-\infty}^\infty \beta_j x_{t-j}
\]

Here, $y_t$ is a filtered version of $x_t$, and it is a *convolution* between two series $x_t$ and $\beta_j$. Its Fourier transform is:

\begin{align}
y_w &= \sum_{t=-\infty}^\infty\sum_{j=-\infty}^\infty \beta_j x_{t-j}\mathrm{exp}(-2\pi iwt) \\\\ 
&= \sum_{t=-\infty}^\infty\sum_{j=-\infty}^\infty \beta_j x_{t-j}\mathrm{exp}(-2\pi iw(t - j +j)) \\\\ 
&= \sum_{t=-\infty}^\infty\sum_{j=-\infty}^\infty \beta_j\mathrm{exp}(-2\pi iwj) x_{t-j}\mathrm{exp}(-2\pi iw(t - j)) \\\\ 
&= \sum_{j=-\infty}^\infty \beta_j\mathrm{exp}(-2\pi iwj)\sum_{t=-\infty}^\infty x_{t-j}\mathrm{exp}(-2\pi iw(t - j)) \\\\ 
&= \sum_{j=-\infty}^\infty \beta_j\mathrm{exp}(-2\pi iwj)\sum_{t=-\infty}^\infty x_{t}\mathrm{exp}(-2\pi iw(t)) \\\\ 
&= \beta_w x_w
\end{align}

Here, the collection of $\beta_w$ as a function of $w$ is called the `transfer function`. It is the Fourier transform of the `impulse response function` $\beta_j$.

If we want to compute the filtered values $y_1, y_2,\dots, y_n$, the naive formula requires to compute the convolution formula n times. But with Fourier transfer, we can:
1. compute the Fourier transform $x_w$ via the FFT for frequencies $w = 0/n, 1/n, \dots, 1/2$
2. compute the FFT $\beta_w$ for these frequencies.
3. compute $y_w = \beta_wx_w$ for these frequencies
4. compute the inverse FFT.

The timeseriesbook then introduced the low-pass filter, high-pass filter, and matching filter. The book also mentioned `Exponential Smoother`. It is such an important topic in time series analysis that we will introduce it in another blog.

### Distributed Lag Models
We consider models of the form:

\[
y_t =\sum_{j=-\infty}^\infty \beta_j x_{t-j} +\epsilon_t
\]

The collection ${\beta_j}$ as a function of the lag j will be referred to as the `distributed lag function`.
A sometimes useful summary statistics is:

\[
\eta = \sum_{j=0}^M\beta_j
\]

The value $\eta$ is often interpretable as a cumulative effect when the outcome is a count. If $\beta_j \neq 0$ and $\eta \approx 0$, then that might suggest that on average across M time points, the effect of a unit increase in $x_t$ is roughly 0.

### Temporal Confounding
The author introduced this topic from a simple linear regression model,

\[
y_t = \beta x_t +\epsilon_t
\]

where $\epsilon_t$ is a Gaussian process with mean 0 and autocovariance $\sigma^2\rho(u)$. It is also assumed that $\mathbb{E}[y_t] = \mathbb{E}[x_t] =0$ and $Var(x_t)=1$.
The least squares estimate for $\beta$ is:

\[
\hat\beta = \frac{1}{n}\sum_{t=0}^{n-1}y_tx_t
\]

Notably, the general form of the least squares estimate for $\beta$ for any matrix X and Y is:

\[
\hat\beta = (X^TX)^{-1}X^TY
\]

In the example, there is only one indepentent variable $x_t$ and one coefficient $\beta$ which does not change as a function of t. The author use the simple linear regression model to illustrate the temporal confounding.

Based on the time scale analysis, we can decompose $x_t$ into:
\[
x_t = \frac{1}{n}\sum_{p=0}^{n-1}z_x(p)\mathrm{exp}(2\pi ipt/n)
\]
where $z_x(p)$ is the complex Fourier coefficient accociated with the frequency p.
\[
z_x(p) = \frac{1}{n}\sum_{p=0}^{n-1}x_t\mathrm{exp}(-2\pi ipt/n)
\]

The least squares estimate can be written as:
\begin{align}
\hat\beta &= \frac{1}{n^2}\sum_{p=0}^{n-1}z_x(p)\bar z_y(p) \\\\ 
\hat\beta_p &= \frac{z_x(p)}{n}\frac{\bar z_y(p)}{n} \\\\ 
\hat\beta &= \sum_{p=0}^{n-1}\hat\beta_p
\end{align}

This shows that the least squares estimate of the coefficient $\beta$ can be written as a sum of the products of the Fourier coefficients between the $y_t$ and $x_t$ series over all of the frequencies. 
This is a very important conclusion, because if we are primarily interested in long-term or short-term trends in $x_t$ and $y_t$, then we do not need to focus on other frequencies components. 

#### Bias from Omitted Temporal Confounders
Suppose the true model for $y_t$ is

\[
y_t = \beta x_t + \gamma w_t + \epsilon_t
\]

If we include $x_t$ but omit $w_t$ from the model, 

\begin{align}
\hat\beta &= \frac{1}{n}\sum_{t=0}^{n-1}y_tx_t \\\\ 
 &= \frac{1}{n}\sum_{t=0}^{n-1}(\beta x_t + \gamma w_t +\epsilon_t)x_t\\\\ 
 &= \beta \widehat{Var}(x_t) + \gamma \frac{1}{n}\sum_{t=0}^{n-1}w_tx_t+\frac{1}{n}\sum_{t=0}^{n-1}\epsilon_tx_t\\\\ 
 & \approx \beta + \gamma\frac{1}{n}\sum_{t=0}^{n-1}w_tx_t
\end{align}

The author illustrated that: the quantity $\sum_{t=0}^{n-1}w_tx_t$ is essentially the least squares estimate of the coefficient between $w_t$ and $x_t$. This implies that if the covariance between the two is zero, then there is no bias in $\hat\beta$. We can also write the bias as a sum of Fourier coefficients:
\[
\mathrm{Bias}(\hat\beta) = \gamma \frac{1}{n^2}\sum_{p=0}^{n-1}z_w(p)\bar z_x(p)
\]
As long as there is no time sacle where the Fourier coefficients for both $x_t$ and $w_t$ are large, the bias should be close to zero. In other words, to avoid the affect of the confounding factors, we should focus our interest in looking at variation at different time scales of the confounding factors. 

Finally, the author mentioned that it’s possible that the residuals are autocorrelated.If there were substantial autocorrelation in the residual, that would suggest that the independent error model was incorrect. To fully understand this, we need to learn heteroscedasticity and generalized least squares. We will introduce this in another post. 
