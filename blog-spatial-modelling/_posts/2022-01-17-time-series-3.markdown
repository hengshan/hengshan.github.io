---
layout: post
title:  "Learning Time Series Analysis: (3) Time Series Regression Modeling"
date:   2022-01-17 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

This post is based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"} chapter 4. In a previous [time scale analysis post]({% post_url 2022-01-11-time-series-2%}), we introduced how to anlalyze the time series of observed outcome $y_t$. In this post, we learn how to analyze the effect of an indepentent variable $x_t$ (with $t=1, \dots, n$) on the observed outcome $y_t$.

In traditional regression models such as a linear regression model, we generally think of relationships as being concurrent in timing. For example, when we study the linear relationship between heights and weights for a group of students (indepentent measures), we generally ignore the effect of time. However, with time series data, we can see if an effect is propagated across time. That is, the effect of a change in $x_t$ is “distributed” across multiple days in the future. This model is often called `distributed lag models`. 

The author introduced two ways of thinking the objectives of this kind of problem.
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

