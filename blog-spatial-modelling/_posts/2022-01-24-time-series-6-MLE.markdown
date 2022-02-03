---
layout: post
title:  "Learning Time Series Analysis: (6) Maximum Likelihood Estimation"
date:   2022-01-24 08:30:32 +0800
category: Spatial Modeling
use_math: true
---

This post is based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"} chapter 6.
In a previous post, we introduced Autoregressive Integrated Moving Average (ARIMA) and Exponential Smoothing, and we mentioned that some ARIMA models can be written in the format of Exponential Smoothing. In another post, we learned State Space Models and Kalman Filter.

In the current post, we will learn how to use State Space model and Kalman Filter for parameter estimation of AR models based on `Maximum Likelihood Estimation` (MLE). That is, we
want to select that parameters that make the observed data the most likely. For example, when we observe independent and identically distibuted (iid) data $y_1,\dots, y_n \sim p(y; \theta)$, the MLE of the parameter vector $\theta$ is:

\begin{align}
L(\theta) = \prod_{i-1}^{n}f(y_i|\theta)
\end{align}

Now we chose the value of $\theta$ that maximizes the likelihood function, $\hat \theta=\underset{\theta}{\mathrm{argmax}}L(\theta)$.

A cool property of argmax is that since log is a monotone function, the argmax of a function is the same as the argmax of the log of the function! That’s nice because logs make the math simpler. Instead of using likelihood, you should instead use log likelihood.

\[
LL(\theta) = \mathrm{log}\prod_{i=1}^nf(y_i|\theta) = \sum_{i=1}^n\mathrm{log}f(y_i|\theta)
\]

However, if the data are not independent such as in AR(1) model the maximize the log-likelihood is hard to write down. 

### An AR(1) Model MLE
For AR(1) model, we can describe it as:
\[
y_t = \phi y_{t-1} +w_t
\]

where $w_t \sim N(0, \tau^2)$ and we assume that $\mathbb{E}[y_t]=0$. What is the joint distribution of the time series?

If we assume that the process is 2nd-order stationary then for the marginal variances, we have
\[
\mathrm{Var}(y_t) = \phi^2 \mathrm{Var}(y_{t-1}) + \tau^2
\]

The stationarity assumption implies that $Var(y_t) = \frac{\tau^2}{1-\phi^2}$, assuming that $\lvert\phi\rvert <0$. Furthermore, we can show that

\begin{align}
\mathrm{Cov}(y_t, y_{t-1}) &= \mathrm{Cov}(\phi y_{t-1},y_{t-1}) \\\\ 
&= \phi\mathrm{Var}(y_{t-1}) \\\\ 
&= \phi\frac{\tau^2}{1-\phi^2}
\end{align}

Because of the sequential dependence of $y_t$s on each other, we have
\begin{align}
\mathrm{Cov}(y_t, y_{t-j}) = \phi^{\lvert j\rvert}\frac{\tau^2}{1-\phi^2} \end{align}

The covariance matrix can be something like:
\begin{bmatrix}
\frac{\tau^2}{1-\phi^2} & \phi\frac{\tau^2}{1-\phi^2} & \dots & \phi^{\lvert {n-1}\rvert}\frac{\tau^2}{1-\phi^2} & \phi^{\lvert {n}\rvert}\frac{\tau^2}{1-\phi^2}\\\\ 
\phi\frac{\tau^2}{1-\phi^2} & \frac{\tau^2}{1-\phi^2} & \phi\frac{\tau^2}{1-\phi^2}& \dots & \phi^{\lvert {n-1}\rvert}\frac{\tau^2}{1-\phi^2} \\\\ 
\vdots & \phi\frac{\tau^2}{1-\phi^2} & \ddots & \phi\frac{\tau^2}{1-\phi^2}& \vdots \\\\ 
\phi^{\lvert {n-1}\rvert}\frac{\tau^2}{1-\phi^2} & \dots & \phi\frac{\tau^2}{1-\phi^2}& \frac{\tau^2}{1-\phi^2} &\phi\frac{\tau^2}{1-\phi^2} \\\\ 
\phi^{\lvert {n}\rvert}\frac{\tau^2}{1-\phi^2} & \phi^{\lvert {n-1}\rvert}\frac{\tau^2}{1-\phi^2} & \dots & \phi\frac{\tau^2}{1-\phi^2}& \frac{\tau^2}{1-\phi^2}
\end{bmatrix}

As n increase, some challenges arise:

1. the covariance matrix quickly grows in size, making the computations more cumbersome, especially because some form of matrix decomposition must occur.

2. we are taking larger and larger powers of $\phi$, which can quickly lead to numerical instability and unfortunately cannot be solved by taking logs. 

3. The formulation above ignores the sequential structure of the AR(1) model, which could be used to simplify the computations.

To addresses both of these problems, the timeseriesbook then introduced how the Kalman filter can provide a computationally efficient way to evaluate this complex likelihood.

### Maximum Likelihood with the Kalman Filter
The basic idea is to re-formulate the time series AR(1) model as a state space model, and then use the Kalman filter to compute the log-Likelihood of the obseverd data for a given set of parameters. 

The general formulation of the state space model is:
\begin{align}
x_t &= \Theta x_{t-1} + W_t \\\\ 
y_t &= A_tx_t + V_t
\end{align}
where $V_t \sim n(0, S)$ and $W_t \sim n(0, R)$.

The joint density of the observations,
\begin{align}
p(y_1, y_2, \dots, y_n)&=p(y_1)p(y_2,\dots,y_n|y_1) \\\\ 
 &= p(y_1)p(y_2|y_1)p(y_3,\dots,y_n|y_1,y_2) \\\\ 
&\vdots \\\\ 
&=p(y_1)p(y_2|y_1)p(y_3|y_1,y_2)\dots p(y_n|y_1,\dots,y_{n-1})
\end{align}

We Initially need to compute $p(y_1)$,
\begin{align}
p(y_1) &= \int p(y_1,x_1)dx_1 \\\\ 
&= \int \underbrace{p(y_1|x_1)}_{\mathrm{density\ for\ the\\\\   observation\ equation}}p(x_1)dx_1
\end{align}

The density $p(x_1)$ is $\mathbb{N(x_1^0,P_1^0)}$, where
\begin{align}
x_1^0 &= \Theta x_0^0 \\\\ 
P_1^0 &= \Theta P_0^0 \Theta' + R
\end{align}

Together, we get
\begin{align}
p(y_1) = N(Ax_1^0, AP_1^0A' + S)
\end{align}

In general, we will have
\begin{align}
p(y_t|y_1,\dots,y_{t-1}) = N(Ax_t^{t-1},AP_t^{t-1}A'+S)
\end{align}

Then we can use standard non-linear maximization routines like Newton’s method or quasi-Newton approaches for MLE. The timeseriesbook then introduced an example of AR(2) model(see [here](https://bookdown.org/rdpeng/timeseriesbook/example-an-ar2-model.html){:target="_blank"})
