---
layout: post
title:  "Three Goals of Building Spatial-Temporal Statistics Models"
date:   2022-01-01 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

[Spatio-Temporal Statistics with R](https://spacetimewithr.org/Spatio-Temporal%20Statistics%20with%20R.pdf){:target="_blank"} states that there are three main goals that one might pursue with a spatio-temporal statistical model:
1. prediction in space and time (filtering and smoothing)
2. inference on parameters
3. forecasting in time

The second goal is about statistical inference. To have a basic understanding of probability and statistics such as population and sample, sampling, parameter estimation, and hypothesis testing, see [here](https://byjus.com/maths/statistical-inference/){:target="_blank"}.
The first and third goals are about filtering, smoothing and forecasting. To fully understand them, it is necessary to understand what state-space model is, as this book introduced the three goals from the perspective of `state-space models`:
> 1. **smoothing** refers to inference on the hidden state process during a fixed time period in which we have observations throughout the time period. 
2. **filtering** refers to inference on the hidden state value at the most current time based on the current and all past data. the most famous example of filtering is the kalman filter (kalman, 1960).
3. **forecasting** refers to inference on the hidden state value at any time point beyond the current time, where data are either not available or not considered in the forecast.
the second goal, inference on parameters, is about statistical inference, which is to use sample data to make inference about the parameters of population.

### What is state-space model?
State-space models cover the broad range of time series models where the aim is to estimate the state of an unobservable random process ${Z_t}$ with $t \in {1, 2, . . . , T}$ from an observed data set ${X_t}$.

When I searched 'state-space model' online, this term is always associated with either time series analysis or control system. It is not surprising as both time series analysis and control system deal with certain functions of time, and state space model is essentially a set of differential equations of functions of time.

What is a function of time f(t)? In physics, finance, and many enginnering fields, a lot of phenomona have been mathematically modeled as functions of time.
The simplest example is stock price, which fluctuates and changes its value at different time.
Another classic example is when describing the position of a object varies with time under the influence of constant acceleration such as a spring mass system.

#### What do `space` and `state` exactly mean in state space models?
The 'space' in state-space model stands for the mathematical space of state variables. See [here](https://en.wikipedia.org/wiki/Space_(mathematics)){:target="_blank"} for the introduction of space (mathematics) in wikipedia. 

The 'state' in state-space model is a latent variable (unobserved data). All states and input together can describe the dynamic system.
An intuitive way to understand state-space model is through a simple physical system such as a simple spring mass system.

![spring mass system](https://cdn1.byjus.com/wp-content/uploads/2021/10/Simple_harmonic_oscillator-0.jpg#center)
The real-time location and speed are so-called state variables both of which are functions of time, as at different time the location and speed are different. The goal of state space model is to describe the dynamic system.

\[u(t) - k{x}(t) - b\dot{x}(t) = m\ddot{x}(t)\tag{1}\]

Here, $u(t)$ is input force such as gravity. $kx(t)$ is spring force. $b\dot{x}(t)$ is friction force, and $\ddot{x}(t)$ is acceleration. Here, we define:

\[
\begin{align}
x(t) &= x_1 \\\\  
\dot{x}(t) &= x_2 =\dot{x}_1\\\\  
u(t) &= u_1
\end{align}\tag{2}
\]

Equation (1) then became:
\[u_1 - kx_1  -bx_2=m\dot{x}_2\tag{3}\]

Based on these equations, the above equation can be rewritten as:
\[\begin{cases}
\dot{x}_1 &=x_2 \\\\  
\dot{x}_2 &=\frac{1}{m}u_1 -\frac{k}{m}x_1-\frac{b}{m}x_2
\end{cases}\tag{4}\]

The above equation can further be rewritten as:
\[\begin{cases}
\dot{x}_1 &=0x_1+1x_2+0u_1 \\\\  
\dot{x}_2 &=-\frac{k}{m}x_1-\frac{b}{m}x_2+\frac{1}{m}u_1 
\end{cases}\tag{5}\]

The above can be rewritten as a matrix form:
\[
\begin{bmatrix}
\dot{x}_1 \\\\  
\dot{x}_2
\end{bmatrix} = \begin{bmatrix} 0 & 1 \\\\ {-k\over m} & {-b\over m}\end{bmatrix}
\begin{bmatrix}
\{x}_1 \\\\  
\{x}_2
\end{bmatrix} + \begin{bmatrix}
0\\\\  
{1\over m}
\end{bmatrix}
\begin{bmatrix}
\{u}_1
\end{bmatrix}
\]

The above matrix equation is the first important equation of the state space model. This equation describes how the states dynamically change (first order derivative) as a function of the states and the inputs.
\[\dot{x}=Ax+Bu\tag{6}\]

The second equation is the observation equation. The y value is observed. Here, we only observed the location.
\[y=x_1=x(t)\]

The above can be rewritten as:
\[y=1x_1+0x_2+0u_1\]

And the matrix form is:
\[
y=
\begin{bmatrix}
1&0
\end{bmatrix}
\begin{bmatrix}
x_1 \\\\  
x_2
\end{bmatrix}+
\begin{bmatrix}
0
\end{bmatrix}u_1
\]

This is the second important equation of the state space model. This equation describes how the observations change as a function of states and inputs.
\[
y=Cx+Du\tag{7}
\]

The equations (6) and (7) are the most common format for state space models. For more details, please refer to this series of [videos](https://www.youtube.com/watch?v=gJzY6jOcgN0&list=PLmK1EnKxphikZ4mmCz2NccSnHZb7v1wV-){:target="_blank"} introduce state-space model (6.1-6.4) to electronic engineering students. It first introduced a very simple spring system, and then introduced circuit state space model. I like this video, as the instructor covered some basics such as basic linear algeba and Laplace transform (1.1-1.3). 

#### How state space model is related to the goal of spatial modeling?

State space model is indeed useful for control system, but how it is related to the goals of spatial modeling. Don't get confused. It will be clear after we figure out the role of state space model on time series analysis. In time series analysis, we often deal with a discrete-time space state model. Equations (6) and (7) will become:

\[
\begin{eqnarray}
x(n+1)=Ax(n)+Du(n)\tag{8} \\\\  
y(n)=Cx(n)+Du(n)\tag{9}
\end{eqnarray}
\]

where 
* $x(n) \in \mathbb{R}^N$ = state vector at time n
* $u(n) = p \times 1$ vector of inputs
* $y(n) = q \times 1$ output vector
* $A = N \times N$ state transition matrix
* $B = N \times p$ input coefficient matrix
* $C = q \times N$ output coefficient matrix
* $D = q \times p$ direct path coefficient matrix

[Statsmodels](https://www.statsmodels.org/stable/statespace.html){:target="_blank"} describes state space model with additional irregular components $\epsilon_t \sim N(0,H_t)$ and $\eta_t \sim N(0, Q_t)$. These terms are neglected in this post. 
The key question is: how to link state space model and spatial-temporal modeling? Obviously, spatial modeling does not model spring mass system. However, both spatial-temporal model and state space model deal with temporal data, which is an observation sequence.

Let's imagine that we observe the temperature at a few sampled locations. These temperature data are output vector $y(n)$.
Then what is the state? From a physical standpoint, a lot of factors such as latitude, longtitude, distance from the sea,prevailing winds, direction of mountains, slope of land and vegetation and so on and so forth can affect the temperature. Do we model all of these states in spatial-temporal modeling? Of course not. This is related the next post topic: two types of spatial-temporal models: descriptive model and dynamical model.
In descriptive model, we only need to add temperature (output vector) and state variables of our research interests such as distance from the sea, longlat.
Given that the temperature is observation at discrete time, we can use hidden markov model here. In the model we  have two layers: observation layer (output) and hidden state layer (state space). An easy way to understand the concept is through [Hidden Markov Model](https://towardsdatascience.com/hidden-markov-model-hmm-simple-explanation-in-high-level-b8722fa1a0d5){:target="_blank"}. 

![HMM](https://assets-global.website-files.com/5f5b931c423b11277a8fe867/5f6e06113b1546c14fd37029_0*8T1XguoIrb8tG8mK.png#center)

We can understand filtering, smoothing, and predicting from a HMM perspective. 

**Filtering** is to compute, given the model's parameters and a sequence of observations, the distribution over hidden states of the last latent variable at the end of the sequence, i.e. to compute $P(z(t)\|x(1),\dots,x(t))$. This problem can be handled efficiently using the [forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm){:target="_blank"}.

**Smoothing** is similar to filtering but asks about the distribution of a latent variable somewhere in the middle of a sequence, i.e. to compute $P(z(k)\|x(1),\dots,x(t))$ for some $k<t$. From the perspective described above, this can be thought of as the probability distribution over hidden states for a point in time k in the past, relative to time t.

**Predicting** is to compute $P(z(k)\|x(1),\dots,x(t))$ for $k>t$ such as $P(z(t+1)\|x(1),\dots,x(t))$. 

After we have basic understanding of HMM, now we can understand the three goals of spatial modeling:
> 1. **smoothing** refers to inference on the hidden state process during a fixed time period in which we have observations throughout the time period. 
2. **filtering** refers to inference on the hidden state value at the most current time based on the current and all past data. the most famous example of filtering is the kalman filter (kalman, 1960).
3. **forecasting** refers to inference on the hidden state value at any time point beyond the current time, where data are either not available or not considered in the forecast.
the second goal, inference on parameters, is about statistical inference, which is to use sample data to make inference about the parameters of population.

This post aims to help have a simple high level understanding of the goals of spatial modeling. It is worth noting that both HMM and state space model can be used to model sequence data.
State space model consists of a series of models such as time-invariant and time variant state space models. 
For continous time, the filtering steps for state space model is in integral form. 
State update:
\[
p(z_t\|X_{t-1})=\int{p(z_{t-1}\|X_{t-1})p(z_t\|z_{t-1})dz_{t-1}}
\]

Bayes' update:
\[
p(z_t\|X_{t})=p(z_t\|X_{t-1})\frac{p(x_t\|z_t)}{p(x_t\|X_{t-1})}
\]
