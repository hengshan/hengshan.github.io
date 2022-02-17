---
layout: post
title:  "Learning Spatial-Temporal Modeling (4) Dynamic Spatio-Temporal Models"
date:   2022-02-13 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

This post is based on the book [Spatio-Temporal Data in R](https://spacetimewithr.org/){:target="_blank"} chapter 5. In this post, we will learn univariate linear dynamic spatio-temporal models (DSTM) in discretized time context.

In previous posts, we have learned the basics of time series analysis. The DSTM could be viewed as a *time series of spatial processes*. DSTMs provide powerful insights into causative mechanisms, as it allows us not only to make predictions of spatial processes but also to make inference on parameters of models of underlying mechanistic. 

The data model and process model are respectively (1) and (2),

$Z_t(\cdot) = H_t(Y_t(\cdot),\theta_{d,t},\epsilon_t(\cdot)), \quad t =1, \dots,T \tag{1}$

$Y_t(\cdot) = M(Y_{t-1}(\cdot),\theta_{p,t},\eta_t(\cdot)), \quad t =1, 2, \dots \tag{2}$

There are two key assumptions that are usually made for DSTMs.
1. $Z_t(\cdot)$ are independent in time when conditioned on the true process $Y_t(\cdot)$ and the parameters $\theta_{d,t}$. The join distribution of the data conditioned on the process and parameters can be represented in product form.

2. The *Markov assumption*: conditioned on the past, only the recent past is important to explain the present. Under the first-order Markov assumption that the process at time $t$ conditioned on all of the past is only dependent on the *most* recent past.

$[Y_0(\cdot),Y_1(\cdot),\dots,Y_T(\cdot)\|\theta_{p,t}] = \left\(\prod_{t=1}^T[Y_t(\cdot)\|Y_{t-1}(\cdot),\theta_{p,t}]\right\)[Y_0(\cdot)\|\theta_{p,0}]\tag{3}$

For the process model (2), it can be linear or nonlinear and the associated conditional distribu-tion, $[Y_t(\cdot)\|Y_{t−1}(\cdot)]$ can be Gaussian or non-Gaussian. As in autoregressive modeling in time series, one can make higher-order Markov assumptions.

$\theta_{d,t},\theta_{p,t}$ are respectively the parameters of the data and process model. The primary challenge in DSTMs is to effectively reduce the parameter space. 

### Latent Linear Gaussian DSTMs
The authors illustrate DSTMs using the simplest but most widely use situation, in which the process operator $M$ in (2) is assumed to be linear with additive Gaussian error distribution. 

We are interested in a Latent process $\\{Y_t(s_i)\\}$ at a set of locations given by $\\{s_i:i =1, \dots, n\\}$, and we have data/observations at locations $\\{r_{jt}: j =1,\dots,m_t; t=0,1,\dots,T\\}$. Note that there could be a different number of data locations at different observation time, but it is assumed that there is finite set of $m$ possible data locations to be considered; so $m_t \le m$. If there are infinite possible data locations, then the location itself becomes a random variable, and it is a possion point process, which is not covered in this book.
The locations can be either point or areal.

#### Linear Data Model with Additive Gaussian Error
Consider the $m_t$-dimensional data vector, $Z_t \equiv (Z_t(r_{1t}),\dots,Z_t(r_{m_tt})')$, and the $n$-dimensional latent-process vector, $Y_t \equiv (Y_t(s_1),\dots,Y_t(s_n))'$. We aim to infer the latent-process vector.

Consider the $j$th observation at time $t$, the linear data model with additive Gaussian error can be written as:

$Z_t(r_{jt}) = b_t(r_{jt}) + \sum_{i=1}^n h_{t,ji}Y_t(s_i)+\epsilon_t(r_{jt})\tag{4}$

The vector-matrix form is:

$Z_t = b_t + H_tY_t +\epsilon_t, \quad \epsilon_t \sim Gau(0,C_{\epsilon,t})\tag{5}$

where $b_t$ is the $m_t$-dimensional offset term, $H_t$ is the $m_t\times n$ mapping matrix, which is typically assumed to be known. $C_{\epsilon,t}$ is an $m_t \times m_t$ error covariance matrix. It can generally include dependence in space or time, although we typically assume that the errors are independent in time. Well, in practice, given that most of the dependence structure in the observations is contained in the process, the structure of $C_{\epsilon,t}$ should be very simple. It is often assumed that these data-model errors are independent with constant variance in time and space, so that $C_{\epsilon,t} = \sigma_\epsilon^2I_{m_t}$.

### Non-Gaussian and Nonlinear Data Model
Non Gaussian data model use a transformation of the mean response $g(Y_t(s))\equiv \hat Y_t(s)$. We can also include a mapping matrix to the non-Gaussian data model.

$Z_t\|Y_{t,\gamma} \sim EF(H_tY_t,\gamma)$

To consider a nonlinear transformation of the latent process (even if the error time is Gaussian), the simplest way is to add a power transformation applied to each element of $Y_t$.

$Z_t = b_t +H_tY_t^a+\epsilon_t, \quad \epsilon_t\sim Gau(0, C_{\epsilon,t}) \tag{6}$

In some applications it is reasonable to assume that the transformation power $a$ vary with space or time and may depend on covariates.

### Process Model
> Linear Markovian spatio-temporal process models generally assume that the value of the process at a given location at the present time is made up of a weighted combination (or is a “smoothed version”) of the process throughout the spatial domain at previous times, plus an additive, Gaussian, spatially coherent “innovation”.

A first-order spatio-temporal inegro-difference equation (IDE) process model is:

$Y_t(s) = \int_{D_s}m(s,x;\theta_p)Y_{t-1}(x)dx +\eta_t(s),\quad s,x\in D_s\tag{7}$

where $m(s,x;\theta_p)$ is a *transition kernel*. Notably, the *asymmetry* and *rate of decay* of the transition kernel m controls *propagation*(linear advection) and *spread* (diffusion) respectively. Spatially coherent disturbances tend to spread across space at a greater rate when the kernel is wider, which leads to more averaging from one time to the next. The offset kernel pulls information from one particular direction and redistributes it in the opposite direction, leading to propagation.

It is often assumed that $\theta_p$ does not vary with time. In (7), it is assumed that the process $Y_t(\cdot)$ has mean zero.
However, it sometimes may be more appropriate to model a non-zero mean, which will be discussed later on. In general, $\int D_sm(s,x;\theta_p)dx < 1$ is required for the process to be stable (non-explosive) in time.

When we consider a finite set of prediction spatial locations or regions (e.g., an irregular lattice or a regular grid), the first order IDE (7) can be discretized as a stochastic difference equation,

$Y_t(s_i) = \sum_{j=1}^nm_{ij}(\theta_p)Y_{t-1}(s_j)+\eta_t(s_i)\tag{8}$

Now, defining the process vector $Y_t \equiv (Y_t(s_1),\dots,Y_t(s_n))'$, (8) can be written in vector-matrix form as:

$Y_t = MY_{t-1}+\eta_t \tag{9}$

The stability condition requires that the maximum modulus of the eigenvalues of M (which may be complex-valued) be less than 1. 
If one fits an unconstrained linear DSTM to data that come from a nonlinear process (many real-world spatial-temporal processes are nonlinear), then the fitted model is unstable, that is, explosive with exponential growth. It indicates that the wrong model is being fitted or that the finite-time window for the observations suggests a `transient period of growth`. Long-lead-time forecasts from such a model are problematic, but for short-term forecasts, it is actually helpful as the mean of the predictive distribution averages over realizations that are both explosive and non-explosive.

> `transient growth` is due to that the transition operator M is stable (the maximum modulus of the eigenvalues of M is less than 1) but M is "non-normal" (in discrete-space case, if $MM'\ne M'M$, in which case, the eigenvectors of M are non-orthogonal).

### Process and Parameter Dimension Reduction
The latent linear Gaussian DSTM has unknown parameters associated with the data model $C_\epsilon$, the transition operator $m(s,x;\theta_p)$ and the initial-condition distribution (e.g., $\mu_0$ and $C_0$). With the linear Gaussian data model, one typically considers a simple parameterization of $C_\epsilon$ such as $C_\epsilon = \sigma_\epsilon^2I$. Or we can use the covariance matrix implied by a simple spatial random process that has just a few parameters such as a `matern spatial covariance function` or a `spatial condtional autoregressive process`. 

One of the biggest challenges in DSTMs in hierachical modeling settings is `the curse of dimensionality` associated with the process model. A common situation is the number of locations $n$ is much larger than the number of time $T$. The DSTM process model will be problematic as there are on the order of $n^2$ parameters to estimate. Two approaches are discussed in the book.

#### Parameter Dimension Reduction
The process-error spatial variance-covariance matrix $C_\eta$ can be represented by one of the common spatial covariance function or a basis-function random effects. No need to estimate the full positive definite matrix in the DSTM.

The authors highlighted that the transition matrix parameters could have as many as $n^2$ parameters and thereby need extra care.For a simple linear DSTM (9), we could parameterize the transition matrix as:
1. a random walk; $M = I$
2. a spatially homogeneous autoregressive process; $M =\theta_pI$
3. a spatially varying autoregressive process; $M =\text{diag}(\theta_p)$

The third parameterization is more realistic and useful for real-world processes than the first two. Based on this parameterization, where $C_\eta = \sigma_\eta^2I$ and $M =\text{diag}(\theta_p)$, we can decompose the first-order conditional distribution as:

$[Y_t\|Y_{t-1},\theta_p,\sigma_\eta^2] = \prod_{i=1}^n[Y_t(s_i)\|Y_{t-1}(s_i),\theta_p(i),\sigma_\eta^2], \quad t = 1,2,\dots\tag{10}$

Conditional on the parameters $\theta_p = (\theta_p(1),\dots,\theta_p(n))'$, we have spatially independent univariate AR(1) processes at each spatial location. However, it is worth noting that, if $\theta_p$ is random and has spatial dependence, then the marginal conditional distribution $[Y_t\|Y_{t-1},\sigma_\eta^2]$ after we integrate $\theta_p$ out, can imply that all of the elements of $Y_{t-1}$ influence the transition to time $t$ at all spatial locations (i.e., this is non-separable spatio-temporal process). For forecasting applications, we often seek parameterizations that directly include interactions in the conditional model.

The authors again pointed that the transition kernel is important. We can model realistic linear behavior by parameterizing the kernel shape (decay and asymmetry) in terms of the kernel width, variance, and shift, or mean. And if we let these parameters to change with space then we can model quite complex dynamics with a relatively small number of parameters. For example, the IDE process model given by (7) can be specifed with a Gaussian-shape transition kernel as a function of $x$ relative to the location $s$ in a 1D spatial domain (for simplicity).

$m(s,x;\theta_p) = \theta_{p,1}\text{exp}\left\(-\frac{1}{\theta_{p,2}}(x-\theta_{p,3}-s)^2\right\)\tag{11}$

On the right side of the above equation, the three parameters about $\theta_p$ are respectively the kernel amplitude, variance (kernal scale), an mean (shift) relative to location s. (11) is positive nut need not integrate to 1 over $x$. Recall the wider kernels suggest faster decay, and postive shift leads to leftward movement. To obtain more complex dynamical behavior, we can allow these parameters to change with space. For example, let the mean (shift) parameter satisfies: $\theta_{p,3} = x(s)'\beta + w(s)$, where $x(s)$ corresponds to covariates at sptial location $s$, and $\beta$ are the regression parameters. The authors also mentioned that we can vary $\theta_{p,2}$ but $\theta_{p,3}$ is often more important. 

Although the IDE kernel is feasible for continous space, three are many occasions that we need efficient parameterizations in (1) a discrete-space setting or (2) in the context of random effects in basis-function expansions. The authors then discussed in discrete space the transition operators only consider local spatial neighborhoods.

#### Lagged-Nearest-Neighbor (LNN) Representation 
For discrete space, a very parsimonious yet realistic dynamical model can be specifed as:

$Y_t(s_i) = \sum_{s_j\in N_i} m_{ij}Y_{t-1}(s_j)+\eta_t(s_i)\tag{12}$

Where $N_i$ corresponds to a pre-specifed neighborhood of the location $s_i$. For those who are familiar with spatial regression and sptial weights, it is easy to understand how neighborhoods of locations can weight $Y_{t-1}$ to $Y_t$. The parameters can be further parameterized to account for decay (spread and diffusion) rate and asymmetry (propagation direction). Of course, we can also let the parameters vary in space. 

#### Motivation of an LNN with a Mechanistic Model
The LNN parameterization can be motivated by mechanistic models such as integro-diffential or partial differential equations (PDEs). In PDE, the parameters $m_{ij}$ in (12) can be parameterized in terms of knowledge of mechanistics, such as spatially varying diffusion or advection coefficients. Consider the basic linear, non-random, advection-diffusion PDE,

$\frac{\partial Y}{\partial t} = a \frac{\partial^2Y}{\partial x^2} + b \frac{\partial^2Y}{\partial x^2} +u \frac{\partial Y}{\partial x}+v\frac{\partial Y}{\partial y}\tag{13}$

where $a$ and $b$ are diffusion coefficients that control the rate of spread, the $u$ and $v$ are advection parameters that control the flow. Simple finite-difference discretization of such PDEs on a 2D equally spaced finite grid can lead to LNN specifications of the form:

$Y_t = M(\theta_p)Y_{t-1} + M_b(\theta_p)Y_{b,t}+\eta_t \tag{14}$

$Y_t$ corresponds to non-boundary grid points, and $Y_{b,t}$ are boundary process. $\eta_t$ is assumed to be Gaussian, mean-zero, and independent in time. These parameters can vary with space as well, and we model them either in terms of covariates or as a spatial random process. 

### Dimension Reduction in the Process Model
In the previous post, we have learned to use spatio-temporal random effect model to reduce process dimensionality. This is particularly helpful for DSTM process model.

Consider an extension to the spatial basis-function mixed-effects model,

$Y_t(s) =x_t(s)'\beta+\sum_{i=1}^{n_\alpha}\phi_i(s)\alpha_{i,t}+\sum_{j=1}^{n_\xi}\psi_j(s)\xi_{j,t}+\nu_t(s)\tag{15}$

where $\alpha_{i,t}$ are the dynamically evolving random coefficients, and $\xi_{j,t}$ are non-dynamical. Both basis functions $\phi_i(\cdot)$ and $\psi_j(\cdot)$ are assumed known.$\nu_t(\cdot)$ is assumed to be a Gaussian process with mean zero and independent in time.

The vector form of (15) is:

$Y_t = X_t\beta + \Phi\alpha_t + \Psi\xi_t+\nu_t\tag{16}$

where $X_t$ is an $n\times (p+1)$ matrix that could be time-varying and can be interpreted as a spatial offset corresponding to large-scale non-dynamical feastures or covariate effects. $\Phi$ is an $n\times n_alpha$ matrix of basis vectors, and $\Psi$ is an $n\times n_\xi$ matrix of basis vectors. $\alpha_t$ and $\xi_t$ are respectively the latent dynamical and non-dynamical coefficients.

The dynamical coefficient process $\alpha_t$ can evolve according to the linear equation. For example, we can specify a first-order vector autoregressive model (VAR(1)),

$\alpha_t = M_\alpha \alpha_{t-1} +\eta_t \tag{17}$

Again, the transition operator has to be non-normal (i.e., $M_\alpha'M_\alpha \neq M_\alpha M_\alpha'$), as almost all real-world linear processes correspond to non-normal transition operators. 

Importantly, the notion of "neighbors" is not always well defined in these formulations. If the basis functions given in $\Phi$ are global basis functions such as some types of splines, Fourier, EOFs, etc., the elements of $\alpha_t$ are not spatially indexed.

The authors then discussed the choice of basis functions. For DSTMs, it is usually important to specify basis functions such that interactions across spatial scales can accommodate transient growth. This is difficult to achieve in "knot-based" representations such as B-splines, kernel convolutions, predictive processes, where $\alpha_t$ are spatially referenced by not multi-resolutional. The dynamical evolution in the DSTM can accommodate scale interactions by using global basis functions such as EOFs, Fouries, etc.

### Nonlinear DSTMs
Many mechanistic processes are nonlinear at spatial and temporal scales of variability. We can write the nonlinear spatio-temporal AR(1) process as:

$Y_t(\cdot) = M(Y_{t-1}(\cdot),\eta_t(\cdot);\theta_p),\quad t=1,2, \dots\tag{18}$

Here, $M$ is a nonlinear function. There are an infinite number of nonlinear statistical models. We can either (1) take a nonparameteric view of the problem and learn the dynamics from the data, or (2) propose specific model classes that can accommodate the dynamical behaviors.

#### State-Dependent Models
`State-Dependent models` consider that the transition matrix $M$ depend on the process (state) value $Y_{t-1}$ at each time and parameters $\theta_p$ which may vary with time and space.

$Y_t = M(y_{t-1};\theta_p)Y_{t-1} + \eta_t \tag{19}$

One type of state-dependent model is the *threshold vector autoregressive model*,

$Y_t = \begin{cases}
M_1Y_{t-1} + \eta_{1,t},  & \text{if $f(w_t) \in d_1$} \\\\ 
\vdots & \vdots \\\\ 
M_KY_{t-1} + \eta_{K,t}  & \text{if $f(w_t) \in d_K$}
\end{cases}\tag{20}$

where $f(w_t)$ is a function of a time-varying parameter $w_t$ that can itself be a function of the process $Y_(t-1)$. 

The transition matrices $\\{M_1,\dots,M_k\\}$ and error covariance matrices $\\{C_{\eta_1},\dots,C_{\eta_K}\\}$ depend on unknown parameters. A big challeng is to reduce the dimensionality of the parameter space.

#### General Quadratic Nonlinearity
A large number of real-world processes in the physical and biological sciences exhibit quadratic interactions. Consider the following one-dimensional reaction-diffusion PDE:

$\frac{\partial Y}{\partial t} = \underbrace{\frac{\partial}{\partial x} \left\(\delta \frac{\partial Y}{\partial x}\right\)}_{\text{diffusion term linear in Y}}  + Y \text{exp}\left\(\gamma_0(1-\frac{Y}{\gamma_1})\right\)\tag{21}$

where the first term corresponds to a diffusion (spread) term that depends on $\delta$, and the seccond term corresponds to a density-dependent growth term with `growth parameter` $\gamma_0$ and `carrying capacity parameter` $\gamma_1$. Each of these parameters can vary with space and time. The first term is linear in Y but the second term is nonlinear.

In discrete space and time, a `general quadratic nonlinear` (GQN) DSTM can be written as:

$Y_t(s_i) = \sum_{j=1}^{n}m_{ij}Y_{t-1}(s_j) + \sum_{k=1}^n\sum_{l=1}^n b_{i,kl}g(Y_{t-1}(s_l);\theta_g)Y_{t-1}(s_k)+\eta_t(s_i)\tag{22}$

$b_{i,kl}$ corresponds to the quadratic interaction transition coefficients. The function $g(\cdot)$ transforms one of the components of the qaudratic interaction. This transformation is important for many processes such as epidemic or invasive-species population processes. The conditional GQN for $Y_t(\cdot)$ on $Y_{t-1}(\cdot)$ is Gaussian, but the marginal model for Y_t(\cdot) is in general not Gaussian due to the non-linear interactions. GQN is a special case of the state-dependent model.

#### Some other Nonlinear Models
The authors introduced other promising approaches for non-linear spatio-temporal modeling.

1. Variants of neural networks (e.g., RNNs). The orignal formulations do not address uncertainty quantification. However, there is increasing interest in considering deep learning models within broader uncertainty-based paradigms. We will have a new series of posts learning deep learning related spatio-temporal modeling.
* due to high dimensionality of the hidden states and parameters, it typically requires complex regularization or prior information to make them work. 
* `echo state network` (ESN) consider sparsely connected hidden layers that allow for sequential interactions yet assume most of the parameters are randomly generated and then fixed, with only parameters estimated being those that connected to the hidden layer to the response. However, ESN does not include quadratic interactions or formal uncertainty quantification.

2. *agent-based model*, the process is built from local individual-scale interactions by simple rules that lead to complex nonlinear behavior. ALthough these models are parsimonious but are computational expensive, and paramter inference can be challenging.

3. "mechanism-free" approach, so called "analogs". It seeks to final historical sequences of analogs that match a similar sequence culminating at the current time.  

> One of the fundamental differences between DSTMs and multivariate time series models is that DSTMs require scalable parameterization of the evo- lution model.
