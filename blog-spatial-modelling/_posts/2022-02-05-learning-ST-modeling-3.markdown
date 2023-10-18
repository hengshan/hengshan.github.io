---
layout: post
title:  "Learning Spatial-Temporal Modeling (3) Descriptive Spatio-Temporal Models"
date:   2022-02-05 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

This post is based on the book [Spatio-Temporal Data in R](https://spacetimewithr.org/){:target="_blank"} chapter 4. In this chapter, the authors focus on (1) prediction at some location in space within the time span of the observations (i.e., smoothing) and (2) 
parameter inference for spatio-temporal covariates.

For both goals, it is assumed that observations can be decomposed to a latent spatio-temporal process and observation error. The latent process can then be written in terms of fixed effects due to covariates plus a spatio-temporally dependent random process.

The main concern of this post is to learn how to describe the dependence structure in the random process.

### Additive Measurement Error and Process Models
In a previous post, we have formalize the observations as $Z(s_i,t_j)$ for location ${s_i: i=1, \dots, m}$ and time ${t_j: j=1, \dots, T}$. The problem with this kind of formalization is that at each time stamp all observations can only have the same locations.

To make it more general, the authors of the book gave a new definition. At each time $t \in {t_1, \dots, t_T}$ we have $m_{t_j}$ observations. Note that at different time the number of observations might be different. We write the number of observations at time $t_j$ as $m_j$ instead of $m_{t_j}$ for simplicity. The vector of all observations is then:

$\mathbf{Z} = (Z(s_{11};t_1),Z(s_{21};t_1),\dots,Z(s_{m_11};t1),\dots,Z(s_{1T};t_T),\dots,Z(s_{m_TT};t_T))'))$

We aim to predict at a spatial-temporal location $(s_0;t_0)$. As we have learned in a [previous post]({% post_url 2022-01-01-the-goal-of-spatial-modeling%}){:target="_blank"}, if $t_0 < t_T$, we are in a smoothing situation; if $t_0 = t_T$, we are in a filtering situation, and if $t_0 > t_T$ then we are in a forecasting situation.

We represent the data in terms of the latent spatial-temporal process plus a measurement error.

$Z(s_{ij};t_j) = Y(s_{ij};t_j) + \epsilon(s_{ij};t_j), \quad i = 1, \dots, m_j; \quad j = 1, \dots, T$

The latent process is denoted as ${Y(s;t); s\in D_s, t \in D_t}$. The errors $\epsilon(s_{ij};t_j)$ represent iid mean-zero measurement error with variance $\sigma_\epsilon^2$.

We typically do not have all observations at all locations, and we would like to predict the latent value $Y(s_0;t_0)$ at a spatio-temporal location $(s_0;t_0)$ as a function of the data vector represented by $\mathbf{Z}$.

The latent process $Y(s;t)$ follows the model

$Y(s;t) = \underbrace{\mu(s;t)}_{\text{process mean} \\\\ \text{not random}}+{\eta(s;t)}$

$\eta(s;t)$ represents a mean-zero random process with spatial and temporal statistical dependence. Our goal is to estimate $\hat Y(s_0;t_0)$ to minimize the mean squared error between $Y(s_0;t_0)$ and $\hat Y(s_0;t_0)$. 

When we choose to let $\mu(s;t)$ to be (1) known, (2) constant but unknown, or (3) modeled in terms of $p$ covariates, $\mu(s;t) = x(s;t)'\beta$, the descriptive spatio-temporal model is respectively (1) simple kriging, (2) ordinary kriging, and (3) universal kriging.

### Prediction for Gaussian Data and Processes
The book first gave a big picture about smoothing. In a previous post, we mentioned inverse distance weighting (IDW). Essentially, it is to smooth out the observation uncertainty by using the inverse distance as a "weights", but whether the weights are the optimal and what the "cost function" of the optimization is, we did not know.

Here, the authors highlighted that the cost funtion to minimize is the interpolation error $E(Y(s_0;t_0)-\hat Y(s_0;t_0))^2$ (aka MSPE, mean squared prediction error). The best linear unbiased predictor that minimize MSPE is `kriging predictor`. The kriging weights are determined by the covariances between observation locations, yet respect the measurement uncertainty. The book assume that the underlying process is a *Gaussian process* and the measurement error has a Gaussian distribution.

We have learned Gaussian process in a previous [post]({%post_url 2022-01-10-time-series-1%}){:target="_blank"}. Here, the book introduced GP again.

`Gaussian process` is a stochastic process denoted by ${Y(r): r \in D}$, where $r$ is a spatial, temporal, or spatio-temporal location in $D$. The process has all its finite-dimensional distribution Gaussian, determined by a mean function $\mu(r)$ and a covariance function $c(r,r') = \text{cov}(Y(r),Y(r'))$ for any location ${r, r'} \in D$. Two points about GP.

1. If the mean and covariance functions are known, the process can describe anywhere in the domain. 

2. only finite distributions need to be considered in practice and any finite collection of Gaussian process random variables has a joint multivariate normal distributions.

In the context of S-T kriging, time is treated as another dimension. The covariance function describes covariance between any two space-time locations. Note that we should use covariance functions that respect that durations in time are different from distances in space.

The data model is:

$Z= Y + \epsilon$

where $Y \equiv (Y(s_{11};t_1),\dots,Y(s_{m_tT};t_T))'$ and $\epsilon \equiv (\epsilon(s_{11};t_1),\dots,\epsilon(s_{m_tT};t_T))'$.

The process model is:

$Y = \mu +\eta$

Note that $\text{cov}(Z) \equiv C_z = C_y + C_\epsilon$, $\text{cov}(Y) \equiv C_y = C_\eta$.

Now defining $c_0' \equiv \text{cov}(Y(s_0;t_0),Z), c_{0,0} \equiv \text{var}(Y(s_0;t_0))$, and $X$ the $(\sum_{j=1}^Tm_j)\times (p+1)$ matrix. Consider the joint Gaussian distribution,

$\begin{bmatrix}Y(s_0;t_0) \\\\ Z \end{bmatrix} \sim Gau\left\(\begin{bmatrix}x(s_0;t_0)' \\\\ X \end{bmatrix}\beta, \begin{bmatrix}c_{0,0} & c_0' \\\\ c_0 & C_z\end{bmatrix}\right\)\tag{1}$

For S-T simple kriging, the mean is known, so $\beta$ is known, we can obtain the conditional distribution,

$Y_(s_0;t_0) \| Z \sim Gau(x(s_0;t_0)'\beta + c_0'C_z^{-1}(Z-X\beta), c_{0,0} - c_0'C_z^{-1}c_0)\tag{2}$

It is important to understand that the process is defined for an uncountable set of locations and the data correspond to a partial realization of this Gaussian process. 

Another important observation is that (2) is a predictor of the hidden value $Y(s_0;t_0)$ not of $Z(s_0;t_0)$. As shown in (1), the conditional mean takes residuals between the observations and their marginal means (i.e., $Z-X\beta$), weighted by $w' \equiv c_0'C_z^{-1}$, and adds the result back to the marginal mean corresponding to the prediction loction (i.e., $x(s_0;t_0)'\beta$)

The weight $w$ is a function of the covariances and the measurement error variance. Recall the $\text{cov}(Z) \equiv C_z = C_y + C_\epsilon$, $\text{cov}(Y) \equiv C_y = C_\eta$. The book also illustrated another way of thinking:

> The trend term $x(s_0;t_0)'\beta$ is the mean of $Y(s_0;t_0)$ *prior to* considering the observations; then the simple S-T kriging predictor combines this prior mean with a weighted average of the mean-corrected observations to get a new, conditional mean. 

> Similarly, one can interprets $c_{0,0}$ as the variance prior to considering the observations, then the conditional variance reduces this initial variance by an amount given by $c_0'C_z^{-1}c_0$.

The authors then gave an example to illustrate simple kriging and pointed out that in most real-world problems, one would not know $\beta$. In this case, the optimal pridiction problem is analogous to the estimation of effects in a linear mixed model including both fiexed effects of regression terms and random effects $\eta$. The *universal kriging* predictor of $Y(s_0;t_0)$ is:

$\hat Y(s_0;t_0) = x(s_0;t_0)'\hat\beta_{gls} + c_0'C_z^{-1}(Z - X\hat\beta_{gls})\tag{3}$

where the generalized least squares (gls) estimator of $\beta$ is:

$\hat\beta_{gls} \equiv (X'C_z^{-1}X)^{-1}X'C_z^{-1}Z\tag{4}$

The associated S-T universal kriging variance is given by

$\sigma_{Y,uk}^2(s_0;t_0) = c_{0,0} - c_0'C_z^{-1}c_0 + \kappa \tag{5}$

where $\kappa \equiv (x(s_0;t_0) - X'C_z^{-1}c_0)'(X'C_z^{-1}X)^{-1}(x(s_0;t_0)-X'C_z^{-1}c_0)$ represents the additional uncertainty brought due to the estimation of $\beta$.

So far we have assumed that we know the variances and covariances that make up $C_y, C_\epsilon, c_0, c_{0,0}$. In reality, we have to estimate the parameters through maximum likelihood, restricted maximum likelihood, or through a Bayesian implementation in which case we specify prior distributions for the parameters. 

### Spatio-temporal Covariance Functions
We need to know the spatio-temporal covariances between the hidden random process evaluated at any two locations in space and time. To achieve this, we need a covariance function, which has to be non-gegative-definite, as the variances are non-negative. 

$c_*(s,s';t,t') \equiv \text{cov}(Y(s;t),Y(s';t'))$

Here, $s'$ represents a different spatio-temporal location.

Classical-kriging implementations assume second-order stationarity so that the model needs less or parsimonious parameters.

$c_*(s,s';t,t') = c(s'-s;t'-t) = c(h;\tau)$

In practice, it is unlikely that the spatio-temporal stationary covariance function is completely known and it is usually specfied in terms of some parameters $\theta$. 

#### Separable (in Space and Time) Covariance Functions
For spatio-temporal covariance functions, it is important to ensure that the function is valid. How do we ensure that?

The most convenient way to guarantee validity is to use separable classes for space and time.

$c(h; \tau) \equiv c^{(s)}(h)\cdot c^{(t)}(\tau)$

If both $c^{(s)}(h)$ and $c^{(t)}(\tau)$ are valid, then the spatio-temporal covariance function is valid. 

The authors listed a number of valid spatial and temporal covariance functions such as the Matern, power exponential, and Gaussian, etc. For example, the exponential covariance function, which is a special case of both the Matern covariance function and the power exponential covariance function, is given by:

$c^{(s)}(h) = \sigma_s^2\text{exp}\left\\{-\frac{\Vert h\Vert}{a_s}\right\\}$

where $\sigma_s^2$ is the variance parameter and $a_s$ is the spatial-dependence parameter in units of distance. The larger $a_s$ is, the more dependent the spatial process is. We can use the same function for time. 

As a result, the spatio-temporal *correlation* function $\rho(h;\tau) \equiv c(h;\tau)/c(0;0)$, is given by

$\rho(h;\tau) =\rho^{(s)}(h;0)\cdot\rho^{(t)}(0;\tau)$

The right sides of the above equation are respectively the marginal spatial and temporal correlation functions. In this case, one an write $C_z = C_z^{(t)} \otimes C_z^{(s)}$, where $\otimes$ is the Kronecker product.

> A consequence of the separability property is that the temporal evolution of the process at a given spatial location does not depend directly on the process’ temporal evolution at other locations.

This is very seldom the case for real world processes. The authors then introduced three approaches to obtain spatio-temporal covariance functions.
1. sums and products formulation
2. Bochner's theorem, which is related to the spectral representation to the covariance representation; e.g., the inverse Fourier transform is a special case
3. covariance functions from the solution of stochastic partial differential equations (SPDEs). 

#### Sums-and-Products Formulation
The product and sum of two non-negative-definite functions is non-negative-definite. Thus, we can construct:

$c(h; \tau) \equiv p c_1^{(s)}(h)\cdot c_1^{(t)}(\tau) + q c_2^{(s)}(h) + rc_2^{(t)}(\tau)$

The authors introduced the concept of *fully symmetric* of covariance, which is defined as:

$\text{cov}(Y(s;t),Y(s';t')) = \text{cov}(Y(s;t'), Y(s';t))$

> separable covariance functions are always fully symmetric, but fully symmetric covariance functions are not always separable.

#### Construction via a Spectral Representation
The authors gave an example of Cressie and Huang (1999). For more details, pls refer to the book.

#### Stochastic Partial Differential Equation (SPDE) Approach
The authors state that:
> The SPDE approach to deriving spatio-temporal covariance functions was originally inspired by statistical physics, where physical equations forced by random processes that describe advective, diffusive, and decay behavior were used to describe the second moments of macro-scale processes, at least in principle.

This book gave an example of Matern spatial covariance function and mentioned that although SPDE can suggest non-suparable spatio-temporal covariance functions, only a few simple cases lead to closed-form functions. The authors pointed out that macro-scale real-world processes of interest are seldom linear and stationary in space and time. The spatio-temporal covariance functions than can be obtained in closed form from SPDEs are usually not appropriate for physical processes although they may provide good fits.

### Spatio-Temporal Semivariograms
Spatio-Temporal variogram is defined as:

$\text{var}(Y(s;t)-Y(s';t')) \equiv 2\gamma(s,s';t,t')$

where $\gamma(\cdot)$ is called the *semivariogram*. The stationary version is denoted by $2\gamma(h;\tau)$. The process Y is considered to be *intrinsically stationary* if it has a constant expectation and a stationary variogram. 

If the process is second-order stationary (a stronger restriction), the relationshiop between semivariogram and the covariance function is:

$\gamma(h;\tau) = c(0;0) - c(h;\tau)$

Notice that stronger spatio-temporal dependence/correlation corresponds to smaller semi-semivariogram values.

Notably, the authors mentioned that in spatio-temporal analysis, there is no preference of using variograms for dependence, mainly due to that *most real-world processes are best characterized in the context of local second-order stationarity*. 

> The difference between intrinsic stationarity and second-order stationarity is most appreciated when the lags h and τ are large. If only local stationarity is expected and modeled, the extra generality given by the variogram is not needed. 

> A price to pay for this extra generality is the extreme caution needed when using the variogram (i.e., intrinsically stationary processes) to find optimal kriging coefficients. 

> the universal-kriging weights may not sum to 1 and, in situations where they do not, the resulting variogram-based kriging predictor will not be optimal. However, when using the covariance-based kriging predictor, there are no such issues and it is always optimal.

### Gaussian Spatio-Temporal Model Estimation
In spatial-temporal context, the authors prefer to consider fully parameterized covariance models and infer the parameters through likelihood-based methods or through fully Bayesian methods.

#### Likelihood Estimation
$C_z = C_y + C_\epsilon$, so $C_z$ depends on parameters $\theta \equiv \\{\theta_y,\theta_\epsilon\\}$. The likelihood can be written as:

$L(\beta,\theta; Z) \propto {\vert C_z(\theta)\vert}^{-1\over 2} \text{exp}\left\\{ -{1\over 2} (Z-X\beta)'(C_z(\theta))^{-1}(Z-X\beta) \right\\} $

We maximumize this with respect to $\\{\beta,\theta\\}$ and get the maximum likelihood estimtion $\\{\hat\beta_{mle},\hat\beta_{mle}\\}$. Because the covariance parameters are in the matrix inverse and determinant, numerical methods have to be used for maximization. 

The authors introduced the profiling method:

$\beta_{gls} = (X'C_z(\theta)^{-1}X)^{-1}X'C_z(\theta)^{-1}Z$

$\beta_{mle} = (X'C_z(\theta_{mle})^{-1}X)^{-1}X'C_z(\theta_{mle})^{-1}Z$

The parameters estimates $\\{\hat\beta_{mle},\hat\beta_{mle}\\}$ are then substituted into the kriging equations (3) and (5) to obtain the empirical best linear unbiased predictor (EBLUP) and the associated empirical prediction variance.

> Among the most popular functions in base R are nlm, which implements a Newton-type algorithm, and optim, which contains a number of general-purpose rou- tines, some of which are gradient-based.

#### Restricted Maximum Likelihood (REML)
REML considers the Likelihood of a linear transformation of the data vector such that the errors are orthogonal to the Xs.

Consider a contrast matrix $K$ such that $E(KZ) = 0$. It follows that $E(KZ) = KX\beta =0$, and $\text{var}(KZ) = KC_Z(\theta)K'$

$L_{reml}(\theta; Z) \propto {\vert KC_z(\theta)K'\vert}^{-1\over 2} \text{exp}\left\\{ -{1\over 2} (KZ)'(KC_z(\theta)K')^{-1}(KZ)\right\\}$

#### Bayesian Inference
In the above method, we treat $\beta$ and $\theta$ as fixed unknown and to be estimated. In Bayesian Inference, prior distributions of these parameters could be provided. The choices of parameters estimations must be obtained through numerical evaluation of the posterior distribution.

### Random-Effects Parameterizations
I like the way this book introduces mixed-effect model, as the authors use a longitudinal study to illustrate the model, and I have worked on longitudinal data analysis for a long time. Let's say in a study 90 subjects were randomly assigned to three treatment groups (control, treament 1, and treatment 2). Each subject had 20 responses through time. The response is generally linear with time, with individual specific random intercepts and slopes. The model can be written as:

$ Z_{ij} =
\begin{cases}
(\beta_0 + \alpha_{0i}) + (\beta_1 + \alpha_{1i})t_j + \epsilon_{ij},  & \text{if the subject receives the control} \\\\ 
(\beta_0 + \alpha_{0i}) + (\beta_2 + \alpha_{1i})t_j + \epsilon_{ij},  & \text{if the subject receives the treatment 1} \\\\ 
(\beta_0 + \alpha_{0i}) + (\beta_3 + \alpha_{1i})t_j + \epsilon_{ij},  & \text{if the subject receives the treatment 2}
\end{cases}$

Where $Z_{ij}$ is the response for the $i$th subject ($i = 1, \dots, n =90$) at time $j = 1, \dots, T = 20$; $\beta_0$ is an overall fixed intercept; $\beta_1,\beta_2,\beta_3$ are fixed time-trend effects; and $\alpha_{0i} \sim iidGau(0,\sigma_1^2)$ and $\alpha_{1i} \sim iidGau(0, \sigma_2^2)$ are subject-specific random intercept and slope effects.

The model is then written in the classical linear mixed model as:

$Z_i = X_i\beta + \Phi\alpha_i +\epsilon_i$

where $Z_i$ is a 20-dimensional vector for the $i$th subject; $X_i$ is a 20 X 4 matrix consisting of a column vector of 1s and three columns indicating the treatment group; $\beta$ is a four-dimensional vector of fixed effects; $\Phi$ is a 20 X 2 matrix with a vectors of 1s and a column consisting of the vector of times. The random effect vector is $\alpha_i \equiv (\alpha_{0i},\alpha_{1i})' \sim Gau(0,C_\alpha)$, where $C_\alpha = \text{diag}(\sigma_1^2,\sigma_2^2)$, and $\epsilon_i \sim Gau(0, \sigma_\epsilon^2\mathbf{I})$ is a 20-dimensional error vector. We assume that the elements of $\\{\alpha_i\\}$ and $\\{\epsilon_i\\}$ are all independent.

Notably, responses that share common random effects exhibit marginal dependence through the marginal covariance matrix, and so the inference on the fixed effects (e.g., via generalized least squares) then accounts for this more complicated marginal dependence.

In the context of spatio-temporal modeling, we can also write the process of interest conditional on random effects (spatial, temporal, or spatio-temporal). It allows us to build spatio-temporal dependence conditionally. In this way, the implied marginal spatio-temporal covariance function is always valid. 

### Basis-Function Representations
We have learned basis functions in a previous post. Here, the authors introduced basis functions from the perspective of mixed-effect model discussed above. 

In the previous post, we used the basis function coefficients are fixed but unknown and to be estimated, then we have a regression model. In this post, we have a mixed-effects model, with basis functions serving as random covariates.

#### Random Effects with Spatio-Temporal Basis Functions
The process model in terms of fixed effects $\beta$ and random effects $\\{\alpha_i:i = 1,\dots,n_\alpha\\}$ can be rewritten as:

$Y(s;t) = x(s;t)'\beta + \eta(s;t) = x(s;t)'\beta + \sum_{i=1}^{n_\alpha}\phi_i(s;t)\alpha_i+\nu(s;t)$

$\\{\alpha_i\\}$ are *random effects*, and $\nu(s;t)$ is a residual error term that is not captured by the basis functions.

The process model $Y$ at $n_y$ spatial-temporal locations can be denote by:

$Y = X\beta + \Phi\alpha + \nu$

where the $i$th column of the $n_y \times n_\alpha$ matrix $\Phi$ corresponds to the ith basis function. The vector $\nu$ corresponds to the spatio-temporal ordering given in Y, and $\nu \sim Gau(0, C_\nu)$. The marginal distribution of $Y$ is given by $Y \sim Gau(X\beta, \Phi C_\alpha\Phi'+C_\nu)$

> The spatio-temporal dependence is accounted for by the spatio-temporal basis functions and in general this could accommodate non-separable dependence. 

> The random effects α are not indexed by space and time, so it should be easier to specify a model for them. For example, we can specify a covariance matrix to describe their dependence, which is easier than specifying a covariance function.

The authors then introduced some computational benefit of mixed effect models. Based on the above formula, we can write $C_z = \Phi C_\alpha \Phi' + C_\nu+C_\epsilon$, and we define $V \equiv C_\nu + C_\epsilon$.

Using the Sherman–Morrison–Woodbury matrix identities, we can write:

$C_z^{-1} = V^{-1}-V^{-1}\Phi(\Phi'V^{-1}\Phi+C_\alpha^{-1})^{-1}\Phi'V^{-1}$

Basis-function implementations may assume that $\nu =0$ and that $\Phi$ is orthogonal, so that $\Phi\Phi' = I$ to reduce the computational burden significantly.

> The definition of “basis function” in our spatio-temporal context is pretty liberal; the matrix $\Phi$ in the product $\Phi\alpha$ is a spatio-temporal basis-function matrix so long as its coefficients α are random and the columns of $\Phi$ are spatio-temporally referenced.

The authors introduced three ways of choosing basis functions:
1. fixed or parameterized basis functions
2. local or global basis functions
3. reduced-rank, complete, or over-complete bases
4. basis func- tions with expansion coefficients possibly indexed by space, time, or space-time. 

The choice of basis functions is affected by the presence and type of residual structure and the distribution of the random effects. One simplification is to use tensor product basis functions (i.e., the product of a spatial basis function and a temporal basis function). This generally yields a non-separable spatio-temporal model. 

#### Random Effects with Spatial Basis Functions
We can also view spatio-temporal-dependence models from the perspective that the statistical dependence comes from spatial-only basis functions whose coefficients are temporal stochastic processes.

$Y(s;t_j) = x(s;t_j)'\beta + \sum_{i=1}^{n_\alpha}\phi_i(s)\alpha_i(t_j)+\nu(s;t_j), \quad j=1,\dots,T$

The basis functions of the spatio-temporal process are functions of space only and their random coefficients are indexed by time.

> We can consider a wide variety of spatial basis functions for this model, and again these might be of reduced rank, of full rank, or over-complete. For example, we might consider com- plete global basis functions (e.g., Fourier), or reduced-rank empirically defined basis functions (e.g., EOFs), or a variety of non-orthogonal bases (e.g., Gaussian functions, wavelets, bisquare functions, or Wendland functions).

It is often not important which basis function is used. Just ensure that the type and number of basis functions are flexible and large enough to model the true dependence. 

To model more complex spatio-temporal dependence structure using spatial-only basis functions, one must  specify the model for the random coefficients such that $\\{\alpha_{t_j}: j = 1,\dots,T\\}$ are dependent in time. This is simplified by assuming conditional temporal dependence (dynamics), which we will learn in the next post. 

#### Random Effects with Temporal Basis Functions
$Y(s;t_j) = x(s;t_j)'\beta + \sum_{i=1}^{n_\alpha}\phi_i(t)\alpha_i(s)+\nu(s;t)$

The authors also mentioned that most spatio-temporal processes have a scientific interpretation of spatial processes evolving in time. Thus, temporal-basis-function representation is less common than spatial-basis-function. However, temporal-basis-functions are increasingly being used in complex seasonal or high-frequency time behaviors that vary across space.

#### Confounding of Fixed Effects and Random Effects
> If primary interest is in inference on the fixed-effect parameters, then one should mitigate potential confounding associated with the random effects. One strategy is to restrict the random effects by selecting basis functions in $\Phi$ that are orthogonal to the column space of $X$.

> If prediction of the hidden process Y is the primary goal, one is typically much less concerned about potential confounding.

### Non-Gaussian Data Models with Latent Gaussian Processes
This is the spatio-temporal manifestation of traditional generalized linear mixed models (GLMMs) and generalized additive mixed models (GAMMs) in statistics.
In the context of spatio-temporal, the situation is a bit more flexible than the GLMM and GAMM, as the data model does not necessarily have to be from the exponential family, as long as allowing conditional independence in the observations conditioned on spatio-temporal structure in the hidden process.

We extend the Gaussian model the transformed mean response in terms of additive fixed effects and random effects.

$g(Y(s;t)) = x(s;t)'\beta + \eta(s;t), \quad s \in D_s, t \in D_t$

where $g(\cdot)$ is a specified monotonic link function. $x(s;t)$ is a p-dimensional vector of covariates for spatial location and time. Note that $\eta(s;t)$ is a spatio-temporal Gaussian random process that can be modeled in three ways:
1. spatial-temporal covariances. 
2. a special case of which uses a basis-function
3. a dynamic spatio-temporal process (we will learn in the next post).

#### Generalized Additive Models (GAMs)
One way to accommodate nolinear structure in the mean function to build more flexible models is to use GAMs.
These models consider a transformation of the mean response to have an additive form in which the additive components are smooth functions (e.g., splines) of the covariates, where generally the functions themselves are expressed as basis-function expansions. Notably, the basis coefficients are treated as random coefficients in the estimtion procedure in practice. The data model can be written as:

$g(Y(s;t)) = x(s;t)'\beta + \sum_{i=1}^{n_f}f_i(x(s;t);s;t)+\nu(s;t)$

Note that, $f_i(\cdot)$ are functions of the covariates, the spatial locations, and the time index; and $\nu(s;t)$ is a spatio-temporal random effect. The random effect basis function models introduced above are essentially GAMMs. However, GAMMs also the basis functions to not only depend non-linearly on spatio-temporal location but also on covariates. On the other hand, GAMMs typically assume that the basis funtions are smooth functions, but there is no such requirement for spatio-temporal-basis-function models. Given that importance of GAMMs in spatio-temporal modeling, we will have another post to focus on GAMMs models.

#### Inference for Spatio-Temporal Hierarchical Models
Essentially, the inference for spatio-temporal Hierarchical models is to integrate out the latent Gaussian spatio-temporal process. In general the likelihood is:

$[Z \| \theta,\beta] = \int[Z\|Y,\theta][Y\|\theta,\beta]dY$

In spatio-temporal, the above integral cannot be evaluated numerically due to high dimensionality of the integral. A traditional approach is to use conditional-independence assumption in the data model and using the latent Gaussian nature of the random effects. These approaches include:

* Laplace approximation
* quasi-likelihood
* generalized estimating equations
* pseudo-likelihood
* penalized quasi-likelihood.

The authors also highlighted that
> recent advances in automatic differentiation have led to very efficient Laplace approximation approaches for performing inference with such likelihoods, even when there are a very large number of random effects (for example, the Template Model Builder (TMB) R package).

#### Bayesian Hierarchical Modeling (BHM)
The author states that: 
> Although these methods are increasingly being used successfully in the spatial context, there has tended to be more focus on Bayesian estimation approaches for spatio-temporal models in the literature.

Based on Bayes' Rule, the model is:

$[Y, \beta, \theta \|Z] \propto [Z\|Y, \theta_\epsilon][Y\|\beta,\theta_y][\beta][\theta_\epsilon][\theta_y]$

To make inference on the parameters, we focus on the posterior distribution $[\beta,\theta\|Z]$. To make pridiction, we focus on the predictive distribution $[Y\|Z]$.

In general, there is no analytical form for the marginal distribution $[Z]$. A common approach is to use Markov chain Monte Carlo (MCMC) techniques to obtain Monte Carlo (MC) samples from the posterior distribution and then to perform inference on the parameters and prediction of the hidden process by summarizing these MC samples. Another popular approach is to use Hamiltonian Monte Carlo (HMC) to get posterior distribution (see, for example, STAN). We will have anothe post to learn STAN.

If the BHM computational complexity is formidable, we can use integrated nested Laplace approximation (INLA) for latent Gaussian spatial and spatio-temporal processes. The method exploits the Laplace approximation in Bayesian latent-Gaussian models and does not require generating samples from the posterior distribution. Thus, it can deal with quite large datasets. We will learn INLA in another post. 
