---
layout: post
title:  "Spatial-Temporal Modeling (2) Statistical Models"
date:   2022-02-02 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

In this blog, I'd like to introduce in detail the three goals of spatial temporal statistical modeling.

1. predicting at a new location in space given spatial-temporal data (aka spatial-temporal interpolation or smoothing)
2. doing parameter inference with spatial-temporal data
3. forecasting a new value at a future time. 

The three examples basically assume that there are no spatio-temporal dependence. In the next post, we will learn how to model with spatio-temporal dependence.

### Predicting at a new location
We first introduce a deterministic method (IDW) for spatio-temporal prediction. However, this methods does not provide direct estimates of the prediction uncertainty. We will not discuss IDW in this post.

Next, the authors showed how to use a linear regression model with spatio-temporal data. This linear model assumes that all of the spatio-temporal dependence can be accounted for by the trend. 

We consider the case where we have observations at discrete times for all spatial data locations.

$Z(s_i;t_j) = \beta_0 + \beta_1X_1(s_i;t_j)+\dots+\beta_pX_p(s_i;t_j)+e(s_i;t_j)$

The covariates $X_k(s_i;t_j)$ describe explanatory features (aka independent variables) such as elevations, time trends, humidity, etc. The authors then introduced *basis functions*. Imagine that the model is a complex surface in space. We can decompose this surface as a linear combination of some "elemental" basis functions.

$Y(s) = \alpha_1\phi_1(s) +\alpha_2\phi_2(s)+\dots+\alpha_i\phi_i(s)$

We can think of coefficients ${\alpha_i}$ as weights that describe how important each basis function ${\phi_i}$ is in representing the function. Examples of basis functions include polynomials, splines, wavelets, sines and cosines, among many others. The authors use NOAA dataset as an example and considered a linear model with the following basis functions:
let $s_i \equiv (s_{1,i},s_{2,i})'$. This is to get the long-lat coordinates of all locations.
1. overall mean: $X_0(s_i; t_j) =1$
2. linear in lon-coordinate: $X_1(s_i;t_j) = s_{1,i}$
3. linear in lat-coordinate: $X_2(s_i;t_j) = s_{2,i}$
4. linear time (day) trend: $X_3(s_i;t_j)=t_j$
5. long-lat interaction: $X_4(s_i;t_j)=s_{1,i}s_{2,i}$
6. long-t interaction: $X_5(s_i;t_j)=s_{1,i}t_j$
7. lat-t interaction: $X_6(s_i;t_j)=s_{2,i}t_j$
8. additional spatial-only basis functions: $X_k(s_i;t_j)=\phi_{k-6}(s_i), \quad k = 7, \dots, 18$

OLS is used for parameter estimations. 

$\hat Z(s;t)=\hat\beta_0 +\hat\beta_1X_1(s;t)+\dots+\hat\beta_pX_p(s;t)$

$\hat\sigma_e^2 = RSS/(mT-p-1)$

Note that the regression model here does not explicitly account for measurement errors in the responses, so the variation due to measurement error is confounded with the variation due to lack of fit in the residual variance.

As long as the residuals do not have spatio-temporal dependence, we can obtain statistically optimal predictions and optimal forecasts using this method. 

#### Model Diagnostics: Dependent Errors
Next, we look for the presence of outliers, influential observations, non-constant error variance, non-normality, dependence in the errors, and so forth.

We can calculate the spatial-temporal covariogram (or semivariogram) and look for dependence structure as a function of spatial and temporal lags. We can apply a statistical test for temporal dependence such as Durbin-Watson test. For a test of spatial dependence, we can use Moran's I. In looking at spatio-temporal dependence, we can use space-time index. 

The Durbin-Watson test statistic is given by:
\begin{align}
d = \frac{\sum_{t=2}^T(\hat e_t - \hat e_{t-1})^2}{\sum_{t=1}^T\hat e_t^2}
\end{align}

d less than 1 indicate strong positive serial dependence. d larger than 4 indicates no positive serial dependence.

For details of Moran's, pls refer to [here](https://en.wikipedia.org/wiki/Moran%27s_I){:target="_blank"}.

Notably, when studying environmental phenomena,
> a linear model of some covariates will not explain all the observed spatio-temporal variability. Thus, fitting such a model will frequently result in residuals that are spatially and temporally correlated. This is not surprising, since several environmental processes are certainly more complex than could be described by simple geographical and temporal trend terms. 

For the dependence in the errors, we use generalized least squares (GLS) to caculate the spatial-temporal covriance matrix, assuming that the vector of errors has the multivariate normal distribution.
We will learn this in the next post.

### Parameter inference
In this section, the authors compared the OLS and GLS parameter estimates. All the standard errors are larger in GLS than in GLS. 

Notably, parameter inference can be misleading in the presence of unmodeled extra variation, dependent errors, multicollinearity, and confounding. When an important variable is ignored, or perhaps when an extraneous variable is included in the model. Since 

The authors then introduced variable selection in R. The following piece of code is used in R for forward variable selection.

```{R}
Tmax_July_lm4 <- list()   # initialize
for(i in 0:4) {           # for four steps (after intercept model)
   ## Carry out stepwise forward selection for i steps
   Tmax_July_lm4[[i+1]] <- step(lm(z ~ 1,
                         data = dplyr::select(Tmax_no_14, -id)),
                         scope = z ~(lon + lat + day)^2 + .,
                         direction = 'forward',
                         steps = i)
}
```

> The choice of criterion can make a substantial difference when doing stepwise selection: the AIC criterion penalizes for model complexity (i.e., the number of variables in the model), whereas the RSS criterion does not.

> forward-selection can be used in the large covariates and small smaple size case.

> subset-selection and regularization can be used to balance the trade-off between variance and bias.

### Forcasting a new value
The authors introduced how to use regression to forecast the sea surface temperature (SST) in the tropical Pacific Ocean six months.

The independent variable is the Southern Oscillation Index (SOI) and the dependent variable is SST. We have monthly SOI, and we use the SOI at time $t$ to forecast the SST at time $t + \tau$ where $\tau = 6$ months. We do this for each spatial location *separately*.
```{R}
fit_one_pixel <- function(data)
                 mod <- lm(sst ~ 1 + soi, data = data)

pixel_lms <- SST_pre_May %>%
             filter(!is.na(sst)) %>%
             group_by(lon, lat) %>%
             nest() %>%
             mutate(model = map(data, fit_one_pixel)) %>%
             mutate(model_df = map(model, tidy))
```

The results showed that the forcast has very biased towards a coller anomaly than that actually observed.
This indicates that we need additional information to perform a long-lead forecast of SST. We will learn this in dynamic models in another post. The example also shows that these regression coefficients estimates show a strong spatial dependence. We will learn how to address this in the next post.

### Generalized Linear Model and Generalized Additive Model
Generalized linear models (GLM) and generalized additive models (GAM) are finally introduced to deal with binary or counts or skewed data.

The GLM has two components: (1) a systematic component that specifies the relationship between the mean response and the covariates, and (2) a random component that is assumed to be independent and come from the exponential family of distributions.

The systematic component is:

$g(Y(s;t)) = \beta_0 + \beta_1X_1(s;t) + \beta_2X_2(s;t)+\dots+\beta_pX_p(s;t)$

where $g(\cdot)$ is monotonic link function. The GLM can be extended to add random effect terms to become *generalized linear mixed model*. 

Generalized Additive Models (GAM) also have a systematic component and a random component, but the systematic component of GAM considers a more flexible function of the covariates. 

$g(Y(s;t)) = \beta_0 + f_1(X_1(s;t)) + f_2(X_2(s;t))+\dots+f_p(X_p(s;t))$

where the functions ${f_k(\cdot)}$ can be some smooth function. It is often written as a basis expansion. The GAM can also be extended to add random effect terms to become *generalized additive mixed model*. 

In the lab, the authors fit a GLM to the Carolina wren counts in the BBS data set, where we assume a Poisson response and a log link.

> This latent spatial surface captures the large-scale trends, but it is unable to reproduce the small-scale spatial and temporal fluctuations in the Carolina wren intensity, and the residuals show both temporal and spatial correlation.

> We could accommodate this additional dependence structure by adding more basis functions and treating their regression coefficients as fixed effects, but this will likely result in overfitting.

In the next post, we will learn the use of random effects to deal with this problem.

### Hierarchical Spatio-Temporal Statistical Model
Finally, the authors introduced hierarchical spatio-temporal statistical model. The hierarchical spatio-temporal model includes at least two stages.

\begin{align}
observations &= true \ process + observation \ error \tag{1} \\\\ 
true \ process &= regression \ component + dependent \ random \ process \tag{2}
\end{align}

There are two general approaches to modeling the dependent random process term in (2): the descriptive approach and the dynamic approach. In the next post, we will learn the descriptive approach, in which the dependent random process in (2) is defined in terms of the first-order and second-order moments (mean, variances, and covariances) of its marginal distribution. Acc oording to the book, this framework is not particularly concerned with the underlying mechenics; instead, it is mostly used for prediction and parameter inference.

In the post after next post, we will learn the dynamic approach. That is, to focus on conditional distributions that describe the evolution of the dependent random process in time. This method is most useful for the third goal: forecasting. 
