---
layout: post
title:  "Learning Time Series Analysis: (4) Exponential Smoother and ARIMA"
date:   2022-01-18 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

Three classical and most commonly used methods in time-series analys include Autoregressive Integrated Moving Average (ARIMA or so-called Box-Jenkins approach), Error, Trend, Seasonality Forecast (ETS), and frequency domain analysis (see a [previous post]({% post_url 2022-01-11-time-series-2%}){:target="_blank"}). ETS approach consists of `exponential smoother` and `loess`.

In this post, we learn `Exponential Smoother` and `ARIMA`. In future posts, we will learn new machine learning methods of time series analysis including Long Short Term Memory (LSTM) and Recurrent Neural Networks (RNN). 

This post is based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"} chapter 5.9 and this [engineering statistics handbook](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm){:target="_blank"}.

### ARIMA vs. ETS models
Before we introduce Exponential smoother, just have a conceptual understanding of the difference between ARIMA and ETS models:
1. ARIMA models
* some are stationary
* do not have exponential smoothing counterparts
* use if you see autocorrelation in the data, i.e. the past data explains the present data well
2. ETS models
* are not stationary
* use exponential smoothing
* use if there is a trend and/or seasonality in the data, as this model explicitly models these components

Note that some models can be written in both ARIMA and ETS forms.

### Simple Exponential Smoothing
The simplest ETS model is `simple exponential smoother`. In ETS terms, it corresponds to the (A, N, N) model. That is, the model has additive errors, no trend, and no seasonality. 

For any time t, the smoothed value $S_t$ is found by computing: 
\[
S_t = \alpha y_{t-1} +(1-\alpha)S_{t-1} \quad 0 < \alpha \le 1 \quad t \ge 3
\]

$y_t$ stands for the original observation. $S_2 = y-1$. 
\The constant $\alpha$ is called the *smoothing constant*. Why it is called "exponential"? Let's expand the basic equation by first substituting for $S_{t-1}$.

\begin{align}
S_t &= \alpha y_{t-1} +(1-\alpha)[\alpha y_{t-2} +(1-\alpha)S_{t-2}] \\\\ 
&= \alpha y_{t-1} +\alpha(1-\alpha)y_{t-2} +(1-\alpha)^2S_{t-2}
\end{align}

By substituting for $S_{t-1}$ then for $S_{t-3}$ and so forth, until we reach $S_2 = y-1$
\begin{align}
S_t &= \alpha\sum_{i=1}^{t-2}(1-\alpha)^{i-1}y_{t-i} + (1-\alpha)^{t-2}S_2, \quad t \ge 2 
\end{align}

Using a property of geometry series, we know that:
\begin{align}
S_t &= \alpha\sum_{i=0}^{t-1}(1-\alpha)^i = \alpha \left[ \frac{1 - (1-\alpha)^t}{1 - (1-\alpha)}\right] = 1 -(1 - \alpha)^t 
\end{align}

From the above formula we can see that the contribution to the smoothed value $S_t$ becomes less and less at each consecutive time period. We choose the best value for $\alpha$ so that the results have the smallest MSE.

The forcasting formula is:
\begin{align}
S_{t+1} &= \alpha y_{t} +(1-\alpha)S_{t} \quad 0 < \alpha \le 1 \quad t \ge 0 \\\\ 
S_{t+1} &= S_t + \alpha y_{t} -\alpha S_{t} \\\\ 
S_{t+1} &= S_t + \alpha \epsilon_t 
\end{align}

From the above formula, the new forcast is the old one plus an adjustment for the error that occured in the last forecast. If the new data is substantially different from the previous data (e.g., there is a trend), single exponential smoother will not be a good model. 

### Double Exponential Smoothing
When there is a trend, the model needs another constant $\gamma$.Here are the two equations of the so-called `double exponential smoothing`:
\begin{align}
S_t &= \alpha y_t +(1- \alpha)(S_{t-1} + b_{t-1}) \quad 0 \le \alpha \le 1 \tag{1} \\\\ 
b_t &= \gamma (S_t-S_{t-1}) +(1-\gamma)b_{t-1} \quad 0 \le \gamma \le 1 \tag{2}
\end{align}

Note that for doulbe smoothing, $S_1 = y_1$. Some suggestions for b1:
\begin{align}
b_1 &= y_2 -y_1 \\\\ 
b_1 &= \frac{1}{3}[(y_2-y_1) + (y_3-y_2)+(y_4 - y_3)] \\\\ 
b_1 &= \frac{y_n - y_1}{n-1}
\end{align}

> The first smoothing equation (1) adjusts $S_t$ directly for the trend of the previous period, $b_{t−1}$, by adding it to the last smoothed value, $S_{t−1}$. This helps to eliminate the lag and brings $S_t$ to the appropriate base of the current value.

> The second smoothing equation (2) then updates the trend, which is expressed as the difference between the last two values. The equation is similar to the basic form of single smoothing, but here applied to the updating of the trend.

The values for α and γ can be obtained via non-linear optimization techniques, such as the Marquardt Algorithm. 

The m-period-ahead forecast is:
\[
F_{t+m} =S_t +mb_t
\]

### Triple Exponential Smoothing (Holt-Winters Method)
If the data shows both trend and seasonality, the above double exponential smoothing will not be a good fit. Instead, we use Triple Exponential Smoothing (Holt-Winters Method). The basic equations are:

\begin{align}
S_t &= \alpha \frac{y_t}{I_{t-L}} + (1 - \alpha)(S_{t-1}+b_{t-1})\tag{3} \\\\ 
b_t &= \gamma(S_t -S_{t-1}) +(1-\gamma)b_{t-1}\tag{4} \\\\ 
I_t &= \beta\frac{y_t}{S_t} +(1-\beta)I_{t-L}\tag{5}
\end{align}

The above formula (3) is the overall smoothing, and formula (4) is the trend smoothing, and formula (5) is seasonal smoothing. $b$ is the trend factir, $I$ is the seasonal index. We need at least one complete season's data to determine initial estimates of the seasonal indices $I_{t-L}$. We need to estimate the trend factor from one period to the next, so if a complete season's data consists of L periods, it is advisable to use two complete seasons; that is, 2L periods. This is somewhat similiar to the `Nyquist Frequency`.

Initial values for the trend factor:

\begin{align}
b = \frac{1}{L}\left(\frac{y_{L+1}-y_1}{L}+ \frac{y_{L+2}-y_2}{L} +\frac{y_{L+3}-y_3}{L} +\dots + \frac{y_{L+L}-y_L}{L} \right)
\end{align}

Initial values for the seasonal indices, refer to [here](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm){:target="_blank"}. Sometimes it happends that the final coefficients for trend $\gamma$ and seasonality $\beta$ are zero. It means that the initial values were incorrect. 
This procedure can be made fully automatic by user-friendly software.
The forcast formula is:
\[
F_{t+m} = (S_t +mb_t)I_{t-L+m}
\]

Seasonality is quite common in economic time series. It is less common in engineering and scientific data. HW method is good for seasonality time series data. 

It is important to know when analyzing a time series if there is a significant seasonality effect. The seasonal subseries plot is an excellent tool for determining if there is a seasonal pattern.

The seasonal subseries plot can provide answers to the following questions:
* Do the data exhibit a seasonal pattern?
* What is the nature of the seasonality?
* Is there a within-group pattern (e.g., do January and July exhibit similar patterns)?
* Are there any outliers once seasonality has been accounted for?

I found [this page](https://www.itl.nist.gov/div898/handbook/eda/section3/eda34.htm){:target="_blank"} about graphical techniques grouped by problem category is very clear and informative.

### ARIMA
Box and Jenkins popularized an approach that combines the moving average(MA) and the autoregressive(AR) approaches into ARIMA.
An autoregressive model is simply a linear regression of the current value of the series against one or more prior values of the series. The value of p is called the order of the AR model.

\[
X_t = \delta + \phi_1X_{t-1}+ \phi_2X_{t-2} +\dots+ \phi_pX_{t-p} +A_t
\]

A moving average model is conceptually a linear regression of the current value of the series against the white noise of one or more prior values of the series.

\[
X_t =\mu +A_t -\theta_1A_{t-1}-\theta_2A_{t-2}-\dots-\theta_qA_{t-q}
\]

The Box_jenkins ARMA model is a combination of the AR and MA models:
\begin{align}
X_t =& \delta + \phi_1X_{t-1}+ \phi_2X_{t-2} + \dots+ \phi_pX_{t-p} + \\\\ 
&A_t -\theta_1A_{t-1}-\theta_2A_{t-2}-\dots-\theta_qA_{t-q}
\end{align}

Some notes on ARMA model:
* Box-Jenkins models can be extended to include seasonal autoregressive and seasonal moving average terms.
* The model assumes that the time series is stationary. Differencing non-stationary series one or more times to achieve stationarity produces an ARIMA model.
* Some formulations transform the series by subtracting the mean of the series from each data point. 
* ETS decomposition methods are recommended if the trend and seasonal components are dominant.
* Effective fitting of Box-Jenkins models requires at least a moderately long series. Many would recommend at least 100 observations.

### ARIMA Model Identification
This [engineering statistics handbook](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm){:target="_blank"} listed the general steps of ARIMA model Identification:

1) Detecting stationarity. It can be detected from an autocorrelation plot. Non-stationarity is often indicated by an autocorrelation plot with very slow decay.

2) Differencing to achieve stationarity. The differencing approach can achieve stationarity. Fitting a curve and subtracting the fitted values from the original data can also achieve stationarity.

3) Detecting seasonality. Seasonality can usually be assessed from an autocorrelation plot, a seasonal subseries plot, or a spectral plot.

4) Seasonal differencing. If seasonality exists, identify the order for the seasonal autoregressive and seasonal moving average terms. For many series, the period is known and a single seasonality term is sufficient. For example, for monthly data we would typically include either a seasonal AR 12 term or a seasonal MA 12 term. For Box-Jenkins models, we do not explicitly remove seasonality before fitting the model. Instead, we include the order of the seasonal terms in the model specification to the ARIMA estimation software. However, it may be helpful to apply a seasonal difference to the data and regenerate the autocorrelation and partial autocorrelation plots. This may help in the model idenfitication of the non-seasonal component of the model p, q.

5) Identify p and q. Once stationarity and seasonality have been addressed, the next step is to identify the order (i.e., the p and q) of the autoregressive and moving average terms. The primary tools for doing this are the autocorrelation plot and the partial autocorrelation plot. The sample autocorrelation plot and the sample partial autocorrelation plot are compared to the theoretical behavior of these plots when the order is known.

6) Order of Autoregressive Process (p). For an AR(1) process, the sample autocorrelation function should have an exponentially decreasing appearance. However, higher-order AR processes are often a mixture of exponentially decreasing and damped sinusoidal components. For higher-order autoregressive processes, the sample autocorrelation needs to be supplemented with a partial autocorrelation plot. The **partial autocorrelation** of an AR(p) process becomes zero at lag p+1 and greater, so we examine the sample partial autocorrelation function to see if there is evidence of a departure from zero.

7) Order of Moving Average Process (q). The autocorrelation function of a MA(q) process becomes zero at lag q+1 and greater, so we examine the sample autocorrelation function to see where it essentially becomes zero. The sample partial autocorrelation function is generally not helpful for identifying the order of the moving average process.

### Identifying the order of differencing
This section is based on this [page](https://people.duke.edu/~rnau/411arim2.htm){:target="_blank"}. I simply make notes to myself in this section.

The first (and most important) step in fitting an ARIMA model is the determination of the order of differencing needed to stationarize the series. Normally, the correct amount of differencing is the lowest order of differencing that yields a time series which fluctuates around a well-defined mean value and whose autocorrelation function (ACF) plot decays fairly rapidly to zero, either from above or below.

* Rule 1: If the series has positive autocorrelations out to a high number of lags, then it probably needs a higher order of differencing.

* Rule 2: If the lag-1 autocorrelation is zero or negative, or the autocorrelations are all small and patternless, then the series does not need a higher order of  differencing. If the lag-1 autocorrelation is -0.5 or more negative, the series may be overdifferenced.  BEWARE OF OVERDIFFERENCING!!

*  Rule 3: The optimal order of differencing is often the order of differencing at which the standard deviation is lowest.

* Rule 4: A model with no orders of differencing assumes that the original series is stationary (mean-reverting). A model with one order of differencing assumes that the original series has a constant average trend. A model with two orders of total differencing assumes that the original series has a time-varying trend.

* Rule 5: A model with no orders of differencing normally includes a constant term (which allows for a non-zero mean value). A model with two orders of total differencing normally does not include a constant term. In a model with one order of total differencing, a constant term should be included if the series has a non-zero average trend.

* Rule 6: If the PACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is positive--i.e., if the series appears slightly "underdifferenced"--then consider adding an AR term to the model. The lag at which the PACF cuts off is the indicated number of AR terms.

* Rule 7: If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series appears slightly "overdifferenced"--then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms.

* Rule 8: It is possible for an AR term and an MA term to cancel each other's effects, so if a mixed AR-MA model seems to fit the data, also try a model with one fewer AR term and one fewer MA term--particularly if the parameter estimates in the original model require more than 10 iterations to converge.

* Rule 9: If there is a unit root in the AR part of the model--i.e., if the sum of the AR coefficients is almost exactly 1--you should reduce the number of AR terms by one and increase the order of differencing by one.

* Rule 10: If there is a unit root in the MA part of the model--i.e., if the sum of the MA coefficients is almost exactly 1--you should reduce the number of MA terms by one and reduce the order of differencing by one.

* Rule 11: If the long-term forecasts appear erratic or unstable, there may be a unit root in the AR or MA coefficients.

* Rule 12: If the series has a strong and consistent seasonal pattern, then you should use an order of seasonal differencing--but never use more than one order of seasonal differencing or more than 2 orders of total differencing (seasonal+nonseasonal).

* Rule 13: If the autocorrelation at the seasonal period is positive, consider adding an SAR term to the model. If the autocorrelation at the seasonal period is negative, consider adding an SMA term to the model. Try to avoid mixing SAR and SMA terms in the same model, and avoid using more than one of either kind. Probably the most commonly used seasonal ARIMA model is the (0,1,1)x(0,1,1) model--i.e., an MA(1)xSMA(1) model with both a seasonal and a non-seasonal difference. This is essentially a "seasonal exponential smoothing" model.
