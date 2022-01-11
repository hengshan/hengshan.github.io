---
layout: post
title:  "Learning Time Series Analysis: (1) The Structure of Temporal Data"
date:   2022-01-09 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

In the [previous post]({% post_url 2022-01-06-two-approaches-spatial-modeling%}), we discussed that understanding time series analysis is a prerequisite for learning spatio-temporal modeling.
This series of posts are based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"}.

### The Structure of Time Series Data
Time series data are:
> Observations or measurements that are indexed according to time

There is an ordering to the time series data, so it is generally not independent. Imagine a simple non-temporal data such as heights $X_{i}$ and weights $Y_{i}$ of all students in a class. We can randomly permute the indices of the data.
However, for time series data such as height and weight of a student $X_{t}$ and $Y_{t}$ from 1 to 20 years old, we cannot change the order of the data. 

#### Time Scales
Depending on the temporal granularity of observations, time series data can be aggregated at different time scales such as yearly, monthly, daily, hourly, or even microseconds for stocks. The [time series book](https://bookdown.org/rdpeng/timeseriesbook/example-air-pollution-and-health.html){:target="_blank"} gave an example about studying the effect of PM10 on deaths at a yearly or a daily basis, and stated that:
> The daily average looks at short-term changes and could prehaps be interpreted as representing “acute” effects of pollution, while the yearly average might represet “chronic” effects of air pollution levels. 

#### Fixed vs. Random Variation

The [time series book](https://bookdown.org/rdpeng/timeseriesbook/example-air-pollution-and-health.html){:target="_blank"} states
> that many real time series in the world are composed of what we might think of as fixed and random variation,, but many time series books tend to image a time series as consising only of random phenomena. For example,

![Baltimore Temperature](https://bookdown.org/rdpeng/timeseriesbook/index_files/figure-html/unnamed-chunk-7-1.png)

One model is to model daily Baltimore temperature as a function flucuating around the mean temperature of all time span $\mu$.
\[
y_t = \mu +\epsilon_t\tag{1}
\]

Another way is to model daily Baltimore temperature as a funciton of the previous data temperature.
\[
y_t = y_{t-1}+\epsilon_t\tag{2}
\]

The model (1) uses fixed variation, and the model (2) uses random variation, as the former assumes that all errors flucuate around the mean, while the latter assumes the errors flucuate around the previous day temperature.  

Note that: fixed vs. random variation are different from fixed and random effects in mulitlevel modeling.
If the readers have the background of pyschology or experimental design, time series data can be easily understood as "repeated measures design" or "within-subject design". It is sometimes called panel data. The data is collected through a series of repeated observations of the same subjects over some extended time frame.

A common approach to model panel data is to use linear mixed-effect model. In matrix notation:
\[
y = X\beta + Zu + \epsilon
\]
where $y$ is the $n \times 1$ vector of reponses, $X$ is an $n \times p$ design/covariate matrix for the fiexed effects $\beta$, and $Z$ is the $n \times q$ design/covariate matrix for the random effects $u$. The $n \times 1$ vector of errors $\epsilon$ is assumed to be the multivariate normal distribution with mean 0 and variation matrix $\sigma_{\epsilon}^2R$.


