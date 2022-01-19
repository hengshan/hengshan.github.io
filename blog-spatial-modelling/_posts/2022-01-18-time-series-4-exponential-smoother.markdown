---
layout: post
title:  "Learning Time Series Analysis: (4) Exponential Smoother"
date:   2022-01-18 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

Three classical methods in time-series analys are Error, Trend, Seasonality Forecast (ETS), Autoregressive Integrated Moving Average (ARIMA) and Holt-Winters (Exponential Smoother).
Two new machine learning (ML) methods include Long Short Term Memory (LSTM) and Recurrent Neural Networks (RNN). In this post, we learn Holt-Winters Exponential Smoother.
This post is based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"} chapter 5.9 and this [engineering statistics handbook](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm){:target="_blank"}.


