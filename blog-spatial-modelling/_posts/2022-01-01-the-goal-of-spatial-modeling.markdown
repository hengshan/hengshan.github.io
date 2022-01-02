---
layout: post
title:  "Three Goals of Building Spatial-Temporal Statistics Models"
date:   2022-01-01 12:30:32 +0800
category: Spatial Modeling
---

[Spatio-Temporal Statistics with R](https://spacetimewithr.org/Spatio-Temporal%20Statistics%20with%20R.pdf){:target="_blank"} states that there are three main goals that one might pursue with a spatio-temporal statistical model:
1. prediction in space and time (filtering and smoothing)
2. inference on parameters
3. forecasting in time

The second goal, inference on parameters, is about statistical inference, which is to use sample data to make inference about the parameters of population. To understand spatial modeling, it is important to have a basic understanding of probability and statistics such as population and sample, sampling, parameter estimation, and hypothesis testing. See [here](https://byjus.com/maths/statistical-inference/){:target="_blank"}.

The first and third goals are about predicting and forecasting. To fully understand filtering, smoothing, and forecasting in spatial-temporal modeling, it is important to understand what state-space model is.
[Spatio-Temporal Statistics with R](https://spacetimewithr.org/Spatio-Temporal%20Statistics%20with%20R.pdf){:target="_blank"} introduced three situations of interest when considering `state-space models`:
1. **Smoothing** refers to inference on the hidden state process during a fixed time period in which we have observations throughout the time period. 
2. **Filtering** refers to inference on the hidden state value at the most current time based on the current and all past data. The most famous example of filtering is the Kalman filter (Kalman, 1960).
3. **Forecasting** refers to inference on the hidden state value at any time point beyond the current time, where data are either not available or not considered in the forecast.

The above statement was a bit hard for me to understand before I learned state-space model. 

### What is state-space model?
When I searched 'state-space model' online, this term is always associated with either time series analysis or control system. It is not surprising as both time series analysis and control system deal with certain functions of time, and state space model is essentially a set of differential equations of functions of time.

What is a function of time, f(*t*)? In physics, finance, and many enginnering fields, a lot of phenomona have been mathematically modeled as functions of time. The simplest example is stock price, which fluctuates and changes its value at different time. Another classic example is when describing the position of a object varies with time under the influence of constant acceleration.

Note that 'space' in state-space model stands for the mathematical metric space of state variables. Our main topic of this blog site, spatial-temporal models, however, mainly model physical space and time. See [here](https://en.wikipedia.org/wiki/Space_(mathematics)){:target="_blank"} for the introduction of space (mathematics) in wikipedia. 
