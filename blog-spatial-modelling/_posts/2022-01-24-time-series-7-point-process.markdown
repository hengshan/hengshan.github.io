---
layout: post-wide
title:  "Time Series Analysis: (7) Point Process Analysis"
date:   2022-01-24 12:30:32 +0800
category: Spatial Modeling
author: Hank Li
use_math: true
---

This post is based on the book [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/){:target="_blank"} chapter 7. It's great that this timeseriesbook covers `point process analysis`, which is an important topic in both spatial and temperal analysis (and spatial-temperal analysis).

In previous posts, we have learned some discrete-time continous-event models such as ARIMA, Exponential Smoothing, and State Space Models (note that state space model can also model continous time). These models mainly deal with discrete time $x_1,x_2,\dots, x_t$ and continous events such as temperature, blood pressure, etc. We also learned Gaussian Processes, which can model continous-time models and continous events.

In this post, we will learn Poisson Point Process, which models continous-time and discrete event, specifically, binary outcome events (i.e., something happens or not happen) such as the occurance of earthquakes. 

Before we dive into Possion Point Process, let's learn Poisson distribution. 
### Poisson distribution
A discrete random variable X is said to have a Poisson distribution, with parameter $\lambda >0$, if it has a probability mass funcion given by:
\begin{align}
f(k;\lambda) = Pr(X=k) = \frac{\lambda^ke^{-\lambda}}{k!}
\end{align}

Notably, the probability mass function is a function of k, which is the number of occurances ($k=0,1,2,\dots$). $\lambda$ is kind of like the shape parameter but I usually understand it as intensity or rate. The Poisson distribution is the limit of a binomial distribution, for which the probability of success for each trial equals λ divided by the number of trials, as the number of trials approaches infinity.

The following materials are based on wikipedia. Assumptions of Poisson distribution include:
1. The occurrence of one event does not affect the probability that a second event will occur.
2. The average rate at which events occur is independent of any occurrences.
3. Two events cannot occur at exactly the same instant.
the Poisson distribution may be useful to model events such as

Examples of Poisson distribution:
> * The number of meteorites greater than 1 meter diameter that strike Earth in a year
* The number of patients arriving in an emergency room between 10 and 11 pm
* The number of laser photons hitting a detector in a particular time interval

Examples of violation of the Possion distribution assumptions. 
1. The number of students who arrive at the student union per minute will likely not follow a Poisson distribution, because the rate is not constant (low rate during class time, high rate between class times) and the arrivals of individual students are not independent (students tend to come in groups).

2. The number of magnitude 5 earthquakes per year in a country may not follow a Poisson distribution if one large earthquake increases the probability of aftershocks of similar magnitude.

3. Examples in which at least one event is guaranteed are not Poisson distributed; but may be modeled using a zero-truncated Poisson distribution.

### Poisson Processes
The Poisson distribution arises as the number of points of a Poisson point process.
> a Poisson point process is a type of random mathematical object that consists of points randomly located on a mathematical space.

If a Poisson point process has a parameter of the form $\Lambda =\nu \lambda$, where $\nu$ is Lebesgue measure (that is, it assigns length, area, or volume to sets) and $\lambda$ is a constant, then the point process is called a homogeneous or stationary Poisson point process. The parameter $\Lambda$ called rate or intensity, is related to the expected (or average) number of Poisson points existing in some bounded region where rate is usually used when the underlying space has one dimension.

The parameter $\lambda$ can be interpreted as the average number of points per some unit of extent such as length, area, volume, or time, depending on the underlying mathematical space, and it is also called the `mean density`.

\begin{align}
P(N(\nu)=n) = \frac{(\lambda\lvert\nu\rvert)^ne^{-\lambda\lvert\nu\rvert}}{n!}
\end{align}

Based on the above equation, we understand that if a collection of random points in some space forms a Poisson process, then the number of points in a region of finite size is a random variable with a Poisson distribution. In the context of time series analysis, we use time t and it becomes:

\begin{align}
P(N(t)=n) = \frac{(\lambda\lvert t\rvert)^ne^{-\lambda\lvert t\rvert}}{n!}
\end{align}

The above can be further written as

\begin{align}
P\\{N(a,b]=n\\} = \frac{(\lambda(b-a))^n}{n!}e^{-\lambda(b-a)}
\end{align}

The homogeneous point process is sometimes called the `uniform Poisson point process`. The positions of these occurrences or events on the real line (often interpreted as time) will be uniformly distributed. 

The timeseriesbook introduced the strong and weak versions of stationarity in the point process. For the weaker version of stationarity says simply that for every h,

\begin{align}
P(N(t,t+h] = k)) = g(h)
\end{align}

In other words, the probability density for the number of points in an interval of size h only depends on h.

### Inhomogeneous Poisson Point Process
Inhomogeneous Poisson Point Process is a Poisson point process with a Poisson parameter set as some *location-dependent* function in the underlying space. For Euclidean space $\mathbb{R}^d$, this is achieved by introducing a locally integrable positive function $\lambda: \mathbb{R}^d \to [0,\infty)$, such that for any bounded region B the volume integral of $\lambda(x)$ over the region is finit.

\begin{align}
\Lambda(B) = \int \lambda(x)dx \lt \infty
\end{align}

A counting process is said to be an inhomogeneous Poisson counting process if it has the four properties:
1. N(0) =0
2. has independent increments
3. $P(N(t+h)-N(t)=1) = \lambda(t)h + o(h)$
4. $P(N(t+h)-N(t)\ge 2) = o(h)$

### Connection to Time Series Analysis
According to the timeseriesbook:
> Depending on how frequent the events are, it may be more reasonable to bin the timeline into equally spaced bins and treat the number of events in each bin as a count time series. If the events are very rare, this may end up being a binary time series. However, in that case it may make sense to simply model the data as a point process anyway.

> One disadvantage to modeling point process data as count time series is that within each bin, you lose the time ordering aspect of the data because past an future mix together within the bin. Therefore, you cannot make predictions of the “next event” using such models because you cannot build a model that is conditional on only the history of the process.

### Conditional Intensities
Given the history of a point process $H_t$ up until but not including t, the conditional intensity of a point process is defined as

\begin{align}
\lambda(t|H_t) = \lim_{h\to 0}\frac{E[N(t,t+h)|H_t]}{h}
\end{align}

For a stationary Poisson process, the conditional intensity function is a constant, i.e.  
\begin{align}
\lambda(t|H_t) = \lambda
\end{align}

If $\lambda(t\|H_t) = \lambda(t)$, so that the conditional intensity does not depend on the history $H_t$ but depende on t, the process is [`inhomogeneous Poisson process`](#inhomogeneous-poisson-point-process).

### Conditional Intensity Models
A simple non-stationary Poisson process could be modeled by a conditional intensity function such as

\begin{align}
\lambda(t) = \alpha + \beta t
\end{align}

so that there is a trend over time governed by the parameter  
β. This model does not depend on any history, but if $\beta >0$ then as time increases, the intensity of events increases. One way to ensure positivity is to sepcify a log-linear model.

\begin{align}
\mathrm{log}\lambda(t) = \alpha + \beta t
\end{align}

Now the components of the model mulitply each other in that
\begin{align}
\lambda(t) = e^{\alpha}\times e^{\beta t}
\end{align}

### Cluster Models
Cluster models typicaly describe processes where the occurrence of an event increases the probability of an event occurring nearby in time.
For example, a main shock earthquake increases the likelihood of an aftershock occurring soon afterwards.

A more formal way to state this is that for two non-overlapping intervals $(a, b)$ and $(b, c)$, we have $\mathrm{Cov}(N(a,b),N(b,c))>0$, which is in contrast to the assumptions of a Poisson process. 

A general class of cluster models is described by the `Hawkes process`. The conditional intensity form is:

\begin{align}
\lambda(t|H_t) = \mu +\int_0^t g(t-u)N(du) = \mu +\sum_{t_i\lt t}g(t-t_i)
\end{align}

we might want to structure the function g such that points in the distant past have less influence than more recent points. Such a function might look like

\begin{align}
g(s) = \sum_{k=1}^K\phi_ks^{k-1}e^{-\alpha s}
\end{align}

Here, K is pre-specified. As s grows large, the exponential term dominates and eventually tapers to zero. g is sometimes called `trigger density`. According to the timeseriesbook,

> In earthquake settings, the parameter μ is thought of as the “background rate” or the rate at which main shock earthquakes occur. Then the trigger density indicates the increased rate over the background rate at which aftershocks occur.

> In infectious disease settings, μ might indicate the background rate of infection and then the trigger density could describe the rate at which other individuals subsequently become infected.

Another option for the trigger density in a Hawkes-type cluster process model is:

\begin{align}
g(s) = \frac{\kappa}{(s + \phi)^\theta}
\end{align}

> The key difference with this trigger density and the previous one is the power-law decay in the clustering rate, as opposed to an exponential decay in the model above.

### Self-Correcting Models
Self-correcting models adjust their conditional intensity each time an event occurs. For two non-overlapping intervals $(a, b)$ and $(b, c)$, we have $\mathrm{Cov}(N(a,b),N(b,c))<0$. 

\begin{align}
\lambda(t|H_t) = \alpha +\beta t -\nu N[0,t)
\end{align}

### Estimation
Given a dataset of event times $t_1, t_2, \dots, t_n$ observed on the interval $[0, T]$, the log-likelihood for a model is:

\begin{align}
\ell(\theta) =\sum_{i=1}^{n}\mathrm{log}\lambda(t_i|H_t;\theta) - \int_0^T\lambda(t|H_t;\theta)dt
\end{align}

The log-likelihood has two parts:
1. The first sum “rewards” a model for having high intensity where event times are located.

2. The integral part, because it is subtracted off, rewards a model for having low intensity where the event times are not located.

The log-likelihood will be a nonlinear function and need to be maximized using standard nonlinear optimization routines.

Point process is an important topic in spatia analysis. Although the spatial-temperal book does not cover the materials of spatial point process, we will learn it from other materials in future.
