---
layout: post
title:  "Learning Spatial-Temporal Modeling (1) Exploratory Analysis"
date:   2022-01-28 12:30:32 +0800
category: Spatial Modeling
use_math: true
---

This post is based on the book [Spatio-Temporal Data in R](https://spacetimewithr.org/){:target="_blank"} chapter 2.

In previous posts, we have learned basic time series analysis mostly for univariate sequential data observed at regular intervals. We also learned the basics of point process in which the temperal event itself is the random event. In this post, we will learn exploratory analysis of spatial-temperal data, which includes both temperal and spatial data analysis. 

### Two Perspectives of Viewing Spatial-Temporal Data
Data from spatio-temporal processes can be considered
from either a time-series perspective or a spatial-random-process perspective:

1. Time-series perspective: univariate or multivariate sequential data; data observed at regular or irregular intervals; Poisson point process (temporal event itself is random event).

2. Spatial data analysis perspective: geostatistical, areal or lattice, or point process.
* *geostatistical data*: we could have observations of the variable at continuous locations over a given spatial domain (e.g., temperature and wind speed). For geostatistical data, one objective is to predict the variable at unknown locations in space.
* *Lattice data*: the data is defined on a finite or countable subset in space such as the population data defined on a specific political geography (e.g., subzones in Singapore) over a specific period of time.
* A *spatial point process* is a stochastic process in which the locations of the points (sometimes called events) are random over the spatial domain. 

The st_data_in_r book primarily focuses on spatio-temporal data that are discrete in time and either geostatistical or on a lattice in space. In other words, spatial point process is not covered in this book. However, we will learn it from other materials.

### Datasets
I like this book as it teaches the model with very informative and representative datasets with R code. Six datasets include:
1. NOAA daily weather data. These data are discrete and regular in time (daily) and geostatistical and irregular in space. There are missing measurements at various stations and at various time points.
2. Sea-surface temperature anomalies. The data are gridded at a $2^\circ$ by $2^\circ$ resolution, so it is geostatistical and regular in space. It is montly data from January 1970 to December 2003.
3. Breeding Bird Survey counts. The counts of house finches from 1967 to 2014 data is discrete in time, geostatistical irregular in space, and non-Gaussian in the sense that they are counts. 
4. Per capita personal income (from US BEA). These data have areal spatial support corresponding to USA counties in the state of Missouri, and they cover the period 1969–2014.
5. Sydney radar reflectivity. These data are a subset of consecutive weather radar reflectivity images. There are 12 images at 10-minute intervals starting at 08:25 UTC on 03 November, 2000. The data used in this book are for a region of dimension 28 × 40, corresponding to a 70km by 100km domain. All reflectivities are given in “decibels relative to Z” (dBZ, a dimensionless logarithmic unit used for weather radar reflec- tivities).
6. Mediterranean winds. Wind observations over a region in the Mediterranean for 28 time periods (every 6 hours for 4 days). There are two data sources: satellite wind observations and surface winds and pressures. The satellite images are only available intermittently in space but at much higher spatial resolution (25 km) than surface winds and pressures when available. Winds and pressures are given on a $0.5^\circ \times 0.5^\circ$ spatial grid (corresponding to 47 longitude locations and 25 latitude locations), and they are available at each time period for all locations.

It is important to understand the characteristics of these datasets, as in future if we have a dataset and do not know how to analyze the data, we can find a similiar dataset here and see how the author analyzed that specific type spatial-temporal data. 

### Representation of Spatial-temporal Data in R
The st_data_in_r book uses R package **spacetime** to represent spatail-temporal data. It is worth noting that in R spatial data can be represented using **sf** or **sp**. The spacetime package extents sp and xts (time series data). Three classes of spacetime:

* time-wide, where columns corresponds to different time points
* space-wide, where where columns correspond to different spatial features (e.g.,locations, regions, grid points, pixels)
* long formats, where each record corresponds to a specific time and space coordinate.

The author gave the tip that:
> it is easy to subset and manipulate data in long format. **dplyr** and **tidyr**, and visualization tools in **ggplot2**, are designed for data in long format.

The spacetime package extents table object to contain additional map projection and time zone information. Four classes of space-time data:
* *full grid* (STF), a combination of any sp object and any xts object to represent all possible locations on the implied space-time lattice.
* *sparse grid* (STS),a combination of any sp object and any xts object to represent all **non-missing** space-time combination on the implied space-time lattice. 
* *irregular* (STI), each point is allocated a spatail coordinate and a time stamp.
* *simple trajectories* (STT), a sequence of space-time points that form trajectories. 

### Exploratory Analysis of Spatio-Temporal Data Visualization

The spatio-temporal book introduced the following exploratory analysis and visualization methods.
1. empirical means and covariances
2. spatio-temporal covariograms and semivariograms
3. emprirical orthogonal functions and principal-component time series
4. spatio-temporal canonical correlation analysis

#### Empirical Means and Covariances 
For observations  $\\{Z(s_i;t_j)\\}$, we can aggregate the data over location $\\{s_i: i = 1, \dots, m\\}$ or over time $\\{t_j: j = 1, \dots, T\\}$.

The formula for average across time is:

$\hat\mu_{z,s} \equiv \begin{bmatrix} \hat\mu_{z,s}(s_1) \\\\ \vdots \\\\ \hat\mu_{z,s}(s_m) \\\\ \end{bmatrix} = \begin{bmatrix} \frac{1}{T}\sum_{j=1}^TZ(s_1;t_j) \\\\ \vdots \\\\ \frac{1}{T}\sum_{j=1}^TZ(s_m;t_j) \\\\ \end{bmatrix} = \frac{1}{T}\sum_{j=1}^T\mathbf{Z}_{t_j}$

Note that here z stands for dependent variable name Z, and s stands for spatial locations. Similarly, we can average across space. The time series is:

$\hat\mu_{z,t}(t_j) \equiv {1 \over m} \sum_{i=1}^mZ(s_i;t_j)$

The $m \times m$ lag-$\tau$ empirical spatial covariance matrix is calculated by:

\begin{align}
\hat C_z^{(\tau)} =\frac{1}{T-\tau}\sum_{j=\tau +1}^T(Z_{t_j}-\hat\mu_{z,s})(Z_{t_j-\tau}-\hat\mu_{z,s})';\quad \tau = 0, 1, \dots, T-1
 \end{align}

Note that the the lat-$\tau$ is the average of all lags from 0 to $T -1$. The author also mentioned that it can be difficult to obtain any intuition from these matrices, as locations in a 2D space do not have a natural ordering as 1D time series data.

### Spatio-Temporal Covariograms
Some may wonder why we need covariograms, given that we've got empirical covariance matrix. The simple answer is that we need to predict covoriance based on distance and time. The objective of covariograms is to re-write the corariability of the data as a function of lags in time and space.

In order to have less parameters for the model, the assumption of covariance function is:
> the mean depends on space but not on time and the covariance depends only on the lags of space and time

The covariance function is a function that can generate the covariance matrix. We have discussed it when we learned GP.
Then the empirical spatio-temporal covariogram for spatial lag h and time lag $\tau$ is:

$\hat C_z(\mathbf{h};\tau) = \frac{1}{\left\vert N_s(\mathbf{h})\right\vert}\frac{1}{\left\vert N_t(\tau)\right\vert}\sum_{s_i,s_k \in N_s(\mathbf{h})}\sum_{t_j,t_l \in N_t(\tau)}(Z(s_i;t_j) - \hat\mu_{z,s}(s_i))(Z(s_k;t_l)-\hat\mu_{z,s}(s_k))$

For a given distance h and time lag $\tau$, we will find pairs of spatial locations with the spatial lag. For example, there are two points within h, then there is only one pair. That is, ${\left\vert N_s(\mathbf{h})\right\vert} = 1$
To calculate the covariance of the two locations, we have to first get the time lag $\tau$. For example, if $\tau =1$, we calculate the pairs of time points of the two locations. 
This depends how the length of time series at the two locations.

Note that, the time series lengths might be different at the two locations. Anyway, we need to calculate the number of pairs of time points ${\left\vert N_t(\tau)\right\vert}$ between the two locations within the lag $\tau$. Next, we sum up all the pairs and average by the number of pairs of locations and time lag. 


### Spatio-Temporal Semivariograms
The Semivariogram is defined as:

$\gamma_z(s_i,s_k;t_j,t_l) \equiv {1 \over 2} \text{var}(Z(s_i;t_j)-Z(s_k;t_l))$

When the covariance depends only on displacements in space and time lag, this can be written as:

\begin{align}
\gamma_z(\mathbf{h};\tau) &= {1 \over 2}\text{var}(Z(s + \mathbf{h}; t + \tau)-Z(s;t)) \\\\ 
&= {1 \over 2}\mathbb{E}[(Z(s + \mathbf{h}; t + \tau) - \mu_{s+h}) -(Z(s;t)-\mu_{s})]^2 \\\\ 
&= {1 \over 2}\mathbb{E}[(Z(s + \mathbf{h}; t + \tau) - \mu_{s+h})^2 +(Z(s;t)-\mu_{s})^2 -(Z(s + \mathbf{h}; t + \tau) - \mu_{s+h})(Z(s;t)-\mu_{s})] \\\\ 
\end{align}

This seems to become too complex, but if we assume a constant spatial mean $\mu$, the above can be written as:
\begin{align}
\gamma_z(\mathbf{h};\tau) &= {1 \over 2}\text{var}(Z(s + \mathbf{h}; t + \tau)-Z(s;t)) \\\\ 
&= {1 \over 2}\mathbb{E}[(Z(s + \mathbf{h}; t + \tau) - \mu_{s+h}) -(Z(s;t)-\mu_{s})]^2 \\\\ 
&= {1 \over 2}\mathbb{E}[Z(s + \mathbf{h}; t + \tau) -Z(s;t)]^2 \\\\ 
&= {1 \over 2}\mathbb{E}[Z(s + \mathbf{h}; t + \tau)^2  +Z(s;t)^2 -2(Z(s + \mathbf{h}; t + \tau))(Z(s;t))] \\\\ 
\end{align}

If there is stationary covariance function $C_z(\mathbf{h};\tau)$, the above can written as:
\begin{align}
\gamma_z(\mathbf{h};\tau) &= {1 \over 2}\text{var}(Z(s + \mathbf{h}; t + \tau)-Z(s;t)) \\\\ 
&= {1 \over 2}\mathbb{E}[(Z(s + \mathbf{h}; t + \tau) - \mu_{s+h}) -(Z(s;t)-\mu_{s})]^2 \\\\ 
&= {1 \over 2}\mathbb{E}[Z(s + \mathbf{h}; t + \tau) -Z(s;t)]^2 \\\\ 
&= {1 \over 2}\mathbb{E}[Z(s + \mathbf{h}; t + \tau)^2  +Z(s;t)^2 -2(Z(s + \mathbf{h}; t + \tau))(Z(s;t))] \\\\ 
&= C_z(\mathbf{0};0) - \text{cov}(Z(s +\mathbf{h};t+\tau),Z(s;t)) \\\\ 
&= C_z(\mathbf{0};0) -C_z(\mathbf{h};\tau)
\end{align}

If there is no stationary covariance function $C_z(\mathbf{h};\tau)$, we will try to fit trend terms that are linear or quadratic in spatail-temporal coordinates. This is similiar to the time series analysis we have learned. 

Based on the above formula, an alternative estimate is:

$\hat \gamma_z(\mathbf{h};\tau) = \frac{1}{\left\vert N_s(\mathbf{h})\right\vert}\frac{1}{\left\vert N_t(\tau)\right\vert}\sum_{s_i,s_k \in N_s(\mathbf{h})}\sum_{t_j,t_l \in N_t(\tau)}(Z(s_i;t_j) - Z(s_k;t_l))^2$

Well, I think semivariogram makes the basic concept unnecessarily more difficult to understand. The covariance function itself is actually easier to understand. That is, given a space displacements and time lag, we can generate a covariance matrix, based on which we can predict.

### Empirical Orthogonal Functions (EOFs)
EOFs is essentially principal component analysis (PCA) in spatail temporal data. PCA is essentially decompose the variance-covariance matrix and find the eigenvector and eigenvalues. The eigenvectors linear transform the original into new variables, and eigenvalues are the variances of the associcated linear combinations. 

For EOFs, just note that each location is a time series and it is one dimension in PCA. First, we calculate the covariance matrix with lag-0 and then decompose the matrix:

$\hat{\mathbf{C}}_z^{(0)} = \Phi\Lambda\Phi'$

Where $\Phi \equiv (\phi_1,\dots,\phi_m)$

According to the book, two primary uses for EOFs are:
1. gain some understanding about important spatial patterns of variability in a sequence of spatio-temporal data by examining the EOF coefficient maps. 
2. these bases can be quite useful for dimension reduction in a random-effects spatio-temporal representation

> But care must be taken not to interpret the EOF spatial structures in terms of dynamical or kinematic properties of the underlying process

### Spatio-Temporal Canonical Correlation Analysis
Each spatial-temporal dataset usually has multivariates (e.g., different locations and each location can be deemed as a time series variable). For a single dataset, we can calculate the correlations among multivariates. However, to calculate the "correlation" between two datasets, we have to figure out a new way. One way is to "aggregate" each dataset as one time series variable and then calculate the correlation between the two time series variables. A new question raises: how to "aggregate" the ataset.

Canonical correlation analysis (CCA) seeks to create new variables that are linear combinations of two multivariate data sets such that the correlations between these new variables are maximized.

\begin{align}
a_k(t_j) &= \sum_{i=1}^m\xi_{ik}Z(s_i;t_j) = \mathbf\xi_k'\mathbf{Z}_{t_j} \\\\  
\end{align}

\begin{align}
b_k(t_j) &= \sum_{l=1}^n\phi_{lk}X(r_l;t_j) = \mathbf\phi_k'\mathbf{X}_{t_j} \\\\  
\end{align}

The kth *canonical correlation* is then the correlation between $a_k$ and $b_k$,
\begin{align}
r_k \equiv \text{corr}(a_k,b_k) &= \frac{\text{cov}(a_k,b_k)}{\sqrt{\text{var}(a_k)}\sqrt{\text{var}(b_k)}} \\\\ 
&=\frac{\xi_k'C_{z,x}^{(0)}\phi_k}{(\xi_k'C_z^{(0)}\xi_k)^{1/2}(\phi_k'C_x^{(0)}\phi_k)^{1/2}}
\end{align}

The weights given by $\xi_k$ and $\phi_k$ are indexed in space, so they can be plotted as spatial maps. The spatial patterns in the weights show the areas in space that are most responsible for the high correlations.

The associated variables $a_k$ and $b_k$ can be plotted as time series. The author mentioned that:
> One has to be careful with the interpretation of canonical variables beyond the first pair, given the restriction that CCA time series are uncorrelated.

> Given that high canonical correlations within a canonical pair naturally result from this procedure, one has to be careful in evaluating the importance of that correlation.
