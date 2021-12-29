---
layout: post
title:  "What is Spatial Optimization?"
date:   2021-12-27 12:41:32 +0800
category: Optimization
---

Two pillars of Data Science are statistics and optimization. Correspondingly, two key pillars of Spatial Data Science are statistics and optimization in a spatial context. The spatial modeling blogs will mainly focus on statistical models including linear and non-linear multilevel regression models, Bayesian statistical models, etc. The spatial optimization blogs will focus on the optimization of mathematical models with geographical or space context.
To understand spatial optimization, let's first learn what optimization is.

### What is Optimization?
Simply speaking, optimization is to find the best solution for a given problem. This is so far the easiest concept I have ever learned in data science. 
Two important but difficult questions arise here:
1. How to model the problem? See [Mathematical Modeling](https://www.math.colostate.edu/~gerhard/MATH331/331book.pdf){:target="_blank"}
2. How to solve the mathematical model? This is related to optimization. See wiki about [Mathematical Optimization](https://en.wikipedia.org/wiki/Mathematical_optimization){:target="_blank"}

The problem can be written as a math formula, which can be a very simple linear or non-linear function.The functions can be very complex such as some differential equations. The functions can be continous (e.g., f = x1 + x2).
Here, x1 and x2 stand for decision variables. We choose values for x1 and x2 so that this function f is optimized (either maximized or minimized).

The formula can be discrete such as for most transportation problems, network, and scheduling problems.
There might be certain constraints on the decision variables which have to be satisfied while solving the optimization problem.

If you are a beginner like me, you might feel overwhelmed when you hear about the terms such as single objective and multi-objective optimization, linear programming, mixed-integer linear programming, nonlinear programming, mixed-integer nonlinear programming.

It is so scary that many beginners may stop learning optimization. I did not find any free online videos that systemcatically introduce optimization that is easy to understand. Well, the topic itself is not easy, as it requires solid math foundation.
If you are not good at math like me, it is better to learn optimization from relatively easy-to-understand optimization problems including:
1. The knapsack problem
2. The bin packing problem
3. The container loading problem
4. The assignment problem
5. The scheduling problem
6. The travelling sales men problem
7. The capacitated vehicle routing problem

I will learn and share how to solve these problems using optimization tools. Instead of learning math and trying to solve the optimization by ourselves, we learn how to use tools or libraries which do all the heavy lifting. After that, we will have a basic understanding of what optimization can help us with, and we may become more interested in learning the math underlying the problems.

Of course, if you are interested, you can learn math in advance from the book [All the mathematics you missed](http://xn--webducation-dbb.com/wp-content/uploads/2018/02/all-the-mathematics-you-missed.pdf){:target="_blank"}.

### What is Spatial optimization?

Spatial optimization is a type of optimization that focuses on geographic decision problems. The decision variables explicitly represent spatial (such as distance, point pattern, partition) or topological (e.g.,connectivity, overlap, containment,etc.) phenomena.
These decision variables are so called geographic decision variables.

My understanding of spatial optimization is that the most challenging part is the mathematical modeling of spatial or topological phenomena. With regards to solving the problem, spatial optimization faces exactly the same mathematical challenge and uses the same method as non-spatial optimization such as using exact or heuristic approaches.
Both network partitioning and travelling sales men problems are somewhat spatial, but I would not define them as spatial optimization, as they are very general optimization problems.  

A typical spatial optimization problem is redistricting problem: dividing space into districts or zones while optimizing a set of spatial criteria under certain constraints.
It is essentially a combinatorial optimization problem which has a wide application on government and business operations. 

I have to admit that there have been way too many terms. Most are necessary, while some are a bit confusing such as spatial optimization, as some may misunderstand that the optimization method instead of the to-be-modeled problem is spatial.  
I feel that it is not easy to find a good name. I simply understand spatial optimization as:
> to find the best solution of a spatial or topological problem.
