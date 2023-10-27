---
layout: post
title:  "What is Spatial Optimization?"
date:   2021-12-27 12:41:32 +0800
category: Optimization
---

Two key pillars of Spatial Data Science are statistics and optimization in a spatial context. The spatial optimization blogs will focus on the optimization of mathematical models with geographical or space context.
To understand spatial optimization, let's first learn what optimization is.

### What is Optimization?
Optimization is to find the best solution for a given problem. Two important questions arise here:
1. How to model the problem? See [Mathematical Modeling](https://www.math.colostate.edu/~gerhard/MATH331/331book.pdf){:target="_blank"}
2. How to solve the mathematical model? This is related to optimization. See wiki about [Mathematical Optimization](https://en.wikipedia.org/wiki/Mathematical_optimization){:target="_blank"}

The problem can be written as a math formula, which can be a very simple linear or non-linear function.The functions can be very complex such as some differential equations. The functions can be continous (e.g., f = x1 + x2).
Here, x1 and x2 stand for decision variables. We choose values for x1 and x2 so that this function f is optimized (either maximized or minimized).

The formula can be discrete such as for most transportation problems, network, and scheduling problems.
There might be certain constraints on the decision variables which have to be satisfied while solving the optimization problem.

For beginners, topics such as single objective and multi-objective optimization, linear programming, mixed-integer linear programming, nonlinear programming, etc. might sound a bit scary. If they start from the wrong location, they can easily get lost and end up no where. As a result, on the one hand, beginners may not be able to self-learn this subject easily; on the other hand, it is hard for a SME company to hire profesional operations researcher or experts on supply chain. What a shame it is, especially optimization is such an useful and important tool for business and operation.

Well, the truth is: although the topic of optimization is indeed complex, which requires solid math foundation for operations research, using optimization based  on fopt, using optimization based on framesworks alone doesn't requires to understand all the math behind it. For most data scientists, learn to mathematically understand the problem and model the objective function and constaints, it is sufficient for most situations. 

Therefore, I will share how to solve the most common optimiation problems using python tools, before we learn the math underlying optimization, I am sure that we will become more interested in learning optimization.

1. The knapsack problem
2. The bin packing problem
3. The container loading problem
4. The assignment problem
5. The scheduling problem
6. The travelling sales men problem
7. The capacitated vehicle routing problem


Of course, if you are interested, you can learn math in advance from the book [All the mathematics you missed](http://xn--webducation-dbb.com/wp-content/uploads/2018/02/all-the-mathematics-you-missed.pdf){:target="_blank"}.

### What is Spatial optimization?

Spatial optimization is a type of optimization that focuses on geographic decision problems. The decision variables explicitly represent spatial (such as distance, point pattern, partition) or topological (e.g.,connectivity, overlap, containment,etc.) phenomena.
These decision variables are so called geographic decision variables.

My understanding of spatial optimization is that the most challenging part is the mathematical modeling of spatial or topological phenomena. With regards to solving the problem, spatial optimization faces exactly the same mathematical challenge and uses the same method as non-spatial optimization such as using exact or heuristic approaches.
Both network partitioning and travelling sales men problems are somewhat spatial, but I would not define them as spatial optimization, as they are very general optimization problems.  

A typical spatial optimization problem is redistricting problem: dividing space into districts or zones while optimizing a set of spatial criteria under certain constraints.
It is essentially a combinatorial optimization problem which has a wide application on government and business operations. 
