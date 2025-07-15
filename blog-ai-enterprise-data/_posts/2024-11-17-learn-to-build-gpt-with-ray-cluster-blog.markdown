---
layout: post-wide
title:  "Building GPT from Scratch with Ray"
date:  2024-11-17 22:41:32 +0800
category: AI 
author: Hank Li
---
In my [previous blog]({% post_url 2024-10-17-learn-to-build-gpt-blog %}), I walked through how to train a simple Bigram language model locally using PyTorch based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" tutorial ([code](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=hoelkOrFY8bN), [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3483s)).

That was a fun and insightful exercise—but it only scratched the surface of modern AI. Real-world machine learning and AI today are not just about clever models or algorithms. They are about infrastructure, scalability, and efficient computation.

In this post, I want to take things a step further. We’ll explore how to build and train GPT-like models from scratch, but this time with an eye toward distributed computing and production readiness using Ray, Metaflow, Docker, and Kubernetes.

