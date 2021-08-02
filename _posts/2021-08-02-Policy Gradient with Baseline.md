---
title: 'Reinforcement Learning-Policy Gradient with Baseline(1)'
date: 2021-08-02
permalink: /posts/2021/08/baseline1/
tags:
  - Reinforcement Learning
  - Baseline
---

# Policy Gradient with Baseline(1)

## Policy Gradient with Baseline

* Let the baseline $b$ be anything independent of $A$

* 
  $$
  \begin{align}
  \begin{aligned}
  E_{A\sim \pi}[b\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial\theta}]&=b\cdot E_{A\sim\pi}[\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]\\
  &=b\cdot\sum_a\pi(a\vert s;\theta)\cdot[\frac{1}{\pi(a\vert s;\theta)}\cdot\frac{\partial\pi(a\vert s;\theta)}{\partial \theta}]\\
  &=b\cdot\sum_a\frac{\partial \pi(a\vert s;\theta)}{\partial \theta}\\
  &=b\cdot\frac{\sum_a\pi(a\vert s;\theta)}{\partial \theta}\\
  &=b\cdot \frac{\partial 1}{\partial \theta}\\
  &=0
  \end{aligned}
  \end{align}
  $$

* 

so that we have this **theorem**

if $b$ is **independent** of $A$, then $E_{A\sim\pi}[b\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]=0$$

* Policy Gradient

$$
\begin{align}
\frac{\partial V_\pi(s)}{\partial \theta}&=E_{A\sim\pi}[Q_\pi(s,A)\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]-E_{A\sim\pi}[b\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]\\
&=E_{A\sim\pi}[\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}(Q_\pi(s,A)-b)]
\end{align}
$$

## Monte Carlo Approximation

* Policy gradient

$$
\frac{\partial V_\pi(s)}{\partial \theta}=E_{A_t\sim\pi}[\frac{\partial \ln\pi(A_t\vert s_t;\theta)}{\partial \theta}(Q_\pi(s_t,A_t)-b)]
$$

* Let $$g(a_t)=\frac{\partial \ln\pi(A_t\vert s_t;\theta)}{\partial \theta}(Q_\pi(s_t,A_t)-b)$$	
  * Randomly sample $$a_t\sim\pi(\cdot\vert s_t;\theta)$$ and compute $g(a_t)$
  * $g(a_t)$ is an unbiased estimate of the policy gradient

  $$
  E_{A_t\sim \pi}[g(A_t)]=\frac{\partial V_\pi(s_t)}{\partial\theta}
  $$

  * Notice that $b$ affects $g(a_t)$
  * A good $b$ leads to small variance and speeds up convergence

## Choice of Baseline

### Choice 1: b=0

$$
\frac{\partial V_\pi(s)}{\partial \theta}=E_{A_t\sim\pi}[\frac{\partial \ln\pi(A_t\vert s_t;\theta)}{\partial \theta}\cdot Q_\pi(s_t,A_t)]
$$

* Standard policy gradient

### Choice 2: b is state-value

* Because $s_t$ has been observed, $$b=V_\pi(s_t)$$ is independent of $A_t$
* $V_\pi(s_t)$ is close to $Q_\pi(s_t,A_t)$, so that it can speed up the convergence

$$
V_\pi(s_t)=E_{A_t}[Q_\pi(s_t,A_t)]
$$

