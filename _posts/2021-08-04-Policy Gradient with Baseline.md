---
title: 'Reinforcement Learning-Policy Gradient with Baseline'
date: 2021-08-04
permalink: /posts/2021/08/baseline/
tags:
  - Reinforcement Learning
  - Baseline
---

# Policy Gradient with Baseline

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

so that we have this **theorem**

if $b$ is **independent** of $A$, then $E_{A\sim\pi}[b\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]=0$$

* Policy Gradient

$$
\begin{align}
\begin{aligned}
\frac{\partial V_\pi(s)}{\partial \theta}&=E_{A\sim\pi}[Q_\pi(s,A)\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]-E_{A\sim\pi}[b\cdot\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}]\\
&=E_{A\sim\pi}[\frac{\partial \ln\pi(A\vert s;\theta)}{\partial \theta}(Q_\pi(s,A)-b)]
\end{aligned}
\end{align}
$$

### Monte Carlo Approximation

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

### Choice of Baseline

#### Choice 1: b=0

$$
\frac{\partial V_\pi(s)}{\partial \theta}=E_{A_t\sim\pi}[\frac{\partial \ln\pi(A_t\vert s_t;\theta)}{\partial \theta}\cdot Q_\pi(s_t,A_t)]
$$

* Standard policy gradient

#### Choice 2: b is state-value

* Because $s_t$ has been observed, $$b=V_\pi(s_t)$$ is independent of $A_t$
* $V_\pi(s_t)$ is close to $Q_\pi(s_t,A_t)$, so that it can speed up the convergence

$$
V_\pi(s_t)=E_{A_t}[Q_\pi(s_t,A_t)]
$$

## REINFORCE with Baseline

policy gradient:

$$
\frac{\partial V_\pi(s)}{\partial \theta}=E_{A_t\sim\pi}[\frac{\partial \ln\pi(A_t\vert s_t;\theta)}{\partial \theta}(Q_\pi(s_t,A_t)-V_\pi(s_t))]
$$

Stochastic policy gradient:

$$
g(a_t)=\frac{\partial \ln\pi(A_t\vert s_t;\theta)}{\partial \theta}(Q_\pi(s_t,A_t)-V_\pi(s_t))
$$

We don't know $Q_\pi(s_t,A_t)$ and $V_\pi(s_t)$ yet, so more approximation is needed

first, we approximate $Q_\pi(s_t,A_t)$

* We have $$Q_\pi(s_t,a_t)=E[U_t\vert s_t,a_t]$$
* Monte Carlo approximation to $Q_\pi(s_t,a_t)\approx u_t$(REINFORCE)
  * Observing the trajectory $$s_t,a_t,r_t,s_{t+1},a_{t+1},r_{t+1},...,s_n,a_n,r_n$$
  * Compute return $$\sum_{i=t}^n \gamma^{i-t}\cdot r_i$$
  * $u_t$ is an unbiased estimate of $Q_\pi(s_t,a_t)$

Next, we approximate $V_\pi(s_t)$

* Approximate  $V_\pi(s_t)$ by the value network $v(s;w)$

So that we have **approximate policy gradient**

$$
\frac{\partial V_\pi(s)}{\partial \theta}\approx g(a_t)\approx \frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial \theta}\cdot (u_t-v(s_t;w))
$$

with **three approximations**:

* Approximate expectation using one sample $a_t$ (Monte Carlo)

* Approximate $Q_\pi(s_t,a_t)$ by $u_t$ (Monte Carlo)
* Approximate $V_\pi(s)$ by the value network $v(s;w)$

### Update Policy Network

$$
\theta\leftarrow \theta+\beta\cdot\frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial \theta}\cdot (u_t-v(s_t;w))
$$

while $$-\delta_t=(u_t-v(s_t;w))$$

$$
\theta\leftarrow \theta+\beta\cdot\delta_t\cdot \frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial \theta}
$$

### Update Value Network

* $v(s_t;w)$ is an approximation to 

$$V_\pi(s_t)=E[U_t\vert s_t]$$

* Prediction error 

$$\delta_t=v(s_t;w)-u_t$$

* Gradient 

$$\frac{\partial \delta_t^2/2}{\partial w}=\delta_t\cdot\frac{\partial v(s_t;w)}{\partial w}$$

* Gradient descent 

$$w\leftarrow w-\alpha\cdot \frac{\partial v(s_t;w)}{\partial w}$$

## Advantage Actor-Critic (A2C)

A2C's value network(critic) $v(s;w)$ is an approximation to the **state-value** function $V_\pi(s)$, which is different from the classical Actor-Critic method.

![capture_20210804160119323](/images/Baseline.bmp)

### Training of A2C

* Observe a transition $(s_t,a_t,r_t,s_{t+1})$
* TD target $$y_t=r_t+\gamma \cdot v(s_{t+1};w)$$
* TD error $$\delta_t=v(s_t;w)-y_t$$
* Update the policy network(actor) by gradient ascent

$$
\theta\leftarrow\theta-\beta\cdot\delta_t\cdot\frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial\theta}
$$

* Update the value network(critic) by gradient descent

$$
w\leftarrow w-\alpha \cdot\delta_t\cdot\frac{\partial v(s_t;w)}{\partial w}
$$

### Explanation(Mathematical Derivation)

Identity

$$
Q_\pi(s_t,a_t)=E_{s_{t+1},A_{t+1}}[R_t+\gamma\cdot Q_\pi(S_{t+1},A_{t+1})]
$$

Thus,

$$
\begin{aligned}
Q_{\pi}(s_t,a_t)&=E_{s_{t+1}}[R_t+\gamma\cdot E_{A_{t+1}}[Q_\pi(S_{t+1},A_{t+1})]]\\
&=E_{S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]
\end{aligned}
$$

**Theorem 1:**

$$
Q_{\pi}(s_t,a_t)=E_{S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]
$$

By definition

$$
\begin{aligned}
V_\pi(s_t)&=E_{A_t}[Q_\pi(s_t,A_t)]\\
&=E_{A_t}[E_{S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]]
\end{aligned}
$$

**Theorem 2**:

$$
V_\pi(s_t)=E_{A_{t},S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]
$$

#### Monte Carlo Approximation

**Theorem 1:**

$$
Q_{\pi}(s_t,a_t)=E_{S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]
$$

* Suppose we have $(s_t,a_t,r_t,s_{t+1})$
* Unbiased estimation

$$
Q_\pi(s_t,a_t)\approx r_t+\gamma\cdot V_\pi(s_{t+1})
$$

**Theorem 2**:

$$
V_\pi(s_t)=E_{A_{t},S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]
$$

* Suppose we have $(s_t,a_t,r_t,s_{t+1})$

* Unbiased estimation

$$
V_\pi(s_t)\approx r_t+\gamma\cdot V_\pi(s_{t+1})
$$

#### Update Policy Network

* Stochastic policy gradient

$$
g(a_t)=\frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial \theta}(Q_\pi(s_t,a_t)-V_\pi(s_t))
$$

After using **Theorem 1**, we have

$$
g(a_t)=\frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial \theta}(r_t+\gamma\cdot V_\pi(s_{t+1})-V_\pi(s_t))
$$

Then approximate $V_\pi(s)$ by the value network $v(s;w)$

$$
g(a_t)=\frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial \theta}(r_t+\gamma\cdot v(s_{t+1};w)-v(s_t;w))
$$

so that we have policy gradient ascent

$$
\theta\leftarrow \theta+\beta\cdot\frac{\partial\ln(a_t\vert s_t;\theta)}{\partial\theta}\cdot(y_t-v(s_t;w))
$$

#### Update Value Network

* Monte Carlo approximation

$$
V_\pi(s_t)=E_{A_{t},S_{t+1}}[R_t+\gamma\cdot V_\pi(S_{t+1})]
$$

Approximation

$$
v(s_t;w)\approx r_t+\gamma\cdot v(s_{t+1};w)
$$

TD learning encourage $v(s_t;w)$ to approach $y_t$

* TD error: $$\delta_t=v(s_t;w)-y_t$$
* Gradient: $$\frac{\partial \delta_t^2/2}{\partial w}=\delta_t\cdot\frac{\partial v(s_t;w)}{\partial w}$$

* Update value network by gradient descent 

$$
w\leftarrow w-\alpha\cdot\delta_t\cdot\frac{\partial v(s_t;w)}{\partial w}
$$

## REINFORCE versus A2C

### REINFORCE with Baseline

* Observing a trajectory form time $t$ to $n$
* Return $$u_t=\sum_{i=t}^n\gamma^{i-t}\cdot r_i$$
* Error $$\delta_t=v(s_t;w)-u_t$$

* Update the policy network

$$
\theta\leftarrow \theta-\beta\cdot\delta_t\cdot\frac{\partial\ln\pi(a_t\vert s_t;\theta)}{\partial \theta}
$$

* Update the value network

$$
w\leftarrow w-\alpha\cdot\delta_t\cdot\frac{\partial v(s_t;w)}{\partial w}
$$

### A2C with Multi-Step TD Target

Observing a transition $(s_t,a_t,r_t,s_{t+1})$

* **One-Step TD Target**

$$
y_t=r_t+\gamma \cdot v(s_{t+1};w)
$$

Observing $m$ transitions $\{(s_{t+i},a_{t+i},r_{t+i},s_{t+i+1})\}_{i=0}^{m-1}$

* **m-Step TD Target**

$$
y_t=\sum_{i=0}^{m-1}\gamma^i\cdot r_{t+i}+\gamma^m\cdot v(s_{t+m};w)
$$

**Algorithm**

* Observe a trajectory from time $t$ to $t+m-1$ 
* TD target $$y_t=y_t=\sum_{i=0}^{m-1}\gamma^i\cdot r_{t+i}+\gamma^m\cdot v(s_{t+m};w)$$
* TD error $$\delta_t=v(s_t;w)-y_t$$
* Update the policy network(actor) by gradient ascent

$$
\theta\leftarrow\theta-\beta\cdot\delta_t\cdot\frac{\partial \ln\pi(a_t\vert s_t;\theta)}{\partial\theta}
$$

* Update the value network(critic) by gradient descent

$$
w\leftarrow w-\alpha \cdot\delta_t\cdot \frac{\partial v(s_t;w)}{\partial w}
$$

### A2C versus REINFORCE

* A2C uses m-step TD target(with bootstrapping)

$$
y_t=\sum_{i=0}^{m-1}\gamma^i\cdot r_{t+i}+\gamma^m\cdot v(s_{t+m};w)
$$

* REINFORCE uses observed return (without bootstrapping)

$$
u_t=\sum_{i=0}^{n-t}\gamma^i\cdot r_{t+i}
$$

