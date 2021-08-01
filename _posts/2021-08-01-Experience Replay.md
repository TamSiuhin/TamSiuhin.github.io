---
title: 'Reinforcement Learning-Experience Replay'
date: 2021-08-01
permalink: /posts/2021/08/Experience_Replay/
tags:
  - Reinforcement Learning
  - Deep Learning
---

# Reinforcement Learning-Experience Replay

## Revisiting DQN & TD Learning

### Deep Q Network (DQN)

![capture_20210801114907283](/images/EP1.bmp)

Approximate the optimal action-value function, $Q^*(s,a)$, by $Q(s,a;w)$ (Deep Q Network)

### Temporal Difference (TD) Learning

* Observe state $s_t$ and perform action $a_t$
* Environment provides new state $s_{t+1}$ and reward $r_t$
* **TD target**

$$
y_t = r_t+\gamma\cdot\max_a Q(s_{t+1},a'w)
$$

* **TD error**

$$
\delta_t =q_t-y_t\text{, where }q_t=Q(s_t,a_t;w)
$$

* **Goal**: Make $q_t$ close to $y_t$, for all $t$

* **TD learning**  Find $w$ by minimizing $L(w)=\frac{1}{T}\sum_{t=1}^T\frac{\delta_t^2}{2}$

* Online gradient descent

  * Observe $(s_t,a_t,r_t,s_{t+1})$ and compute $\delta_t$
  * compute gradient $g_t=\frac{\partial\delta_t^2/2}{\partial w}=\delta_t\cdot\frac{\partial Q(s_t,a_t;w)}{\partial w}$

  * Gradient descent $w\leftarrow w-\alpha\cdot g_t$

#### Shortcoming

* Waste of Experience
* Correlated Updates

## Experience Replay

* A transition: $(s_t,a_t,r_t,s_{t+1})$
* Score recent $n$ transitions in a **replay buffer**
* Remove old transition so that the buffer has at most $n$ transitions
* Buffer capability $n$ is a tuning hyper-parameter
  * $n$ is typically large ($10^5\sim 10^6$)
  * Setting of $n$ is application-specific

### TD with Experience Replay

* Find $w$ by minimizing $L(w)=\frac{1}{T}\sum_{t=1}^T\frac{\delta_t^2}{2}$

* Stochastic gradient descent(SGD)

  * Randomly sample a transition $(s_i,a_i,r_i,s_{i+1})$ from the buffer
  * Compute TD error $\delta_i$
  * Stochastic gradient

  $$
  g_i=\frac{\partial \delta_i^2/2}{\partial w}=\delta_i\cdot\frac{\partial Q(s_i,a_i;w)}{\partial w}
  $$

  * SGD: $$w\leftarrow w-\alpha\cdot g_i$$

## Prioritized Experience Replay

### Basic Idea

* Not all transitions are equally important 
* If a transition has high TD error $\vert\delta_t\vert$, it will be given high priority

![capture_20210801135645608](/images/EP2.bmp)

### Scaling Probabilities

Use importance sampling instead of uniform sampling

* Sampling probability $p_t\propto\vert \delta_t\vert +\epsilon$
* Sampling probability $p_t\propto\frac{1}{\text{rank}(t)}$

### Scaling Learning Rate

* If uniform sampling is used, $\alpha$ is the same for all transitions
* If importance sampling is used, $\alpha$ shall be adjusted according to the importance

Scale the learning rate by $(n p_t)^{-\beta}$, where $\beta\in(0,1)$

* If $p_1=p_2=...=p_n=\frac{1}{n}$ (uniform sampling), the scaling factor is equal to 1
* High-importance transitions (with high $p_t$) have low learning rates
* In the beginning, set $\beta$ small, increase $\beta$ to 1 over time

### Update TD Error

* Associate each transition $(s_t,a_t,r_t,s_{t+1})$ with a TD error $\delta_t$
* If a transition is newly collected, we do not know its $\delta_t$
  * Simply set its $\delta_t$ to the maximum
  * It has the highest priority
* Each time $(s_t,a_t,r_t,s_{t+1})$ is selected from the buffer, we update its $\delta_t$