---
title: 'Reinforcement Learning-Overestimation & Solutions'
date: 2021-08-01
permalink: /posts/2021/08/Overestimation/
tags:
  - Reinforcement Learning
  - Deep Learning
---

# Reinforcement Learning-Overestimation & Solutions

## TD learning for DQN

* TD target :

$$
y_t=r_t+\gamma\cdot\max_a Q(s_{t+1},a;w)
$$

* SGD

$$
w\leftarrow w-\alpha\cdot(Q(s_t,a_t;w)-y_t\cdot\frac{\partial Q(s_t,a_t;w)}{\partial w}
$$

We use $y_t$, which is partly based on $Q$, to update $Q$ itself (**Bootstrapping**)

## Problem of Overestimation

### Reason 1: Maximization

* TD target:  $$y_t=r_t+\gamma\cdot\max_a Q(s_{t+1},a;w)$$
* TD target is bigger than the real action-value

Derivation procedure is given below

* True action-value: $$x(a_1), x(a_2),..., x_(a_n)$$
* Noisy estimation made by DQN: $$Q(s,a_1;w), ..., Q(s,a_n;w)$$
* Suppose the estimation is unbiased

$$
mean_a(x(a))=mean_a(Q(s,a;w))
$$

* $$q=\max_a Q(s,a;w)$$ is typically an overestimation

$$
q\geqslant\max_a(x(a))
$$

### Reason 2: Bootstrapping propagates the overestimation

* TD learning performs bootstrapping

  * TD target in part uses $$q_{t+1}=\max_a Q(s_{t+1},a;w)$$

  * Use the TD target for updating $Q(d_t,a_t;w)$

![capture_20210801145251052](/images/OE1.bmp)

![capture_20210801145307108](/images/OE2.bmp)

### Why is overestimation a shortcoming?

* The agent is controlled by the DQN $$a_t = \arg\max_{a} Q(s_t,a;w)$$
* Uniform overestimation is not a problem

* Non-uniform overestimation is problematic
  * TD algorithm pushes $Q(s_t,a_t;w)$ towards $y_t$
  * The more frequently $(s,a)$ appears in the replay buffer, the worse $Q(s,a;w)$ overestimates $Q^*(s,a)$

## Solutions

![capture_20210801161837341](/images/OE3.bmp)

### Target Network

Use a target network to compute TD targets

* Target network $Q(s,a;w^-)$
  * Same structure as the DQN, $Q(s,a;w)$
  * Different parameters $w^-\neq w$
* Use $Q(s,a;w)$ to control the agent and collect experience

$$
\{(s_t,a_t,r_t,s_{t+1})\}
$$

* Use $Q(s,a;w^-)$ to compute TD target

$$
y_t=r_t+\gamma\cdot \max_a Q(s_{t+1},a;w^-)
$$

**Notice!**

SGD: Only update the weights $w$ of DQN network 
$$
w\leftarrow w-\alpha\cdot\delta_t\cdot\frac{\partial Q(s_t,a_t;w)}{\partial w}
$$

* **Periodically update $w^-$**
  * Option1: $$w^-\leftarrow w$$
  * Option2: $$w^-\leftarrow \tau\cdot w+(1-\tau)\cdot w^-$$ ($\tau$ is a hyper-parameter)

**Though better than naive update, TD learning with target network nevertheless overestimates action-values**

### Double DQN

* Select using DQN

$$
a^*=\arg\max_a Q(s_{t+1},a;w)
$$

* Evaluation using target network

$$
y_t=r_t+\gamma\cdot Q(s_{t+1},a^*;w^-)
$$

* Double DQN alleviates overestimation

$$
Q(s_{t+1},a^*;w^-)\leqslant \max_a Q(s_{t+1},a;w^-)
$$

​	left side is the estimation by Double DQN, while right side is the estimation by target network

**It is the best method among three mentioned**