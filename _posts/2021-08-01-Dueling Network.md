---
title: 'Reinforcement Learning-Dueling Network'
date: 2021-08-01
permalink: /posts/2021/08/dueling_network/
tags:
  - Reinforcement Learning
  - Deep Learning
---

# Reinforcement Learning-Dueling Network

## Advantage Function

* Return 
  $$
  U_t = R_t+\gamma\cdot R_{t+1}+\gamma^2\cdot R_{t+2}+...
  $$

* Action-value function

$$
Q_\pi(s_t,a_t)=E[U_t\vert S_t=s_t, A_t=a_t]
$$

* State-value function

$$
V_\pi(s_t)=E_A[Q_\pi(s_t,A)]
$$

* Optimal action-value function

$$
Q^*(s,a)=\max_a Q_\pi(s,a)
$$

* Optimal state-value function

$$
V^*(s)=\max_\pi V_\pi(s)
$$

* **Optimal advantage function**

$$
A^*(s,a)=Q^*(s,a)-V^*(s)
$$

## Properties of Advantage Function

* **Theorem 1**:

$$
V^*(s)=\max_a Q^*(s,a)
$$

so that we have
$$
\max_a A^*(s,a)=\max_a Q^*(s,a)-V^*(s)=0
$$

* **Theorem 2**:

$$
Q^*(s,a)=V^*(s)+A^*(s,a)-\max_a A^*(s,a)
$$

## Dueling Network

* **Theorem 2**:

$$
Q^*(s,a)=V^*(s)+A^*(s,a)-\max_a A^*(s,a)
$$

* Approximate $V^*(s)$ by a neural network $V(s;w^V)$
* Approximate $A^*(s,a)$ by a neural network $A(s,a;w^A)$
* Thus, approximate $Q^*(s,a)$ by the **dueling network**

$$
Q(s,a;w^A,w^V)=V(s;w^V)+A(s,a;w^A)-\max_a A(s,a;w^A)
$$

![capture_20210801174303414](/images/DN1.bmp)

## Training 

* Dueling network $Q(s,a;w)$ is an approximation to $Q^*(s,a)$
* Learn the parameter $w=(w^A,w^V)$ in the same way as the DQN's
* Tricks can be used to improve the performance
  * Prioritized experience replay
  * Double DQN
  * Multi-step TD target

## Problem of Non-identifiability

### Equation 1: 

$$
Q^{*}(s,a)=V^{*}(s)+A^{*}(s,a)
$$

* Problem of non-identifiability
  * Let $$V'=V^{*}+10$ and $A'=A^{*}-10$$
  * Then $$Q^{*}(s,a)=V^{*}(s)+A^{*}(s,a)=V'(s)+A'(s,a)$$



### Equation 2: 

$$
Q^*(s,a)=V^{*}(s)+A^{*}(s,a)-\max_a A^{*}(s,a)
$$

* Does not have the problem 

* **Alternative:**

$$
Q(s,a;w)=V(s;w^V)+A(s,a;w^A)-\text{mean }A(s,a;w^A)
$$

mean() seems to be more effective in the experiments

## Summary

* Dueling network

$$
Q(s,a;w)=V(s;w^V)+A(s,a;w^A)-\text{mean }A(s,a;w^A)
$$

* Dueling network controls the agent in the same way as DQN
* Train dueling network by TD algorithm in the same way as DQN
* Do not train V and A separately