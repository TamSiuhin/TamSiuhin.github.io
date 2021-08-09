---
title: 'Multi-Agent Reinforcement Learning-Basic Concepts'
date: 2021-08-02
permalink: /posts/2021/08/MARL_basic/
tags:
  - Multi-Agent Reinforcement Learning
  - Deep Learning
---

# Multi-Agent Reinforcement Learning-Basic Concepts

## Settings

* **Fully cooperative:** Agents collaborate to optimize a common return

* **Fully competitive:** One agent's gain is other agent's loss
* **Mixed Cooperative & competitive:** There are both cooperative setting and competitive setting
* **Self-interested:** Their rewards may or may not conflict

## Terminologies

### State & Action & State Transition

* There are $n$ agents
* Let $S$ be the state
* Let $A^i$ be the $i^{th}$ agent's action
* State transition

$$
p(s'\vert s,a^1,...,a^n)=P(S'=s'\vert S=s,A^1=a^1,...,A^n=a^n)
$$

* The next state $S'$ depends on all agents actions 

### Rewards

* Let $R^i$ be the reward received by the $i^{th}$ agent
* Fully cooperative: $$R^1=R^2=...=R^n$$
* Fully competitive: $$R^1\propto -R^2$$

* $R^i$ depends on $A^i$ as well as all the other agents' action $\{A^j\}_{j\neq i}$

### Returns

* Let $R_t^i$ be the reward received by the $i^{th}$ agent at time $t$
* Return of the $i^{th}$ agent

$$
U_t^i=R_t^i+R_{t+1}^i+R_{t+2}^i+R_{t+3}^i
$$

* Discounted return of the $i^{th}$ agent

$$
U_t^i=R_t^i+\gamma\cdot R_{t+1}^i+\gamma^2\cdot R_{t+2}^i+\gamma^3\cdot R_{t+3}^i+...
$$

where $\gamma$ is the discounted rate

### Policy Network

* Each agent has its own policy network $$\pi(a^i\vert s;\theta^i)$$
* Policy networks can be exchangeable

$$
\theta^1=\theta^2=...=\theta^n
$$

* Policy networks can be nonexchangeable: $$\theta^i\neq \theta^j$$

### Uncertainty in the Return

* Reward $R_t^i$ depends on $S_t$ and $$A_t^1,A_t^2,...,A_t^n$$

* Uncertainty in $S_t$ is from the state transition $p$

* Uncertainty in $A_t^i$ is from the policy network $$\pi(\cdot\vert s_t;\theta^i)$$

* The return $$U_t^i=\sum_{k=0}^\infty \gamma^k\cdot R_{t+k}^i$$ depends on 
  * all the future states $$\{S_t,S_{t+1},S_{t+2,...}\}$$
  * all the future actions $$\{A_{t}^i,A_{t+1}^i,A_{t+2}^i,...\}$$, for $$i=1,2,...,n$$

### State-Value Function

* State-Value of the $i^{th}$ agent 

$$
V^i(s_t;\theta^1,...,\theta^n)=E[U_t^i\vert S_t=s_t]
$$

* Randomness in actions $$A_t^j\sim\pi(\cdot\vert s_t;\theta^j)$$, for all $j = 1,2,...,n$. That is also why state value $V^i$ depends on $$\theta^1,\theta^2,...\theta^n$$

## Convergence: Nash Equilibrium

* While all the other agents' policy remain the same, the $i^{th}$ agent cannot get better expected return by changing its own policy
* Every agent is playing a best-response to the other agents' policies
* Nash equilibrium indicates convergence because no one has any incentive to deviate

## Single-Agent Policy Gradient for MARL

![capture_20210801212400271](/images/MARL1.bmp)

* The $i^{th}$ agent's policy network is $$\pi(a^i\vert s;\theta^i)$$
* The $i^{th}$ agent's state-value function is $$V^i(s;\theta^1,...\theta^n)$$
* Objective function

$$
J^i(\theta^1,...,\theta^{n})=E_s[V^i(S;\theta^1,...,\theta^n)]
$$

* Learn the policy network's parameter $\theta^i$ by

$$
\max_{\theta^i} J^i(\theta^1,...,\theta^n)
$$

#### It may not converge!

* If $i^{th}$ agent found

  $$
  \theta_*^i = \arg\max_{\theta^i} J^i(\theta^1,...,\theta^n)
  $$

* Now, another agent changes its policy 

* $\theta_*^i$ is no longer the best policy of the $i^{th}$ agent, the $i^{th}$ agent has to find a new $\theta^i$

* The other agent's objective functions will change, and therefore they will change their policy

So, we have to take agents' cooperation into consideration.