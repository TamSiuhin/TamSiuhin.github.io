---
title: 'Multi-Agent Reinforcement Learning-Centralize vs. Decentralize'
date: 2021-08-02
permalink: /posts/2021/08/centralize&decentralize/
tags:
  - Multi-Agent Reinforcement Learning
  - Centralize & Decentralize
---

# Multi-Agent Reinforcement Learning: Centralized vs. Decentralized

## Architectures

* **Fully decentralized:** Every agent uses its own observations and rewards to learn its policy. Agents do not communicate.
* **Fully centralized:** The agents send everything to the central controller. The controller makes decision for all the agents.
* **Centralized training with decentralized execution:** A central controller is used during training. The controller is disabled after training.

## Partial Observation

* An agent may or may not have full knowledge of the state $s$
* Let $o^i$ be the $i^{th}$ agent's observation 
* Partial observation: $o^i\neq s$

* Full observation: $o^1=...=o^n=s$

## Fully Decentralized

![capture_20210801221327175](/images/CD1.bmp)

same as single-agent reinforcement learning 

### Fully Decentralized Actor-Critic Method

* The $i^{th}$ agent has a policy network (actor): $$\pi(a^i\vert o^i;\theta^i)$$

* The $i^{th}$ agent has a value network (critic): $$q(i^i,a^i;w^i)$$

* Agent do not share observations and actions
* Train the policy and value network in the same way as the single-agent setting

* This does not work well

## Fully Centralized

![capture_20210801221939727](/images/CD2.bmp)

### Centralized Actor-Critic Method

* Let $$a=[a^1,a^2,...,a^n]$$ contain all the agents' action
* Let $$o=[o^1,o^2,...,o^n]$$ contain all the agents' observation
* The central controller knows $a,o$ and all the rewards
* The controller has $n$ policy networks and $n$ value networks:
  * Policy network (actor) for the $i^{th}$ agent: $$\pi(a^i\vert o;\theta^i)$$
  * Value network (critic) for the $i^{th}$ agent: $$q(o,a;w^i)$$
* **Centralized Training:**  Training is performed by the controller
  *  The central controller knows $a,o$ and all the rewards
  * Train $$\pi(a^i\vert o;\theta^i)$$ using policy gradient
  * Train $$q(o,a;w^i)$$ using TD algorithm
* **Centralized Execution:** Decision are made by the controller
  * For all $i$, the $i^{th}$ agent sends its observation, $o^i$ to the controller
  * The controller knows $$o=[o^1,o^2,...,o^n]$$
  * For all $i$, the controller samples action by $a^i\sim \pi(\cdot\vert;\theta^i)$ and send $a^i$ to the $i^{th}$ agent

#### Shortcoming: Slow during Execution

* All the agents send their observations to the central controller
* The central controller makes decisions $$a = [a^1,a^2,...,a^n]$$ and sends $a^i$ to the $i^{th}$ agent
* Communication and synchronization cost time
* Real-time decision is impossible

### Centralized Training with Decentralized Execution

*  Each agent has its own policy network (actor): $$\pi(a^i\vert o^i;\theta^i)$$
* The central controller has $n$ value networks (critic): $$q(o,a;w^i)$$
* **Centralized Training:** During training, the central controller knows all the agents' observation, actions, and rewards

* **Decentralized Execution:** During execution, the central controller and its value networks are not used

#### Centralized Training

![capture_20210802133014908](/images/CD3.bmp)

* The central controller trains the critics $$q(o,a;w^i)$$ for all $i$
* To update $w^i$, TD algorithm takes as inputs
  * All actions $$a=[a^1,a^2,...,a^n]$$
  * All the observations $$o=[o^1,o^2,...,o^n]$$
  * The $i^{th}$ reward $r^i$

![capture_20210802134328302](/images/CD4.bmp)

* Each agent locally trains the actor $$\pi(a^i\vert o^i;\theta^i)$$, using policy gradient
* To update $\theta^i$, the policy gradient algorithm takes as input $$(a^i,o^i,q^i)$$

#### Decentralized Execution

![capture_20210802135002039](/images/CD5.bmp)

* All actors interact with the environment without communicating with central controller

