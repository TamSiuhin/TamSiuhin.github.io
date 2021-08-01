# Reinforcement Learning-Stochastic Policy for Continuous Control

## Policy Network

### One Dim Normal Distribution

* Assume the degree of freedom is one
* Let $\mu$ (mean) and $\sigma$ (std) be function of $s$
* Let Policy function be the PDF of normal distribution $N(\mu, \sigma^2)$

$$
\pi(a\vert s)=\frac{1}{\sqrt{2\pi}\sigma}\cdot \exp(-\frac{(a-\mu)^2}{2\sigma^2})
$$

### Multivariate Normal Distribution

* Let the degree of freedom be $d$
* Let $\mu,\sigma$ : $s\to R^d$ be function of $s$
* Let $\mu_i$ and $\sigma_i$ be the $i^{th}$ elements of $\mu(s)$ and $\sigma(s)$, respectively
* Let policy function be the PDF of multivariate normal

$$
\pi(a\vert s)=\prod_{i=1}^{d} \frac{1}{\sqrt{2\pi}\sigma_i}\cdot \exp(-\frac{(a_i-\mu_i)^2}{2\sigma_i^2})
$$

but $\mu$ and $\sigma$ is unknown

**sample action $a$ from $N(\mu,\sigma^2)$**

### Function Approximation

* Approximate the mean $\mu(s)$, by the neural network $\mu(s;\theta^\mu)$

* Use log variance to approximate the std $\sigma(s)$ (More precisely)

$$
\rho_i=\ln \sigma_i^2,\quad for\;i=1,2,...,d
$$

* Approximate $\rho$ , by the neural network $\rho(s;\theta^\rho)$

![capture_20210801094122922](/images/DP3.bmp)

## Continuous Control

* Observe state $s$
* Compute mean and log variance using the neural network

$$
\begin{align}
\hat\mu&=\mu(s;\theta^\mu)\\
\hat\rho&=\rho(s;\theta^\rho)
\end{align}
$$

* Compute $\hat\sigma_i=\exp(\hat\rho_i)$, for all $i=1,2,...,d$

* Randomly sample action $a$ by 

$$
a_i\sim N(\hat\mu_i,\hat\sigma_i^2)\quad\text{for all }i=1,2,...,d
$$

## Training Policy Network

1. Auxiliary network

2. Policy gradient methods

   * Reinforce
   * Actor-Critic

   ### Auxiliary Network

   Stochastic policy gradient
   $$
   g(a)=\frac{\partial \ln\pi(a\vert s;\theta)}{\partial \theta}\cdot Q_\pi(s,a)
   $$
   **Policy network is:**
   $$
   \pi(a\vert s;\theta^\mu,\theta^\rho)=\prod_{i=1}^d\frac{1}{\sqrt{2\pi}\sigma_i}\cdot\exp(-\frac{(a_i-\mu_i)^2}{2\sigma_i^2})
   $$
   Natural log of the policy network is 
   $$
   \begin{align}
   \ln\pi(a\vert s;\theta^\mu,\theta^\rho)&=\sum_{i=1}^{d}[-\ln\sigma_i=\frac{(a_i-\mu_i)^2}{2\sigma_i^2}]+\text{const}\\
   &=\sum_{i=1}^d[-\frac{\rho_i}{2}-\frac{(a_i-\mu_i)^2}{2\cdot\exp(\rho_i)}]+\text{const}
   \end{align}
   $$
   **Auxiliary Network**
   $$
   f(s,a;\theta)=\sum_{i=1}^d[-\frac{\rho_i}{2}-\frac{(a_i-\mu_i)^2}{2\cdot \exp(\rho_i)}]
   $$
   ![capture_20210801101136382](/images/DP2.bmp)

   gradient $\frac{\partial f}{\partial \theta}$ can be automatically computed by pytorch

   #### Recap

   We have built three neural networks
   $$
   \mu(s;\theta^\mu),\rho(s;\theta^\rho),f(s,a;\theta)
   $$

* $\mu(s,\theta^\mu)$ computes the mean (control agent)
* $\rho(s;\theta^\rho)$ computes the log variance (control agent)
* Auxiliary network combines $\mu$ and $\rho$ , helps with training
* Use $\frac{\partial f}{\partial \theta}$ for computing policy gradient

### Policy Gradient

Stochastic policy gradient
$$
g(a)=\frac{\partial \ln\pi(a\vert s;\theta)}{\partial \theta}\cdot Q_\pi(s,a)
$$
Auxiliary network
$$
f(s,a;\theta)=\ln\pi(a\vert s;\theta)+\text{const}
$$
So that we have
$$
g(a)=\frac{\partial f(s,a;\theta)}{\partial\theta}\cdot Q_\pi(s,a)
$$
But $Q_\pi(s,a)$ is unknown, it needs approximation

* REINFORCE
* Actor-Critic

#### REINORCE

* Approximates $Q_\pi(s_t,a_t)$ by the observed return 

$$
u_t=r_t+\gamma\cdot r_{t+1}+\gamma^2\cdot r_{t+2}+\gamma^3\cdot r_{t+3}+...
$$

* Update policy network 

$$
\theta\leftarrow\theta+\beta\cdot\frac{\partial f(s,a;\theta)}{\partial \theta}\cdot u_t
$$

#### Actor-Critic

* Approximates $Q_\pi$ by the value network $q(s,a;w)$
* Update policy network

$$
\theta\leftarrow\theta+\beta\cdot\frac{\partial f(s,a;\theta)}{\partial \theta}\cdot q(s,a;w)
$$

* Update value network $q(s,a;w)$ by TD learning

## Summary of Continuous Control

* Discretize the action space and use standard DQN or policy network
* Deterministic policy network
* Stochastic policy network

