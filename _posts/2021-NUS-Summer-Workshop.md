---
title: 'NUS SWS 2021 (Online)'
date: 2021-07-30
permalink: /posts/2021/07/2021-NUS-Summer-Workshop/
tags:
  - Summer Workshop
  - Deep Learning
  - Twitter Bot Detection
---

# 2021 NUS Summer Workshop--Lecture1

Some Foundation knowledge of machine learning. 

## F-score

The F-score, also called the F1-score, is a measure of a model’s accuracy on a dataset. It is used to evaluate binary classification systems, which classify examples into ‘positive’ or ‘negative’

The F-score is commonly used for evaluating information retrieval systems such as search engines, and also for many kinds of machine learning models, in particular in natural language processing.

### F-score Formula

The formula for the standard F1-score is the harmonic mean of the precision and recall. **A perfect model has an F-score of 1 **.
$$
\begin{align}
F_{1}&=\frac{2}{\frac{1}{recall}+\frac{1}{precision}}\\
&=2\times\frac{precision\times recall}{precision + recall}\\
&=\frac{tp}{tp+\frac{1}{2}(fp+fn)}
\end{align}
$$

### F-score Formula Symbols Explained

| precision                                                    | Precision is the fraction of true positive examples among the examples that the model classified as positive. In other words, the number of true positives divided by the number of false positives plus true positives. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **recall**                                                   | Recall, also known as sensitivity, is the fraction of examples classified as positive, among the total number of positive examples. In other words, the number of true positives divided by the number of true positives plus false negatives. |
| ![img](https://images.deepai.org/user-content/9385586353-thumb-5207.svg) | The number of true positives classified by the model.        |
| ![img](https://images.deepai.org/user-content/5344178495-thumb-9914.svg) | The number of false negatives classified by the model.       |
| ![img](https://images.deepai.org/user-content/1189295773-thumb-9043.svg) | The number of false positives classified by the model.       |

from the table above, we have
$$
precision=\frac{tp}{tp+fp}\\
recall=\frac{tp}{tp+fn}
$$

### Generalized Fβ-score Formula

The adjusted F-score allows us to weight precision or recall more highly if it is more important for our use case. Its formula is slightly different:
$$
\begin{align}
F_{\beta}&=(1+\beta^{2})\times\frac{precision\times recall}{(\beta^2\times precision)+recall}\\
&=\frac{(1+\beta^2)tp}{(1+\beta^2)tp+\beta^2fn+fp}
\end{align}
$$


*Mathematical definition of the Fβ-score*

A factor indicating how much more important recall is than precision. For example, if we consider recall to be twice as important as precision, we can set *β* to 2. The standard F-score is equivalent to setting *β* to one.

## Activation Function

### Sigmoid Function

Sigmoid functions are used in machine learning for logistic regression and basic neural network implementations and they are the introductory activation units. But for advanced Neural Network Sigmoid functions **are not preferred due to various drawbacks (vanishing gradient problem)**. It is one of the most used activation function for beginners in Machine Learning and Data Science when starting out.
$$
f'(x)=f(x)(1-f(x))\\
f(x)=sigmoid(x)=\frac{1}{1+e^{-x}}
$$
There is a major drawback of info loss due to the derivative having a short range. And vanishing and exploding problem in present, with sigmoid function is positive to output, all our output neurons have a positive output too which is not ideal.

### Tanh Function

Sigmoid functions are used in machine learning for logistic regression and basic neural network implementations and they are the introductory activation units. But for advanced Neural Network Sigmoid functions are not preferred due to various drawbacks (vanishing gradient problem). It is one of the most used activation function for beginners in Machine Learning and Data Science when starting out.

 ![](D:\MarkDown\1_3OerqzdzmcDWbo6XC_zG_g.png)

The formula for hyperbolic tangent (tanh) can be given as follows
$$
tanh(x)=\frac{e^{2x}-1}{e^{2x}+1}
$$
This does not however mean that tanh is devoid of the vanishing or exploding gradient problem, it persists even in the case of tanh but unlike Sigmoid as it is centered at Zero, it is more optimal than Sigmoid Function. Therefore other functions are employed more often which we will see below for machine learning.

### ReLU (Rectified Linear Units) and Leaky ReLU

#### ReLU

A **Rectified Linear Unit** (A unit employing the rectifier is also called a rectified linear unit ReLU) has output 0 if the input is less than 0, and *raw* output otherwise. That is, if the input is greater than 0, the output is equal to the input. The operation of ReLU is closer to the way our *biological neurons* work.
$$
f(x)=max(x,0)
$$
ReLU aren’t without any drawbacks some of them are that ReLU is **Non Zero centered** and is **non differentiable at Zero**, but differentiable anywhere else.

#### Leaky ReLU

Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. The slope coefficient is determined before training, it is not learnt during training. This type of activation function is popular in tasks where we we may suffer from sparse gradients, for example training generative adversarial networks.

![1_siH_yCvYJ9rqWSUYeDBiRA](D:\MarkDown\1_siH_yCvYJ9rqWSUYeDBiRA.png)

With Leaky ReLU there is a small negative slope, so instead of not firing at all for large gradients, our neurons do output some value and that makes our layer much more optimized too.

### PReLU(Parametric ReLU) Function

In Parametric ReLU as seen from the figure above, instead of using a fixed slope like 0.01 used in Leaky ReLU, a parameter ‘a’ is made that will change depending on the model, for x < 0

Using weights and biases, we tune the parameter that is learned by employing backpropagation across multiple layers .
$$
f(x)=\begin{cases}
x &if \quad x>0\\
ax &otherwise.
\end{cases}
\\f(x)=max(x,ax)
$$
Therefore as PReLU relates to the maximum value, we use it in something called “maxout” networks too.

### ELU(Exponential LU) Function

Exponential Linear Units are are used to speed up the deep learning process, this is done by making the mean activations closer to Zero, here an alpha constant is used which must be a positive number.

![1_mWEL2mKKC_y8les1hBd9VA](D:\MarkDown\1_mWEL2mKKC_y8les1hBd9VA.png)


$$
f(x)=\begin{cases}
x&if \; x>0\\
a(e^x-1)&otherwise.
\end{cases}
$$
ELU have been shown to **produce more accurate results than ReLU and also converge faster**. ELU and ReLU are same for positive inputs, but for **negative inputs ELU smoothes (to -alpha) slowly whereas ReLU smooths sharply**.

## [Kaiming Initialization](https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138)

If we random initialize the weight, it will cause two problems, the **vanishing gradient problem** and **exploding gradient problem**.

**Vanishing gradient problem** means weights vanish to 0. Because these weights are multiplied along with the layers in the backpropagation phase. If we initialize weights very small(<1), the gradients tend to get smaller and smaller as we go backward with hidden layers during backpropagation. Neurons in the earlier layers learn much more slowly than neurons in later layers. This causes minor weight updates.

**Exploding gradient problem** means weights explode to infinity (NaN). Because these weights are multiplied along with the layers in the backpropagation phase. If we initialize weights very large(>1), the gradients tend to get larger and larger as we go backward with hidden layers during backpropagation. Neurons in the earlier layers update in huge steps, `W = W — ⍺ * dW`, and the downward moment will increase.

[Kaiming et al.](https://arxiv.org/pdf/1502.01852.pdf) derived a sound initialization method by cautiously modeling non-linearity of ReLUs, which makes **extremely deep models (>30 layers)** to converge. Below is the Kaiming initialization function.
$$
std=\sqrt{\frac{2}{(1+a^2)\times fan\_in}}
$$

+ a: the negative slope of the rectifier used after this layer (0 for ReLU by default)

+ fan_in: the number of input dimension. If we create a `(784, 50)`, the fan_in is 784. `fan_in` is used **in the feedforward phase**. If we set it as `fan_out `, the fan_out is 50. `fan_out` is used in the **backpropagation phase**. 

### How to choose `mode`  ?

If you **create weight implicitly by creating a linear layer**, you should set `mode='fan_in' `, for example

```python
linear = torch.nn.Linear(node_in, node_out)
init.kaiming_normal_(linear.weight, mode=’fan_in’)
t = relu(linear(x_valid))
```

If you **create weight explicitly by creating a random matrix**, you should set `mode='fan_out'`, for example

```python
w1 = torch.randn(node_in, node_out)
init.kaiming_normal_(w1, mode=’fan_out’)
b1 = torch.randn(node_out)
t = relu(linear(x_valid, w1, b1))
```

According to the [document](https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_normal_), choosing `'fan_in'` preserves the magnitude of the variance of the weights **in the forward pass**. Choosing `'fan_out'` preserves the magnitudes in the backward pass. 

## Learning Rate

Stochastic gradient descent is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the [back-propagation of errors algorithm](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/), referred to as simply backpropagation.

The amount that the weights are updated during training is referred to as the step size or the “learning rate”.
$$
W_1=W_1-\alpha\cdot\frac{\partial Loss}{\partial W_{1}}\\
b_1=b_1-\alpha\cdot\frac{\partial Loss}{\partial b_1}
$$
The learning affects how quickly our model can converge to a local minima.

### ADAM(Adaptive Moment Estimation)

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t\\
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat{m_t}=\frac{m_t}{1-\beta_1^t}\\
\hat{v_t}=\frac{v_t}{1-\beta_2^t}
$$

update:
$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}
$$


### SGD with Momentum





## Weight Decay

Weight decay is a common way to deal with overfitting 

最广泛使用的正则化技术之一

### 均方范数作为硬性限制

$$
\min L(W,b) \quad subject \; to \;||w||^2\leqslant\theta
$$

* 通常不限制偏移b

* 小$\theta$意味着更小的正则项

### 均方范数作为柔性限制

对每个$\theta$都可以找到$\lambda$使得硬性限制等价于下式
$$
\min \{L(W,b)+\frac{\lambda}{2}||w||^2\}
$$
$\lambda$为超参数，控制正则项的重要程度

* $\lambda=0:$无作用

* $\lambda\rightarrow\infin,W^*\rightarrow 0$

$$
W^*=\arg \min \{L(W,b)+\frac{\lambda}{2}||w||^2\}
$$

加入罚（Penalty）降低模型复杂度

### 参数更新法则

* 计算梯度

$$
\frac{\partial}{\partial W}( L(W,b)+\frac{\lambda}{2}||w||^2 )=\frac{\partial L(W,b)}{\partial W}+\lambda W
$$

* 更新参数

$$
W_{t+1}=(1-\eta\lambda)W_t-\eta\frac{\partial L(W_t,b_t)}{\partial W_t}
$$

通常$\eta \lambda<1$，在深度学习中叫做权重衰退（Weight Decay）

## Dropout(丢弃法)

主流的控制多层感知机方法

* 一个好的模型需要对输入数据的扰动鲁棒
* 丢弃法：在层之间加入噪音

### 无偏差地加入噪音

对$x$加入噪音得到$x'$，我们希望$E(x')=x$

丢弃法对每个元素进行如下扰动
$$
x'_i=\begin{cases}
0&with\; probability \; p\\
\frac{x_i}{1-p}&otherwise
\end{cases}
$$
训练过程中：

记$x_1$为第一个隐藏层的输出
$$
\begin{align}
x_1&=\alpha (W_1 x+b_1)\\
x_1'&=dropout(h)\\
o&=W_2x_1'+b_2\\
y&=softmax(o)\\
\end{align}
$$
可以认为每次取一部分神经网络做平均，结果更好

看作正则项

尝试建立更复杂的模型，并用正则化（Dropout）进行调整
