+++
title = '生成式模型中的变分推断'
date = 2025-06-28T20:52:24+08:00
draft = false
summary = ""

+++



本文主要探究变分推断在生成式模型中（以 VAE、DDPM 为例）是如何应用的。

## 什么是变分推断

首先我们要知道什么是变分推断（Variational Inference；VI）。变分通常同样用于近似某些难以直接表达的分布，例如我们现在已经有观测数据 x，我们希望知道后验分布 <span> $p(z|x)$ </span>。但是我们难以直接写出 <span> $p(z|x)$ </span>。此时我们可以选择一个比较简单的 <span> $q(z|x)$ </span>，通过优化（最小化 KL 散度）让两者尽可能接近。

而在生成式模型中，我们通常会认为观测数据遵从某些潜在变量（Latent Variable）$z$ 的控制，即 <span> $p(x)=\int p(z)p(x|z)dz$ </span>。举个例子，如果我们希望建立模型，这个模型可以生成各式各样的图片。此时如果我们让模型直接给我们随机生成一张图片，那么你可能会得到一只狗的图片，或者一辆车。但是如果我们给定一些约束，例如我们希望得到一只猫的图片，那么模型就会正确地生成一只猫。此时我们先不管如何添加约束，而是思考一下，这个过程中模型是如何控制生成图片的类型的呢？显然模型中存在某些部分（参数）控制着模型每次输出的图片的类型（或者其他约束信息，例如颜色、形状等）。那么我们认为模型实际上学习到了一些能够描述图片本质的信息的变量，而由于我们无法描述或者观测到这些变量，所以将其称之为潜在变量。

> 如果我们不考虑潜在变量的控制效果，那基本就是原始的自编码器（AutoEncoder；AE）的思想，即让模型想办法直接学习 <span> $p(x)$ </span>，但这会导致严重的过拟合问题。因此变分自编码器（Variational AutoEncoder；VAE） 提出了潜在变量的概念，并要求该潜在变量遵循标准正态分布，然后使用变分推断来优化模型的学习过程，从而解决了正则化和连续采样的问题。

回到开始，我们有观测数据 <span> $x$ </span>，我们认为模型的学习目标是 <span> $ p(x)=\int p(x|z)p(z)dz$ </span>。其中 <span> $p(x)$ </span> 是先验分布，且我们假设其为标准正态分布。$p(x|z)$ 是似然，即给定潜在变量 $z$ 后观测数据 $x$ 出现的概率。

> 由于模型实际上建模了 $x$ 和 $z$ 的联合分布，因此也可以用 $p(x,z)$ 来表示整个模型。

>  在实际生成过程中，往往通过两步采样的方式，即先采样 $z\sim p(z)$ ，然后采样 $x\sim p(x|z)$ 来生成结果。

## 变分推断在 VAE 中的应用

模型的最终目的是最大化 $log\left(p\left(x\right)\right)$ ，但是边缘似然 $log(p(x))$ 存在对 $z$ 的积分，因为潜在变量 $z$ 往往对应着一个高维空间，因此实际中这个积分是很难解的。因此我们引入一个近似的后验分布 $q(z|x)$ ，也称为变分分布，进行如下恒等变化：

<div>

$$
\begin{align*}
\log\left(p\left(x\right)\right)&=\log\int p\left(x|z\right)p(z)dz \\\\
&=\log\int q(z|x)\frac{p(x|z)p(z)}{q(z|x)}dz \\\\
&=\log\mathbb{E}_{q(z|x)}\left[\frac{p(x|z)p(z)}{q(z|x)}\right]
\end{align*}
$$

</div>

> 其中应用了重要性采样的思想，即对于一个难以求采样的分布 $p(x)$ ，我们希望计算某个 $f(x)$ 在该分布下的期望，可以使用一个容易采样的分布 $q(x)$ 来间接估计，其核心思想可以表述为：<span> $ \int p(x)f(x)dx=\int q(x)\frac{f(x)p(x)}{q(x)}dx=\mathbb E_{q(x)}\left[f(x)\frac{p(x)}{q(x)}\right] $ </span>。
> 通过对 $q(x)$ 的采样 $N$ 个样本来近似的求出 $f(x)$ 在 $x\sim p(x)$ 下的期望，其中 <span> $w_i=\frac{p(x)}{q(x)} $</span> 称为重要性权重。

再根据 Jensen 不等式得出：

<div>
$$
\log(p(x))=log\mathbb{E}_{q(z|x)}\left[\frac{p(x|z)p(z)}{q(z|x)}\right] \ge \mathbb{E}_{q(z|x)}\left[\log\left(\frac{p(x|z)p(z)}{q(z|x)}\right)\right]
$$
</div>

> Jensen 不等式是指，对于凸函数有 $\mathbb E[f(x)] \ge f(\mathbb E(x))$，对于凹函数有 $\mathbb E[f(x)] \le f(\mathbb E(x))$。

此时我们得到了 $\log(p(x)) $ 的下限，我们也将其称之为证据下界（Evidence Lower Bound；ELBO）。即：

<div>

$$
\begin{align*}
\mathbf{ELBO}&=\mathbb E_{q(z|x)}\left[\log\left(\frac{p(x|z)p(z)}{q(z|x)}\right)\right] \\\\
&= \mathbb E_{q(z|x)}\left[\log p(x|z)\right]+ \mathbb E_{q(z|x)}\left[\log p(z) - \log q(z|x)\right] \\\\
&= \mathbb E_{q(z|x)}\left[\log p(x|z)\right] + D_{KL}(q(z|x)\mid\mid p(z))
\end{align*}
$$

</div>

经过变换，我们将 ELBO 转为了两项之和。其中第一项代表重构项，描述了模型的生成能力。第二项表示正则项，约束了潜在变量不能偏离高斯先验。且此时我们的目标变成了最大化 ELBO。

> 最后一步推导涉及到 KL 散度，如果不懂可以先看一下 [从零理解熵、交叉熵、KL散度 | 周弈帆的博客](https://zhouyifan.net/2022/10/13/20221012-entropy/) 。



## 变分推断在 DDPM 中的应用

对于去噪扩散模型（Denosing Diffusion Probabilistic Model）而言，整个模型分为两个过程，即加噪过程 <span> $q(x_t|x_{t-1})$</span> 和去噪过程 <span> $p(x_{t-1}|x_t)$ </span>，观测数据为 <span>$x_0$ </span>，潜在变量就是每一步的噪声 <span> $x_1,\dots,x_t$ </span>。此时建模目标是是：

<div>

$$
p(x_0)=\int p(x_{0:T})dx_{1:T} \\\\
x_T\sim \mathcal N(\mathbf 0, \mathbf I)
$$

</div>
由于加噪过程是人为定义的 <span>$q(x_t|x_{t-1})=\mathcal N(x_t;\sqrt{1-\beta} x_{t-1},\beta_t \mathbf I) $ </span>，因此整个近似后验 <span> $q(x_{1:T}|x_0)$ </span> 都可以推导。通过变分推断，我们得到如下结果：

<div>

$$
\begin{align*}
\log p(x_0)&=\log \int q(x_{1:T}|x_0)\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}dx_{1:T}\\\\
&=\log \mathbb E_{q(x_{1:T}|x_0)}\left[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right]\\\\
&\ge  \mathbb E_{q(x_{1:T}|x_0)}\left[\log\frac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right]
\end{align*}
$$

</div>

继续简化 ELBO：

<div>
$$
\begin{align}
\mathbb{E}_{q}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = \mathbb{E}_{q} \left[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \right]
\end{align}
$$
</div>

由于 <span> $p(x_T)$  </span> 是标准正态分布，因此 ELBO 简化为每一步的加噪和去噪的 KL 散度之和。