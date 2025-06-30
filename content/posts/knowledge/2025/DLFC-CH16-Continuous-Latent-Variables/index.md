+++
title = 'DLFC CH16 Continuous Latent Variables'
date = 2025-06-30T16:08:03+08:00
draft = true
summary = ""

+++

本文内容来自 [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) 一书的第十六章。

其中的数学公式会可能简洁，某些公式的详细推导过程在 Appendix 中给出。

## CH16 Continuous Latent Variables

什么是潜在变量（Latent Variables）？

这个概念主要是相对于高维数据而言的，对于很多高维数据，我们可以认为它们实际上是由一些低维的潜在变量控制。举个例子，对于手写数字图片（见下图），一般而言我们会认为每一个像素点都是一个维度，假如整个图片有 100x100 个像素，那么每个图片都是一个高达 10000 维的数据。但是换个角度思考，我们可以认为这些图片实际上是由第一张图片经过一些变化，例如平移和旋转得到的。让我们暂时忽略数字部分在像素级别的细微差异，我们是否可以认为，以下五张图片实际上是由三个变量（纵向平移，横向平移，中心旋转）所控制的呢？完全可以，此时我们可以认为这几张图片是由这三个潜在变量控制生成的，因为你只要拿到第一张图片，然后通过修改这三个变量，就可以得到后续所有的图片。而在观测数据中我们只能看到像素，而看不到控制生成过程的三个变量，所以这些变量就称之为潜在变量。

![手写数字3](Figure1.png)

上面的例子也许不是很严谨，例如控制生成过程的也许还有好多个其他的潜在变量，但是我们主要关注这种思想。实际上现实中绝大多数数据都符合这种特点，**即高维的数据往往集中在某些低维流形上，且其维度远小于原始数据维度**。此时我们就可以通过这些低维流形所代表在潜在变量来生成一些数据，但是生成的数据往往和现实不太一致，因为现实数据往往带有噪音。因此我们在生成的数据上添加一些噪音，从而使其更加接近观测数据。

>  这个过程很自然的启发了一些生成式模型。

而要从观测数据中找到控制其生成过程的潜在变量，我们就需要想办法抛弃无用的维度，将高维的观测数据降维到真正的低维流形上。

### 16.1 Principal Component Analysis

主成分分析（PCA）是一种非常常用的降维手段，也被称之为 Kosambi–Karhunen–Loeve transform。PCA 有两种定义，一是最大化投影后数据方差的线性投影；二是最小化投影代价的线性投影，投影代价通常定义为投影前后数据的均方距离。我们会从两个角度分别尝试推导出 PCA 的数学过程。

> 所有推导过程以单变量为基础，多变量降维请参考 canonical correlation analysis。

#### 16.1.1 Maximum variance formulation

我们有 $N$ 个观测数据 <span> $\{\mathbf{x}_n\}=\{\mathbf{x}_1,\dots,\mathbf{x}_N\}$ </span>，每个数据由一个 D 维向量表示，即 <span>$\mathbf{x}_i\in \mathbb{R}^D $</span> 。我们希望将其降维到一个 $M$ 维空间中（$M<D$），即每个数据点使用 $M$ 维向量表示。通常而言 $M$ 是给定的，后面我们会讨论一些选择 $M$ 的技巧。

现在考虑一个简单情况，即 $M=1$。我们希望将数据投影到一条线上，我们可以使用一个向量<span> $\mathbf{u}_t$ </span>来表示，而且我们关注的是向量的方向，而不在乎其长度，因为这个向量代表的是我们所要投影到的一维空间方向，至于这个向量到底多长，我们并不在乎，所以我们规定 <span>$\mathbf{u}_t^T\mathbf{u}_t=1$ </span>。投影后的向量就可以表示为 <span> $\mathbf{u}_t\mathbf{x}_n$</span>，此时我们不难求出投影后数据的均值为 <span>$\mathbf{u}_t^T\bar{\mathbf{x}}$ </span>，方差为 <span>$\mathbf{u}_t^T \mathbf{S}\mathbf{u}_t$ </span>。

> 均值和方差的推导见 Appendix A1。

我们要最大化投影后数据的方差，即 <span>$\max \mathbf{u}_t^T \mathbf{S} \mathbf{u}_t$ </span>，但是可能出现 <span>$\mathbf{u}_t^T \mathbf{S} \mathbf{u}_t =a$，有 </span> <span> $\hat{\mathbf{u}}_t=k\mathbf{u_t}$ </span>， 令 <span> $\hat{\mathbf{u}}_t^T \mathbf{S} \hat{\mathbf{u}}_t=k^2\mathbf{u}_t^T \mathbf{S} \mathbf{u}_t=k^2 a$</span> 。但我们不想关注向量的长度，因此我们要引入约束项 <span> $\mathbf{u}_t^T\mathbf{u}_t=1$ </span>。

此时使用拉格朗日乘子法，构建带约束的最大化方程：

<div>

$$
\begin{align}
\mathcal{L}(\mathbf{x},\lambda_1)=\mathbf{u}_t^T\mathbf{S}\mathbf{u}_t + \lambda_1(1-\mathbf{u}_t^T\mathbf{u}_t) \tag{1.1}
\end{align}
$$

</div>

该方程对 <span>$\mathbf{u}_t$</span> 求偏导得：

<div>

$$
\mathbf{S}\mathbf{u}_t=\lambda_1\mathbf{u}_t \tag{1.2}
$$

</div>

> 求导过程见 Appendix A2。

显然投影向量 <span> $\mathbf{u}_t$ </span> 是协方差矩阵 $\mathbf{S}$ 的一个特征向量，此时两边左乘 <span>$\mathbf{x}_t^T$</span> 得到投影后的方差 <span>$\mathbf{u}_t^T \mathbf{S}\mathbf{u}_t=\lambda_1$ </span>。当 $M>1$ 时，令投影后方差最大化的投影向量就是协方差矩阵的前 $M$ 个最大特征值所对应的特征向量。由此我们就推导出了 PCA 降维的数学过程，即对协方差矩阵做特征值分解，从最大的特征值开始选择对应的特征向量，就是降维所需要的投影向量。

但是这种方法的计算量太大，对于维度为 $D$ 的数据，其计算成本为 $O(D^3)$ 。

#### 16.1.2 Minimal error formulation

有 $D$ 维基向量 <span>$\{\mathbf{u}_n\}=\{\mathbf{u}_1,\dots,\mathbf{u}_N\}$</span>，满足 <span>$\mathbf{u}_t^T\mathbf{u}_t=\delta_{ij}$ </span>，意思是 $i=j$ 时 <span>$\mathbf{u}_t^T\mathbf{u}_t=1$ </span>，否则 <span>$\mathbf{u}_t^T\mathbf{u}_t=0$ </span>，这意味着 <span>$\{\mathbf{u}_n\}$ </span> 是一组完全正交的基向量。

对于 $D$ 维的观测数据  <span> $\{\mathbf{x}_n\}=\{\mathbf{x}_1,\dots,\mathbf{x}_N\}$ </span>，我们可以使用新的基向量做如下转换：

<div>

$$
\mathbf{x}_n=\sum_{i=1}^{D}\alpha_{ni}\mathbf{u}_i \tag{1.3}
$$

</div>

这相当于在新的基向量张成的空间中表示观测数据，原始数据 <span>$\mathbf{n}_n$</span> 的分量 <span>$\mathbf{x}_n=\{x_1,\dots,x_D\}$</span>，被替换为 <span>$\{\alpha_1,\dots,\alpha_D\}$</span>。此时我们在上式基础上同时左乘某个基向量 <span>$\mathbf{u}_j^T$ </span>，同时应用基向量之间的正交性，可以得到：

<div>

$$
\begin{align}
\mathbf{u}_j^T\mathbf{x}_n &= \sum_{i=1}^D\alpha_{ni}\mathbf{u}_j^T\mathbf{u}_i=\alpha_{nj}\\\\
\alpha_{nj}&=\mathbf{u}_j^T\mathbf{x}_n=\mathbf{x}_n^T\mathbf{u}_j
\end{align}\tag{1.4}
$$

</div>

我们将公式（1.4）的结果带入公式（1.3），可以改写为：

<div>

$$
\begin{align}
\mathbf{x}_n = \sum_{i=1}^D \left( \mathbf{x}_n^T \mathbf{u}_i \right) \mathbf{u}_i
\end{align}\tag{1.5}
$$
</div>

此时我们成功使用了自定义的完全正交基向量来表示观测数据，但是我们的目标是对数据进行降维，这意味着我们不能使用全部的 $D$ 个基向量，如果要降到 $M$ 维，那么我们就只能使用其中的 $M$ 个基向量来重新表示，但是选择哪些基向量比较好呢？

假设我们选择了前 $M$ 个基向量，我们可以将观测的数据近似的表示为：

<div>

$$
\begin{align}
\tilde{\mathbf{x}}_n = \sum_{i=1}^{M} z_{ni} \mathbf{u}_i + \sum_{i=M+1}^{D} b_i \mathbf{u}_i
\end{align}\tag{1.6}
$$

</div>

其中 <span>$\{z_{ni}\}$</span> 依赖于特定的观测数据点，而 <span>$\{b_i\}$</span> 则是对于所有数据点都一致的常量。这种表示下观测数据只有 $M$ 维基向量描述，而公式中的第二项用来补偿降维所失去维度，从而使我们可以在 $D$ 维中表示降维后的数据。

## Appendix

### A1

<div>

$$
\begin{align}
\frac{1}{N}\sum_{n=1}^{N}\mathbf{u}_t^T\mathbf{x}_n=\mathbf{u}_t^T\frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_n=\mathbf{u}_t^T\bar{\mathbf{x}} 
\end{align}\tag{A1.1}
$$

</div>

<div>

$$
\begin{align}
\frac{1}{N}\sum_{n=1}^{N}\{\mathbf{u}_t^T\mathbf{x}_n-\mathbf{u}_t^T\bar{\mathbf{x}}\}^2
&=\frac{1}{N}\sum_{n=1}^{N}\left[\mathbf{u}_t^T (\mathbf{x}_n - \bar{\mathbf{x}})\right]^2\\\\
&=\frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n-\bar{\mathbf{x}})^T \mathbf{u}_t \mathbf{u}_t^T (\mathbf{x}_n-\bar{\mathbf{x}})\\\\
&=\mathbf{u}_t^T \left( \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n-\bar{\mathbf{x}})(\mathbf{x}_n-\bar{\mathbf{x}})^T \right) \mathbf{u}_t\\\\
&= \mathbf{u}_t^T \mathbf{S} \mathbf{u}_t 
\end{align}\tag{A1.2}
$$

</div>

### A2

<div>

$$
\begin{align}
\mathcal{L}(\mathbf{x},\lambda_1)&=\mathbf{u}_t^T\mathbf{S}\mathbf{u}_t + \lambda_1(1-\mathbf{u}_t^T\mathbf{u}_t) \\\\
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_t}&=\frac{\partial (\mathbf{u}_t^T\mathbf{S}\mathbf{u}_t)}{\partial \mathbf{u}_t}+
\frac{\partial \lambda_1}{\partial \mathbf{u}_t} - \frac{\partial (\lambda_1\mathbf{u}_t^T\mathbf{u}_t)}{\partial \mathbf{u}_t}=
2\mathbf{S}\mathbf{u_t}-2\lambda_1\mathbf{u_t}=0\\\\
\mathbf{S}\mathbf{u}_t&=\lambda_1\mathbf{u}_t 
\end{align}\tag{A2.1}
$$

</div>