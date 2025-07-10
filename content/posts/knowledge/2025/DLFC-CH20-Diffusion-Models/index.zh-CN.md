+++
title = '[Deep Learning: Foudations and Concepts] CH20-Diffusion Models'
date = 2025-07-06T09:36:30+08:00
draft = false
outdatedInfoWarning = false
collections = ["Deep Learning: Foundations and Concepts"]
tags = ["Book", "Deep Leaning", "Diffusion Models"]
categories = ["Deep Learning", "Diffusion Models"]
summary = "本文主要讲了扩散模型的基本原理，以及其和分数匹配之间的关系，并且扩展了如何进行条件控制的扩散生成。"

+++

本文内容来自 [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) 一书的第二十章——扩散模型

正文的数学公式会尽可能详细易懂，但是某些公式的详细推导太长，因此其过程在 Appendix 中给出。

应当注意的是本文只是博主在学习过程中对于原书内容的摘要性记录，并不能完全代替原书内容。

## CH20-Diffusion Models

扩散模型的核心思想是（以图片生成为例），对一个图片进行多步加噪声操作，最终将图片变的类似一个从高斯噪声中采样出的样本。然后训练一个神经网络来逆向这个加噪过程，一旦这个神经网络训练完毕，就可以直接从一个高斯分布中采样，然后使用这个神经网络生成一张图片。

扩散模型也可以被视为一个多层次的变分自编码器（VAE），每一步加噪去噪就对应了 VAE 的一次编码和解码，只不过编码过程被设置为一个固定的加噪操作，只有去噪操作是需要学习的。

### 20.1 Forward Encoder

假设我们有一张图片 $\mathbf{x}$ ，我们对其进行一步加噪操作，具体而言，就是对每一个像素独立的添加一个高斯噪声。得到加噪后的结果 <span> $\mathbf{z}_1$ </span>。

<span>

$$
\begin{align*}
\mathbf{z}=\sqrt{1-\beta_1}\mathbf{x}+\sqrt{\beta_1}\boldsymbol{\epsilon}_1
\end{align*}\tag{1.1}
$$

</span>

其中 <span> $\boldsymbol{\epsilon}_1\in \mathcal{N}(\mathbf{\epsilon}_1|\mathbf{0},\mathbf{I})$</span> 且 <span> $ \beta_1 < 1$ </span>。这保证了每一次加噪之后的 <span> $\mathbf{z}_t$ </span> 相比加噪前的 <span> $\mathbf{z}_{t-1}$ </span>而言其均值更接近 0 而方差更接近 $\mathbf{I}$。我们可以写出 <span> $\mathbf{z}_1$ </span> 的分布。

<span>

$$
\begin{align*}
q(\mathbf{z}_1|\mathbf{x})=\mathcal{N}(\mathbf{z}_1|\sqrt{1-\beta_1}\mathbf{x},\beta_1\mathbf{I})
\end{align*}\tag{1.2}
$$

</span>
后续的 <span> $\mathbf{z}_2,\dots,\mathbf{z}_T$ </span>可以用类似的方法定义。

<span>

$$
\begin{align*}
\mathbf{z}_t &= \sqrt{1-\beta_t}\mathbf{z}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t\tag{1.3} \\\\
q(\mathbf{z}_t|\mathbf{z}_{t-1}) &= \mathcal{N}(\mathbf{z}_t|\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I})\tag{1.4}
\end{align*}
$$

</span>
其中 <span> $\boldsymbol{\epsilon}_t\in \mathcal{N}(\boldsymbol{\epsilon}|\mathbf{0},\mathbf{I})$ </span>。整个加噪过程可以视为一个马尔科夫链。其中 <span> $\beta_t \in (0,1)$ </span> 的具体数值通常是人为指定的，一般会随 t 递增，即 <span> $\beta_1 < \beta_2 <\dots<\beta_T$ </span>。

#### 20.1.1 Diffusion Kernel
我们将每一步加噪的结果<span> $\mathbf{z}_t$ </span> 作为潜在变量，根据上面的定义，我们可以写出所有潜在变量的联合概率分布
<span>

$$
\begin{align*}
q(\mathbf{z}_1\dots\mathbf{z}_t|\mathbf{x})=q(\mathbf{z}_1|\mathbf{x})\prod_{t=2}^T q(\mathbf{z}_t|\mathbf{z}_{t-1})
\end{align*}\tag{1.5}
$$

</span>
现在我们边缘化 <span> $\mathbf{z}_1,\dots,\mathbf{z}_{t-1}$ </span>，就可以得到 t 步加噪之后的潜在变量分布
<span>

$$
\begin{align*}
q(\mathbf{z}_t|x)&=\mathcal{N}(\mathbf{z}_t|\sqrt{\alpha_t}\mathbf{x},(1-\alpha_t)\mathbf{I})\tag{1.6}\\\\
\mathbf{z}_t&=\alpha_t\mathbf{x}+(1-\alpha_t)\boldsymbol{\epsilon}_t\tag{1.7}
\end{align*}
$$

</span>
其中 <span> $\alpha_t=\prod_{1}^t(1-\beta_t)$ </span>，边缘概率的推导过程见 Appendix 1.1。
我们可以直接写出每个中间过程的潜在变量 <span> $\mathbf{z}_t$ </span> 的高斯分布形式，意味我们不需要完整的执行整个马尔科夫链，就可以直接采样出加噪过程中间的任意一步的结果，从而允许我们高效地进行训练。
需要注意的是边缘分布中的噪声项 <span> $\boldsymbol{\epsilon}_t$ </span> 表示所有 t 步加噪的总和，而不仅仅是第 t 步的噪声。
经过足够多步的加噪之后，最终结果几乎和高斯噪声一模一样，因此我们可以写出 $T\to \infin$ 的边缘概率分布为
<span>

$$
\begin{align*}
    q(\mathbf{z}_T|x)&=\mathcal{N}(\mathbf{z}_T|\mathbf{0},\mathbf{I})\tag{1.8}\\\\
    q(\mathbf{z}_T)&=\mathcal{N}(\mathbf{z}_T|\mathbf{0},\mathbf{I})\tag{1.9}
\end{align*}
$$

</span>

实际上公式 1.8 右边的结果表示 T 足够大之后的边缘概率和 $\mathbf{x}$ 就没有关系了，因此我们可以直接写成公式 1.9 的形式。
这个前向的加噪过程就是一个马尔科夫链过程，在扩散模型中被称为前向过程（Forward Process）。

#### 20.1.2 Conditional Distribution
我们的目标是学习去噪过程，也就是 <span> $q(\mathbf{z}_t|\mathbf{z}_{t-1})$ </span>的逆过程，使用贝叶斯定理我们可以得到
<span>

$$
\begin{align*}
    q(\mathbf{z}_{t-1}|\mathbf{z}_t)=\frac{q(\mathbf{z}_{t}|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1})}{q(\mathbf{z}_t)}
\end{align*}\tag{1.10}
$$

</span>
> 这个公式等号右侧看起来各个项都是已知的，但实际上并不是，如果我们代入之前的公式会发现最终的计算会需要 p(x)。

首先考虑以下公式
<span>

$$
\begin{align*}
    q(\mathbf{z}_{t-1}) = \int q(\mathbf{z}_{t-1}|\mathbf{x})p(\mathbf{x}) d\mathbf{x}
\end{align*}\tag{1.11}
$$

</span>
我们要边缘化潜在变量的概率分布就需要知道观测数据的分布，但是观测数据的分布就是我们的目标，我们知道了观测数据的分布就可以直接采样生成观测数据了，那还学什么呢？因此这个公式是无法直接指导训练过程的。

我们虽然不知道观测数据的分布，但是我们有观测数据本身，因此我们可以使用条件分布来重写上述公式

<span>

$$
\begin{align*}
    q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) = \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x})q(\mathbf{z}_{t-1}|\mathbf{x})}{q(\mathbf{z}_t|\mathbf{x})} 
\end{align*}\tag{1.12}
$$

</span>

根据马尔科夫链的性质，每一步加噪操作只依赖于上一步的信息，因此有 <span>$q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x}) = q(\mathbf{z}_t|\mathbf{z}_{t-1})$</span>

<span>

$$
\begin{align*}
    q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) = \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1}|\mathbf{x})}{q(\mathbf{z}_t|\mathbf{x})} 
\end{align*}\tag{1.13}
$$

</span>

上式中的分母可以忽略掉，因为我们希望得到的是 <span> $\mathbf{z}_{t-1}$ </span>的表达式，因此对于最终结果而言分母是一个固定的分布。将公式 1.4 和公式 1.6 代入，可以得到以下结果

<span>

$$
\begin{align*}
    q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) = \mathcal{N} (\mathbf{z}_{t-1}|\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t), \sigma_t^2\mathbf{I})
\end{align*}\tag{1.14}
$$

</span>

其中

<span>

$$
\begin{align*}
    \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) &= \frac{(1 - \alpha_{t-1})\sqrt{1 - \beta_t}\mathbf{z}_t + \sqrt{\alpha_{t-1}\beta_t}\mathbf{x}}{1 - \alpha_t} \tag{1.15} \\
    \sigma_t^2 &= \frac{\beta_t(1 - \alpha_{t-1})}{1 - \alpha_t} \tag{1.16}
\end{align*}
$$

</span>

现在我们就得到了反向过程的表达式，具体推导过程见 Appendix 1.2。

### 20.2 Reverse Decoder
由于 <span> $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ </span>涉及到对所有观测数据的积分，因此我们使用一个神经网络模型来学习一个分布 <span> $p(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{w})$ </span>来近似这个逆向过程。

> 需要注意的是，难以计算的是 <span> $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ </span>，但是上一小节我们已经得到了 <span> $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x})$ </span> 的表达式，这一小节也确实会用到，但作者并没有详细解释为什么在逆向过程存在闭式解的情况下还要使用神经网络近似。

对于公式 1.12，我们可以利用高斯分布的一个性质，即如果 <span> $q(\mathbf{z}_{t}|\mathbf{z}_{t-1})$ </span> 是一个足够窄（方差足够小）的高斯分布，那么 <span> $q(\mathbf{z}_{t}|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1})$ </span> 也会近似的遵循高斯分布。而要让前向过程所代表的高斯分布足够窄，根据公式 1.4 只需要 <span> $\beta_t$ </span> 足够小即可。如果对逆向过程泰勒展开，我们也可以发现逆向过程<span> $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ </span>的协方差也会接近正向过程的协方差。

因此我们使用如下形式的高斯分布来建模逆向过程

<span>

$$
\begin{align*}
    p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w}) = \mathcal{N} (\mathbf{z}_{t-1}|\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t), \beta_t\mathbf{I})
\end{align*}\tag{2.1}
$$

</span>

其中 <span> $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$ </span> 是由神经网络参数 $\mathbf{w}$ 控制的，并且其接受 t 作为输入，这意味着这个神经网络可以通过 t 缩放 <span>$\beta_t$</span> 从而直接建模任意一步的去噪过程，因此只需要一个神经网络就可以实现整个马尔科夫链上的逆向过程。

> 通常而言马尔科夫链上相邻两步的数据维度是一致的，因此一个输入和输出同纬度的模型是最好的，在图像领域，很自然的就可以想到 U-net。

我们写出整个马尔科夫链逆向过程的联合概率分布

<span>

$$
\begin{align*}
    p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T |\mathbf{w}) = p(\mathbf{z}_T) \prod_{t=2}^{T} p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w}) p(\mathbf{x}|\mathbf{z}_1, \mathbf{w}).
\end{align*}\tag{2.2}
$$

</span>

其中 <span> $p(\mathbf{z}_T)$ </span>我们假定其和 <span> $p(\mathbf{z}_T)$ </span>的分布一致，是一个标准高斯分布。

#### 20.2.1 Training the decoder

我们接下来要确定训练神经网络的目标函数，一个很显然的选择是似然函数

<span>

$$
\begin{align*}
    p(\mathbf{x}|\mathbf{w}) = \int \cdots \int p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T |\mathbf{w}) d\mathbf{z}_1 \ldots d\mathbf{z}_T
\end{align*}\tag{2.3}
$$

</span>
但是这涉及到对高度复杂的神经网络函数进行积分，往往是不可解的。

#### 20.2.2 Evidence Lower Bound
既然上面的精确似然不可解，那我们变分推断来构建似然的下界 ELBO，只要最大化下界就可以近似地最大化似然。

根据 [DLFC-CH16-Continuous-Latent-Variables](/DLFC-CH16-Continuous-Latent-Variables) 中的公式 3.1~3.5，我们引入变分分布 <span> $q(\mathbf{z})$ </span> 之后可以直接写出如下结果

<span>

$$
\begin{align*}
    \ln p(\mathbf{x}|\mathbf{w}) &= \mathcal{L}(\mathbf{w}) + KL (q(\mathbf{z})\|p(\mathbf{z}|\mathbf{x}, \mathbf{w}))\tag{2.4}\\\\
    \mathcal{L}(\mathbf{w}) &= \int q(\mathbf{z}) \ln \frac{p(\mathbf{x}, \mathbf{z}|\mathbf{w})}{q(\mathbf{z})} d\mathbf{z}\tag{2.5}
\end{align*}
$$

</span>

现在我们的目标是最大化 ELBO，首先我们先推导出 ELBO 的显式表达式。很多其他应用会选择一个可以改变的 <span> $q(\mathbf{z})$ </span>，通过交替优化 <span> $q(\mathbf{z})$ </span> 和模型参数项 <span> $p(\mathbf{x}, \mathbf{z}|\mathbf{w})$ </span> 来不断逼近最大似然。但是在扩散模型中，我们选择一个固定的 <span> $q(\mathbf{z}_1,\dots,\mathbf{z}_T|\mathbf{x})$ </span>，此时唯一可以学习的就只有模型中的参数了。

然后我们将公式 1.5 和 2.2 代入公式 2.5，得到如下形式

<span>

$$
\begin{align*}
    \mathcal{L}(\mathbf{w}) &= \mathbb{E}_q \left[ \ln \frac{p(\mathbf{z}_T) \prod_{t=2}^T p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w}) p(\mathbf{x}|\mathbf{z}_1, \mathbf{w})}{q(\mathbf{z}_1|\mathbf{x}) \prod_{t=2}^T q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x})} \right] \\\\
    &= \mathbb{E}_q \left[ \ln p(\mathbf{z}_T) + \sum_{t=2}^T \ln \frac{p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x})} - \ln q(\mathbf{z}_1|\mathbf{x}) + \ln p(\mathbf{x}|\mathbf{z}_1, \mathbf{w}) \right] \tag{2.6}
\end{align*}
$$

</span>

并且我们定义上式中取期望的过程如下

<span>

$$
\begin{align*}
    \mathbb{E}_q [ \cdot ] \equiv \int \cdots \int q(\mathbf{z}_1|\mathbf{x}) \prod_{t=2}^T q(\mathbf{z}_t|\mathbf{z}_{t-1}) [ \cdot ] d\mathbf{z}_1 \ldots d\mathbf{z}_T\tag{2.7}
\end{align*}
$$

</span>

公式 2.6 中的第一项 <span> $\ln p(\mathbf{z}_T)$ </span>是固定的标准高斯分布，而第三项 <span> $\ln q(\mathbf{z}_1|\mathbf{x})$ </span> 和参数 $\mathbf{w}$ 无关，因此这两项都可忽略。

第四项类似于 VAE 中的重建误差，可以使用蒙特卡洛估计来近似其期望。

<span>

$$
\begin{align*}
    \mathbb{E}_q [\ln p(\mathbf{x}|\mathbf{z}_1, \mathbf{w})] \approx \frac{1}{L} \sum_{l=1}^L \ln p(\mathbf{x}|\mathbf{z}_1^{(l)}, \mathbf{w})\tag{2.8}
\end{align*}
$$

</span>

其中 <span> $\mathbf{z}_1^{(l)} \sim \mathcal{N}(\mathbf{z}_1|\sqrt{1-\beta_1}\mathbf{x}, \beta_1\mathbf{I})$ </span>。

> 原书中蒙特卡洛估计的右侧没有取平均的系数项，可能是因为这个系数对于参数优化而言不重要。

现在 ELBO 中还剩下第二项，这一项由若干子项求和组成，每一个子项都是马尔科夫链中相邻的一对潜变量组成。由于前向过程是固定的，因此我们可以直接通过公式 1.6 来计算 <span> $p(\mathbf{z}_{t-1})$ </span> 并从其中采样得到样本，然后根据公式 1.4 得到 <span> $p(\mathbf{z}_{t})$ </span>，但是由于我们需要进行足够多次数的采样（趋于无穷大），这个两步采样的过程会造成很大的方差。

通过一些技巧我们可以重写 ELBO，变成对每一项只采样一个值的形式

#### 20.2.3 Rewriting the ELBO

根据贝叶斯定理，我们重写前向过程

<span>

$$
\begin{align*}
    q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x}) &= \frac{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})q(\mathbf{z}_t|\mathbf{x})}{q(\mathbf{z}_{t-1}|\mathbf{x})}\tag{2.9}\\\\
    \ln \frac{p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{x})} &= \ln \frac{p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})} + \ln \frac{q(\mathbf{z}_{t-1}|\mathbf{x})}{q(\mathbf{z}_t|\mathbf{x})}\tag{2.10}
\end{align*}
$$

</span>

公式 2.10 中右侧的第二项与参数无关，可以直接忽略，然后代入 ELBO 中，此时就只剩下对 <span> $\mathbf{z}_{t-1}$ </span>的采样。

<span>

$$
\begin{align*}
    L(\mathbf{w}) = \mathbb{E}_q \left[ \sum_{t=2}^T \ln \frac{p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})} + \ln p(\mathbf{x}|\mathbf{z}_1, \mathbf{w}) \right] .
\end{align*}
$$

</span>

我们可以将 ELBO 重写为如下形式

<span>

$$
\begin{align*}
L(\mathbf{w}) &= \underbrace{\int q(\mathbf{z}_1|\mathbf{x}) \ln p(\mathbf{x}|\mathbf{z}_1, \mathbf{w}) d\mathbf{z}_1}_{\text{reconstruction term}}\\\\ 
&- \underbrace{\sum_{t=2}^T \int \text{KL}(q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})\|p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w}))q(\mathbf{z}_t|\mathbf{x}) d\mathbf{z}_t}_{\text{consistency terms}} \tag{2.11}
\end{align*}
$$

</span>

对于重建项，我们可以通过公式 2.8 进行近似采样，并使用和 VAE 一样的重参数化技巧进行训练。而对于一致性项，两个分布分别由公式 1.14 和 2.1 给出，代入后得到如下形式

<span>

$$
\begin{align*}
    \text{KL}(q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})\|p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w})) = \frac{1}{2\beta_t} \|\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) - \boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)\|^2 + \text{const}\tag{2.12}
\end{align*}
$$

</span>

所有和参数 $\mathbf{w}$ 无关的项都被合并到常数项中。由此我们得到了最大化似然的完整形式。

#### 20.2.4 Predicting the noise
有人发现改变一下模型的目标可以有效地提高结果的质量，那就是不再预测每一步的去噪过程，而是直接预测当前步添加到原始数据中的总噪声。为了实现这一目标，我们重写第 t 步噪声的表达式，

<div>

$$
\begin{align*}
\mathbf{x} = \frac{1}{\sqrt{\alpha_t}} \mathbf{z}_t - \frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} \boldsymbol{\epsilon}_t\tag{2.13}
\end{align*}
$$

</div>

这个表达式描述了第 t 步加噪结果 <span> $\mathbf{z}_t$ </span> 和原始数据 <span> $\mathbf{x}$ </span> 之间的噪声分量 <span> $\boldsymbol{\epsilon}_t$ </span>。

然后我们继续重写反向过程的均值，将公式 2.13 代入 <span> $\mathbf{m}_t(\mathbf{x},\mathbf{z}_t)$ </span>，可以得到

<div>

$$
\begin{align*}
    \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \boldsymbol{\epsilon}_t \right)\tag{2.14}
\end{align*}
$$

</div>

使用一个新的符号 <span> $g(\mathbf{z}_t,\mathbf{w},t)$ </span>来表示预测总噪声的神经网络。根据公式 2.14 可以得知，<span> $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$ </span> 可以使用下面的形式来替代

<div>

$$
\begin{align*}
    \boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t) = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} g(\mathbf{z}_t, \mathbf{w}, t) \right)\tag{2.15}
\end{align*}
$$

</div>

将公式 2.14 和 2.15 代入 2.12，我们可以得到新的 KL 散度表达式

<div>

$$
\begin{align*}
    &KL(q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})‖p(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{w}))  \\\\
    &= \frac{\beta_t}{2(1 - \alpha_t)(1 - \beta_t)} \|\mathbf{g}(\mathbf{z}_t, \mathbf{w}, t) - \boldsymbol{\epsilon}_t\|^2 + const  \\\\
    &= \frac{\beta_t}{2(1 - \alpha_t)(1 - \beta_t)}  \|\mathbf{g}(\sqrt{\alpha_t}\mathbf{x} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_t, \mathbf{w}, t) - \boldsymbol{\epsilon}_t\|^2 + const\tag{2.16}
\end{align*}
$$

</div>

现在再来考虑公式 2.11 的重构项，其中 <span> $\ln p(\mathbf{x}|\mathbf{z}_1,\mathbf{w})$ </span>可以通过公式 2.8 进行蒙特卡洛近似。同时根据公式 2.1 我们已知该分布的均值可以通过 <span> $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$ </span> 来表示，我们再次使用<span> $g(\mathbf{z}_t,\mathbf{w},t)$ </span>来替代 <span> $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$ </span>，并展开高斯分布的显式表达式，可以得到如下形式

<div>

$$
\begin{align*}
    \ln p(\mathbf{x}|\mathbf{z}_1, \mathbf{w}) = -\frac{1}{2(1-\beta_1)} \|\mathbf{g}(\mathbf{z}_1, \mathbf{w}, 1) - \boldsymbol{\epsilon}_1\|^2 + const\tag{2.17}
\end{align*}
$$

</div>

对比公式 2.17 和 公式 2.16，不难发现 2.17 就是 $t=1$ 情况下的 2.16，因此我们可以合并重构项和一致项这两个公式。此外还有人发现如果我们直接忽略掉公式 2.16 前面那一坨系数的话，模型的性能会有进一步的提升，忽略掉系数之后整个马尔科夫链的每一步都具有了相同的权重，此时我们化简之后的 ELBO 是如下形式

<div>

$$
\begin{align*}
    \mathcal L(\mathbf{w}) = - \sum_{t=1}^{T} \| g(\sqrt{\alpha_t}\mathbf{x} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_t, \mathbf{w}, t) - \boldsymbol{\epsilon}_t \|^2 \tag{2.18}
\end{align*}
$$

</div>
现在我们可以从新的角度来解释这个 ELBO。对于给定的观测数据 <span> $\mathbf{x}$ </span> 和步数 t，我们得到加噪之后的潜在变量 <span> $\mathbf{z}_t$ </span>。上述 ELBO 计算了预测噪声和真实噪声之间的距离平方。



#### 20.2.5 Generating new samples

一旦模型训练好，我们就可以从高斯分布 <span> $p(\mathbf{z}_T)$ </span>中采样，然后通过马尔科夫链进行去噪。对于马尔科夫链中某一步的潜在变量 <span> $\mathbf{z}_t$ </span>，我们需要三步来生成 <span>$\mathbf{z}_{t-1}$ </span>，首先我们计算神经网络模型 <span> $\mathbf{g}(\mathbf{z}_t,\mathbf{w},\mathbf{t})$ </span>，然后根据公式 2.15 计算 <span> $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$ </span>，最后根据公式 2.1 采样得到 <span> $\mathbf{z}_{t-1}$  </span>。但是需要注意的是，我们还要在 $\mathbf{z}_{t-1}$ 上添加一个额外的经过缩放的噪声项，所以最终的结果如下式

<div>

$$
\begin{align}
\mathbf{z}_{t-1}=\boldsymbol{\mu}(\mathbf{z}_t,\mathbf{w},\mathbf{t})+\sqrt{\beta_t}\boldsymbol{\epsilon}\tag{2.19}
\end{align}
$$

</div>

其中噪声项遵循标准高斯分布。需要注意的是当我们从 <span>$\mathbf{z}_1$ </span>生成 $\mathbf{x}$ 的时候就不再加噪声项了，因为我们最终希望得到没有噪声的数据。

> 关于为什么去噪过程要加噪声项，有很多说法。例如去噪过程本身就是具有不确定性的，因此添加噪声项能反映这种不确定性。此外噪声项也是正则化的一部分，可以避免去噪过程过于自信，使得最终结果具有一定的多样性。

以上生成过程最大的问题在于需要经过多步去噪，因此计算量比较大。所以有人提出了 denoising diffusion implicit models（DDIM）来提高采样速度。



### 20.3 Score Matching

还有一个叫做分数匹配（Score Matching）的技术一直和扩散模型同步发展，但是两者在本质上几乎就是同一个东西，而且 Score Matching 中的很多思想可以帮助我们更好的理解和改进扩散模型。

分数匹配是利用一种叫做得分函数（Score Function）建立起来的，得分函数被定义为相对于观测数据 $\mathbf{x}$ 的对数似然的梯度。

<div>

$$
\begin{align*}
s(\mathbf{x})=\nabla_{\mathbf{x}}\ln p(\mathbf{x})\tag{3.1}
\end{align*}
$$

</div>

这个函数有什么用呢？我们考虑两个函数 $q(\mathbf{x})$ 和 $p(\mathbf{x})$，并且 <span> $\nabla_{\mathbf{x}}\ln q(\mathbf{x})=\nabla_{\mathbf{x}}\ln p(\mathbf{x})$ </span>，然后我们在等式两边对 $\mathbf{x}$ 进行积分，就可以得到 $q(\mathbf{x})=Kp(\mathbf{x})$ 其中 K 是独立于 $\mathbf{x}$ 的常数。因此如果我们建立一个模型 $s(\mathbf{x}, \mathbf{w})$ 能够代表这个得分函数，那么我们就相当于建模了原始数据的密度。

#### 20.3.1 Score loss function

接下来我们定义一个损失函数，来确定如何训练一个模型 $s(\mathbf{x}, \mathbf{w})$。首先我们有观测数据 $\mathbf{x}$ 遵循分布 $p(\mathbf{x})$，那么我们可以定义平方误差函数来衡量模型和真实得分函数之间的误差：

<div>

$$
\begin{align*}
J(\mathbf{w})=\frac{1}{2}\int \|s(\mathbf{x},\mathbf{w})-\nabla\ln p(\mathbf{x})\|^{2}p(\mathbf{x})\mathrm{d}\mathbf{x}\tag{3.2}
\end{align*}
$$

</div>

这个误差函数非常的直观，就是衡量了模型和真实的得分函数之间的平方误差。

> 模型结构有两种选择，一种是采样输入和输出形状一致的模型，这符合得分函数输入和输出形状一致的特点，第二种是采用单一输出的模型，但是需要经过两次反向传播。通常而言都会采用第一种方式，训练过程比较快。



#### 20.3.2 Modified score loss

公式 3.2 存在一个问题，就是我们并不知道原始数据的分布 $p(\mathbf{x})$ 是什么，我们只能获得有限的观测数据 <span> $\mathcal{D}=(\mathbf{x}_1,\dots,\mathbf{x}_N)$ </span>，虽然我们可以构建出经验分布

<div>

$$
\begin{align*}
p_{\mathcal{D}}(\mathbf{x})=\frac{1}{N}\sum_{n=1}^N\delta(\mathbf{x}-\mathbf{x}_n)\tag{3.3}
\end{align*}
$$

</div>

其中 $\delta$ 表示 Dirac delta 函数。但是公式 3.3 对于 $\mathbf{x}$ 而言是不可微的，为了解决这个问题，我们使用核密度估计来抹平 $\delta$ 函数。

<div>

$$
\begin{align*}
q_{\sigma}(\mathbf{z})=\int q(\mathbf{z}|\mathbf{x},\sigma)p(\mathbf{x})\mathrm{d}\mathbf{x}\tag{3.4}
\end{align*}
$$

</div>

其中 $q(\mathbf{z}|\mathbf{x},\sigma)$ 就是噪声核，通常采样高斯分布，$p(\mathbf{x})$ 则是原始数据分布，我们可以使用公式 3.3 的经验分布来代替。此时我们最小化的目标变成了如下形式

<div>

$$
J(\mathbf{w})=\frac{1}{2}\int \|s(\mathbf{z},\mathbf{w})-\nabla_{\mathbf{z}}q_{\sigma}(\mathbf{z})\|^2q_{\sigma}(\mathbf{z})\mathrm{d}\mathbf{z}\tag{3.5}
$$

</div>

我们将公式 3.4 和公式 3.3 代入，就可以得到如下形式

<div>

$$
J(\mathbf{w}) = \frac{1}{2N} \sum_{n=1}^{N} \int \left\| \mathbf{s}(\mathbf{z}, \mathbf{w}) - \nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}_n, \sigma) \right\|^2 q(\mathbf{z} \mid \mathbf{x}_n, \sigma) \, \mathrm{d}\mathbf{z} + \text{const}\tag{3.6}
$$

</div>

如果我们使用高斯噪声核 $q(\mathbf{z} \mid \mathbf{x}, \sigma) = \mathcal{N}(\mathbf{z} \mid \mathbf{x}, \sigma^2 \mathbf{I})$，我们可以直接写出得分函数的表达式

<div>

$$
\begin{align*}
\nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}, \sigma) = -\frac{1}{\sigma}\boldsymbol{\epsilon}\tag{3.7}
\end{align*}
$$

</div>

其中噪声 $\boldsymbol{\epsilon}=\mathbf{z}-\mathbf{x}$ 属于标准高斯分布。将公式 3.7 代回公式 3.5 或者 3.6 就会发现，损失函数实际上描述了模型输出和噪声之间的关系，这和扩散模型公式 2.17 如出一辙。

> 从 Score Matching 模型中采样可以使用 Langevin 动态采样。



#### 20.3.3 Noise Variance

上面虽然解决了模型训练的问题，但是仍然存在一些潜在的隐患。首先，如果数据分布落在一个低维流形上，我们又要面对数据点偏离流形的问题。其次损失函数中使用概率密度作为权重，这会导致在低概率密度区域的估计不准确。最后如果数据分布是由不相交的分布混合而成，那么 Langevin 采样会得到不准确的结果。

以上问题都可以通过选择一个方差足够大的噪声核来解决。但是过大的方差会扭曲原始的分布，一个平衡的方法是我们从小到大选择一系列方差取值，将方差也作为输入，并通过损失函数来进行训练。此时损失函数变成如下形式

<div>

$$
\begin{align*}
\frac{1}{2} \sum_{i=1}^{L} \lambda(i) \int \bigg\| s(\mathbf{z}, \mathbf{w}, \sigma_i^2) - \nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}_n, \sigma_i) \bigg\|^2 q(\mathbf{z} \mid \mathbf{x}_n, \sigma_i) \, d\mathbf{z}\tag{3.8}
\end{align*}
$$

</div>

其中 $\lambda$ 决定了每个不同大小的方差在损失函数中所占的比重。这个方法类似于扩散模型中训练不同的扩散步。训练完成后可以使用 langevin 采样依次从方差由大到小的模型中进行采样。

#### 20.3.4 Stochastic differential equations

前面我们看到扩散模型往往需要进行上千步的计算，那么如果我们进行无限步的计算呢？那考虑这样的极限我们必须保证噪声方差 <span> $\beta_t$ </span>随着步数增加而减小。这会引出随机微分方程，实际上扩散模型和分数匹配都是离散化的随机微分方程。我们可以将随机微分方程写成对向量的无穷小更新。

<div>

$$
\mathrm{d}\mathbf{z}=f(\mathbf{z},t)\mathrm{d}t+g(t)\mathrm{d}\mathbf{v}\tag{3.9}
$$

</div>

右边第一项称为漂移项，是确定性的。而第二项扩散项则是随机的。扩撒模型中的前向传播过程就可以写成上式的形式。

上述随机微分方程还有反向形式

<div>

$$
\begin{align*}
d\mathbf{z} = \{ f(\mathbf{z}, t) - g^2(t) \nabla_{\mathbf{z}} \ln p(\mathbf{z}) \} \, dt + g(t) \, d\mathbf{v}\tag{3.10}
\end{align*}
$$

</div>

不难发现其形式类似于分数匹配模型。

### 20.4 Guided diffusion

在前面我们讨论的都是无条件的扩散模型，即没有任何输入来控制建模过程，但是在实际使用中我们往往希望通过某些信息来控制生成过程，例如我希望只生成某一类别的图像。最简单的方法就是将条件作为模型的输入，构建 <span> $g(\mathbf{z},\mathbf{w},t,\mathbf{c})$ </span>，使用数据对 <span> $\{\mathbf{x}_n,\mathbf{c}_n\}$ </span>进行训练。但是这种方法的缺点是模型往往会忽略类别变量，因此我们需要一种方法来控制类别在训练过程中的比重。根据是否需要训练额外的分类模型我们可以将控制方法分为分类引导和无分类引导。



#### 20.4.1 Classifier guidance

假设我们已经有了一个训练好的分类模型 $p(\mathbf{c}|\mathbf{x})$ ，我们可以使用贝叶斯定理写出 $p(\mathbf{x}|\mathbf{c})$ 并求它的得分函数

<div>

$$
\begin{align*}
\nabla_{\mathbf{x}} \ln p(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \ln \left\{ \frac{p(\mathbf{c} \mid \mathbf{x}) p(\mathbf{x})}{p(\mathbf{c})} \right\} = \nabla_{\mathbf{x}} \ln p(\mathbf{x}) + \nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})\tag{4.1}
\end{align*}
$$

</div>

其中右侧第二项可以将去噪过程推向最大化类别概率 $\mathbf{c}$ 的方向。如果我们引入一个权重系数来控制分类损失的占比

<div>

$$
\begin{align*}
\text{score}(\mathbf{x}, \mathbf{c}, \lambda) = \nabla_{\mathbf{x}} \ln p(\mathbf{x}) + \lambda \nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})\tag{4.2}
\end{align*}
$$

</div>

系数 $\lambda$ 越大表示模型越注重分类概率，但是这也容易导致模型倾向于生成使得分类概率最大化的“简单”结果，而丧失一定的多样性。此外这种方法还要求我们有一个已经训练好的分类模型。

#### 20.4.2 Classifier-free guidance

我们重写公式 4.2，将公式 4.1 中的分类概率代入 4.2 中。

<div>

$$
\begin{align*}
\text{score}(\mathbf{x}, \mathbf{c}, \lambda) = \lambda \nabla_{\mathbf{x}} \ln p(\mathbf{x} \mid \mathbf{c}) + (1 - \lambda) \nabla_{\mathbf{x}} \ln p(\mathbf{x})\tag{4.3}
\end{align*}
$$

</div>

当权重系数 $\lambda$ 在 0-1 之间时，表示条件对数密度和无条件对数密度的凸组合。当权重系数超过一时，无条件对数密度将会变为负值，此时模型会倾向于尽可能不生成无条件控制的数据，从而提高生成符合条件的数据的概率。此外，我们通过在训练过程中按照一定的概率将条件变量设为空值（即无条件），对一部分样本的条件置零，就可以只需要一个模型来同时建立条件密度和无条件密度。


## Appendix

### Appendix 1

#### Appendix 1.1
已知
<span>

$$
\begin{align*}
\mathbf{z}_1&=\sqrt{1-\beta_1}\mathbf{x}+\sqrt{\beta_1}\boldsymbol{\epsilon}_1\\\\
\mathbf{z}_t &= \sqrt{1-\beta_t}\mathbf{z}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t 
\end{align*}
$$

</span>
则有 
<span>

$$
\begin{align*}
    \mathbf{z}_2 &= \sqrt{1-\beta_2}\mathbf{z}_1+\sqrt{\beta_2}\epsilon_2\\\\
    &= \sqrt{1-\beta_2}\sqrt{1-\beta_1}\mathbf{x}+\sqrt{1-\beta_2}\sqrt{\beta_1}\boldsymbol{\epsilon}_1+\sqrt{\beta_2}\boldsymbol{\epsilon_2}\\\\
    &= \sqrt{(1-\beta_2)(1-\beta_1)}\mathbf{x}+\sqrt{(1-\beta_2)\beta_1}\boldsymbol{\epsilon}_1+\sqrt{\beta_2}\boldsymbol{\epsilon_2}
\end{align*}
$$

</span>
其中 <span> $\boldsymbol{\epsilon}_1,\boldsymbol{\epsilon}_2$ </span> 都是服从标准正态分布的噪声项，根据正态分布的性质，两个均值相同的分布相加后其方差也是两者的相加。
<span>

$$
\begin{align*}
    \mathbf{z}_2 &= \sqrt{(1-\beta_2)(1-\beta_1)}\mathbf{x}+\sqrt{(1-\beta_2)\beta_1}\boldsymbol{\epsilon}_1+\sqrt{\beta_2}\boldsymbol{\epsilon_2}\\\\
    &= \sqrt{(1-\beta_2)(1-\beta_1)}\mathbf{x} + \sqrt{(1-\beta_2)\beta_1+\beta_2}\boldsymbol{\epsilon}\\\\
    &=\sqrt{(1-\beta_2)(1-\beta_1)}\mathbf{x} + \sqrt{(1-\beta_1)(1-\beta_2)}\boldsymbol{\epsilon}
\end{align*}
$$

</span>
依次类推不难得出根号下的因子会逐渐累乘。我们令 <span> $\alpha_t=\prod_{1}^t(1-\beta_t)$ </span>，就可以将 <span> $\mathbf{z}_t$ </span> 及其分布化简为公式 1.6 和 1.7。

#### Appendix 1.2
> 我需要声明一点的是，下面的推导过程是 AI 写的，说实话我真的不知道怎么有人能推出这么一长串东西。

这里我们给出公式1.13到1.16的详细推导过程。

我们从公式1.13开始：

<span>

$$
\begin{align*}
    q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) = \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1}|\mathbf{x})}{q(\mathbf{z}_t|\mathbf{x})} 
\end{align*}
$$

</span>

首先，根据公式1.4，我们知道：

<span>

$$
\begin{align*}
q(\mathbf{z}_t|\mathbf{z}_{t-1}) &= \mathcal{N}(\mathbf{z}_t|\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I})
\end{align*}
$$

</span>

根据公式1.6，我们有：

<span>

$$
\begin{align*}
q(\mathbf{z}_{t-1}|\mathbf{x}) &= \mathcal{N}(\mathbf{z}_{t-1}|\sqrt{\alpha_{t-1}}\mathbf{x}, (1-\alpha_{t-1})\mathbf{I}) \\\\
q(\mathbf{z}_t|\mathbf{x}) &= \mathcal{N}(\mathbf{z}_t|\sqrt{\alpha_t}\mathbf{x}, (1-\alpha_t)\mathbf{I})
\end{align*}
$$

</span>

现在，我们需要将这三个高斯分布代入公式1.13。首先，我们回顾一下多元高斯分布的概率密度函数：

<span>

$$
\begin{align*}
\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
\end{align*}
$$

</span>

代入我们的三个高斯分布，并将它们相除，我们需要计算：

<span>

$$
\begin{align*}
q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) &= \frac{\mathcal{N}(\mathbf{z}_t|\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I}) \cdot \mathcal{N}(\mathbf{z}_{t-1}|\sqrt{\alpha_{t-1}}\mathbf{x}, (1-\alpha_{t-1})\mathbf{I})}{\mathcal{N}(\mathbf{z}_t|\sqrt{\alpha_t}\mathbf{x}, (1-\alpha_t)\mathbf{I})}
\end{align*}
$$

</span>

为了简化计算，我们首先关注分子中的两个高斯分布的乘积。当我们将两个高斯分布相乘时，结果仍然是一个高斯分布（可能需要归一化）。

让我们先计算分子：

<span>

$$
\begin{align*}
&\mathcal{N}(\mathbf{z}_t|\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I}) \cdot \mathcal{N}(\mathbf{z}_{t-1}|\sqrt{\alpha_{t-1}}\mathbf{x}, (1-\alpha_{t-1})\mathbf{I}) \\\\
&\propto \exp\left(-\frac{1}{2\beta_t}(\mathbf{z}_t-\sqrt{1-\beta_t}\mathbf{z}_{t-1})^T(\mathbf{z}_t-\sqrt{1-\beta_t}\mathbf{z}_{t-1})\right) \cdot \\\\
&\quad \exp\left(-\frac{1}{2(1-\alpha_{t-1})}(\mathbf{z}_{t-1}-\sqrt{\alpha_{t-1}}\mathbf{x})^T(\mathbf{z}_{t-1}-\sqrt{\alpha_{t-1}}\mathbf{x})\right)
\end{align*}
$$

</span>

将指数项展开并合并同类项后，我们可以得到一个新的二次型表达式。经过复杂的代数运算，这个二次型可以重新表示为关于 $\mathbf{z}_{t-1}$ 的高斯分布：

<span>

$$
\begin{align*}
q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) &\propto \exp\left(-\frac{1}{2\sigma_t^2}(\mathbf{z}_{t-1} - \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t))^T(\mathbf{z}_{t-1} - \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t))\right)
\end{align*}
$$

</span>

这表明 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$ 是一个均值为 $\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t)$ 且协方差为 $\sigma_t^2\mathbf{I}$ 的高斯分布，即：

<span>

$$
\begin{align*}
q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) &= \mathcal{N}(\mathbf{z}_{t-1}|\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t), \sigma_t^2\mathbf{I})
\end{align*}
$$

</span>

这就是公式1.14。

现在我们需要具体推导 $\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t)$ 和 $\sigma_t^2$ 的表达式，也就是公式1.15和1.16。

通过进一步展开指数项并收集关于 $\mathbf{z}_{t-1}$ 的二次项和一次项，我们可以得到：

<span>

$$
\begin{align*}
&\exp\left(-\frac{1}{2\beta_t}(\mathbf{z}_t-\sqrt{1-\beta_t}\mathbf{z}_{t-1})^T(\mathbf{z}_t-\sqrt{1-\beta_t}\mathbf{z}_{t-1})\right) \cdot \\\\
&\exp\left(-\frac{1}{2(1-\alpha_{t-1})}(\mathbf{z}_{t-1}-\sqrt{\alpha_{t-1}}\mathbf{x})^T(\mathbf{z}_{t-1}-\sqrt{\alpha_{t-1}}\mathbf{x})\right) \\\\
&\propto \exp\left(-\frac{1}{2}\left(\frac{(1-\beta_t)}{\beta_t}\mathbf{z}_{t-1}^T\mathbf{z}_{t-1} - \frac{2\sqrt{1-\beta_t}}{\beta_t}\mathbf{z}_t^T\mathbf{z}_{t-1} + \frac{1}{1-\alpha_{t-1}}\mathbf{z}_{t-1}^T\mathbf{z}_{t-1} - \frac{2\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}\mathbf{x}^T\mathbf{z}_{t-1}\right)\right)
\end{align*}
$$

</span>

收集 $\mathbf{z}_{t-1}$ 的二次项系数，得到精度矩阵（协方差矩阵的逆）：

<span>

$$
\begin{align*}
\frac{1}{\sigma_t^2}\mathbf{I} &= \frac{(1-\beta_t)}{\beta_t}\mathbf{I} + \frac{1}{1-\alpha_{t-1}}\mathbf{I} \\\\
&= \left(\frac{(1-\beta_t)}{\beta_t} + \frac{1}{1-\alpha_{t-1}}\right)\mathbf{I}
\end{align*}
$$

</span>


因此：

<span>

$$
\begin{align*}
\sigma_t^2 &= \frac{1}{\frac{(1-\beta_t)}{\beta_t} + \frac{1}{1-\alpha_{t-1}}} \\\\
&= \frac{\beta_t(1-\alpha_{t-1})}{(1-\beta_t)(1-\alpha_{t-1}) + \beta_t} \\\\
\end{align*}
$$

</span>

利用 $\alpha_t = \alpha_{t-1}(1-\beta_t)$，我们可以得到：

<span>

$$
\begin{align*}
1-\alpha_t &= 1-\alpha_{t-1}(1-\beta_t) \\
&= 1-\alpha_{t-1}+\alpha_{t-1}\beta_t \\
&= (1-\alpha_{t-1}) + \alpha_{t-1}\beta_t
\end{align*}
$$

</span>

代入上面的 $\sigma_t^2$ 表达式：

<span>

$$
\begin{align*}
\sigma_t^2 &= \frac{\beta_t(1-\alpha_{t-1})}{(1-\beta_t)(1-\alpha_{t-1}) + \beta_t} \\\\
&= \frac{\beta_t(1-\alpha_{t-1})}{(1-\alpha_{t-1}) - \beta_t(1-\alpha_{t-1}) + \beta_t} \\\\
&= \frac{\beta_t(1-\alpha_{t-1})}{(1-\alpha_{t-1}) + \beta_t\alpha_{t-1}} \\\\
&= \frac{\beta_t(1-\alpha_{t-1})}{1-\alpha_{t-1}+\alpha_{t-1}\beta_t} \\\\
&= \frac{\beta_t(1-\alpha_{t-1})}{1-\alpha_t}
\end{align*}
$$

</span>

这就是公式1.16。

现在，我们来推导均值 $\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t)$。收集 $\mathbf{z}_{t-1}$ 的一次项系数：

<span>

$$
\begin{align*}
\frac{1}{\sigma_t^2}\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) &= \frac{\sqrt{1-\beta_t}}{\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}\mathbf{x}
\end{align*}
$$

</span>

因此：

<span>

$$
\begin{align*}
\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) &= \sigma_t^2 \left( \frac{\sqrt{1-\beta_t}}{\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}\mathbf{x} \right) \\\\
&= \frac{\beta_t(1-\alpha_{t-1})}{1-\alpha_t} \left( \frac{\sqrt{1-\beta_t}}{\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}\mathbf{x} \right) \\\\
&= \frac{(1-\alpha_{t-1})\sqrt{1-\beta_t}}{1-\alpha_t}\mathbf{z}_t + \frac{\beta_t\sqrt{\alpha_{t-1}}}{1-\alpha_t}\mathbf{x}
\end{align*}
$$

</span>

这就是公式1.15。

综上，我们已经完成了公式1.13到1.16的推导过程。
