+++

title = '什么是 DDIM'
date = '2025-08-12T15:26:41+08:00'
draft = false
outdatedInfoWarning = false
collections = []
tags = ["Deep Learning", "Diffusion Models"]
categories = ["Deep Learning", "Diffusion Models"]
summary = "什么是 DDIM？DDIM 是如何实现采样加速的？"

+++

## 回顾 DDPM

在讨论 DDIM 之前，我们先回顾一下 DDPM 的相关内容。

DDPM 的目的是建模 $p(\mathbf{x})$ 从而实现数据生成，其具体的步骤是首先对 $\mathbf{x}$ 进行多步加噪，加噪公式如下：
$$
\begin{align*}
q(\mathbf{x}_t|\mathbf{x}_{t-1})&=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})\tag{1}\\\\
q(\mathbf{x}_t|\mathbf{x}_0)&=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})\tag{2}
\end{align*}
$$
其中 $\alpha_t=1-\beta_t,\bar{\alpha}_t=\prod_{s=1}^t\alpha_s$。

由于 $\beta_t$ 随着 $t$ 的增大而不断增大，因此经过多步加噪后，$\mathbf{x}_t$ 基本就是一个高斯噪声。然后我们希望模型可以学习到从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t-1}$ 的去噪过程，从而通过多步迭代去噪，实现从高斯噪声 $\mathbf{x}_t$ 到观测数据 $\mathbf{x}_0$ 过程。一旦这个模型训练好，我们就可以直接从高斯噪声中采样，然后经过迭代去噪实现数据生成。

为了实现这个目标，我们建立一个带参数的模型 $p_\theta(x)$。由于加噪过程和去噪过程都是马尔科夫链，所以我们可以定义模型为如下形式：
$$
\begin{align*}
p(\mathbf{x}_{0:T})&=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})\tag{3}\\\\
p_\theta(\mathbf{x}_t|\mathbf{x}_{t-1})&=\mathcal{N}(\mathbf{x}_{t-1},\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\Sigma_\theta(\mathbf{x}_t,t))\tag{4}
\end{align*}
$$

我们的最终目标是建模 $p(\mathbf{x})$，根据马尔可夫链的性质，我们可以写出如下形式的对数似然函数：


$$
\begin{align*}
\log p(\mathbf{x})&=\log \int p(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}\\\\
&=\dots\\\\
&=\underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)]}_{\text{reconstruction}}
+\underbrace{\mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_-)}[D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0)||p(\mathbf{x}_T))]}_{\text{prior mathching}}\\\\
&-\underbrace{\sum_{t=1}^{T-1}\mathbb{E}_{q(\mathbf{x}_T|\mathbf{x}_0)}[D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)||p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))]}_{\text{consistency}}\tag{5}
\end{align*}
$$


>  这里省略了其中的详细推导过程， 如果有兴趣可以查看 DDPM 原论文或者阅读 *Understanding diffusion models: a unified perspective* 这篇论文。

在上述对数似然中，第一项表示了去噪过程的最后一步，也就是生成数据的这一步，因此被称为重构项，在高斯分布的先验下，其表达式可以化简为 MSE Loss。第二项是先验匹配项，通过 KL 散度约束了加噪之后的数据应该是一个标准高斯噪声。第三项则是一致项，让去噪模型接近加噪的逆向过程。

我们的重点在于第三项，这里<span> $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$</span> 拟合的是 <span>$q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$</span>，而不是 <span>$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$</span>，这是因为后者实际上就是我们希望真正得到的去噪过程，但是它是未知的。所以这里我们退一步，引入 <span>$\mathbf{x}_0$ </span>之后，<span> $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$</span> 是一个已知的过程，其表达式如下：

$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}\tag{6}
\end{align*}
$$
公式（6）中全部分布都是已知的，我们展开并整理，发现其依然是一个高斯分布，并且可以得到这个分布的均值和方差分别如下：
$$
\begin{align*}
\mathbf{x}_{t-1}:\text{mean}&=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t\mathbf{x}_0+\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1}\mathbf{x}_t)}{1-\bar\alpha_t}\tag{7}\\\\
\mathbf{x}_{t-1}:\text{var}&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \mathbf{I}\tag{8}
\end{align*}
$$
如果我们用公式（2）的表达式 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ 来替换掉均值中的 $\mathbf{x}_0$ 则可以得到如下形式的均值：
$$
\begin{align*}
\mathbf{x}_{t-1}:\text{mean}=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon)\tag{9}
\end{align*}
$$
此时我们可以发现，对于 $\mathbf{x}_{t-1}$ 而言，其方差是固定的，均值也只有 $\epsilon$ 是未知的。因此我们的模型可以只预测去噪过程中的噪音即可。一旦我们完成了噪声的预测，那么公式（9）就可以用来实现去噪操作，实际上公式（9）和 DDPM 论文中所使用的采样公式是完全一致的。

因此我们的模型可以写成 $\epsilon_\theta(\mathbf{x}_t,t)=\epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)$，此时我们既然只需要预测噪声，那么损失函数可以直接写成 $||\epsilon-\epsilon_\theta(\mathbf{x}_t,t)||^2$。完成训练后，采样公式就是 $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t))$。我们首先采样一个高斯噪声，应用采样公式经过足够多步迭代后就可以得到最终的结果。

实际上在 DDPM 论文中，采样过程额外添加了一个噪声项，即 $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t))+\sigma_t \mathbf{z}$，其中 $\mathbf{z}$ 是一个标准高斯噪声，方差系数 $\sigma_t$ 控制噪声大小。这一项是为了模拟加噪过程中，加噪操作的随机性。并且在生成时引入一定的多样性。此外在 DDPM 的实践中发现，如果采样过程完全不加噪声，其生成结果反而很差。

> 在采样过程中，DDPM 中尝试了两种方差系数 $\sigma_t$ 的取值，第一种是由公式（8）得到系数 $\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t $，这是效果最好的。第二种则是直接令方差系数和加噪过程一致，即 $\beta_t$，效果同样很好。

## DDIM 加速采样

在 DDPM 中，我们推导似然函数时使用到了 $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$，但是在最终的损失函数和采样函数中，我们只用到了 $q(\mathbf{x}_t|\mathbf{x}_0)$ 和 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$。而 DDPM 受限于马尔可夫性质，其加噪去噪过程都必须逐步进行，这一性质是由于 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 所限制的。但是既然最终我们使用的损失函数和采样函数中都没有涉及这一性质，那么是不是有办法去掉这个约束，实现跳步采样呢？

如果采样过程不是马尔可夫性的，那么 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 只需要满足如下性质：
$$
\begin{align*}
\int q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)q(\mathbf{x}_t|\mathbf{x}_0)d\mathbf{x}_t=q(\mathbf{x}_{t-1}|\mathbf{x}_0)\tag{10}
\end{align*}
$$
为了保持加噪过程不变，我们令 $q(\mathbf{x}_t|\mathbf{x}_0)$ 和 $q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ 保持和 DDPM 一样的过程。通过待定系数法求解公式（10）可以得到如下结果：
$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1};\frac{\sqrt{(1-\bar{\alpha}_{t-1}) - \sigma_t^2}}{\sqrt{1-\bar{\alpha}_t}} \mathbf{x}_t + \left( \sqrt{\bar{\alpha}_{t-1}} -\frac{\sqrt{\bar{\alpha}_t} \sqrt{(1-\bar{\alpha}_{t-1}) - \sigma_t^2}}{\sqrt{1-\bar{\alpha}_t}} \right) \mathbf{x}_0,\sigma_t^2 \mathbf{I}\right)\tag{11}
\end{align*}
$$

> 求解过程可以参考以下文章，需要注意本文的符号和下面所示文章中的符号不一致。
>
> 苏剑林. (Jul. 27, 2022). 《生成扩散模型漫谈（四）：DDIM = 高观点DDPM 》[Blog post]. Retrieved from https://kexue.fm/archives/9181

对比 DDPM 中推导出的均值公式（7）和方差公式（8），可以发现公式（11）中多出现了一个 $\sigma_t$ ，这是求解过程中使用配方法引入的自由变量，这允许我们在采样过程中使用不同的方差。

> 在 DDPM 推导中，我们可以发现采样过程的方差是有闭式解的，在 DDIM 推导中则完全没有限制。

对于公式（11），我们同样使用 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ 替换 $\mathbf{x}_0$，化简后得到：
$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \epsilon) = \mathcal{N}\left(\mathbf{x}_{t-1};\ \sqrt{\frac{\bar{\alpha}_{t-1}}{\bar{\alpha}_t}} \mathbf{x}_t + \left( \sqrt{(1-\bar{\alpha}_{t-1}) - \sigma_t^2} - \sqrt{\frac{\bar{\alpha}_{t-1}(1-\bar{\alpha}_t)}{\bar{\alpha}_t}} \right) \epsilon,\ \sigma_t^2 \mathbf{I}\right)\tag{12}
\end{align*}
$$
同样的，我们使用模型来预测误差，可以得到采样结果为
$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \epsilon) = \mathcal{N}\left(\mathbf{x}_{t-1};\ \sqrt{\frac{\bar{\alpha}_{t-1}}{\bar{\alpha}_t}} \mathbf{x}_t + \left( \sqrt{(1-\bar{\alpha}_{t-1}) - \sigma_t^2} - \sqrt{\frac{\bar{\alpha}_{t-1}(1-\bar{\alpha}_t)}{\bar{\alpha}_t}} \right) \epsilon_\theta(\mathbf{x}_t,t),\ \sigma_t^2 \mathbf{I}\right)\tag{13}
\end{align*}
$$
由于我们抛弃了马尔可夫性质，因此其中 t 完全可以任意取值，从而实现加速采样。

> 需要注意，修改了 t 的含义后，其中各个变量的含义也发生了改变，例如我们用 $\tau$ 代替 t，那么  $\alpha_\tau $ 的含义也会发生变化。当然，在实际代码实现中，我们直接令 $\tau$ 等于我们希望跳到的 t 即可。

知道了采样均值的计算过程，那么该如何确定采样时的方差呢？原则上而言，方差系数 $\sigma_t$ 并没有约束，原作尝试了几种设置。其中如果直接照搬 DDPM 的方差设置 $\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t $，并添加一个权重系数用于从零开始调整其大小 $\eta \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t $，当 $\eta=1$ 时实际上退化为原始 DDPM，当 $\eta=0$ 时采样过程不再具有随机性，而是一个确定性采样，这也是称其为 DDIM 的原因。

此外，当方差为零时，由于采样变成了确定性变化，因此我们可以直接对噪声向量进行插值，从而实现生成结果的变换。而原始 DDPM 无法直接对噪声空间进行插值操作。

> 如果使用 DDPM 的另一个方差配置，即采样方差和加噪方差相同，DDIM 的采样表现会很糟糕。
