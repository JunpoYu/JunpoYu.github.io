+++
title = 'Denoising Diffusion Probabilistic Models'
date = 2025-06-25T19:21:01+08:00
draft = true

+++

本文内容来自 [Denoising Diffusion Probablilistic Models](http://arxiv.org/abs/2006.11239)（DDPM），也是 diffusion model 发扬光大的起点。

## 论文部分

### Introductin

> 这一章节主要介绍了一下 DDPM 的基本理念，并且说明了本文的成果。

扩散模型是一个参数化的马尔科夫链，通过变分推断进行训练，模型的目标是在经过有限步的处理后生成和观测数据相符合的样本。而这个马尔科夫链实际上是逆转了一个扩散过程，所谓扩散过程就是逐步向一个样本中不断地添加噪声，知道样本信息完全丢失，实际上扩散过程也是一个马尔科夫链，只是这两个马尔科夫链所代表的是反向的两个过程。

由于加噪过程每次添加的都是很小的高斯噪声，所以反向的去噪过程也可以表示为一个条件高斯分布，从而可以方便的进行参数化。

### Background



## 代码实现

该部分代码参考[Annotated Research Paper Implementations](https://nn.labml.ai/diffusion/ddpm/index.html) 提供的代码示例。



## 附录

