---
title: '深度学习之分类问题'
date: 2025-02-15T09:54:01+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - machine learning
---



 

## 一、构建一个分类模型的三种方法

1. 判别函数：直接建模一个函数将输入映射到分类，例如支持向量机。
2. 判别式概率模型：建模一个条件概率模型 $p(C_k|x)$ ，根据输出的概率选择其分类结果，通常概率最高的类别就是分类类别。
3. 生成式概率模型：首先建模条件概率 $p(x|C_k)$，然后根据先验概率 $p(C_k)$ 通过贝叶斯公式 $p(C_k|x)=\frac{p(x|C_k)p(C_k)}{p(x)}$ 得出后验概率分布，使用后验概率分布进行分类。

> $x$ 表示输入向量，$C_k$ 表示 k 个分类类别。
>
> 在生成式概率模型中，由于建模了 $p(x|C_k)$ ，所以可以使用模型生成新的样本 $x$ ，因此称之为生成式模型。



## 二、判别函数

### 1. 二分类问题

对于一个二分类问题构建决策函数：
$$
y(x)=\textbf w^Tx+\textbf w_0
$$
 则决策面就是 $y(x)=0$ 所代表的超平面，通过该决策面作为边界进行分类。

在这个决策函数中，其参数的含义如下：

1. $\textbf w$ 决定了决策面的方向，因为 $\textbf w$ 一定与决策面正交。
2. $\textbf w_0$ 决定了决策面的偏移量。
3. $y(x)$ 的值与其代表的点与决策面的距离成正比，正负则代表了在决策面的哪一侧。


### 2. 多分类问题

在多分类问题中，如果照搬二分类中超平面的方法，会导致决策空间中存在模棱两可的区域。无论是 one-vs-the-rest 还是 one-vs-one 方法都不能避免。

> one-vs-the-rest 表示使用一个决策面，将当前类别和其他所有类别区分，通过 k 个决策面实现 k 分类。
>
> one-vs-one 表示使用一个决策面，将当前类别和另一个类别区分，通过 k(k-1) 个决策面实现 k 分类。

但是我们仍然可以使用判别函数，考虑由 k 个线性判别函数组成的 k 分类器。
$$
y_k(x)=\textbf w_k^T(x) + \textbf w_{k0}
$$
k 表示 k 个判别函数。此时将 x 的类别定义为使 $y_k(x)$ 最大的类别，即当 $y_k(x)>y_j(x),j\neq k$ 时，认为 x 输入类别 k。

此时类别 $C_k$ 和类别 $C_j$ 之间的超平面由
$$
y_k(x)-y_j(x)=(\textbf w_k^T-\textbf w_j^T)x+(\textbf w_{k0}-\textbf w_{j0})=0
$$
给出。这种方法建立起来的判别区域一定是**单连通且凸**的（singly connected and convex）。

> 所谓单连通是指区域内没有任何空洞或者开口，即整个区域是一个有限整体。例如一个实心圆内部的区域是单连通的，但是内部加一个同心圆，则外部的圆不再是单连通的。
>
> 所谓凸是指，在区域内任取两点，则两点连线之间的任何点也在这个区域内。

### 3. one-hot 编码

对于多变量分类的目标值，我们通常采用 one-hot 编码来表示其所属的类别。这种编码的含义是，输入变量分类到正确类别的概率为 1，分类到其他类别的概率为 0。

使用 one-hot 编码时同样可以使用最小二乘法进行建模。例如对于以下判别函数：
$$
y_k(x)=\textbf w_k^Tx+\textbf w_{k0}
$$
我们采用增广矩阵 $\tilde W_k^T=(w_{k0},w_k^T),\tilde x=(1,x)$ 来简化表示为
$$
y_k(x)=\tilde W^T\tilde x
$$
对于训练数据 $\{x_n,tn\}$ 表示为 ${X,T}$，其中 $X$ 每一行代表一个输入 $x$ ，$T$ 每一行代表一个目标 ont-hot $t$，构建平方和误差函数
$$
E_D(\tilde W)=\frac{1}{2}\{ (\tilde W \tilde X - T)^T(\tilde W \tilde X-T)\}
$$
使用最小二乘法最小化该误差函数，令其为零，则有
$$
\tilde W =\tilde X^{\dagger} T
$$
其中 $\tilde X^{\dagger}$ 表示 $\tilde X$ 的伪逆。

可得
$$
y(x)=\tilde W^T\tilde X=\tilde X^{\dagger}T\tilde x
$$
从而得到判别函数。

> 在最小二乘法中有一个特性，即若 t 满足 $a_n^T t_n+b=0$ ,其中 $a_n^T,b$ 都是常数向量，则有 $a_n^Ty(x)+b=0$。这可以证明当 $t_n$ 作为 one-hot 编码表示的概率之和为 1 时，则使用判别公式得到的概率分布之和同样为 1，但是不能保证每个概率值都在 0~1 之间。

需要注意的是，由于最小二乘是噪声符合高斯分布时极大似然的特殊形式，所以如果噪声不符合高斯分布，则最小二乘会得到糟糕的结果。另外平方误差函数对离群值非常敏感，因此稳定性较低。



## 三、生成式概率模型

假如我们希望建立一个参数化的生成式概率模型，则我们需要建模类条件概率 $p(x|C_k)$ 并根据贝叶斯公式来计算后验概率。假设类条件概率服从高斯分布，则对于一个二分类问题，我们定义两个类别具有相同的协方差，同时我们有一个数据集 $\{x_n,t_n\},n=1,2...N$ ，以 $t_n=1$ 表示其属于类别 $C_1$，$t_n=0$ 表示属于类别 $C_2$ ，定义先验概率 $p(C_1)=\pi,p(C_2)=(1-\pi)$，则对于任一样本，有
$$
p(x_n,C_1)=p(x|C_1)p(C_1)=\pi\mathcal N(x_n|\mu_1,\sum) \newline
p(x_n,C_2)=p(x|C_2)p(C_2)=(1-\pi)\mathcal N(x_n|\mu_1,\sum)
$$
我们可以很自然的建立起似然函数
$$
p(t,X|\pi,\mu_1,\mu_2,\sum)=\prod_{n=1}^N[\pi\mathcal N(x_n|\mu_1,\sum)]^{t_n}[(1-\pi)\mathcal N(x_n|\mu_2,\sum)]^{1-t_n}
$$
转为对数似然后对其中的各参数求偏导，令偏导等于零就可以得到各参数的解，这个方法同样可以扩展到多类别分类中。

## 四、判别式概率模型

在前面的描述中，我们总是假设某个分布服从高斯分布，这其实是为了方便我们理解，实际上真实的数据分布不一定是遵循某种特定的分布类型。

因此当我们假定的分布和真实的分布近似程度比较低时，基于类条件概率的方法可能产生比较差的结果，此时可以考虑判别式概率模型，即直接使用极大似然确定广义线性模型的概率分布，直接拟合后验概率，这种做法可以大大降低需要学习的参数的数量，同时有助于解决假定分布与真实分布不近似的问题。

### 逻辑回归

对于二分类问题，在一般的假设下，后验概率可以写成如下形式：
$$
p(C_1|\phi)=\sigma(w^T\phi)
$$
其中 $$\phi$$ 表示基函数向量，用以对原始输入空间 $$x$$ 进行变换，通过基函数变化可以将某些线性不可分的输入空间转为线性可分。$$\sigma$$ 表示 logistic sigmoid，这种形式也叫做逻辑回归，但其实这是一个分类模型，并不是回归模型。

对于 M 维输入空间，这个模型只有 M 个可调整参数，如果我们采用高斯分布的类条件概率密度，并使用极大似然估计，则需要 $$M(M+5)/2+1$$ 的参数量。

现在让我们使用最大似然来求解逻辑回归模型，对于一个数据集 $$\{\phi_n,t_n\}$$，其中 $$\phi_n=\phi(x_n)$$ $$t_n\in\{0,1\}$$ ，似然函数可以写作：
$$
p(\textbf t|w)=\prod_{n=1}^Ny_n^{t_n}\{1-y_n\}^{1-t_n}
$$
其中 $$\textbf t=(t_1,t_2...t_N)^T$$ 并且 $$y_n=p(C_k|\phi_n)$$ ，构建负对数似然，我们就可以得到交叉熵误差函数：
$$
E(\textbf w)=-\ln p(\textbf t|\textbf w)=-\sum_{n=1}^N{t_n\ln y_n+(1-t_n)\ln (1-y_n)}
$$
其中 $$y _n=\sigma(a_n)$$ $$a_n=\textbf w^T\phi_n$$ ，对该误差函数做关于 $$\textbf w$$ 的偏导，并且使用 logistic sigmoid 函数的简化求导形式 $$\frac{\mathrm{d} \sigma}{\mathrm{d} a}=\sigma(1-\sigma)$$ ，我们得到如下公式：
$$
\nabla E(\textbf w)=\sum_{n=1}^N(y_n-t_n)\phi_n
$$
令其等于零即可求解参数。需要注意的是，由于引入了非线性函数，这个等式不在具有闭式解，一种求解方法是使用随机梯度下降，但是这个极大似然等式实际上只有一点非线性，所以误差函数实际上是一个关于参数 $$\textbf w$$ 的凸函数，这就允许我们使用一种叫做迭代重加权最小二乘法的方法来求解。但是这种方法不适用深度神经网络，所以在深度神经网络中我们还是以梯度下降为主。

另外需要注意的是，如果数据集是线性可分的，那么极大似然会产生严重的过拟合，这是因为 $$\sigma$$ 函数的性质，如果数据集本身线性可分，则这个函数会将其参数趋于无限，从而使计算出的概率无限趋向0或1，以实现完美分类。而对于 $$\sigma(\textbf w^T\phi)=0.5$$ 即 $$\textbf w^T\phi=0$$ 的超平面，由于参数本身趋于无穷，因此分类边界处会变得非常陡峭。这个问题可以通过添加正则化系数来缓解。

对于多类别逻辑回归，可以使用线性代数的相关知识建立同样的求解过程，此处不再赘述。像这样的线性分类模型可以使用单层神经网络表示。

> 还有一种 probit regression，即采用 probit 函数作为激活函数的广义线性模型，这种模型的表现和 logistic regression 类似，但是其对于离群值更加敏感。

## 规范链接函数

广义线性模型被定义为：假设目标变量 t 的条件分布属于指数族分布，并且通过一个激活函数将线性组合 $$\textbf w^T\phi$$ 映射到目标变量的期望值。

所谓规范链接函数是就是指按照某种特定规则选定的激活函数，这种激活函数可以将广义线性模型的误差函数的梯度简化为预测误差和特征向量的乘积。即使对参数的梯度呈现如下形式：
$$
\nabla E(\textbf w)=\sum_{n=1}^N(y_n-t_n)\phi_n
$$
sigmoid 函数和 softmax 函数都是规范链接函数。

## 类别不平衡问题

某些时候我们使用的训练数据可能存在类别不平衡的问题，例如二分类问题中正样本占了 99%，这种情况会导致模型学习到“平凡解”，即模型将所有输入全部划分为正样本即可得到 99% 的准确率。很明显这样模型无法学习到负样本的分布情况，难以泛化到真实的分布中。

此时通常可以采用认为平衡数据集的方式，手动使数据集中的各样本占比平衡，这有助于模型学习到少数样本的特征。但这毫无以为会带来另一个问题，即模型是在一个调整过的数据集上训练出来的，根据贝叶斯公式
$$
p(C_k|x)=\frac{p(x|C_k)p(C_k)}{p(x)}
$$
经过调整后的数据集其先验概率 $p(C_k)$ 已经发生了改变，因此从该数据集中得到的后验概率 $p(C_k|x)$ 不能直接用于分类，而是要经过如下变化
$$
p_{real}(C_k|x)=\frac{p_{real}(C_k)}{p_{balanced}(C_k)}p_{balanced}(C_k|x)
$$
这是因为当模型建立其类别条件概率 $p(x|C_k)$ 之后，后验概率就和先验概率成正比了。

> 此处仍然存在一个问题，即我们假定模型从平衡数据中学习到的类别条件概率 $p(x|C_k)$ 是正确的，因此这个调整之后的后验概率才是正确的，但是为什么认为调整过的数据集训练出的 $p(x|C_k)$ 是正确的呢？

## 广义线性模型

这一节其实是引出了广义线性模型的定义，而不是专门讨论广义线性模型。

首先我们假设一个多分类问题中（二分类同样适用），其类别条件概率密度 $p(x|C_k)$ 服从高斯分布：
$$
p(x|C_k)=\frac{1}{(2\pi ^{D/2})}\frac{1}{|\sum|^{1/2}}\exp\{-\frac{1}{2}(x-\mu_k)^T\sum\nolimits^{-1}(x-\mu_k)\}
$$
其中 $\sum$ 表示不同类别之间的协方差矩阵，我们假设所有类别的协方差矩阵相同。

根据贝叶斯公式，后验概率如下：
$$
p(C_k|x)=\frac{p(x|C_k)p(C_k)}{p(x)}=\frac{p(x|C_k)p(x)}{\sum_{j=1}^{K}p(x|C_j)p(C_j)}
$$
由于我们定义类别条件概率密度服从高斯分布，而其分布中的常数系数在分式中会相互抵消，因此我们忽略常数系数，化简高斯分布公式，并乘上先验概率：
$$
p(C_k|x)p(C_k)=\exp \{\mu_k^T\sum\nolimits^{-1}x-\frac{1}{2}\mu_k^T\sum\nolimits^{-1}\mu_k+\ln p(C_k)\}
$$
此时我们可以发现这是一个关于 $x$ 的函数，我们定义一个线性判别公式：
$$
a_k(x)=\mu_k^T\sum\nolimits^{-1}x-\frac{1}{2}\mu_k^T\sum\nolimits^{-1}\mu_k+\ln p(C_k)
$$
则后验概率简化为下式：
$$
p(C_k|x)=\frac{\exp(a_k(x))}{\sum_{j=1}^K\exp(a_j(x))}
$$
这个公式，就是 softmax 的函数形式，也就是说后验概率被表示为了一个关于 $x$ 的线性函数经过 softmax 变换后的结果。由于判别函数是线性函数，因此这个分类模型的决策边界也是线性的。

这种模型就被称之为广义线性模型，即由一个线性模型和一个非线性激活函数结合，从而扩展了线性模型。

> 在这个公式推导中，由于先验概率只线性模型的 bias 中出现，因此先验概率只影响了决策边界的平移。

当不同类别间的协方差矩阵不同时，二次项不会相互抵消，因此判别函数会是二次型，也就是决策边界是二次型的。
