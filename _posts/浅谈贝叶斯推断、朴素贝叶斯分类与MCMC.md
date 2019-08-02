贝叶斯推断/决策论是概率框架下实施决策的基本方法。贝叶斯方法考虑如何基于概率和损失来最小化总体损失。

[TOC]

# 常见的概率分布
Let $Z$ be some random variable. Then associated with $Z$ is a probability distribution function that assigns probabilities to the different outcomes $Z$ can take.

##Poisson Distributon
If $Z$ is discrete, then its distribution is called a **probability mass function** , which measures the probability that $Z$ takes on the value $k$ , denoted $P(Z = k )$. Let’s introduce the first very useful probability mass function. We say that $Z$ is *Poisson -distributed* ($Z \sim \mathrm{Poi}(\lambda)$) if:
$$P(Z=k)=\frac{\lambda^{k} e^{-\lambda}}{k!},\quad \lambda>0, k=0,1,2,\cdots$$ One useful property of the Poisson distribution is that its expected value is equal to its parameter. That is,
$$\mathrm{E}[Z|\lambda]=\lambda$$

##Binary Distribution and Bernoulli Distribution
Similar to Poisson distribution, Binary distribution ($Z\sim\mathrm{Bin}(N,p)$) is also a discrete distribution, nevertheless, it only cares about $0$ to $N$ rather than $0$ to $\infty$. 
$$P(Z=k)=\begin{pmatrix}N\\k\end{pmatrix}p^{k}(1-p)^{N-k}$$ The expected value is
$$\mathrm{E}[Z|N,p]=Np$$ Once $N=1$ the Binary distribution reduces to Bernoulli distribution.

##Exponential Density
Instead of a probability mass function, a continuous random variable has a **probability density function**. An example of a continuous random variable is a random variable with *exponential density* ($Z\sim \mathrm{Exp}(\lambda)$).
$$f_{z}(z|\lambda)=\lambda e^{-\lambda z},\quad z\geq 0$$ Given a specific $\lambda$, the expected value of an exponential random variable is equal to the inverse of $\lambda$, i.e.
$$\mathrm{E}[Z|\lambda]=1/\lambda$$

##Normal Distribution
For the sake of data analysis, we denote a normal distribution as $Z\sim N(\mu, 1/\tau)$ . Then the smaller $\tau$ means the broader bandwidth. The probability density of a normal-distributed random variable is:
$$f(z|\mu,\tau)=\sqrt{\frac{\tau}{2\pi}}\mathrm{Exp}\left(-\frac{\tau}{2}(z-\mu)^{2}\right)$$ The expectation of a normal distribution is nothing but $\mu$. 

# 贝叶斯方法简述

统计学中有两个主要学派：频率学派（又称经典学派）和贝叶斯学派。频率学派利用**总体信息**和**样本信息**进行统计推断，贝叶斯学派与之的区别在于还用到了**先验信息**。简单地说，贝叶斯方法是*通过新得到的证据不断地更新我们的信念*。

## 基本原理

贝叶斯学派最基本的观点是：任一未知量 $\theta$ （或者前文中的 $\lambda$，$\mu$ 和 $\tau$） 都可以看做随机变量，可用一个概率分布区描述，这个分布称为**先验分布** （记为 $\pi(\theta)$）。因为任一未知量都有不确定性，而在表述不确定性地程度时，概率与概率分布是最好的语言。依赖于参数 $\theta$ 的密度函数在经典统计学中记为 $p(x,\theta)$，它表示参数空间 $\Theta$ 中不同的 $\theta$ 对应不同的分布。在贝叶斯统计中应记为 $p(x|\theta)$ ，表示随机变量 $\theta$ 给定某个值时，$X$ 的条件密度函数。

从贝叶斯观点看，样本 $x$ 的产生要分两步进行：首先，设想从先验分布 $\pi(\theta)$ 中产生一个样本 $\theta'$ ，这一步人是看不到的，所以是“设想”（注意设想也是有意义的，它反映了同一事件不同人的不同看法，为个人之见的差异留有余地）；再从 $p(x|\theta')$ 中产生一个样本 $x=(x_{1},x_{2},x_{3},\dots,x_{n})$ 。这时样本 $x$ 的联合条件密度函数为：
$$ p(x|\theta')=\prod_{i=1}^{n}p(x_{i}|\theta') $$ 这个联合分布综合了**总体信息**和**样本信息**，又称为**似然函数**。它与极大似然估计中的似然函数没有什么区别。$\theta'$ 仍然是未知的，它是按照先验分布 $\pi(\theta)$ 产生的，为了把**先验信息**综合进去，不能只考虑 $\theta'$，对 $\theta$ 的其它值发生的可能性也要加以考虑，故要用 $\pi(\theta)$ 进行综合。这样一来，样本 $x$ 和参数 $\theta$ 的联合分布为：
$$ h(x,\theta)=p(x|\theta)\pi(\theta) $$ 这个联合分布综合了**总体信息**、**样本信息**和**先验信息**。

我们的核心目标是对 $\theta$ 进行估计，若把 $h(x,\theta)$ 作如下分解：
$$ h(x,\theta) = \pi(\theta|x)m(x) $$ 其中 $m(x)$ 是 $X$ 的**边际密度函数**:
$$ m(x) = \int_{\Theta}h(x,\theta)\mathrm{d}\theta = \int_{\Theta}p(x|\theta)\pi(\theta)\mathrm{d}\theta $$ 它与 $\theta$ 无关。因此，能用来对 $\theta$ 进行估计的只有条件分布 $\pi(\theta|x)$，它的计算公式是：
$$ \pi(\theta|x)=\frac{h(x,\theta)}{m(x)} = \frac{p(x|\theta)\pi(\theta)}{m(x)} =  \frac{p(x|\theta)\pi(\theta)}{\int_{\Theta}p(x|\theta)\pi(\theta)\mathrm{d}\theta} $$ 这就是**贝叶斯公式的密度函数形式**。 这个条件分布称为 $\theta$ 的**后验分布**，它集中了**总体信息**、**样本信息**和**先验信息**中有关 $\theta$ 的一切信息。也可以说是总体和样本对先验分布 $\pi(\theta)$ 作调整的结果，比先验分布更接近 $\theta$ 的实际情况。上述公式是在 $x$ 和 $\theta$ 都是连续随机变量场合下的贝叶斯公式。其它场合下的贝叶斯公式如下：

 1. $x$ 离散，$\theta$ 连续： $$\pi(\theta|x_{j})=\frac{p(x_{j}|\theta)\pi(\theta)}{\int_{\Theta}p(x_{j}|\theta)\pi(\theta)\mathrm{d}\theta}$$
 2. $x$ 连续，$\theta$ 离散：$$\pi(\theta_{i}|x) =\frac{p(x|\theta_{i})\pi(\theta_{i})}{\sum_{i}p(x|\theta_{i})\pi(\theta_{i})} $$
 3. $x$ 离散，$\theta$ 离散：$$\pi(\theta_{i}|x_{j}) =\frac{p(x_{j}|\theta_{i})\pi(\theta_{i})}{\sum_{i}p(x_{j}|\theta_{i})\pi(\theta_{i})} $$

通过引入先验信息的不确定性，我们事实上允许了我们的初始信念可能是错误的。在观察数据、证据或其它信息之后，我们不断更新我们的信念使之更符合目前的证据。

回到我们的核心目标，寻求参数 $\theta$ 的估计 $\hat{\theta}$ 只需要从后验分布 $\pi(\theta| x)$ 中合理提取信息即可。常用的提取方式是用**后验均方误差准则**，即选择这样的统计量
$$ \hat{\theta} = \hat{\theta}(x_{1},x_{2},\dots,x_{n}) $$ 使得后验均方误差达到最小，即
$$ \min\mathrm{MSE}(\hat{\theta} | x) =\min E^{\theta|x}(\hat{\theta}-\theta)^{2}$$ 这样的估计 $\hat{\theta}$ 称为 $\theta$ 的贝叶斯估计，其中 $E^{\theta|x}$ 表示用后验分布 $\pi(\theta|x)$ 求期望。求解上式并不困难，
$$\begin{split}
E^{\theta|x}(\hat{\theta}-\theta)^{2} &= \int_{\Theta}(\hat{\theta}-\theta)^{2}\pi(\theta | x)\mathrm{d}\theta \\
	&= \hat{\theta}^{2} -2\hat{\theta}\int_{\Theta}\theta\pi(\theta|x)\mathrm{d}\theta+\int_{\Theta}\theta^{2}\pi(\theta|x)\mathrm{d}\theta
\end{split}$$ 这是关于 $\hat{\theta}$ 的二次三项式，二次项系数为正，必有最小值：
$$\hat{\theta} = \int_{\Theta}\theta\pi(\theta|x)\mathrm{d}\theta=E(\theta|x)$$ 也就是说，在均方误差准则下， $\theta$ 的贝叶斯估计 $\hat{\theta}$ 就是 $\theta$ 的后验期望 $E(\theta|x)$。

类似的可证，在已知后验分布为 $\pi(\theta|x)$ 的情况下，参数函数 $g(\theta)$ 在均方误差下的贝叶斯估计为 $\hat{g}(\theta)=E[g(\theta)|x] $。

贝叶斯公式中，$m(x)$ 为样本的边际分布，它不依赖于 $\theta$ ，在后验分布计算中仅起到一个正则化因子的作用，加入把 $m(x)$ 省略，贝叶斯公式可改写为如下形式：
$$\pi(\theta|x) \propto p(x|\theta)\pi(\theta)$$ 上式右边虽然不是 $\theta$ 的密度函数或分布列，但在需要时利用正则化立即可以恢复密度函数或分布列原型。这时，可把上式右端称为**后验分布的核**，加入其中还有不含 $\theta$ 的因子，仍可剔去，使核更为精炼。

## 损失函数
这一节我们介绍统计学和决策理论中的损失函数。损失函数时一个关于真实参数及对该参数的估计的函数：
$$L(\theta,\hat{\theta})$$ 如上一节所说的后验均方差就是一种损失函数。损失函数的重要性在于：*他们能够衡量我们的估计的好坏。损失越大，那么根据损失函数来说，这个估计越差。* 均方差是一种常见的损失函数，有着广泛的应用。但是它的缺点在于他们过于强调大的异常值。因为随着估计值的偏离，损失是平方增加的，而不是线性增加的。更稳健的损失函数是**误差的线性函数**，即绝对损失函数：
$$L(\theta,\hat{\theta})=| \theta - \hat{\theta} |$$ 损失函数的设计要考虑到两个关键问题：

 1. 数学上的易于计算性；
 2. 实际应用的稳健性。

我们可以把关注的重心从更精确的参数估计转移到参数估计带来的结果上来。根据具体情形优化估计要求我们设计出能反映目标和结果的损失函数。

## 选择先验分布
**先验分布**的确定十分关键，其原则有二：一是要根据经验信息；二是要使用方便，即在数学上处理方便。贝叶斯先验可分为两类：

 1. **客观先验**，旨在让数据最大程度影响后验，如扁平先验；
 2. **主观先验**：让使用者来表达自己对先验的看法。

选择先验，无论是主观的还是客观的，仍是建模过程的一部分。目前选择先验已经有了一些成熟的方法，如**共轭先验**。共轭先验指的是满足如下关系的先验：
$$p_{\beta}\cdot f_{\alpha}(X) = p_{\beta'}$$ 其中 $p_{\beta}$ 是先验，$p_{\beta'}$ 是后验，$X$ 是数据集。例如Beta先验和二项式数据集意味着Beta后验。
$$f_{X}(x|\alpha,\beta） = \frac{x^{(\alpha-1)}(1-x)^{(\beta-1)}}{B(\alpha,\beta)}$$ Beta分布是均匀分布的更一般形式。共轭先验是一个非常有用的特性，它可以让我们避免使用MCMC，因为我们已知封闭形式的后验。

# 朴素贝叶斯分类

对于分类任务来说，在所有相关概率都已知的理想情况下，贝叶斯分类考虑如何基于这些概率和误判损失来选择最优的类别标记。

**问题描述**
假设有 $N$ 种类别 $C = \{c_{1},c_{2},\dots,c_{N}\}$ ，$\lambda_{ij}$ 是将一个真实类别为 $c_{i}$ 的样本误判为 $c_{j}$ 所产生的损失。基于后验概率 $P(c_{j}|x)$ 可获得将样本 $x$ 分类为 $c_{j}$ 所产生的**期望损失**，即样本 $x$ 上的条件风险：
$$R(c_{j}|x) = \sum_{i=1}^{N}\lambda_{ij}P(c_{i} | x)$$ 我们的目标是寻找一个分类器 $h:X\to C$ 来最小化总体损失：
$$R(h) = E_{x}[R(h(x)|x)]$$

这就产生了**贝叶斯判定准则**（Bayes decision rule）：*为最小化总体风险，只需要在每个样本上选择哪个能使条件风险 $R(c|x)$ 最小的类别标记，即：
$$h^{*}(x) = \arg\max_{c\in C}R(c|x)$$ 如果误判损失 $\lambda_{ij}$ 写为：
$$\lambda_{ij} = \left\{ \begin{aligned} &0,\quad i=j \\ &1,\quad\text{otherwise} \end{aligned} \right.$$ 此时的条件风险为：
$$R(c|x) = 1-P(c|x)$$ 于是，最小化错误率的贝叶斯最优分类器为：
$$h^{*}(x) = \arg\max_{c\in C}P(c|x)$$ 即对每个样本 $x$，选择能使后验概率 $P(c|x)$ 最大的类别。这个原理非常直观。

根据贝叶斯定理，$P(c|x)$ 可写为：
$$P(c|x)=\frac{P(c)P(x|c)}{P(x)}$$ 对于条件概率 $P(x|c)$，它涉及到关于样本所有属性的联合概率，直接根据样本出现的频率来估计将会遇到严重困难。（假设样本的$d$个属性都是二值的，那么样本空间将有$2^{d}$种可能取值，这个值往往大于训练样本数。）为了避开这个困难，**朴素贝叶斯分类器**（Naive Bayes Classifier）采样用**属性条件独立性假设**：对已知类别，假设所有属性相互独立。换言之，假设每个属性独立地对分类结果发生影响。基于此假设，上述公式可展开为：
$$P(c|x) = \frac{P(c)P(x|c)}{P(x)} = \frac{P(c)}{P(x)}\prod_{i=1}^{d}P(x_{i} | c)$$ 其中 $d$ 为属性数量，$x_{i}$为样本的第 $i$ 个属性上的值。因此，
$$h_{nb}(x) = \arg\max_{c\in C}P(c)\prod_{i=1}^{d}P(x_{i}|c)$$ 这就是朴素贝叶斯分类器的表达式。朴素贝叶斯法实际上学习到生成数据的机制，属于**生成模型**。朴素贝叶斯分类算法的简答步骤描述如下：

 1. 计算先验概率及条件概率 $$\begin{split} &P(Y=c_{k}) = \frac{\sum_{i=1}^{N}I(y_{i}=c_{k})}{N} \\  &P(X^{(j)}=a_{jl}|Y=c_{k}) = \frac{\sum_{i=1}^{N}I(x_{i}^{(j)}=a_{jl},y_{i}=c_{k})}{\sum_{i=1}^{N}I(y_{i}=c_{k})} \\  &j=1,2,\dots,n; l=1,2,\dots,S_{j}; k=1,2,\dots,K \end{split}$$ 
 2. 对于给定的实例 $x=(x^{(1)},x^{(2)},\dots,x^{(n)})^{T}$ 计算 $$P(Y=c_{k})\prod_{j=1}^{n}P(X^{(j)}=x^{(j)} | Y = c_{k}),\quad k=1,2,\dots,K$$ 
 3. 确定 $x$ 属于的类 $$y = \arg\max_{c_{k}}P(Y=c_{k})\prod_{j=1}^{n}P(X^{(j)}=x^{(j)} | Y=c_{k})$$

对于离散的属性，我们若有足够的独立同分布样本，可根据期望估计出先验与条件概率。对于连续的属性可考虑概率密度函数。需要注意，若某个属性值在训练集中没有与某个类别同时出现过，则直接基于期望进行概率估计并基于上式进行判别将会出现问题。因为计算出的概率值始终为零，无论其它属性是什么。避免此问题常用的方法是**拉普拉斯修正**（Laplacian correction）：令 $N$ 表示训练集 $D$ 中可能的类别数，$N_{i}$ 表示第 $i$ 个属性可能的取值数，则：
$$P(c) = \frac{|D_{c}|+1}{|D|+N}$$ $$P(x_{i}|c) = \frac{|D_{c,x_{i}}|+1}{|D_{c}|+N_{i}}$$ 随着训练集变大，修正所引入的先验的影响会逐渐变得可忽略。

# 贝叶斯方法与MCMC
MCMC全称**马尔可夫链蒙特卡洛**方法（Markov Chain Monte Carlo）。任何讲贝叶斯推断的书后会讲MCMC，这里我们也稍作探讨，对于MCMC的详细讨论请参考[此文](https://blog.csdn.net/philthinker/article/details/80735037)。关于具体的MCMC实例请参考[此文](https://blog.csdn.net/philthinker/article/details/78314087)。

## 贝叶斯景象图
对于一个含有 $N$ 个未知元素的贝叶斯推断问题，我们隐式地为其先验分布创建一个 $N$ 维空间。先验分布上的某一点的概率，都投射到某个高维曲面或曲线上，其形状由其先验分布决定。例如指数分布：
![指数分布](https://img-blog.csdn.net/20171028161513664?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
这是一个二维的例子，实际运用中先验分布所在的空间和其概率面往往具有更高维度。概率面仅描述了先验分布。得到观测样本后，观测样本会影响概率面的形状，在某些局部产生拉伸或者挤压，以表明参数对的真实值在哪里。最后得到的是后验分布的形状。这些拉伸和挤压的结果是形成几座“山峰”。“山峰”的位置标明从后验分布上看，各未知变量的真实值可能在哪。

注意：如果某一点处先验概率是0，那么在这一点推不出后验概率。

## MCMC
遍历一个 $N$ 维空间的复杂度将随着 $N$ 呈指数增长，如此一来我们将如何找到贝叶斯景象图中隐藏的山峰呢？实际上，MCMC背后的思想就是如何聪明地对空间进行搜索。**MCMC 的返回值是后验分布上的一些样本点，而非后验分布本身**。我们可以把MCMC的过程想象成不断重复地问一块石头“你是不是来自我们要找的那座山？”并试图用大量的给肯定答案的石头来重塑那座山，最后将它们返回。这里的“石头”就是样本，累计起来称之为“迹”。MCMC每次都会探索附近位置上的概率值，并朝着概率值增加的方向前进。

一般看来，用大量样本描述一个后验分布是一种效率极低的方法。但是对于高维问题，用数学公式来描述“山峰所在范围”是非常困难的。若仅返回“峰顶”的位置也是可行的，但是图形的形状对于判定未知量的后验概率也十分重要。

MCMC可由一系列算法实现，这些算法大多可以描述为以下几步：

 1. 从当前位置开始。
 2. 尝试移动一个位置。
 3. 根据新位置是否服从于该概率数据和先验分布，来决定是否采纳这次移动。
 4. 如果采纳，就在新位置重复第1步；如果不采纳，那么留在原处并重复第1步。
 5. 大量迭代之后返回所有被采纳的点。

整体上看，MCMC向着后验分布所在的方向前进，并沿途谨慎地收集样本。而一旦达到后验分布所在的区域，就可以轻松地采集更多样本，因为那里的点几乎都位于后验分布的区域里。**MCMC收敛的涵义就是迹收敛到概率较大的点集。** 

最初的几千个点与最终目标分布关系不大，所以使用这些点参与估计并不明智。我们可以剔除这些点后进行估计。产生这些遗弃点的过程被称为**预热期**。数学上可以证明，若让MCMC通过更多的采样运行的足够久，就可以忽略起始位置。理想情况下，我们希望起始位置就在分布图形的山峰处，因为这是后验概率所在的区域，如果以山峰为起点，就能避免很长的预热器以及错误的估计结果。我们将山峰位置称为**最大后验**，或简称MAP。

MCMC会天然地返回具有**自相关性**的采样结果。因为“行走”过程总是从前位置移动到附近某个位置。如果一次采样过程的探索效果很好，那么表现出的自相关性也会很高。如果后验样本的自相关性很高，又会引起另一个问题：很多后处理算法要求样本彼此相互独立。这个问题可以通过每隔 $n$ 返回一个样本来解决或减轻。

- Thanks Cameron Davidson-Pilon for the great work of *Bayesian Methods for Hackers: Probabilistic programming and Bayesian Inference*.

- 感谢周志华——《机器学习》清华大学出版社。

- 感谢李航——《统计学习方法》清华大学出版社。