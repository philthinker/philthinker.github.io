EM算法是一种迭代算法，用于含有**隐变量**（latent variable）的概率模型参数的[极大似然估计](https://blog.csdn.net/philthinker/article/details/80487967)，或极大后验概率估计。EM算法也是一种**无监督**机器学习算法与[K-means算法](https://blog.csdn.net/philthinker/article/details/70226326)十分类似。

@[toc]
## Expectation-Maximization 算法简述
如果概率模型仅含有可观测变量，那么给定数据，可以直接用极大似然法估计模型参数，或用贝叶斯法估计模型参数。然而如果模型含有隐变量，就不能简单的使用这些估计方法。下面我们讨论含有隐变量的概率模型参数极大似然估计法，即EM算法。

变量定义：

 - $Y$：观测的随机变量数据，或称不完全数据（incomplete-data）。
 - $Z$：隐随机变量数据。$Y$ 和 $Z$ 连在一起成为完全数据（complete-data）。
 - $\theta$：需要估计的模型参数。

EM算法通过迭代的方式求不完全数据对数似然函数 $L(\theta)=\log P(Y|\theta)$ 的极大似然估计。介绍EM算法之前，我们首先介绍其核心——**Q函数**。
$$Q(\theta,\theta^{(i)}) = E_{Z}[\log P(Y,Z | \theta)|Y,\theta^{(i)}] $$ 它是完全数据的对数似然函数 $\log P(Y,Z | \theta)$ 关于给定的观测数据 $Y$ 和当前参数 $\theta^{(i)}$ 下对隐藏数据 $Z$ 的条件概率分布 $P(Z|Y,\theta^{(i)})$ 的期望值。下面我们导出EM算法：


----------


我们的目标是极大化观测数据关于参数的对数似然函数，即极大化[^1] $$L(\theta) = \log P(Y|\theta) = \log\sum_{Z}P(Y,Z|\theta) = \log\sum_{Z}P(Y |Z,\theta)P(Z|\theta)$$ 

[^1]: 这是对于离散隐变量而言的。对于连续的隐变量：$$L(\theta) = \log\int P(Y,Z|\theta)dZ$$ 后面的推导一样需要将求和调整为积分。

由于包含了隐变量和求和（或积分）的对数，直接极大化非常困难。因此我们采用迭代的方法逐步近似，假设第 $i$ 次迭代后参数的估计值是 $\theta^{(i)}$。我们希望新的估计值能提高 $L(\theta)$ 的值，并逐步收敛到极大值。基于此思路，我们考虑更新后的增量：
$$L(\theta)-L(\theta^{(i)}) =  \log\left(\sum_{Z}P(Y |Z,\theta)P(Z|\theta)\right) - \log P(Y|\theta^{(i)})$$
利用Jensen不等式[^2]进行放缩，可求出其下界：

[^2]:Jensen不等式：$$\log\sum_{j}\lambda_{j}y_{j}\geq \sum_{j}\lambda_{j}\log y_{j}, \quad \lambda_{j}\geq 0, \sum_{j}\lambda_{j} =1$$

$$\begin{aligned} L(\theta)-L(\theta^{(i)})  &= \log\left(\sum_{Z}P(Y |Z,\theta^{(i)})\frac{P(Y |Z,\theta)P(Z|\theta)}{P(Y |Z,\theta^{(i)})}\right) - \log P(Y|\theta^{(i)}) \\ &\geq \sum_{Z}P(Y |Z,\theta^{(i)})\log\frac{P(Y |Z,\theta)P(Z|\theta)}{P(Y |Z,\theta^{(i)})}  - \log P(Y|\theta^{(i)}) \\ &= \sum_{Z}P(Y |Z,\theta^{(i)})\log\frac{P(Y |Z,\theta)P(Z|\theta)}{P(Y |Z,\theta^{(i)})P(Y|\theta^{(i)})}  \end{aligned}$$ 到这里，我们得到了更新后的一个下界，当然不是下确界：
$$L(\theta) \geq L(\theta^{(i)})  + \sum_{Z}P(Y |Z,\theta^{(i)})\log\frac{P(Y |Z,\theta)P(Z|\theta)}{P(Y |Z,\theta^{(i)})P(Y|\theta^{(i)})} $$ 

任何使这个下界增大的 $\theta$ 都可以使我们的目标函数增大，此时我们希望使这个下界尽可能增大，即求解：$$\theta^{(i+1)} = \arg\max_{\theta}\left[ L(\theta^{(i)})  + \sum_{Z}P(Y |Z,\theta^{(i)})\log\frac{P(Y |Z,\theta)P(Z|\theta)}{P(Y |Z,\theta^{(i)})P(Y|\theta^{(i)})} \right]$$ 求解过程非常直接，我们去掉与 $\theta$ 无关的项，得到：
$$ \theta^{(i+1)} = \arg\max_{\theta} \left( \sum_{Z}P(Z|Y,\theta^{(i)})\log P(Y,Z|\theta) \right) = \arg\max_{\theta}Q(\theta,\theta^{(i)}) $$ 反复执行上述过程即可。EM算法的收敛性此处不做讨论。EM算法的另外一个推导参考[此文](https://blog.csdn.net/philthinker/article/details/77481814)。

----------

总结一下，EM算法可分为如下几步：

 1. 初始化参数 $\theta^{(1)}$，注意EM算法对初值是敏感的。
 2. **E 步**：基于 $\theta^{(i)}$，求出 $Q(\theta,\theta^{(i)})$。即求期望。
 3. **M 步**：以 $\theta$ 为参数极大化 $Q(\theta,\theta^{(i)})$，得到 $\theta^{(i+1)}$。
 4. 当参数变化足够小时停止迭代，否则返回步骤2。

EM算法其实是在给定观测数据的情况下**最大化完全数据的似然**。

---------
这里给出一个**混合伯努利分布**的例子，简单展示一下EM算法的执行步骤。

假设有一类二值化的数据：$x_{n} = \{000110\dots\}$ 其中 $[x_{n}]_{i} = 0 \text{ or } 1, i=1,2,\dots,D$ 是其组成元素。显然，这类数据的概率分布满足伯努利分布。
$$p(x_{i})=\mu_{i}^{x_{i}}(1-\mu_{i})^{1-x_{i}},\quad p(x)= \prod_{i=1}^{D}p(x_{i})$$ 其中 $\mu_{i}$ 是 $x_{i}=1$ 的概率。基于上述定义，我们可以利用极大似然法定义一个生成器 $\mu=\frac{1}{N}\sum_{n=1}^{N}x_{n}$，其中 $\mu\in\mathbb{R}^{D}$ 且 $0<[\mu]_{i}<1$。极大似然法推导该生成器的过程并无特别技巧，此处从略。

现在考虑一个更复杂的问题，假设我们的这个 $N$ 个数据 $x_{n}$ 来自 $K$ 个生成器 $\{\mu_{k}\}_{k=1}^{K}$ ，我们要对这些数据进行分类，判断它们分别属于哪一个生成器。此时这 $K$ 个生成器相当于隐变量，采用EM方法可以估计这 $K$ 个生成器的参数，并实现数据分类。执行EM算法之前，我们首先给出选择生成器 $\mu_{k}$ 生成数据 $x$ 的概率：
$$p_{\mu_{k}}(x)=\prod_{i=1}^{D}[\mu_{k}]_{i}^{x_{i}}(1-[\mu_{k}]_{i})^{1-x_{i}}$$ 生成器 $\mu_{k}$ 被选择的概率：
$$p(\mu_{k})=\pi_{k}, \quad \sum_{k=1}^{K}\pi_{k}=1$$ 那么可以得出数据 $x$ 被生成的概率：
$$p(x)=\sum_{k=1}^{K}\pi_{k}p_{\mu_{k}}(x)=\sum_{k=1}^{K}\prod_{i=1}^{D}\pi_{k}[\mu_{k}]_{i}^{x_{i}}(1-[\mu_{k}]_{i})^{1-x_{i}}$$ 这便是**混合伯努利分布**。可以看出，此时公式中同时包含求和与求积符号，难以直接利用极大似然法计算。

下面我们使用EM算法迭代式求解 $\pi_{k}$ 与 $\mu_{k}$：首先初始化 $\{\pi_{k}\}_{k=1}^{K}$ 与 $\{\mu_{k}\}_{k=1}^{K}$，注意EM算法对初值的选取是敏感的，适当的初值对算法的性能影响很大。然后要做的是写出**完全数据**并求其对数似然，即 $Q$ 函数。

**E步**（求期望）： 我们已知数据 $x_{n}$ 被生成的概率是： $$p(x_{n}) = \sum_{k=1}^{K}\pi_{k}p_{\mu_{k}}(x_{n})$$ 那么考虑完整数据的期望，其实这里的期望就是对隐含变量的估计，即“数据 $x_{n}$ 是被生成器 $\mu_{k}$ 所生成”的概率： $$p(\mu_{k}|x_{n})=\frac{p(\mu_{k},x_{n})}{p(x_{n})}=\frac{\pi_{k}p_{\mu_{k}}(x_{n})}{\sum_{k=1}^{K}\pi_{k}p_{\mu_{k}}(x_{n})} \triangleq \gamma_{nk} $$  那么，完全数据的对数似然为

**M步**（解参数）：根据上式，我们可得参数 $\pi_{k}$ 与 $\mu_{k}$ 的更新值：$$\mu_{k}=\frac{\sum_{n=1}^{N}\gamma_{nk}x_{n}}{\sum_{n=1}\gamma_{nk}},\quad \pi_{k}=\frac{\sum_{n=1}^{N}\gamma_{nk}}{N}$$

读者也可参考[此文](https://www.jianshu.com/p/1121509ac1dc)。

---------

## 高斯混合模型（Gaussian Mixture Model, GMM）

高斯混合模型是指具有如下形式的概率分布模型： $$P(y|\theta) = \sum_{k=1}^{K}\alpha_{k}\phi(y|\theta_{k})$$ 其中 $\alpha_{k}\geq 0,\sum_{k=1}^{K}\alpha_{k}=1$；$\phi(y|\theta_{k})$ 是高斯分布密度，$\theta_{k} = (\mu_{k},\sigma_{k}^{2})$， 
$$\phi(y|\theta_{k}) = \frac{1}{\sqrt{2\pi}\sigma_{k}}\exp\left( -\frac{(y-\mu_{k})^{2}}{2\sigma_{k}^{2}} \right)$$ 称为第 $k$ 个分模型。

## 用EM算法学习高斯混合模型

假设有一组观测数据 $Y=\{y_{1},y_{2},\dots,y_{N}\}$ 由上述高斯混合模型生成，下面我们用EM算法估计参数：$\theta = (\alpha_{1},\alpha_{2},\dots,\alpha_{K},\theta_{1},\theta_{2},\dots,\theta_{K})$。

在高斯混合模型中，我们的观测数据 $y_{j}$ 是这样产生的：首先依照概率 $\alpha_{k}$ 选择第 $k$ 个高斯分布模型 $\phi_{k}$；然后依照该模型的概率分布生成数据 $y_{j}$。这时观测数据 $y_{j}$ 是已知的；反映观测数据来自第 $k$ 个高斯高分的数据是未知的，以 $\gamma_{jk}$ 表示，其定义如下：
$$\gamma_{jk} = \left\{ \begin{aligned} &1,\quad \text{第 $j$ 个观测数据来自第 $k$ 个高斯模型} \\ &0,\quad \text{Otherwise} \end{aligned} \right.$$ 有个这个隐变量的定义，样本的完全数据是：
$$(y_{j},\gamma_{j1},\gamma_{j2},\dots,\gamma_{jK}),\quad j=1,2,\dots,N$$ 选定参数初值后，我们开始进行EM算法：


----------
### E-Step
根据当前模型参数，计算分模型 $k$ 对观测数据 $y_{j}$ 的**响应度**。

首先写出 Q 函数：
$$Q(\theta,\theta^{(i)}) = E[\log P(y,\gamma|\theta)|y,\theta^{(i)}]$$ 我们需要的是完全数据的对数似然，然后在当前的模型参数下对其求期望。
$$\begin{aligned} P(y,\gamma|\theta) &= \prod_{j=1}^{N}P(y_{j},\gamma_{j1},\gamma_{j2},\dots,\gamma_{jK}|\theta) \\  &= \prod_{k=1}^{K}\prod_{j=1}^{N}[\alpha_{k}\phi(y_{j}|\theta_{k})]^{\gamma_{jk}} \\ &= \prod_{k=1}^{K}\alpha_{k}^{n_{k}}\prod_{j=1}^{N}\left[ \frac{1}{\sqrt{2\pi}\sigma_{k}}\exp\left( -\frac{(y_{j}-\mu_{k})^{2}}{2\sigma_{k}^{2}} \right) \right]^{\gamma_{jk}}  \end{aligned}$$ 其中 $n_{k} = \sum_{j=1}^{N}\gamma_{jk}, \sum_{k=1}^{K}n_{k} = N$。注意 $n_{k}$ 表示第 $k$ 个高斯模型共产生了几个实例 $y$。那么，完全数据的对数似然函数为：
$$\log P(y,\gamma|\theta) = \sum_{k=1}^{K}\left[ n_{k}\log\alpha_{k}+\sum_{j=1}^{N}\gamma_{jk}\left( -\log\sqrt{2\pi}-\log\sigma_{k}-\frac{1}{2\sigma_{k}^{2}}(y_{j}-\mu_{k})^{2} \right) \right] $$

有了完全数据的对数似然函数后，在给定 $y$ 和 $\theta^{(i)}$ 的条件下，我们对其求期望，即确定 $Q$ 函数。
$$Q(\theta,\theta^{(i)}) = \sum_{k=1}^{K}\left[ \sum_{j=1}^{N} (E\gamma_{jk}) \log\alpha_{k}+\sum_{j=1}^{N}(E\gamma_{jk}) \left( -\log\sqrt{2\pi}-\log\sigma_{k}-\frac{1}{2\sigma_{k}^{2}}(y_{j}-\mu_{k})^{2} \right) \right] $$ 
这里的关键是计算 $(E\gamma_{jk})$，我们称其为模型 $k$ 对数据 $y_j$ 的**响应度**，用 $\hat{\gamma}_{jk}$ 表示。它指的是当前模型参数下第 $j$ 个观测数据来自第 $k$ 个高斯模型的概率。$n_{k} = \sum_{j=1}^{N}\hat{\gamma}_{jk}$。
$$\hat{\gamma}_{jk} = E(\gamma_{jk}|y,\theta) = P(\gamma_{jk}=1|y,\theta)=\frac{\alpha_{k}\phi(y_{j}|\theta_{k})}{\sum_{k=1}^{K}\alpha_{k}\phi(y_{j}|\theta_{k})}$$ 那么
$$Q(\theta,\theta^{(i)}）=  \sum_{k=1}^{K}\left[ n_{k} \log\alpha_{k}+\sum_{j=1}^{N}\hat{\gamma}_{jk} \left( -\log\sqrt{2\pi}-\log\sigma_{k}-\frac{1}{2\sigma_{k}^{2}}(y_{j}-\mu_{k})^{2} \right) \right] $$

### M-Step
计算新一轮迭代的模型参数。
$$\theta^{(i+1)} = \arg\max_{\theta}Q(\theta,\theta^{(i)}）$$ 求解最优化问题的方法有很多，这里可以简单地对各个参数求偏导数并令其等于0，注意求 $\alpha_{k}^{(i+1)}$时，前提在 $\sum_{k=1}^{K}\alpha_{k}=1$ 的条件下。结果如下：
$$\mu_{k}^{(i+1)}=\frac{\sum_{j=1}^{N}\hat{\gamma}_{jk}y_{j}}{\sum_{j=1}^{N}\hat{\gamma}_{jk}},\quad k=1,2,\dots,K \tag{1}$$
$$(\sigma_{k}^{(i+1)})^{2}=\frac{\sum_{j=1}^{N}\hat{\gamma}_{jk}(y_{j}-\mu_{k})^{2}}{\sum_{j=1}^{N}\hat{\gamma}_{jk}},\quad k=1,2,\dots,K \tag{2}$$
$$\alpha_{k}^{(i+1)}=\frac{n_{k}}{N}=\frac{\sum_{j=1}^{N}\hat{\gamma}_{jk}}{N},\quad k=1,2,\dots,K \tag{3}$$

重复上述计算，之道参数基本不再变化为止。不过要注意初值的选择尽量合适，EM算法对初值是敏感的。

总结一下，采用EM算法进行无监督学习，估计高斯混合模型参数的步骤为：
输入： 观测数据 $y_{1},y_{2},\dots,y_{N}$，高斯混合模型；
输出：高斯混合模型参数。

 1. 取参数的初值；
 2. E步：依据当前模型参数，计算分模型 $k$ 对观测数据 $y_{j}$ 的响应度 $$\hat{\gamma}_{jk} = \frac{\alpha_{k}\phi(y_{j}|\theta_{k})}{\sum_{k=1}^{K}\alpha_{k}\phi(y_{j}|\theta_{k})}, j= 1,2,\dots,N; k = 1,2,\dots,K$$
 3. M步：计算新一轮迭代的模型参数，同上述 (1) (2) (3) 式。
 4. 重复上述步骤2和3，直到收敛。

----------


EM算法的用途不限于估计高斯混合模型的参数，EM算法在**强化学习**策略搜索中的应用请参考[此文](https://blog.csdn.net/philthinker/article/details/77481814)。

- 感谢李航——《统计学习方法》清华大学出版社
- 感谢中井悦司——《机器学习入门之道》姚待艳 译，人民邮电出版社 
