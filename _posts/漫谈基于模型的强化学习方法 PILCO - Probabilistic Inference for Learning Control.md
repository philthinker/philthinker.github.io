基于模型的强化学习方法最大的问题是模型误差。针对此类问题，业界提出了 PILCO （Probabilistic Inference for Learning Control）算法。它把模型误差纳入考虑的范围。它解决模型偏差的方法不是集中于一个单独的动力学模型，而是建立了概率动力学模型，即动力学模型上的分布。也就是说，PILCO建立的模型并不是具体的某个确定性函数，而是建立一个可以描述一切可行模型（所有通过已知训练数据的模型）上的概率分布。

该概率模型有两个目的：

 1. 它表达和表示了学习到的动力学模型的不确定性；
 2. 模型不确定性被集成到长期的规划和决策中。

本文中我们不对PILCO的推导做详细讨论，PILCO的具体内容可参考如下论文：
> Deisenroth M P, Rasmussen C E. PILCO: A Model-based and Data-efficient Approach to Policy Search. Int. Conf. on Machine Learning, Bellevue, Washington, USA, Jane 28-July, pp. 465-472, 2011.
> Deisenroth M P, Rasmussen C E and Fox D. Learning to Control a Low-Cost Manipulator Using Data-Efficient Reinforcement Learning. Robotics: Science and Systems, 2011.

[TOC]

# PILCO
## PILCO 算法概述
PILCO算法包含三个层次：

```flow
layer01=>operation: 底层 - 学习转移概率模型
layer02=>operation: 中层 - 对长期预测进行近似推断
layer03=>operation: 顶层 - 策略更新

layer03->layer02->layer01
```
底层学习一个状态转移概率模型 $f$；中层利用该状态转移概率模型和策略 $\pi$，预测在策略 $\pi$ 下，后续的状态分布，并利用 $V^{\pi}(x_{0})=\sum_{t=0}^{T}\int c(x_{t})p(x_{t})dx_{t}$ 对策略进行评估；顶层利用基于梯度的方法对策略 $\pi$ 的参数进行更新。PILCO的伪代码如下：

 1. Set policy to random
 2. loop
	 3. execute policy and record collected experience
	 4. learn probabilistic dynamics model
	 5. loop
		 6. simulate system with policy $\pi$
		 7. compute expected long-term cost $V^{\pi}$
		 8. improve policy
	 9. end loop
 10. end loop

下面我们对每一层稍加说明。

## 底层：学习转移概率模型
PILCO算法用的概率模型是高斯过程模型。假设系统的动力学可以由下列公式描述：
$$x_{t}=f(x_{t-1},u_{t-1})$$ 
PILCO的概率模型不直接对模型建模，而是引入一个差分变量 $\Delta_{t}$，通过如下变换：
$$\Delta_{t}=x_{t}-x_{t-1}+\varepsilon$$ 
设 $\Delta_{t}$ 符合高斯分布，则 $x_{t}$ 也符合高斯分布：
$$p(x_{t} | x_{t-1},u_{t-1})=\mathcal{N}(x_{t}|\mu_{t},\Sigma_{t})$$ 
其中均值 $\mu_{t}=x_{t-1}+E_{f}[\Delta_{t}]$。令 $\tilde{x}=(x,u)$，PILCO动力学概率模型学习的是输入 $\tilde{x}$ 和输出 $\Delta$ 之间的拟合关系。差分变换很少，学习差分近似于学习函数的梯度。

此处省略具体的求解方法。

## 中层：对长期预测进行近似推断
这一层的目的是实现策略评估，即计算 $V^{\pi}(x_{0})=\sum_{t=0}^{T}\int c(x_{t})p(x_{t})dx_{t}$ 其中 $c(x_{t})$ 为人为指定的奖励/惩罚函数。由于底层算法学到了概率动力学模型，因此此处值函数的计算可以利用该模型，不需要与环境交互。

概率分布的具体计算方法比较复杂，此处省略。

## 顶层：策略更新
策略更新采用基于梯度的策略搜索方法。得到最优的策略即找到最优的策略参数，使得：
$$\pi^{*}\in\arg\min_{\pi\in\Pi}V^{\pi_{\psi}}(x_{0})$$ 
其中 $\Pi$ 为所有参数空间所对应的策略空间。具体的优化方法此处省略。

# PILCO 算法的改进
## 滤波PILCO
PILCO算法假设了状态完全可观，不存在测量误差。实际中状态并非完全可观的，且存在噪声。该问题可以通过在执行步和预测步计入滤波器解决。滤波器的使用过程包含滤波器的更新步和滤波器预测步两个阶段。此处不细说。
## 有向探索PILCO
在策略改善步，PILCO利用优化的方法最小化累计代价函数的均值得到新的参数。这样的优化方法其实只有“exploitation”没有“exploration”，没有考虑到模型的不确定性。能平衡利用和探索的算法应该既要考虑累计代价的均值函数，也要考虑累计代价函数的方差函数。
## 深度PILCO
PILCO算法一直被诟病的问题是其模型计算复杂度随着观测状态的维数指数增长，因此难以应用到高维度系统中。我们可以利用贝叶斯网络代替高斯回归模型对PILCO进行扩展，即深度PILCO。
