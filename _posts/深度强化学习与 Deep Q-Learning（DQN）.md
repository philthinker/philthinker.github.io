本文中我们来探讨一下深度强化学习（以DQN为主）的基本原理。这里假设读者对强化学习的基本原理与神经网络的基本知识已经有了一定的了解。

[TOC]


## Deep Q-Learning
这里介绍的 DQN 就是 DeepMind 发表在 *Nature* 上的一篇论文：

> Mnih V, Kavukuuoglu K, Silver D, et al. Human-level Control through Deep Reinforcement Learning. *Nature*, 2015, 518(7540):529.

DQN 技术是 Q-Learning 算法的一种变体，具体改变的是以下三个方面：

 1. DQN 利用深度卷积神经网络估计值函数；
 2. DQN 利用经验回放进行学习；
 3. DQN 独立设置了目标网络来单独处理时间差分算法中的 TD 偏差。

由于训练神经网络时，存在的假设是训练数据是独立同分布的，而通过强化学习采集的数据之间总是存在着关联性，易造成神经网络不稳定。经验回放技术可以打破数据间的关联。独立的目标网络使用 $\bar{\theta}$ 而不是 $\theta$ 来计算 $TD$ 偏差，这样做也为了打破关联性。DQN的算法伪代码如下：

 1. Initialize replay memory $D$ to capacity $N$
 2. Initialize $Q$-function with ramdom weights $\theta$
 3. Initialize target $Q$-function with weights $\bar{\theta}=\theta$
 4. For $episode=1,M$ do
	 5. Initalize sequence $s_{1}=\{x_{1}\}$ and preprocessed sequence $\phi_{1}=\phi(s_{1})$
	 6. For $t=1,T$ do
		 7. Select action $a_{t}$, then observe reward $r_{t}$ and image $x_{t+1}$
		 8. Processed $\phi_{t+1}=\phi(x_{t+1})$ and store transition $(\phi_{t},a_{t},r_{t},\phi_{t+1})$ in $D$
		 9. Sample minibatch of transitions $(\phi_{j},a_{j},r_{j},\phi_{j+1})$ from $D$
		 10. Set $y_{j}=\left\{\begin{aligned} r_{j},\qquad\qquad &\text{ if episode terminates at step }j+1 \\r_{j}+\gamma\max_{a'}Q(\phi_{j+1},a',\bar{\theta}),&\text{ otherwise} \end{aligned}\right.$
		 11. Perform a gradient descent step on $(y_{j}-Q(\phi_{j},a_{j},\theta))$ w.r.t. network parameter $\theta$
		 12. Every $C$ steps reset $\bar{\theta} = \theta$
	 13. End for
 14. End for

其中第5行通过预处理得到状态对应的特征输入。DQN 已经实现了很大的飞跃，但是还不够，下面的内容将陆续介绍 DQN 的各种改进方法。

## Double DQN
DQN无法克服 Q-Learning 本身固有的过估计问题，原因是其中的最大化操作。Double Q-Learning 将动作的选择和动作的评估分别用不同的值函数来实现，可以缓解此问题。

这里介绍的Double DQN 方法来自论文：

> Van Hasselt H, Guez A, Silver D. Deep Reinforcement Learning with Double Q-Learning. *Computer Science*,2015.

DQN 中采用两个结构相同的网络：Behavior Network 和 Target Network。虽然这个方法提升了模型的稳定性，但是 Q-Learning 对价值的过高估计的问题未能解决。Q-Learning 在计算时利用下一个时刻的最优价值，所以它通常会给出一个状态行动的估计上限。由于训练过程中模型并不够稳定，因此对上限的估计也会存在偏差。如果偏差不一致，那么这个偏差会造成模型对行动优劣的判断偏差，这样会影响模型效果。

在 Q-Learning 中，我们已经知道 Target Network 求解价值目标值得公式：
$$Y_{t}=R_{t+1}+\gamma\max_{a}Q(S_{t+1},a,\bar{\theta}_{t})$$ 进一步展开：
$$Y_{t}=R_{t+1}+\gamma Q(S_{t+1},\arg\max_{a}Q(S_{t+1},a,\bar{\theta}_{t}),\bar{\theta}_{t})$$ 为了尽可能地减少过高估计的影响，一个简单的方法是把选择最优行动和估计最优行动两部分的工作分离。我们用Behavior Network 完成最优行动的选择工作。将该思想运动到 DQN 中，得到 Double DQN，其 $TD$ 目标为：
$$Y_{t}^{DQN} = R_{t+1}+\gamma Q(S_{t+1},\arg\max_{a}Q(S_{t+1},a,\theta_{t}),\bar{\theta}_{t})$$ 通过这个变化，算法的三个环节模型安排如下：

 1. 采样：Behavior Network $Q(\theta)$；
 2. 选择最优行动：Behavior Network $Q(\theta)$；
 3. 计算目标价值：Target Network $Q(\bar{\theta})$。

## 带有优先回放的 DQN( Prioritized Replay Buffer)
这里仅讨论优先回放思想，不给出具体算法。这里介绍的PRB来自论文：

> Schaul T, Quan J, Antonoglou I, et al. Prioritized Experience Replay. *Computer Science*, 2015.

在DQN中，选取训练集合的方法是均匀采样，然而并非所有数据集的效率一致。某些状态的学习效率远比其他状态高。如果平等地看待每一个样本，就会在那些简单的样本上话费比较多的时间，而学习潜力没有被重复挖掘出来。优先回放的接班思想就是赋予学习效率高的状态以更大的采样权重。交互时表现越差，对应的权重就越高，这样可以更高效地利用样本。

那么如何选择采样权重呢？一个选择是 $TD$ 偏差 $\delta$ 。例如：我们设样本 $i$ 处的 $TD$ 偏差为 $\delta$， 则该处的采样概率为
$$P_{i}=\frac{p_{i}^{\alpha}}{\sum_{k}p_{k}^{\alpha}}$$ 其中 $p_{i}=|\delta_{i}|+\epsilon$ 或者 $p_{i}=\frac{1}{rank(i)}$ 。$|rank(i)|$ 根据 $|\delta_{i}|$ 排序得到。

采用优先回放的概率分布采样时，动作值的估计是一个**有偏估计**。因为采样分布于动作值函数分布完全不同，为了矫正这个偏差，我们需要乘以一个重要性采样系数 $\omega_{i}=\left( \frac{1}{N}\cdot\frac{1}{P_{i}} \right)^{\beta}$ 。这里当 $\beta = 1$ 时，更新效果实际上等同于 Replay Buffer，当 $\beta <1$ 时，Priority Replay Buffer 就能够发挥作用了。我们使用 RRB 的目的正式让更新变得有偏，加入这个纠正是为了说明我们可以通过一些设定让它变回RB那样的更新方式。这样虽无好处，但也没有坏处。我们可以根据实际问题调整权重，即调整 $\beta$ ，让它在两种更新效果之间做一个权衡。

从另一个角度讲，我们使用PRB的收敛性并不确定，所以我们还是希望PRB最终变回RB，所以可以让 $\beta$ 在训练开始时复制为一个小于1的数，然后随着训练迭代数的增加，让 $\beta$ 不断变大，并最终达到 1。

总结一下PRB的原理：

 1. 在样本放入 Replay Buffer 时，计算 $P(i) = \frac{p(i)^{\alpha}}{\sum_{j}p(j)^{\alpha}}$；
 2. 在样本取出时，以上一步计算的概率进行采样；
 3. 在更新时，为每一个样本添加 $w_{i}=(\frac{1}{N\cdot P(i)})^{\beta}$ 的权重；
 4. 随着训练进行，让 $\beta$ 从某个小于 1 的值渐进靠近 1。

## Dueling DQN
Dueling DQN 从网络结构上改进了 DQN。它利用模型结构将值函数表示成更细致的形式，这使得模型能够拥有更高的表现。这里的Dueling DQN来自论文：

> Wang Z, Schaul T, Hessel M, et al. Dueling Network Architectures for Deep Reinforcement Learning. 2015.

值函数 $Q$ 被分解为状态值函数和优势函数（Advantage Function），即：
$$Q^{\pi}(s,a)=V^{\pi}(s)+A^{\pi}(s,a)$$ 优势函数可以表现出当前行动和平均表现之间的区别：如果优于平均表现，那么优势函数为正，反之则为负。我们对优势函数加一个限定，我们知道优势函数的期望为 0，将公式变成：
$$Q^{\pi}(s,a) = V^{\pi}(s)+\left( A^{\pi}(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a') \right)$$ 让每个A值减去档期状态所有A值得平均值，可以保证期望值为0的约束，从而增加了整理输出的稳定性。

这样做到底有什么好处呢？首先，如果在某些场景下序号使用 V 的值，我们不用再训练一个网络。同时，通过显式地给出 V函数的输出值，每一次更新时，我们都会显式地更新 V函数。这样V函数的更新频率会得到确定性的增加，对于单一输出的Q网络来说，它的更新就显得有些晦涩。从网络训练角度看，这样做使得网络训练更友好且容易。

## 解决 DQN 的冷启动问题
对于以值函数为核心的Q-Learning算法来说，前期的算法迭代很难让模型快速进入一个相对理想的环境。更何况前期的值函数估计存在较大偏差，与环境交互得到的采样与最优策略存在一定的差别，这更增加了学习的难度。论文：

> Hester T, Vecerik M, Pietquin O, et al. Deep Q-Learning from Demonstration. 2018.

给出了一种强化学习和监督学习结合的方案。作者的主要思路是利用预先准备好的优质采样轨迹加快模型前期的训练速度。模型的目标函数变成多个学习目标结合的形式：
$$J(q) = J_{DQ}(q)+\lambda_{1}J_{n}(q)+\lambda_{2}J_{E}(q)+\lambda_{3}J_{L2}(q)$$ 

 1. $J_{DQ}(q)$：Deep Q-Learning 的目标函数；
 2. $J_{n}(q)$：以[$n$步回报估计法](https://blog.csdn.net/philthinker/article/details/72670095)为目标的Q-Learning目标函数；
 3. $J_{E}(q)$：利用准备数据进行监督学习的目标函数；
 4. $J_{L2}(q)$：L2正则的目标函数。

一般来说，事先准备好的数据比较有限，很难支撑一个完整的模型训练，因此它必然只能影响很小一部分的状态行动值。如果它不能尽可能地覆盖更多的状态，那么这些数据反而有可能对模型造成不好的影响。同时，准备好的数据也可能存在噪声，其中的行动并不是真正的行动。因此监督学习的目标函数被定义为如下形式：
$$J_{E}(Q) = \max_{a\in A}[Q(s,a)+l(a_{E},a)]-Q(s,a_{E})$$ 其中 $a_{E}$ 表示当前状态下专家给出的行动，$l(x,x)$函数是一个指示函数，当模型选择的行动与 $a_{E}$ 相同时，函数值为0；不同时为某个固定值。如果 $a\neq a_{E}$，则说明其它某个行动的价值至少不弱于专家行动太多，这对模型来说是一个比较合理的约束。

## Distributional DQN
Distributional DQN 以类似直方图的形式表示了价值的分布。这个算法同Dueling DQN类似，都是要对价值模型的结构进行改进。它来自论文：

> Bellemare M G, Dabney W, Munos R. A Distributional Perspective on Reinforcement Learning. 2017.

Distributional DQN 模型希望通过建立更复杂细致的值函数，让估计结果更细致可信。此处对其具体原理暂不做细致讨论。

## Noisy DQN
探索问题是强化学习中经常遇到的问题，常见的$\epsilon-greedy$ 方法相当于在执行策略环节增加了一定的噪声，使得模型具有一定的探索能力。Noisy DQN 通过增加参数的随机性增强模型的探索性能，相比较 $\epsilon-greedy$ 方法，它使用了一种更平滑的手段增加探索能力。本方法来自论文：

> Fortunato M, Azar M G, Piot B, et al. Noisy Networks for Exploration. 2017.

此处对其具体原理暂不做细致讨论。

## Rainbow
Rainbow 模型将前面的改进融合在了一起。它来自论文：

> Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements in Deep Reinforcement Learning. 2017.

在Baselines中关于DQN的实现，对应的代码在Baselines/deepq文件夹中。这个子项目实现了一个类似于Rainbow的算法。此处暂不做详细讨论。

*感谢冯超——《强化学习精要：核心算法与Tensorflow实现》电子工业出版社*
*感谢郭宪 等——《深入浅出强化学习：原理入门》电子工业出版社*