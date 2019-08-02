下面我们简单讨论下强化学习中的函数估计问题，这里对于强化学习的基本原理、常见算法以及凸优化的数学基础不作讨论。假设你对强化学习（Reinforcement Learning）有最基本的了解。

[TOC]

#概述
对于状态空间为连续空间的强化学习问题，我们需要利用函数估计的方法表示各种映射关系。函数估计方法可分为参数估计和非参数估计，其中参数化估计又分为线性参数化估计和非线性参数化估计。本文中我们主要讨论参数化估计。对于基础较薄弱读者，可以参考[这篇更基础的文章](http://blog.csdn.net/philthinker/article/details/72119513)。
#价值函数估计
价值函数估计的过程可以看作是一个监督学习的过程，其中数据和标签对为 $(S_{t}, U_{t})$ 。训练的目标函数为： $$\arg\min_{\theta}(q(s,a)-\hat{q}(s,a,\theta))\quad \text{or} \quad  \arg\min_{\theta}(v(s)-\hat{v}(s,\theta))$$
##增量式/梯度下降方法
梯度下降的基本原理可以参考凸优化问题中的[无约束规划方法](http://blog.csdn.net/philthinker/article/details/78191864)。这里我们要求估计偏差最小，因此采用**梯度下降**方法：$$\theta_{t+1}=\theta_{t}+\alpha d_{t}$$这里 $d_{t}$ 是偏差下降的方向，此处应为 $-\nabla_{\theta}(U_{t}-\hat{v}(S_{t},\theta_{t}))$ 即负梯度方向。代入上式可得：
$$\theta_{t+1}=\theta_{t}+\alpha[U_{t}-\hat{v}(S_{t},\theta_{t})]\nabla_{\theta}\hat{v}(S_{t},\theta)$$ 注意此处 $U_{t}$ 与 $\theta$ 无关，但情况并非总是这样。如果采用蒙特卡罗方法对实验进行采样，即 $U_{t} = G_{t}$ 时，上述公式直接成立；但如果采样 $TD(0)$ 方法采样，由于用到了 bootstrapping，即 $U_{t}=R_{t+1}+\gamma\hat{v}(S_{t+1},\theta)$ ， $U_{t}$ 中也包含 $\theta$。 使用上式忽略了这个影响，因此被称为**部分梯度**（semi-gradient）法。

下面讨论线性估计问题，即 $\hat{v}(s,\theta)=\theta^{T}\phi(s)$。常用的线性基函数类型如下：

 1. 多项式基函数：$(1,s_{1},s_{2},s_{1}s_{2}, s_{1}^{2},s_{2}^{2},\dots )$ 
 2. 傅里叶基函数：$\phi_{i}(s)=\cos(i\pi s),s\in[0,1]$
 3. 径向基函数：$\phi_{i}(s)=\exp\left(-\frac{\|s-c_{i}\|^{2}}{2\sigma_{i}^{2}}\right)$

不同的更新公式如下：

 1. 蒙特卡罗方法：$\Delta\theta = \alpha[G_{t}-\theta^{T}\phi(s)]\phi(s)$ 
 2. $TD(0)$方法：$\Delta\theta = \alpha[R+\gamma\theta^{T}\phi(s')-\theta^{T}\phi(s)]\phi(s)$
 3. 正向视角的$TD(\lambda)$方法：$\Delta\theta = \alpha[G_{t}^{\lambda}-\theta^{T}\phi(s)]\phi(s)$
 4. 反向视角的$TD(\lambda)$方法：$$\begin{split} \delta_{t} &= R_{t+1}+\gamma\theta^{T}\phi(s')-\theta^{T}\phi(s) \\ E_{t}&=\gamma\lambda E_{t-1}+\phi(s) \\ \Delta\theta &= \alpha\delta_{t}E_{t} \end{split}$$

关于这些更新方法的具体含义可以参考[这篇文章](http://blog.csdn.net/philthinker/article/details/72519083)。

##批处理方法
批处理方法的计算比较复杂，但是计算效率高。批处理方法是指给定经验数据集 $D=\{(s_{1},v_{1}^{\pi}), (s_{2},v_{2}^{\pi}),\dots, (s_{T},v_{T}^{\pi}) \}$，找到最好的拟合函数 $\hat{v}(s,\theta)$ 使得 $LS(\theta)=\sum_{t=1}^{T}(v_{t}^{\pi}-\hat{v}_{t}^{\pi}(s_{t},\theta))^{2}$ 最小（此处为最小二乘）。此处我们不做详细介绍。
##深度强化学习浅析（DQN）
这里介绍的 DQN 就是 DeepMind 发表在 *Nature* 上的一篇论文：

> Human-level Control through Deep Reinforcement Learning

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

其中第5行通过预处理得到状态对应的特征输入。

###Double DQN
DQN无法克服 Q-Learning 本身固有的过估计问题，原因是其中的最大化操作。Double Q-Learning 将动作的选择和动作的评估分别用不同的值函数来实现，可以缓解此问题。

在 Double Q-Learning 中，$$Y_{t}=R_{t+1}+\gamma Q(S_{t+1},\arg\max_{a}Q(S_{t+1},a,\theta_{t}),\theta_{t}')$$ 将该思想运动到 DQN 中，得到 Double DQN，其 $TD$ 目标为：
$$Y_{t}^{DQN} = R_{t+1}+\gamma Q(S_{t+1},\arg\max_{a}Q(S_{t+1},a,\theta_{t}),\bar{\theta}_{t})$$

###带有优先回放的Double DQN( Prioritized Replay )
这里仅讨论优先回放思想，不给出具体算法。在DQN中，选取训练集合的方法是均匀采样，然而并非所有数据集的效率一致。某些状态的学习效率远比其他状态高。优先回放的接班思想就是赋予学习效率高的状态以更大的采样权重。

那么如何选择采样权重呢？一个选择是 $TD$ 偏差 $\delta$ 。例如：我们设样本 $i$ 处的 $TD$ 偏差为 $\delta$， 则该处的采样概率为
$$P_{i}=\frac{p_{i}^{\alpha}}{\sum_{k}p_{k}^{\alpha}}$$ 其中 $p_{i}=|\delta_{i}|+\epsilon$ 或者 $p_{i}=\frac{1}{rank(i)}$ 。$|rank(i)|$ 根据 $|\delta_{i}|$ 排序得到。

采用优先回放的概率分布采样时，动作值的估计是一个**有偏估计**。因为采样分布于动作值函数分布完全不同，为了矫正这个偏差，我们需要乘以一个重要性采样系数 $\omega_{i}=\left( \frac{1}{N}\cdot\frac{1}{P_{i}} \right)^{\beta}$ 。

###Dueling DQN
Dueling DQN 从网络结构上改进了 DQN。动作值函数可以被分解为状态值函数和优势函数，即：
$$Q^{\pi}(s,a)=V^{\pi}(s)+A^{\pi}(s,a)$$ 这也是为了消除训练数据的关联性，此处不做具体讨论。

##非参数化估计方法
除了参数化方法之外，价值函数估计还有非参数化方法。非参数化函数估计指参数的个数和基底形式并非固定，由样本决定的估计方法。例如基于**核函数**的方法和基于**高斯过程**的方法。此处不做细致介绍，有兴趣可以参考如下书籍：

> 李航. 统计学习方法[M]. 清华大学出版社，2012.
> Rasmussen C E, Williams C K I. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning)[M]. The MIT Press, 2005.

#直接策略搜索
基于价值函数的方法往往适用于有限的状态空间集合。策略搜索是将策略参数化，即 $\pi_{\theta}(s)$，寻找最优的参数 $\theta$ ，使强化学习目标——累计回报的期望最大。这里不介绍过多细节，有兴趣的读者可以参考这篇[更具体的文章](http://blog.csdn.net/philthinker/article/details/71104095)。

## 无模型的策略搜索
### 随机策略
#### REINFORCE
随机策略搜索法最典型的算法是 [REINFORCE](http://blog.csdn.net/philthinker/article/details/73201410) 这里不给出具体算法，只推导基本原理。

我们用 $\tau$ 表示一组状态-行为序列 $s_{0}, u_{0}, \dots, s_{H}, u_{H}$，用符号 $R(\tau)=\sum_{t=0}^{H}R(s_{t},u_{t})$ 表示轨迹 $\tau$ 的回报，$P(\tau,\theta)$ 表示轨迹 $\tau$ 出现的概率，此时直接策略搜索的目标可以表示为：
$$U(\theta)=\sum_{\tau}P(\tau,\theta)R(\tau)$$ 此时强化学习的目标是找到最优参数 $\theta$ 使得
$$\max_{\theta}U(\theta) = \max_{\theta}\sum_{\tau}P(\tau,\theta)R(\tau)$$ 此时搜索问题转化为优化问题，下面我们采用[最速下降法](http://blog.csdn.net/philthinker/article/details/78191864)求解（这里其实是上升）。
$$\theta_{t+1}=\theta_{t}+\alpha\nabla_{\theta}U(\theta)$$ 下面研究如何求 $\nabla_{\theta}U(\theta)$:
$$\begin{split} \nabla_{\theta}U(\theta) &= \nabla_{\theta}\sum_{\tau}P(\tau,\theta)R(\tau) \\ &= \sum_{\tau}\nabla_{\theta}P(\tau,\theta)R(\tau) \\ &= \sum_{\tau}P(\tau,\theta)\frac{\nabla_{\theta}P(\tau,\theta)}{P(\tau,\theta)}R(\tau) \\ &= \sum_{\tau}P(\tau,\theta)\nabla_{\theta}\log P(\tau,\theta)R(\tau) \end{split}$$ 这样一来求 $\nabla_{\theta}U(\theta)$ 变成了估计 $\nabla_{\theta}\log P(\tau,\theta)R(\tau)$ 的期望。这可以利用经验平均，即利用 $m$ 条轨迹的经验计算平均值来估计：
$$\nabla_{\theta}U(\theta) \approx \frac{1}{m}\sum_{i=1}^{m}\nabla_{\theta}\log P(\tau_{i},\theta)R(\tau_{i})$$ 下面再研究如何估计 $\nabla_{\theta}\log P(\tau,\theta)$：
$$\begin{split} \nabla_{\theta}\log P(\tau,\theta) &= \nabla_{\theta}\log \left[ \prod_{t=0}^{H}P(s_{t+1} | s_{t},u_{t})\cdot\pi_{\theta}(u_{t} | s_{t}) \right] \\ &= \nabla_{\theta} \left[ \sum_{t=0}^{H}\log P(s_{t+1} | s_{t},u_{t})+\sum_{t=0}^{H}\log\pi_{\theta}(u_{t} | s_{t}) \right] \\ &= \nabla_{\theta}\sum_{t=0}^{H}\log\pi_{\theta}(u_{t} | s_{t}) \\ &= \sum_{t=0}^{H}\nabla_{\theta}\log\pi_{\theta}(u_{t} | s_{t}) \end{split}$$ 到这一步可以看出，似然概率 $P$ 的梯度变化仅与策略 $\pi_{\theta}$ 有关，与环境本身的动力学模型无关，这个结果被称为**策略梯度定理**。因此：
$$\nabla_{\theta}U(\theta) \approx \frac{1}{m}\sum_{i=1}^{m} \sum_{t=0}^{H}\nabla_{\theta}\log\pi_{\theta}(u_{t}^{(i)} | s_{t}^{(i)})R(\tau_{i}^{(i)})$$ 这个估计是无偏的，但是方差很大。我们可以在回报中引入常数基线 $b$ 来减小方差：
$$\begin{split} \nabla_{\theta}U(\theta) &\approx \frac{1}{m}\sum_{i=1}^{m}\nabla_{\theta}\log P(\tau^{(i)},\theta)\left(R(\tau^{(i)})-b\right) \\ &= \frac{1}{m}\sum_{i=1}^{m} \sum_{t=0}^{H}\nabla_{\theta}\log\pi_{\theta}(u_{t}^{(i)} | s_{t}^{(i)})\left(R(\tau_{i}^{(i)})-b\right) \end{split}$$ 两个估计等价，证明很简单，此处从略。

####G(PO)MDP
从之前的讨论中可以看出，每个动作 $u_{t}^{(i)}$ 所对应的 $\nabla_{\theta}\log\pi_{\theta}(u_{t}^{(i)}|s_{t}^{(i)})$ 都乘以相同的轨迹总回报 $\left(R(\tau_{i}^{(i)})-b\right)$ 。然而，当前的动作与过去的回报实际上没有关系。因此，我们可以修改回报函数，有一种方法称为 G(PO)MDP：
$$\nabla_{\theta}U(\theta) \approx \frac{1}{m}\sum_{i=1}^{m} \sum_{j=0}^{H-1} \sum_{t=0}^{j}\nabla_{\theta}\log\pi_{\theta}(u_{t}^{(i)} | s_{t}^{(i)})\left(r_{j}-b_{j}\right)$$

#### TRPO
策略梯度算法的硬伤就是更新步长 $\alpha$ 的取法问题，当步长不合适时，更新的参数所对应的策略可能是一个更不好的策略。TRPO（Trust Region Policy Optimization）证明解决了此问题，使得当策略更新后，回报函数的值不能更差。TRPO的具体介绍请参考[此文](http://blog.csdn.net/philthinker/article/details/79551892)。

#### Actor-Critic
异策略（off-policy）是指行动策略和评估测录不是同一个策略。AC框架是一种实现异策略强化学习的典型框架。

关于Actor-Critic 框架的具体讨论请参考[此文](https://blog.csdn.net/philthinker/article/details/71104095)。

### 确定性策略
2014年，Silver 在论文

> Deterministic Policy Gradient Algorithm

中首次提出了确定性策略理论。2015年 DeepMind 将该理论与 DQN 结合，在论文

> Continuous Control with Deep Reinforcement Learning

中提到了DDPG算法。

确定性策略的公式如下：$$a=\mu_{\theta}(s)$$ 和随机策略不同，相同的策略参数，在状态为 $s$ 时，动作是唯一确定的。确定性策略的优点在于**需要采样的数据少，算法效率高**。随机策略的梯度计算公式：
$$\nabla_{\theta}J(\pi_{\theta})=E_{s\sim\rho^{\pi},a\sim\pi_{\theta}}[\nabla_{\theta}\log\pi_{\theta}(a|s)Q^{\pi}(s,a)]$$ 此式表明，策略梯度公式是关于状态和动作的期望，在求期望时，需要对状态分布和动作分布求积分，这就要求在状态空间和动作空间采集大量的样本，这样求均值才能近似期望。然而，确定性策略的动作是确定的，因此不需要再动作空间采样积分，所以确定性策略需要的样本数据更小。确定性策略梯度如下：$$\nabla_{\theta}J(\mu_{\theta})=E_{s\sim\rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}]$$ 

#### DPG 与 DDPG
言归正传，确定性策略动作是确定的，无法探索环境，那么如何学习呢？答案就是利用异策略方法，这里采用AC框架。AC算法包含两个同等地位的元素，一个是 Actor 即行动策略，另一个是 Critic 即评估策略，这里指的是利用函数逼近的方法估计值函数。Actor 方法用来调整 $\theta$ 值；Critic 方法逼近值函数 $Q^{\omega}(s,a)\approx Q^{\pi}(s,a)$，其中 $\omega$ 为待逼近的参数，可用 TD 学习的方法评估值函数。

异策略随机策略梯度为 $$\nabla_{\theta}J(\pi_{\theta})=E_{s\sim\rho^{\pi},a\sim\pi_{\theta}}\left[ \frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)} \nabla_{\theta}\log\pi_{\theta}(a|s)Q^{\pi}(s,a) \right]$$ 采样策略为 $\beta$ 。
异策略确定性策略梯度为：$$\nabla_{\theta}J_{\beta}(\mu_{\theta})=E_{s\sim\rho^{\beta}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}]$$  对比上述两式不难发现，确定性策略梯度求解少了重要性权重。这是因为重要性采样是用简单的概率分布去估计复杂的概率分布，而确定性策略的动作为确定值而不是概率分布；此外，确定性策略的值函数评估用的是 Q-Learning 方法，即 TD(0)。有了上式，确定性异策略AC算法的更新过程如下：
$$\begin{split} \delta_{t} &= r_{t} + \gamma Q^{\omega}(s_{t+1},\mu_{\theta}(s_{t+1})) - Q^{\omega}(s_{t},a_{t}) \\ \omega_{t+1} &= \omega_{t} + \alpha_{\omega}\delta_{t}\nabla_{\omega}Q^{\omega}(s_{t},a_{t}) \\ \theta_{t+1} &=  \theta_{t} + \alpha_{\theta}\nabla_{\theta}\mu_{\theta}(s_{t})\nabla_{a}Q^{\omega}(s_{t},a_{t})|_{a=\mu_{\theta}(s)} \end{split}$$ 以上介绍的是 Deterministic Policy Gradient 方法，简称 DPG。

有了 DPG，我们再看 DDPG，即Deep Determinstic Policy Gradient。这里所谓的深度是指利用神经网络估计行为值函数 $Q^{\omega}(s_{t},a_{t})$ 和确定策略 $\mu_{\theta}(s)$。如前介绍DQN时所说，这里用了两个技巧：经验回放和独立的目标网络。此处不再重复。这里需要修改的是对 $\omega$ 和 $\theta$ 利用独立的网络进行更新。DDPG的更新公式为：
$$\begin{split} \delta_{t} &= r_{t} + \gamma Q^{\omega^{-}}(s_{t+1},\mu_{\theta^{-}}(s_{t+1})) - Q^{\omega}(s_{t},a_{t}) \\ \omega_{t+1} &= \omega_{t} + \alpha_{\omega}\delta_{t}\nabla_{\omega}Q^{\omega}(s_{t},a_{t}) \\ \theta_{t+1} &=  \theta_{t} + \alpha_{\theta}\nabla_{\theta}\mu_{\theta}(s_{t})\nabla_{a}Q^{\omega}(s_{t},a_{t})|_{a=\mu_{\theta}(s)} \\ \theta^{-} &= \tau\theta+(1-\tau)\theta^{-} \\ \omega^{-} &= \tau\omega + (1-\tau)\omega^{-} \end{split}$$

## 基于模型的策略搜索
无模型强化学习算法有很多优点，比如无需环境建模。但是因为没有模型，无模型方法必须不断试探环境，效率低下。解决该问题的方法是利用模型探索。例如有了模型之后，可以利用基于模型的优化方法得到好的数据，并稳定训练策略网络；而且，有了模型我们可以充分利用示教（Demonstration）数据学习。

### GPS
引导策略搜索方法（Guided Policy Search）最早见于2015年 Sergey Levine 的博士论文

> Levine S, "Motor skill learning with local trajectory methods," PhD thesis, Stanford University, 2014.

GPS将策略搜索分为两步：**控制相**和**监督相**。控制相通过轨迹最优、传统控制器或随机最优等方法产生好的数据；监督相利用产生的数据进行监督学习。

关于GPS的具体讨论详见[此文](http://blog.csdn.net/philthinker/article/details/79575794)。

### PILCO
基于模型的强化学习方法最大的问题是模型误差。针对此类问题，业界提出了 PILCO （Probabilistic Inference for Learning Control）算法。它把模型误差纳入考虑的范围。

关于PILCO的具体实现方法详见[此文](https://blog.csdn.net/philthinker/article/details/79749038)。