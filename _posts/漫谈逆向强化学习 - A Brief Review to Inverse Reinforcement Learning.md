下面我们来探讨下逆向强行学习的基本原理和典型方法，我们假设您已经对强化学习和凸优化的基本原理有一定的了解。

@[toc]
# 概述
我们先介绍下逆向强化学习的概念预分类：

什么是逆向强化学习呢？当完成复杂的任务时，强化学习的回报函数很难指定，我们希望有一种方法找到一种高效可靠的回报函数，这种方法就是逆向强化学习。我们假设专家在完成某项任务时，其决策往往是最优的或接近最优的，当所有的策略产生的累积汇报函数期望都不比专家策略产生的累积回报期望大时，强化学习所对应的回报函数就是根据示例学到的回报函数。即逆向强化学习就是从专家示例中学习回报函数。当需要基于最优序列样本学习策略时，我们可以结合逆向强化学习和强化学习共同提高回报函数的精确度和策略的效果。逆向强化学习的基本理论可参考如下论文：

> Ng A Y, Russell S J. Algorithms for Inverse Reinforcement Learning. ICML, 2000

逆向强化学习一般流程如下：

 1. 随机生成一个策略作为初始策略；
 2. 通过比较“高手”的交互样本和自己交互样本的差别，学习得到回报函数；
 3. 利用回报函数进行强化学习，提高自己策略水平；
 4. 如果两个策略差别不大，就可以停止学习了，否则回到步骤2。

逆向强化学习分类如下：

 1. 最大边际形式化：学徒学习、MMP方法、结构化分类、神经逆向强化学习。
 6. 基于概率模型的形式化：最大熵IRL、相对熵IRL、深度逆向强化学习。

最大边际化方法的缺点是很多时候不存在单独的回报函数使得专家示例行为既是最优的又比其它任何行为好很多，或者不同的回报函数挥导致相同的专家策略，也就是说这种方法无法解决歧义问题。基于概率模型的方法可以解决此问题。

逆向强化学习项目可参考：

> https://github.com/MatthewJA/Inverse-Reinforcement-Learning


# 基于最大边际的逆向强化学习

## 学徒学习
学徒学习指的是从专家示例中学到回报函数，使得在该回报函数下所得的最优策略在专家示例策略附近。设未知的回报函数
$$R(s)=w\cdot \phi(s)$$ 其中 $\phi(s)$ 为基函数，可以是多项式基底、傅里叶基底等。此时逆向强化学习要求得的是灰板函数的系数 $w$。

根据值函数定义：$$E_{s_{0}\sim D}[V^{\pi}(s_{0})] = E\left[\left.\sum_{t=0}^{\infty}\gamma^{t}R(s_{t})\right|\pi\right]=E\left[\left.\sum_{t=0}^{\infty}\gamma^{t}w\cdot\phi(s_{t})\right|\pi\right]=w\cdot E\left[\left.\sum_{t=0}^{\infty}\gamma^{t}\phi(s_{t})\right|\pi\right]$$ 定义**特征期望** 为 $\mu(\pi)=E\left[\left.\sum_{t=0}^{\infty}\gamma^{t}\phi(s_{t})\right|\pi\right]$ 因此 $E_{s_{0}\sim D}[V^{\pi}(s_{0})] =w\cdot \mu(\pi)$。给定 $m$ 跳专家轨迹后，我们可以估计特征期望为 $$\hat{\mu}_{E}=\frac{1}{m}\sum_{i=1}^{m}\sum_{t=0}^{\infty}\gamma^{t}\phi\left(s_{t}^{(i)}\right)$$ 
我们要找一个策略，使得它的表现与专家策略相近，其实就是找到一个策略 $\tilde{\pi}$ 的特征期望与专家策略的特征期望相近，即 $$\| \mu(\tilde{\pi})-\mu_{E} \|_{2}\leq \epsilon$$ 对于任意的权重 $\|w\|_{2}\leq 1$，值函数满足如下不等式
$$\left| E\left[\left.\sum_{t=0}^{\infty}\gamma^{t}R(s_{t})\right|\pi_{E}\right] - E\left[\left.\sum_{t=0}^{\infty}\gamma^{t}R(s_{t})\right|\tilde{\pi}\right] \right| = \left| w^{T}\mu(\tilde{\pi})-w^{T}\mu_{E} \right| \leq \|w\|_{2}\cdot \| \mu(\tilde{\pi}) - \mu_{E} \|_{2} \leq \epsilon$$ 

学徒学习的伪代码如下：

 1. Randomly pick some policy $\pi^{(0)}$, compute or approximate via Monte Carlo $\mu^{(0)}=\mu(\pi^{(0)})$, and set $i=1$.
 2. Compte $t^{(i)}=\max_{w:\|w\|_{2}\leq 1}\min_{j\in\{0,1,\dots,m\}} w^{T}(\mu_{E}-\mu^{(j)})$ and let $w^{(i)}$ be the value of $w$ that attains this maximum.
 3. If $t^{(i)}\leq \epsilon$, then terminates.
 4. Using RL algorithm, compute the optimal policy $\pi^{(i)}$ for the MDP using rewards $R=(w^{(i)})^{T}\phi(s)$.
 5. Compute or estimate $\mu^{(i)}=\mu(\pi^{(i)})$.
 6. Set $i=i+1$ and go back to step 2.

其中step 2 的目标函数写成标准的优化形式为：
$$\begin{aligned} &\max_{t,w}   t \\ \text{s.t. } & w^{T}\mu_{E}\geq w^{T}\mu^{(j)},j=0,1,\dots,i-1 \\ & \|w\|_{2}\leq 1 \end{aligned}$$
综上所述，学徒学习方法可分两步：

 1. 在已经迭代得到的最优策略中，利用最大边际方法求出当前的回报函数的参数值；
 2. 将求出的回报函数作为当前系统的回报函数，并利用强化学习方法求出此时的最优策略。

详细内容请参考：

> Abbeel P, Ng A Y. Apprenticeship learning via inverse reinforcement learning, ICML. ACM, 2004.

代码可参考 irl/linear_irl.py 中的 large_irl 函数。

## 最大边际规划（MMP）
最大边际规划将逆向强化学习建模为 $D=\{(\mathcal{X}_{i}, \mathcal{A}_{i},p_{i},F_{i},y_{i},\mathcal{L}_{i})\}_{i=1}^{n}$ 该式从左到右依次为状态空间，动作空间，状态转移概率，回报函数的特征向量，专家轨迹和策略损失函数。在MMP框架下，学习者试图找到一个特征到回报的线性映射也就是参数 $w$，在这个线性映射下最好的策略在专家示例策略附近。该问题可形式化为：
$$\begin{aligned} &\min_{w,\zeta_{i}}\frac{1}{2}\|w\|^{2}+\frac{\gamma}{n}\sum_{i}\beta_{i}\zeta_{i}^{q} \\ &\text{s.t. }\forall i   \quad w^{T}F_{i}\mu_{i}+\zeta_{i}\geq \max_{\mu\in\mathcal{G}_{i}}w^{T}F_{i}\mu+l_{i}^{T}\mu \end{aligned}$$ 第二行为约束，其含义如下：

 1. 约束只允许专家示例得到最好的回报的权值存在；
 2. 回报的边际差，即专家示例的值函数与其他策略的值函数的差值，与策略损失函数成正比。

损失函数可以利用轨迹中两种策略选择不同动作的综合来衡量，此处策略 $\mu$ 指的是每个状态被访问的频次。关于MMP的求解内容此处暂时不做详细讨论。

详细内容请参考：

> Ratliff N D, Bagnell J A, Zinkevich M A. Maximum margin planning, International Conference. DBLP, 2006.

## 基于结构化分类的方法
MMP方法在约束不等式部分，变量是策略，需要迭代求解MDP的解，计算代价很高。为了避免迭代计算MDP的解，我们可以这样考虑问题：对于一个行为空间很小的问题，最终的策略其实是找到每个状态所对应的最优动作。若把每个动作看作是一个标签，那么所谓的策略其实就是把所有的装填分成几类，分类的表示是值函数，最好的分类对应着最大的值函数。利用这个思想，逆向强化学习可以形式化为：
$$\begin{aligned} &\min_{\theta,\zeta}\frac{1}{2}\|\theta\|^{2}+\frac{\eta}{N}\sum_{i=1}^{N}\zeta_{i} \\ &\text{s.t. }\forall i,\quad \theta^{T}\hat{\mu}^{\pi_{E}}(s_{i},a_{i})+\zeta_{i}\geq \max_{a}\theta^{T}\hat{\mu}^{\pi_{E}}(s_{i},a)+\mathcal{L}(s_{i},a)  \end{aligned}$$ 约束中的$\{s_{i},a_{i}\}$ 为专家轨迹，$\hat{\mu}^{\pi_{E}}(s_{i},a_{i})$ 可以利用Monte Carlo方法求解，而对于 $\hat{\mu}^{\pi_{E}}(s_{i},a\neq a_{i})$ 则可利用启发的方法得到。

从数学形式看，结构化分类的方法与MMP非常相似，但是两者有本质不同：

 1. 结构化分类的方法约束每个状态处的每个动作；
 2. MMP约束一个MDP的解。

详细内容请参考：

> Klein E, Geist M, Piot B, et al. Inverse reinforcement learning through structured classification, Advances in Neural Information Processing Systems. 2012.

## 神经逆向强化学习
逆向强化学习要学习的是回报函数，但是学习回报函数时又引入了需要人为指定的基底。对于大规模问题，人为指定基底表示能力不足，只能覆盖部分回报函数形式，难以泛化到其他状态空间。解决方法之一就是利用神经网络表示回报函数的基底。此处暂不对神经逆向强化学习做深入讨论。

详细内容请参考：

> Chen X, Kamel A E. Neural inverse reinforcement learning in autonomous navigation. Robotics & Automation Systems,2016.

# 基于最大熵的逆向强化学习
基于最大边际的方法往往会产生歧义，比如许多不同的回报函数会导致相同的专家策略。在这种情况下，所学到的回报函数往往具有随机的偏好。最大熵方法就是为解决此问题提出的。

在概率论中，熵是不确定性的度量。不确定性越大，熵越大。

> **最大熵原理**：在学习概率模型时，在所有满足约束的概率模型（分布）中，熵最大的模型是最好的模型。

原因在于通过熵最大所选取的模型，没有对未知做任何主观假设。从概率模型的角度建模逆向强化学习，我们可以这样考虑：**存在一个潜在的概率分布，在该概率分布下，产生了专家轨迹。**此时，已知条件为
$$\sum_{\zeta_{i}}P(\zeta_{i})f=\tilde{f}$$ 这里用 $f$ 表示特征期望，$\tilde{f}$ 表示专家特征期望。在满足上述约束条件的所有概率分布中，熵最大的概率分布式除了约束外对其他任何位置信息没有做任何假设的分布。所以，最大熵的方法可以避免歧义性的问题。

## 基于最大信息熵的逆向强化学习
熵最大，是最优问题，我们将该问题转化为标准型：
$$\begin{aligned} &\max -p\log p \\ &\text{s.t. } \sum_{\zeta_{i}}P(\zeta_{i})f_{\zeta_{i}}=\tilde{f} \\ &\sum P =1 \end{aligned}$$ 该问题可用拉格朗日乘子法求解：
$$\min L = \sum_{\zeta_{i}}p\log p -\sum_{j=1}^{n}\lambda_{j}(pf_{i}-\tilde{f})-\lambda_{0}(\sum p-1)$$ 对概率 $p$ 进行微分，并令导数为0，得到有最大熵的概率为：
$$p=\frac{\exp\left( \sum_{j=1}^{n}\lambda_{j}f_{j} \right)}{\exp(1-\lambda_{0})}=\frac{1}{Z}\exp\left( \sum_{j=1}^{n}\lambda_{j}f_{j} \right)$$ 参数 $\lambda_{j}$ 对应着回报函数中的参数，改参数可以利用**最大似然法**求解。

一般而言，利用最大似然法求解上式中的参数时，往往会遇到未知的配分函数项 $Z$ ，因此不能直接求解。一种可行的方法是利用**次梯度**的方法，如：
$$\nabla L(\lambda)=\tilde{f}-\sum_{\zeta}P(\zeta|\lambda,T)f_{\zeta}$$ 其中轨迹的概率 $P(\zeta)$可表示为： $$Pr(\tau|\theta,T)\propto d_{0}(s_{1})\exp\left( \sum_{i=1}^{k}\theta_{i}f_{i}^{\tau} \right)\prod_{t=1}^{H}T(s_{t+1}|s_{t},a_{t})$$ 求解该式的前提是装填转移概率 $T$ 是已知的。

详细内容请参考：

> Ziebart B D, Mass A, Bagnell J A, et al. Maximum entropy inverse reinforcement learning. National Conference on Artifical Intelligence. AAAI Press, 2008.

## 基于相对熵的逆向强化学习
在无模型强化学习中，状态转移概率 $T$ 往往是未知的，为解决此问题，我们可将问题建模为求解相对熵最大。设 $Q$ 为利用均匀分布策略产生的轨迹分布，要求解的概率分布为 $P(\tau)$，问题可形式化为：
$$\begin{aligned} &\min_{P}\sum_{\tau\in\mathcal{T}}P(\tau)\ln \frac{P(\tau)}{Q(\tau)} \\ &\text{s.t. }\forall i\in\{1,2,\dots,k\}: \\ &\left| \sum_{\tau\in\mathcal{T}}P(\tau)f_{i}^{\tau}-\hat{f}_{i} \right| \leq\epsilon \\ &\sum_{\tau\in\mathcal{T}}P(\tau)=1 \\ & \forall \tau\in\mathcal{T}:P(\tau)\geq 0 \end{aligned}$$ 同样利用拉格朗日乘子法和KKT条件求解，可以得到相对熵最大的解。参数的求解过程同样利用次梯度方法。此处对具体的求解过程暂不做详细讨论。

详细内容请参考：

> Boularias A, Kober J, Peters J. Relative entropy inverse reinforcement learning. 2011.

## 深度逆向强化学习
最大熵逆向强化学习虽然解决了歧义性问题，但在实际应用中难以应用，这是因为：

 1. 回报函数的学习需要人为选定特征，对于很多实际问题，特征的选择是很困难的；
 2. 很多逆向强化学习的子循环包含正向强化学习，而正向强化学习本身就是很难解决的问题。

解决上述第一个问题可以利用深度神经网络来估计回报函数；针对第二个问题，可以采用基于采样的方法替代正向强化学习。此时暂不对具体算法做详细讨论。

详细内容请参考：

> Wulfmeier M, Ondruska P, Posner I, Maximum Entropy Deep Inverse Reinforcement Learning.  2015. 
> Finn C, Levine S, Abbeel P. Guided cost learning: deep inverse optimal control via policy optimization. 2016. 

## GAIL

GAIL (Generative Adversarial Imitation Learning) 是一种使用 GAN (Generative Adversarial Network) 完成 Imitation Learning 工作的方法。我们知道回报函数的目标是使专家轨迹的长期回报尽可能高于其它策略生成的轨迹。把这个目标对应到 GAN 的场景下：*在 GAN 中我们希望生成分布尽可能靠近真实分布；而在逆向强化学习中我们希望策略模型轨迹尽可能靠近专家轨迹。*我们可以采用类似的方法，由策略模型充当 GAN 中的生产模型，以状态为输入生成行动；而回报函数模型可以充当判别模型，用于判别行动近似专家行动的程度。

GAIL 的实现代码在 baselines/gail 文件夹中。这里不作详细介绍，具体内容请参考：

> Goodfellow I J, Pouget-Abadie J, Mirza M, et al. Generative Adversarial Networks. NIPS, 2014.

