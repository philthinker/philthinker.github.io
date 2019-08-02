本文中我们不对拓扑学中的相关概念进行阐述，并假设读者对三维空间中的[刚体运动](https://blog.csdn.net/philthinker/article/details/73008139)以及机器人有一定了解。

[TOC]

## 位形空间（Configuration Space）
机器人上各点位置的一个完整规范被称为位形（Configuration），所有可能的位形组合的集合被称为位形空间。我们将用 $\mathcal{Q}$ 来表示位形空间，用 $W$ 表示工作空间， 用 $\mathcal{O}_{i}$ 表示工作空间中的障碍物。我们使用 $\mathcal{A}$ 来表示机器人，那么，处于位形 $q$ 的机器人所占据的工作空间可被表示为 $\mathcal{A}(q)$ 。会与障碍物碰撞的所有位形所组成的集合成为位形空间障碍（Configuration Space Obstacle），其定义如下：
$$\mathcal{QO}=\{q\in\mathcal{Q} | \mathcal{A}(q) \cap \mathcal{O}\neq\emptyset\}$$
其中
$$\mathcal{O}=\cup\mathcal{O}_{i}$$
那么，无碰撞位形的集合可表示为下面的差集
$$\mathcal{Q}_{free}=\mathcal{Q} \setminus \mathcal{QO}$$
路径规划问题是寻找一条从初始位形 $q_{s}$ 到最终位形 $q_{f}$ 的路径，使得机器人在通过路径时不与任何障碍物发生碰撞。

## 基于势场的路径规划（Planning via Artificial Potentials）
势场法是一种高效的在线规划方法。势场法背后的基本思想是：把机器人当作位形空间中处于人工势场 $U$ 影响下的一个点。势场 $U$ 的构造如下：机器人能够被吸引到最终位形 $q_{f}$，同时因受到排斥而远离障碍区域 $\mathcal{QO}$ 的边界。如果可能的话，构造 $U$ 使得势场 $U$ 中只有单个全局最小值，没有局部最小值。规划过程以增量方式发生：**在每个位形点 $q$ 处，由势场产生的人工作用力定义为该势场的负梯度 $-\nabla U(q)$，表示最有可能的局部运动方向**。

势场是一种合力场，它由引导机器人到达 $q_{f}$ 的引力分量，以及排斥机器人以远离 $\mathcal{QO}$ 边界的斥力分量组成：
$$U(q)=U_{att}(q)+U_{rep}(q)$$
我们将直接在机器人的工作空间中定义势场。特别是，对于一个带有 $n$ 个连杆的机械臂，对于其中的 $n$ 个[DH坐标系](http://blog.csdn.net/philthinker/article/details/73014407)的各原点，我们将定义对应的势场。此类工作空间势场（Workspace Potential Field）将会把DH坐标系的原点吸引到各自的目标位置。
### 引力场（Attractive Potentials）
引力场需要满足几个条件：

 1. $U_{att,i}$ 应该是从$o_{i}$ 到目标位置距离的单调增函数。例如圆锥形势阱（Conic Well Potential）：$U_{att,i}(q)=\|o_{i}(q)-o_{i}(q_{f})\|$，以及抛物线势阱（Parabolic Well Potential）：$U_{att,i}(q)=\frac{1}{2}\xi_{i}\|o_{i}(q)-o_{i}(q_{f})\|^{2}$。
 2. 工作空间内对 $o_{i}$ 的引力等于 $U_{att,i}$ 的负梯度：$F_{att,i}=-\nabla U_{att,i}(q)=-\xi_{i}(o_{i}(q)-o_{i}(q_{f})$。

我们也可以选择将二次型势场和圆锥形势场结合起来：当 $o_{i}$ 远离目标位置时，使用圆锥形势场；当 $o_{i}$ 离目标位置比较近时，使用二次型势场。当然交界处需要对梯度有定义：
$$U_{att,i}(q)=\left\{\begin{aligned}&\frac{1}{2}\xi_{i}\|o_{i}(q)-o_{i}(q_{f})\|^{2},\quad &\|o_{i}(q)-o_{i}(q_{f})\|\leq d \\&d\xi_{i}\|o_{i}(q)-o_{i}(q_{f})\|-\frac{1}{2}\xi_{i}d^{2}, &\|o_{i}(q)-o_{i}(q_{f})\|> d \end{aligned}\right.$$
其中，$d$ 是从圆锥形势场变为二次型势场时所对应的距离。在此情况下，$o_{i}$ 上的工作空间力由下式给出
$$F_{att,i}(q)=\left\{\begin{aligned}&-\xi_{i}(o_{i}(q)-o_{i}(q_{f})),\quad &\|o_{i}(q)-o_{i}(q_{f})\|\leq d \\&-d\xi_{i}\frac{(o_{i}(q)-o_{i}(q_{f}))}{\|o_{i}(q)-o_{i}(q_{f})\|}, &\|o_{i}(q)-o_{i}(q_{f})\|> d \end{aligned}\right.$$

### 斥力场（Repulsive Potential）
斥力场应该满足若干标准。它们应该使机器人排斥并远离障碍物，并且，当机器人离障碍物比较远时，障碍物对机器人运动的影响应该很小或者没有。需要注意的是：仅对DH 坐标系原点定义斥力场，我们无法确保永远不发生碰撞。

我们定义 $\rho_{0}$ 为一个障碍物影响的距离。一个标准的势场函数由下式给出：
$$U_{rep,i}(q)=\left\{\begin{aligned}&\frac{1}{2}\eta_{i}\left(\frac{1}{\rho(o_{i}(q))}-\frac{1}{\rho_{0}}\right)^{2},\quad &\rho(o_{i}(q))\leq \rho_{0} \\&0, &\rho(o_{i}(q))> \rho_{0} \end{aligned}\right.$$
其中， $\rho(o_{i}(q))$ 为 $o_{i}$ 与工作空间内任意障碍物之间的最短距离。排斥力可由下式给出：
$$F_{rep,i}=\eta_{i}\left(\frac{1}{\rho(o_{i}(q))}-\frac{1}{\rho_{0}}\right)\frac{1}{\rho^{2}(o_{i}(q))}\nabla\rho(o_{i}(q))$$
符号 $\nabla\rho(o_{i}(q))$ 表示在 $x=o_{i}(q)$ 处计算得到的梯度值。如果障碍物区域为凸多边形并且 $b$ 为障碍物边界上距离 $o_{i}$ 最近的点，那么，$\rho(o_{i}(q))=\|o_{i}(q)-b\|$，并且它的梯度为
$$\nabla\rho(x)|_{x=o_{i}(q)}=\frac{o_{i}(q)-b}{\|o_{i}(q)-b\|}$$
即从 $b$ 到 $o_{i}(q)$ 的单位向量。

如果障碍物的形状不是凸形的，那么距离函数 $\rho$ 不一定在每个地方都可微，这意味着力向量不连续。有多种方法可以解决这个问题，其中最简单的一种是，确保不同的障碍影响区域不重叠。因此，仅仅对DH坐标系的原点定义斥力场并不能保证机器人不与障碍物发生碰撞。为解决该问题，我们可以使用一组**浮动斥力控制点**（Floating Repulsive Control Point） $o_{float,i}$。通常在各个连杆上设置一个。浮动控制点的定义为连杆边界上距离任何工作空间障碍物最近的点。也可以将非凸障碍在建立势场前将其分解为各个凸的部分。

### 将工作空间力映射到关节力矩
我们已经展示了如何在机器人的工作空间内构建势场，从而在机器人的手臂的DH坐标系原点 $o_{i}$ 处诱导生成人工力（Artificial Force）。下面我们介绍如何将这些力映射到关节力矩。

如果用 $\tau$ 来表示由施加在末端执行器上的工作空间力 $F$ 诱导生成的关节力矩向量，那么
$$J_{v}^{T}F=\tau$$
其中 $J_{v}$ 包括机械臂[雅克比矩阵](http://blog.csdn.net/philthinker/article/details/73744745)的前三行。我们不使用后三行，因为只考虑工作空间中的引力和斥力，而不考虑工作空间中的吸引和排斥力矩。对每个 $o_{i}$ ，必须构建合适的雅克比矩阵。

作用在机械臂上的总人工力矩源自所有引力和斥力势场的人工力矩之和
$$\tau(q)=\sum_{i}J^{T}_{o_{i}}(q)F_{att,i}(q)+\sum_{i}J^{T}_{o_{i}}(q)F_{rep,i}(q)$$
至关重要的是，我们对关节力矩而非工作空间力求和。

### 梯度下降规划
从初始位形开始，在负梯度方向（尽快降低势能的方向）上前进一小步。这提供一种新位形，然后重复该过程，直至达到最终的位形。大致过程如下：

 1. $q^{0} \leftarrow q_{s}$, $i\leftarrow 0$
 2. If $\|q^{i}-q_{f}\|>\epsilon$
	 3. $q^{i+1}\leftarrow q^{i}+\alpha^{i}\frac{\tau(q^{i})}{\|T(q^{i})\|}$
	 4. $i\leftarrow i+1$
 3. Or return $q^{0}, q^{1}, \cdots, q^{i}$
 4. Repeat step 2.

使用该算法必须要做出很多设计选择。$\xi_{i}$ 控制着对点 $o_{i}$ 引力场的相对影响。没有必要将所有 $\xi$ 设置为同一个值。通常我们为一个点分配较大权重，用来生成一个**跟随引导者**（follow the leader）类型的运动，然后机器人重新定位自身是其他点到达最终目的。$\eta_{i}$ 控制着对点 $o_{i}$ 斥力场的相对影响。对于靠近机器人目标位置的障碍物，我们通常将对应的 $\eta_{i}$ 值设置得很小。$\rho_{0}$ 定义了障碍物的影响距离。我们可以对各障碍物定义不同的取值。

困扰所有梯度下降算法的一个问题是势场中可能存在局部最小值。为解决该问题，人们已经开发了诸如模拟退火等概率方法。类似地，机器人路径规划领域也开发了随机化方法。

## 概率规划（Probabilistic Planning）
概率规划属于离线规划，它们需要预先知道关于机器人工作空间中障碍物的几何形状和位置信息。概率规划是一类非常高效地规划方法。它们属于基于抽样（Sampling-based）方法族。其基本思想是：**确定一个能充分表示 $Q_{free}$ 连通性的有限避碰位形集合并利用该集合建立用于解决运动规划问题的路径图**。实现该思路的途径是在**每一步迭代的过程中抽取一个位形样本并检查是否会使机器人与工作空间内的障碍发生碰撞**。如果发生碰撞，则丢弃该样本；否则将其加入当前路径图中并于其它记录的位形建立可行连接。

具体策略有赖于一些特别的设计，区别主要体现在**用什么样的标准选择样本进行碰撞检测**。比如通过规则的栅格采样。但是更可取的办法是随机采样。

### 概率路线图（Probabilistic Road Map, PRM）
为了解决多个路径规划问题，一种方法是构造 $\mathcal{Q}_{free}$ 的一个表示，当遇到新的路径规划问题时，这个表示可被用来快速生成路径。此类的一种表示被称为**位形空间路线图**（Configuration Space Roadmap）。此时，规划算法包括三个阶段：

 1. 在路线图中寻找一条从 $q_{s}$ 到位形 $q_{a}$ 的路径。
 2. 在路线图中寻找一条从 $q_{f}$ 到位形 $q_{b}$ 的路径。
 3. 在路线图中寻找一条从 $q_{a}$ 到位形 $q_{b}$ 的路径。

我们将介绍**概率路线图**（Probabilistic Road Map, PRM）。 PRM是由在节点处相连的简单曲线段或者弧线段组成的一个网络。其中每个节点对应一个位形。两个节点之间的每个弧线段对应两个位形之间的无碰撞路径。下面将详细介绍构建PRM的过程。

**Step 1: 采样**
生成一组随机位形作为路线图中的节点。最简单的方法是对位形空间采用均匀随机采样，然后丢弃 $\mathcal{QO}$ 内部点位形样本。该方法的缺点是：$Q_{free}$ 中任何特定区域的样本数目与该区域的体积成正比，因此不太可能在狭窄的通道内进行。这被称为**狭窄通道问题**（Narrow Passage Problem）。

**Step 2: 连接**
使用一个简单的局部路径规划算法，来生成连接位形对的路径。典型的方法是尝试把各个节点与其 $k$ 个最近邻节点连接起来。

**Step 3: 增强**
加入新的节点和弧线，把路线图中不相连的元素连接起来。增强的目标是最大程度地把不相交的单元连接起来。一种增强方法是：将两个不相交单元中的节点连接起来，采用较复杂的规划算法。另一种方法是：将更多随机节点添加到路线图中。

![概率图](http://img.blog.csdn.net/20170711125221194?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

PRM方法不利的一面在于它只是**概率完全**（Probabilistically Complete）的，即当求解规划问题的计算时间趋于无穷大时，求得一个解（如果存在）的概率趋于1。这意味着：如果问题无解，算法运行时间不确定，需要加一个最大迭代次数以保证算法终止。

### 快速搜索随机树（Rapidly-Exploring Random Tree, RRT）
与PRM这样多重查询规划方法不同，单次查询的方法针对快速求解规划问题的一个特定情况。这种方法并不依赖产生一个尽可能表征自由位形空间的连通的路网图，而是在 $Q_{free}$ 中探索一个与有把握解决该问题相关的子集。这使得计算量大幅减少。
单次查询方法的一个典型例子是双向RRT算法，该方法使用一个被称为RRT的数据结构，记为 $T$。一个RRT的增量扩张依靠每步迭代中重复一个随机化的程序来实现。如下图所示：
![RRT](https://img-blog.csdn.net/20180820202344166?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BoaWx0aGlua2Vy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**Step 1: 采样**
依据一个空间中的随机采样生成随机位形 $q_{rand}$（与PRM类似）。

**Step 2: 连接**
在 $T$ 上找到最靠近 $q_{rand}$ 的位形 $q_{near}$，并在从 $q_{near}$ 到 $q_{rand}$ 的连线上找到一个候选点 $q_{new}$，使之到 $q_{near}$ 的距离为预先给定的值 $\delta$。接下来进行碰撞检测，如果 $q_{near}$ 到 $q_{new}$ 的连线段属于 $Q_{free}$，则将该连线段和 $q_{new}$ 添加到 $T$ 中。

注意 $q_{rand}$ 并未被添加到 $T$ 中，它只是标识 $T$ 的生长方向。为加速搜索过程，双向RRT方法采用两个分别以 $q_{s}$ 和 $q_{g}$ 为根节点的树 $T_{s}$ 和 $T_{g}$。迭代到一定步数的时候，算法进入下一步：

**Step 3: 两棵树连接**
通过产生一个 $q_{new}$ 作为 $T_{s}$ 的生长点并尝试将 $T_{g}$ 连接到该点。此时的计算步长是可变的而非常数 $\delta$。

双向RRT方法也是概率完全的。对于不满足自由运动假设的机器人——比如非完整约束的机器人——基于RRT的方法也是适用的。

## 其它规划方法
机器路径规划方法还包括**基于回缩的路径规划（Planning via Retraction）**方法和**基于单元分解的路径规划（Planning via Cell Decomposition）**方法等，此处不作介绍。

----------

- Thanks Mark W. Spong for his great work of *Robot Modelling and Control*.
- Thanks Steven M. LaValle for his great work of *Planning Algorithm*.
- Thanks Bruno Siciliano et al for their great work of *Robotics: Modelling, Planning and Control*.