Thanks Richard S. Sutton for his masterpiece - *Reinforcement Learning: An Introduction* and some papers.

[TOC]

#Gradient Bandit Algorithm
Most reinforcement learning methods for *multi-armed bandit* problems use *action-value functions* to select actions given a particular state. This is often a good approach, but not the only one possible. Now we consider learning a numerical **reference** for each action $a$, which we denote $H_{t}(a)$. The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. We can determine an action according to a [soft-max distribution](http://blog.csdn.net/philthinker/article/details/70911997) as follows:
$$\mathrm{Pr}\{A_{t}=a\}=\frac{e^{H_{t}(a)}}{\sum_{b=1}^{k}e^{H_{t}(b)}}=\pi_{t}(a)$$ There is natrual learning algorithm for this setting based on **stochastic gradient ascent**. On each step, after selecting action $A_{t}$ and receiving the reward $R_{t}$, preferences are updated by:
$$\begin{split} H_{t+1}(A_{t}) &=H_{t}(A_{t})+\alpha(R_{t}-\bar{R}_{t})(1-\pi_{t}(A_{t})) \\ H_{t+1}(a) &= H_{t}(a)-\alpha(R_{t}-\bar{R}_{t})\pi_{t}(a),\qquad \forall a\neq A_{t} \end{split}$$ where $\bar{R}_{t}$ is the average of all the rewards up through and including time $t$, which can be computed incrementally. The $\bar{R}_{t}$ serves as a **baseline** with which the reward is compared. If the reward is higher than the baseline, then the probability of taking $A_t$ in the future is increased, and if the reward is below baseline, then probability is decreased. The non-selected actions move in the opposite direction.

Let's take a deeper insight into the gradient bandit algorithm by viewing it as a stochastic approximation to [gradient ascent](http://blog.csdn.net/philthinker/article/details/78191864). In exact gradient ascent, each preference $H_{t}(a)$ would be incremented proportional to the increment's effect on the performance:
$$H_{t+1}(a) = H_{t}(a)+\alpha\frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)}$$where the *measure of performance* here is the expected reward
$$\mathbb{E}[R_{t}]=\sum_{b}\pi_{t}(b)q_{*}(b)$$Since we do not know the $q_{*}(b)$ exactly, we have to find an alternative update that is equal to the original one in expected value. 
$$\begin{split} \frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)} &= \frac{\partial}{\partial H_{t}(a)}\left[ \sum_{b}\pi_{t}(b)q_{*}(b) \right] \\ &=\sum_{b}q_{*}(b)\frac{\partial \pi_{t}(b)}{\partial H_{t}(a)} \\ &= \sum_{b}(q_{*}(b)-X_{t})\frac{\partial \pi_{t}(b)}{\partial H_{t}(a)}  \end{split}$$ where $X_t$ can be any scalar that does not depend on $b$. We can include it here because the gradient sums to zero over all the actions, $\sum_{b}\frac{\partial \pi_{t}(b)}{\partial H_{t}(a)}=0$. As $H_{t}(a)$ is changed, some actions' probabilities go up and some down, but the sum of the changes must be zero because the sum of the probabilities must remain one.
$$\begin{split} \frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)} &= \sum_{b}\pi_{t}(b)(q_{*}(b)-X_{t})\frac{\partial \pi_{t}(b)}{\partial H_{t}(a)}/\pi_{t}(b) \\ &= \mathbb{E}\left[ (q_{*}(A_{t})-X_{t})\frac{\partial \pi_{t}(A_{t})}{\partial H_{t}(a)}/\pi_{t}(A_{t}) \right] \\ &=\mathbb{E}\left[ (R_{t}-\bar{R}_{t})\frac{\partial \pi_{t}(A_{t})}{\partial H_{t}(a)}/\pi_{t}(A_{t}) \right]  \end{split}$$where here we have chosen $X_{t}=\bar{R}_{t}$ and substituted $R_{t}$ for $q_{*}(A_{t})$ which is permitted because $\mathbb{E}[R_{t}|A_{t}]=q_{*}(A_{t})$. 

$$\begin{split} \frac{\partial\pi_{t}(A_{t})}{\partial H_{t}(a)} &= \frac{\partial}{\partial H_{t}(a)}\left[\frac{e^{H_{t}(A_{t})}}{\sum_{c=1}^{k}e^{H_{t}(c)}}\right] \\ &= \frac{\frac{\partial e^{H_{t}(A_{t})}}{\partial H_{t}(a)}\sum_{c=1}^{k}e^{H_{t}(c)}-e^{H_{t}(A_{t})}\frac{\partial \sum_{c=1}^{k}e^{H_{t}(c)}}{\partial H_{t}(a)}}{\left(\sum_{c=1}^{k}e^{H_{t}(c)}\right)^{2}} \\ &= \frac{\mathbb{I}_{a=A_{t}}e^{H_{t}(A_{t})}\sum_{c=1}^{k}e^{H_{t}(c)}-e^{H_{t}(A_{t})}e^{H_{t}(a)}}{\left(\sum_{c=1}^{k}e^{H_{t}(c)}\right)^{2}} \end{split}$$ where $\mathbb{I}_{a=b}=\left\{\begin{aligned}&1,\quad \text{if }a=b\\ &0,\quad \text{otherwise}\end{aligned}\right.$. Then
$$\begin{split} \frac{\partial\pi_{t}(A_{t})}{\partial H_{t}(a)} &=\frac{\mathbb{I}_{a=A_{t}}e^{H_{t}(A_{t})}}{\sum_{c=1}^{k}e^{H_{t}(c)}}-\frac{e^{H_{t}(A_{t})}e^{H_{t}(a)}}{\left(\sum_{c=1}^{k}e^{H_{t}(c)}\right)^{2}} \\ &=\mathbb{I}_{a=A_{t}}\pi_{t}(A_{t})-\pi_{t}(A_{t})\pi_{t}(a) \\ &= \pi_{t}(A_{t})(\mathbb{I}_{a=A_{t}}-\pi_{t}(a)) \end{split}$$

Thus
$$\begin{split} \frac{\partial \mathbb{E}[R_{t}]}{\partial H_{t}(a)} &= \mathbb{E}\left[(R_{t}-\bar{R}_{t})\pi_{t}(A_{t})(\mathbb{I}_{a=A_{t}}-\pi_{t}(a))/\pi_{t}(A_{t})\right] \\ &=\mathbb{E}\left[(R_{t}-\bar{R}_{t})(\mathbb{I}_{a=A_{t}}-\pi_{t}(a)) \right]  \end{split}$$

Our plan has been to write the performance gradient as an expectation of something that we can sample on each step, as we have just done, and then update on each step proportional to the sample. By substituting a sample of the expectation above for the performance gradient yields:
$$H_{t+1}(a) = H_{t}(a)+\alpha(R_{t}-\bar{R}_{t})(\mathbb{I}_{a=A_{t}}-\pi_{t}(a)),\quad \forall a$$ which is exactly what we have given at the beginning. 

#Policy Approximation
To begin with, let us focus on the **episodic** case with **discrete action spaces** in a MDP. Here the performance is defined as the value of the start state under the parameterized policy 
$$\eta(\theta)=v_{\pi_{0}}(x_{0})$$
In policy gradient methods, the policy can be parameterized in any way as long as $\nabla_{\theta}\pi(a|s,\theta)$ exists and is always finite. In practice, to ensure exploartion we generally require that the poliyc never becomes deterministic.

For simplicity, we can form parameterized numerical preferences $h(s,a,\theta)\in\mathbb{R}$ for each state-action pair. For example: the softmax distribution[^footnote]
[^footnote]: An immediate advantages of selecting actions according to the softmax in action preferences is that the approximatepolicy can approach determinism.

$$\pi(a|s,\theta)=\frac{\exp(h(s,a,\theta))}{\sum_{b}\exp(h(s,b,\theta))}$$
The perferences themselves can be parameterized arbitrarily, such as the simplest linear features:
$$h(s,a,\theta)=\theta^{T}\phi(s,a)$$

#Policy Gradient Theorem
WIth function approximation, it is challenging to change the policy weights in a way that ensures improvement. The effect of the policy on the state distribution is completely a function of the environment and is typically unknown. How can we estimate the performance w.r.t. the policy weights when the gradient depends on the unknown effect of changing the policy on the state distribution? 
This brings us to the policy gradient theorem, which provides us an analytical expression for the gradient of the performance w.r.t. the policy weights that does involve the derivative of the state distribution. The **policy gradient theorem** is that
$$\nabla\eta(\theta)=\sum_{s}d_{\pi}(s)\sum_{a}q_{\pi}(s,a)\nabla_{\theta}\pi(a|s,\theta)$$
In the episodic case, $d_{\pi}(s)$ is defined to be the expected number of time steps $t$ on which $S_{t}=s$ in a randomly generated episode starting in $s_{0}$ and following $\pi$ and the dynamics of the MDP.

#REINFORCE
Now all we need is some way of sampling to approximate. Notice that the right hand side policy gradient theorem equation is a sum over states weighted by how often the states occurs under the target policy $\pi$ weighted again by $\gamma$ times how many steps it takes to get to those states. Thus
$$\nabla\eta(\theta)=\mathbb{E}_{\pi}\left[\gamma^{t}\sum_{a}q_{\pi}(S_{t},a)\nabla_{\theta}\pi(a|S_{t},\theta)\right]$$
Then the remaining part of the expectation above is a sum over actions.If only each term was weighted by the probability of selecting the actions, that is, according to $\pi(a|S_{t},\theta)$. So let us make it that way:
$$\nabla\eta(\theta)=\mathbb{E}_{\pi}\left[\gamma^{t}q_{\pi}(S_{t},A_{t})\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}\right]$$
where $q(\cdot)$ is also unknown for sure. However, we can approximate it by Monte Carlo method. Then
$$\nabla\eta(\theta)=\mathbb{E}_{\pi}\left[\gamma^{t}G_{t}\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}\right]$$
which is exactly what we want, a quantity that we can sample on each time step whose expectation is equal to the gradient. By the gradient ascent rule:
$$\theta_{t+1}=\theta_{t}+\alpha\gamma^{t}G_{t}\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}$$
The update steps are given below:

 1. Initialize: $\pi(a|s,\theta)$, $\forall a\in\mathcal{A}, s\in\mathcal{S}, \theta\in\mathbb{R}^{n}$
 2. Initialize policy weights $\theta$
 3. Repeat forever:
	 4. Generate an episdoe $S_{0}, A_{0}, R_{1}, \dots, S_{T-1}, A_{T-1}, R_{T}$ following $\pi(\cdot|\cdot,\theta)$.
	 5. For each step of the episode $t=0,1,\dots,T-1$:
		 6. $G_{t}\leftarrow$return from step $t$.
		 7. $\theta\leftarrow\theta+\alpha\gamma^{t}G_{t}\nabla_{\theta}\log\pi(A_{t}|S_{t},\theta)$

The vector $\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}$, or in the compact form $\nabla_{\theta}\log\pi(A_{t}|S_{t},\theta)$, is the only place the policy parameterization appears in the algorithm, which we refer to simply as the *eligibility vector*. In the softmax policies with linear action preferences, the eligibility vector is 
$$\nabla_{\theta}\log\pi(A_{t}|S_{t},\theta)=\phi(s,a)-\sum_{b}\pi(b|,s,\theta)\phi(s,b)$$

Refer to [here](http://blog.csdn.net/philthinker/article/details/71104095) for more details about REINFORCE algorithm.

#Actor-Critic Method
In reinforcement learning systems, only through bootstrapping do we introduce bias and an asymptotic dependence on the qualilty of the function approximation. REINFORCE is unbiased and will converge asymptotically to a local minimum. 

Here we introduce Actor-Critic methods with a true bootstraping critic.
Consider the one-step actor-critic methods firstly. The main appeal of them is that they are fully online and incremental. One-step actor-critic methods replace the full return of REINFORCE with the one-step return and use a learned state-value function as the baseline:
$$\begin{split}\theta_{t+1} &= \theta_{t}+\alpha\left(G_{t}^{(1)}-\hat{v}(S_{t},w)\right)\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)} \\ &= \theta_{t}+\alpha\left( R_{t+1}+\gamma\hat{v}(S_{t+1},w)-\hat{v}(S_{t},w) \right)\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}\end{split}$$

The precise steps go as follows:

 1. Initialize $\pi(a|s,\theta), \forall a\in\mathcal{A}, s\in\mathcal{S}, \theta\in\mathbb{R}^{n}$
 2. Initialize $\hat{v}(s,w), \forall s\in\mathcal{S}, w\in\mathbb{R}^{m}$
 3. Repeat forever:
	 4. Initialize the first state of episode $S$
	 5. $I\leftarrow 1$
	 6. While $S$ is not terminal:
		 7. Take action $A\sim\pi(\cdot|S,\theta)$, observe $S', R$
		 8. $\delta\leftarrow R+\gamma\hat{v}(S',w)-\hat{v}(S,w)$, (if $S'$ is terminal, $\hat{v}(S',w)=0$)
		 9. $w\leftarrow w+\beta\delta\nabla_{w}\hat{v}(S,w)$
		 10. $\theta\leftarrow \theta+\alpha I\delta\nabla_{\theta}\log\pi(A|S,\theta)$
		 11. $I\leftarrow \gamma I$
		 12. $S\leftarrow S'$

The forward view of multi-step methods can be found [here](http://blog.csdn.net/philthinker/article/details/72519083).

Actor-Critic Algorithm with Eligibility Traces:

 1. Initialize $\pi(a|s,\theta), \forall a\in\mathcal{A}, s\in\mathcal{S}, \theta\in\mathbb{R}^{n}$
 2. Initialize $\hat{v}(s,w), \forall s\in\mathcal{S}, w\in\mathbb{R}^{m}$
 3. Repeat forever:
	 4. Initialize the first state of episode $S$
	 5. $e^{\theta}=0$, $e^{w}=0$.
	 5. $I\leftarrow 1$
	 6. While $S$ is not terminal:
		 7. Take action $A\sim\pi(\cdot|S,\theta)$, observe $S', R$
		 8. $\delta\leftarrow R+\gamma\hat{v}(S',w)-\hat{v}(S,w)$, (if $S'$ is terminal, $\hat{v}(S',w)=0$)
		 9. $e^{w}\leftarrow \lambda^{w}e^{w}+I\nabla_{w}\hat{v}(S,w)$
		 10. $e^{\theta}\leftarrow \lambda^{\theta}e^{\theta}+I\nabla_{\theta}\log\pi(A|S,\theta)$
		 9. $w\leftarrow w+\beta\delta\nabla_{w}\hat{v}(S,w)$
		 10. $\theta\leftarrow \theta+\alpha I\delta\nabla_{\theta}\log\pi(A|S,\theta)$
		 11. $I\leftarrow \gamma I$
		 12. $S\leftarrow S'$

Refer to [here](http://blog.csdn.net/philthinker/article/details/71104095) for more details about Actor-Critic algorithm.

