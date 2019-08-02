#Actor - Critic
A class of algorithms that precede Q-Learning and SARSA are **actor - critic** methods. Refer to 

> V. Konda and J. Tsitsiklis: *Actor -critic algorithms.* SIAM Journal on Control and Optimization 42(4), 1143-1166 (2003).

for more details. We first give some basics:
##Policy Gradient Methods & Policy Gradient Theorem
We consider methods for learning the policy weights based on the gradient of some performance measure $\eta(\theta)$ with respect to the policy weights. These methods maximize performance, so their updates approximate gradient ascent in $\eta$:
$$\theta_{t+1}=\theta_{t}+\alpha\widehat{\nabla\eta(\theta_{t})}$$
All methods that follow this general schema are known as *policy gradient methods*, whether or not they also learn an approximate value function. A value function may still be used to learn the policy weight $\theta\in\mathbb{R}^{n}$, but may not be required for action selection. 

We consider the episodic case only, in which performance is defined as the value of the start state under the parameterized policy, $\eta(\theta)=v_{\pi_{\theta}}(s_{0})$. (In continuing case, the performance is defined as the average reward rate.) In policy gradient methods, the policy can be parameterized in any way as long as $\pi(a|s,\theta)$ is differentiable with respect to its weights, i.e. $\nabla_{\theta}\pi(a|s,\theta)$ always exists and is finite. To ensure exploration, we generally require the policy never becomes deterministic.

Without losing any meaningful generality, we assume that every episode starts in some particular state $s_{0}$. Then we define performance as:
$$\eta(\theta)=v_{\pi_{\theta}}(s_{0})$$
The **policy gradient theorem** is that
$$\nabla\eta(\theta)=\sum_{s}d_{\pi}(s)\sum_{a}q_{\pi}(s,a)\nabla_{\theta}\pi(a|s,\theta)$$
where the gradients in all cases are the column vectors of partial derivatives with respect to the components of $\theta$, $\pi$ denotes the policy corresponding to weights vector $\theta$ and the distribution $d_\pi$ here is the expected number of time steps $t$ on which $S_{t}=s$ in a randomly generated episode starting in $s_{0}$ and following $\pi$ and the dynamics of the MDP.

##REINFORCE
Now we have an exact expression for updating gradient. We need some way of sampling whose expectation equals or approximates this expression. Notice that the right-hand side is a sum over states weighted by how often the states occurs under the target policy $\pi$ weighted again by $\gamma$ times how many steps it takes to get to those states. If we just follow $\pi$ we will encounter states in these proportions, which we can then weighted by $\gamma^{t}$ to preserve the expected value:
$$\nabla\eta(\theta)=\mathbb{E}_{\pi}\left[ \gamma^{t}\sum_{a}q_{\pi}(S_{t},a)\nabla_{\theta}\pi(a|S_{t},\theta) \right]$$
Then we approximate a sum over actions:
$$\begin{split}\nabla\eta(\theta)&=\mathbb{E}_{\pi}\left[\gamma^{t}\sum_{a}q_{\pi}(S_{t},a)\pi(a|S_{t},\theta)\frac{\nabla_{\theta}\pi(a|S_{t},\theta)}{\pi(a|S_{t},\theta)}\right]\\ &=\mathbb{E}_{\pi}\left[\gamma^{t}q_{\pi}(S_{t},A_{t})\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}\right]\\ 
&=\mathbb{E}_{\pi}\left[\gamma^{t}G_{t}\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}\right]\end{split}$$
Using the sample to instantiate the generic stochastic gradient ascent algorithm, we obtain the update:
$$\theta_{t+1}=\theta_{t}+\alpha\gamma^{t}G_{t}\frac{\nabla_{\theta}\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}$$
It is evident that REINFORCE is a type of Monte Carlo Policy Gradient methods. The update increases the weight vector in this direction proportional to the return but inversely proportional to the action probability. The former makes sense because it causes the weights to move most in the directions that favor actions yielding the highest return. The latter makes sense because otherwise actions that are selected frequently are at an advantage and might win out even if they do not yield the highest return.

The update law can also be written as:
$$\theta_{t+1}=\theta_{t}+\alpha\gamma^{t}G_{t}\nabla_{\theta}\log\pi(A_{t}|S_{t},\theta)$$

The **policy gradient theorem** can be generalized to include a comparison of the action value to an arbitrary **baseline** $b(s)$:
$$\nabla\eta(\theta)=\sum_{s}d_{\pi}(s)\sum_{a}(q_{\pi}(s,a)-b(s))\nabla_{\theta}\pi(a|s,\theta)$$
the baseline can be any function, even a random variable, as long as it does not vary with $a$. Then we rewrite the update law to be:
$$\theta_{t+1}=\theta_{t}+\alpha(G_{t}-b(S_{t}))\nabla_{\theta}\log\pi(A_{t}|S_{t},\theta)$$
The baseline leaves the expected value of the update unchanged, but it can have a large effect on its variance. One natural choice is an estimate of the state value $\hat{v}(S_{t},w)$.

##Actor-Critic
Methods that learn approximations to both policy and value functions are called *actor-critic methods*. REINFORCE with baseline methods use value functions only as a baseline, not a critic, i.e. not for bootstrapping. This is a useful distinction, for only through bootstrapping do we introduce bias and an asymptotic dependence on the quality of the function approximation.

Consider one-step action-critic methods. The main appeal of one step methods is that they are fully online and incremental such as TD(0), SARSA(0) and Q-Learning. One-step actor-critic methods replace the full return of REINFORCE with the one-step return:
$$\theta_{t+1}=\theta_{t}+\alpha(R_{t+1}+\gamma\hat{v}(S_{t+1},w)-\hat{v}(S_{t},w))\nabla_{\theta}\log\pi(A_{t}|S_{t},\theta)$$

Full algorithm can be formulated as follows:

 1. Initialize a differentiable policy parameterization $\pi(a|s,\theta)$.
 2. Initialize a differentiable state-value parameterization $\hat{v}(s,w)$.
 3. Set step sizes $\alpha >0, \beta>0$.
 4. Initialize policy weights $\theta$ and state-value weights $w$.
 5. Repeat:
	 6. Initialize $S$ - first state
	 7. $I=1$
	 7. While $S$ is not terminal
		 8. $A \sim \pi(\cdot|S,\theta)$
		 9. Take $A$, observe $S',R$
		 10. $\delta=R+\gamma\hat{v}(S',w)-\hat{v}(S,w)$
		 11. $w=w+\beta\delta\nabla_{w}\hat{v}(S,w)$
		 12. $\theta=\theta+\alpha I \delta\nabla_{\theta}\log\pi(A|S,\theta)$
		 13. $I=\gamma I$
		 14. $S=S'$

##Parameterization for Continuous Actions
Policy based methods offer practical ways of dealing with large actions spaces, even continuous spaces. All of the notes above are dealing with discrete actions, nevertheless, we begin with continuous ones from now on.

Instead of computing learned probabilities for each of many actions, we learn the statistics of the probability distributions. Take the Gaussian distribution for example. The actions set might be the real numbers with actions chosen from a Gaussian distribution:
$$p(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)$$
The value $p(x)$ is the *density* of the probability at $x$. 

To produce a policy parameterization, we can define the policy as the normal probability density over a real-valued scalar action, with mean and standard deviation give by parametric function approximators:
$$\pi(a|s,\theta)=\frac{1}{\sigma(s,\theta)\sqrt{2\pi}}\exp\left(-\frac{(a-\mu(s,\theta))^{2}}{2\sigma(s,\theta)^{2}}\right)$$
Then we divide the policy weight vector into two parts: $\theta=[\theta^{\mu}, \theta^{\sigma}]^{T}$. The mean can be approximated as a linear function:
$$\mu(s,\theta)={\theta^{\mu}}^{T}\phi(s)$$
The standard deviation must always be positive and is better approximated as the exponential of a linear function:
$$\sigma(s,\theta)=\exp\left({\theta^{\sigma}}^{T}\phi(s)\right)$$\
where $\phi(s)$ is a basis function of some type. With these definitions, all the algorithms described above can be applied to learn to select real-valued actions.

#Summary
Actor -Critic, which is a branch of TD methods, keeps a separate policy independent of the value function. The policy is called the *actor* and the value function is called the *critic*. An advantage of having a separate policy representation is that if there are many actions, or when the action space is continuous, there is no need to consider all actions' Q-values in order to select one of them. A second advantage is that they can learn stochastic policies naturally. Furthermore, a prior knowledge about policy constraints can be used. 