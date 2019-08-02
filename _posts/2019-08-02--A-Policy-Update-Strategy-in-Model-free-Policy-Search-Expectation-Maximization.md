---
layout: post
title:  "A Policy Update Strategy in Model-Free Policy Search - Expectation-Maximization"
date:   2019-08-02 12:43:19 +0800
categories: jekyll update
---

[TOC]

#Expectation-Maximization Algorithm
[Policy gradient](http://blog.csdn.net/philthinker/article/details/77452891) methods require the user to specify the learning rate which can be problematic and often results in an unstable learning process or slow convergence. By formualting policy search as an **inference problem** with **latent variables** and using the EM algorithm to infer a new policy, this problem can be avoided since no learning rate is required.

The standard EM algorithm, which is well-known for determining the maximum likelihood solution of a probabilitistic **latent variable model**, takes the parameter update as a **weighted maximum likelihood estimate** which has a closed form solution for most of the used polices. 

Let's assume that :
$$\begin{split} y &: \text{ observed random variable }\\ z &:\text{ unobserved random variable }\\ p_{\theta}(y,z) &: \text{ parameterized joint distribution } \end{split}$$ Given a data set $Y=[y^{[1]},\dots,y^{[N]}]^{T}$, we wanna approximate the parameter $\theta$ which means maximizing the **log likelihood**:
$$\max_{\theta}\log p_{\theta}(Y,Z)$$ Since  $Z$ is latent variable we cannot solve the maximization problem directly. But by computing the expectation of $Z$, we can maximize the  **log-marginal likelihood** of $Y$:
$$\log p_{\theta}(Y)=\sum_{i=1}^{N}\log p_{\theta}(y^{[i]})=\sum_{i=1}^{N}\log \int p_{\theta}(y^{[i]},z)dz$$ However, it is evident that we cannot obtain a **closed-form solution** for the parameter $\theta$ of our probability model $p_{\theta}(y,z)$. Do you know what is closed-form solution? 

[Closed form solution](http://mathworld.wolfram.com/Closed-FormSolution.html) :
> An equation is said to be a closed-form solution if it solves a given problem in terms of functions and mathematical operations from a given generally-accepted set. For example, an infinite sum would generally not be considered closed-form. However, the choice of what to call closed-form and what not is rather arbitrary since a new "closed-form" function could simply be defined in terms of the infinite sum.

**EM(Expectation-Maximization)** is a powerful method to estimate parameterized latent variables. The basic idea behind is that *if parameter $\theta$ is known then we can estimate the optimal latent variable $Z$ in view of $Y$ (E-Step); If latent variable $Z$ is known, we can estimate$\theta$ by maximum likelihood estimation.* EM method can be seen as a kind of **coordinate descent** method to maximize the lower-bound of the log likelihood.



The iterative procedure for estimating the maximum log-likelihood consists of two main segments: **Expectation Steps** and **Maximization Steps** as mentioned above. Assume that we begin at the $\theta^{0}$. Then we execute the following iterative steps:

 - Based on $\theta^{t}$ estimating the expectation of the latent variable $Z^{t}$.
 - Based on $Y$ and $Z^{t}$ estimating the parameter $\theta^{t+1}$ by maximum likelihood estimation.

In general, we do not feel like the expectation of $Z$ but the distribution of $Z$, i.e. $p_{\theta^{t}}(Z | Y)$. To be specific, let's introduce an **auxiliary distribution** $q(Z)$, which is variational, to decompose the marginal log-likelihood by using the identity $p_{\theta}(Y)=p_{\theta}(Y,Z)/p_{\theta}(Z|Y)$ :
$$\begin{split}\log p_{\theta}(Y) =& \log p_{\theta}(Y)\int q(Z)dZ \\ =& \int q(Z)\log p_{\theta}(Y)dZ \\ =& \int q(Z)\log\frac{q(Z)p_{\theta}(Y,Z)}{q(Z)p_{\theta}(Z|Y)}dZ \\ =&\int q(Z)\log\frac{p_{\theta}(Y,Z)}{q(Z)}dZ + \int q(Z)\log\frac{q(Z)}{p_{\theta}(Z|Y)}dZ \\ =& \mathcal{L}_{\theta}(q)+KL(q(Z)\|p_{\theta}(Z|Y)) \end{split}$$ Since the [KL divergence](http://blog.csdn.net/philthinker/article/details/70172905) is always larger or equal to zero, the term $\mathcal{L}_{\theta}(q)$ is a lower bound of the log-marginal likelihood. 

##E-Step
In E-step we update the variational distribution $q(Z)$ by minimizing the KL divergence $KL(q(Z)\|p_{\theta}(Z|Y))$, i.e. setting $q(Z)=p_{\theta}(Z|Y)$ . Note that the value of the log-likelihood $\log p_{\theta}(Y)$ has nothing to do with the variational distribution $q(Z)$. In summary, E-step :
$$\text{Update } q(Z) \Leftarrow \text{ Minimize } KL(q(Z)\|p_{\theta}(Z|Y)) \Leftarrow \text{ Set } q(Z)=p_{\theta}(Z|Y)$$

##M-Step
In M-step we optimize the lower bound w.r.t. $\theta$ , i.e. 
$$\begin{split}\theta_{new} =& \arg\max_{\theta}\mathcal{L}_{\theta}(q) \\ =&\arg\max_{\theta} \int q(Z)\log\frac{p_{\theta}(Y,Z)}{q(Z)}dZ \\ =& \arg\max_{\theta}\int q(Z)\log p_{\theta}(Y,Z)dZ + H(q) \\=&\arg\max_{\theta}\mathbb{E}_{q(Z)}[\log p_{\theta}(Y,Z)] \\ =&\arg\max_{\theta}\mathcal{Q}_{\theta}(q) \end{split}$$ where $H(q)$ denotes the **entropy** of $q$ , $\mathcal{Q}$ is the **expected complete data log-likelihood**. The log now acts on the joint distribution directly. So M-step can be obtained in closed form. Moreover, 
$$\mathcal{Q}_{\theta}(q)=\int q(Z)\log p_{\theta}(Y,Z)dZ = \sum_{i=1}^{N}\int q_{i}(z)\log p_{\theta}(y^{[i]},z)dZ$$ The M-step is based on a **weighted maximum likelihood estimate** of $\theta$ using the complete data points $[y^{[i]},z]$ weighted by $q_{i}(z)$. In summary, M-step:
$$\text{Update } \theta \Leftarrow \text{ Maximize }\mathcal{Q}_{\theta}(q) $$

#Reformulate Policy Search as an Inference Problem
Let's assume that :
$$\begin{split} \text{Binary reward event }R &: \text{ observed variable} \\ \text{Trajectory }\tau &: \text{ unobserved variable} \end{split}$$ Maximizing the reward implies maximizing the probability of the reward event, and, hence, our trjectory distribution $p_{\theta}(\tau)$ needs to assign high probability to trajectories with high reward probability $p(R=1|\tau)$ . 

We would like to find a parameter vector $\theta$ that maximizes the probability of the reward event, i.e. 
$$\log p_{\theta}(R)=\int_{\tau}p(R|\tau)p_{\theta}(\tau)d\tau$$ As for the standard EM algorithm, a variational distribution $q(\tau)$ is used to decompose the log-marginal likelihood into tow terms:
$$\log p_{\theta}(R)=\mathcal{L}_{\theta}(q)+KL(q(\tau)\|p_{\theta}(\tau|R))$$ where the **reward-weighted** trajectory distribution: 
$$p_{\theta}(\tau|R)=\frac{p(R|\tau)p_{\theta}(\tau)}{p_{\theta}(R)}=\frac{p(R|\tau)p_{\theta}(\tau)}{\int p(R|\tau)p_{\theta}(\tau)d\tau} \propto p(R|\tau)p_{\theta}(\tau)$$ 

##E-Step
$$\text{Update }q(\tau) \Leftarrow \text{ Minimize }KL(q(\tau)\|p_{\theta}(\tau|R)) \Leftarrow \text{ Set }q(\tau)=p_{\theta}(\tau|R) $$

##M-Step
$$\begin{split} \theta_{new} &= \arg\max_{\theta}\mathcal{L}_{\theta}(q) \\ &= \arg\max_{\theta}\int q(\tau)\log\frac{p_{\theta}(R,\tau)}{q(\tau)}d\tau \\ &=\arg\max_{\theta}\int q(\tau)\log p_{\theta}(R,\tau) d\tau + H(q) \\ &=\arg\max_{\theta}\underbrace{\int q(\tau)\log( p(R|\tau)p_{\theta}(\tau)) d\tau}_{\mathcal{Q}_{\theta}(q)}\\ &=\arg\max_{\theta}\int q(\tau)\log p_{\theta}(\tau)d\tau+f(q) \\ &= \arg\min_{\theta}\int q(\tau)(-\log p_{\theta}(\tau))d\tau \\ &=\arg\min_{\theta}\left[\int q(\tau)\log\frac{q(\tau)}{p_{\theta}(\tau)}d\tau +  \int q(\tau)\log\frac{1}{q(\tau)}d\tau \right] \\ &=\arg\min_{\theta}KL(q(\tau)\| p_{\theta}(\tau)) \end{split}$$ i.e. 
$$\text{Update } \theta \Leftarrow \text{ Maximize }\mathcal{Q}_{\theta}(q) \Leftarrow \text{ Minimize }KL(q(\tau)\|p_{\theta}(\tau)) $$

#EM-based Policy Search Algorithms
##Monte-Carlo EM-based Policy Search
MC-EM-algorithm uses a sample-based approximation for the variational distribution $q$, i.e. in the E-step, MC-EM minimizes the KL divergence $KL(q(Z)\|p_{\theta}(Z|Y))$ by using samples $Z_{j}\sim p_{\theta}(Z|Y)$. Subsequently, these samples $Z_{j}$ are used to estimate the expectation of the complete data log-liklihood:
$$\mathcal{Q}_{\theta}(q)=\sum_{j=1}^{K}\log p_{\theta}(Y,Z_{j})$$ In terms of policy search, MC_EM methods use samples $\tau^{[i]}$ from the old trajectory distribution $p_{\theta'}$ to represent the variational distribution $q(\tau)\propto p(R|\tau)p_{\theta'}(\tau)$ over trajectories. As $\tau^{[i]}$ has already been sampled from $p_{\theta'}(\tau)$, $q(\tau^{[i]})\propto p(R|\tau^{[i]})$ . Consequently, in the M-step, we maximize:
$$\mathcal{Q}_{\theta}(\theta')=\sum_{\tau^{[i]}\sim p_{\theta'}(\tau)}p(R|\tau^{[i]})\log p_{\theta}(\tau^{[i]})$$

There are **Episode-based EM-algorithms** such as **Reward-Weighted Regression**(RWR) and **Cost-Regularized Kernel Regression**(CrKR), and **Step-based EM-algorithms** such as **Episodic Reward-Weighted Regression**(eRWR) and **Policy Learning by Weighting Exploration with Returns**(PoWER). 

##Variational Inference-based Methods
The MC-EM approach uses a weighted maximum likelihood estimate to obtain the new parameters $\theta$ of the policy. It averages over several modes of the reward function. Such a behavior might result in slow convergence to good policies as the average of several modes might be in an area with low reward. 

The maximization used for the MC-EM approach is equivalent to minimizing: 
$$KL(p(R|\tau)p_{\theta'}(\tau)\|p_{\theta}(\tau))=\int p(R|\tau^{[i]})p_{\theta'}(\tau^{[i]})\log \frac{p(R|\tau)p_{\theta'}(\tau^{[i]})}{p_{\theta}(\tau^{[i]})}$$ w.r.t. parameter $\theta$ . This minimization is also called the **Moment Projection** of the reward-weighted trajectory distribution as it matches the moments of $p_{\theta}(\tau)$ with the moments of $p(R|\tau)p_{\theta'}(\tau)$ . 

Alternatively, we can use the **Information projection** $\arg\min_{\theta}KL(p_{\theta}(\tau)\| p(R|\tau)p_{\theta'}(\tau))$ to update the policy. This projection forces the new trajectory distribution $p_{\theta}(\tau)$ to be zero everywhere where the reward-weighted trajectory distribution is zero. 

 - Thanks J. Peters et al for their great work of *A Survey on Policy Search for Robotics* . 
 - 感谢周志华——《机器学习》清华大学出版社