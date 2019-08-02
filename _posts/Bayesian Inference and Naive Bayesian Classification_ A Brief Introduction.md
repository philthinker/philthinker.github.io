Thanks Cameron Davidson-Pilon for the great work of *Bayesian Methods for Hackers: Probabilistic programming and Bayesian Inference*.

[TOC]

Bayesian inference is simply updating your beliefs after considering new evidence.

#Popular Probability Distribution
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

#Naive Bayesian Classification
**Naive Bayesian Algorithm**
Given some conditional probability, how to solve the conditional probability when we exchange positions of the two events? Or given $P(A|B)$, how to solve $P(B|A)$?
$$P(B|A)=\frac{P(A|B)P(B)}{P(A)}$$

**Formal Definition**

 1. Let $x=\{a_{1},a_{2},\dots,a_{m}\}$ be a sample, every $a$ stands for an attribute.
 2. Given a classes set $C=\{y_{1},y_{2},\dots,y_{n}\}$.
 3. Calculate $P(y_{1}|x),P(y_{2}|x),\dots,P(y_{n}|x)$.
 4. If $P(y_{k}|x)=\max\{P(y_{1}|x),P(y_{2}|x),\dots,P(y_{n}|x)\}$, then $x\in y_{k}$.

Then, how to calculate the conditional probabilities? Follow the steps below:

 1. Find a training set.
 2. Calculate the conditional probability of each attribute under each class, i.e. $$\begin{aligned}&P(a_{1}|y_{1}), &P(a_{2}|y_{1}), &\quad \cdots, &P(a_{m}|y_{1});\\&P(a_{1}|y_{2}), &P(a_{2}|y_{2}), &\quad \cdots, &P(a_{m}|y_{2});\\&P(a_{1}|y_{n}), &P(a_{2}|y_{n}), &\quad \cdots, &P(a_{m}|y_{n}); \end{aligned}$$
 3. Suppose that those attributes are independent of each other, by Bayesian theorem,$$P(y_{i}|x)=\frac{P(x|y_{i})P(y_{i})}{P(x)}$$ Since the denominator is always a constant, we need only maximize the numerator. By supposition, those attributes are (conditionally) independent of each other, we can write:$$P(x|y_{i})P(y_{i})=P(a_{1}|Y_{i})P(a_{2}|Y_{i})\cdots P(a_{m}|Y_{i})P(y_{i})=P(y_{i})\prod_{j=1}^{m}P(a_{j}|y_{i})$$