
[TOC]

1. Logistic Regression
--

Likelihood function, as interpreted by wikipedia:
> https://en.wikipedia.org/wiki/Likelihood_function

plays one of the key roles in statistic inference, especially methods of estimating a parameter from a set of statistics. In this article, we'll make full use of it.
Pattern recognition works on the way that learning the posterior probability $p(y|x)$ of pattern $x$ belonging to class $y$. In view of a pattern $x$, when the posterior probability of one of the class $y$ achieves the maximum, we can take $x$ for class $y$, i.e.$$\hat{y}=\arg\max_{y=1,\dots,c}p(u|x)$$
The posterior probability can be seen as the credibility of model $x$ belonging to class $y$.
In Logistic regression algorithm, we make use of linear logarithmic function to analyze the posterior probability:
$$q(y|x,\theta)=\frac{\exp\left( \sum_{j=1}^{b}\theta_{j}^{(y)}\phi_{j}(x) \right)}{\sum_{y'=1}^{c}\exp\left( \sum_{j=1}^{b}\theta_{j}^{(y')}\phi_{j}(x) \right)}$$
Note that the denominator is a kind of regularization term. Then the Logistic regression is defined by the following optimal problem:
$$\max_{\theta}\sum_{i=1}^{m}\log q(y_{i}|x_{i},\theta)$$
We can solve it by gradient descent method:

 1. Initialize $\theta$.
 2. Pick up a training sample $(x_{i},y_{i})$ randomly.
 3. Update $\theta=({\theta^{(1)}}^{T},\dots, {\theta^{(c)}}^{T})^{T}$ along the direction of gradient ascent:$$\theta^{(y)}\leftarrow \theta^{(y)}+\epsilon\nabla_{y}J_{i}(\theta),\quad y=1,\dots,c$$where $$\nabla_{y}J_{i}(\theta)=-\frac{\exp\left( {\theta^{(y)}}^{T}\phi(x_{i}) \right)\phi(x_{i})}{\sum_{y'=1}^{c}\exp\left( {\theta^{(y')}}^{T}\phi(x_{i}) \right)}+\left\{\begin{aligned} &\phi(x_{i})\quad &(y=y_{i})\\ &0 &(y\neq y_{i}) \end{aligned}\right.$$
 4. Go back to step 2,3 until we get a $\theta$ of suitable precision.

Take the Gaussian Kernal Model as an example: $$q(y|x,\theta) \propto \exp\left( \sum_{j=1}^{n}\theta_{j}K(x,x_{j}) \right)$$
Aren't you familiar with Gaussian Kernal Model? Refer to this article:

> http://blog.csdn.net/philthinker/article/details/65628280

Here are the corresponding MATLAB codes:

```MATLAB
n=90; c=3; y=ones(n/c,1)*(1:c); y=y(:);
x=randn(n/c,c)+repmat(linspace(-3,3,c),n/c,1);x=x(:);

hh=2*1^2; t0=randn(n,c);
for o=1:n*1000
    i=ceil(rand*n); yi=y(i); ki=exp(-(x-x(i)).^2/hh);
    ci=exp(ki'*t0); t=t0-0.1*(ki*ci)/(1+sum(ci));
    t(:,yi)=t(:,yi)+0.1*ki;
    if norm(t-t0)<0.000001
        break;
    end
    t0=t;
end

N=100; X=linspace(-5,5,N)';
K=exp(-(repmat(X.^2,1,n)+repmat(x.^2',N,1)-2*X*x')/hh);

figure(1); clf; hold on; axis([-5,5,-0.3,1.8]);
C=exp(K*t); C=C./repmat(sum(C,2),1,c);
plot(X,C(:,1),'b-');
plot(X,C(:,2),'r--');
plot(X,C(:,3),'g:');
plot(x(y==1),-0.1*ones(n/c,1),'bo');
plot(x(y==2),-0.2*ones(n/c,1),'rx');
plot(x(y==3),-0.1*ones(n/c,1),'gv');
legend('q(y=1|x)','q(y=2|x)','q(y=3|x)');
```
![这里写图片描述](http://img.blog.csdn.net/20170408150856039?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


2. Least Square Probability Classification
---------------------------------------
In LS probability classifiers, linear parameterized model is used to express the posterior probability:$$q(y|x,\theta^{(y)})=\sum_{j=1}^{b}\theta_{j}^{(y)}\phi_{j}(x)={\theta^{(y)}}^{T}\phi(x),\quad y=1,\dots,c$$
These models depends on the parameters $\theta^{(y)}=（\theta_{1}^{(y)},\dots, \theta_{b}^{(y)}）^{T}$ correlated to each classes $y$ that is diverse from the one used by Logistic classifiers. Learning those models means to minimize the following quadratic error:$$\begin{split}J_{y}(\theta^{(y)})= & \frac{1}{2}\int\left( q(y|x,\theta^{(y)})-p(y|x) \right)^{2}p(x)\mathrm{d}x \\ =& \frac{1}{2}\int q(y|x,\theta^{(y)})^{2}p(x) \mathrm{d}x-\int q(y|x,\theta^{(y)})p(y|x)p(x)\mathrm{d}x  \\ &+ \frac{1}{2}\int p(y|x)^{2}p(x) \mathrm{d}x\end{split}$$where $p(x)$ represents the probability density of training set $\{x_{i}\}_{i=1}^{n}$.
By the Bayesian formula,$$p(y|x)p(x)=p(x,y)=p(x|y)p(y)$$
Hence $J_{y}$ can be reformulated as
$$J_{y}(\theta^{(y)})=\frac{1}{2}\int q(y|x,\theta^{(y)})^{2}p(x) \mathrm{d}x-\int q(y|x,\theta^{(y)})p(x|y)p(y)\mathrm{d}x+ \frac{1}{2}\int p(y|x)^{2}p(x) \mathrm{d}x$$
Note that the first term and second term in the equation above stand for the mathematical expectation of $p(x)$ and $p(x|y)$ respectively, which are often impossible to calculate directly. The last term is independent of $\theta$ and thus can be omitted.
Due to the fact that $p(x|y)$ is the probability density of sample $x$ belonging to class $y$, we are able to estimate term 1 and 2 by the following averages:$$\frac{1}{n}\sum_{i=1}^{n}q(y|x_{i},\theta^{(y)})^{2},\quad \frac{1}{n_{y}}\sum_{i:y_{i}=y}^{}q(y|x_{i},\theta^{(y)})p(y)$$
Next, we introduce the regularization term to get the following calculation rule:$$\hat{J}_{y}(\theta^{(y)})=\frac{1}{2n}\sum_{i=1}^{n}q(y|x_{i},\theta^{(y)})^{2}-\frac{1}{n_{y}}\sum_{i:y_{i}=y}^{}q(y|x_{i},\theta^{(y)})+\frac{\lambda}{2n}\|\theta^{(y)}\|^{2}$$
Let $\pi^{(y)}=(\pi_{1}^{(y)},\dots,\pi_{n}^{(y)})^{T}$ and $\pi_{i}^{(y)}=\left\{\begin{aligned}&1\quad (y_{i}=y)\\ &0 \quad (y_{i}\neq y)\end{aligned}\right.$, then
$$\hat{J}_{y}(\theta^{(y)})=\frac{1}{2n}{\theta^{(y)}}^{T}\Phi^{T}\Phi\theta^{(y)}-\frac{1}{n}{\theta^{(y)}}^{T}\Phi^{T}\pi^{(y)}+\frac{\lambda}{2n}\|\theta^{(y)}\|^{2}$$.
Therefore, it is evident that the problem above can be formulated as a *convex optimization* problem, and we can get the analytic solution by setting the twice order derivative to zero:
$$\hat{\theta}^{(y)}=\left( \Phi^{T}\Phi+\lambda I \right)^{-1}\Phi^{T}\pi^{(y)}$$.
In order not to get a negative estimation of the posterior probability, we need to add a constrain on the negative outcome:$$\hat{p}(y|x)=\frac{\max(0,{\hat{\theta}^{(y)}}^{T}\phi(x))}{\sum_{y'=1}^{c}\max(0,{\hat{\theta}^{(y')}}^{T}\phi(x))}$$

We also take Gaussian Kernal Models for example:

```matlab
n=90; c=3; y=ones(n/c,1)*(1:c); y=y(:);
x=randn(n/c,c)+repmat(linspace(-3,3,c),n/c,1);x=x(:);

hh=2*1^2; x2=x.^2; l=0.1; N=100; X=linspace(-5,5,N)';
k=exp(-(repmat(x2,1,n)+repmat(x2',n,1)-2*x*(x'))/hh);
K=exp(-(repmat(X.^2,1,n)+repmat(x2',N,1)-2*X*(x'))/hh);
for yy=1:c
    yk=(y==yy); ky=k(:,yk);
    ty=(ky'*ky +l*eye(sum(yk)))\(ky'*yk);
    Kt(:,yy)=max(0,K(:,yk)*ty);
end
ph=Kt./repmat(sum(Kt,2),1,c);

figure(1); clf; hold on; axis([-5,5,-0.3,1.8]);
C=exp(K*t); C=C./repmat(sum(C,2),1,c);
plot(X,C(:,1),'b-');
plot(X,C(:,2),'r--');
plot(X,C(:,3),'g:');
plot(x(y==1),-0.1*ones(n/c,1),'bo');
plot(x(y==2),-0.2*ones(n/c,1),'rx');
plot(x(y==3),-0.1*ones(n/c,1),'gv');
legend('q(y=1|x)','q(y=2|x)','q(y=3|x)');
```
![这里写图片描述](http://img.blog.csdn.net/20170408162048871?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

3. Summary
-------

Logistic regression is good at dealing with sample set with small size since it works in a simple way. However, when the number of samples is large to some degree, it is better to turn to the least square probability classifiers.