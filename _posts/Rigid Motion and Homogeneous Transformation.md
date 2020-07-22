A rigid motion of an object is a motion which **preserves distance between points**. In this article, we provide a description of rigid body motion using the tools of linear algebra and screw theory.

[TOC]

# Rotation Matrix
## Representation
Suppose there are two frames $o_{0}x_{0}y_{0}z_{0}$ and $o_{1}x_{1}y_{1}z_{1}$. A **rotation matrix** is defined as unit vectors $x_{1}$, $y_{1}$ and $z_{1}$ in frame $o_{0}x_{0}y_{0}z_{0}$. A good way to represent rotation operation is to project $o_{1}x_{1}y_{1}z_{1}$ to $o_{0}x_{0}y_{0}z_{0}$:
$$R_{1}^{0}=\begin{pmatrix}x_{1}\cdot x_{0} & y_{1} \cdot x_{0} & z_{1}\cdot x_{0} \\ x_{1}\cdot y_{0} & y_{1} \cdot y_{0} & z_{1}\cdot y_{0} \\ x_{1}\cdot z_{0} & y_{1} \cdot z_{0} & z_{1}\cdot z_{0} \end{pmatrix}$$Or
$$R_{1}^{0}=[x_{10}, y_{10}, z_{10}]$$where $x_{10}, y_{10}, z_{10}$ are the coordinates of the principal axes of $1$ relative to $0$.

In fact, each entry of the rotation matrix is a dot product[^dotproduct] of two unit vectors, in other words, is the cosine of the angle between the two vectors, i.e. directional cosine.
[^dotproduct]: **Dot Product**: $a=(a_{1}, a_{2}, a_{3}), b=(b_{1}, b_{2}, b_{3})$
$$a\cdot b = |a||b|\cos\theta = a_{1}b_{1}+a_{2}b_{2}+a_{3}b_{3}$$


$R^{0}_{1}$ is **orthogonal** ($R^{0}_{1}=(R^{1}_{0})^{T}$, $(R^{0}_{1})^{T}=(R^{0}_{1})^{-1}$) and $\det R^{0}_{1}=\pm1$. To determine the sign of the determinant of $R_{1}^{0}$, we recall from linear algebra that $\det R^{0}_{1}=r_{1}^{T}(r_{2}\times r_{3})$ where $r$ is its column. Since the coordinate frame is **right-handed**, we have that $r_{2}\times r_{3}=r_{1}$. Then $\det R^{0}_{1} = 1$.

Define $R\in SO(n)$ which is a special orthogonal group[^group], satisfying:
[^group]: $X$ is a **group** if and only if
(1)$\forall x_{1}, x_{2}\in X, x_{1}*x_{2}\in X$; 
(2)$(x_{1}*x_{2})*x_{3}=x_{1}*(x_{2}*x_{3})$; 
(3)$\exists I\in X, \forall x\in X \rightarrow I*x=x*I=x$; 
(4)$\forall x\in X, \exists y\in X s.t. x*y=y*x=I$.

 1. $R^{T}=R^{-1}\in SO(n)$;
 2. Columns and rows of R are orthogonal;
 3. Columns and rows of R are unit vectors;
 4. $\det R=1$.

Obviously, $R^{0}_{1}\in SO(3)$.

For illustration: 
Assume that frame $o_{1}x_{1}y_{1}z_{1}$ rotate $\theta$ degrees around $z_{0}$,
$$R^{0}_{1}=\begin{pmatrix}c_{\theta} & -s_{\theta} & 0\\ s_{\theta} & c_{\theta} & 0 \\ 0 & 0 & 1 \end{pmatrix}$$
For simplicity, we define $R_{z,\theta}=R^{0}_{1}$ temporarily. Then we see
$$R_{z,0}=I,\quad R_{z,\theta}R_{z,\phi}=R_{z.\theta+\phi},\quad (R_{z,\theta})^{-1}=R_{z,-\theta}$$
Similarly,
$$R_{x,\theta}=\begin{pmatrix} 1 & 0 & 0 \\0 & c_{\theta} & -s_{\theta}\\ 0 & s_{\theta} & c_{\theta} \end{pmatrix},\quad R_{y,\theta}=\begin{pmatrix}c_{\theta} & 0 & s_{\theta} \\ 0 & 1 & 0 \\ -s_{\theta} & 0 & c_{\theta} \end{pmatrix}$$

## Rotation Transformation
Given a point $p=[u,v,w]^{T}$, its coordinate in frame $o_{1}x_{1}y_{1}z_{1}$ is $p^{1}$ satisfying
$$p=ux_{1}+vy_{1}+wz_{1}$$
Since the principal axes of $o_{1}x_{1}y_{1}z_{1}$ have coordinates $x_{10}, y_{10}, z_{10}$ with respect to $o_{0}x_{0}y_{0}z_{0}$,  the coordinate of $p$ relative to frame $o_{0}x_{0}y_{0}z_{0}$ is given by 
$$p^{0}=\begin{pmatrix} x_{10} & y_{10} & z_{10} \end{pmatrix}\begin{pmatrix} u \\ v \\ w \end{pmatrix}=R^{0}_{1}p^{1}$$
A rotation matrix preserves distance and orientation. This can be proved partially by using some algebraic properties of the **cross product**[^cross_product] operation between two vectors. Given $R\in SO(3)$
$$R(v\times w)=(Rv) \times (Rw),\quad R(w)^{\land}R^{T}=(Rw)^{\land}$$

[^cross_product]: The cross product between two vectors $a, b\in\mathbb{R}^{3}$ is defined as $$a\times b = \begin{pmatrix}a_{2}b_{3}-a_{3}b_{2} \\ a_{3}b_{1}-a_{1}b_{3} \\ a_{1}b_{2}-a_{2}b_{1} \end{pmatrix}$$We can also write:$$a\times b = (a)^{\land}b,\quad (a)^{\land}=\begin{pmatrix}0& -a_{3} & a_{2} \\ a_{3} & 0 & -a_{1} \\ -a_{2} & a_{1} & 0  \end{pmatrix}$$

Given a linear transformation $A$ defined in frame $o_{1}x_{1}y_{1}z_{1}$. And $B$ is the same linear transformation defined in frame $o_{0}x_{0}y_{0}z_{0}$. It can be showed that
$$B=(R^{0}_{1})^{-1}AR^{0}_{1}$$

## Rotation Superposition
Rotation w.r.t current frame:
$$\left\{\begin{aligned}p^{0}=R^{0}_{1}p^{1}\\ p^{1}=R^{1}_{2}p^{2}\\p^{0}=R^{0}_{2}p^{2} \end{aligned}\right.\quad\Rightarrow\quad p^{0}=R^{0}_{1}R^{1}_{2}p^{2}, R^{0}_{2}=R^{0}_{1}R^{1}_{2}$$
Rotation w.r.t fixed frame: Just reverse the multiplier order of the above equation.
$$R^{0}_{2}=R_{2}^{1}R^{0}_{1}$$The equation above is called the **composition rule**.

# Exponential Coordinates for Rotation
A common motion encountered in robotics is the rotation of a body about a given axis by some amount. Take the figure below as an illustration:
![fig01](http://img.blog.csdn.net/20171129205450261?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
Hypothesize that we rotate the point $q$ at a constant unit angular velocity around $\omega$, then $\dot{q}$ may be written as:
$$\dot{q}(t) = \omega\times q(t) = \hat{\omega}q(t)$$Thus$$q(t)=e^{\hat{\omega}t}q(0)$$Due to the unit velocity hypothesis:
$$R(\omega, \theta)=e^{\hat{\omega}\theta}$$ It can be seen that $\hat{\omega}$ is a **skew-symmetric** matrix, i.e. $\hat{\omega}^{T} = -\hat{\omega}$ . The vector space of $3\times 3$ skew-symmetric matrix is demoted $so(3)$. The sum of two elements of $so(3)$ is an element of $so(3)$ and the scalar multiple of any element of $so(3)$ is an element of $so(3)$. Furthermore, $(v + \omega)^{\land}=\hat{v}+\hat{\omega}$ .

It will be convenient to represent a skew-symmetric matrix as the product of a unit skew-symmetric matrix and a real number, i.e. $\hat{\omega}\in so(3), \|\omega\|=1, \theta\in \mathbb{R}$ . Then by Taylor expansion, we can write:
$$e^{\hat{\omega}\theta}=I+\hat{\omega}\sin\theta+\hat{\omega}^{2}(1-\cos\theta)$$ This formula, commonly referred to as *Rodrigues’formula*[^Rodrigues], gives an efficient method for computing $\exp(ωθ)$. 

[^Rodrigues]:If $\|\omega\|\neq 1$, it is verified that $$e^{\hat{\omega}\theta}=I+\frac{\hat{\omega}}{\|\hat{\omega}\|}\sin(\|\hat{\omega}\|\theta) + \frac{\hat{\omega}^{2}}{\|\hat{\omega}\|^{2}}(1-\cos(\|\hat{\omega}\|\theta))$$

Every rotation matrix can be represented as the matrix exponential of some skew-symmetric matrix, i.e. the map $\exp : so(3) → SO(3)$ is surjective (onto).

**Euler Theorem**: Any Orientation $R\in SO(3)$ is equivalent to a rotation about a fixed axis $\omega\in \mathbb{R}^{3}$ through an angle $\theta\in [0, 2\pi)$.

# Rotation Parameterization
## Euler-angle
Around the current frame ($R_{z,\phi} \to R_{y,\theta} \to R_{z,\psi}$)
$$\begin{split}R_{ZYZ} &=R_{z,\phi}R_{y,\theta}R_{z,\psi} \\ &=\begin{pmatrix}c_{\phi} & -s_{\phi} & 0\\s_{\phi} & c_{\phi} & 0\\0 & 0 & 1 \end{pmatrix}\begin{pmatrix}c_{\theta} &0& s_{\theta}\\0 & 1 & 0 \\-s_{\theta}  & 0& c_{\theta}\end{pmatrix}\begin{pmatrix}c_{\psi} & -s_{\psi} & 0\\s_{\psi} & c_{\psi} & 0\\0 & 0 & 1 \end{pmatrix}  \\ &= \begin{pmatrix}c_{\phi}c_{\theta}c_{\psi}-s_{\phi}s_{\psi} & -c_{\phi}c_{\theta}s_{\psi}-s_{\phi}c_{\psi} & c_{\phi}s_{\theta}\\s_{\phi}c_{\theta}c_{\psi}+c_{\phi}s_{\psi} & -s_{\phi}c_{\theta}s+c_{\phi}c_{\psi} & s_{\phi}s_{\theta}\\ -s_{\theta}c_{\psi} & s_{\theta}s_{\psi} &c_{\theta}  \end{pmatrix}\end{split}$$
## Roll-Pitch-Yaw
Around the fixed frame ($R_{x,\psi} \to R_{y,\theta} \to R_{z,\phi}$)
$$\begin{split}R_{XYZ} &=R_{z,\phi}R_{y,\theta}R_{x,\psi} \\ &=\begin{pmatrix}c_{\phi} & -s_{\phi} & 0\\s_{\phi} & c_{\phi} & 0\\0 & 0 & 1 \end{pmatrix}\begin{pmatrix}c_{\theta} &0& s_{\theta}\\0 & 1 & 0 \\-s_{\theta}  & 0& c_{\theta}\end{pmatrix}\begin{pmatrix}1 & 0 & 0\\0& c_{\psi} &- s_{\psi}\\0 & s_{\psi}& c_{\psi}\end{pmatrix} \\ &= \begin{pmatrix} c_{\phi}c_{\theta} & -s_{\phi}c_{\psi}+c_{\phi}s_{\theta}s_{\psi} & s_{\phi}s_{\psi} + c_{\phi}s_{\theta}c_{\psi}\\ s_{\phi}c_{\theta} & c_{\phi}c_{\psi}+s_{\phi}s_{\theta}s_{\psi} & -c_{\phi}s_{\psi}+s_{\phi}s_{\theta}c_{\psi}\\-s_{\theta} & c_{\theta}s_{\psi} & c_{\theta}c_{\psi} \end{pmatrix}\end{split}$$
## Axis Angle
Let $k=[k_{x},k_{y},k_{z}]^{T}$ be an unit vector in frame $o_{0}x_{0}y_{0}z_{0}$. It can be seen as an axis. Let $R=R_{z,\alpha}R_{y,\beta}$ make the vector $k$ rotate to axis $z$. Then
$$R_{k,\theta}=RR_{z,\theta}R^{-1}=R_{z,\alpha}R_{y,\beta}R_{z,\theta}R_{y,-\beta}R_{z,-\alpha}$$

In fact, for any $R\in SO(3)$ we can always define $R=R_{k,\theta}$ in which
$$\theta=\cos^{-1}\left(\frac{r_{11}+r_{22}+r_{33}-1}{2}\right),\quad k=\frac{1}{2\sin\theta}\begin{pmatrix}r_{32}-r_{23}\\r_{13}-r_{31}\\r_{21}-r_{12} \end{pmatrix}$$

# Rigid Motion

A mapping $g:\mathbb{R}^{3} \to \mathbb{R}^{3}$ is a **rigid body transformation** if it satisfies the following properties:

 1. Length is preserved: $\|g(p)-g(q)\| = \|p-q\|$ for all points $p,q\in\mathbb{R}^{3}$.
 2. The cross product is preserved: $g_{*}(v\times w) = g_{*}(v)\times g_{*}(w)$ for all vectors $v,w\in\mathbb{R}^{3}$.

The representation of general rigid body motion, involving both translation and rotation, is more involved. Rigid motions is defined as a sequence $(d,R)$, where $s\in\mathbb{R}^{3}$, $R\in SO(3)$. All the rigid motions form a group called **Special Euclidean Group** represented by $SE(3)$. 

Consider three frames $o_{0}x_{0}y_{0}z_{0}$, $o_{1}x_{1}y_{1}z_{1}$ and $o_{2}x_{2}y_{2}z_{2}$. There happened some rigid motions:
$$\begin{gather}p^{1}=R^{1}_{2}p^{2}+d^{1}_{2}\\ p^{0}=R^{0}_{1}p^{1}+d^{0}_{1}\\p^{0}= R^{0}_{1}R^{1}_{2}p^{2}+R^{0}_{1}d^{1}_{2}+d^{0}_{1}\\p^{0}=R^{0}_{2}p^{2}+d^{0}_{2}\end{gather}$$
Finally, we get
$$\begin{gather}R^{0}_{2}=R^{0}_{1}R^{1}_{2}\\ d^{0}_{2}=d^{0}_{1}+R^{0}_{1}d^{1}_{2} \end{gather}$$

# Homogeneous Transformation
## Homogeneous Representation
The sequential rigid motions above can be simplified as 
$$\begin{pmatrix}R^{0}_{1} & d^{0}_{1}\\0 & 1\end{pmatrix}\begin{pmatrix}R^{1}_{2} & d^{2}_{1}\\0 & 1\end{pmatrix}=\begin{pmatrix}R^{0}_{1}R^{1}_{2} & R^{0}_{1}d^{2}_{1}+d^{0}_{1}\\0 & 1\end{pmatrix}$$The equation above is called the *composition rule* for rigid body transformations to be the standard matrix multiplication. 
Define
$$H=\begin{pmatrix}R & d\\ 0 & 1\end{pmatrix}, R\in SO(3), d\in \mathbb{R}^{3}$$
as a **homogeneous transformation**. And let 
$$P^{0}=\begin{pmatrix}p^{0}\\ 1\end{pmatrix},\quad P^{1}=\begin{pmatrix}p^{1}\\  1\end{pmatrix}$$
Then
$$P^{0}=H^{0}_{1}P^{1}$$
It is evidently that 
$$H^{-1}=\begin{pmatrix}R^{T} & -R^{T}d\\ 0 & 1\end{pmatrix}$$
since $R$ is orthogonal. It may be verified that the set of rigid transformations is a group, i.e. 

 1. If $g_{1}, g_{2}\in SE(3)$, then $g_{1}g_{2}\in SE(3)$.
 2. The $4\times 4$ identity element $I$ is in $SE(3)$. 
 3. If $g=(d, R)\in SE(3)$, then $$\begin{gather} \bar{g}=\begin{pmatrix}R & d \\ 0 & 1 \end{pmatrix} \end{gather},\quad \bar{g}^{-1}=\begin{pmatrix}R^{T} & -R^{T}d \\ 0 & 1 \end{pmatrix}\in SE(3) \\ $$So that $g^{-1} = (-R^{T}d,R^{T})$.
 4. The composition rule for rigid body transformations is associative.

*Ex*: Consider the example below:
![ex2](http://img.blog.csdn.net/20171207201729412?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
The orientation of coordinate frame $B$ with respect to $A$ is 
$$R_{ab}=\begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$ The coordinates for the origin of frame $B$ are
$$p_{ab}=\begin{pmatrix} 0 \\ l_{1} \\ 0 \end{pmatrix}$$again relative to frame $A$. The homogeneous representation of the configuration of the rigid body is given by 
$$g_{ab}(\theta)=\begin{pmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & l_{1} \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

## Exponential coordinates for rigid motion and twist
The notion of the exponential mapping introduced for $SO(3)$ can be generalized to the **Euclidean group**, $SE(3)$.  Analogous to the definition of $so(3)$, we define 
$$se(3):=\{(v, \hat{\omega}): v\in\mathbb{R}^{3}, \hat{\omega}\in so(3) \}$$ In homogeneous coordinates, we write an element $\hat{\xi}\in se(3)$ as 
$$\hat{\xi} =\begin{pmatrix} \hat{\omega} & v \\ 0 & 0 \end{pmatrix}$$ An element of se(3) is referred to as a **twist**, or a (infinitesimal) generator of the Euclidean group. We also define
$$\begin{pmatrix}\hat{\omega} & v \\ 0 & 0 \end{pmatrix}^{\vee} = \begin{pmatrix}v \\ \omega \end{pmatrix},\quad \begin{pmatrix}v \\ \omega \end{pmatrix}^{\land} = \begin{pmatrix}\hat{\omega} & v \\ 0 & 0 \end{pmatrix}$$ 

Given $\hat{\xi}\in se(3)$ and $\theta\in \mathbb{R}$, the exponential of $\hat{\xi}\theta$ is an element of $SE(3)$. Moreover
$$e^{\hat{\xi}\theta} = \begin{pmatrix} e^{\hat{\omega}\theta} & (I-e^{\hat{\omega}\theta})(\omega\times v)+\omega\omega^{T}v\theta \\ 0 & 1 \end{pmatrix},\quad \omega\neq 0$$
$$e^{\hat{\xi}\theta} = \begin{pmatrix} I & v\theta \\ 0 & 1 \end{pmatrix},\quad \omega= 0$$
We interpret $g=\exp(\hat{\xi}\theta)$ not as mapping points from one coordinate frame to another, but rather as mapping points from their initial coordinates to their coordinates after the rigid motion is applied:
$$p(\theta)=e^{\hat{\xi}\theta}p(0)$$In this equation, both $p(0)$ and $p(θ)$ are specified with respect to a single reference frame. Similarly, if we let $g_{ab}(0)$ represent the initial configuration of a rigid body relative to a frame $A$, then the final configuration, still with respect to $A$, is given by 
$$g_{ab}(\theta)=e^{\hat{\xi}\theta}g_{ab}(0)$$ Thus, the exponential map for a twist gives the relative motion of a rigid body. Every rigid transformation can be written as the exponential of some twist.

# Screw
In this section, we explore some of the geometric attributes associated with a **twist** $\xi=(v,\omega)$ . Consider a rigid body motion which consists of rotation about an axis in space through an angle of $\theta$ radians, followed by translation along the same axis by an amount $d$ as shown below
![screw](http://img.blog.csdn.net/20171207212416964?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGhpbHRoaW5rZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
We call such a motion a **screw motion**.

A screw $S$ consists of an axis $l$, a pitch $h (h=d/\theta)$, and a magnitude $M$. A screw motion represents rotation by an amount $\theta = M$ about the axis $l$ followed by translation by an amount $h\theta$ parallel to the axis $l$. If $h = \infty$ then the corresponding screw motion consists of a pure translation along the axis of the screw by a distance $M$. 

Recall the figure above, it can be seen that 
$$gp = q + e^{\hat{\omega}\theta}(p-q)+h\theta\omega$$ $$g\begin{pmatrix}p\\1\end{pmatrix}=\begin{pmatrix}e^{\hat{\omega}\theta} & (I-e^{\hat{\omega}\theta})q+h\theta\omega \\ 0 & 1\end{pmatrix}\begin{pmatrix}p\\1\end{pmatrix}$$ This transformation maps points attached to the rigid body from their initial coordinates $(\theta = 0)$ to their final coordinates, and all points are specified with respect to the fixed reference frame.

In fact, if we choose $v = −\omega × q + h\omega$, then $\xi = (v, \omega)$ generates the screw motion. A screw motion corresponds to motion along a constant twist by an amount equal to the magnitude of the screw.
we define a **unit twist** to be a twist such that either $\|\omega\|=1$, or $\omega = 0$ and $\|v\| = 1$; that is, a unit twist has magnitude $M = 1$. Unit twists are useful since they allow us to express rigid motions due to revolute and prismatic joints as $g=\exp(\hat{\xi}\theta)$ where $\theta$ corresponds to the amount of rotation or translation.

**Chasles Theorem**: Every rigid body motion can be realized by a rotation about an axis combined with a translation parallel to that axis.



----------

**Acknowledgement **

- Thanks Mark W. Spong for his great work of *Robot Modeling and Control*.
- Thanks John J. Craig for his great work of *Introduction to Robotics - Mechanics and Control, 3rd-Edition*
- Thanks Zexiang Li for his great work of *A Mathematical Introduction to Robotic Manipulation*