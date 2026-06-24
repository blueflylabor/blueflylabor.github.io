---
title: Advanced Mathematics: Vector Algebra and Analytic Geometry in Space
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
lang: en

---

# Chapter 4 Vector Algebra and Analytic Geometry in Space

## Section 1 Vectors and Vector Algebra

### Definition 1:

![image-20210709112236540](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709112236540.jpg)
$$
\begin{align}
&(1) \text{ Vector: } \vec{a}=x\vec{i}+y\vec{j}+z\vec{k}=\{x,y,z\}\\
&(2) \text{ Magnitude (Norm) of a Vector: } |\vec{a}|=\sqrt{x^2+y^2+z^2}\\
&(3) \text{ Unit Vector: } |\vec a|=1, \; \vec a=\left(\frac{x}{\sqrt{x^2+y^2+z^2}},\frac{y}{\sqrt{x^2+y^2+z^2}},\frac{z}{\sqrt{x^2+y^2+z^2}}\right)=(\cos \alpha,\cos\beta,\cos \gamma)\\
&(4) \text{ Direction Cosines (Direction Numbers) of Vector } \vec{a}:\\
&\text{Direction angles } \alpha,\beta,\gamma\in[0,\pi]\\
&\vec{a}_0=(\cos \alpha,\cos\beta,\cos \gamma)\\
\end{align}
$$
![image-20210709112311950](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709112311950.jpg)

### Theorem 1:

$$
\begin{align}
&\text{Let } A(a_1,a_2,a_3),B(b_1,b_2,b_3)\in R^3, \text{ then } \vec{AB}=\{b_1,b_2,b_3\}-\{a_1,a_2,a_3\}=\{b_1-a_1,b_2-a_2,b_3-a_3\}\\
\end{align}
$$



### Definition 2

![image-20210709121110125](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709121110125.jpg)
$$
\begin{align}
&(1) \text{ Linear Operations: } \vec{a}+\vec{b}, \; \vec{a}-\vec{b}\\
&\lambda\vec{a}=\begin{cases}&|\lambda||\vec a|\hat{a}, \; \lambda>0, \text{ i.e., same direction as } \vec a\\&\vec{0}, \; \lambda=0, \text{ i.e., zero vector}\\&-|\lambda||\vec a|\hat{a}, \; \lambda<0, \text{ i.e., opposite direction to } \vec a\\\end{cases}
\end{align}
$$
![image-20210709121037300](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709121037300.jpg)
$$
\begin{align}
&(2) \text{ Scalar Product (Inner Product, Dot Product): The scalar product } \vec{a}\cdot\vec{b} \text{ yields a scalar.}\\
&\vec{a}\cdot\vec{b}=|\vec a||\vec b|\cos \theta\\
&\text{Note: Used to determine orthogonality }(\vec{a}\cdot\vec{b}=0 \Leftrightarrow \vec{a}\perp\vec{b})\\
&(3) \text{ Vector Product (Outer Product, Cross Product): The vector product } \vec{a}\times\vec{b} \text{ yields a vector, satisfying:}\\
&[1] \; |\vec a\times\vec b|=|\vec{a}||\vec{b}|\sin \theta\\
&[2] \; \vec a\times\vec b\perp\vec{a} \text{ and } \vec a\times\vec b\perp\vec{b}. \text{ The orientation of } \vec{a}, \; \vec{b}, \text{ and } \vec a\times\vec b \text{ follows the right-hand rule.}\\
&\text{Notes:}\\
&1) \text{ Used to determine parallelism } (\vec{a}\times\vec{b}=\vec{0} \Leftrightarrow \vec{a}\parallel\vec{b})\\
&2) \text{ Geometric Meaning of } |\vec{a}\times\vec{b}|: \text{ Represents the area of the parallelogram formed by } \vec{a} \text{ and } \vec{b}.\\
\end{align}
$$
![image-20210709122353830](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709122353830.jpg)
$$
\begin{align}
&(4) \text{ Scalar Triple Product: } [\vec a\vec b \vec c]\overset{\Delta}{=}(\vec a \times\vec b)\cdot \vec c=\vec c\cdot(\vec a \times\vec b)=(\vec b \times\vec c)\cdot \vec a\\
&\text{Note: The geometric meaning of } [\vec a\vec b \vec c] \text{ is the volume of the parallelepiped spanned by the three vectors.}\\
\end{align}
$$
![image-20210709123026193](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709123026193.jpg)

### Theorem 2

$$
\begin{align}
&\text{Let } \vec{a}=\{a_1,a_2,a_3\}, \; \vec{b}=\{b_1,b_2,b_3\}\\
&\text{Then } \lambda\vec{a}+\mu\vec{b}=\{\lambda a_1+\mu b_1, \; \lambda a_2+\mu b_2, \; \lambda a_3+\mu b_3\}\\
\end{align}
$$

### Theorem 3

![image-20210709135655589](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709135655589.jpg)

### Theorem 4

![image-20210709135720672](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709135720672.jpg)

### Theorem 5

![image-20210709150311140](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709150311140.jpg)

### Theorem 6

![image-20210709150345143](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709150345143.jpg)

### Example Problems

![image-20210709150453265](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210709150453265.jpg)
$$
\begin{align}
&\text{Solution:}\\
&(\vec a,\vec b,\vec c) \text{ denotes the scalar triple product of vectors } \vec a, \; \vec b, \text{ and } \vec c, \text{ meaning } (\vec a,\vec b,\vec c)=(\vec a \times \vec b)\cdot \vec c.\\
&\text{Moreover, } (\vec a \times \vec b)\cdot \vec c=(\vec b \times \vec c)\cdot \vec a=(\vec c \times \vec a)\cdot \vec b.\\
&\text{The geometric meaning of the triple product is the volume enclosed by the three vectors in space.}\\
&\vec A \times \vec B \text{ follows the right-hand rule, and its magnitude is given by } AB\sin\theta. \text{ The dot product involves } \cos\phi.\\
&\because (\vec a+\vec b) \times (\vec b+\vec c)=\vec a \times \vec b+\vec a \times \vec c+\vec b \times \vec b+\vec b \times \vec c = \vec a \times \vec b+\vec a \times \vec c+\vec b \times \vec c \quad (\text{since } \vec b \times \vec b = \vec 0)\\
&\therefore (\vec a+\vec b) \times (\vec b+\vec c)\cdot (\vec c+\vec a)=(\vec a \times \vec b+\vec a \times \vec c+\vec b \times \vec c)\cdot (\vec c+\vec a)\\
&=(\vec a,\vec b,\vec c)+(\vec a,\vec b,\vec a)+(\vec a,\vec c,\vec c)+(\vec a,\vec c,\vec a)+(\vec b,\vec c,\vec c)+(\vec b,\vec c,\vec a).\\
&\because (\vec a \times \vec b) \text{ is perpendicular to vector } \vec a,\\
&\therefore (\vec a,\vec b,\vec a)=(\vec a \times \vec b)\cdot \vec a=0.\\
&\text{By the same property, } (\vec a,\vec c,\vec c)=(\vec a,\vec c,\vec a)=(\vec b,\vec c,\vec c)=0.\\
&\text{Given that } (\vec a,\vec b,\vec c)=2,\\
&\text{it follows from cyclic permutation that } (\vec b,\vec c,\vec a)=(\vec a,\vec b,\vec c)=2.\\
&\therefore (\vec a+\vec b) \times (\vec b+\vec c)\cdot (\vec c+\vec a) = 2 + 2 = 4.
\end{align}
$$

## Section 2 Planes and Lines

### Planes

#### Point-Normal Equation

![image-20210720163141583](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720163141583.jpg)
$$
\begin{align}
&\vec{P_0P}\cdot\vec{n}=0\\
&\Leftrightarrow a(x-x_0)+b(y-y_0)+c(z-z_0)=0\\
&\text{where } \vec{n}=\{a,b,c\}\neq\vec{0} \text{ is the normal vector, and } P_0(x_0,y_0,z_0) \text{ is a known point on the plane.}\\
\end{align}
$$

#### General Equation

$$
\begin{align}
&ax+by+cz+d=0, \quad \vec{n}=\{a,b,c\}\neq\vec{0} \text{ serves as the normal vector.}\\
\end{align}
$$

Notes:

![image-20210720165839118](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720165839118.jpg)

#### Intercept Equation

![image-20210720165758211](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720165758211.jpg)
$$
\begin{align}
&\frac{x}{a}+\frac{y}{b}+\frac{z}{c}=1\\
&\text{where } a,b,c \text{ represent the intercepts of the plane on the three coordinate axes, respectively.}\\
&\text{Note: The intercept form can only represent planes that intersect all three coordinate axes and do not pass through the origin.}\\
\end{align}
$$

#### Distance from a Point to a Plane

![image-20210720170315037](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720170315037.jpg)
$$
\begin{align}
&\text{Distance from point } P_0(x_0,y_0,z_0) \text{ to the plane } ax+by+cz+d=0:\\
&\rho=\frac{|ax_0+by_0+cz_0+d|}{\sqrt{a^2+b^2+c^2}}
\end{align}
$$

#### Angle Between Two Planes (Angle Between Normal Vectors)

![image-20210720170411120](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720170411120.jpg)
$$
\begin{align}
&\cos\theta=\frac{|\vec{n_1}\cdot\vec{n_2}|}{|\vec{n_1}||\vec{n_2}|}\\
\end{align}
$$

#### Spatial Relationships Between Planes

![image-20210720170615555](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720170615555.jpg)

### Lines

#### Symmetric Equation (Point-Direction Equation)

$$
\begin{align}
&\vec{s}\parallel\vec{P_0P}\\
&\frac{x-x_0}{l}=\frac{y-y_0}{m}=\frac{z-z_0}{n}\\
&\text{where } P_0(x_0,y_0,z_0) \text{ is a point on the line, and } \vec{s}=\{l,m,n\}\neq\vec{0} \text{ is the direction vector of the line.}\\
\end{align}
$$



![image-20210720170902650](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720170902650.jpg)

#### General Equation

![image-20210720171628330](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720171628330.jpg)

#### Parametric Equation

![image-20210720171558344](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720171558344.jpg)

#### Distance from a Point to a Line

![image-20210720171649489](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720171649489.jpg)

#### Angle Between Two Lines

![image-20210720171701565](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720171701565.jpg)

#### Spatial Relationships Between Lines

![image-20210720171527919](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720171527919.jpg)

### Spatial Relationships Between a Line and a Plane

![image-20210720171500856](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210720171500856.jpg)

### Basic Methods