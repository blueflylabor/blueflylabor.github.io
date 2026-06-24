---
title: "Advanced Mathematics: Differential Calculus of Functions of Several Variables"
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
---

# Chapter 4 Differential Calculus of Functions of Several Variables

## Section 1 Basic Concepts and Conclusions

### Definition 1: (Function of Two Variables)

$$
\begin{align}
&z=f(x,y),(x,y)\in D\subset R^2\\
\end{align}
$$

![image-20210624214055676](/images/image-20210624214055676.jpg)

Example:
$$
\begin{align}
&\text{Find the domain of } f(x,y)=\arcsin(2x)+\ln y+\frac{\sqrt{4x-y^2}}{\ln{(1-x^2-y^2)}}.\\
&\text{Solution:}\\
&-1\leq 2x\leq 1, \quad y > 0, \quad 1-x^2-y^2 > 0, \quad 1-x^2-y^2\neq 1, \quad 4x-y^2\geq 0\\
&D=\left\{(x,y)\,\middle|\,-\frac{1}{2}\leq x\leq\frac{1}{2}, \quad y > 0, \quad x^2+y^2 < 1, \quad x\geq\frac{1}{4}y^2\right\}\\
\end{align}
$$


### Definition 2: (Limit of a Function of Two Variables)

$$
\begin{align}
&\lim_{(x,y)\rightarrow(x_0,y_0)}f(x,y)=A \quad \text{or} \quad \lim_{x\rightarrow x_0, y\rightarrow y_0}f(x,y)=A\\
\end{align}
$$

Examples:
$$
\begin{align}
&\text{1. Evaluate the limit:}\\
&\text{Solution Method 1:}\\
&\lim_{x\rightarrow 0,y\rightarrow 0}xy\frac{x^2-y^2}{x^2+y^2}\\
&\xlongequal[\text{Let } y=r\sin\theta]{\text{Let } x=r\cos\theta}\lim_{r\rightarrow0}{r\cos\theta \cdot r\sin\theta \cdot \frac{r^2\cos2\theta}{r^2}}=0\\
&\text{Solution Method 2:}\\
&0\leq\left|xy\frac{x^2-y^2}{x^2+y^2}\right|\leq{|xy|}\\
&\because \lim_{x\rightarrow 0,y\rightarrow 0}{0}=0 \quad \text{and} \quad \lim_{x\rightarrow 0,y\rightarrow 0}|xy|=0\\
&\text{By the Squeeze Theorem: } \Leftrightarrow\lim_{x\rightarrow 0,y\rightarrow 0}xy\frac{x^2-y^2}{x^2+y^2}=0\\
&\\
&\text{2. Verify that the following limits do not exist:}\\
&(1)\lim_{x\rightarrow 0,y\rightarrow 0}\frac{xy}{x^2+y^2}\\
&(2)\lim_{x\rightarrow 0,y\rightarrow 0}\frac{x^3+y^3}{x^2+y}\\
&\text{Solution:}\\
&(1) \text{ Let } y=kx:\\
&\lim_{x\rightarrow0,y=kx}\frac{x \cdot kx}{x^2+k^2x^2}=\frac{k}{1+k^2} \quad \text{(The limit depends on } k\text{, so it does not exist.)}\\
&(2) \text{ Let } y=-x^2+x^4:\\
&\lim_{x\rightarrow0,y=-x^2+x^4}\frac{x^3+(x^4-x^2)^3}{x^2+(-x^2+x^4)}=\lim_{x\rightarrow0}\left[\frac{1}{x}+\frac{(x^4-x^2)^3}{x^4}\right]=\infty \quad \text{(The limit does not exist.)}\\
\end{align}
$$
Notes:
$$
\begin{align}
&\text{Function of a single variable: } \{(x,f(x))|x\in D\}\\
&\text{Function of several variables: } \{(x,y,f(x,y))|(x,y)\in D\}
\end{align}
$$

### Definition 3: (Continuity of a Function of Two Variables)

$$
\begin{align}
&f(x,y) \text{ is continuous at point } P_0: \lim_{x\rightarrow x_0,y\rightarrow y_0}f(x,y)=f(x_0,y_0)\\
&\text{Note 1: } z=f(x,y) \text{ is continuous at point } P_0 \Leftrightarrow \Delta z=f(x,y)-f(x_0,y_0)\rightarrow 0 \text{ as } (x\rightarrow x_0,y\rightarrow y_0)\\
&\text{Note 2: "Elementary functions of two variables" are continuous everywhere within their domains. e.g., } \lim_{x\rightarrow 1,y\rightarrow2}\frac{x+y}{x-y}=-3\\
\end{align}
$$

### Theorem 1

$$
\begin{align}
&\text{A continuous function on a bounded closed region } D\subset R^2 \text{ must be bounded, and achieves both its maximum and minimum values.}\\
\end{align}
$$

![image-20210624223606852](/images/image-20210624223606852.jpg)

### Definition 4: (Partial Derivatives)

$$
\begin{align}
&\frac{\partial{z}}{\partial x}\bigg|_{(x_0,y_0)}=f_x'(x_0,y_0)=\lim_{\Delta x\rightarrow 0}\frac{\Delta z_x}{\Delta x}=\lim_{\Delta x\rightarrow0}\frac{f(x_0+\Delta x,y_0)-f(x_0,y_0)}{\Delta x}\\
&f_y'(x_0,y_0)=\lim_{\Delta y\rightarrow0}\frac{\Delta z_y}{\Delta y}=\lim_{y\rightarrow y_0}\frac{f(x_0,y)-f(x_0,y_0)}{y-y_0}
\end{align}
$$

![image-20210624223625497](/images/image-20210624223625497.jpg)

Examples:
$$
\begin{align}
&(1) \text{ Let } f(x,y)=\begin{cases}\frac{xy}{x^2+y^2},&(x,y)\neq(0,0)\\0,&(x,y)=(0,0)\end{cases}. \text{ Find } f_x'(0,0) \text{ and } f'_y(0,0), \text{ though } \lim_{x\rightarrow 0,y\rightarrow 0}f(x,y) \text{ does not exist.}\\
&(2) \text{ Find the partial derivatives of } f(x,y)=\sqrt{x^2+y^2} \text{ at } (0,0), \text{ and analyze its continuity at this point.}\\
\end{align}
$$
![image-20210625110703853](/images/image-20210625110703853.jpg)
$$
\begin{align}
&\text{(1) } f_x'(0,0)=\lim_{x\rightarrow0}\frac{f(x,0)-f(0,0)}{x-0}=\lim_{x\rightarrow 0}{\frac{0-0}{x}}=0\\
&\text{Similarly, } f_y'(0,0)=0\\
\end{align}
$$

$$
\begin{align}
&\text{(2) For } f(x,y)=\sqrt{x^2+y^2}:\\
&\lim_{x\rightarrow 0^{\pm}}\frac{f(x,0)-f(0,0)}{x-0}=\lim_{x\rightarrow 0^{\pm}}\frac{\sqrt{x^2}-0}{x}=\pm 1\\
&\Rightarrow f'_x(0,0) \text{ and } f'_y(0,0) \text{ do not exist.}\\
\end{align}
$$

### Definition 5: (Total Differential)

$$
\begin{align}
&\text{If } z=f(x,y), \, \Delta z=A\Delta x+B\Delta y+o(\rho) \text{ as } (\rho\rightarrow 0), \text{ where } \rho=\sqrt{\Delta x^2+\Delta y^2}, \text{ then } z=f(x,y) \text{ is said to be}\\
&\text{differentiable at point } P_0, \text{ and its total differential is denoted as:}\\
&dz\bigg|_{(x_0,y_0)}=df\bigg|_{(x_0,y_0)}=A\Delta x+B\Delta y \quad (\text{where } dx = \Delta x, dy = \Delta y)
\end{align}
$$

Notes:
$$
\begin{align}
&(1) \text{ If } \exists A, B \text{ such that } \lim_{\Delta x\rightarrow0,\Delta y\rightarrow 0}\frac{\Delta f-A\Delta x-B\Delta y}{\sqrt{\Delta x^2+\Delta y^2}}=0, \text{ then } f \text{ is differentiable at } (x_0,y_0).\\
&(2) \text{ If } f \text{ is differentiable at } (x_0,y_0), \text{ then } \lim\frac{\Delta f-df}{\sqrt{\Delta x^2+\Delta y^2}}=0\\
\end{align}
$$

### Theorem 2

$$
\begin{align}
&\text{If } z=f(x,y) \text{ is differentiable at point } P_0(x_0,y_0), \text{ then its partial derivatives must exist, and:}\\
&dz\bigg|_{(x_0,y_0)}=f'_x(x_0,y_0)dx+f_y'(x_0,y_0)dy\\
\end{align}
$$

### Theorem 3: Relationship Between Statements

$$
\begin{align}
&\text{Differentiable} \Rightarrow \begin{cases}\text{Continuous} \Rightarrow \text{Limit Exists}\\\text{Partial Derivatives Exist}\\\text{Directional Derivatives Exist}\end{cases}\\
\end{align}
$$

[Connection between Differentiability and Partial Derivatives of Functions of Two Variables - blueflylabor - Blogs (cnblogs.com)](https://www.cnblogs.com/blueflylabor/p/14947754.html)

Examples:

1.

![image-20210625155720429](/images/image-20210625155720429.jpg)

Note:
$$
\begin{align}
&\text{Sufficient Condition for Differentiability: Continuous partial derivative functions at } (x_0,y_0) \Rightarrow \text{Differentiable.}\\
\end{align}
$$


2.

![image-20210625155145956](/images/image-20210625155145956.jpg)

![image-20210625155207836](/images/image-20210625155207836.jpg)

## Section 2 Methods of Differentiation for Functions of Several Variables

### Differentiation of Elementary Functions

![image-20210627213536850](/images/image-20210627213536850.jpg)

Notes:
$$
\begin{align}
&(1) \text{ When differentiating with respect to } x\text{, treat } y \text{ as a constant; when differentiating with respect to } y\text{, treat } x \text{ as a constant.}\\
&(2) u \rightarrow \begin{cases}u_x' \rightarrow \begin{cases}u''_{xx}\\u''_{xy}\end{cases}\\u_y' \rightarrow \begin{cases}u''_{yx}\\u''_{yy}\end{cases}\end{cases}\\
&\text{When the partial derivative functions are continuous, } u''_{xy}=u''_{yx}.
\end{align}
$$


Examples:
$$
\begin{align}
&\text{1. Let } z=\arcsin\frac{x}{\sqrt{x^2+y^2}}, \text{ find } \frac{\partial^2z}{\partial x^2} \text{ and } \frac{\partial^2z}{\partial x\partial y}.\\
&\text{Solution:}\\
&z'_x=\frac{|y|}{x^2+y^2}\\
&z''_{xx}=\frac{\partial}{\partial x}\left(\frac{|y|}{x^2+y^2}\right)=\begin{cases}\frac{-2xy}{(x^2+y^2)^2},&y > 0\\0,&x\neq0,y=0\\\frac{2xy}{(x^2+y^2)^2},&y < 0\end{cases}\\
&z''_{xy}=\frac{\partial}{\partial y}\left(\frac{|y|}{x^2+y^2}\right)=\begin{cases}\frac{x^2-y^2}{(x^2+y^2)^2},&y > 0\\\text{Does not exist},&x\neq 0,y=0\\\frac{y^2-x^2}{(x^2+y^2)^2},&y < 0\end{cases}\\
\end{align}
$$
![image-20210629140548382](/images/image-20210629140548382.jpg)

### Differentiation of Composite Functions (Chain Rule)

![image-20210627214607550](/images/image-20210627214607550.jpg)

2.

![image-20210627214917783](/images/image-20210627214917783.jpg)
$$
\begin{align}
&\frac{\partial z}{\partial x}=-\frac{1}{x^2}f(xy)+\frac{1}{x}f'(xy)y+y\phi'(x+y)\\
&\frac{\partial^2 z}{\partial x\partial y}=-\frac{1}{x^2}f'(xy)x+\frac{1}{x}[f''(xy)xy+f'(xy)]+\phi'(x+y)+y\phi''(x+y)\\
\end{align}
$$
3.

![image-20210627215756345](/images/image-20210627215756345.jpg)
$$
\begin{align}
&\text{Method 1: Direct Differentiation}\\
&\frac{\partial f}{\partial x}=e^{-(xy)^2}y-e^{-(x+y)^2}\\
&\frac{\partial^2 f}{\partial x\partial y}=\dots\\
&\text{Method 2: Variable Substitution}\\
&\text{Let } u=x+y, v=xy. \text{ Then } f(x,y) \text{ is the composite of } \int_v^ue^{-t^2}dt \text{ with } \begin{cases}u=x+y\\v=xy\end{cases}\\
&\frac{\partial f}{\partial x}=e^{-u^2}\frac{\partial u}{\partial x}-e^{-v^2}\frac{\partial v}{\partial x} = e^{-(x+y)^2}\cdot 1 - e^{-(xy)^2}\cdot y\\
&\frac{\partial^2 f}{\partial x\partial y}=\dots\\
\end{align}
$$

## Differentiation of Implicit Functions of Several Variables

![image-20210628214256991](/images/image-20210628214256991.jpg)

Example:

![image-20210628214429361](/images/image-20210628214429361.jpg)
$$
\begin{align}
&\text{Method 1: Formula Method}\\
&F(x,y,u)=u+e^u-xy, \quad u'_x=\frac{\partial u}{\partial x}=-\frac{F'_x}{F'_u}=-\frac{-y}{1+e^u} = \frac{y}{1+e^u}\\
&\text{Method 2: Implicit Differentiation}\\
&u'_x+e^u u'_x=y \Rightarrow u_x'=\frac{y}{1+e^u}, \quad u'_y={\frac{x}{1+e^u}}\\
&\frac{\partial ^2u}{\partial x\partial y}=\frac{\partial}{\partial y}\left(\frac{y}{1+e^u}\right) = {\frac{1\cdot[1+e^u]-y\cdot e^u\cdot u_y'}{[1+e^u]^2}}\\
\end{align}
$$
![image-20210628215810134](/images/image-20210628215810134.jpg)

### Methods for Finding Extrema and Absolute Maximum/Minimum Values

![image-20210701154053195](/images/image-20210701154053195.jpg)

![image-20210701154208433](/images/image-20210701154208433.jpg)

![image-20210701154128709](/images/image-20210701154128709.jpg)

#### Unconstrained Extrema (Two Variables)

$$
\begin{align}
&(1) \text{ Definition}\\
&(2) \text{ Necessary Conditions: } \begin{cases}z_x'(x_0,y_0)=0\\z_y'(x_0,y_0)=0\end{cases}\\
&(3) \text{ Sufficient Conditions: Let } \Delta=AC-B^2, \text{ where:}\\
&A=f_{xx}''(x_0,y_0), \quad B=f_{xy}''(x_0,y_0), \quad C=f''_{yy}(x_0,y_0)\\
&\begin{cases}\Delta > 0, \text{ Extrema exists } (A<0 \text{ for maximum, } A>0 \text{ for minimum})\\\Delta < 0, \text{ No extrema exists (Saddle point)}\\\Delta = 0, \text{ Inconclusive}\end{cases}\\
\end{align}
$$

#### Bounded Closed Regions

$$
\begin{align}
&\text{Absolute Extrema of a Continuous Function } f(x,y) \text{ on a Bounded Closed Region } D:\\
&\text{If a function } f(x,y) \text{ is continuous on a bounded closed region } D\subset R^2, \text{ then } f(x,y) \text{ must attain both its}\\
&\text{absolute maximum and absolute minimum values within } D.\\
\end{align}
$$

![image-20210628224325667](/images/image-20210628224325667.jpg)

#### Problem Solving Steps

![image-20210629135428912](/images/image-20210629135428912.jpg)

![image-20210629135455737](/images/image-20210629135455737.jpg)

Examples:

![image-20210628215938737](/images/image-20210628215938737.jpg)

![image-20210628220016932](/images/image-20210628220016932.jpg)