---
title: Advanced Mathematics: Derivatives and Differentials
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
lang: en
permalink: /en/:title/
---

# Derivatives and Differentials

## (I) Concepts of Derivatives and Differentials

### 1. Derivative

$$
\begin{aligned}
&\text{Definition 1: (Derivative) (Represents the rate of change at } x_0) \\
&f'(x_0)=\lim_{\Delta x\to 0} \tfrac{\Delta y}{\Delta x}=\lim_{\Delta x\to 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}=\lim_{h\to 0}\frac{f(x_0+h)-f(x_0)}{h}\\
&\text{Definition 2: (Left Derivative) (Differentiable within the left neighborhood)}\\
&f_-'(x_0)=\lim_{\Delta x\to 0^-} \tfrac{\Delta y}{\Delta x}=\lim_{\Delta x\to 0^-}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}=\lim_{h\to 0^-}\frac{f(x_0+h)-f(x_0)}{h}\\
&\text{Definition 3: (Right Derivative) (Differentiable within the right neighborhood)}\\
&f_+'(x_0)=\lim_{\Delta x\to 0^+} \tfrac{\Delta y}{\Delta x}=\lim_{\Delta x\to 0^+}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}=\lim_{h\to 0^+}\frac{f(x_0+h)-f(x_0)}{h}\\
\end{aligned}
$$

$$
\text{Theorem 1: } f'(x) \text{ exists } \Leftrightarrow f'_-(x) \text{ exists, } f'_+(x) \text{ exists, and } f'_-(x)=f'_+(x)
$$

### 2. Differential

$$
\begin{align}
&\text{Definition 4: (Differential) If } \Delta y=f(x_0+\Delta x)-f(x_0) \text{ can be expressed as:}\\
&\Delta y=A\Delta x+o(\Delta x)\\
&\text{Then the function } f(x) \text{ is said to be differentiable at point } x_0, \text{ and } A\Delta x \text{ is called the differential, denoted as } dy=A\Delta x\\
&dy\approx \Delta y \text{ (Replacing a non-uniform variable with a uniform variable in a tiny region)}\\
&\text{The differential is the linear principal part of the function increment.}
\end{align}
$$

$$
\text{Theorem 2: The function } y=f(x) \text{ is differentiable at point } x_0 \Leftrightarrow f(x) \text{ is derivable at point } x_0, \text{ and } dy=f'(x_0)\Delta x=f'(x_0)dx
$$

![QQ图片20210424223657](https://raw.githubusercontent.com/blueflylabor/images/main/QQ%E5%9B%BE%E7%89%8720210424223657.jpg)
$$
\begin{align}
&S(x)=x^2, S(x+\Delta x)=(x+\Delta x)^2\\
&\Delta S=(x+\Delta x)^2-x^2=2x\Delta x+(\Delta x)^2=2x\Delta x+o(\Delta x)\\
&\text{Linear Principal Part } (ds=2x\Delta x=S'(x)\Delta x) + \text{Higher-Order Infinitesimal}\\
&\Delta S\approx 2x\Delta x \quad (\Delta x\rightarrow 0)\\
&\\
&\Delta f(x)=A\Delta x+o(\Delta x) \quad (\Delta x\rightarrow 0)\Leftrightarrow \lim_{\Delta x \rightarrow 0}\frac{\Delta f(x)-A\Delta x}{\Delta x}=0\\
&(1) \text{ If } \exists A \text{ such that } \lim_{\Delta x \rightarrow 0}\frac{\Delta f-A\Delta x}{\Delta x} = 0\\
&(2) \text{ If } f \text{ is differentiable, then } \lim_{\Delta x \rightarrow 0}\frac{\Delta f-f'(x)\Delta x}{\Delta x}=0\\
\end{align}
$$


## 3. Geometric Meaning of Derivatives and Differentials

![image-20210317203802573](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210317203802573.jpg)

![image-20210317204445451](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210317204445451.jpg)

## 4. Relationships between Continuity, Derivability, and Differentiability

![image-20210318155550511](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318155550511.jpg)

$$
\begin{align}
&\text{Note: } f(x) \text{ is continuous at } x_0 \Leftrightarrow \lim_{x\rightarrow x_0}f(x)=f(x_0)\\
&\Leftrightarrow \lim_{x\rightarrow x_0}[f(x)-f(x_0)]=0 \quad \text{i.e., } \lim_{\Delta x\rightarrow 0}\Delta f=0\\
\end{align}
$$


## (II) Differentiation Formulas and Rules

![image-20210318161354989](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318161354989.jpg)

![image-20210318161623611](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318161623611.jpg)

![image-20210318161758591](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318161758591.jpg)

![image-20210318162220162](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318162220162.jpg)

![image-20210318162441013](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318162441013.jpg)

![image-20210318162721402](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318162721402.jpg)

![image-20210318163229142](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318163229142.jpg)

"Transform multiplication/division into addition/subtraction via logarithms"
$$
u^v=e^{v\ln{u}}
$$

$$
y=u^v \Leftrightarrow \ln{y}=v\ln{u}
$$

## Higher-Order Derivatives

![image-20210318163730625](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318163730625.jpg)![image-20210318163851039](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210318163851039.jpg)
$$
\begin{align}
&\text{Given } y=\sin 3x, \text{ find } y^{(n)}:\\
&y'=\cos 3x \cdot 3 = \sin\left(3x+\frac{\pi}{2}\right) \cdot 3\\
&y''=\cos\left(3x+\frac{\pi}{2}\right) \cdot 3^2 = \sin\left(3x+2\cdot\frac{\pi}{2}\right) \cdot 3^2\\
&\Rightarrow y^{(n)}=\sin\left(3x+n\cdot\frac{\pi}{2}\right) \cdot 3^n\\
&y=\sin(ax+b) \Rightarrow y^{(n)}=a^n \sin\left(ax+b+n\cdot\frac{\pi}{2}\right)
\end{align}
$$

$$
\begin{align}
&\text{Given } y=x^2\cos x, \text{ find } y^{(n)}:\\
&\text{Let } u=x^2, v=\cos x\\
&u'=2x, u''=2, u'''=0, \dots, u^{(n)}=0\\
&(uv)^{(n)}=\sum_{k=0}^{n} C_n^k u^{(k)}v^{(n-k)}\\
&y^{(n)}=C_{n}^{0}x^2\cos\left(x+n\cdot\frac{\pi}{2}\right)+C_{n}^1(2x)\cos\left(x+(n-1)\frac{\pi}{2}\right)+C^2_{n}(2)\cos\left(x+(n-2)\cdot\frac{\pi}{2}\right)
\end{align}
$$

## (III) Methods of Differentiation

### 1. Differentiation of Composite and Elementary Functions

### (1) Basic Differentials

![1619770648(1)](https://raw.githubusercontent.com/blueflylabor/images/main/1619770648(1).jpg)

### (2) Composite Function Differentiation (Chain Rule)

$$
\begin{align}
&\text{Assume that } u=\Phi(x) \text{ is differentiable at point } x, \text{ and } y=f(u) \text{ is differentiable at point } u=\Phi(x).\\
&\text{Then the composite function } y=f(\Phi(x)) \text{ is differentiable at point } x, \text{ and } \frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}\\
&\text{Or written as: } [f(\Phi(x))]'=f'(\Phi(x))\Phi'(x)\\
\end{align}
$$

###### Examples

$$
\begin{align}
&\text{Let } u=\tan y, x=e^t. \text{ Transform the following equation regarding } y(x): F\left(\frac{d^2y}{dx^2},\frac{dy}{dx},y,x\right)=0\\
&\text{Solution:}\\
&y=y(u(t(x))), t=\ln x, y=\arctan u \Rightarrow y=\arctan(u(\ln x))\\
&\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dt}\cdot\frac{dt}{dx}=\frac{1}{1+u^2}\cdot\frac{du}{dt}\cdot\frac{1}{x}\\
&\frac{d^2y}{dx^2}=\frac{d}{dx}\left[\frac{dy}{dx}\right] = \frac{d}{dt}\left[\frac{1}{1+u^2}\cdot\frac{du}{dt}\cdot\frac{1}{x}\right]\frac{dt}{dx}\\
&= \left( \frac{-2u}{(1+u^2)^2}\left(\frac{du}{dt}\right)^2\frac{1}{x} + \frac{1}{1+u^2}\frac{d^2u}{dt^2}\frac{1}{x} - \frac{1}{1+u^2}\frac{du}{dt}\frac{1}{x^2} \right) \cdot \frac{1}{x}\\
\end{align}
$$

### 2. Implicit Differentiation

###### Examples

$$
\begin{align}
&\text{Let } y=y(x) \text{ be defined by the equation } e^y+6xy+x^2-1=0, \text{ find } y''(0);\\
&\text{Let } y=y(x) \text{ be defined by the equation } xe^{f(y)}=e^y, f \text{ is twice differentiable, and } f'\neq1, \text{ find } y''(x).\\
&e^y \cdot y'+6y+6xy'+2x=0\\
&e^y \cdot y''+e^y \cdot (y')^2+6xy''+12y'+2=0\\
\end{align}
$$


## Examples

##### 1.

$$
\begin{align}
&\text{Given } f'(x_0)=-1, \text{ find } \lim_{x\rightarrow 0}\frac{x}{f(x_0-2x)-f(x_0-x)}=?\\
&\\
&\text{Solution 1:}\\
&I=\lim_{x\rightarrow 0}\frac{1}{-2\cdot\frac{f(x_0-2x)-f(x_0)}{-2x}+\frac{f(x_0-x)-f(x_0)}{-x}}=\frac{1}{-2(-1) + (-1)} = 1 \quad \text{(Note: Check sign alignment below)}\\
&\text{Alternative approach via definition:}\\
&I = \lim_{x\rightarrow 0}\frac{1}{\frac{f(x_0-2x)-f(x_0-x)}{x}} = \frac{1}{-2f'(x_0) - (-f'(x_0))} = \frac{1}{-f'(x_0)} = \frac{1}{-(-1)} = 1\\
\end{align}
$$

##### 2.

$$
\begin{align}
&\text{Suppose } f(x) \text{ is continuous at } x=0, \text{ and } \lim_{h\rightarrow0}\frac{f(h^2)}{h^2}=1, \text{ then:}\\
&(A) f(0)=0 \text{ and } f_{-}'(0) \text{ exists.}\\
&(B) f(0)=1 \text{ and } f_{-}'(0) \text{ exists.}\\
&(C) f(0)=0 \text{ and } f_{+}'(0) \text{ exists.}\\
&(D) f(0)=1 \text{ and } f_{+}'(0) \text{ exists.}\\
&\\
&\text{Solution:}\\
&\text{Since } f(x) \text{ is continuous at } x=0, \lim_{x\rightarrow 0}f(x)=f(0).\\
&\text{Since } \lim_{h\rightarrow0}\frac{f(h^2)}{h^2}=1, \text{ as } h\rightarrow0, h^2\rightarrow 0^+, \text{ meaning } \lim_{h\rightarrow 0}f(h^2)=0 \Rightarrow f(0)=0.\\
&\text{Let } t=h^2 \rightarrow 0^+, \text{ then } \lim_{t\rightarrow0^+}\frac{f(t)}{t}=\lim_{t\rightarrow0^+}\frac{f(t)-f(0)}{t-0}=1 \Leftrightarrow f_{+}'(0) \text{ exists and equals } 1. \text{ Thus, (C) is correct.}\\
\end{align}
$$

##### 3.

$$
\begin{align}
&\text{Let } f(x)=|x^3-1|\Phi(x), \text{ where } \Phi(x) \text{ is continuous at } x=1. \text{ Find the necessary and sufficient condition for } f(x) \text{ to be differentiable at } x=1 \text{ for } \Phi(1)=?\\
&\text{Analysis: Under what condition is } f(x) \text{ differentiable at } x=1?\\
&\text{Solution:}\\
&\text{Note that } f(1) = 0.\\
&\lim_{x\rightarrow 1^+}\frac{f(x)-f(1)}{x-1}=\lim_{x\rightarrow 1^+}\frac{(x^3-1)\Phi(x)}{x-1}=\lim_{x\rightarrow 1^+}\frac{(x-1)(x^2+x+1)\Phi(x)}{x-1}=3\Phi(1)\\
&\lim_{x\rightarrow 1^-}\frac{f(x)-f(1)}{x-1}=\lim_{x\rightarrow 1^-}\frac{-(x^3-1)\Phi(x)}{x-1}=\lim_{x\rightarrow 1^-}\frac{-(x-1)(x^2+x+1)\Phi(x)}{x-1}=-3\Phi(1)\\
&\\
&\text{For } f(x) \text{ to be differentiable at } x=1, \text{ the left and right derivatives must be equal: } 3\Phi(1)=-3\Phi(1) \Rightarrow \Phi(1)=0.\\
&\text{Note: } a^n-b^n=(a-b)(a^{n-1}b^0+a^{n-2}b^1+\dots+a^0b^{n-1})
\end{align}
$$

##### 4.

$$
\begin{align}
&\text{Suppose } f(x) \text{ is continuous at } x=1, \text{ and } \lim_{x\rightarrow 1}\frac{f(x)}{x-1}=2, \text{ find } f'(1).\\
&\text{Solution:}\\
&\lim_{x \rightarrow 1}f(x)=f(1)\\
&\text{Since the limit exists, } f(1) = \lim_{x \rightarrow 1}\frac{f(x)}{x-1}(x-1)=2\cdot 0=0\\
&f'(1)=\lim_{x \rightarrow 1}\frac{f(x)-f(1)}{x-1}=\lim_{x \rightarrow 1}\frac{f(x)}{x-1}=2
\end{align}
$$

### III. Differentiation of Functions Defined by Parametric Equations

$$
\begin{align}
&\text{Let } y=y(x) \text{ be defined by the parametric equations } \begin{cases}x=x(t)\\y=y(t)\end{cases}, \text{ then:}\\
&\frac{dy}{dx}=\frac{y'(t)}{x'(t)}\\
&\frac{d^2y}{dx^2}=\frac{d}{dt}\left(\frac{dy}{dx}\right)\cdot\frac{dt}{dx} = \frac{\frac{d}{dt}\left[\frac{y'(t)}{x'(t)}\right]}{x'(t)} = \frac{y''(t)x'(t)-y'(t)x''(t)}{[x'(t)]^3}\\
\end{align}
$$

Example:
$$
\begin{align}
&y=y(x) \text{ is defined by } \begin{cases}x=\ln (1+t^2)\\y=t-\arctan t\end{cases}, \text{ find } \frac{d^2y}{dx^2}.\\
&\frac{dy}{dx}=\frac{1 - \frac{1}{1+t^2}}{\frac{2t}{1+t^2}}=\frac{\frac{t^2}{1+t^2}}{\frac{2t}{1+t^2}}=\frac{t}{2}\\
&\frac{d^2y}{dx^2}=\frac{\frac{d}{dt}\left(\frac{t}{2}\right)}{x'(t)} = \frac{\frac{1}{2}}{\frac{2t}{1+t^2}}=\frac{t^2+1}{4t}\\
\end{align}
$$

### IV. Differentiation of Inverse Functions

$$
\begin{align}
&\text{Let } y=f(x) \text{ be twice differentiable and } f'(x)\neq0, \text{ its inverse function is } x=f^{-1}(y).\\
&\text{Then: } \frac{dx}{dy}=\frac{1}{\frac{dy}{dx}}=\frac{1}{f'(x)}, \quad \frac{d^2x}{dy^2}=\frac{d}{dy}\left(\frac{1}{f'(x)}\right) = -\frac{f''(x)\frac{dx}{dy}}{[f'(x)]^2} = -\frac{f''(x)}{[f'(x)]^3}
\end{align}
$$

### V. Differentiation of Piecewise Functions

Differentiate normally using standard rules within open intervals. At boundary points, apply definitions of limits/derivatives and verify if the left and right derivatives match.

$$
\begin{align}
&\text{Let } f(x)=\begin{cases}1-2x^2, &x\leq -1\\x^3, &-1< x\leq2\\12x-16, &x> 2\end{cases}\\
&(1) \text{ Find the inverse function } g(x).\\
&(2) \text{ Check if } g(x) \text{ has any points of discontinuity or non-differentiability.}
\end{align}
$$