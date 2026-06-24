---
title: "Advanced Mathematics: Functions"
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
---

### Functions

#### Definition: Two Elements—Domain & Rule of Correspondence

$$
\begin{align}
&y=f(x),x\in R  \Leftrightarrow y=f(t),t\in R\\
&\int_{a}^{b}f(t)dt=\int_{a}^{b}f(x)dx\\
&\sqrt{x^2}=|x|=(x^{2})^{\frac{1}{2}}\\
&-\sqrt{x^2}=-|x|=-(x^2)^{\frac{1}{2}}
\end{align}
$$

Example:
$$
\begin{align}
&\text{Prove: } \left|\frac{x}{1+x^2}\right|\leq\frac{1}{2}\\
&\text{Using } a^2+b^2\geq 2ab:\\
&\text{Proof: } 1+x^2 \geq 2|x| \Rightarrow \frac{1}{1+x^2} \leq \frac{1}{2|x|} \quad (x \neq 0)\\
&\left|\frac{x}{1+x^2}\right| = \frac{|x|}{1+x^2} \leq \frac{|x|}{2|x|} = \frac{1}{2}\\
&\text{When } x=0, \text{ the inequality holds trivially. Hence, verified.}
\end{align}
$$


#### Basic Elementary Functions:

$$
\begin{align}
&\text{Constant function: } y=c\\
&\text{Power function: } y=x^a\\
&\text{Exponential function: } y=a^x\\
&\text{Logarithmic function: } y=\log_{a}x\\
&\text{Trigonometric functions: } y=\sin x ,y=\cos x,y=\tan x,y=\csc x=\frac{1}{\sin x},y=\sec x=\frac{1}{\cos x},y=\cot x=\frac{1}{\tan x}\\
&\text{Inverse trigonometric functions: } y=\arcsin x,y=\arccos x,y=\arctan x,y=\text{arccot } x
\end{align}
$$



##### $\arcsin x$ & $\sin x$

![arcsinx&sinx](https://raw.githubusercontent.com/blueflylabor/images/main/arcsinx&sinx.jpg)
$$
\begin{align}
&y=\arcsin x,x \in [-1,1],y \in \left[-\frac{\pi}{2},\frac{\pi}{2}\right]\\
&\arcsin \left(x+\frac{\sqrt{2}}{2}\right)=+\frac{\pi}{4} \quad (\text{Note: standard reference evaluation when inner value is } \frac{\sqrt{2}}{2})\\
&\arcsin \left(x-\frac{\sqrt{2}}{2}\right)=-\frac{\pi}{4} \quad (\text{Note: standard reference evaluation when inner value is } -\frac{\sqrt{2}}{2})\\
&\arcsin (+1)=+\frac{\pi}{2}\\
&\arcsin (-1)=-\frac{\pi}{2}\\
&\arcsin x \text{ is not the inverse function of } \sin x \text{ in a global sense; it is merely the inverse mapping of } \sin x \text{ restricted within } x \in \left[-\frac{\pi}{2},\frac{\pi}{2}\right].
\end{align}
$$


##### $\arccos x$ & $\cos x$

![arccosx&cosx](https://raw.githubusercontent.com/blueflylabor/images/main/arccosx&cosx.jpg)
$$
\begin{align}
&y=\arccos x,x \in [-1,1],y \in [0,\pi]\\
&\arccos (+1)=0\\
&\arccos (-1)=\pi\\
&\arccos \left(\frac{1}{2}\right)=\frac{\pi}{3}\\
&\arccos x \text{ is not the inverse function of } \cos x \text{ in a global sense; it is merely the inverse mapping of } \cos x \text{ restricted within } x \in [0,\pi].
\end{align}
$$

##### $\arctan x$ & $\tan x$

![arctanx&tanx](https://raw.githubusercontent.com/blueflylabor/images/main/arctanx&tanx.jpg)
$$
\begin{align}
&y=\arctan x,x \in [-\infty,+\infty],y \in \left(-\frac{\pi}{2},\frac{\pi}{2}\right)\\
&\arctan (+\infty)=\frac{\pi}{2}\\
&\arctan (-\infty)=-\frac{\pi}{2}
\end{align}
$$

##### $\text{arccot } x$ & $\cot x$

##### ![arc&noarc](https://raw.githubusercontent.com/blueflylabor/images/main/arc&noarc.jpg)

$$
\begin{align}
&y=\text{arccot } x,x \in [-\infty,+\infty],y \in (0,\pi)\\
&\text{arccot } (+\infty)=0\\
&\text{arccot } (-\infty)=\pi
\end{align}
$$

$$
\begin{align}
&\text{If a continuous function has an inverse function, its inverse function is also a continuous function.}\\
&\text{If a differentiable function has an inverse function, its inverse function is not necessarily differentiable.}\\
&\text{(e.g., the differentiable function } x^3 \text{ has an inverse function } x^{\frac{1}{3}}\text{, which is not differentiable at } x=0.)\\
\end{align}
$$

Common Trigonometric Identities:
$$
\begin{align}
&\sin^{2}x+\cos^{2}x=1 \\
&\sin 2x=2\sin x \cos x=\frac{2\tan x}{1+\tan^{2}x}\\
&\cos 2x=\cos^{2}x-\sin^{2}x=1-2\sin^{2}x=2\cos^{2}x-1=\frac{1-\tan^{2}x}{1+\tan^{2}x}\\
&1+\tan^{2}x=\sec^{2}x\\
&1+\cot^{2}x=\csc^{2}x\\
&\arcsin x+\arccos x=\frac{\pi}{2} \quad (\forall x\in[-1,1])\\
&\arctan x+\text{arccot } x=\frac{\pi}{2} \quad (\forall x\in(-\infty,+\infty))\\
&\arcsin x+\arcsin \sqrt{1-x^2}=\frac{\pi}{2} \quad (\forall x\in[0,1])\\
&\arctan x+\arctan \frac{1}{x}=\frac{\pi}{2} \quad (\forall x \in(-\infty,0)\cup(0,+\infty))
\end{align}
$$


#### Properties of Functions

##### 1) Parity (Even / Odd)

$$
\begin{align}
&\text{Odd function: } f(-x)=-f(x), \quad \text{Even function: } f(-x)=f(x)\\
&\text{If } f \text{ and } g \text{ are odd functions, is the composite function } f(g(x)) \text{ odd or even?}\\
&\text{Using the definition:}\\
&f(g(-x))=f(-g(x))=-f(g(x)) \Leftrightarrow f(g(x)) \text{ is an odd function.}
\end{align}
$$

###### Methods to Determine Parity:

(1) Definition

(2) Algebraic operations: The algebraic sum of odd functions is an odd function; the algebraic sum of even functions is an even function. The product of an odd function and an even function is an odd function.

(3) Composite operations: If at least one of the inner or outer functions is even, the composite function is even. If both the inner and outer functions are odd, the composite function is odd.

(4) Calculus operations: The derivative of an odd function is an even function, and the derivative of an even function is an odd function. The antiderivative (primitive function) of an odd function is an even function, but the antiderivative of an even function is not necessarily an odd function.
$$
\text{e.g., if } f(x)=1 \text{ (even), its primitive } F(x)=x+1 \text{ is not odd, but } \int_{0}^{x}f(t)dt=x \text{ is odd.}
$$


##### 2) Periodicity

##### 3) Monotonicity

#### Elementary Functions

Functions obtained from basic elementary functions through a finite number of algebraic operations (addition, subtraction, multiplication, division) and composite operations.

The absolute value of an elementary function is still an elementary function.

#### Boundedness of Functions

$$
\begin{align}
&\text{Bounded} \Leftrightarrow \text{Bounded both above and below}\\
&\text{Unbounded} \Leftrightarrow \text{Lacks at least one bound (above, below, or both)}
\end{align}
$$



Methods to Determine Boundedness:
$$
\begin{align}
&1) \text{ Definition: apply inequalities to the absolute value of the function to bound it by a positive constant;}\\
&\text{ or determine that the function is bounded via its global maximum and minimum values.}\\
&2) \text{ If } f(x) \text{ is bounded on both domains } D_1 \text{ and } D_2, \text{ then it is also bounded on } D_1\cup D_2.\\
&3) \text{ A continuous function on a closed interval must be bounded (a continuous function on an open interval is not necessarily bounded, e.g., } \tan x\text{).}\\
&4) \text{ A convergent sequence must be bounded.}\\
&5) \text{ A function possessing a limit is locally bounded:}\\
&\lim_{x \rightarrow x_0}f(x) \Rightarrow f(x) \text{ must be bounded in a certain deleted neighborhood of } x_0.\\
&\lim_{x \rightarrow +\infty}f(x) \Rightarrow \exists M, \text{ such that } f(x) \text{ is bounded in } (M,+\infty).
\end{align}
$$

Methods to Determine Unboundedness:
$$
\begin{align}
&\text{An infinitely large quantity must be unbounded (e.g., } \lim_{x\rightarrow \frac{\pi}{2}}\tan x=\infty\text{).}\\
&\text{Local unboundedness implies global unboundedness (e.g., } f(x)=x\sin x\text{).}
\end{align}
$$
Example:
$$
\begin{align}
&\text{Prove: } f(x)=\frac{\ln x}{x-1} \text{ is unbounded in } (0,1) \text{ and bounded in } (1,+\infty).\\
&\text{Proof: } \lim_{x\rightarrow0^{+}}f(x)=+\infty \Rightarrow f(x) \text{ is unbounded in } (0,\epsilon) \Rightarrow f(x) \text{ is unbounded in } (0,1).\\
&\lim_{x\rightarrow1^{+}}f(x)=1, \text{ hence } \exists \epsilon > 0, \text{ such that } f(x) \text{ is bounded in } (1,1+\epsilon).\\
&\lim_{x\rightarrow+\infty}f(x)=0, \text{ hence } \exists M > 1+\epsilon, \text{ such that } f(x) \text{ is bounded in } (M,+\infty).\\
&\text{Since } f(x) \text{ is continuous on the closed interval } [1+\epsilon, M], \text{ it is bounded on it. Therefore, } f(x) \text{ is bounded in } (1,+\infty).\\
\end{align}
$$
![image-20210331002025536](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210331002025536.jpg)
$$
\begin{align}
&\text{Notes: } f(x) \text{ is undefined at } x=1, \text{ causing a global discontinuity, which requires a case-by-case analysis.}\\
&\text{As } x\rightarrow0^+, \text{ the limit of the function in the deleted neighborhood } (0,\epsilon) \text{ is } +\infty. \text{ Thus, it has no upper bound in this neighborhood,}\\
&\text{which means it is unbounded in the neighborhood, and consequently } f(x) \text{ is unbounded in } (0,1).\\
&\text{As } x\rightarrow1^+, \text{ the limit of the function in the deleted neighborhood } (1,1+\epsilon) \text{ equals } 1, \text{ hence it is bounded within this neighborhood.}\\
&\text{As } x\rightarrow +\infty, \text{ the limit in the neighborhood is } 0. \text{ Since } f(x) \text{ is a continuous function on } (1,+\infty), \text{ it can be concluded that } f(x) \text{ is bounded for } x\in (1,+\infty).
\end{align}
$$

#### Composite Functions

$$
y=f(u), \; u=g(x) , \quad y\leftarrow u \leftarrow x \quad (\text{Chain propagation})
$$

Example:
$$
\begin{align}
&\text{Given } f(\sqrt[3]{x}-1)=x-1, \text{ find } f(x).\\
&\text{Solution: Let } u=\sqrt[3]{x}-1.\\
&x=(u+1)^3 \Rightarrow f(u)=(u+1)^3-1\\
&f(x)=(x+1)^3-1\\
& \\
&\text{Notes: Match and construct the right-hand expression directly when inverse solving is difficult.}\\
\end{align}
$$

#### Inverse Functions

Notes:
$$
\begin{align}
&1. \text{ A function with a one-to-one correspondence possesses an inverse function; hence, a strictly monotonic function on an interval must have an inverse.}\\
&\text{The function } y=\sin x \text{ has no global inverse function, but } y=\sin x \text{ restricted to } x\in\left[-\frac{\pi}{2},\frac{\pi}{2}\right] \text{ does.}\\
&2. \text{ The functions } x=f^{-1}(y) \text{ and } y=f^{-1}(x) \text{ represent the same structural inverse mapping. The graph of the former is identical to } y=f(x),\\
&\text{whereas the graph of the latter is symmetric to } y=f(x) \text{ with respect to the line } y=x.\\
&3. \text{ For } \forall x\in D, \, f^{-1}(f(x))=x; \text{ when } y\in f(D), \, f(f^{-1}(y))=y.
\end{align}
$$
Example:
$$
\begin{align}
&\text{Given } \arcsin (\sin \theta)=\frac{\pi}{4}, \text{ and } \theta \in\left(\frac{\pi}{2},\pi\right), \text{ find } \theta.\\
&\text{Solution: } \theta \in \left(\frac{\pi}{2},\pi\right) \Rightarrow \pi-\theta \in \left(0,\frac{\pi}{2}\right) \\
&\arcsin (\sin(\pi - \theta))=\frac{\pi}{4} \Rightarrow \pi - \theta=\frac{\pi}{4}\\
&\theta=\frac{3\pi}{4}
\end{align}
$$