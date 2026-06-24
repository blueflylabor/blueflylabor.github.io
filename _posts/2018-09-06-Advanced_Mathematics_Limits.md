---
title: "Advanced Mathematics: Limits"
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
---

# Limits

#### Definition of Limit

##### 1) Limit of a Sequence and a Function

$$
\begin{align}
&\lim_{n \rightarrow \infty}{x_n}=A \Leftrightarrow \text{For } \forall \epsilon > 0, \exists N, \text{ such that when } n > N, \text{ we have } |x_n-A|<\epsilon\\
&\lim_{x\rightarrow \infty}f(x)=A \Leftrightarrow \forall \epsilon > 0, \exists M > 0, \text{ such that when } |x| > M, \text{ we have } |f(x)-A|<\epsilon\\
&\lim_{x\rightarrow +\infty}f(x)=A \Leftrightarrow \forall \epsilon > 0, \exists M > 0, \text{ such that when } x > M, \text{ we have } |f(x)-A|<\epsilon\\
&\lim_{x\rightarrow -\infty}f(x)=A \Leftrightarrow \forall \epsilon > 0, \exists M > 0, \text{ such that when } x < -M, \text{ we have } |f(x)-A|<\epsilon\\
&\lim_{x\rightarrow x_0}f(x)=A \Leftrightarrow \forall \epsilon > 0, \exists \delta > 0, \text{ such that when } 0 < |x-x_0| < \delta, \text{ we have } |f(x)-A|<\epsilon\\
\end{align}
$$



#### Properties of Limits

##### 1) Local Preservation of Sign

$$
\begin{align}
&\text{If } \lim_{x\rightarrow x_0}f(x)=A > 0 \; (<0),\\
&\text{then } \exists \delta > 0, \text{ such that when } x \in U^0(x_0,\delta), \text{ we have } f(x) > 0 \; (<0)\\
\end{align}
$$

Corollary: Preservation of Order:
$$
\begin{align}
&\text{If } \lim_{x\rightarrow x_0}f(x)=A, \text{ and } A > \alpha \; (< \beta),\\
&\text{then } \exists \delta > 0, \text{ such that when } x \in U^0(x_0,\delta), \text{ we have } f(x) > \alpha \; (f(x) < \beta)\\
\end{align}
$$

##### 2) Local Boundedness

$$
\begin{align}
\text{If } \lim_{x\rightarrow x_0}f(x)=A, \text{ then } \exists U^0(x_0), \text{ such that } f(x) \text{ is bounded within } U^0(x_0).
\end{align}
$$

##### 3) Inequality Properties

$$
\begin{align}
&\text{If both } \lim_{x \rightarrow x_0}f(x)=A \text{ and } \lim_{x \rightarrow x_0}g(x)=B \text{ exist,}\\
&\text{and } f(x) \geq g(x) \text{ in a deleted neighborhood of } x_0,\\
&\text{then } A \geq B\\
&\\
&\text{Note: If the condition } f(x)\geq g(x) \text{ is replaced by } f(x) > g(x), \text{ the conclusion } A > B \text{ does not necessarily hold.}\\
&\text{For example, as } x\rightarrow +\infty, \; \frac{1}{x} > \frac{1}{x+1}, \text{ which cannot deduce } \lim_{x\rightarrow +\infty}\frac{1}{x} > \lim_{x\rightarrow +\infty}\frac{1}{x+1}.\\
&\text{However, it still deduces that } \lim_{x\rightarrow +\infty}\frac{1}{x} \geq \lim_{x\rightarrow +\infty}\frac{1}{x+1}.\\
\end{align}
$$

Corollary:
$$
\text{If } \lim_{x\rightarrow x_0}f(x) = A \text{ exists, and } f(x) \geq 0 \; (\leq 0), \text{ then } A \geq 0 \; (\leq 0).
$$

##### 4) Algebraic Operations

$$
\begin{align}
&\text{If } \lim f(x)=A \text{ and } \lim g(x)=B \text{ exist, then:}\\
&\lim [f(x) \pm g(x)]=A \pm B\\
&\lim f(x)g(x)=A \cdot B\\
&\lim \frac{f(x)}{g(x)}=\frac{A}{B}, \quad (B \neq 0)\\
&\\
&\text{Note: If } \lim f(x) \text{ does not exist, and } \lim g(x)=B \text{ exists,}\\
&\text{then } \lim [f(x) \pm g(x)] \text{ definitely does not exist.}\\
&\text{However, } \lim f(x)g(x) \text{ may or may not exist.}\\
&\\
\end{align}
$$

#### Sequence Limits

$$
\begin{align}
&\text{Definition 1: } \lim_{n\rightarrow \infty}x_n=A:\\
&\forall \epsilon > 0, \exists N > 0, \text{ such that when } n > N, \text{ we always have } |x_n-A|<\epsilon\\
&\text{Notes:}\\
&(1) \text{ The roles of } \epsilon \text{ and } N:\\
&\epsilon \text{ quantifies the closeness between } x_n \text{ and } A, \text{ whereas } N \text{ describes the process of } n\rightarrow \infty.\\
&(2) \text{ Geometric Meaning:}\\
&\text{For any given } \epsilon, \text{ in the } \epsilon\text{-neighborhood of } A, \text{ when } n \text{ is sufficiently large, only a finite number of terms}\\
&\text{lie outside the neighborhood, while all the remaining infinite terms fall inside } (A-\epsilon, A+\epsilon).\\
\end{align}
$$

![img](https://raw.githubusercontent.com/blueflylabor/images/main/%7DK27TS8@YE9O01$$JFWZL%7BGI.jpg)
$$
\begin{align}
&(3) \text{ Whether a sequence has a limit is completely independent of its first finite number of terms.}\\
\end{align}
$$
![img](https://raw.githubusercontent.com/blueflylabor/images/main/EYFSSCG11QCROZVR6BRRWCW.jpg)
$$
\begin{align}
&(4) \lim_{n\rightarrow\infty}x_n=a \Leftrightarrow \lim_{k\rightarrow\infty}x_{2k-1}=\lim_{k\rightarrow\infty}x_{2k}=a:\\
&\text{The existence of a sequence limit } \Rightarrow (\text{but } \notin) \text{ the existence of sub-sequence limits of odd and even terms.}\\
&\text{The existence of a sequence limit } \Leftrightarrow \text{ Limit of odd terms exists } = \text{ Limit of even terms exists.}\\
&eg: \text{ For } a_n=(-1)^n, \; a_{2k-1}=-1,-1,-1,\dots,-1; \; a_{2k}=1,1,1,\dots,1. \quad \lim_{k\rightarrow \infty}a_{2k-1} \neq \lim_{k\rightarrow \infty}a_{2k}.\\
\end{align}
$$

###### Example 1

![img](https://raw.githubusercontent.com/blueflylabor/images/main/%5BPJU%608@8W1YOB(~)XA%60J%7BYA.jpg)
$$
\begin{align}
&\text{Method 1: Splitting into Odd and Even Sequences}\\
&\text{Odd terms: } \lim_{n\rightarrow\infty}\left(\frac{n+1}{n}\right)^{-1}=1\\
&\text{Even terms: } \lim_{n\rightarrow\infty}\left(\frac{n+1}{n}\right)^{1}=1\\
&\text{Method 2: Scaling Method + Squeeze Theorem}\\
&\left(\frac{n+1}{n}\right)^{-1} \leq \left(\frac{n+1}{n}\right)^{(-1)^n} \leq \frac{n+1}{n}\\
&\because \lim_{n\rightarrow \infty}\left(\frac{n+1}{n}\right)^{-1}=1 \quad \text{and} \quad \lim_{n\rightarrow \infty}\frac{n+1}{n}=1\\
&\therefore I=\lim_{n\rightarrow \infty}\left(\frac{n+1}{n}\right)^{(-1)^n}=1\\
\end{align}
$$

###### Example 2

![eqfdsfas1312dsafdwftht](https://raw.githubusercontent.com/blueflylabor/images/main/eqfdsfas1312dsafdwftht.jpg)
$$
\begin{align}
&\text{(1) Solution: Apply the crucial triangle inequality } \left||a|-|b|\right| \leq |a-b|.\\
&\text{Since } \lim_{n\rightarrow\infty}x_n=a, \text{ by the definition of limits:}\\
&\forall \epsilon > 0, \exists N > 0, \text{ such that when } n > N, \; |x_n-a|<\epsilon.\\
&\text{According to } \left||x_n|-|a|\right| \leq |x_n-a|,\\
&\text{we have } \forall \epsilon > 0, \exists N > 0, \text{ such that when } n > N, \; \left||x_n|-|a|\right|<\epsilon.\\
&\text{Thus, } \lim_{n\rightarrow\infty}x_n=a \Rightarrow \lim_{n\rightarrow\infty}|x_n|=|a|.\\
&\text{The converse does not hold. For instance, if } x_n=(-1)^n, \text{ then } \lim_{n\rightarrow \infty}|x_n|=1=|1|, \text{ but } \lim_{n\rightarrow \infty}(-1)^n \text{ does not exist.}\\
&\text{(2) From (1), it follows that if } \lim_{n\rightarrow\infty}x_n=0, \text{ then } \lim_{n\rightarrow \infty}|x_n|=|0|=0.\\
&\text{Conversely, if } \lim_{n\rightarrow\infty}|x_n|=0, \text{ then } \forall \epsilon > 0, \exists N > 0, \text{ such that when } n > N, \; \left||x_n|-0\right|<\epsilon,\\
&\text{which directly means } |x_n-0|<\epsilon. \text{ Thus, } \lim_{n\rightarrow\infty}x_n=0 \Leftrightarrow \lim_{n\rightarrow\infty}|x_n|=0.\\
\end{align}
$$


#### Methods for Finding Sequence Limits:

$$
\begin{align}
&\text{(1) Transform the sequence limit into a function limit.}\\
\end{align}
$$

#### Function Limits

##### 1) Limits of Functions as the Independent Variable Approaches Infinity

![image-20210428170621601](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210428170621601.jpg)

###### Example

$$
\begin{align}
&\text{Evaluate the limit: } \lim_{x\rightarrow \infty}\frac{\sqrt{x^2+1}}{x}=?\\
&\text{Solution:}\\
&\because \sqrt{x^2}=|x|\\
&\text{We must split it into left and right limits:}\\
&\lim_{x\rightarrow +\infty}\frac{x\sqrt{1+\frac{1}{x^2}}}{x}=1\\
&\lim_{x\rightarrow -\infty}\frac{-x\sqrt{1+\frac{1}{x^2}}}{x}=-1\\
&\because \lim_{x\rightarrow -\infty}\frac{\sqrt{x^2+1}}{x} \neq \lim_{x\rightarrow +\infty}\frac{\sqrt{x^2+1}}{x}\\
&\therefore \lim_{x\rightarrow \infty}\frac{\sqrt{x^2+1}}{x} \text{ does not exist.}\\
\end{align}
$$

##### 2) Limits of Functions as the Independent Variable Approaches a Finite Value

![image-20210428172430632](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210428172430632.jpg)
$$
\begin{align}
&\text{Notes: (1) Due to the arbitrariness of } \epsilon, \text{ when relating } \epsilon \text{ and } \delta, \text{ we always have } |f(x)-A|<\epsilon\\
&\Rightarrow A-\epsilon<f(x)<A+\epsilon\\
&(2) \text{ Geometric Meaning: The function value exactly at } f(x_0) \text{ can be undefined;}\\
&\text{ the limit depends entirely on the function values within the deleted neighborhood.}\\
\end{align}
$$
![image-20210502180033951](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210502180033951.jpg)

##### Common Pitfall:

$$
\begin{align}
&\text{Correct: } \lim_{x\rightarrow 0}\frac{\sin x}{x}=1 \quad (x\rightarrow 0, \; x\neq 0)\\
&\text{Incorrect: } \lim_{x\rightarrow 0}\frac{\sin(x\sin \frac{1}{x})}{x\sin\frac{1}{x}}=1\\
&\text{To apply the standard limit, it must be guaranteed that } x\sin\frac{1}{x}\rightarrow 0 \text{ AND } x\sin\frac{1}{x}\neq 0.\\
&\text{That is, } x\sin\frac{1}{x}\neq0 \text{ must hold throughout a deleted neighborhood of } 0.\\
&\text{However, whenever } x=\frac{1}{n\pi} \text{ (where } n \in \mathbb{Z} \setminus \{0\}\text{), it makes } x\sin\frac{1}{x}=0.\\
&\text{Therefore, the original limit does not exist.}\\
\end{align}
$$

![image-20210502181857574](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210502181857574.jpg)

![223567833](https://raw.githubusercontent.com/blueflylabor/images/main/223567833.jpg)

![1619951164(1)](https://raw.githubusercontent.com/blueflylabor/images/main/1619951164(1).jpg)

#### Properties of Limits

$$
\begin{align}
&\text{1) Boundedness}\\
&\text{(1) Boundedness of Convergent Sequences: If a sequence } \{x_n\} \text{ converges, then it must be bounded.}\\
&\text{Proof outline: If } x_n\rightarrow A, \text{ for } n>N, \; |x_n| \leq M_1. \text{ Including the first } N \text{ terms, } |x_n| \leq M.\\
\end{align}
$$

![123](https://raw.githubusercontent.com/blueflylabor/images/main/1222222222222222321.jpg)
$$
\begin{align}
&\text{The first finite terms form a finite set, so there must exist a bound greater than the maximum of those terms.}\\
&\text{Convergence } \Rightarrow (\text{but } \notin) \text{ Boundedness.}\\
&eg: x_n=(-1)^n \text{ is bounded but does not converge.}\\
&\text{(2) Local Boundedness of Functions: If } \lim_{x\rightarrow x_0}f(x) \text{ exists, then } f(x) \text{ is bounded in some deleted neighborhood of } x_0.\\
&\lim_{x\rightarrow x_0}f(x) \text{ exists } \Rightarrow (\text{but } \notin) \text{ Local boundedness (bounded in a deleted neighborhood).}\\
&eg: \text{ For } f(x)=\sin\frac{1}{x}, \; f(x) \text{ is bounded in a deleted neighborhood of } 0, \text{ but } \lim_{x\rightarrow 0}{\sin\frac{1}{x}} \text{ does not exist.}\\
&\text{2) Preservation of Sign}\\
&\text{(1) Preservation of Sign for Sequence Limits:}\\
&\text{Let } \lim_{n\rightarrow \infty}{x_n}=A.\\
&[1] \text{ If } A > 0 \; (\text{or } A < 0), \text{ then there exists } N > 0, \text{ such that when } n > N, \; x_n > 0 \; (\text{or } x_n < 0).\\
&[2] \text{ If there exists } N > 0 \text{ such that when } n > N, \; x_n\geq 0 \; (\text{or } x_n\leq0), \text{ then } A\geq0 \; (\text{or } A\leq 0).\\
&\text{(2) Preservation of Sign for Function Limits:}\\
&[1] \text{ If } A > 0 \; (\text{or } A < 0), \text{ then there exists } \delta > 0, \text{ such that when } x\in \dot{U}(x_0,\delta), \; f(x) > 0 \; (\text{or } f(x) < 0).\\
&[2] \text{ If there exists } \delta > 0 \text{ such that when } x\in \dot{U}(x_0,\delta), \; f(x)\geq0 \; (\text{or } f(x)\leq0), \text{ then } A\geq0 \; (\text{or } A\leq0).\\
\end{align}
$$

![](https://raw.githubusercontent.com/blueflylabor/images/main/sssaaaa.jpg)