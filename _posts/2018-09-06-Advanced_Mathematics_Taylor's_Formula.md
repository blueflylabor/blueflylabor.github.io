---
title: "Advanced Mathematics: Taylor's Formula"
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
lang: en
---

## Taylor's Formula

In mathematics, Taylor's formula is a formula that utilizes information about a function at a specific point to describe its values in the vicinity of that point. If a function is sufficiently smooth, and given the values of its derivatives of all orders at a single point, Taylor's formula can use these derivative values as coefficients to construct a polynomial that approximates the function's value within a neighborhood of this point. Taylor's formula also provides the deviation between this polynomial and the actual function value.

Core Idea: Approximating and substituting a target function using a polynomial function.

The following derivations focus on Taylor's formula with the Peano form of the remainder.

### 1. Derivation of Taylor's Formula

---

$$
(1) \sin x
$$

---

First, by finding the $n$-th order derivatives of $f(x)=\sin x$, a clear pattern emerges:
$$
\sin x \rightarrow \cos x \rightarrow -\sin x \rightarrow -\cos x
$$
Approximating and substituting with a polynomial function:
$$
g(x)=\sum_{i=0}^{n}a_ix^i
$$
We obtain the following derivation:
$$
\begin{align}
g^{(0)}(x)&=\sin x =a_0x^0+a_1x^1+a_2x^2+a_3x^3+a_4x^4+a_5x^5+...+a_nx^n\\
g^{(1)}(x)&=\cos x =a_1x^0+2a_2x^1+3a_3x^2+4a_4x^3+5a_5x^4+...+a_nx^n\\
g^{(2)}(x)&=-\sin x =2*1a_2x^0+3*2a_3x^1+4*3a_4x^2+5*4a_5x^3+...+a_nx^n\\
g^{(3)}(x)&=-\cos x=3*2*1a_3x^0+4*3*2a_4x^1+5*4*3a_5x^2+...+a_nx^n\\
g^{(4)}(x)&=\sin x=4*3*2*1a_4x^0+5*4*3*2a_5x^1+...+a_nx^n\\
g^{(5)}(x)&=\cos x=5*4*3*2*1a_5x^0+...+a_nx^n
\end{align}
$$
When $x=0$:
$$
\begin{align}
0&=a_0\\
+1&=1*a_1\\
0&=2*1*a_2\\
-1&=3*2*1*a_3\\
0&=4*3*2*1a_4\\
+1&=5*4*3*2*1*a_5\\
\end{align}
$$
By induction, we get:
$$
a_k= \begin{cases}
0 & \text{remainder is 0 when divided by 4} \\
\frac{1}{k!} & \text{remainder is 1 when divided by 4} \\
0 & \text{remainder is 2 when divided by 4} \\
\frac{-1}{k!} & \text{remainder is 3 when divided by 4} \\
\end{cases}
$$
Thus, it can be concluded that:
$$
\sin x=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+...+(-1)^{n-1}\frac{x^{2n-1}}{(2n-1)!}+o(x^{2n-1})
$$
Based on the same core concept and derivation method, Taylor expansions can be performed on other basic elementary functions.

---

$$
(2) e^x
$$

---

Discovering the differentiation pattern:
$$
e^x \rightarrow e^x \rightarrow e^x \rightarrow e^x
$$

$$
\begin{align}
g^{(0)}(x)&=e^x =a_0x^0+a_1x^1+a_2x^2+a_3x^3+a_4x^4+a_5x^5+...+a_nx^n\\
g^{(1)}(x)&=e^x =a_1x^0+2a_2x^1+3a_3x^2+4a_4x^3+5a_5x^4+...+a_nx^n\\
g^{(2)}(x)&=e^x =2*1a_2x^0+3*2a_3x^1+4*3a_4x^2+5*4a_5x^3+...+a_nx^n\\
\end{align}
$$

When $x=0$:
$$
\begin{align}
1&=a_0\\
1&=1*a_1\\
1&=2*1*a_2\\
\end{align}
$$
By induction, we get:
$$
\begin{align}
a_k=\frac{1}{k!}
\end{align}
$$
Thus, it can be concluded that:
$$
e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+...+\frac{x^n}{n!}+o(x^n)
$$

---

$$
(3) \ln(1+x)
$$

---

Discovering the differentiation pattern:
$$
\ln(1+x)\rightarrow (1+x)^{-1}\rightarrow (-1)(1+x)^{-2}\rightarrow (-2)(1+x)^{-3}
$$

$$
\begin{align}
g^{(0)}(x)&=\ln(1+x) =a_0x^0+a_1x^1+a_2x^2+a_3x^3+a_4x^4+a_5x^5+...+a_nx^n\\
g^{(1)}(x)&=(1+x)^{-1} =a_1x^0+2a_2x^1+3a_3x^2+4a_4x^3+5a_5x^4+...+a_nx^n\\
g^{(2)}(x)&=(-1)(1+x)^{-2} =2*1a_2x^0+3*2a_3x^1+4*3a_4x^2+5*4a_5x^3+...+a_nx^n\\
g^{(3)}(x)&=(-1)^2(1+x)^{-3}=3*2*1a_3x^0+4*3*2a_4x^1+5*4*3a_5x^2+...+a_nx^n\\
g^{(4)}(x)&=(-1)^3(1+x)^{-4}=4*3*2*1a_4x^0+5*4*3*2a_5x^1+...+a_nx^n\\
g^{(5)}(x)&=(-1)^4(1+x)^{-5}=5*4*3*2*1a_5x^0+...+a_nx^n
\end{align}
$$
When $x=0$:
$$
\begin{align}
0&=a_0\\
1&=1*a_1\\
-1&=2*1*a_2\\
1&=3*2*1*a_3\\
-1&=4*3*2*1*a_4\\
1&=5*4*3*2*1*a_5\\
\end{align}
$$
By induction, we get:
$$
\begin{align}
a_k=\frac{(-1)^{k-1}}{k}
\end{align}
$$
Thus, it can be concluded that:
$$
\ln(1+x)=x-\frac{x^2}{2}+\frac{x^3}{3}+...+ \frac{(-1)^{n-1}x^n}{n}+o(x^n)
$$


---

$$
(4) \cos x
$$

---

Discovering the differentiation pattern:
$$
\cos x\rightarrow -\sin x\rightarrow -\cos x\rightarrow \sin x\rightarrow \cos x
$$



$$
\begin{align}
g^{(0)}(x)&=\cos x =a_0x^0+a_1x^1+a_2x^2+a_3x^3+a_4x^4+a_5x^5+...+a_nx^n\\
g^{(1)}(x)&=-\sin x =a_1x^0+2a_2x^1+3a_3x^2+4a_4x^3+5a_5x^4+...+a_nx^n\\
g^{(2)}(x)&=-\cos x =2*1a_2x^0+3*2a_3x^1+4*3a_4x^2+5*4a_5x^3+...+a_nx^n\\
g^{(3)}(x)&=\sin x=3*2*1a_3x^0+4*3*2a_4x^1+5*4*3a_5x^2+...+a_nx^n\\
g^{(4)}(x)&=\cos x=4*3*2*1a_4x^0+5*4*3*2a_5x^1+...+a_nx^n\\
g^{(5)}(x)&=\sin x=5*4*3*2*1a_5x^0+...+a_nx^n
\end{align}
$$
When $x=0$:
$$
\begin{align}
1&=a_0\\
0&=1*a_1\\
-1&=2*1*a_2\\
0&=3*2*1*a_3\\
1&=4*3*2*1*a_4\\
0&=5*4*3*2*1*a_5\\
\end{align}
$$
By induction, we get:
$$
a_k= \begin{cases}
\frac{1}{k!}& \text{remainder is 0 when divided by 4} \\
0 & \text{remainder is 1 when divided by 4} \\
\frac{-1}{k!}  & \text{remainder is 2 when divided by 4} \\
0& \text{remainder is 3 when divided by 4} \\
\end{cases}
$$
Thus, it can be concluded that:
$$
\cos x=1-\frac{x^{2}}{2!}+\frac{x^4}{4!}-\frac{x^6}{6!}+...+(-1)^{n}\frac{x^{2n}}{(2n)!}+o(x^{2n})
$$

---

$$
(5) (1+x)^\alpha
$$

---

Discovering the differentiation pattern:
$$
(1+x)^\alpha\rightarrow \alpha(1+x)^{\alpha-1}\rightarrow \alpha(\alpha-1)(1+x)^{\alpha-2}\rightarrow \alpha(\alpha-1)(\alpha-2)(1+x)^{\alpha-3}
$$

$$
\begin{align}
g^{(0)}(x)&=(1+x)^\alpha =a_0x^0+a_1x^1+a_2x^2+a_3x^3+a_4x^4+a_5x^5+...+a_nx^n\\
g^{(1)}(x)&=\alpha(1+x)^{\alpha-1} =a_1x^0+2a_2x^1+3a_3x^2+4a_4x^3+5a_5x^4+...+a_nx^n\\
g^{(2)}(x)&=\alpha(\alpha-1)(1+x)^{\alpha-2} =2*1a_2x^0+3*2a_3x^1+4*3a_4x^2+5*4a_5x^3+...+a_nx^n\\
g^{(3)}(x)&=\alpha(\alpha-1)(\alpha-2)(1+x)^{\alpha-3}=3*2*1a_3x^0+4*3*2a_4x^1+5*4*3a_5x^2+...+a_nx^n\\
\end{align}
$$

When $x=0$:
$$
\begin{align}
1&=a_0\\
\alpha&=1*a_1\\
\alpha(\alpha-1)&=2*1*a_2\\
\alpha(\alpha-1)(\alpha-2)&=3*2*1*a_3\\
\end{align}
$$
By induction, we get:
$$
a_k=\frac{\alpha(\alpha-1)(\alpha-2)...(\alpha-k+1)}{k!}
$$
Thus, it can be concluded that:
$$
(1+x)^\alpha=1+\alpha x+\frac{\alpha(\alpha-1)x^2}{2!}+\frac{\alpha(\alpha-1)(\alpha-2)x^3}{3!}+...+\frac{\alpha(\alpha-1)(\alpha-2)...(\alpha-n+1)x^n}{n!}+o(x^n)
$$

---

## 2. Peano vs. Lagrange Form of the Remainder

### (1) Taylor's Formula with Peano Form of the Remainder
$$
\begin{align}
&\text{If } f(x) \text{ possesses derivatives up to the } n\text{-th order at point } x_0, \text{ then:}\\
&f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2!}f''(x_0)(x-x_0)^2+...+\frac{1}{n!}f^{(n)}(x_0)(x-x_0)^n+o[(x-x_0)^{n}]\\
&\text{When } x_0=0, \text{ we obtain the Maclaurin formula:}\\
&f(x)=f(0)+f'(0)x+\frac{1}{2!}f''(0)x^2+...+\frac{1}{n!}f^{(n)}(0)x^n+o(x^n)
\end{align}
$$

### (2) Taylor's Formula with Lagrange Form of the Remainder
$$
\begin{align}
&\text{Assume that the function } f(x) \text{ has an } (n+1)\text{-th order derivative in an open interval } (a,b) \text{ containing } x_0. \text{ Then for } x\in(a,b):\\
&f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2!}f''(x_0)(x-x_0)^2+...+\frac{1}{n!}f^{(n)}(x_0)(x-x_0)^n+R_n(x)\\
&\text{where } R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}. \text{ Here, } \xi \text{ lies between } x_0 \text{ and } x, \text{ which is known as the Lagrange remainder.}
\end{align}
$$

### (3) Key Differences

**1. Difference in Target Scope:**

Taylor's formula with the Lagrange remainder describes global behavior.
$$
\text{Lagrange Remainder (Global)} \rightarrow \begin{cases}
\text{Extrema / Optimization}\\
\text{Inequalities}
\end{cases}
$$

Taylor's formula with the Peano remainder describes local behavior.
$$
\text{Peano Remainder (Local)} \rightarrow \begin{cases}
\text{Limits}\\
\text{Local Extremum / Critical Points}
\end{cases}
$$

**2. Difference in Structural Expression:**

The Lagrange remainder uses a concrete evaluation expression, defined as an $(n+1)$-th order derivative evaluated at $\xi$ multiplied by $(x-x_0)^{n+1}$.

The Peano form of the remainder lacks a specific valuation expression and is strictly an expression of a higher-order infinitesimal: $R_n(x)=o((x-x_0)^n)$.

**3. Difference in Formula Calculation Approaches:**

The Maclaurin formula is a specialized variation of Taylor's formula evaluated at $a=0$ (setting $\xi=\theta x$).

The Peano form of the remainder is written as $R_n(x) = o(x^n)$.

Therefore, during expansion, one only needs to carry out the calculation to the specific order required by the problem statement.