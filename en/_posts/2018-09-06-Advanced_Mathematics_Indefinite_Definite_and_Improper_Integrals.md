---
title: Advanced Mathematics: Indefinite, Definite, and Improper Integrals
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
lang: en

---

# Indefinite, Definite, and Improper Integrals

## Indefinite Integrals

### I. Concepts of Indefinite Integral

#### 1. Definition

$$
\begin{align}
&\text{Antiderivative: If for every point } x \text{ in the interval } I, \text{ we have } F'(x)=f(x), \text{ then } F(x) \text{ is called an antiderivative of } f(x) \text{ on } I.\\
&\text{Indefinite Integral: If } f(x) \text{ has antiderivatives on } I, \text{ the set of all its antiderivatives is called the indefinite integral of } f(x) \text{ on } I, \text{ denoted as } \int{f(x)dx}.\\
&\text{Linearity: } \int[\alpha f(x)+\beta g(x)]dx=\alpha\int f(x)dx+\beta\int g(x)dx\\
\end{align}
$$

#### 2. Evaluation

$$
\begin{align}
&\text{Methods of Evaluation}\begin{cases}&1. \text{Basic Formulas}\\&2. \text{Linearity}\\&3. \text{Integration Methods}\begin{cases}&1. \text{Substitution Method}\\&2. \text{Integration by Parts}\\\end{cases}\\\end{cases}\\
\end{align} 
$$

##### (1) Integration by Substitution - First Method (Differential Matching)

$$
\begin{align}
&\text{Let } F'(u)=f(u), \text{ then } \int{f(\Phi(x))\Phi'(x)}dx=\int{f(\Phi(x))d(\Phi(x))}=F(\Phi(x))+C\\
&\text{Note: Find a suitable differential matching } \Phi'(x)dx=d(\Phi(x))
\end{align}
$$

Common Differential Matchings:
$$
\begin{align}
&1. \int{f(ax+b)dx=\frac{1}{a}\int{f(ax+b)d(ax+b)}} \quad (a\neq0)\\
&\text{eg1. } \int{\sin (2x+3)}dx=\frac{1}{2}\int\sin (2x+3)d(2x+3)=-\frac{1}{2}\cos{(2x+3)}+C\\
&2. \int{f(ax^n+b)x^{n-1}dx}=\frac{1}{na}\int{f(ax^n+b)d(ax^n+b)}\\
&\text{eg2. } \int{\cos(2x^4+3)x^3dx}=\frac{1}{4\cdot2}\int{\cos(2x^4+3)d(2x^4+3)}=\frac{1}{8}\sin{(2x^4+3)}+C\\
&3. \int{f(a^x+c)a^xdx}=\frac{1}{\ln{a}}\int{f(a^x+c)}d(a^x+c)\\
&\text{eg3. } \int{\sin(2^x+3)2^xdx}=\frac{1}{\ln2}\int{\sin{(2^x+3)}d(2^x+3)}=-\frac{1}{\ln 2}\cos{(2^x+3)}+C\\
&4. \int{f\left(\frac{1}{x}\right)\frac{1}{x^2}}dx=-\int{f\left(\frac{1}{x}\right)}d\left(\frac{1}{x}\right)\\
&\text{eg4. } \int{\ln\left(\frac{1}{x}\right)}\frac{1}{x^2}dx=-\int\ln \left(\frac{1}{x}\right)d\left({\frac{1}{x}}\right)+C\\
&5. \int{f(\ln |x|})\frac{1}{x}dx=\int{f(\ln{|x|)}}{d(\ln|x|)}\\
&\text{eg5. } \int{\sin ({\ln{|x|}}})\frac{1}{x}dx=\int{\sin(\ln|x|)d(\ln{|x|})}=-\cos(\ln |x|)+C\\
&6. \int{f(\sqrt x)\frac{1}{\sqrt x}}dx=2\int{f(\sqrt x)}d(\sqrt x)\\
&7. \int f(\sin x)\cos xdx=\int{f(\sin x)}d(\sin x)\\
&8. \int{f(\cos x)\sin x}dx=-\int{f(\cos x)d(\cos x)}\\
&9. \int{f(\tan x)\sec^2 xdx}=\int{f(\tan x)d(\tan x)}\\
&10. \int{f(\cot x)\csc^2xdx}=-\int{f(\cot x)d{(\cot x)}}\\
&11. \int{f\left(\arcsin x\right)\frac{1}{\sqrt{1-x^2}}}dx=\int{f(\arcsin x)d({\arcsin x})}\\
&12. \int{f(\arccos x)\left(-\frac{1}{\sqrt{1-x^2}}\right)}dx=\int{f(\arccos x)d(\arccos x)}\\
&13. \int{f(\arctan x)\frac{1}{1+x^2}dx}=\int{f(\arctan x)d(\arctan x)}\\
&14. \int{f(\sqrt{x^2\pm a})}\frac{x}{\sqrt{x^2\pm a}}dx=\int{f(\sqrt{x^2\pm a})}d(\sqrt{x^2\pm a})\\
&\text{Note: } (\sqrt{x^2\pm a})'=\frac{x}{\sqrt{x^2\pm a}}, \quad (\sqrt{a^2-x^2})'=\frac{-x}{\sqrt{a^2-x^2}}\\
\end{align}
$$

##### (2) Integration by Substitution - Second Method

$$
\begin{align}
&\text{Let } F'(u)=f(\Phi(u))\Phi'(u), \text{ then }\\
&\int{f(x)dx}\overset{x=\Phi(u)}{=}\int{f(\Phi(u))\Phi'(u)du}=F(u)+C=F(\Phi^{-1}(x))+C\\
&\text{Note: Find a suitable substitution } x=\Phi(u)\\
\end{align}
$$

1) Trigonometric Substitution
$$
\begin{align}
&x=a\sin u, \quad x=a\tan u, \quad x=a \sec u\\
&\sqrt{a^2-x^2}\overset{x=a\sin u}{=}a\cos u, \quad u\in\left[-\frac{\pi}{2},\frac{\pi}{2}\right], \quad x\in[-a,a]\\
&\sqrt{a^2+x^2}\overset{x=a\tan u}{=}a\sec u, \quad u\in{\left(-\frac{\pi}{2},\frac{\pi}{2}\right)}, \quad x\in{(-\infty,\infty)}\\
&\sqrt{x^2-a^2}\overset{x=a\sec u}{=}a\tan u, \quad u\in\left(\frac{\pi}{2},\pi\right]\cup\left(0,\frac{\pi}{2}\right]\\
\end{align}
$$
2) Reciprocal Substitution
$$
\begin{align}
&x=\frac{1}{u} \quad \text{commonly used for functions containing } \frac{1}{x}\\
\end{align}
$$
3) Exponential (or Logarithmic) Substitution
$$
\begin{align}
&a^x=u \text{ or } x=\frac{\ln u}{\ln a} \quad \text{commonly used for functions containing } a^x\\
\end{align}
$$
4) Substitutions for Rationalization
$$
\begin{align}
&\text{For } \frac{1}{\sqrt{x}+\sqrt[3]{x}}, \text{ use } x=u^6\\
&\text{For } \sqrt[n]{\frac{ax+b}{cx+d}}, \text{ use } u=\sqrt[n]{\frac{ax+b}{cx+d}} \text{ or } x=-\frac{du^n-b}{cu^n-a}\\
\end{align}
$$

##### (3) Integration by Parts

$$
\begin{align}
&\int{u(x)v'(x)dx}=\int{u(x)d(v(x))}=u(x)v(x)-\int{v(x)u'(x)dx}\\
&\text{Note: Find suitable functions } u(x) \text{ and } v(x)\\
\end{align}
$$

1) Degree Reduction Method
$$
\begin{align}
&\int{x^ne^{ax}dx}, \quad \int{x^n\sin axdx}, \quad \int{x^n\cos ax dx}\\
&\text{Choose } u(x)=x^n\\
\end{align}
$$
2) Order Raising Method
$$
\begin{align}
&\int{x^a\ln xdx}, \quad \int{x^a\arcsin xdx}, \quad \int{x^a\arccos x dx}, \quad \int{x^a\arctan x dx}\\
&\text{Choose } u(x)=\ln x, \arcsin x, \text{ etc.}\\
\end{align}
$$
3) Cyclic Method
$$
\begin{align}
&\int{e^{ax}\sin bx dx}, \quad \int{e^{ax}\cos {bx} dx}\\
&\text{Choose } u(x)=e^{ax} \text{ or } \sin{bx}\\
\end{align}
$$
4) Reduction Formula Method
$$
\begin{align}
&\text{For results } I_n \text{ involving } n, \text{ establish a recursive relation } I_n=f(I_{n-1}) \text{ or } f(I_{n-2})\\
\end{align}
$$

## Definite Integrals

### I. Concepts of Definite Integral

#### 1. Definition

$$
\begin{align}
&\text{Definition: Let the function } f(x) \text{ be defined and bounded on the interval } [a,b].\\
&(1) \text{ Partition: Divide } [a,b] \text{ into } n \text{ subintervals } [x_{i-1},x_{i}].\\
&(2) \text{ Summation: Choose a point } \xi_{i} \in [x_{i-1},x_{i}], \text{ form the sum } \sum_{i=1}^{n}{f(\xi_{i})\Delta x_i}, \text{ where } \lambda=\max\{\Delta x_{1},\Delta x_{2},...,\Delta x_{n}\}.\\
&(3) \text{ Limit: If } \lim_{\lambda \rightarrow 0}{\sum_{i=1}^{n}f(\xi_{i})\Delta x_i} \text{ exists and the limit is independent of the partition of } [a,b] \text{ and choice of } \xi_{i},\\
&\text{then } f(x) \text{ is integrable on } [a,b], \text{ denoted as } \int^{b}_{a}{f(x)dx}=\lim_{\lambda \rightarrow 0}{\sum_{i=1}^{n}f(\xi_i)\Delta x_{i}}\\
&\\
&\text{Notes:}\\
&(1) \lambda \rightarrow 0 \implies n \rightarrow \infty, \text{ but } n \rightarrow \infty \nimplies \lambda \rightarrow 0 \text{ unless the partition is regular.}\\
&(2) \text{ A definite integral represents a single value, dependent on } [a,b] \text{ but independent of the variable of integration } x.\\
&\int_{a}^{b}{f(x)dx}=\int_{a}^{b}{f(t)dt}\\
&(3) \text{ If } \int_{0}^{1}{f(x)dx} \text{ exists, dividing } [0,1] \text{ into } n \text{ equal parts gives } \Delta{x_{i}}=\frac{1}{n}. \text{ Choosing } \xi_{i}=\frac{i}{n}:\\
&\int_{0}^{1}f(x)dx=\lim_{\lambda \rightarrow 0}{\sum_{i=1}^{n}{f(\xi_{i})\Delta x_{i}}}=\lim_{n\rightarrow \infty}\sum_{i=1}^{n}\frac{1}{n}f\left(\frac{i}{n}\right)\\
\end{align}
$$

$$
\begin{align}
&\int^{b}_{a}{f(x)dx}=\lim_{\lambda \rightarrow 0}\sum^{n}_{i=1}f(\xi_i)\Delta x_i=\begin{cases}&\lim_{n\rightarrow \infty}{\sum_{i=1}^{n}{f\left(a+(i-1)\frac{b-a}{n}\right)\frac{b-a}{n}}}, \text{ Left endpoints}\\\\&\lim_{n\rightarrow \infty}{\sum_{i=1}^{n}{f\left(a+i\frac{b-a}{n}\right)\frac{b-a}{n}}}, \text{ Right endpoints}\\\end{cases}\\
&\text{Midpoints: } \xi_i=a+(i-1)\frac{b-a}{n}+\frac{b-a}{2n}\\
\end{align}
$$

![image-20210613172601984](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210613172601984.jpg)

Theorem: (Linearity)
$$
\begin{align}
&\int_a^b[\alpha f(x)+\beta g(x)]dx=\alpha\int_a^b f(x)dx+\beta\int_a^b g(x)dx\\
\end{align}
$$
Note: Integration is subtle
$$
\begin{align}
&\int{e^{\pm x^2}dx}, \int{\frac{\sin x}{x}dx} \text{ cannot be integrated in terms of elementary functions.}\\
&\text{If } F'(x)=f(x), x\in I, \text{ any continuous function must have antiderivatives (infinitely many).}\\
&[F(x)+C]'=f(x)
\end{align}
$$


#### 2. Sufficient Conditions for the Existence of Definite Integrals

$$
\begin{align}
&\text{If } f(x) \text{ is continuous on } [a,b], \text{ then } \int^{b}_{a}{f(x)dx} \text{ definitely exists.}\\
&\text{If } f(x) \text{ is bounded on } [a,b] \text{ and has only finitely many discontinuities, then } \int^{b}_{a}{f(x)dx} \text{ definitely exists.}\\
&\text{If } f(x) \text{ has only finitely many jump or removable discontinuities on } [a,b], \text{ then } \int^{b}_{a}{f(x)dx} \text{ definitely exists.}\\
\end{align}
$$

#### 3. Geometric Meaning of Definite Integrals

![image-20210405155729433](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210405155729433.jpg)
$$
\begin{align}
&(1) \text{ If } f(x)\geqslant{0}, \quad \int_{a}^{b}{f(x)dx}=S\\
\end{align}
$$
![image-20210405155859329](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210405155859329.jpg)
$$
\begin{align}
&(2) \text{ If } f(x)\leqslant{0}, \quad \int_{a}^{b}{f(x)dx}=-S\\
\end{align}
$$


![image-20210405155556537](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210405155556537.jpg)
$$
\begin{align}
&(3) \text{ General case: } \int_{a}^{b}{f(x)dx}=S_1+S_3-S_2\\
\end{align}
$$

Notes:
$$
\begin{align}
&\text{(1) When } f(x)\geq0, \text{ the geometric meaning is the area of the curvilinear trapezoid bounded by } y=f(x), x=a, x=b, \text{ and the } x\text{-axis.}\\
&\text{(2) A definite integral is a constant that depends only on } f \text{ and the interval } [a,b], \text{ not on the variable symbol.}\\
&\int_a^b{f(x)}dx=\int_a^b{f(t)dt}\\
&\text{(3) } \int_a^bdx=b-a\\
&\text{(4) } \int_{a}^{a}f(x)dx=0, \quad \int_a^bf(x)dx=-\int_b^a{f(t)}dt
\end{align}
$$

### II. Properties of Definite Integrals

#### 1. Inequality Properties

$$
\begin{align}
&(1) \text{ Monotonicity: If } f(x)\leqslant{g(x)} \text{ on } [a,b], \text{ then } \int_a^{b}{f(x)dx}\leqslant{\int_a^{b}{g(x)dx}}\\
&\text{Corollaries:}\\
&(1) \text{ If } f(x)\geq0 \text{ for all } x\in[a,b], \text{ then } \int_a^b{f(x)dx}\geq0\\
&(2) \text{ If } f(x)\geq0 \text{ for all } x\in[a,b] \text{ and } [c,d]\subset[a,b], \text{ then } \int_a^b{f(x)dx}\geq\int_c^d{f(x)dx}\\
&(3) \left|\int_a^bf(x)dx\right|\leq\int_a^b{|f(x)|dx}\\
&\quad -|f|\leq f\leq |f|\implies \int_a^b-|f|dx\leq \int_a^bfdx\leq \int_a^b|f|dx\implies \left|\int_a^bfdx\right|\leq\int_a^b|f|dx\\
&\quad \text{Example: Since } x^3\leq x^2 \text{ on } [0,1], \text{ then } \int_0^1{x^3dx}\leq\int_0^1{x^2dx}\\
\end{align}
$$

$$
\begin{align}
&(4) \text{ (Estimation Theorem) If } M \text{ and } m \text{ are the maximum and minimum values of } f(x) \text{ on } [a,b] \text{ respectively,}\\
&\text{then } m(b-a)\leqslant{\int_a^{b}{f(x)dx}\leqslant{M(b-a)}}\\
\end{align}
$$

![geogebra-export](https://raw.githubusercontent.com/blueflylabor/images/main/geogebra-export.jpg)
$$
\begin{align}
&\text{Proof: } M(b-a)=S_{AFDC}=S_1+S_2+S_3\\
&m(b-a)=S_{EBDC}=S_3\\
&\int_a^{b}{f(x)dx}=S_{ADBC}=S_2+S_3\\
&\text{Since } S_3\leqslant{S_2+S_3\leqslant{S_1+S_2+S_3}}\\
&\implies{m(b-a)\leqslant{\int_a^{b}{f(x)dx}\leqslant{M(b-a)}}}\\
\end{align}
$$


#### 2. Mean Value Theorems

$$
\begin{align}
&(1) \text{ First Mean Value Theorem for Integrals: If } f(x) \text{ is continuous on } [a,b], \text{ then } \int_a^{b}{f(x)dx}=f(\xi)(b-a), \quad (a<\xi<b)\\
&\text{The value } \frac{1}{b-a}{\int_{a}^{b}{f(x)dx}} \text{ is called the average value of the function } y=f(x) \text{ on the interval } [a,b].\\
&\text{Note: Let } F'(x)=f(x), \text{ then } F(b)-F(a)=\int_a^b{f(x)dx}, \text{ and } f(\xi)(b-a)=F'(\xi)(b-a) \text{ by MVT.}\\
&(2) \text{ Generalized MVTI: If } f(x), g(x) \text{ are continuous on } [a,b], \text{ and } g(x) \text{ does not change sign, then } \int_{a}^{b}{f(x)g(x)dx}=f(\xi)\int_a^b{g(x)dx}\\
\end{align}
$$

Notes:
$$
\begin{align}
&\text{Consider } \int_0^1{\frac{x}{\sin x}}dx\\
&f(x)=\begin{cases}&\frac{x}{\sin x}, \quad x\in(0,1]\\&1, \quad x=0\\\end{cases}\\
&\text{Conclusion: Modifying a function at finitely many points does not affect its definite integral.}\\
&\text{For } f(x)={\begin{cases}&x+1, \quad x\in[1,2]\\&x, \quad x\in[0,1)\\\end{cases}}\\
&\int_0^2{f(x)dx}=\int_0^1{xdx}+\int_1^2{(x+1)dx}\\
\end{align}
$$

$$
\begin{align}
&\text{Proof Example: Show that } \frac{1}{2}\leq\int_0^{\frac{1}{2}}\frac{1}{\sqrt{1-x^n}}dx\leq\frac{\pi}{6} \quad (n \geq 2)\\
&\text{Estimation: For } x\in\left[0,\frac{1}{2}\right], \text{ we have } 1 \leq \frac{1}{\sqrt{1-x^n}} \leq \frac{1}{\sqrt{1-x^2}}\\
&\implies \int_0^{\frac{1}{2}} 1 dx \leq \int_0^{\frac{1}{2}}\frac{1}{\sqrt{1-x^n}}dx \leq \int_0^{\frac{1}{2}}\frac{1}{\sqrt{1-x^2}}dx = \arcsin\left(\frac{1}{2}\right) = \frac{\pi}{6}\\
\end{align}
$$

Examples:
$$
\begin{align}
&1. \text{ Find the limit } \lim_{n\rightarrow \infty}\int_0^1{\frac{x^ne^x}{1+e^x}dx}\\
&\text{Since } 0\leq\frac{x^ne^x}{1+e^x}\leq x^n \text{ for } x\in[0,1], \text{ by monotonicity:}\\
&0\leq\int_0^1{\frac{x^ne^x}{1+e^x}dx}\leq \int_0^1{x^n}dx=\frac{1}{n+1}\\
&\text{Using the Squeeze Theorem and knowing } \lim_{n\rightarrow\infty}\frac{1}{n+1}=0:\\
&\lim_{n\rightarrow \infty}\int_0^1{\frac{x^ne^x}{1+e^x}dx}=0\\
\end{align}
$$

$$
\begin{align}
&2. \text{ Let } I_1=\int_0^{\frac{\pi}{4}}\frac{\tan x}{x}dx, \quad I_2=\int_0^{\frac{\pi}{4}}\frac{x}{\tan x}dx. \text{ Then:}\\
&\text{(A) } I_1 < I_2 < 1 \quad \text{(B) } I_2 < 1 < I_1 \quad \text{(C) } I_2 < I_1 < 1 \quad \text{(D) } 1 < I_2 < I_1\\
&\text{Solution: Use monotonicity. Since } \tan x > x \text{ for } x\in\left(0,\frac{\pi}{2}\right):\\
&\frac{\tan x}{x} > 1 > \frac{x}{\tan x} \quad \text{on } \left(0,\frac{\pi}{4}\right]\\
&\text{Integrating over } \left[0,\frac{\pi}{4}\right]:\\
&\int_0^{\frac{\pi}{4}}\frac{x}{\tan x}dx < \int_0^{\frac{\pi}{4}}1dx = \frac{\pi}{4} < 1\\
&\text{For } I_1, \text{ by the integral mean value theorem:}\\
&I_1 = \int_0^{\frac{\pi}{4}}\frac{\tan x}{x}dx = \frac{\tan \xi}{\xi}\left(\frac{\pi}{4}\right) \quad \text{for some } \xi\in\left(0,\frac{\pi}{4}\right)\\
&\text{Since } \frac{\tan x}{x} \text{ is strictly increasing on } \left(0,\frac{\pi}{4}\right], \text{ its minimum value approaches } 1 \text{ as } x \rightarrow 0^.\\
&\text{Thus } \frac{\tan \xi}{\xi} > 1, \text{ but this doesn't immediately force } I_1 > 1 \text{ since } \frac{\pi}{4} < 1.\\
&\text{Let's re-evaluate the relation: } I_2 < \frac{\pi}{4} < 1.\\
&\text{Actually, evaluating at the endpoint } x=\frac{\pi}{4}, \frac{\tan(\pi/4)}{\pi/4} = \frac{4}{\pi} > 1.\\
&\text{Therefore, } I_2 < 1. \text{ For } I_1, \text{ since } \frac{\tan x}{x} > 1 \implies I_1 > \frac{\pi}{4}. \text{ Let's look at choice structural design:}\\
&\text{Since } \frac{\tan x}{x} > 1 > \frac{x}{\tan x}, \text{ then } I_2 < \frac{\pi}{4} < I_1. \text{ Also } I_1 = \frac{\tan\xi}{\xi}\frac{\pi}{4}. \text{ Since } \xi < \frac{\pi}{4}, \frac{\tan\xi}{\xi} < \frac{4}{\pi} \implies I_1 < 1.\\
&\text{Thus } I_2 < I_1 < 1. \text{ Choose (C).}\\
\end{align}
$$



### III. Functions Defined by Integrals (Variable Upper Limit)

![image-20210405152647772](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210405152647772.jpg)
$$
\begin{align}
&\text{If } f(x) \text{ is continuous on } [a,b], \text{ then } \Phi(x)=\int_a^x{f(t)dt} \text{ is differentiable on } [a,b], \text{ and:}\\
&\left(\int_a^xf(t)dt\right)'=f(x), \quad \left(\int_a^{x^2}f(t)dt\right)'=f(x^2)\cdot 2x\\
&\text{If } f(x) \text{ is continuous on } [a,b], \text{ and } \phi_1(x), \phi_2(x) \text{ are differentiable functions, then:}\\
&\left(\int_{\phi_1(x)}^{\phi_2(x)}{f(t)dt}\right)' = f[\phi_2(x)]\cdot\phi_2'(x)-f[\phi_1(x)]\cdot\phi_1'(x)\\
\\
&\text{Let } f(x) \text{ be continuous on } [-l,l]:\\
&\text{If } f(x) \text{ is an odd function, then } \int_0^xf(t)dt \text{ must be an even function.}\\
&\text{If } f(x) \text{ is an even function, then } \int_0^xf(t)dt \text{ must be an odd function.}\\
\end{align}
$$


$$
\begin{align}
&\text{Proof sketch: Take } x\in[a,b), \Delta x > 0 \text{ such that } x+\Delta x\in[a,b):\\
&\frac{\Delta F}{\Delta x}=\frac{F(x+\Delta x)-F(x)}{\Delta x}=\frac{1}{\Delta x}\left[\int_a^{x+\Delta x}f(t)dt-\int_a^xf(t)dt\right]=\frac{1}{\Delta x}\int_x^{x+\Delta x}f(t)dt=f(x+\theta\Delta x)\rightarrow f(x) \quad (\Delta x\rightarrow 0^+)\\
\end{align}
$$
Corollaries:
$$
\begin{align}
&\text{If } f(x), \phi'(x), \psi'(x) \text{ are continuous on } [a,b], \text{ then:}\\
&(1) \left(\int_a^{\phi(x)}f(t)dt\right)'=f(\phi(x))\phi'(x)\\
&(2) \left(\int_{\psi(x)}^b f(t)dt\right)'=-f(\psi(x))\psi'(x)\\
&(3) \left(\int_{\psi(x)}^\phi(x)}f(t)dt\right)'=f(\phi(x))\phi'(x)-\f(\psi(x))\psi'(x)\\
\end{align}
$$
Examples:
$$
\begin{align}
&1. \text{ If } f(x) \text{ is continuous and odd on } \mathbb{R}, \text{ show that its antiderivatives are even. What if } f(x) \text{ is even?}\\
&\text{Proof:}\\
&\text{Let } F_0(x) = \int_0^xf(t)dt, \quad x\in \mathbb{R}\\
&F_0(-x)=\int_0^{-x}f(t)dt\overset{t=-u}{=}\int_0^x f(-u)d(-u)=\int_0^x -f(u)(-du)=\int_0^x f(u)du=F_0(x) \implies F_0(x) \text{ is even.}\\
\end{align}
$$


$$
\begin{align}
&\text{Find the derivatives of the following functions:}\\
&(1) F(x)=\int_x^{e^{-x}}f(t)dt\\
&(2) F(x)=\int_0^{x^2}(x^2-t)f(t)dt\\
&(3) F(x)=\int_0^{x}f(x^2-t^2)dt\\
&(4) \text{ Let } y=y(x) \text{ be defined parametrically by } \begin{cases}&x=1+2t^2\\&y=\int_1^{1+2\ln t}\frac{e^u}{u}du\\\end{cases} \quad (t>0), \text{ find } \frac{d^2y}{dx^2}\Big|_{x=9}\\
&\text{Solution:}\\
&(1) F'(x) = f(e^{-x})(-e^{-x})-f(x)\\
&(2) F(x) = x^2\int_0^{x^2}f(t)dt - \int_0^{x^2}tf(t)dt\\
&\quad F'(x) = 2x\int_0^{x^2}f(t)dt + x^2 f(x^2)(2x) - x^2 f(x^2)(2x) = 2x\int_0^{x^2}f(t)dt\\
&(3) \text{ To find } F'(x) \text{ for } F(x)=\int_0^{x}f(x^2-t)dt, \text{ substitute } u = x^2-t \implies dt = -du:\\
&\quad F(x) = -\int_{x^2}^{x^2-x} f(u)du = \int_{x^2-x}^{x^2} f(u)du\\
&\quad F'(x) = f(x^2)(2x) - f(x^2-x)(2x-1)\\
&(4) \frac{dy}{dt} = \frac{e^{1+2\ln t}}{1+2\ln t} \cdot \frac{2}{t} = \frac{et^2}{1+2\ln t} \cdot \frac{2}{t} = \frac{2et}{1+2\ln t}\\
&\quad \frac{dx}{dt} = 4t \implies \frac{dy}{dx} = \frac{\frac{2et}{1+2\ln t}}{4t} = \frac{e}{2(1+2\ln t)}\\
&\quad \frac{d^2y}{dx^2} = \frac{d}{dt}\left(\frac{dy}{dx}\right) \Big/ \frac{dx}{dt} = \left( \frac{-2e}{2t(1+2\ln t)^2} \right) \Big/ (4t) = -\frac{e}{4t^2(1+2\ln t)^2}\\
&\quad \text{At } x=9 \implies 1+2t^2=9 \implies t=2 \text{ (since } t>0).\\
&\quad \frac{d^2y}{dx^2}\Big|_{x=9} = -\frac{e}{16(1+2\ln 2)^2}\\
\end{align}
$$

$$
\begin{align}
&2. \text{ Evaluation involving variable limits:}\\
&(1) \text{ Let } f(x)=\int_0^x{\frac{\sin t}{\pi -t}dt}, \text{ evaluate } \int_0^\pi{f(x)}dx.\\
&\text{Solution: Using Integration by Parts}\\
&\int_0^\pi{f(x)}dx = xf(x)\Big|_0^{\pi} - \int_0^{\pi}x f'(x) dx = \pi f(\pi) - \int_0^{\pi}\frac{x\sin x}{\pi -x}dx\\
&= \pi\int_0^{\pi}\frac{\sin x}{\pi -x}dx - \int_0^{\pi}\frac{x\sin x}{\pi -x}dx = \int_0^{\pi}\frac{(\pi-x)\sin x}{\pi-x}dx = \int_0^{\pi}\sin xdx = 2\\
&(2) \lim_{x\rightarrow\infty}{\frac{\left(\int_0^x{e^{t^2}}dt\right)^2}{\int_0^xe^{2t^2}dt}} = \lim_{x\rightarrow\infty}{\frac{2\left(\int_0^{x}e^{t^2}dt\right)e^{x^2}}{e^{2x^2}}} = \lim_{x\rightarrow\infty}\frac{2\int_0^{x}e^{t^2}dt}{e^{x^2}} = \lim_{x\rightarrow\infty}\frac{2e^{x^2}}{2xe^{x^2}} = \lim_{x\rightarrow\infty}\frac{1}{x} = 0\\
\end{align}
$$

$$
\begin{align}
&(3) \text{ Let } f(x) \text{ be continuous, } \phi(x)=\int_0^1{f(tx)dt}, \text{ and } \lim_{x\rightarrow0}\frac{f(x)}{x}=A, \text{ find } \phi'(x) \text{ and discuss its continuity at } x=0.\\
&\text{When } x\neq0:\\
&\text{Substitute } u=tx \implies dt = \frac{du}{x}. \text{ When } t=0 \implies u=0; t=1 \implies u=x.\\
&\phi(x) = \frac{1}{x}\int_0^x{f(u)du}\\
&\phi'(x) = \frac{xf(x)-\int_0^xf(u)du}{x^2}\\
&\text{When } x=0, \text{ since } \lim_{x\rightarrow0}\frac{f(x)}{x}=A \implies f(0)=0. \text{ Thus } \phi(0)=f(0)=0.\\
&\phi'(0) = \lim_{x\rightarrow0}\frac{\phi(x)-\phi(0)}{x-0} = \lim_{x\rightarrow0}\frac{\int_0^xf(u)du}{x^2} = \lim_{x\rightarrow 0}\frac{f(x)}{2x} = \frac{1}{2}A\\
&\lim_{x\rightarrow0}\phi'(x) = \lim_{x\rightarrow 0}{\frac{xf(x)-\int_0^xf(u)du}{x^2}} \overset{\text{L'Hopital}}{=} \lim_{x\rightarrow 0}\frac{f(x)+xf'(x)-f(x)}{2x} = \frac{1}{2}f'(0) = \frac{1}{2}A\\
&\text{Since } \lim_{x\rightarrow0}\phi'(x) = \phi'(0), \phi'(x) \text{ is continuous at } x=0.\\
\end{align}
$$

Note:
$$
\begin{align}
&\text{Be mindful of limit transformations when changing variables in integral functions:}\\
&\text{For example, } F(x)=\int_0^x{f(t)dt}\overset{t=-u}{=}-\int_0^{-x}f(-u)du = \int_{-x}^0 f(-u)du\\
\end{align}
$$


### IV. Evaluation of Definite Integrals

#### 1. Newton-Leibniz Formula

$$
\int_a^bf(x)dx=F(x)\Big|_a^b=F(b)-F(a)
$$

#### 2. Integration by Substitution

$$
\int_a^bf(x)dx=\int_\alpha^\beta{f(\Phi(t))\Phi'(t)dt} \quad \text{where } \Phi(\alpha)=a, \Phi(\beta)=b
$$

#### 3. Integration by Parts

$$
\int_a^budv=uv\Big|_a^b-\int_a^bvdu
$$

#### 4. Symmetry and Periodicity

$$
\begin{align}
&(1) \text{ Let } f(x) \text{ be continuous on } [-a,a] \quad (a>0):\\
&\int_{-a}^{a}f(x)dx=\begin{cases}0,&f(x) \text{ is odd}\\2\int_0^af(x)dx,&f(x) \text{ is even}\end{cases}\\
\end{align}
$$

$$
\begin{align}
&(2) \text{ If } f(x) \text{ is continuous with period } T, \text{ then for any } a:\\
&\int_a^{a+T}f(x)dx=\int_0^T{f(x)dx}\\
\end{align}
$$

$$
\begin{align}
&\text{Linear mapping: } \Phi: x\in[a,b]\rightarrow y\in[c,d], \text{ set } \frac{x-a}{b-a}=\frac{y-c}{d-c} \implies y=c+\frac{d-c}{b-a}(x-a)\\
\end{align}
$$

![image-20210617160041903](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210617160041903.jpg)

#### 5. Parity of Derivatives and Integrals

##### 1. Parity of Derivatives

$$
\begin{align}
&(1) \text{ If } f(x) \text{ is odd:}\\
&f(-x)=-f(x) \implies -f'(-x)=-f'(x) \implies f'(-x)=f'(x) \implies f'(x) \text{ is even.}\\
&(2) \text{ If } f(x) \text{ is even:}\\
&f(-x)=f(x) \implies -f'(-x)=f'(x) \implies f'(-x)=-f'(x) \implies f'(x) \text{ is odd.}\\
\end{align}
$$

##### 2. Parity of Integrals

$$
\begin{align}
&\text{Let } F(x) = \int_0^x f(t)dt:\\
&(1) \text{ If } f(x) \text{ is odd, then } F(x) \text{ is even.}\\
&(2) \text{ If } f(x) \text{ is even, then } F(x) \text{ is odd.}\\
\end{align}
$$

##### 3. Parity of Composite Functions

$$
\begin{align}
&\text{Let } F(x)=f(g(x)):\\
&\text{If } f(x) \text{ is even } \implies F(x) \text{ is always even.}\\
&\text{If } f(x) \text{ is odd and } g(x) \text{ is odd } \implies F(x) \text{ is odd.}\\
&\text{If } f(x) \text{ is odd and } g(x) \text{ is even } \implies F(x) \text{ is even.}\\
&\text{Rule of thumb: "Outer even makes it even, outer odd follows the inner."}\\
\end{align}
$$



Examples:
$$
\begin{align}
&1. \text{ Let } M=\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}{\frac{\sin x}{1+x^2}\cos^4xdx}, \quad N=\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}{(\sin^3 x+\cos^4x)dx}, \quad P=\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}(x^2\sin^3x-\cos^4x)dx. \text{ Then:}\\
&\text{(A) } N<P<M \quad \text{(B) } M<P<N \quad \text{(C) } N<M<P \quad \text{(D) } P<M<N\\
&\text{Solution: By symmetry,}\\
&M: \text{ The integrand is odd } \implies M = 0.\\
&N = \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\sin^3 xdx + \int_{-\frac{\frac{\pi}{2}}^{\frac{\pi}{2}}\cos^4 xdx = 0 + 2\int_0^{\frac{\pi}{2}}\cos^4 xdx > 0 \implies N > 0.\\
&P = \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}x^2\sin^3 xdx - \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\cos^4 xdx = 0 - 2\int_0^{\frac{\pi}{2}}\cos^4 xdx < 0 \implies P < 0.\\
&\implies P < M < N. \quad \text{Choose (D).}\\
\end{align}
$$

$$
\begin{align}
&2. \text{ Given } f(x)=\begin{cases}&kx, \quad 0\leq x\leq \frac{1}{2}a\\&c, \quad \frac{1}{2}a<x\leq a\\\end{cases}, \text{ find } F(x)=\int_0^xf(t)dt \text{ for } x\in[0,a].\\
&\text{Solution:}\\
&\text{For } 0 \leq x \leq \frac{1}{2}a: \quad F(x) = \int_0^x kt dt = \frac{1}{2}kx^2\\
&\text{For } \frac{1}{2}a < x \leq a: \quad F(x) = \int_0^{\frac{1}{2}a} kt dt + \int_{\frac{1}{2}a}^x c dt = \frac{1}{8}ka^2 + c\left(x-\frac{1}{2}a\right)\\
\end{align}
$$

$$
\begin{align}
&3. \text{ Prove: } \int_0^{2\pi}f(|\cos x|)dx=4\int_0^{\frac{\pi}{2}}f(|\cos x|)dx\\
\end{align}
$$

![1111](https://raw.githubusercontent.com/blueflylabor/images/main/e0e1f27ff16b0cf00a8f3d155bfc3423.jpg)

#### 6. Established Formulas

$$
\begin{align}
&(1) \text{ Wallis' Formula: } \int_0^{\frac{\pi}{2}}{\sin^nxdx}=\int_0^{\frac{\pi}{2}}\cos^n xdx=\begin{cases}\frac{n-1}{n}\cdot\frac{n-3}{n-2}\cdots\frac{1}{2}\cdot\frac{\pi}{2},&n \text{ is even}\\\frac{n-1}{n}\cdot\frac{n-3}{n-2}\cdots\frac{2}{3},&n \text{ is odd } (>1)\end{cases}\\
&(2) \int_0^{\pi}xf(\sin x)dx=\frac{\pi}{2}\int_0^{\pi}f(\sin x)dx \quad (f(x) \text{ is continuous})\\
\end{align}
$$

#### 7. Definite Integral Proofs



#### 8. Classical Examples:

##### Example 1:

$$
\begin{align}
&\text{Evaluate } \lim_{n\rightarrow \infty}{\left(\frac{1}{n+1}+\frac{1}{n+2}+\cdots+\frac{1}{n+n}\right)}\\
&\text{Method 1: Riemann Sum (Definite Integral Definition)}\\
&\lim_{n\rightarrow \infty} \sum_{i=1}^n \frac{1}{n+i} = \lim_{n\rightarrow \infty} \frac{1}{n} \sum_{i=1}^n \frac{1}{1+\frac{i}{n}} = \int_0^1 \frac{1}{1+x} dx = \ln(1+x)\Big|_0^1 = \ln 2\\
\end{align}
$$



##### Example 2

$$
\begin{align}
&\text{Let } f(x)=\int_0^x{\frac{\sin t}{\pi-t}dt}, \text{ evaluate } \int_0^{\pi}f(x)dx.\\
&\text{Method 1: Integration by Parts}\\
&\text{Identical to Example III.2(1), yielding a result of } 2.\\
\end{align}
$$

##### Example 3

![img](https://raw.githubusercontent.com/blueflylabor/images/main/AN%L6IJ6TF[%1UB3OUWMRCR.jpg)

![123](https://raw.githubusercontent.com/blueflylabor/images/main/123.jpg)
$$
\begin{align}
&\text{Method 1: Construct an auxiliary function}\\
&\text{From the problem, } f(1)=f(-1)=1, f(0)=-1. \text{ Assuming an even polynomial function like } f(x)=2x^2-1:\\
&\text{We can test the conditions to match the graph trends directly.}\\
\end{align}
$$


![image-20210408160543049](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210408160543049.jpg)



##### Example 4

![img](https://raw.githubusercontent.com/blueflylabor/images/main/Q8%7DOT_25(HC79%5BS_21)AZZK.jpg)

$$
\begin{align}
&\text{Given } \lim_{x\rightarrow 0}{\frac{ax-\sin x}{\int_b^x{\frac{\ln(1+t^3)}{t}dt}}}=c \quad (c\neq 0)\\
&\text{Since the limit exists and the numerator goes to } 0, \text{ the denominator must go to } 0 \implies b=0.\\
&\text{Applying L'Hopital's Rule:}\\
&\lim_{x\rightarrow 0}{\frac{a-\cos x}{\frac{\ln(1+x^3)}{x}}} = \lim_{x\rightarrow 0}{\frac{a-\cos x}{x^2}} = c\\
&\text{For the limit to be non-zero and finite, the numerator must be } 0 \text{ at } x=0 \implies a=1.\\
&\lim_{x\rightarrow 0}{\frac{1-\cos x}{x^2}} = \frac{1}{2} \implies c=\frac{1}{2}. \text{ Thus, } a=1, b=0, c=\frac{1}{2}.\\
\end{align}
$$

## Improper Integrals

### I. Improper Integrals over Infinite Intervals

$$
\begin{align}
&(1) \int_a^{+\infty}{f(x)}dx=\lim_{t\rightarrow +\infty}{\int_{a}^{t}f(x)dx}\\
&(2) \int_{-\infty}^{b}{f(x)}dx=\lim_{t\rightarrow -\infty}{\int_{t}^{b}f(x)dx}\\
&(3) \int_{-\infty}^{+\infty}{f(x)}dx = \int_{-\infty}^{c}{f(x)}dx + \int_{c}^{+\infty}{f(x)}dx. \text{ If both converge, the total integral converges.}\\
&\text{Common convergence p-test criterion: } \int_a^{+\infty}{\frac{1}{x^p}dx} \quad (a>0) \begin{cases}\text{Converges}, & p>1\\\text{Diverges}, & p\leq1\end{cases}\\
\end{align}
$$

### II. Improper Integrals of Unbounded Functions (Inproper Integrals with Discontinuities / Type II)

$$
\begin{align}
&\text{If } f(x) \text{ becomes unbounded near a point } a, \text{ then } a \text{ is called a singularity (or vertical asymptote).}\\
&(1) \text{ If } a \text{ is the singularity on } (a,b]: \quad \int_{a}^{b}f(x)dx=\lim_{t\rightarrow a^+}{\int_{t}^{b}{f(x)dx}}\\
&(2) \text{ If } b \text{ is the singularity on } [a,b): \quad \int_a^bf(x)dx=\lim_{t\rightarrow b^-}{\int_a^tf(x)dx}\\
&(3) \text{ If } c \in (a,b) \text{ is a singularity, split the integral at } c. \text{ Both pieces must converge for the whole to converge.}\\
&\text{Common convergence p-test criterion:}\\
&\int_a^b{\frac{1}{(x-a)^p}dx} \text{ or } \int_a^b{\frac{1}{(b-x)^p}dx} \begin{cases}\text{Converges}, & p<1\\\text{Diverges}, & p\geq 1\end{cases}\\
\end{align}
$$

### III. Examples

##### Example 1

![12edsadada](https://raw.githubusercontent.com/blueflylabor/images/main/12edsadada.jpg)
$$
\begin{align}
&\text{Analyze convergence by splitting at the singularity } x=1 \text{ and assessing infinity behavior.}\\
\end{align}
$$

## Applications of Definite Integrals

#### The Element Method (Differential Elements)

$$
\begin{align}
&dA = f(x)dx, \quad dV = \pi f^2(x)dx\\
\end{align}
$$



### I. Geometric Applications

#### 1. Area of a Plane Region

$$
\begin{align}
&(1) \text{ Cartesian Coordinates: } S=\int_a^b{[f(x)-g(x)]dx \quad \text{where } f(x)\geq g(x)}\\
&(2) \text{ Polar Coordinates: } S=\frac{1}{2}\int_{\alpha}^{\beta}{\rho^2(\theta)d\theta}\\
\end{align}
$$

#### 2. Volume of a Solid of Revolution

$$
\begin{align}
&\text{Let a region } D \text{ be bounded by } y=f(x) \geq 0, x=a, x=b, \text{ and the } x\text{-axis:}\\
&(1) \text{ Rotation about the } x\text{-axis: } V_x=\pi\int_a^b{f^2(x)dx}\\
&(2) \text{ Rotation about the } y\text{-axis (Shell Method): } V_y=2\pi\int_a^b{xf(x)dx}\\
\end{align}
$$

![img](https://raw.githubusercontent.com/blueflylabor/images/main/U1%7D97(ZE)HIN4FCVUKI$$%5DZB.jpg)
$$
\begin{align}
&\text{Volumes of revolution for regions bounded by lines like } y=x, y=x^2:\\
&V_x = \pi\int_0^1 (x^2 - (x^2)^2)dx\\
&V_y = 2\pi\int_0^1 x(x - x^2)dx\\
\end{align}
$$

#### 3. Arc Length of a Plane Curve

$$
\begin{align}
&(1) \text{ Explicit Form } y=y(x): \quad s=\int_a^b{\sqrt{1+y'^2}dx}\\
&(2) \text{ Parametric Form } \begin{cases}x=x(t)\\y=y(t)\end{cases}: \quad s=\int_{\alpha}^{\beta}{\sqrt{x'^2+y'^2}dt}\\
&(3) \text{ Polar Form } \rho=\rho(\theta): \quad s=\int_{\alpha}^{\beta}{\sqrt{\rho^2+\rho'^2}d\theta}\\
\end{align}
$$

#### 4. Lateral Surface Area of a Solid of Revolution

$$
\begin{align}
&\text{Surface area generated by revolving curve } y=f(x) \text{ around the } x\text{-axis:}\\
&S=2\pi\int_a^b{f(x)\sqrt{1+f'^2(x)}dx}\\
\end{align}
$$

### II. Physical Applications

#### 1. Hydrostatic Pressure

#### 2. Work Done by a Variable Force

#### 3. Gravitational Attraction



#### Example 1
![img](https://raw.githubusercontent.com/blueflylabor/images/main/X2PPU~%@L@NM4Y}W6GZTT_R.jpg)

$$
\begin{align}
&\text{Analysis: The profile indicates cross-sectional slices can be integrated with respect to } y.\\
&V = \pi \int_{-1}^{1/2} (1-y^2) dy + \text{upper contribution. By symmetry, the problem can be set up carefully:}\\
\end{align}
$$

![屏幕截图 2021-04-19 203327](https://raw.githubusercontent.com/blueflylabor/images/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202021-04-19%20203327.jpg)
$$
\begin{align}
&\text{Work elements: } dW = \rho g \cdot A(y) \cdot (\text{distance}) dy\\
\end{align}
$$
![屏幕截图 2021-04-19 204534](https://raw.githubusercontent.com/blueflylabor/images/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202021-04-19%20204534.jpg)

#### Example 2

![image-20210419211039410](https://raw.githubusercontent.com/blueflylabor/images/main/image-20210419211039410.jpg)
$$
\begin{align}
&F = \int \rho g h \cdot b(y) dy\\
\end{align}
$$


![1618837868(1)](https://raw.githubusercontent.com/blueflylabor/images/main/1618837868(1).jpg)