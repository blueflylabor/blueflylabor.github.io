---
title: Advanced Mathematics: Differential Equations
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
lang: en

---

# Differential Equations

### I. Basic Concepts of Ordinary Differential Equations

##### 1. Differential Equation

$$
y'=2x
$$

An equation containing the derivatives or differentials of an **unknown function**.

##### 2. Order of a Differential Equation

$$
\text{1st-order equation}
$$

The **highest order of the derivative of the unknown function** that appears in the differential equation.

##### 3. Solution of a Differential Equation

$$
y=f(x)=x^2
$$

A **function that satisfies the differential equation**.

##### 4. General Solution of a Differential Equation

$$
y=f(x)=x^2+c
$$

A solution of a differential equation that contains **arbitrary constants**, where the **number of arbitrary constants** matches the order of the differential equation.

##### 5. Particular Solution of a Differential Equation

$$
y=f(x)=x^2+1
$$

A solution of a differential equation that **does not contain arbitrary constants**.

##### 6. Initial Conditions

A set of constants used to determine a particular solution.

##### 7. Integral Curve

The geometric curve on a plane that corresponds to a single solution of the differential equation.

### II. First-Order Differential Equations $y'=f(x,y)$

##### 1. Separable Variables Equations

$$
\begin{align}
&y'=f(x)g(y)\Leftrightarrow \frac{dy}{dx}=f(x)g(y)\Leftrightarrow \frac{dy}{g(y)}=f(x)dx \\
&\text{Solution method: integrate both sides } \Leftrightarrow \int\frac{dy}{g(y)}=\int f(x)dx\\
\end{align}
$$

##### 2. Homogeneous Differential Equations

$$
\begin{align}
&\frac{dy}{dx}=\Phi\left(\frac{y}{x}\right)\\
&\text{Method: Let } u=\frac{y}{x}, \; y=ux, \; y'=u+u'x\\
\end{align}
$$

###### Example

![1321sdads](https://raw.githubusercontent.com/blueflylabor/images/main/1321sdads.jpg)
$$
\begin{align}
&y'+\frac{y}{x}=\left(\frac{y}{x}\right)^2\\
&\text{Let } u=\frac{y}{x}, \; y=ux, \; y'=u+u'x\\
&\Leftrightarrow u+u'x+u=u^2\\
&\Leftrightarrow u+\frac{du}{dx}x+u=u^2\\
&\Leftrightarrow \frac{du}{u^2-2u}=\frac{1}{x}dx\\
&\Leftrightarrow \int\frac{du}{u^2-2u}=\int\frac{1}{x}dx\\
&\Leftrightarrow \frac{1}{2}(\ln|u-2|-\ln|u|)=\ln|x|+C\\
&\Leftrightarrow \frac{u-2}{u}=Cx^2\\
\end{align}
$$

##### 3. First-Order Linear Differential Equations

$$
\begin{align}
&\text{Form: } y'+p(x)y=Q(x)\\
&\text{General Solution: } y=e^{-\int p(x)dx}\left[\int Q(x)e^{\int p(x)dx}dx+C\right]\\
\end{align}
$$

##### 4. Bernoulli Equations

$$
\begin{align}
&\text{Form: } y'+p(x)y=Q(x)y^n \quad (n \neq 0,1)\\
&\text{Method: Let } u=y^{1-n}\\
&y^{-n}y'+p(x)y^{1-n}=Q(x)\\
&\text{Let } u=y^{1-n}\\
&(1-n)y^{-n}y'=\frac{du}{dx}\\
\end{align}
$$

![img](https://raw.githubusercontent.com/blueflylabor/images/main/0P50G64~_SZYBCKV~_$$WHJG.jpg)
$$
\begin{align}
&\text{Observing that the differential equation contains } y'', \; y', \text{ and } y, \text{ it belongs to the form } y''=f(y,y').\\
&\text{Let } y'=p, \; y''=p\frac{dp}{dy}.\\
&\text{Yielding: } yp\frac{dp}{dy}+p^2=0\\
&\Leftrightarrow p\left(y\frac{dp}{dy}+p\right)=0\\
&\text{When } p=0, \text{ the equation holds, but according to the initial condition } y'|_{x=0}=1,\\
&\text{the solution } p=0 \text{ contradicts it.}\\
&\Leftrightarrow \int\frac{dp}{p}=-\int\frac{dy}{y}\\
&\Leftrightarrow |py|=e^C\\
&\Leftrightarrow p=\frac{C}{y}\\
&\Leftrightarrow \frac{dy}{dx}=\frac{C}{y}\\
&\text{According to the initial condition } y'|_{x=0}=\frac{1}{2} \Leftrightarrow C=\frac{1}{2}.\\
&\Leftrightarrow y^2=x+C_2\\
&\text{According to the initial condition } y|_{x=0}=1 \Rightarrow 1^2=0+C_2 \Rightarrow C_2=1 \Rightarrow y^2=x+1.\\
&\text{Since } y|_{x=0}=1 > 0, \text{ we have } y=\sqrt{x+1}.
\end{align}
$$

##### 5. Exact Differential Equations

![sdaghyrsa1](https://raw.githubusercontent.com/blueflylabor/images/main/dojTOJ9EAYhkmSl.jpg)

### III. Higher-Order Equations Reducible in Order

![asfgjkh](https://raw.githubusercontent.com/blueflylabor/images/main/asfgjkh.jpg)
$$
\begin{align}
&y'=p=\frac{dy}{dx}, \; y''=\frac{dp}{dx}\\
&y''=f(x,y') \Leftrightarrow \frac{dp}{dx}=f(x,p)
\end{align}
$$


![asdvvxzz](https://raw.githubusercontent.com/blueflylabor/images/main/asdvvxzz.jpg)
$$
\begin{align}
&x\frac{dp}{dx}+3p=0\\
&\int{\frac{1}{p}dp}=-3\int{\frac{1}{x}dx} \quad (\text{Note: standard substitution step})\\
&p=\frac{C_1}{x^3}\\
&\frac{dy}{dx}=\frac{C_1}{x^3}\\
&y=\frac{C_2}{x^2}+C_1\\
\end{align}
$$
![sadsasdc cxzczcxzc](https://raw.githubusercontent.com/blueflylabor/images/main/sadsasdc%20cxzczcxzc.jpg)
$$
\begin{align}
&y'=p=\frac{dy}{dx}, \; y''=\frac{dp}{dx}\\
&\text{Yielding } \frac{dp}{dy}=f(y,p) \text{ when handled via substitution.}\\
&\text{Let } y'=p=\frac{dy}{dx}, \; y''=\frac{dp}{dx}=\frac{dp}{dy}\frac{dy}{dx}=\frac{dp}{dy}p\\
&\text{Yielding: } \frac{dp}{dy}p=f(y,p)\\
\end{align}
$$


![feeasdasdsa](https://raw.githubusercontent.com/blueflylabor/images/main/feeasdasdsa.jpg)
$$
\begin{align}
&y\frac{dp}{dy}p+p^2=0\\
&\int\frac{dp}{p}=-\int\frac{dy}{y}\\
&py=C_1\\
&p=\frac{C_1}{y}\\
&\frac{dy}{dx}=\frac{C_1}{y}\\
&\text{Since } y'|_{x=0}=\frac{1}{2}\\
&\frac{dy}{dx}=\frac{\frac{1}{2}}{y}\\
&y^2=x+C_2\\
&\text{Since } y|_{x=0}=1\\
&y=\sqrt{x+1}\\
\end{align}
$$

### IV. Higher-Order Linear Differential Equations

![1621739485(1)](https://raw.githubusercontent.com/blueflylabor/images/main//1621739485(1).jpg)

![1621739593(1)](https://raw.githubusercontent.com/blueflylabor/images/main//1621739593(1).jpg)
$$
\begin{align}
&y''+py'+qy=0 \Leftrightarrow r^2+pr+q=0\\
&\text{Finding conjugate complex roots: } r_{1},r_2=\frac{-b\pm i\sqrt{4ac-b^2}}{2a} \quad (i^2=-1)\\
&\text{Example: If } y=xe^x \text{ is a solution to } y''+py'+qy=0,\\
&\text{then } (r-1)^2=0 \text{ must be the characteristic equation.}\\
&r^2-2r+1=0\\
&\text{Hence, } p=-2, \; q=1.\\
\end{align}
$$
![1621740803(1)](https://raw.githubusercontent.com/blueflylabor/images/main//1621740803(1).jpg)
$$
\begin{align}
&r^2-r+\frac{1}{4}=0\\
&\left(r-\frac{1}{2}\right)^2=0\\
&r_1=r_2=\frac{1}{2}\\
&y=e^{\frac{1}{2}x}(C_1+C_2x)\\
\end{align}
$$
![1621740979(1)](https://raw.githubusercontent.com/blueflylabor/images/main//1621740979(1).jpg)
$$
\begin{align}
&r^2+2r+5=0\\
&r_{1,2}=\frac{-2\pm i\sqrt{4-20}}{2}=-1\pm 2i\\
&y=e^{-x}(C_1\cos{2x}+C_2\sin{2x})
\end{align}
$$

![1621755536(1)](https://raw.githubusercontent.com/blueflylabor/images/main/1621755536(1).jpg)
$$
\begin{align}
&r^3-2r^2+r-2=0\\
&r^2(r-2)+(r-2)=0\\
&(r-2)(r^2+1)=0\\
&r_1=2, \; r_{2,3}=\pm i\\
&y=C_1e^{2x}+C_2\cos{x}+C_3\sin{x}\\
\end{align}
$$
![1621756093](https://raw.githubusercontent.com/blueflylabor/images/main/1621756093.jpg)
$$
\begin{align}
&\text{Core Solution Strategy: Find the particular solution of the homogeneous equation.}\\
\end{align}
$$
![1621764390(1)](https://raw.githubusercontent.com/blueflylabor/images/main/1621764390(1).jpg)
$$
\begin{align}
&D=\frac{d}{dt}\\
&\text{Let } x=e^t \text{ or } t=\ln x, \; t'=\frac{dt}{dx}=\frac{1}{x}\\
&y'=\frac{dy}{dx}=\frac{dy}{dt}\frac{dt}{dx}=\frac{1}{x}\frac{dy}{dt}\\
&xy'=\frac{dy}{dt}=Dy\\
&y''=\left(\frac{1}{x}\frac{dy}{dt}\right)'=\left(\frac{dy}{dt}\right)'\frac{1}{x}-\frac{1}{x^2}\frac{dy}{dt}=\frac{d^2y}{dt^2}\frac{dt}{dx}\frac{1}{x}-\frac{1}{x^2}\frac{dy}{dt}=\frac{d^2y}{dt^2}\frac{1}{x^2}-\frac{1}{x^2}\frac{dy}{dt} \Rightarrow x^2y''=D(D-1)y\\
\end{align}
$$

$$
\begin{align}
&D(D-1)y+4Dy+2y=0\\
&r(r-1)+4r+2=0\\
&r^2+3r+2=0\\
&(r+1)(r+2)=0\\
&y=C_1e^{-t}+C_2e^{-2t}\\
&y=\frac{C_1}{x}+\frac{C_2}{x^2}\\
\end{align}
$$

### Examples

##### 1.

$$
\begin{align}
&\text{If the general solution of the second-order constant coefficient linear homogeneous differential equation }\\
&y''+ay'+by=0 \text{ is given by } y=(C_1+C_2x)e^x,\\
&\text{find the general solution of the non-homogeneous equation } y''+ay'+by=x \text{ satisfying } y(0)=2, \; y'(0)=0.\\
&\\
&\text{Solution:}\\
&\text{From the general solution of the linear homogeneous equation, we know that } r=1 \text{ is a double root of the characteristic equation.}\\
&\text{Thus, the characteristic equation is } (r-1)^2=0 \Rightarrow r^2-2r+1=0.\\
&\text{This gives } a=-2, \; b=1.\\
&\text{The non-homogeneous equation becomes } y''-2y'+y=x, \text{ where the forcing term is } x=e^{0x}x. \text{ The particular solution takes the form } y^*=a'x+b'.\\
&\text{Substituting this into the non-homogeneous equation: } 0-2a'+(a'x+b')=x \Rightarrow a'x + (b'-2a') = x.\\
&\text{Equating coefficients yields: } a'=1, \; b'=2.\\
&\text{Thus, the general solution of the non-homogeneous equation is } y=(C_1+C_2x)e^x+x+2.\\
&\text{Applying the initial conditions } y(0)=2 \text{ and } y'(0)=0:\\
&y(0) = C_1 + 2 = 2 \Rightarrow C_1 = 0.\\
&y'(x) = C_2e^x + C_2xe^x + 1 \Rightarrow y'(0) = C_2 + 1 = 0 \Rightarrow C_2 = -1.\\
&\text{Therefore, the final particular solution is } y=-xe^x+x+2 = x(1-e^x)+2.\\
\end{align}
$$

##### 2.

$$
\begin{align}
&\text{Given that } y=\frac{1}{2}e^{2x}+\left(x-\frac{1}{3}\right)e^x \text{ is a particular solution to the second-order constant coefficient non-homogeneous}\\
&\text{linear differential equation } y''+ay'+by=ce^x, \text{ determine the values of } a, \; b, \text{ and } c.\\
&\\
&\text{Solution:}\\
&y=\frac{1}{2}e^{2x}+xe^x-\frac{1}{3}e^x \text{ matches the linear structure } y=C_1y_1+y^* \text{ or similar compositions.}\\
&\text{We can observe that } r=2 \text{ and } r=1 \text{ correspond to characteristic roots from the homogeneous part structure.}\\
&\text{The characteristic equation is therefore } (r-1)(r-2)=0 \Rightarrow r^2-3r+2=0.\\
&\text{Hence, } a=-3, \; b=2.\\
&\text{The equation is written as } y''-3y'+2y=ce^x. \text{ Substituting the non-homogeneous part term } y^*=xe^x \text{ into it:}\\
&\begin{cases}y^*{}'=e^x+xe^x\\y^*{}''=2e^x+xe^x\\\end{cases}\\
&\text{Substituting these derivatives back into the differential equation:}\\
&(2e^x+xe^x)-3(e^x+xe^x)+2(xe^x)=ce^x\\
&-e^x=ce^x \Rightarrow c=-1.\\
&\text{Thus, } a=-3, \; b=2, \; c=-1.\\
\end{align}
$$