---
title: "Advanced Mathematics: Partial Differentiability of Functions of Two Variables"
date:   2018-09-06
last_modified_at: 2018-09-06
categories: notes
tags: [Advanced Mathematics]
---

**1. Partial Differentiability of Functions of Two Variables**

In functions of two variables, the concept of differentiability for single-variable functions evolves into partial differentiability, and the concept of derivative functions evolves into partial derivative functions. See the example below for details:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/9d82d158ccbf6c818200af6499c37c3132fa400c.jpeg)

The partial derivative functions of a two-variable function $f(x,y)$ with respect to $x$ and $y$ are respectively:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/728da9773912b31b5d52d2cda3e5fb7edbb4e11e.jpeg)

When finding the partial derivative function of a two-variable function, we always assume the other variable is a constant, and then differentiate with respect to the remaining variable. For example, to find the partial derivative function of $f(x,y)$ with respect to $x$, we treat $y$ as a constant and then differentiate $f(x,y)$ with respect to the variable $x$.

At a specific point, both partial derivatives of the function $f(x, y)$ may exist, only one may exist, or neither may exist.

An example of a function where only one of the two partial derivatives exists at the point $(0, 0)$:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/2cf5e0fe9925bc31f7e7c860782240b5c9137096.jpeg)

An example of a function where neither of the two partial derivatives exists at the point $(0, 0)$:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/023b5bb5c9ea15ce6c2fde3c90fdf7f73887b2c7.jpeg)

An example of a function where both partial derivatives exist at the point $(0, 0)$:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/0824ab18972bd407221931d55d7453550eb30939.jpeg)

For the three examples above, it is highly recommended to calculate the partial derivatives by hand to deepen your understanding of partial derivatives of two-variable functions.

**2. Differentiability of Functions of Two Variables**

Differentiability at a specific point describes the linear relationship between the function increment and the independent variable increments. In a single-variable function, if the coefficient of the principal linear part depends only on that specific point, the function is differentiable. By extension, in a two-variable function, if the coefficients of the principal linear parts for multiple independent variables depend only on that specific point, the function is differentiable. The relationships between the function increment and the independent variable increments for single-variable and two-variable functions are listed below:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/d439b6003af33a877c62e124e1a1dd3c5243b587.jpeg)

For a specific point, when $A$ and $B$ are constants (meaning $A$ and $B$ are independent of the increments of the independent variables), the function is differentiable at that point, and $A$ and $B$ are the partial derivatives of the function evaluated at that point with respect to $x$ and $y$, respectively.

**3. Relationships Among Differentiability, Partial Differentiability, Continuity, and Continuity of Partial Derivatives**

To facilitate a comparison with single-variable functions, let us first look at the relationship diagram regarding differentiability, continuity, and continuity of the derivative function at a point $C$ for a single-variable function. In Figure 1, differentiability and differentiability (derivability) are equivalent for a single-variable function, hence the two-way arrow between them. Differentiability at point $C$ definitely implies continuity at point $C$, but continuity cannot guarantee differentiability at point $C$. Therefore, there is a one-way arrow from differentiability to continuity. If the derivative function is continuous at point $C$, it obviously implies that the function is differentiable and continuous at point $C$, but conversely, it cannot guarantee that the derivative function is continuous at point $C$.

![img](https://raw.githubusercontent.com/blueflylabor/images/main/7acb0a46f21fbe09fc1d4d2f4d9dc1378644ad48.jpeg)Figure 1. Relationship diagram of differentiability and continuity for single-variable functions

A friendly reminder: make sure to memorize the above diagram frequently, and memorize it with true understanding—for instance, if a single-variable function is differentiable, you must know what differentiability means and how to write out its defining formula!

Compared to single-variable functions, functions of two variables are much more complex. Let us first look at the relationship diagram among differentiability, partial differentiability, continuity, and continuity of partial derivative functions for multivariable functions.

![img](https://raw.githubusercontent.com/blueflylabor/images/main/3c6d55fbb2fb43165d025ae105598b2708f7d309.jpeg)Figure 2. Relationship diagram of differentiability and partial differentiability for multivariable functions

Naturally, when memorizing these relationships, we usually need to spend more time on those that are counter-intuitive, which are precisely the differences that emerge when compared to single-variable functions.

**3.1 Differentiability and Partial Differentiability are Not Equivalent**

Before explaining why differentiability and partial differentiability are not equivalent for two-variable functions, let's briefly review why differentiability and differentiability (derivability) are equivalent in single-variable functions.

In a single-variable function, if the function $f(x)$ is differentiable at $x=x_0$, the following relationship holds:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/242dd42a2834349b8a998272ec17d8ca34d3bea8.jpeg)

Suppose that in a single-variable function, the function increment and the independent variable increment satisfy the following relationship:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/6a600c338744ebf8019d6c0aff041a2e6159a71c.jpeg)

By dividing both sides of the above equation by $\Delta x$ and then taking the limit as $\Delta x \to 0$ on both sides, we find that $A=m$. According to the definition of single-variable differentiability, $A$ depends only on $x=x_0$ and is independent of $\Delta x$, so $f(x)$ is differentiable at $x=x_0$. Similarly, it is easy to prove that differentiability also implies differentiability (derivability) in a single-variable function.

So, how do we demonstrate that differentiability necessarily implies partial differentiability in a two-variable function?

Suppose a two-variable function is differentiable at point $C(x_0, y_0)$. By the definition of differentiability, there must exist a certain neighborhood of $(x_0, y_0)$ such that the following formula holds:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/203fb80e7bec54e7797010c09cc556544dc26a55.jpeg)

Let us set $\Delta y=0$ and $\Delta x=0$ separately. According to formula ①, we can obtain:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/f703738da977391283a65691dde44b1c347ae244.jpeg)

The reason we are allowed to set $\Delta x=0$ or $\Delta y=0$ is that the points $(x_0, y_0+\Delta y)$ and $(x_0+\Delta x, y_0)$ both lie within the differentiable neighborhood of the point $(x_0, y_0)$.

Taking the limits of the two expressions in ② yields:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/e850352ac65c103845d7a20797ec5e17b27e894f.jpeg)

Combining the definition of partial derivatives with the two limits in ③, we can see that under the condition of differentiability, both partial derivatives of the function at point $C$ must exist. Therefore, differentiability definitely implies partial differentiability.

Although differentiability guarantees partial differentiability, the converse is not true. Consider the following example:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/08f790529822720ef5a584c15d36c742f31faba0.jpeg)

Both partial derivatives of the function $F$ at $(0, 0)$ exist and equal $0$. Now we use proof by contradiction to demonstrate that the function $F$ is not differentiable at the point $(0, 0)$. Suppose the function $F$ is differentiable at the origin. According to the definition of differentiability, the following limit must exist. However, it is easy to verify that this limit does not exist by testing two different paths. Thus, the initial assumption is false, meaning that having partial derivatives does not guarantee differentiability.

![img](https://raw.githubusercontent.com/blueflylabor/images/main/c8177f3e6709c93d685e88a4b9c035d8d0005432.jpeg)

**3.2 Partial Differentiability Does Not Guarantee Continuity**

Another puzzling aspect of the relationship diagram for two-variable functions is that even if both partial derivatives exist at a certain point, the function is not necessarily continuous at that point. To illustrate this, examine the following function:

![img](https://raw.githubusercontent.com/blueflylabor/images/main/0b46f21fbe096b6339eceea129ce4a40e9f8ac48.jpeg)

It is clear that you can smoothly compute that both partial derivatives of the function $F$ with respect to $x$ and $y$ at the origin are $0$. However, when approaching the origin along the line $y=x$, the function value approaches $1$, which is not equal to $0$. Therefore, the function $F$ is discontinuous at the origin.

From an abstract perspective, the existence of both partial derivatives at a point only indicates that the function value approaches its defined value when moving strictly along the $x$-direction and the $y$-direction toward that point. It provides no guarantee that the function value will approach the same target when approaching from any other direction.