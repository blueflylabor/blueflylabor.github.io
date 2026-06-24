
---
title: Frog Jump Problem: Dynamic Programming Optimization
date:   2021-11-20
last_modified_at: 2026-06-23
categories: algorithms
tags: [Algorithm, Dynamic Programming, Python]
lang: en

---

# Problem Statement

A frog can jump up $1$ step, $2$ steps, ..., up to $m$ steps at a time. Write a function to calculate the total number of unique ways the frog can jump to the top of an $n$-level staircase.

* **Input:** $n$ (total steps), $m$ (maximum steps per jump)
* **Output:** Total number of unique jumping methods.

---

## 1. Mathematical Analysis & State Equations

Let $f(n)$ be the total number of unique ways to reach the $n$-th step. 

The frog's last jump to reach step $n$ could have been from step $n-1$, step $n-2$, ..., or step $n-m$ (provided these step numbers are valid, i.e., greater than or equal to $0$). Therefore, the problem satisfies the following transition relations:

* When $n \le m$: The frog can reach step $n$ from any step from $0$ to $n-1$.
  $$f(n) = f(n-1) + f(n-2) + \dots + f(0)$$
* When $n > m$: The frog can only jump from at most $m$ steps back.
  $$f(n) = f(n-1) + f(n-2) + \dots + f(n-m)$$

**Base Case:** $f(0) = 1$ (There is exactly $1$ way to stay at the ground level: doing nothing).

---

## 2. Dynamic Programming Implementation (O(n * m))

Instead of utilizing deep recursion with backtracking which results in massive redundant calculations ($O(m^n)$), we can use an array `dp` of size $n+1$ to store calculated states.

```python
def frog_jump_dp(n: int, m: int) -> int:
    if n < 0:
        return 0
    if n == 0 or n == 1:
        return 1
        
    # Initialize DP table, dp[i] stores the ways to reach step i
    dp = [0] * (n + 1)
    dp[0] = 1 # Base case
    
    for i in range(1, n + 1):
        # The frog can come from any of the previous min(i, m) steps
        for j in range(1, min(i, m) + 1):
            dp[i] += dp[i - j]
            
    return dp[n]

# Example execution
if __name__ == '__main__':
    n = int(input("Enter the total number of steps (n): "))
    m = int(input("Enter the maximum jump height (m): "))
    print(f"Total unique jumping methods: {frog_jump_dp(n, m)}")

```

---

## 3. Mathematical Optimization to O(n)

We can optimize the transition formula further to remove the inner loop entirely.

When $n > m$:


$$\begin{aligned}
f(n) &= f(n-1) + f(n-2) + \dots + f(n-m) \\
f(n-1) &= f(n-2) + f(n-3) + \dots + f(n-1-m)
\end{aligned}$$

By substituting the second equation into the first, we get:


$$f(n) = 2 \times f(n-1) - f(n-1-m)$$

When $n \le m$, it simplifies directly to the classic generalized variant:


$$f(n) = 2 \times f(n-1)$$

Using this sliding window logic, the time complexity drops cleanly to **$O(n)$**.

```python
def frog_jump_optimized(n: int, m: int) -> int:
    if n < 0: return 0
    if n == 0 or n == 1: return 1
    
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        if i <= m:
            dp[i] = 2 * dp[i - 1]
        else:
            dp[i] = 2 * dp[i - 1] - dp[i - 1 - m]
            
    return dp[n]

```

---

## Summary of Complexities

| Implementation | Time Complexity | Space Complexity | Notes |
| --- | --- | --- | --- |
| **Backtracking (Original)** | $O(m^n)$ | $O(n)$ | Redundant computations cause Stack Overflow for large $n$. |
| **Basic DP** | $O(n \times m)$ | $O(n)$ | Standard approach, robust and clear logic. |
| **Sliding Window DP** | $O(n)$ | $O(n)$ | Optimal approach via algebraic substitution. |

```