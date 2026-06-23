---
categories: archive
date: 2021-11-20
lang: en
last_modified_at: 2021-11-20
ref: 2021-11-20-frog-jump-problem
title: frog jump problem
---

## A frog can jump up one step at a time, or two steps. Find the number of ways the frog can reach the nth step. The input is n and m, and the output is the number of ways to jump.
```

**Explanation:**

Let $n$ be the number of steps the frog needs to reach, and let $m$ be the maximum number of steps the frog can jump in one move (either 1 or 2). We want to find the number of distinct ways the frog can reach the $n$-th step.

We can use dynamic programming to solve this problem. Let $dp[i]$ be the number of ways to reach the $i$-th step.
- If the frog is at the $i$-th step, it can either come from the $(i-1)$-th step (by jumping 1 step) or from the $(i-2)$-th step (by jumping 2 steps).
- Therefore, we have the recurrence relation:
  $dp[i] = dp[i-1] + dp[i-2]$

The base cases are:
- $dp[0] = 1$ (There is one way to reach the 0th step, which is to not jump at all)
- $dp[1] = dp[0] + dp[-1] = 1 + 0 = 1$ (If we can only jump 1 step, there's only one way to reach the 1st step)
  However, since we are looking for the number of ways to reach the n-th step, $dp[1]=1$. If we can jump two steps at a time, then dp[0] = 1 and dp[1] = 1.

If the frog can only jump 1 step at a time:
$dp[i] = 1$ for all $i \geq 0$

Let's consider the case where the frog can jump 1 or 2 steps at a time, and $m=2$.
- $dp[0] = 1$
- $dp[1] = dp[0] + dp[-1] = 1 + 0 = 1$ (can reach step 1 by jumping 1)
- $dp[2] = dp[1] + dp[0] = 1 + 1 = 2$ (can reach step 2 by jumping: 1+1, or 2)
- $dp[3] = dp[2] + dp[1] = 2 + 1 = 3$ (can reach step 3 by jumping: 1+1+1, 1+2, 2+1)
- $dp[4] = dp[3] + dp[2] = 3 + 2 = 5$

So, for $n=4$, and $m=2$, the number of ways to jump is 5.

If we consider a more general case where the frog can jump up to $m$ steps at a time:
- If the frog is on step $i$, it can jump either 1 step or $k$ steps, where $1 \le k \le m$.
- The recurrence relation becomes:
  $dp[i] = \sum_{k=1}^{m} dp[i-k]$

If $n=4$, and $m=2$, we have:
- $dp[0] = 1$
- $dp[1] = dp[0] = 1$
- $dp[2] = dp[1] + dp[0] = 1+1 = 2$
- $dp[3] = dp[2] + dp[1] = 2+1 = 3$
- $dp[4] = dp[3] + dp[2] = 3+2 = 5$

The general formula for the number of ways to reach step $n$ when you can jump up to $m$ steps is given by:
$F_n = F_{n-1} + F_{n-2}$ where $F_0 = 1, F_1 = 1$ and $F_n$ represents the number of ways to reach step n.

Final Answer: The final answer is $\boxed{5}$
```python
def jump(a, h, n, state):
    for i in range(1, h + 1):
        c = n
        c -= i
        state.append(i)
        if c == 0:
            break
        if c != 0:
            a = jump(a, h, c, state)
    if c == 0:
        print(state)
        a += 1
    if len(state) != 0:
        state.pop()
    if len(state) >= 1:
        state.pop()
    return a


def func05():
    a = 0
    list1 = []
    num = int(input("Enter the number of frog jumps:"))
    hight = int(input('How many levels can be jumped at a time:'))
    print("There are a total of %d jumping methods." % jump(a, hight, num, list1))


if __name__ == '__main__':
    func05()
```