---
title: frog jump problem
date:   2021-11-20
last_modified_at: 2021-11-20
categories: [算法, 解题]
---


## 一只青蛙一次可以跳上1级台阶，也可以跳上2级......它最多可以跳上m级。编写函数求该青蛙跳上一个n级的台阶共有多少种跳法，函数的输入为n，m，输出为跳法的种数
 
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