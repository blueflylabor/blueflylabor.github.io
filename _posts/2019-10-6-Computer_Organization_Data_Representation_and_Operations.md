---
title: "Computer Organization: Data Representation and Operations"
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Computer Organization and Architecture]
---

# Data Representation and Operations
## Base Systems (Radix)
- **Radix-$r$ System ($r$-base)** $$K_{n} K_{n-1} K_{n-2} \dots K_{0} K_{-1} \dots K_{-m}$$    
  Numerical expansion value:
  $$K_{n} r^{n} + K_{n-1} r^{n-1} + \dots + K_{0} r^{0} + K_{-1} r^{-1} + \dots + K_{-m} r^{-m} = \sum_{i=n}^{-m} K_{i} r^{i}$$
- **Binary System** $$\text{Digits: } 0, 1 \quad (r=2)$$
- **Octal System** $$\text{Digits: } 0, 1, 2, 3, 4, 5, 6, 7 \quad (r=8=2^3)$$
- **Hexadecimal System** $$\text{Digits: } 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F \quad (r=16=2^4)$$

## Base Conversions
* A 4-bit binary sequence directly corresponds to a single 1-digit hexadecimal character.  
* A 3-bit binary sequence directly corresponds to a single 1-digit octal character.  

## Signed Number Representations (Sign-and-Magnitude, Ones' Complement, Two's Complement, Biased)

### 1. Sign-and-Magnitude (Original Code)
* **Definition:** The highest-order bit represents the sign ($0$ for positive numbers, $1$ for negative numbers), while the remaining bits represent the absolute magnitude of the numerical value.
* **Characteristics:** Has two distinct representations for zero ($+0$ and $-0$). It is intuitive for humans but complicates arithmetic hardware implementation.

### 2. Ones' Complement (Inverse Code)
* **Definition:** * For positive numbers: Identical to its sign-and-magnitude representation.
  * For negative numbers: The sign bit is maintained as $1$, while all magnitude bits are inverted ($0 \to 1$ and $1 \to 0$).
* **Characteristics:** Also retains twin representations for zero ($+0$ and $-0$). Historically used as an intermediate state to simplify subtraction into addition.

### 3. Two's Complement (Complementary Code)
* **Definition:** * For positive numbers: Identical to its sign-and-magnitude representation.
  * For negative numbers: Formed by taking its ones' complement and adding $1$ to the least significant bit (LSB).
* **Characteristics:** Features a unique representation for zero ($0000\dots00$). It unifies subtraction and addition into unified addition operations, and expands the representation range for negative values by one unit (e.g., matching $-128$ in an 8-bit integer system).

### 4. Biased Representation (Offset Binary / Excess Code)
* **Definition:** Shifting the true value by adding a fixed constant displacement bias value (typically $2^{n-1}$ or $2^{n-1}-1$, where $n$ represents the total tracking bit width count).
* **Characteristics:** Preserves natural lexicographical ordering for values, making comparisons straightforward. Commonly used to represent the exponent field (characteristic) in floating-point notations (such as IEEE 754 formats).