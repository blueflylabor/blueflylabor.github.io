---
title: "Brief Overview of Data Structures"
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Data Structures]
---

# Introduction to Data Structures
## I. Data Structure: A collection of data elements that share one or more specific relationships
- **Structure:** In any scenario, data elements do not exist in isolation; there are inherent relationships among them.  
- **Logical Structure**
- **Storage Structure (Physical Structure)**
- **Data Operations**
- Logical structure and storage structure are inextricably linked.
- The design of an algorithm depends on the logical structure, while its implementation relies heavily on the storage structure.

## II. Logical Structure: The logical relationships existing among data elements
- Independent of physical storage and detached from the computer hardware representation.
- Broadly categorized into linear structures and non-linear structures:
  - **Linear Structures:** Linear Lists, Stacks, Queues, Strings, Arrays, Generalized Lists.
  - **Non-Linear Structures:** Trees, Binary Trees, Directed Graphs, Undirected Graphs.

## III. Storage Structure (Physical Structure): The representation of a data structure within computer memory
- The digital representation of data elements.
- The digital representation of relational structures.
- Highly dependent on programming language frameworks.
- Divided into Sequential Storage, Linked Storage, Indexed Storage, and Hash Storage.

### 1. Sequential Storage: Memory addresses are physically contiguous; elements that are logically adjacent are also adjacent in physical memory
- **Pros:** Supports random access ($O(1)$ capacity); requires less memory overhead per individual element.
- **Cons:** Constrained to a single contiguous block of storage units; prone to generating substantial external fragmentation or allocation bottlenecks.

### 2. Linked Storage: Memory positions do not need to be physically contiguous; elements that are logically adjacent are not necessarily adjacent in memory, and connections are managed by explicitly storing reference pointers to neighboring addresses
- **Pros:** No external fragmentation issues; maximizes utilization of scattered storage slots.
- **Cons:** Limited exclusively to sequential access structures; introduces pointer storage overhead.

### 3. Indexed Storage: Manages elements via an auxiliary index table, acting essentially like a lookup directory.

### 4. Hash Storage (Scattered Storage): Computes the direct physical address of an element instantly using its key value via a mapping function.

## IV. Data Operations

## V. Five Core Characteristics of an Algorithm
- **Finiteness:** Must terminate completely after executing a finite number of steps.
- **Definiteness:** Each instruction must bear a clear, unambiguous meaning, ensuring identical inputs consistently produce identical outputs.
- **Feasibility:** Every operation must be basic enough to be executed a finite number of times using real, actionable programmatic actions to achieve deterministic results.
- **Input:** Possesses zero or more discrete external inputs.
- **Output:** Yields one or more distinct programmatic output results.

## VI. Algorithmic Complexity
### 1. Time Complexity: Measures how the theoretical execution time of an algorithm scales as the problem input size increases.
### 2. Space Complexity: Measures how the auxiliary memory workspace footprint of an algorithm scales as the problem input size increases.