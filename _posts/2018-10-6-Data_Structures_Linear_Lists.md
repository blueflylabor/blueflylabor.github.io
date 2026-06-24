
---
title: Data Structures: Linear Lists
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Data Structures]
lang: en
permalink: /en/:title/
---

# Linear Lists
## I. Logical Structure and Basic Operations
### 1. Logical Structure
- A finite sequence consisting of $n$ data elements of the same data type, where $n$ is the length of the list. When $n=0$, it is an empty list.
- **Head:** The first element of the list.
- **Tail:** The last element of the list.
- Except for the first element, every element has exactly one unique **direct predecessor**.
- Except for the last element, every element has exactly one unique **direct successor**.
### 2. Basic Operations
```c++
initList(&L);          // Initialize an empty list
len(L);                // Return the length of the list
locateElem(L, e);      // Find the position of an element by its value
getElem(L, i);         // Retrieve the element at a specific index
listInsert(&L, i, e);  // Insert a new element at position i
listDelete(&L, i, &e); // Delete the element at position i and return its value
printList(L);          // Print all elements in the list
isEmptyList(L);        // Check if the list is empty
destroyList(&L);       // Destroy the list and free all allocated memory

```

## II. Sequential Storage Structure

### 1. Definition: Also known as a sequential list (SqList), it uses a set of contiguous memory addresses to store the data elements of a linear list in sequence, ensuring that elements that are logically adjacent are also adjacent in physical memory.

* Starting position of the storage space: `data[]`
* Maximum capacity of the sequential list: `MaxSize`
* Current length of the sequential list: `len`

**Characteristics**

* Random access: Elements can be accessed directly in $O(1)$ time complexity.
* High storage density: Each node only stores the data element itself without additional structural overhead.
* No extra space is required to establish logical relationships between data, as relationships are implicitly defined by their physical adjacency.
* Since elements are adjacent both logically and physically, insertion and deletion operations require moving a large number of elements.

### 2. Basic Operations

#### (1) Insert Element

```c++
// Insert an element
bool listInsert(SqList &L, int i, Elemtype e){
    if(i < 1 || i > L.len + 1)
        return false;
    if(L.len >= MaxSize)
        return false;
    for(int j = L.len; j >= i; j--)
        L.data[j] = L.data[j - 1];
    L.data[i - 1] = e;
    L.len++;
    return true;
}

```

**Analysis**

* **Best-Case Scenario:** Inserting at the tail of the list ($i = n + 1$). No elements need to be shifted. Time complexity is $O(1)$.
* **Worst-Case Scenario:** Inserting at the head of the list ($i = 1$). All existing $n$ elements must be shifted backward. Time complexity is $O(n)$.
* **Average-Case Scenario:** Assuming $P_i$ ($P_i = \frac{1}{n+1}$) represents the probability of inserting a node at the $i$-th position, the average number of shifts required to insert a node in a linear list of length $n$ is $\frac{n}{2}$, giving a time complexity of $O(n)$.

#### (2) Delete Element

```c++
// Delete an element
bool listDelete(SqList &L, int i, Elemtype &e){
    if(i < 1 || i > L.len)
        return false;
    e = L.data[i - 1];
    for(int j = i; j < L.len; j++)
        L.data[j - 1] = L.data[j];
    L.len--;
    return true;
}

```

**Analysis**

* **Best-Case Scenario:** Deleting the element at the tail ($i = n$). No elements need to be shifted. Time complexity is $O(1)$.
* **Worst-Case Scenario:** Deleting the element at the head ($i = 1$). The remaining $n-1$ elements must be shifted forward. Time complexity is $O(n)$.
* **Average-Case Scenario:** Assuming $P_i$ ($P_i = \frac{1}{n}$) represents the probability of deleting a node at the $i$-th position, the average number of shifts required to delete a node in a linear list of length $n$ is $\frac{n-1}{2}$, giving a time complexity of $O(n)$.

#### (3) Locate Element (Search)

```c++
int locateElem(SqList L, Elemtype e){
    int i;
    for(i = 0; i < L.len; i++)
        if(e == L.data[i])
            return i + 1; // Return 1-based position
    return 0; // Return 0 if not found
}

```

**Analysis**

* **Best-Case Scenario:** The target element is at the head of the list. Only 1 comparison is needed. Time complexity is $O(1)$.
* **Worst-Case Scenario:** The target element is at the tail or does not exist. It requires $n$ comparisons. Time complexity is $O(n)$.
* **Average-Case Scenario:** Assuming $P_i$ ($P_i = \frac{1}{n}$) represents the probability of the target element being at the $i$-th position, the average number of comparisons required in a linear list of length $n$ is $\frac{n+1}{2}$, giving a time complexity of $O(n)$.

---

# Linked Storage Structure

## 1. Creating a Singly Linked List

### (1) Head Insertion Method (Insert at Head)

* Allocate memory space for the new node.
* Insert the new node immediately after the head node (at the front of the list).
* The order of nodes in the resulting linked list is the reverse of the input sequence.

### (2) Tail Insertion Method (Insert at Tail)

* Allocate memory space for the new node.
* Append the new node after the current tail node, then update the tail pointer.
* The order of nodes in the resulting linked list matches the input sequence.

## 2. Search Node by Value

* Starting from the first data node, traverse the list sequentially by following the `next` pointer until a node matching the target value is found. If no such node exists, return the terminal pointer `NULL`.

## 3. Search Node by Position Index

* Starting from the first node, traverse forward sequentially to track and match the given index position. Returns the pointer to the node if found; requires a check to ensure the index does not exceed valid boundaries.

## 4. Insertion

* The insertion operation adds a new node with value $x$ at the $i$-th position of the singly linked list.
* Check if the target position index is valid.
* Locate the predecessor node at position $i-1$.
* Insert the new node immediately after it.

```c++
p = getElem(L, i - 1); // Locate predecessor node
s->next = p->next;
p->next = s;

```

## 5. Deletion

* Removals target and delete the node at the $i$-th position of the singly linked list.
* Check if the target position index is valid.
* Locate the predecessor node at position $i-1$.
* Unlink and free the node that follows it.

```c++
p = getElem(L, i - 1); // Locate predecessor node
q = p->next;           // q points to the node to be deleted
p->next = q->next;     // Unlink node q
free(q);               // Deallocate memory space

```

---

# Doubly Linked Lists

* Each node in a doubly linked list contains two pointers, `prior` and `next`, pointing to its direct predecessor and direct successor nodes, respectively.
* **Insertion Operation:**

```c++
s->next = p->next;
if (p->next != NULL) {
    p->next->prior = s;
}
s->prior = p;
p->next = s;

```

* **Deletion Operation:**

```c++
q = p->next;
p->next = q->next;
if (q->next != NULL) {
    q->next->prior = p;
}
free(q);

```

---

# Circular Linked Lists

* Includes circular singly linked lists and circular doubly linked lists, where the terminal node points back to the head node to form a continuous loop.
* **Static Linked Lists:** Utilize a pre-allocated array structure to describe and implement the link-pointer behaviors of a linear list layout.

```

