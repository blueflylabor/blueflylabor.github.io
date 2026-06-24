---
title: Data Structures: Search Algorithms
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Data Structures]
lang: en
permalink: /en/:title/
---

# Search
## 1. Sequential Search
### General Linear List
(1) Code
```c++
typedef struct{
    ElemType *elem;
    int tableLen;
}SSTable;

int searchSeq(SSTable ST, ElemType key){
    ST.elem[0] = key;   // Set sentinel
    for(int i = ST.tableLen; ST.elem[i] != key; i--)
        return i;   // Returns index if found, returns 0 if not found
}

```

(2) Setting a Sentinel: Eliminates the need to check for array boundary overflows during each iteration. Note that the data index starts at 1.

(3) ASL (Average Search Length)

$$\text{If search probabilities are unknown, records can be sorted by their access probability from smallest to largest.}\\
ASL_{\text{success}} = \sum_{i=1}^{n} P_i(n-i+1) = \frac{n+1}{2}\\
ASL_{\text{unsuccess}} = n+1\\$$

(4) Advantages & Disadvantages

* **Advantages:** No structural requirements for data storage; it works for both sequential (array-based) allocation and linked storage.
* **Disadvantages:** When $n$ is large, the average search length becomes massive, resulting in low performance efficiency.

### Ordered List

```mermaid
graph LR
id1((10))--id2((20))
id1((10)).--infinity,10
id2((20))--id3((30))
id2((20)).--infinity,20
id3((30))--id4((40))
id3((30)).--infinity,30
id4((40))--id5((50))
id4((40)).--infinity,40
id5((50))--id6((60))
id5((50)).--infinity,50
id6((60))--60,=
id6((60)).--infinity,60

```

(1) The execution process terminates immediately once it encounters an element greater than the target search key.
(2) The rectangular boxes represent mock fail-states (external nodes). The exact search length matches the number of circular nodes traversed to reach them.
(3) ASL (Average Search Length)

$$ASL_{\text{success}} = \sum_{i=1}^{n} P_i(n-i+1) = \frac{n+1}{2}\\
ASL_{\text{unsuccess}} = \sum_{j=1}^{n+1} Q_j(l_j-1) =  \frac{1+2+...+n+n}{n+1} = \frac{n}{2} +  \frac{n}{n+1}\\$$

### Binary Search

```mermaid
graph LR
id29((29))--id37((37))--id41((41))--id43((43))
id43((43))--43,+infinity
id43((43))--37,43
id37((37))--id32((32))--id33((33))
id32((32))--29,32
id33((33))--33,37
id33((33))--32,33
id13((13))--id16((16))--id19((19))--19,29
id19((19))--16,19
id29((29))--id13((13))--id7((7))--id10((10))--10,13
id10((10))--7,10
id7((7))---infinity,7

```

(1) Strictly applicable only to sequential lists (arrays).
(2) Code

```c++
int binarySearch(SeqList L, ElemType key){
    int low = 0, high = L.tableLen - 1, mid = 0;
    while(low <= high){
        mid = (low + high) / 2;
        if(L.elem[mid] == key)
            return mid;
        else if(L.elem[mid] > key)
            high = mid - 1;
        else
            low = mid + 1;
    }
    return -1;
}

```

(3) ASL (Average Search Length)

$$ASL = \frac{1}{n}\sum_{i=1}^{n} l_i = \frac{1}{n}(1*1+2*2+...+h*2^{h-1}) = \frac{n+1}{n} \log_2(n+1)-1 \approx \log_2(n+1)-1\\
h=\lceil\log_2(n+1)\rceil \text{ (Rounded up)}$$

#### Trace Example: Searching for 11

##### low=7, high=43, mid=29

##### 11 < 29

```mermaid
graph 
7--low
10
13
16
19
29--mid
32
33
37
41
43--high


```

##### low=7, high=mid-1=19, mid=13

##### 11 < 13

```mermaid
graph 
7--low
10
13--mid
16
19--high
29
32
33
37
41
43


```

##### low=7, high=mid-1=10, mid=7

##### 11 > 7

```mermaid
graph 
7--low--mid
10--high
13
16
19
29
32
33
37
41
43


```

##### low=mid+1=10, high=10, mid=10

##### 11 != 10 (Not Found)

```mermaid
graph 
7
10--low--mid--high
13
16
19
29
32
33
37
41
43


```

##### Search ends with failure, tracking pointer stops at index 'low'.

### Block Search (Index Sequential Search)

(1) Divides the search table into a series of distinct sub-blocks. Items inside a block can be completely unordered, but the blocks themselves must remain strictly ordered.

```mermaid
graph 
id24((24))--id((Index Block 24,54,78,88))
id21((21))
id6((6))
id11((11))
id8((8))
id22((22))
id32((32))--id((Index Block 24,54,78,88))
id31((31))
id54((54))
id72((72))--id((Index Block 24,54,78,88))
id61((61))
id78((78))
id88((88))--id((Index Block 24,54,78,88))
id83((83))


```

(2) ASL (Average Search Length)

$$n: \text{Total length}\\
b: \text{Number of allocated blocks}\\
s: \text{Number of records per block}\\
P: \text{Equal probability distribution}\\
ASL = L_I+L_S = \frac{b+1}{2}+\frac{s+1}{2}=\frac{s^2+2s+n}{2s}\\
\text{When } s=\sqrt{n}, \text{ optimal } ASL=\sqrt{n}+1\\
\text{Using Binary Search for index blocks: } ASL=L_I+L_S=\lceil\log_2(b+1)\rceil+\frac{s+1}{2} \text{ (Rounded up)}$$

### B-Tree (Balanced Multi-Way Search Tree)

$$\text{An } m\text{-order B-Tree is either an empty tree or satisfies:}\\
\text{Each internal node has at most } m \text{ subtrees and contains at most } m-1 \text{ keys.}\\
\text{If the root node is not a leaf node, it must have at least 2 subtrees.}\\
\text{All non-leaf internal nodes except the root must contain at least } \lceil\frac{m}{2}\rceil \text{ subtrees (keys).}$$

```mermaid
graph 
id[22]--id0[5,11]
id[22]--id1[36,45]
id0[5,11]--id00[1,3]
id0[5,11]--id01[6,8,9]
id0[5,11]--id02[13,15]
id1[36,45]--id10[30,35]
id1[36,45]--id11[40,42]
id1[36,45]--id12[47,48,50,56]

```

```