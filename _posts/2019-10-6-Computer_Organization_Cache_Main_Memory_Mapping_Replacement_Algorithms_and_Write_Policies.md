---
title: Computer Organization: Cache-Main Memory Mapping, Replacement Algorithms, and Write Policies
sdate:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Computer Organization and Architecture]
lang: en

---

# Cache-Main Memory Mapping, Replacement Algorithms, and Write Policies

## I. Address Mapping Schemes
The data stored within a Cache line is an exact replica of a specific block in main memory. **Address Mapping** is the foundational mechanism that bridges the main memory address space to the Cache address space, dictating exactly how data blocks from memory are allocated into Cache slots.

Because the number of Cache lines is significantly smaller than the number of blocks in main memory, only a fraction of the total memory capacity can occupy the Cache at any given time. Therefore, each Cache line must include a **Tag** field—acting as a unique structural block identifier—alongside a **Valid Bit** ($1 = \text{Valid}$, $0 = \text{Invalid}$) to confirm whether the cached data is current and usable. 

There are three primary address mapping strategies:
* Direct Mapping
* Fully Associative Mapping
* Set-Associative Mapping

---

### 1. Direct Mapping
In a direct-mapped scheme, each block in main memory is assigned to exactly one specific, predetermined Cache line. 
$$\text{Formula: } \text{Cache Line Index} = \text{Main Memory Block Number} \pmod{\text{Total Cache Lines}}$$

Assuming a Cache with $2^c$ lines and a main memory with $2^m$ blocks:
* Memory blocks $0, 2^c, 2^{c+1}, \dots$ map exclusively to Cache Line $0$.
* Memory blocks $1, 2^c+1, 2^{c+1}+1, \dots$ map exclusively to Cache Line $1$.

The lower $c$ bits of the memory block number explicitly define the target Cache line location. To distinguish which memory block is currently occupying that line, a Tag field of length $t = m - c$ bits is added to each line.



A logical address is parsed by the hardware into three distinct fields:
| Tag ($t$ bits) | Cache Line Index ($c$ bits) | Word/Block Offset ($b$ bits) |
| :---: | :---: | :---: |

#### Hardware Translation and Access Workflow:
1. The hardware extracts the middle $c$ bits (**Cache Line Index**) from the logical address to target the exact corresponding Cache line.
2. The **Tag** field of that selected Cache line is compared directly with the highest $t$ bits of the inbound address.
3. **Cache Hit:** If the tags match identically and the **Valid Bit** is $1$, a hit occurs. The CPU accesses the data immediately using the lower $b$ bits (**Block Offset**).
4. **Cache Miss:** If the tags mismatch or the **Valid Bit** is $0$, a miss occurs. The CPU fetches the entire missing block from main memory, routes it to the designated Cache line, overwrites any existing data, updates the line's Tag to match the upper $t$ bits, and sets the Valid Bit to $1$. Concurrently, the requested data is forwarded to the CPU.

---

### 2. Fully Associative Mapping
In a fully associative scheme, any block from main memory can be loaded into **any available line** within the Cache. This format offers maximum flexibility but requires specialized, expensive hardware to search all lines concurrently.



Because a memory block has no fixed index location, the entire memory block number must be used as the **Tag** field:
| Tag ($t$ bits) | Word/Block Offset ($b$ bits) |
| :---: | :---: |

#### Hardware Translation and Access Workflow:
1. When the CPU issues an address lookup, its upper $t$ bits are compared **in parallel across all Cache lines simultaneously** using specialized Content-Addressable Memory (CAM) circuitry.
2. If a matching tag is found and its Valid Bit is $1$, it is a cache hit, and the lower $b$ offset bits extract the target word.
3. If no matching tag is found, it is a cache miss. The block is fetched from main memory and placed into any empty Cache line. If the Cache is full, a **Replacement Algorithm** must be executed to select a victim block.

---

### 3. Set-Associative Mapping
Set-associative mapping is a hybrid approach that combines the simplicity of direct mapping with the flexibility of fully associative mapping. The Cache lines are divided into $Q$ equal-sized groups called **Sets**. 

A memory block first maps directly to a specific Set based on its address (direct mapping behavior). Once the Set is identified, the block can occupy *any* individual line within that designated Set (fully associative behavior).
* If a Set contains $r$ Cache lines, the architecture is referred to as an **$r$-way Set-Associative Cache**.
* If $Q = 1$, the layout collapses into a *Fully Associative Cache*.
* If $r = 1$, the layout becomes a *Direct-Mapped Cache*.

$$\text{Formula: } \text{Cache Set Index} = \text{Main Memory Block Number} \pmod{\text{Total Cache Sets } (Q)}$$



A logical address is parsed by the hardware into three distinct fields:
| Tag ($t$ bits) | Cache Set Index ($g$ bits) | Word/Block Offset ($b$ bits) |
| :---: | :---: | :---: |

#### Hardware Translation and Access Workflow:
1. The hardware extracts the middle $g$ bits (**Cache Set Index**) to instantly pinpoint the target Set.
2. The CPU compares the upper $t$ bits of the address against the tags of all $r$ lines inside that specific Set simultaneously.
3. **Cache Hit:** If a match is found and its Valid Bit is $1$, a hit occurs. Data is read out via the lower $b$ offset bits.
4. **Cache Miss:** If no match occurs, the block is fetched from main memory and loaded into any empty line within that specific Set. If all lines within the target Set are already occupied, a replacement algorithm chooses a victim line within that set to be evicted.

---

## II. Cache Block Replacement Algorithms
When operating a Fully Associative or Set-Associative Cache, the system must choose which existing block to evict when a new memory block needs to be loaded into a completely full Cache or Set.

* **RAND (Random Algorithm):** Randomly selects a victim line for eviction. It is simple to implement in hardware and does not require tracking access history, but it typically yields lower hit rates because it ignores the principle of locality.
* **FIFO (First-In, First-Out):** Evicts the block that has resided in the Cache the longest. It is simple to manage via queue pointers but can inadvertently evict frequently used loop parameters (a phenomenon known as Belady's Anomaly).
* **LRU (Least Recently Used):** Tracks access history and evicts the block that has gone the longest period without being accessed by the CPU. This aligns perfectly with the **Principle of Locality** and generally yields excellent hit ratios, though it requires tracking bits (counters) to maintain the recency order of entries.

---

## III. Cache Write Policies
Because the Cache contains copies of data blocks from main memory, any write operation performed by the CPU must be managed carefully to maintain data consistency between the Cache and physical RAM.

### 1. Handling Cache Write Hits
When the CPU executes a write operation and the target address is already present in the Cache:

* **Write-Through (Store-Through) Policy:** The CPU writes the updated data to both the Cache and physical main memory simultaneously. 
  * *Pros:* Main memory always holds up-to-date data, making error recovery and multi-core data sharing straightforward.
  * *Cons:* Every write operation is constrained by the slower access speed of main memory, which can create a write bottleneck.
* **Write-Back (Copy-Back) Policy:** The CPU updates the data only within the Cache line, postponing the main memory update. Each Cache line features an extra control bit called a **Dirty Bit** (or Modification Bit). When a line is updated, its Dirty Bit is set to $1$.
  * *Pros:* Extremely fast; consecutive writes to the same block occur at Cache speeds without generating bus traffic.
  * *Cons:* When a dirty Cache line is selected for eviction by a replacement algorithm, it must first be written back to main memory to avoid data loss, complicating the eviction workflow. If the Dirty Bit is $0$, the line can simply be overwritten.

### 2. Handling Cache Write Misses
When the CPU executes a write operation and the target address is missing from the Cache:

* **Write Allocate Policy:** The missing block is fetched from main memory into the Cache, and the write operation is then executed as a standard cache hit. This approach is typically paired with the **Write-Back** policy, operating under the assumption that recently modified data will likely be accessed again soon.
* **No-Write Allocate (Write-Around) Policy:** The data is written directly to main memory, bypassing the Cache entirely. The missing block is not loaded into the Cache. This approach is typically paired with the **Write-Through** policy.