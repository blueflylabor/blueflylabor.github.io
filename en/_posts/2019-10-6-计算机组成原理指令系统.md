---
title: Computer Organization: Virtual Memory (Paged, Segmented, and Segmented-Paged)
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Computer Organization and Architecture]
---

# Virtual Memory: Paged, Segmented, and Segmented-Paged Schemes
Main memory (physical RAM) and auxiliary memory (secondary storage, such as HDDs/SSDs) together form the **Virtual Memory** system, managed cooperatively by dedicated hardware mechanisms and the operating system software. To application programmers, virtual memory is transparent, presenting a unified, contiguous address space that bridges both main and auxiliary storage layouts.

* **Virtual Address (Logical Address):** The address space accessible within a user program. The corresponding storage bounds are referred to as the **Virtual Space**.
* **Real Address (Physical Address):** The actual hardware address of a memory cell in physical RAM. The corresponding storage bounds are referred to as the **Main Memory Address Space** or **Real Space**.

Address components map as follows:
* $\text{Real Address} = \text{Physical Page Number (PPN)} + \text{Page Offset}$
* $\text{Virtual Address} = \text{Virtual Page Number (VPN)} + \text{Page Offset}$
* <NoLaTeX>\text{Auxiliary Address} = \text{Disk ID} + \text{Surface ID} + \text{Track ID} + \text{Sector ID}</NoLaTeX>

When the CPU executes instructions using a virtual address, auxiliary hardware components (the Memory Management Unit, or MMU) resolve the mapping between the virtual and physical addresses, checking whether the targeted block has already been loaded into physical RAM.
* **If present in RAM:** The system performs a hardware address translation, enabling the CPU to access the target memory cell directly.
* **If absent from RAM:** A page fault or segment fault is triggered. The operating system swaps the missing block from secondary storage into physical RAM, updates the translation mappings, and resumes execution.

---

## 1. Paged Virtual Memory
This scheme divides both the virtual address space and the physical main memory into fixed-size, uniform blocks called **Pages**. Pages in physical memory are called **Real Pages** or **Page Frames**, while pages in virtual memory are called **Virtual Pages**.



### The Page Table
A **Page Table** is a foundational data structure maintained in main memory that maps virtual page numbers (VPN) to physical page numbers (PPN). It tracks exactly where a program's virtual pages reside within physical RAM.

Each page table entry (PTE) generally contains the following control fields:

| Valid Bit (Present) | Dirty Bit (Modified) | Reference Bit (Used) | Physical Page Number (PPN) |
| :---: | :---: | :---: | :---: |

* **Valid Bit (Present Bit):** Indicates whether the corresponding virtual page currently resides in physical RAM ($1 = \text{Present}$, $0 = \text{Absent}$).
* **Dirty Bit (Modified Bit):** Tracks whether the page has been modified since it was loaded. Used by write-back caching policies to determine if the page must be written back to disk before being replaced.
* **Reference Bit (Used Bit):** Tracks recent page access history. Used by page replacement algorithms (such as LRU or Clock) to select victim pages during replacement.

#### Address Translation Workflow:
1. The **Page Table Base Register (PTBR)** stores the starting physical address of the current process's page table.
2. The hardware extracts the higher-order **Virtual Page Number (VPN)** from the target virtual address to locate the correct page table entry (PTE).
3. If the **Valid Bit** is $1$, the hardware extracts the **Physical Page Number (PPN)** and concatenates it with the lower-order **Page Offset** from the original virtual address to construct the final physical memory address.
4. If the **Valid Bit** is $0$, a **Page Fault** exception is raised, transferring execution control to the OS kernel page-fault handler.

### Translation Lookaside Buffer (TLB)
Operating under the principle of locality, a **Translation Lookaside Buffer (TLB)**, or **Fast Table**, is a small, high-speed associative hardware cache integrated directly into the processor. The primary page table stored in RAM is conversely called the **Slow Table**. When translating addresses, the MMU queries the TLB first. A TLB hit resolves the physical address instantly, bypassing the need to read the page table from slower main memory.

The TLB is highly parallelized, typically implementing **Set-Associative** or **Fully-Associative** cache mapping architectures:

| Tag | Valid Bit | Physical Page Number (PPN) |
| :---: | :---: | :---: |

### Hierarchical Memory Operations (TLB, Page Table, and Cache Interaction)
The table below maps the potential outcomes across a multi-level storage system consisting of a TLB, a Page Table (Page), and a hardware data Cache:

| Case | TLB | Page Table | Cache | Operational Description |
| :---: | :---: | :---: | :---: | :--- |
| **1** | Hit | Hit | Hit | Since the TLB hit, the page table must be a hit. The target data resides in physical RAM and is cached inside the L1/L2/L3 data cache. Fast execution path. |
| **2** | Hit | Hit | Miss | The TLB hit guarantees a page table hit. The page resides in RAM, but the specific data line is missing from the data cache. The system fetches the data block from RAM into the cache. |
| **3** | Miss | Hit | Hit | The TLB missed, but the page table lookup succeeded in RAM. The specific data item happens to reside inside the hardware data cache. The entry is loaded into the TLB for future use. |
| **4** | Miss | Hit | Miss | The TLB missed, but the page table lookup succeeded in RAM (the page is in memory). However, the specific data block is missing from the cache. Data must be fetched from RAM. |
| **5** | Miss | Miss | Miss | A complete miss. The TLB missed, and the page table entry indicates the page is not in memory (Page Fault). Because the page is absent from RAM, it cannot exist in the CPU cache. Requires secondary storage I/O swapping. |

---

## 2. Segmented Virtual Memory
This scheme allocates memory spaces based on the logical structure of a program (e.g., separating code sections, variable blocks, stacks, or subroutines). A logical virtual address is split into a **Segment Number** and a **Segment Offset**. Address translation is performed using a **Segment Table**, which maps these variable-length logical segments to contiguous regions within main memory.

Each segment table entry (STE) contains structural fields:

| Segment Number | Segment Base Address | Valid Bit | Segment Length |
| :---: | :---: | :---: | :---: |

#### Address Translation Workflow:
1. The hardware evaluates the **Segment Number** alongside the process's **Segment Table Base Register (STBR)** to locate the corresponding segment table row.
2. The hardware checks the **Valid Bit** to verify if the segment is loaded in RAM.
3. It validates that the **Segment Offset** does not exceed the recorded **Segment Length** (raising a segmentation fault exception if a boundary violation occurs).
4. If valid, the **Segment Base Address** is extracted and arithmetically added to the **Segment Offset** to generate the final physical memory address.

---

## 3. Segmented-Paged Virtual Memory
This hybrid scheme combines the logical organizational advantages of segmentation with the efficient memory management of paging. Programs are first segmented according to their logical components. Each variable-length logical segment is then internally subdivided into fixed-size, uniform **Pages**. Physical memory is likewise mapped into identical page frames. Data transfers between RAM and secondary disk storage remain strictly page-based.

Under this configuration, an individual program maintains one master Segment Table, and each segment entry references its own unique Page Table. The length of any segment must be an integer multiple of the uniform system page size.

A logical virtual address is structured into three distinct fields:

| Segment Number | Virtual Page Number | Page Offset |
| :---: | :---: | :---: |



#### Address Translation Workflow:
1. The MMU uses the **Segment Number** to look up the appropriate entry in the process's Segment Table.
2. The segment entry provides the base address of the **Page Table** dedicated to that specific segment.
3. The **Virtual Page Number** is used as an index into this designated Page Table to locate the target page frame.
4. The **Physical Page Number (PPN)** is extracted and concatenated directly with the original **Page Offset** to form the final physical main memory address.

---

## Summary Comparison: Virtual Memory vs. Cache Systems

| Characteristic / Metric | Hardware Cache System | Virtual Memory System |
| :--- | :--- | :--- |
| **Primary Objective** | Maximizes overall system execution speed by matching processor performance with memory performance. | Expands the logical storage capacity of main memory to accommodate large programs. |
| **Basic Transfer Unit** | Small information blocks (Cache Lines, typically 64 bytes). | Large structural information blocks (Pages or Segments, typically 4KB or larger). |
| **Implementation Layer** | Implemented purely via dedicated hardware components; completely transparent to all programmers. | Implemented co-operatively by the Operating System (OS) and hardware architectures. Transparent to application programmers, but visible to systems programmers. |
| **Design Mechanisms** | Both rely on identical computer architectural principles, including address mapping methods, block replacement policies, and data write update strategies. | Both rely on identical computer architectural principles, including address mapping methods, block replacement policies, and data write update strategies. |
| **Core Governing Principle** | **Principle of Locality** (both temporal and spatial locality properties). | **Principle of Locality** (both temporal and spatial locality properties). |