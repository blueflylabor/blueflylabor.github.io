
---
title: Computer Organization: The Central Processing Unit (CPU)
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Computer Organization and Architecture]
lang: en

---

# Central Processing Unit (CPU)
The CPU is the core hardware component responsible for executing instructions. It consists of two primary parts: the **Control Unit (CU)**, which orchestrates data flow and handles the instruction sequence (Fetch, Decode, Execute), and the **Execution Unit / Arithmetic Logic Unit (ALU)**, which processes and manipulates data.

### Core Architecture



#### 1. Arithmetic Logic Unit (ALU) & Execution Components:
* **Arithmetic Logic Unit (ALU):** Performs basic arithmetic (addition, subtraction) and logical operations (AND, OR, NOT).
* **Temporary Registers:** Holds transient source operands or intermediate outputs safely during internal operations.
* **Accumulator Register (ACC):** A dedicated register that implicitly buffers primary input values or running calculation results.
* **General-Purpose Registers (GPRs):** Fast local storage array (e.g., `AX`, `BX`, `CX`, `DX`, `SP`) used to hold user metrics, counting increments, or physical data tracking offsets.
* **Program Status Word Register (PSW):** Stores system status condition flags representing execution properties:
  * **OF (Overflow Flag):** Indicates arithmetic signed overflow boundaries were breached.
  * **SF (Sign Flag):** Reflects negative computation traits ($1 = \text{Negative}$, $0 = \text{Positive}$).
  * **ZF (Zero Flag):** Set to $1$ if the result of an operation is exactly zero.
  * **CF (Carry Flag):** Tracks high-order unsigned bit carry-outs or borrow actions.
* **Shifter / Counters:** Specialized logic blocks that handle hardware bit-shifting and internal timing increments.

#### 2. Control Unit (CU) Components:
* **Program Counter (PC):** Holds the memory address of the next sequential instruction to be fetched.
* **Instruction Register (IR):** Temporarily stores the raw binary instruction machine code fetched from main memory.
* **Instruction Decoder (ID):** Translates the operational binary opcode from the IR into localized functional control signals.
* **Memory Address Register (MAR):** Buffers the physical memory address being read from or written to via the system address bus.
* **Memory Data Register (MDR):** Buffers the actual data word being transferred to or from main memory via the system data bus.
* **Timing System / Micro-operation Signal Generator:** The combinational clock networks that emit control signals in sequence to synchronize hardware actions.

---

## Instruction Execution Cycle

### 1. The Instruction Cycle
An **Instruction Cycle** is the total time required for the CPU to fetch, decode, and execute a single machine instruction. It is divided into several **Machine Cycles** (which can be of equal or variable lengths). Each machine cycle corresponds to a single major hardware step and contains multiple **Clock Cycles (T-states or ticks)**, which represent the smallest fundamental units of CPU timing.



* **Unconditional Branch Instructions:** Skip memory accesses during execution. They consist only of the *Fetch* stage and a quick *Execute* internal update stage.
* **Indirect Addressing Instructions:** Require an additional **Indirect Cycle** between Fetch and Execute to fetch the true effective operand address from memory.
* **Interrupt Cycle:** At the end of every execution sequence, the CPU samples its interrupt lines. If a pending request is detected, it enters an Interrupt Cycle to save execution context state parameters.

The four phases are tracked by internal hardware state flags: **FE (Fetch)**, **IND (Indirect)**, **EX (Execute)**, and **INT (Interrupt)**.

### 2. Instruction Cycle Data Flows

#### Fetch Cycle (FE)
Fetches the machine instruction code from main memory into the Instruction Register using the current address tracked inside the Program Counter:
1. $\text{PC} \to \text{MAR} \to \text{Address Bus} \to \text{Main Memory}$
2. $\text{CU issues a READ command} \to \text{Control Bus} \to \text{Main Memory}$
3. $\text{Main Memory} \to \text{Data Bus} \to \text{MDR} \to \text{IR (Instruction stored)}$
4. $\text{CU triggers increment signal (FE): } \text{PC} + 1 \to \text{PC}$

#### Indirect Cycle (IND)
Resolves indirect pointer variables to find the actual effective address of data operands:
1. $\text{Address Field(IR)} \text{ or } \text{MDR} \to \text{MAR} \to \text{Address Bus} \to \text{Main Memory}$
2. $\text{CU issues a READ command} \to \text{Control Bus} \to \text{Main Memory}$
3. $\text{Main Memory} \to \text{Data Bus} \to \text{MDR (Now contains the true effective address)}$

#### Execute Cycle (EX)
Decodes the operation code (opcode) inside the IR and routes operands through the ALU. Data paths vary significantly depending on the instruction type.

#### Interrupt Cycle (INT)
Saves the program breakpoint context onto the system stack and branches execution to an Interrupt Service Routine (ISR):
1. $\text{CU updates stack pointer: } \text{SP} - 1 \to \text{SP} \to \text{MAR} \to \text{Address Bus}$
2. $\text{CU issues a WRITE command} \to \text{Control Bus} \to \text{Main Memory}$
3. $\text{PC} \to \text{MDR} \to \text{Data Bus} \to \text{Main Memory (Saves return address on the stack)}$
4. $\text{CU loads the vector address of the target ISR entry point} \to \text{PC}$

### 3. Execution Strategies
* **Single-Cycle Design:** Every instruction executes completely within one long clock cycle. This requires a slow clock rate because the clock period is constrained by the slowest possible instruction.
* **Multi-Cycle Design:** Instructions are broken down into discrete steps, each executing in a single clock cycle. Faster instructions finish in fewer steps, optimizing overall performance.
* **Pipelined Design:** Multiple instructions overlap execution simultaneously across a sequence of dedicated hardware stages, maximizing throughput.

---

## Data Paths

### Definition & Layout
A **Data Path** is the physical route that binary data travels as it moves between functional units (such as registers, the ALU, and memory multipliers). It is orchestrated by control signals generated by the Control Unit.



#### 1. Internal Single-Bus Architecture
All internal registers share a single common communication bus. While simple and economical, this design introduces resource contention conflicts, as only one data word can occupy the bus per clock cycle.

#### 2. Internal Multi-Bus Architecture
Registers are linked across multiple independent internal buses. This allows different data sets to move concurrently within the same clock tick, significantly increasing execution speed.

#### 3. Dedicated Point-to-Point Data Paths
Direct connections are established between specific functional components where data frequently flows. This eliminates bus contention and yields excellent performance, but requires a massive amount of hardware wiring.

### Basic Data Path Transfers

#### Register-to-Register Transfers:
Controlled by matching gating flags (e.g., `PCout` enables an output onto the bus, while `MARin` latches data into the destination register from the bus):
* Example ($\text{PC} \to \text{MAR}$): Enable `PCout` and `MARin` control lines concurrently.

#### Main Memory-to-CPU Read Operations:
1. $\text{PC} \to \text{MAR}$ — Enable `PCout` and `MARin`.
2. $\text{CU issues Read control signal}$ — Set $R = 1$ over the system bus.
3. $\text{Memory(MAR)} \to \text{MDR}$ — Latch the fetched memory word into the MDR by enabling `MDRin`.
4. $\text{MDR} \to \text{IR}$ — Enable `MDRout` and `IRin` to move the instruction code into position for decoding.

#### Arithmetic Logic Execution ($\text{Memory Operand} + \text{ACC} \to \text{ACC}$):
Because the ALU lacks internal memory storage, both source operands must be presented to its inputs simultaneously. This requires buffering one operand in a temporary register (e.g., Register Y):
1. $\text{Address(IR)} \to \text{MAR}$ — Enable `MDRout` and `MARin`.
2. $\text{CU issues Read control signal}$ — Set $R = 1$.
3. $\text{Memory} \to \text{MDR}$ — The operand is fetched from memory into the MDR.
4. $\text{MDR} \to \text{Y}$ — Move the operand to buffer register Y (`MDRout` and `Yin`).
5. $\text{ACC} + \text{Y} \to \text{Z}$ — Enable `ACCout` and activate the ALU addition control signal. The result is captured in output buffer register Z.
6. $\text{Z} \to \text{ACC}$ — Route the final calculation result back to the Accumulator (`Zout` and `ACCin`).

---

## Control Unit Mechanics

Based on how control signals are generated, controllers are divided into **Hardwired Control Units** and **Microprogrammed Control Units**.

### 1. Hardwired Control Units
Hardwired controllers generate control signals using fixed, combinational logic gates and state flip-flops. Signals are derived directly from the current instruction opcode, active step timing, and status flags.



#### Control Unit Inputs:
* **Instruction Decoder (ID):** Decodes the active opcode from the IR.
* **Timing State Generator:** Emits machine cycle and clock tick (step) markers.
* **Execution Flags:** Status indicators (e.g., PSW conditional bits like zero or overflow flags).
* **System Bus Controls:** Handles inbound platform lines like interrupt requests (`INTR`).

* **Pros:** Extremely fast operation with minimal signal propagation delay.
* **Cons:** Complex, irregular structure. Modifying or expanding the instruction set requires redesigning the physical chip layout.

---

### 2. Microprogrammed Control Units
Microprogrammed controllers treat instruction execution as a sequence of tiny hardware steps called **Micro-operations**. The control signals required for each step are stored as a binary code word called a **Micro-instruction**. A collection of these micro-instructions forms a **Micro-program**, which defines the execution of a single machine instruction. All micro-programs are stored in a dedicated read-only memory within the CPU called the **Control Memory (CM)**.



#### Key Terms:
* **Micro-command:** An individual control signal (e.g., `PCout`). It is the smallest functional control element.
* **Micro-operation:** The actual physical action executed in response to a micro-command.
* **Micro-instruction:** A control word containing a set of micro-commands that execute simultaneously in a single clock tick, along with sequencing information to locate the next micro-instruction.
* **Control Memory (CM):** A high-speed, internal ROM used to store micro-programs. (The main memory, by contrast, holds user code and data).

#### Comparison of Register Roles:
| Core Memory Component | Address Pointer Register | Data / Output Buffer Register |
| :--- | :--- | :--- |
| **Main Memory (RAM)** | **MAR** (Memory Address Register) | **MDR** (Memory Data Register) / **IR** |
| **Control Memory (ROM)** | **CMAR** (Control Memory Address Register) | **CMDR** or **$\mu$IR** (Micro-Instruction Register) |

#### Execution Workflow:
1. The CPU automatically loads the initialization micro-address of the **Fetch Micro-program** into the CMAR (typically address `0x0`).
2. The entry micro-instruction is read from the CM into the CMDR to execute the standard machine fetch step, loading a user instruction from RAM into the main Instruction Register (IR).
3. The **Micro-address Generation Logic** translates the user opcode inside the IR into the starting address of its corresponding micro-program, then latches it into the CMAR.
4. The controller steps through the micro-program from the CM, executing the micro-instructions sequentially to perform the instruction's task.
5. The final micro-instruction in the sequence resets the CMAR back to the Fetch Micro-program entry point to process the next instruction.

#### Micro-instruction Fields:
Micro-instructions typically consist of two main fields:
1. **Control Field (Micro-opcode):** Emits the actual micro-commands to the hardware components.
2. **Next-Address Field (Micro-operand):** Contains sequencing bits used to compute or jump to the next micro-instruction address.

#### Control Field Encoding Formats:
* **Direct Encoding:** Each bit in the control field directly represents a single micro-command ($1 = \text{Active}$, $0 = \text{Inactive}$). It is extremely fast and requires no decoding logic, but results in very wide micro-instructions if the system has many control signals.
* **Field Direct Encoding:** Micro-commands are grouped into mutually exclusive fields. Commands that cannot occur simultaneously share a field, and each field is decoded independently. This significantly reduces the width of the micro-instruction word.

#### Structural Classification:
* **Horizontal Micro-instructions:** Wide words that can execute many micro-commands in parallel. They offer high performance and fast execution but are complex to program.
* **Vertical Micro-instructions:** Narrow words resembling standard machine instructions, where each micro-instruction performs only one or two operations. They require decoding logic, resulting in longer execution times, but are easier to program and require less storage space.

---

### Structural Comparison

| Metric / Feature | Microprogrammed Controller | Hardwired Controller |
| :--- | :--- | :--- |
| **Operating Principle** | Control signals are looked up from a micro-program stored in internal ROM. | Control signals are generated in real time by combinational logic gates. |
| **Execution Speed** | Slower (requires an internal memory access to fetch each micro-instruction). | Faster (limited only by logic gate propagation delays). |
| **Structural Regularity**| Highly regular, clean design layout. | Complex, irregular "random logic" design layout. |
| **Flexibility & Upgrades**| Highly flexible; easy to modify or expand via microcode firmware adjustments. | Extremely difficult; requires physical rewiring or a total hardware redesign. |
| **Target Architecture** | Typically used in Complex Instruction Set Computers (**CISC**). | Typically used in Reduced Instruction Set Computers (**RISC**). |

---

## Exceptions (Internal Interrupts) and Interrupts

* **Exceptions (Internal Interrupts):** Synchronous events generated internally by the CPU, directly tied to the execution of a specific instruction (e.g., division by zero, arithmetic overflow, page faults, or illegal opcodes).
* **Interrupts (External Interrupts):** Asynchronous events generated outside the CPU by external hardware devices (e.g., I/O peripheral ready alerts, timer expiration ticks, or hardware faults).

### Categories of Exceptions:
1. **Faults:** Anomalies detected during instruction execution (e.g., page faults). If fixed by the OS handler, the CPU rolls back its state and re-executes the faulting instruction.
2. **Traps:** Intentional software breakpoints or system calls embedded in a program. Once handled, control returns to the next sequential instruction.
3. **Aborts:** Severe, unrecoverable hardware failures. Execution cannot be resumed; the current process is terminated immediately.

### Interrupt Vectors and Handling:
* **Software Identification:** The CPU updates a status cause register. The OS runs a centralized routine to poll and evaluate priority flags sequentially to find the source.
* **Hardware Identification (Vectored Interrupts):** The interrupting device sends an interrupt type number to the CPU over the bus. The CPU uses this number as an index into an **Interrupt Vector Table** in memory to immediately locate and jump to the correct Interrupt Service Routine (ISR).

---

## Instruction Pipelining
Instruction pipelining increases processor throughput by overlapping the execution of multiple instructions. The complete execution cycle of an instruction is divided into independent, sequential stages:



1. **Instruction Fetch (IF):** Fetches the instruction code from the memory system.
2. **Instruction Decode (ID):** Decodes the instruction and reads source operands from the register file.
3. **Execute (EX):** Performs the operation in the ALU or computes an effective memory address.
4. **Memory Access (MEM):** Performs a data read or write operation if required by a load or store instruction.
5. **Write-Back (WB):** Writes the final calculation or memory result back into the register file.

---
---

# Instruction Set Architecture (ISA)
The **Instruction Set Architecture (ISA)** defines the boundary interface between hardware execution units and software applications. It specifies the supported instruction formats, native data types, available registers, addressing modes, and memory layout configurations.

### Standard Instruction Component Format
A standard machine instruction consists of two primary bit fields:

| Opcode Field (OP) | Address Field (A) |
| :---: | :---: |

* **Opcode (Operation Code):** Specifies the operation to be performed (e.g., `ADD`, `SUB`, `MOV`) and identifies the instruction type.
* **Address Field:** Provides the memory addresses or register identifiers of the operands, destination targets, or branch destinations.

#### Instruction Classifications by Address Count:
* **Zero-Address Instructions:** Contain no explicit address fields. Used for operations that do not require operands (e.g., `NOP`, `CLI`) or in stack-based architectures where operands are implicitly popped from the top of the stack.
* **One-Address Instructions:** Contain a single operand address field. Used for unary operations (e.g., `INC`) or operations where a second operand and the destination are implicitly assumed to be a dedicated register like the Accumulator (ACC):
  $$\text{Expression: } (\text{ACC}) \text{ OP } (A_1) \to \text{ACC}$$
* **Two-Address Instructions:** The most common format for binary operations. Specifies a source operand and a destination operand (which also holds the result):
  $$\text{Expression: } (A_1) \text{ OP } (A_2) \to A_1$$
* **Three-Address Instructions:** Explicitly specifies two source operands and a separate destination target, avoiding the overwrite of source data:
  $$\text{Expression: } (A_1) \text{ OP } (A_2) \to A_3$$

---

## Opcode Schemes

### 1. Fixed-Length Opcodes
The opcode field uses a fixed number of bits for every instruction in the ISA. For an $n$-bit field, the system can support up to $2^n$ unique instructions. This simplifies decoding hardware but can waste instruction word bits if many instructions require few operands.

### 2. Variable-Length (Expanding) Opcodes
The size of the opcode field varies depending on the number of operands an instruction requires. Instructions with fewer explicit operands utilize the unused address bits to expand their opcodes.

#### Critical Rule:
To ensure unambiguous decoding, a shorter opcode **must never** match the prefix of a longer opcode.

#### Example Expansion Strategy (32-bit Instruction Word with 8-bit Address Fields):


```

[4-bit Master Opcode Zone]
0000 -> 3-Address Instruction 1
0001 -> 3-Address Instruction 2
...
1110 -> 3-Address Instruction 15  (Leaves 1 prefix pattern '1111' free for expansion)

If Prefix == 1111: Read next 4 bits to resolve a 2-Address Instruction:
1111 0000 -> 2-Address Instruction 1
1111 1011 -> 2-Address Instruction 12 (Leaves remaining pattern blocks free to cascade down)

```

---

## Operand Addressing Modes
Addressing modes define how the system interprets instruction fields to calculate the **Effective Address (EA)** of an operand in memory or registers.

[Image showing different addressing modes like direct, indirect, register and indexed]

### 1. Implied Addressing
The operand's location is implicitly defined by the instruction type itself, so no explicit address bits are required (e.g., modifying the stack pointer or manipulating the accumulator).

### 2. Immediate Addressing
The address field does not contain an address, but holds the actual operand value itself (typically represented by a prefix like `#`).
* **Pros:** Extremely fast execution; no memory access is required to fetch the operand.
* **Cons:** The size of the operand value is limited by the bit width of the address field.

### 3. Direct Addressing
The address field contains the actual physical or logical effective address of the operand in memory:
$$\text{Formula: } \text{EA} = A$$
* **Pros:** Simple to implement; requires only a single memory access to fetch the operand.
* **Cons:** The addressable memory space is constrained by the bit width of the address field, and addresses cannot be modified dynamically at runtime.

### 4. Indirect Addressing
The address field points to a memory location that contains the true effective address of the operand:
$$\text{Formula: } \text{EA} = (A)$$
* **Pros:** Greatly expands the addressable memory space, as a wide address pointer can be stored in a standard memory slot. It also simplifies implementing pointers and subroutine returns.
* **Cons:** Slower performance, as it requires multiple sequential memory accesses to retrieve a single operand.

### 5. Register Addressing
The address field specifies a register identifier within the CPU that contains the operand value:
$$\text{Formula: } \text{EA} = R_i$$
* **Pros:** Fast execution; accessing registers is much faster than accessing main memory. It also reduces instruction size because fewer bits are needed to address a register file than a memory space.
* **Cons:** The number of available registers is limited by hardware costs.

### 6. Register Indirect Addressing
The address field specifies a register that contains the memory address of the operand:
$$\text{Formula: } \text{EA} = (R_i)$$
* **Pros:** Faster than memory indirect addressing because the initial pointer lookup occurs within a high-speed CPU register, requiring only one subsequent memory access.

### 7. Relative Addressing
The effective address is computed by adding the address field's displacement value (treated as a signed two's complement offset) to the current value of the Program Counter (PC):
$$\text{Formula: } \text{EA} = (\text{PC}) + A$$
* **Pros:** Supports position-independent code (program relocation), allowing code blocks to execute correctly regardless of where they are loaded into physical memory. It is widely used for conditional branch and jump instructions.

### 8. Base-Register Addressing
The effective address is computed by adding a displacement offset from the instruction field to the contents of a dedicated Base Register (BR):
$$\text{Formula: } \text{EA} = (\text{BR}) + A$$
* **Pros:** Managed primarily by the operating system to facilitate multitasking and program relocation. It dynamically aligns a program's logical address space with its assigned physical memory footprint.

### 9. Indexed Addressing
The effective address is computed by adding the contents of an Index Register (IX) to a fixed base address specified in the instruction's address field:
$$\text{Formula: } \text{EA} = (\text{IX}) + A$$
* **Pros:** The user can dynamically modify the index register value at runtime while keeping the base address constant. This makes it highly efficient for iterating through arrays, vectors, and linear data blocks in loops.

---

## Addressing Mode Summary

| Addressing Mode | Effective Address Formula ($\text{EA}$) | Memory Accesses (To Fetch Operand) | Primary Advantages |
| :--- | :--- | :---: | :--- |
| **Implied** | Implicitly defined by instruction | $0$ | Reduces instruction size. |
| **Immediate**| Address field holds the operand | $0$ | Maximum speed; no memory lookup. |
| **Direct** | $\text{EA} = A$ | $1$ | Simple; requires no complex calculations. |
| **Indirect** | $\text{EA} = (A)$ | $2+$ | Expands addressable memory space. |
| **Register** | $\text{EA} = R_i$ | $0$ | High speed; eliminates memory access overhead. |
| **Register Indirect** | $\text{EA} = (R_i)$ | $1$ | Efficient pointer manipulation and array access. |
| **Relative** | $\text{EA} = (\text{PC}) + A$ | $1$ | Enables position-independent code relocation. |
| **Base-Register**| $\text{EA} = (\text{BR}) + A$ | $1$ | Simplifies segmented memory allocation by the OS. |
| **Indexed** | $\text{EA} = (\text{IX}) + A$ | $1$ | Simplifies array traversal and loop structures. |
