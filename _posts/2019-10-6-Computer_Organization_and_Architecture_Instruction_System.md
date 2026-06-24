---
title: "Computer Organization and Architecture - Instruction System"
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Computer Organization and Architecture]
---

# Instruction System
## Instruction System
It is the core of the Instruction Set Architecture (ISA).  
The ISA mainly includes:
- Instruction formats
- Data types and formats
- Storage methods for operands
- Number, bit-width, and numbering of registers accessible by programs
- Storage space size and addressing methods
- Addressing modes
- Control methods for the instruction execution process, etc.

### Basic Format of Instructions
An instruction consists of an opcode field and an address code field.
|||
|-|-|
|Opcode|Address Code|

Opcode:
- Specifies the operation to be executed by the instruction
- Identifies the instruction
- Clarifies the function of the instruction
- Distinguishes the composition and usage of the operand address content  

Address Code:
- Provides the address of the information to be operated on
- The address where one or more operands involved in the operation are located
- The storage address for the operation result
- The jump address for programs
- The entry address of a called subroutine, etc.

Instruction length refers to the number of binary bits contained within an instruction.  
The instruction word length depends on:
- The length of the opcode
- The length of the operand address code
- The number of operand addresses  

Single-word length instruction: Equal to the machine word length.  
Half-word length instruction: Half of the machine word length.  
Double-word length instruction: Twice the machine word length.  
Fixed-length instruction word structure: The lengths of all instructions in an instruction system are equal.  

#### Zero-Address Instruction: No explicit address

||
|-|
|OP|

- Instructions that do not require an operand.
- Zero-address arithmetic instructions are only used in stack computers. Usually, the two operands involved in the operation are implicitly popped from the top of the stack and the next-to-top of the stack, sent to the ALU, and the operation result is implicitly pushed back onto the stack.  

One-Address Instruction:

|||
|-|-|
|OP|$$A_1$$|

OP($$A_1$$) $$\to$$ $$A_1$$
- Contains only the destination operand. The operand is read according to address $$A_1$$, the OP operation is performed, and the result is stored back into the original address.  

(ACC)OP($$A_1$$) $$\to$$ ACC
- Dual-operand instruction with an implicitly agreed destination address. The operand is read according to the instruction address $$A_1$$, and the instruction implicitly agrees that the other operand is provided by the ACC. The operation result will also be stored in the ACC.
- If the instruction length is 32 bits, the opcode occupies 8 bits, and 1 address code field occupies 24 bits, the direct addressing range of the instruction operand is $$2^{24}=16M$$.

#### Two-Address Instruction
||||
|-|-|-|
|OP|$$A_1$$|$$A_2$$|

($$A_1$$)OP($$A_2$$) $$\to$$ $$A_1$$

- Commonly used arithmetic and logical operation instructions require two operands. It is necessary to provide the destination operand and the source operand respectively, where the destination operand address is also used to store the result of this operation.
- If the instruction word length is 32 bits, the opcode occupies 8 bits, and the two address codes each occupy 12 bits, then the direct addressing range of the instruction operand is $$2^{12}=4K$$.

#### Three-Address Instruction
|||||
|-|-|-|-|
|OP|$$A_1$$|$$A_2$$|$$A_3$$(Result)|

($$A_1$$)OP($$A_2$$) $$\to$$ $$A_3$$

- If the instruction word length is 32 bits, the opcode occupies 8 bits, and the 3 address codes each occupy 8 bits, the direct addressing range is $$2^8=256$$. If the address field is a main memory address, completing a single three-address instruction requires 4 memory accesses: 1 time to fetch the instruction, 2 times to fetch the two operands, and 1 time to store the result.

#### Four-Address Instruction

||||||
|-|-|-|-|-|
|OP|$$A_1$$|$$A_2$$|$$A_3$$|$$A_4$$|

($$A_1$$)OP($$A_2$$) $$\to$$$$A_3$$ ,$$A_4$$ = Address of the next instruction to be executed

- The address word length is 32 bits, the opcode occupies 8 bits, and the 4 address codes each occupy 6 bits, making the direct addressing range $$2^6=64$$.

### Fixed-Length Opcode Instruction Format
A fixed number of bits (fixed length) are allocated in the highest bits of the instruction word to represent the opcode.
An instruction system with an n-bit opcode field can represent a maximum of $$2^{n}$$ instructions.

### Expanding Opcode Instruction Format
- Short codes are not allowed to be prefixes of long codes.
- The opcodes of each instruction must not overlap.

|||||
|-|-|-|-|
|0000|0001|0010|0011|
|0100|0101|0110|0111|
|1000|1001|1010|1011|
|1100|1101|1110|1111|

|||||||
|-|-|-|-|-|-|
|Opcode Case|OP|$$A_1$$|$$A_2$$|$$A_3$$|Explanation|
|15 Three-Address|0000-1110||||16-15=1 left over, 1*2^4=16 combinations|
|12 Two-Address|1111|0000-1011|||16-12=4 left over, 4*2^4=64 combinations|
|62 One-Address|1111|(1100-1110)/1111|(0000-1111)/(0000-1101)||64-62=2 left over, 2*2^4=32 combinations|
|32 Zero-Address|1111|1111|1110-1111|0000-1111|||

### Operation Types of Instructions
- Data transfer
- Arithmetic and logical operations
- Shift
- Transfer/Jump
- Input/Output

## Instruction Addressing Modes
Determines the data address of the current instruction and the address of the next instruction to be executed, divided into:
- Instruction Addressing: Finding the next instruction to be executed  
(1) Sequential Addressing  
Formed automatically via PC+(1) to get the next instruction.  
(2) Skip Addressing  
Implemented through jump instructions. The address of the next instruction is not automatically given by the PC, but the calculation method of the next instruction address is given by the current instruction. Whether it can jump is controlled by status registers and operands. The result of a jump modifies the PC value of the current instruction, and the next instruction is still provided through the PC.

- Data Addressing: Finding the data for the current instruction  
Usually, a field is set in the instruction word to indicate the addressing mode.

    ||||
    |-|-|-|
    |Opcode|Addressing Characteristic|Formal Address A|

### Common Data Addressing Modes

#### Implied Addressing
The operand address is not explicitly given, but is implied within the instruction.
- Pros: Conducive to shortening the instruction word length.
- Cons: Requires additional hardware to store operands or implied addresses.

#### Immediate Addressing
The address field of the instruction does not indicate the operand address, but rather the operand itself, also known as an immediate value. '#' denotes immediate addressing, and it is represented using two's complement.
- Pros: The instruction does not access main memory during the execution phase, making the instruction execution time the shortest.
- Cons: The bit-width of A limits the range of the immediate value.

#### Direct Addressing
The formal address A in the instruction is the actual effective address EA of the operand, EA=A.  
- Pros: Simple, requires 1 memory access, and does not require special calculation of the operand address.
- Cons: The bit-width of A determines the addressing range of the instruction operand, and the address of the operand is not easily modified.

#### Indirect Addressing
The formal address given in the address field of the instruction is not the actual address of the operand, but rather the storage unit address of the effective address of the operand, EA=(A). Indirect addressing can be iterated multiple times.  
For indirect addressing, the first bit of the main memory cell indicates whether there are multiple levels of indirect addressing.
- Pros: Can expand the addressing range (the bit-width of the effective address EA is larger than that of the formal address A), convenient for programming (indirect addressing can easily complete subroutine returns).
- Cons: Slow access speed.

#### Register Addressing
The register number where the operand is located is directly given in the instruction word, EA= $$R_i$$, and the operand is inside the register pointed to by $$R_i$$.
- Pros: The instruction execution phase does not access memory, it only accesses registers. The address code length corresponding to registers is relatively small, making the instruction word short. Because it does not access memory, its execution speed is fast, and it supports vector/matrix operations.
- Cons: Registers are expensive and limited in number.

#### Register Indirect Addressing
The register $$R_i$$ does not contain an operand, but rather the address of the main memory cell where the operand is located, EA=($$R_i$$).
- Pros: Faster speed compared to general indirect addressing.
- Cons: Requires memory access.

#### Relative Addressing
The content of the PC is added to the formal address A of the instruction format to form the effective address EA of the operand, EA=(PC)+A. A is the displacement amount relative to the current PC value, which can be positive or negative, expressed in two's complement. The bit-width of A determines the addressing range.
- The address of the operand is not fixed; it changes as the value of the PC changes, and maintaining a fixed difference from the instruction address makes it convenient for code floating. It is widely used in jump instructions.
- JMP A: The CPU fetches one byte from memory, automatically executes (PC)+1 $$\to$$ PC. If the address of the jump instruction is X and it occupies 2B, after fetching this instruction, the PC auto-increments by 2, (PC)=X+2. After executing this instruction, it will automatically jump to the address X+2+A to continue execution.

#### Base-Register Addressing
The content of the CPU's base register BR is added to the formal address A of the instruction format to form the effective address EA of the operand, EA=(BR)+A. The base register can be a dedicated register or a general-purpose register.
- The base register faces the operating system, and its content is determined by the operating system or management program. It is mainly used to solve the independence between the logical space of programs and the physical space of memory.
- During execution, the content of the base register remains unchanged, while the formal address can change (offset).
- When a general-purpose register is used as a base register, the user can decide which register to use, but the content is determined by the operating system.

- Pros: Can expand the addressing range (the bit-width of the base register is larger than that of the formal address A), users do not need to consider which area of the main memory their program is stored in, conducive to multi-programming, and can be used to construct floating programs.
- Cons: The offset bit-width is relatively short.

#### Indexed Addressing
The effective address EA is equal to the sum of the formal address A in the instruction word and the content of the index register IX, EA=(IX)+A.
- IX can use a dedicated register or a general-purpose register.
- The index register faces the user. During program execution, the content of the index register can be changed by the user (acting as an offset), while the formal address A remains unchanged (acting as a base address).
- Can expand the addressing range (the bit-width of the index register is larger than that of the formal address A), suitable for writing loop programs, and the bit-width of the offset (IX) is sufficient to represent the entire storage space.

#### Stack Addressing
A stack is a specific storage area in memory (or a dedicated register set) managed according to the Last-In-First-Out (LIFO) principle. The read/write unit address of the storage area is given by a specific register called the Stack Pointer (SP). It is divided into hard stacks (not suitable for large-capacity stacks) and soft stacks (a segment of the main memory allocated for it).

|Addressing Mode|Effective Address|Memory Access Count|
|-|-|-|
|Implied Addressing|Specified by Program|0|
|Immediate Addressing|A is the operand|0|
|Direct Addressing|EA=A|1|
|Single Indirect Addressing|EA=(A)|2|
|Register Addressing|EA= $$R_i$$|0|
|Register Indirect Single Addressing|EA = ($$R_i$$)|1|
|Relative Addressing|EA=(PC)+A|1|
|Base-Register Addressing|EA=(BR)+A|1|
|Indexed Addressing|EA=(IX)+A|1|