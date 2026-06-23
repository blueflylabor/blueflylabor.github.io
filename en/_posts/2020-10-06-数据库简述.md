---
title: Database Management Systems: An Introduction
date:   2020-10-06
last_modified_at: 2020-10-06
categories: notes
tags: [Database Systems]
---

# Database Systems Overview

## 1. Introduction to Database Systems

### Core Concepts
The architecture of data management centers around four fundamental concepts: **Data**, **Database (DB)**, **Database Management System (DBMS)**, and **Database System (DBS)**.

* **Data:** Symbolic records that describe real-world objects or events. The meaning behind these symbols is known as data semantics. Data and its semantics are fundamentally inseparable.
* **Database (DB):** A long-term, structurally organized, highly shareable collection of related data stored systematically within a computer system. It organizes, describes, and stores records based on specific data models to guarantee low redundancy, substantial data independence, and easy scalability.
* **Database Management System (DBMS):** A layer of system software positioned between the operating system (OS) and end-users. It facilitates data definition, physical storage organization, structural management, query manipulation, transaction processing, concurrency monitoring, and system maintenance.
* **Database System (DBS):** A comprehensive system configuration composed of the physical **Database**, the managing **DBMS**, front-end application programs, and the **Database Administrator (DBA)** working together to store, maintain, and process information.

### Historical Evolution of Data Management
The mechanisms for tracking electronic information evolved through three discrete historical stages:


```

[Manual Management Stage] ──> [File System Stage] ──> [Database System Stage]

```

* **Manual Management Stage:** Early computing where application programs managed data internally. Data was not preserved after execution, lacked independent structures, and was bound strictly to individual programs.
* **File System Stage:** Data was externalized into structured files stored on long-term media. While files could be opened across applications, data remained scattered across isolated silos, causing high redundancy and low data independence.
* **Database System Stage:** The modern era where data is structurally integrated globally. Data management is entirely centralized within a DBMS, providing high shareability, physical/logical independence, and structural flexibility.

---

### Key Features of a Database System

* **Structured Data Realization:** This is the foundational difference between a database system and a simple file system. Data is no longer modeled as isolated flat files; instead, it reflects the inherent structural relationships of the real world.
* **High Shareability, Low Redundancy, and Dynamic Extensibility:** Data is shared globally across multiple users and varied applications concurrently. This eliminates redundant duplicate records, reduces storage waste, and allows schemas to expand easily without affecting existing applications.
* **High Data Independence:** Bridges the gap between user programs and physical structures.
  * **Physical Independence:** Internal storage layout updates do not alter logical schemas.
  * **Logical Independence:** Macro logical schema shifts do not break micro user application code.
* **Centralized Security and Control:** The DBMS actively coordinates four structural guardrails to guarantee information health:
  1. *Security Protection:* Restricts unauthorized data viewing and tampering.
  2. *Integrity Controls:* Validates data against predefined business boundaries to prevent garbage inputs.
  3. *Concurrency Control:* Manages simultaneous transactions safely to prevent race conditions or data corruption.
  4. *Database Recovery:* Provides mechanisms to restore data states following system crashes or hardware failures.

---

## 2. Data Modeling Concepts

A **Data Model** is an abstract representation of real-world features and relationships. It forms the core foundation of any database management system architecture.

### The Three Layers of Data Models
Data modeling is processed across three progressive abstraction layers, moving from human conceptual models down to physical computer storage:



1. **Conceptual Models (Information Models):** Models data based on human perception and real-world business perspectives. It is strictly **user-oriented** and forms the primary blueprint for database design, independent of specific software implementations (e.g., the E-R Model).
2. **Logical Models:** Formally maps conceptual designs into concrete, logical structures supported by computer software platforms. It is **designer-oriented** and dictates how data is queried and structured (e.g., Hierarchical, Network, Relational, or Object-Oriented models).
3. **Physical Models:** The lowest level of abstraction. It defines exactly how bits, indexes, pages, and pointer blocks are arranged on physical storage media, directly interfacing with the underlying OS and hardware.

---

### The Conceptual Model: Entities and Relationships

* **Entity:** An objective, identifiable real-world object or concept that can be distinctly isolated (e.g., a specific student, an item, a corporate account).
* **Attribute:** An individual trait or characteristic that describes an entity (e.g., `Student_ID`, `Name`, `Age`).
* **Key (Identifier):** An attribute or minimal set of attributes that uniquely identifies a single entity within an entity set.
* **Entity Type:** The blueprint schema that defines a class of similar entities using a distinct name and a set of attributes (e.g., `Student(StudentID, Name, Age)`).
* **Entity Set:** A collection of actual entities sharing an identical Entity Type (e.g., all students currently enrolled in a university).
* **Relationship:** The logical associations between real-world entities. Relationships can exist within an entity set or between different entity sets, categorized into three cardinalities:
  * **One-to-One (1:1):** One entity in set $A$ maps to at most one entity in set $B$ (e.g., Manager to Department).
  * **One-to-Many (1:$N$):** One entity in set $A$ can map to multiple entities in set $B$, but a $B$ entity maps back to at most one $A$ entity (e.g., Class to Students).
  * **Many-to-Many ($M$:$N$):** Multiple entities in set $A$ can link to multiple entities in set $B$ (e.g., Students to Courses).

#### Entity-Relationship (E-R) Model
The primary notation tool used to diagram conceptual models during the initial design phase of a database.



---

### Core Structural Components of a Data Model
Every formal data model must strictly define three components:

1. **Data Structures:** Specifies the structural layout rules and composition of objects within the database (e.g., tables, trees, nodes, arrays).
2. **Data Operations:** The valid set of actions allowed on data instances at runtime, including execution behavior rules (e.g., SQL queries, record insertions, updates, deletions).
3. **Integrity Constraints:** A collection of rules that ensure data accuracy, validity, and consistency over time. It guarantees that values map perfectly to real-world business rules.

---

### Common Logical Data Models

* **Hierarchical Model:** Organizes data records in an inverted tree structure. It strictly dictates that:
  * There is exactly one root node that has no parent.
  * Every child node outside the root has exactly one parent node.
* **Network Model:** A graph-based model that loosens hierarchical constraints. It allows:
  * Multiple nodes to exist without parent records.
  * An individual child node to link to multiple parent records.
* **Relational Model:** Represents all data and relationships uniformly using flat, two-dimensional tables. It is mathematically grounded in set theory and forms the basis of modern SQL databases.
* **Object-Oriented Model:** Models data as self-contained objects encapsulating both data attributes and behavioral methods, directly mapping to object-oriented programming paradigms.
* **Semi-Structured Model:** Abandons rigid schema constraints. Data is self-describing, containing embedded tags and variable properties (e.g., XML, JSON layouts).

---

## 3. Database Architecture: The Three-Schema Architecture

Modern database systems separate internal storage mechanisms from end-user applications using the **Three-Schema Architecture** (comprising the External, Conceptual, and Internal views), connected by two dynamic mapping layers.



### System Schema vs. Database Instance
* **Schema:** The static, macro-level logical description of structural frameworks and data types. It outlines the structural design and properties of the system and rarely changes.
* **Instance (State):** The actual collection of data values stored inside those schema boundaries at a specific moment. It is dynamic and changes constantly as rows are added, updated, or removed.

### The Three Schema Layers
* **External Schema (User View):** The highest layer. It describes the specific subset of the database visible to an individual user group or application. A single database can support multiple distinct external schemas, tailoring data views based on application needs or access permissions.
* **Conceptual Schema (Logical View):** The global, centralized schema of the entire database. It defines all data entities, attributes, relationships, and constraints for the whole system, completely independent of physical storage layouts or front-end software configurations.
* **Internal Schema (Physical View):** The lowest layer. It describes the physical organization and storage layouts inside the computer, defining paths, data page groupings, indexing schemas, compression tools, and hardware device allocations. There is exactly one internal schema per database.

---

### Two-Level Mappings and Data Independence

The separation between these architectural layers is maintained by two hardware/software mapping layers managed by the DBMS. These mappings provide **Data Independence**.

#### 1. External/Conceptual Mapping (Logical Data Independence)
* **Mechanism:** Defines the exact structural translation between an individual user's External View and the global Conceptual Schema.
* **Independence Benefit:** If the macro logical structure of the database changes (e.g., adding a new attribute column or splitting a global table), the DBA simply adjusts this mapping layer. The user's external schema remains untouched. Because front-end application programs are written directly against the external view, **application code does not break**, achieving **Logical Data Independence**.

#### 2. Conceptual/Internal Mapping (Physical Data Independence)
* **Mechanism:** Links the global logical data definitions directly to the physical files and indexes on disk.
* **Independence Benefit:** If the physical storage layout is modified (e.g., migrating data to new hard drives, changing file block sizes, or creating new B-Tree indexes), the DBA updates this internal mapping. The global conceptual schema remains completely unchanged. Consequently, user views and front-end application workflows are unaffected, achieving **Physical Data Independence**.

---

## 4. The Relational Database Model

### Formal Definitions and Structure
The Relational Model represents all data logically as mathematical relation structures, which present to users as clear, flat, two-dimensional tables.

* **Domain:** A set of atomic values sharing an identical data type (e.g., the domain of integers, or a domain of valid names).
* **Cartesian Product:** An operation across a sequence of domains ($D_1, D_2, \dots, D_n$) that constructs a comprehensive set of all possible ordered combinations of those values:
  $$D_1 \times D_2 \times \dots \times D_n = \{(d_1, d_2, \dots, d_n) \mid d_i \in D_i, i = 1, 2, \dots, n\}$$
  * **Tuple:** An individual ordered row element $(d_1, d_2, \dots, d_n)$ generated within a Cartesian product. An $n$-attribute row is called an $n$-tuple.
  * **Component (Value):** A specific, singular value $d_i$ located inside a tuple.
  * **Cardinality:** The total number of distinct rows generated by the product, computed as:
    $$M = \prod_{i=1}^{n} m_i$$
* **Relation:** A mathematical relation $R$ is a finite subset of the Cartesian product of its underlying domains. It maps directly to a standard database table layout:
  $$R(D_1, D_2, \dots, D_n) \subseteq D_1 \times D_2 \times \dots \times D_n$$

---

### Core Structural Constraints and Keys

* **Candidate Key:** An attribute or minimal group of attributes whose combined values uniquely identify a single tuple within a relation, such that no proper subset can achieve this identification.
* **Primary Key:** The single, definitive candidate key chosen by the database designer to act as the primary structural identifier for rows within the table.
* **All-Key (Superkey):** In extreme cases where no subset of columns can uniquely identify a row, the entire set of attributes combined forms the key, known as an **All-Key**.
* **Primary Attribute vs. Non-Primary Attribute:** Attributes that form part of *any* candidate key are called primary attributes; columns that do not participate in any candidate key are non-primary attributes.

#### Structural Properties of Base Relations:
To be considered a valid relation, a database table must satisfy the following strict rules:
1. **Homogeneous Columns:** Every value within a specific column must originate from the exact same underlying domain.
2. **Distinct Columns:** Each column must have a unique attribute name within that table.
3. **Unordered Columns:** The horizontal order of columns is mathematically irrelevant.
4. **Unique Rows:** No two tuples within a base relation can share identical values across all columns; every row must be unique via its key.
5. **Unordered Rows:** The vertical sorting order of rows does not alter the logical validity of the relation.
6. **First Normal Form (1NF) Obligation:** Every individual attribute value must be completely **atomic**. Multivalued fields, embedded arrays, or nested tables are strictly prohibited.

---

### Relational Schema vs. Relation Instance
* **Relational Schema:** The structural definition of a relation, written as:
  $$R(U, D, DOM, F)$$
  Where $R$ is the relation name, $U$ is the set of attributes, $D$ is the underlying domains, $DOM$ maps attributes to domains, and $F$ defines data dependencies. It represents the fixed structural layout.
* **Relation Instance:** The actual collection of tuples occupying that schema's rows at a single moment. It changes dynamically as data is modified over time.

### Relational Database Languages
Relational languages are categorized by their underlying mathematical paradigm:
* **Relational Algebra:** A procedural query language where operations are applied to relations to compute a result relation.
* **Relational Calculus:** A non-procedural query language based on first-order predicate logic. It specifies *what* data to retrieve rather than *how* to retrieve it, split into Tuple Relational Calculus and Domain Relational Calculus.
* **SQL (Structured Query Language):** The unified industry standard that integrates Data Definition Language (DDL), Data Manipulation Language (DML), and Data Control Language (DCL). It combines principles from both relational algebra and relational calculus.

---

## 5. Relational Integrity Constraints

The relational model enforces three categories of integrity constraints to guarantee data accuracy and prevent invalid inputs.

### 1. Entity Integrity
* **Rule:** If an attribute $A$ is a component of the Primary Key of a base relation $R$, then $A$ **cannot accept null values (NULL)**.
* **Purpose:** Every tuple within a relation represents a distinct real-world entity and must be uniquely identifiable. Allowing a primary key to be null would break this identifier mechanism, compromising data integrity.

### 2. Referential Integrity
* **Rule:** Defines relationships between tables via Foreign Keys. If an attribute set $F$ in relation $R$ is a **Foreign Key** pointing to the Primary Key $K_s$ of target relation $S$, then for every tuple in $R$, the value of $F$ must either:
  1. Match an existing Primary Key value in some row of target table $S$.
  2. Be entirely **NULL** (if business logic permits).



### 3. User-Defined Integrity
* **Rule:** Custom constraints defined by database designers to enforce specific real-world business logic (e.g., setting a rule that an `Age` integer column must fall within the range `18` to `65`, or forcing an `Email` field to contain an `@` symbol).

---

## 6. Relational Algebra Operations

Relational algebra is a mathematical query framework that takes relations as inputs and returns a relation as output. It is divided into traditional set operations and specialized relational operations.

### Traditional Set Operations (Binary Operators)
These operations require the input relations to be **Union-Compatible** (meaning they must have the exact same number of attributes, and their corresponding attributes must share identical domains).

* **Union ($R \cup S$):** Combines all tuples from both relations, automatically eliminating duplicate rows:
  $$R \cup S = \{ t \mid t \in R \vee t \in S\}$$
* **Difference ($R - S$):** Extracts tuples that reside in relation $R$ but do not exist in relation $S$:
  $$R - S = \{ t \mid t \in R \wedge t \notin S\}$$
* **Intersection ($R \cap S$):** Extracts only the common tuples that reside simultaneously in both $R$ and $S$:
  $$R \cap S = \{ t \mid t \in R \wedge t \in S\}$$
* **Cartesian Product ($R \times S$):** Combines every row of relation $R$ with every row of relation $S$, constructing an extended tuple space. If $R$ has $n$ attributes and $S$ has $m$ attributes, the result relation will contain $n+m$ attributes.

#### Concrete Example Walkthrough:
Given the following sample relations, $R$ and $S$:

$$\large{R}$$
| A | B | C |
| :---: | :---: | :---: |
| $a_1$ | $b_1$ | $c_1$ |
| $a_1$ | $b_2$ | $c_2$ |
| $a_2$ | $b_2$ | $c_1$ |

$$\large{S}$$
| A | B | C |
| :---: | :---: | :---: |
| $a_1$ | $b_2$ | $c_2$ |
| $a_1$ | $b_3$ | $c_2$ |
| $a_2$ | $b_2$ | $c_1$ |

---

$$\large{R \cup S \text{ (Union Outputs)}}$$
| A | B | C |
| :---: | :---: | :---: |
| $a_1$ | $b_1$ | $c_1$ |
| $a_1$ | $b_2$ | $c_2$ |
| $a_2$ | $b_2$ | $c_1$ |
| $a_1$ | $b_3$ | $c_2$ |

$$\large{R \cap S \text{ (Intersection Outputs)}}$$
| A | B | C |
| :---: | :---: | :---: |
| $a_1$ | $b_2$ | $c_2$ |
| $a_2$ | $b_2$ | $c_1$ |

$$\large{R \times S \text{ (Cartesian Product Outputs)}}$$
| R.A | R.B | R.C | S.A | S.B | S.C |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $a_1$ | $b_1$ | $c_1$ | $a_1$ | $b_2$ | $c_2$ |
| $a_1$ | $b_1$ | $c_1$ | $a_1$ | $b_3$ | $c_2$ |
| $a_1$ | $b_1$ | $c_1$ | $a_2$ | $b_2$ | $c_1$ |
| $a_1$ | $b_2$ | $c_2$ | $a_1$ | $b_2$ | $c_2$ |
| $a_1$ | $b_2$ | $c_2$ | $a_1$ | $b_3$ | $c_2$ |
| $a_1$ | $b_2$ | $c_2$ | $a_2$ | $b_2$ | $c_1$ |
| $a_2$ | $b_2$ | $c_1$ | $a_1$ | $b_2$ | $c_2$ |
| $a_2$ | $b_2$ | $c_1$ | $a_1$ | $b_3$ | $c_2$ |
| $a_2$ | $b_2$ | $c_1$ | $a_2$ | $b_2$ | $c_1$ |

---

### Specialized Relational Operations

#### 1. Selection ($\sigma$)
Filters horizontal rows from a relation based on a logical predicate condition $F$:
$$\sigma_{F}(R) = \{t \mid t \in R \wedge F(t) = \text{True}\}$$
* The condition $F$ typically uses comparison operators ($\theta \in \{>, <, =, \neq\}$) to evaluate attribute values.

#### 2. Projection ($\pi$)
Extracts vertical columns from a relation, discarding the unselected attributes. Because projection can create duplicate rows, any identical resulting tuples are automatically combined to preserve set properties:
$$\pi_{A}(R) = \{t[A] \mid t \in R\}$$
* Where $A$ represents the targeted group of attribute column names.

##### Projection Example:
Projecting the `Sdept` column from the sample `Student` relation:

$$\large{\text{Source Table: } Student}$$
| Sname | Sdept |
| :---: | :---: |
| Nick | CS |
| Cay | CS |
| John | MA |
| West | IS |

$$\large{\text{Resulting Projection Table: } \pi_{Sdept}(Student)}$$
| Sdept |
| :---: |
| CS |
| MA |
| IS |

Notice that the duplicate `CS` entry from the second row was automatically merged into a single unique row.

#### 3. Theta Join ($\bowtie_{\theta}$)
Combines related rows from two separate relations based on a comparison condition between their attributes. It is logically equivalent to performing a full Cartesian Product followed by a Selection filter:
$$R \bowtie_{A \, \theta \, B} S = \{\widehat{t_r t_s} \mid t_r \in R \wedge t_s \in S \wedge t_r[A] \, \theta \, t_s[B]\}$$

#### 4. The Concept of an Image Set ($Z_x$)
Given a relation $R(X, Z)$, where $X$ and $Z$ represent sets of attributes, let $x$ be a specific value combination for attribute group $X$. The **Image Set** $Z_x$ contains all values for attribute group $Z$ that appear alongside $x$ within relation $R$:
$$Z_x = \{t[Z] \mid t \in R \wedge t[X] = x\}$$

##### Image Set Example:
Given the following relation $R(X, Z)$:

$$\large{R}$$
| X | Z |
| :---: | :---: |
| $x_1$ | $z_1$ |
| $x_1$ | $z_2$ |
| $x_1$ | $z_3$ |
| $x_2$ | $z_2$ |
| $x_2$ | $z_3$ |
| $x_3$ | $z_1$ |
| $x_3$ | $z_3$ |

The computed image sets for each distinct $X$ value are:
* For $x_1$: $Z_{x_1} = \{z_1, z_2, z_3\}$
* For $x_2$: $Z_{x_2} = \{z_2, z_3\}$
* For $x_3$: $Z_{x_3} = \{z_1, z_3\}$

