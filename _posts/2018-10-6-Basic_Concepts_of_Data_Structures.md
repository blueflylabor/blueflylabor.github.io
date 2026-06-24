---
title: "Basic Concepts of Data Structures"
date:   2018-10-06
last_modified_at: 2018-10-06
categories: notes
tags: [Data Structures]
lang: en
---

# Basic Concepts of Data Structures

### 1. Data
The structural collection of facts, figures, concepts, or characters managed and processed by a computer system.

### 2. Data Element
The fundamental unit of data. A data element can consist of several **data items**. A data item is the smallest, indivisible unit of data.

### 3. Data Type
A collection of values and a set of operations defined on those values.

### 4. Abstract Data Type (ADT)
A mathematical model implemented in a computer system, which specifies a collection of data objects, the relationships among them, and a set of basic operations (e.g., modeling a Finite State Machine).

### 5. Data Structure
The specific structural relationship existing among different data elements. A data structure encompasses three core aspects: **Logical Structure**, **Storage Structure**, and **Data Operations** ($\text{Program} = \text{Algorithm} + \text{Data Structure}$).

### 6. Logical Structure
The conceptual relationships between data items, independent of how they are physically stored in memory. It is broadly categorized into linear and non-linear configurations.

```mermaid
graph TD
Logical_Structure[Logical Structure] --> Linear_Structure[Linear Structure]
Logical_Structure --> NonLinear_Structure[Non-Linear Structure]
Linear_Structure --> General_Linear_List[General Linear List]
Linear_Structure --> Restricted_Linear_List[Restricted Linear List]
Linear_Structure --> Generalized_Linear_Extension[Linear Extension]
Restricted_Linear_List --> Stacks_and_Queues[Stacks and Queues]
Restricted_Linear_List --> Strings[Strings]
Generalized_Linear_Extension --> Arrays[Arrays]
Generalized_Linear_Extension --> Generalized_Lists[Generalized Lists]
NonLinear_Structure --> NonLinear_Tables[Non-Linear Tables]
NonLinear_Tables --> Sets[Sets]
NonLinear_Tables --> Tree_Structures[Tree Structures]
NonLinear_Tables --> Graph_Structures[Graph Structures]
Tree_Structures --> General_Trees[General Trees]
Tree_Structures --> Binary_Trees[Binary Trees]
Graph_Structures --> Directed_Graphs[Directed Graphs]
Graph_Structures --> Undirected_Graphs[Undirected Graphs]