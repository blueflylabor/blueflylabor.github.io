---
title: Operating Systems: Classical Process Synchronization and Mutual Exclusion
date:   2021-04-06
last_modified_at: 2026-06-23
categories: notes
tags: [Operating Systems, Concurrency]
lang: en

---

# Classical Process Synchronization Problems

This section explores the standard synchronization templates used to resolve race conditions, data corruption, and deadlocks in concurrent environments.

---

## 1. The Bounded-Buffer (Producer-Consumer) Problem

### Problem Description
* A set of **Producer** processes and a set of **Consumer** processes share a common, bounded buffer containing $n$ slots.
* Producers generate items and place them into the buffer; consumers extract items from the buffer.
* **Constraints:** 1. Producers must block if the buffer is completely full.
  2. Consumers must block if the buffer is completely empty.
  3. Only one process can modify the buffer at any given time (Mutual Exclusion).

> ⚠️ **Critical Bug Fix Note:** In the initial pseudocode, the producer invoked `P(mutex)` before `P(empty)`. If the buffer was full ($empty = 0$), the producer would grab the mutex lock, then block on `P(empty)`. Since it holds the lock, consumers can never enter to clear space—causing an immediate **Deadlock**. Always request resource semaphores *before* mutual exclusion locks.



### Corrected PV Implementation
```cpp
semaphore mutex = 1;  // Controls critical section access to the buffer
semaphore empty = n;  // Counts available empty slots (Resource semaphore)
semaphore full = 0;   // Counts populated items in the buffer (Resource semaphore)

void Producer() {
    while(true) {
        item data = Produce();
        
        wait(empty);   // Check/decrement empty slots first (Blocks if buffer full)
        wait(mutex);   // Lock the buffer boundary
        add2Buffer(data);
        signal(mutex); // Unlock buffer
        signal(full);  // Increment and signal full items
    }
}

void Consumer() {
    while(true) {
        wait(full);    // Check/decrement full items first (Blocks if buffer empty)
        wait(mutex);   // Lock the buffer boundary
        item data = getFromBuffer();
        signal(mutex); // Unlock buffer
        signal(empty); // Increment and signal empty slots
        
        Consume(data);
    }
}

```

---

## 2. The Readers-Writers Problem

### Problem Description

* Multiple reader processes and writer processes share a single document/dataset.
* **Constraints:**
1. Multiple readers can read simultaneously.
2. Only one writer can modify the data at a time; no readers can read while a write is in progress.



### Variation A: Reader-Priority (Read-Preferred)

Writers can starve if there is a continuous stream of reader processes arriving at the system.

```cpp
int count = 0;         // Tracks active readers accessing the file
semaphore mutex = 1;   // Protects modification loops of 'count'
semaphore rw = 1;      // Ensures mutual exclusion for writers / first reader

void Reader() {
    while(true) {
        wait(mutex);
        if(count == 0) {
            wait(rw);  // First reader blocks writers from entering
        }
        count++;
        signal(mutex);

        Read();

        wait(mutex);
        count--;
        if(count == 0) {
            signal(rw); // Last reader releases the block for writers
        }
        signal(mutex);
    }
}

void Writer() {
    while(true) {
        wait(rw);
        Write();
        signal(rw);
    }
}

```

### Variation B: Writer-Priority (Write-Preferred)

If a writer requests access, newly arriving readers are blocked until the writer completes its task.

```cpp
int count = 0;         // Tracks active readers
semaphore mutex = 1;   // Protects 'count' modification
semaphore rw = 1;      // Document access control
semaphore w = 1;       // Queue barrier to let writers jump ahead of oncoming readers

void Writer() {
    while(true) {
        wait(w);       // Lock queue barrier; blocks new readers
        wait(rw);      // Request document write access
        Write();
        signal(rw);
        signal(w);     // Release queue barrier
    }
}

void Reader() {
    while(true) {
        wait(w);       // Check queue barrier; blocks if a writer is waiting
        wait(mutex);   // Safely enter reader tracking section
        if(count == 0) {
            wait(rw);  // First reader locks out writers
        }
        count++;
        signal(mutex);
        signal(w);     // Release queue barrier for next reader

        Read();

        wait(mutex);
        count--;
        if(count == 0) {
            signal(rw); // Last reader unlocks the document
        }
        signal(mutex);
    }
}

```

---

## 3. The Dining Philosophers Problem

### Problem Description

* Five philosophers sit around a circular table, alternating between thinking and eating.
* There are five chopsticks available, one between each pair of adjacent philosophers.
* A philosopher needs **both** adjacent chopsticks to eat.

> ⚠️ **Deadlock Risk:** If all five philosophers sit down simultaneously and grab their respective left chopstick, every `P(Chopsticks[i])` succeeds, but everyone blocks infinitely on their right chopstick (`P(Chopsticks[(i+1)%5])`).

### Solution: Mutual Exclusion Solution

Wrapping the pick-up sequence inside a `mutex` lock ensures a philosopher can only pick up their chopsticks if *both* are available.

```cpp
semaphore Chopsticks[5] = {1, 1, 1, 1, 1};
semaphore mutex = 1;   // Restricts chopstick allocation to one philosopher at a time

void Philosopher(int i) {
    while(true) {
        Think();
        
        wait(mutex);                      // Lock chopstick allocation state
        wait(Chopsticks[i]);              // Grab left chopstick
        wait(Chopsticks[(i + 1) % 5]);    // Grab right chopstick
        signal(mutex);                    // Release allocation lock
        
        Eat();
        
        signal(Chopsticks[i]);
        signal(Chopsticks[(i + 1) % 5]);
    }
}

```

---

## 4. The Sleeping Smoker Problem

### Problem Description

* Three smokers and one agent (provider) run concurrently.
* To smoke a cigarette, three ingredients are required: **Tobacco**, **Paper**, and **Match/Glue**.
* Each smoker possesses an infinite supply of exactly one ingredient:
* Smoker 1 has Tobacco.
* Smoker 2 has Paper.
* Smoker 3 has Glue.


* The agent randomly places two ingredients on the table. The smoker who has the third ingredient collects the items, rolls a cigarette, and smokes, signaling the agent when finished via the `end` semaphore.

```cpp
int random_num = 0;
semaphore offer1 = 0;  // Represents Paper + Glue on table (For Smoker 1)
semaphore offer2 = 0;  // Represents Tobacco + Glue on table (For Smoker 2)
semaphore offer3 = 0;  // Represents Paper + Tobacco on table (For Smoker 3)
semaphore end = 0;     // Signals that the current smoker has finished

void Agent() {
    while(true) {
        random_num = get_random_choice() % 3;
        if(random_num == 0)      signal(offer1);
        else if(random_num == 1) signal(offer2);
        else                     signal(offer3);
        
        wait(end);     // Wait for the smoker to finish before placing new ingredients
    }
}

void Smoker1() { // Has Tobacco, needs Paper + Glue
    while(true) {
        wait(offer1);
        Smoke();
        signal(end);
    }
}

void Smoker2() { // Has Paper, needs Tobacco + Glue
    while(true) {
        wait(offer2);
        Smoke();
        signal(end);
    }
}

void Smoker3() { // Has Glue, needs Paper + Tobacco
    while(true) {
        wait(offer3);
        Smoke();
        signal(end);
    }
}

```

---

## 5. Practical Implementation Challenge (Extended Case)

### Problem Definition (Completed `*eg1*`)

* Three processes ($P_1, P_2, P_3$) share a bounded buffer with $N$ slots.
* **$P_1$ (Producer):** Generates numerical data (`int num`) and places it into the buffer.
* **$P_2$ (Odd Consumer):** Removes data from the buffer **only if the number is Odd**.
* **$P_3$ (Even Consumer):** Removes data from the buffer **only if the number is Even**.

### Implementation Strategy

This variation requires separate resource counters for odd and even items to ensure the correct consumer is notified.

```cpp
semaphore mutex = 1;     // Controls critical section access to the shared buffer
semaphore empty = N;     // Tracks available empty slots in the buffer
semaphore odd = 0;       // Tracks odd numbers currently in the buffer
semaphore even = 0;      // Tracks even numbers currently in the buffer

void P1() {
    while(true) {
        int num = Produce();
        
        wait(empty);     // Verify there is an empty slot available
        wait(mutex);
        putItemToBuffer(num);
        signal(mutex);
        
        if(num % 2 != 0) {
            signal(odd);  // Signal the Odd Consumer (P2)
        } else {
            signal(even); // Signal the Even Consumer (P3)
        }
    }
}

void P2() { // Odd Consumer
    while(true) {
        wait(odd);       // Block until an odd number is available
        wait(mutex);
        int num = getOddItemFromBuffer();
        signal(mutex);
        signal(empty);   // Free a slot in the buffer
        
        ConsumeOdd(num);
    }
}

void P3() { // Even Consumer
    while(true) {
        wait(even);      // Block until an even number is available
        wait(mutex);
        int num = getEvenItemFromBuffer();
        signal(mutex);
        signal(empty);   // Free a slot in the buffer
        
        ConsumeEven(num);
    }
}

```

```