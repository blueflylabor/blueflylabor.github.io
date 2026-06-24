---
layout: post
title: "Some knowledge about go routine and channel"
categories: archive
lang: en
permalink: /en/:title/
tags:
  - go
  - routines
  - channel
---


## I. Basic Concepts of Channel Buffers

A channel buffer is a memory region used for temporary data storage. The buffer capacity is defined when the channel is initialized using `make(chan Type, capacity)`.

* **Unbuffered Channels:** Created via `make(chan Type)` or `make(chan Type, 0)`. The buffer capacity is 0.
* **Buffered Channels:** Created via `make(chan Type, N)` (where `N > 0`). The buffer can hold up to `N` elements.

---

## II. Unbuffered vs. Buffered Channels: Core Differences

| Feature | Unbuffered Channel (Capacity 0) | Buffered Channel (Capacity N > 0) |
| --- | --- | --- |
| **Send Operation** | Must wait for a receiver to be ready; blocks otherwise (Synchronous). | Non-blocking if the buffer has space; blocks if the buffer is full (Asynchronous). |
| **Receive Operation** | Must wait for a sender to be ready; blocks otherwise (Synchronous). | Non-blocking if the buffer is not empty; blocks if the buffer is empty (Synchronous). |
| **Synchronicity** | Strong synchronization: Requires a "handshake" between sender and receiver. | Weak synchronization: Decoupled via the buffer. |
| **Use Cases** | Strict goroutine synchronization (e.g., signaling, ordering). | Balancing producer-consumer speeds (e.g., task queues, traffic smoothing). |

---

## III. Buffer Operation Examples

### 1. Unbuffered Channel (Deadlock & Correct Usage)

```go
package main

import "fmt"

func main() {
    // Unbuffered channel
    ch := make(chan int)

    // Incorrect: Deadlock occurs if sending without a receiver
    // ch <- 10 

    // Correct: Launch receiver goroutine before sending
    go func() {
        fmt.Println("Received:", <-ch) // Waits for sender
    }()
    
    ch <- 10 // Sends and waits for receiver to finish (synchronous)
    fmt.Println("Send complete")
}

```

**Output:**

```text
Received: 10
Send complete

```

### 2. Buffered Channel (Full/Not Full Scenarios)

```go
package main

import "fmt"

func main() {
    // Buffered channel with capacity 2
    ch := make(chan int, 2)

    // Buffer not full, send does not block
    ch <- 10
    ch <- 20

    // Check buffer state (length and capacity)
    fmt.Println("Length:", len(ch), "Capacity:", cap(ch)) // Length: 2, Capacity: 2

    // Buffer is full: sending more would block (commented out to prevent deadlock)
    // ch <- 30

    // Receiving frees up space in the buffer
    fmt.Println("Received:", <-ch) // Received: 10
    fmt.Println("Length:", len(ch)) // Length: 1

    // Can now send another value
    ch <- 30
    fmt.Println("Length:", len(ch)) // Length: 2
}

```

---

## IV. Critical Considerations for Buffers

1. **Deadlock Risks:**
* **Unbuffered:** Sending without a receiver (or receiving without a sender) results in a deadlock.
* **Buffered:** Sending to a full buffer or receiving from an empty buffer will cause the goroutine to block indefinitely if not managed correctly.


2. **Length vs. Capacity:**
* `len(ch)`: Returns the current number of elements in the buffer.
* `cap(ch)`: Returns the total maximum capacity defined at initialization.


3. **Channel Closure:**
* Closing allows continued reception of existing buffered data; after the buffer is drained, further receives return the zero-value and `false`.
* **Panic:** Sending to a closed channel will trigger a runtime panic.


4. **Performance:**
* Buffered channels reduce goroutine context-switching overhead by decoupling producers from consumers. However, choose capacity wisely: excessive capacity wastes memory, while insufficient capacity causes frequent blocking.



---
