---
categories: archive
lang: en
layout: post
ref: 2025-08-09-some-knowledge-about-go-routine-and-channel
title: Some knowledge about Go routines and channels
---

### One, Basic Concepts of Channel Buffers

A channel buffer is a memory area used to temporarily store data. When creating a channel using `make(chan Type, capacity)`, the capacity (capacity) specifies the size of the buffer.

- **Unbuffered Channels**: `make(chan Type)` or `make(chan Type, 0)`, the buffer capacity is 0.
- **Buffered Channels**: `make(chan Type, N)` (N > 0), the buffer can store a maximum of N elements.

### Two, Core Differences Between Unbuffered and Buffered Channels

| Feature              | Unbuffered Channel (Capacity 0)                        | Buffered Channel (Capacity N > 0)                      |
|----------------------|-------------------------------------------------------|-------------------------------------------------------|
| **Send Operation**   | Must wait for the receiver to be ready, otherwise blocks (synchronous operation) | Can send directly if the buffer is not full, otherwise blocks (asynchronous operation until there's space) |
| **Receive Operation**| Must wait for the sender to be ready, otherwise blocks (synchronous operation) | Can receive directly if the buffer is not empty, otherwise blocks (until data is available)       |
| **Synchronicity**    | Strong synchronization: Sending and receiving must "handshake" to complete | Weak synchronization: Depends on buffer status, can temporarily store data              |
| **Suitable Scenarios**| Used for strict synchronization between goroutines (e.g., passing signals, ensuring order) | Used to balance the speed of producers and consumers (e.g., task queues, traffic smoothing)      |

### Three, Example of Buffer Operations
#### 1. Unbuffered Channel (Deadlock and Correct Usage)
```go
package main

import "fmt"

func main() {
    // 无缓冲通道
    ch := make(chan int)

    // 错误：主 goroutine 发送后无接收方，死锁
    // ch <- 10 

    // 正确：启动接收 goroutine 后再发送
    go func() {
        fmt.Println("接收:", <-ch) // 等待发送方
    }()
    ch <- 10 // 发送后等待接收方完成（同步）
    fmt.Println("发送完成")
}
```
**Output:**
```
接收: 10
发送完成
```
#### 2. With Buffering Channel (Buffer is not full/full situation)
```go
package main

import "fmt"

func main() {
    // 容量为 2 的带缓冲通道
    ch := make(chan int, 2)

    // 缓冲区未满，发送不阻塞
    ch <- 10
    ch <- 20

    // 查看缓冲区状态（长度和容量）
    fmt.Println("长度:", len(ch), "容量:", cap(ch)) // 长度: 2 容量: 2

    // 缓冲区已满，再发送会阻塞（注释掉避免死锁）
    // ch <- 30

    // 接收数据，缓冲区腾出空间
    fmt.Println("接收:", <-ch) // 接收: 10
    fmt.Println("长度:", len(ch)) // 长度: 1

    // 此时可再发送一个数据（缓冲区有空间）
    ch <- 30
    fmt.Println("长度:", len(ch)) // 长度: 2
}
```
### Four, Key Considerations for Buffers
1.  **Deadlock Risk**:
    -   Unbuffered channels: If there is no receiver (or sender) after sending, it will cause a deadlock.
    -   Buffered channels: If data is sent when the buffer is full, or if data is received when the buffer is empty, it will also cause a deadlock.

2.  **Length and Capacity**:
    -   `len(ch)`: Returns the number of elements in the current buffer.
    -   `cap(ch)`: Returns the maximum capacity of the buffer (specified when created).

3.  **Impact of Closing a Channel**:
    -   After closing, you can continue to receive data from the buffer, and it will return 0 and `false` after receiving all data.
    -   After closing, you cannot send any more data (it will cause a panic).

4.  **Performance Considerations**:
    -   Buffered channels can reduce goroutine switching overhead (by temporarily storing data in the buffer), but you need to set the capacity reasonably (too large will waste memory, and too small may still frequently block).

### Five, Summary of Suitable Scenarios
-   **Unbuffered Channels**: Suitable for scenarios that require strict synchronization (such as "relay" tasks or signal notifications), ensuring that the sender and receiver keep pace with each other.
-   **Buffered Channels**: Suitable for the "producer-consumer" model, when the production speed does not match the consumption speed, through the buffer to balance the two (such as log collection, task scheduling).