---
title: run-cuda-cpp-on-jupyter
date:   2024-08-05
last_modified_at: 2024-08-05
categories: [cuda, cpp, jupyter, thread]
---

### Install
run this on jupyter(*.ipynb) files

```python
!pip3 install nvcc4jupyter
```

### Usage
load the extension to enable the magic commands:

```python
%load_ext nvcc4jupyter
```

### Run cuda test

```python
%%cuda
#include <stdio.h>

__global__ void hello(){
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main(){
    hello<<<6, 12>>>();
    cudaDeviceSynchronize();
}
```

output
Hello from block: 0, thread: 0  
Hello from block: 0, thread: 1  
Hello from block: 0, thread: 2  
Hello from block: 0, thread: 3  
Hello from block: 0, thread: 4  
Hello from block: 0, thread: 5  
Hello from block: 0, thread: 6  
Hello from block: 0, thread: 7  
Hello from block: 0, thread: 8  
Hello from block: 0, thread: 9  
Hello from block: 0, thread: 10  
Hello from block: 0, thread: 11  
Hello from block: 2, thread: 0  
Hello from block: 2, thread: 1  
Hello from block: 2, thread: 2  
Hello from block: 2, thread: 3  
Hello from block: 2, thread: 4  
Hello from block: 2, thread: 5  
Hello from block: 2, thread: 6  
Hello from block: 2, thread: 7  
Hello from block: 2, thread: 8  
Hello from block: 2, thread: 9  
Hello from block: 2, thread: 10  
Hello from block: 2, thread: 11  
Hello from block: 1, thread: 0  
...
Hello from block: 5, thread: 9  
Hello from block: 5, thread: 10  
Hello from block: 5, thread: 11

Output is truncated. 

### Other Problem

**ModuleNotFoundError: No module named 'nvcc_plugin'**

try this
```python
!pip3 install nvcc_plugin
```

or
```python
!pip3 install nvcc4jupyter==1.0.0
```