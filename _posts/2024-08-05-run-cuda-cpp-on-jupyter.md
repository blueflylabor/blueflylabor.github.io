---
title: run cuda cpp on jupyter
date:   2024-08-05
last_modified_at: 2024-08-05
categories: [cuda, cpp, jupyter, thread]
---
### Pre-Install
- For Windows Users  
*Install N-GPU drivers*
- Ubuntu22.04 WSL Users  
*[WSL CUDA Drivers Install](https://blueflylabor.github.io/2021/04/06/%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AEUbuntu1804%E5%AE%89%E8%A3%85CUDA%E5%92%8CPytorch)*
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