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
    hello<<<2, 2>>>();
    cudaDeviceSynchronize();
}
```

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