```python
from gadget import gadget, undo_magic_name
from IPython.core.getipython import get_ipython
ipython = get_ipython()
ipython.register_magic_function(gadget, 'cell')
```


```python
%%gadget -c hello.cpp

#include<cstdio>

void hello(){
    printf("hello.\n");
}
```

    文件 hello.cpp 创建成功，内容已写入。



```python
%%gadget -c main.cpp

#include<cstdio>

void hello();

int main(){
    hello();
    return 0;
}
```

    文件 main.cpp 创建成功，内容已写入。



```python
!g++ -c hello.cpp -o hello.o
!g++ -c main.cpp -o main.o
!g++ hello.o main.o -o a.out
```


```python
!./a.out
```

    hello.


- import head file for once

```c++
#pragma once
```

- submodule setting

```cmake
add_subdirectory(*submodule_directory*)
```

- glm  (math lib)


```python
%%gadget -c glm_test.cpp

#include<glm/vec3.hpp>
#include<iostream>

inline std::ostream &operator<<(std::ostream &os, glm::vec3 const &v){
    return os << v.x << " " << v.y << " " << v.z;
}

int main(){
    glm::vec3 v(1, 2, 3);
    v += 1;
    std::cout << v << std::endl;
    return 0;
}
```

    文件 glm_test.cpp 创建成功，内容已写入。



```python
%%gadget -c glm_test_.cpp

#include<glm/vec3.hpp>
#include<iostream>

int main(){
    glm::vec3 v(1, 2, 3);
    v += 1;
    std::cout << v.x << " " << v.y << " " << v.z << std::endl;
    return 0;
}
```

    文件 glm_test_.cpp 创建成功，内容已写入。



```python
%%gadget -c CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(glm_test LANGUAGES CXX)
add_executable(a.out glm_test.cpp)
find_package(glm CONFIG REQUIRED)
target_link_libraries(a.out PRIVATE glm::glm)
```

    文件 CMakeLists.txt 创建成功，内容已写入。



```python
!git clone https://github.com//g-truc/glm.git
```

    fatal: destination path 'glm' already exists and is not an empty directory.



```python
!cmake -B build
!cmake --build build --target a.out
!build/a.out
```

    -- Configuring done
    -- Generating done
    -- Build files have been written to: /mnt/c/Users/Jeff/Desktop/cpp_course_from_scratch/build
    gmake: Warning: File 'Makefile' has modification time 1.1 s in the future
    gmake[1]: Warning: File 'CMakeFiles/Makefile2' has modification time 1.2 s in the future
    gmake[2]: Warning: File 'CMakeFiles/Makefile2' has modification time 1 s in the future
    gmake[3]: Warning: File 'CMakeFiles/a.out.dir/progress.make' has modification time 0.99 s in the future
    [35m[1mConsolidate compiler generated dependencies of target a.out[0m
    gmake[3]: warning:  Clock skew detected.  Your build may be incomplete.
    gmake[3]: Warning: File 'CMakeFiles/a.out.dir/progress.make' has modification time 0.5 s in the future
    [ 50%] [32mBuilding CXX object CMakeFiles/a.out.dir/glm_test.cpp.o[0m
    [100%] [32m[1mLinking CXX executable a.out[0m
    gmake[3]: warning:  Clock skew detected.  Your build may be incomplete.
    [100%] Built target a.out
    gmake[2]: warning:  Clock skew detected.  Your build may be incomplete.
    gmake[1]: warning:  Clock skew detected.  Your build may be incomplete.
    gmake: warning:  Clock skew detected.  Your build may be incomplete.
    2 3 4



```python
!ls glm/glm
```

    CMakeLists.txt	 fwd.hpp	integer.hpp  mat3x4.hpp   simd
    common.hpp	 geometric.hpp	mat2x2.hpp   mat4x2.hpp   trigonometric.hpp
    detail		 glm.cppm	mat2x3.hpp   mat4x3.hpp   vec2.hpp
    exponential.hpp  glm.hpp	mat2x4.hpp   mat4x4.hpp   vec3.hpp
    ext		 gtc		mat3x2.hpp   matrix.hpp   vec4.hpp
    ext.hpp		 gtx		mat3x3.hpp   packing.hpp  vector_relational.hpp



```python
!cmake .
```

    -- Configuring done
    -- Generating done
    -- Build files have been written to: /mnt/c/Users/Jeff/Desktop/cpp_course_from_scratch



```python
!./a.out
```

    hello.


- fmt (alternative for std::format)


```python
!git clone https://github.com/fmtlib/fmt.git
```

    fatal: destination path 'fmt' already exists and is not an empty directory.



```python
%%gadget -c fmt_test.cpp

#include <fmt/core.h>

int main() {
  fmt::print("Hello, {}!\n", 42);
  return 0;
}
```

    文件 fmt_test.cpp 创建成功，内容已写入。



```python
%%gadget -c CMakeLists.txt

# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(FmtTestProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 fmt 库
find_package(fmt REQUIRED)

# 添加可执行文件
add_executable(fmt_test fmt_test.cpp)

# 链接 fmt 库到可执行文件
target_link_libraries(fmt_test PRIVATE fmt::fmt)
```

    文件 CMakeLists.txt 创建成功，内容已写入。



```python
!cmake -B build
!cmake --build build --target fmt_test
!build/fmt_test
```

    -- The C compiler identification is GNU 11.4.0
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working C compiler: /bin/cc - skipped
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /mnt/c/Users/Jeff/Desktop/cpp_course_from_scratch/build
    [ 50%] [32mBuilding CXX object CMakeFiles/fmt_test.dir/fmt_test.cpp.o[0m
    [100%] [32m[1mLinking CXX executable fmt_test[0m
    [100%] Built target fmt_test
    Hello, 42!


- 可以通过 find_package 命令寻找系统中的包/库：  
find_package(fmt REQUIRED)
target_link_libraries(myexec PUBLIC fmt::fmt)
- 为什么是 fmt::fmt 而不是简单的 fmt？  
现代 CMake 认为一个包 (package) 可以提供多个库，又称组件 (components)，比如 TBB 这个包，就包含了 tbb, tbbmalloc, tbbmalloc_proxy 这三个组件。
因此为避免冲突，每个包都享有一个独立的名字空间，以 :: 的分割（和 C++ 还挺像的）。
- 你可以指定要用哪几个组件：  
find_package(TBB REQUIRED COMPONENTS tbb tbbmalloc REQUIRED)
target_link_libraries(myexec PUBLIC TBB::tbb TBB::tbbmalloc)
