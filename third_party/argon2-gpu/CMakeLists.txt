cmake_minimum_required(VERSION 3.7)

project(argon2-gpu CXX)
set(BINARY_INSTALL_DIR bin)
set(LIBRARY_INSTALL_DIR lib)
set(INCLUDE_INSTALL_DIR include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

option(ARGON2_GPU_BUILD_OPENCL "Build OpenCL backend" OFF)
option(ARGON2_GPU_BUILD_TOOLS "Build test/bench tools" OFF)

set(CUDA_FOUND FALSE)

if(NOT NO_CUDA)
    find_package(CUDA)
endif()

if(CUDA_FOUND)
    message("INFO: Using CUDA version ${CUDA_VERSION}")
    add_definitions(-DHAVE_CUDA=1)
else()
    message("INFO: Building without CUDA support")
    add_definitions(-DHAVE_CUDA=0)
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;-O3;--ptxas-options=-v;-arch sm_89;-lineinfo
    )
endif()

set(ARGON2_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/argon2)

add_library(argon2-gpu-common STATIC
    lib/argon2-gpu-common/argon2params.cpp
    lib/argon2-gpu-common/blake2b.cpp
)
target_include_directories(argon2-gpu-common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-gpu-common PRIVATE
    include/argon2-gpu-common
    lib/argon2-gpu-common
)

if(CUDA_FOUND)
    cuda_add_library(argon2-cuda STATIC
        lib/argon2-cuda/device.cpp
        lib/argon2-cuda/globalcontext.cpp
        lib/argon2-cuda/programcontext.cpp
        lib/argon2-cuda/processingunit.cpp
        lib/argon2-cuda/kernels.cu
    )
else()
    add_library(argon2-cuda STATIC
        lib/argon2-cuda/nocuda.cpp
    )
endif()

target_include_directories(argon2-cuda PRIVATE
    include/argon2-cuda
    lib/argon2-cuda
)
target_include_directories(argon2-cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(argon2-cuda argon2-gpu-common)

if(ARGON2_GPU_BUILD_OPENCL)
    add_library(argon2-opencl STATIC
        lib/argon2-opencl/device.cpp
        lib/argon2-opencl/globalcontext.cpp
        lib/argon2-opencl/kernelloader.cpp
        lib/argon2-opencl/programcontext.cpp
        lib/argon2-opencl/processingunit.cpp
        lib/argon2-opencl/kernelrunner.cpp
    )
    target_include_directories(argon2-opencl INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(argon2-opencl PRIVATE
        include/argon2-opencl
        lib/argon2-opencl
    )
    target_link_libraries(argon2-opencl
        argon2-gpu-common -lOpenCL
    )
endif()

if(ARGON2_GPU_BUILD_TOOLS)
    add_executable(argon2-gpu-test
        src/argon2-gpu-test/main.cpp
        src/argon2-gpu-test/testcase.cpp
    )
    target_include_directories(argon2-gpu-test PRIVATE src/argon2-gpu-test)
    target_link_libraries(argon2-gpu-test
        argon2-cuda argon2-opencl argon2 -lOpenCL
    )

    add_executable(argon2-gpu-bench
        src/argon2-gpu-bench/cpuexecutive.cpp
        src/argon2-gpu-bench/cudaexecutive.cpp
        src/argon2-gpu-bench/openclexecutive.cpp
        src/argon2-gpu-bench/benchmark.cpp
        src/argon2-gpu-bench/main.cpp
    )
    target_include_directories(argon2-gpu-bench PRIVATE src/argon2-gpu-bench)
    target_link_libraries(argon2-gpu-bench
        argon2-cuda argon2-opencl argon2 -lOpenCL
    )

    add_test(argon2-gpu-test-opencl argon2-gpu-test -m opencl)
    add_test(argon2-gpu-test-cuda argon2-gpu-test -m cuda)
endif()

install(
    TARGETS argon2-gpu-common argon2-cuda
    DESTINATION ${LIBRARY_INSTALL_DIR}
)
if(ARGON2_GPU_BUILD_OPENCL)
    install(
        TARGETS argon2-opencl
        DESTINATION ${LIBRARY_INSTALL_DIR}
    )
endif()
install(FILES
    include/argon2-gpu-common/argon2-common.h
    include/argon2-gpu-common/argon2params.h
    include/argon2-cuda/cudaexception.h
    include/argon2-cuda/kernels.h
    include/argon2-cuda/device.h
    include/argon2-cuda/globalcontext.h
    include/argon2-cuda/programcontext.h
    include/argon2-cuda/processingunit.h
    DESTINATION ${INCLUDE_INSTALL_DIR}
)
if(ARGON2_GPU_BUILD_OPENCL)
    install(FILES
        include/argon2-opencl/cl.hpp
        include/argon2-opencl/opencl.h
        include/argon2-opencl/device.h
        include/argon2-opencl/globalcontext.h
        include/argon2-opencl/programcontext.h
        include/argon2-opencl/processingunit.h
        include/argon2-opencl/kernelrunner.h
        DESTINATION ${INCLUDE_INSTALL_DIR}
    )
endif()
if(ARGON2_GPU_BUILD_TOOLS)
    install(
        TARGETS argon2-gpu-bench argon2-gpu-test
        DESTINATION ${BINARY_INSTALL_DIR}
    )
endif()
