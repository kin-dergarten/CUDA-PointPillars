# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.1)
project(pointpillar)
set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

set(POINTPILLAR_TARGET "auto" CACHE STRING "Jetson target: tx2nx, orinnx, or auto")
set_property(CACHE POINTPILLAR_TARGET PROPERTY STRINGS auto tx2nx orinnx)
if(POINTPILLAR_TARGET STREQUAL "tx2nx")
    set(POINTPILLAR_CUDA_ARCH 62)
elseif(POINTPILLAR_TARGET STREQUAL "orinnx")
    set(POINTPILLAR_CUDA_ARCH 87)
elseif(POINTPILLAR_TARGET STREQUAL "auto")
    if(NOT "$ENV{CUDASM}" STREQUAL "")
        set(POINTPILLAR_CUDA_ARCH "$ENV{CUDASM}")
    else()
        message(FATAL_ERROR "Set CUDASM or configure with -DPOINTPILLAR_TARGET=tx2nx|orinnx")
    endif()
else()
    message(FATAL_ERROR "Unsupported POINTPILLAR_TARGET '${POINTPILLAR_TARGET}'; use tx2nx, orinnx, or auto")
endif()

if(NOT CMAKE_CXX_STANDARD)
    if(POINTPILLAR_TARGET STREQUAL "orinnx" OR
       (POINTPILLAR_TARGET STREQUAL "auto" AND POINTPILLAR_CUDA_ARCH STREQUAL "87"))
        set(CMAKE_CXX_STANDARD 17)
    else()
        set(CMAKE_CXX_STANDARD 14)
    endif()
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-O0 -g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    --std=c++${CMAKE_CXX_STANDARD}
    -gencode arch=compute_${POINTPILLAR_CUDA_ARCH},code=compute_${POINTPILLAR_CUDA_ARCH}
)

include_directories(
    $ENV{CUDA_Inc}
    $ENV{TensorRT_Inc}
    src
	src/pointpillar
)

link_directories(
    $ENV{CUDA_Lib}
    $ENV{TensorRT_Lib}
    build
)

file(GLOB_RECURSE CORE_FILES 
    src/pointpillar/*.cu 
    src/pointpillar/*.cpp
    src/common/tensor.cu
    src/common/tensorrt.cpp
)

####################### libpointpillar_core.so ##########################
cuda_add_library(pointpillar_core SHARED 
  	${CORE_FILES}
)

target_link_libraries(pointpillar_core
	libnvinfer.so
)
#######################################################################


####################### pointpillar program #############################
cuda_add_executable(pointpillar 
	src/main.cpp
)

target_link_libraries(pointpillar
	pointpillar_core
)
#######################################################################
