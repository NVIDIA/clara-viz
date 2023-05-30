# Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.12)

macro(setup_project)
    # avoid 'Up-to-date' install messages
    set(CMAKE_INSTALL_MESSAGE LAZY)

    # Ensure user selected a build type
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
    endif()

    # CUDA settings

    # use the stub libs instead of real binaries, with that building in docker without using the nvidia runtime works
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda/lib64/stubs)

    # Generate code for multiple architectures, if we don't do this the JIT compiler will compile SM specific
    # code from the embedded PTX code. This will increase program startup times.
    # Pascal (60, 61), Volta (70), Turing (75), Ampere (80)
    set(CMAKE_CUDA_ARCHITECTURES 60-real 61-real 70-real 75-real 80)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    # allow calling constexpr host function from device code and vice versa, used to call std::numeric_limits functions from device
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


    if (CLARA_VIZ_CUDA_DEBUG)
        # enable for device debugging in debug mode
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --device-debug")
    endif()

    if (CLARA_VIZ_CUDA_PROFILING)
        # enable for profiling
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
    endif()

    # statically link the runtime
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

    # enable c++17
    set(CMAKE_CXX_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    # set flags for coverage builds
    set(CMAKE_CXX_FLAGS_COVERAGE
        "-DNDEBUG"
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
        FORCE)
    set(CMAKE_C_FLAGS_COVERAGE
        "-DNDEBUG"
        CACHE STRING "Flags used by the C compiler during coverage builds."
        FORCE)
    MARK_AS_ADVANCED(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_C_FLAGS_COVERAGE)

    # setup nvsharedmemory
    find_package(nvsharedmemory CONFIG)
    if(nvsharedmemory_FOUND)
        add_compile_definitions(CLARA_VIZ_USE_NVSHAREDMEMORY)
    endif()

endmacro()
