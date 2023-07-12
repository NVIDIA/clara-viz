# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

# Options
option(CLARA_VIZ_CUDA_PROFILING "Enable CUDA profiling" OFF)
option(CLARA_VIZ_CUDA_DEBUG "Enable CUDA debug" OFF)
option(CLARA_VIZ_WITH_GRPC "Build with gRPC frontend" ON)
option(CLARA_VIZ_WITH_OPENH264 "Build with OpenH264 support for encoding videos when no HW encoder is available" ON)
option(CLARA_VIZ_WITH_EXAMPLES "Build examples" ON)

include(ExternalProject)

project(ClaraViz
    DESCRIPTION "ClaraViz"
    VERSION 0.3.1
    LANGUAGES CXX
)

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
set(CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR "${PROJECT_SOURCE_DIR}/cmake")

include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Project.cmake")
setup_project()

# Build the thirdparty dependencies
add_subdirectory(thirdparty)

set(_DEPENDENCIES CudaHelper)
if (CLARA_VIZ_WITH_OPENH264)
    set(_DEPENDENCIES ${_DEPENDENCIES} OpenH264)
endif()
if (CLARA_VIZ_WITH_GRPC)
    set(_DEPENDENCIES ${_DEPENDENCIES} gRPC grpc-web)
endif()

# Build the ClaraViz project
# This is done with an external project. Thirdparty projects are also
# external projects and the only way to for CMake to correctly handle
# the dependencies is that our source is also build as an external
# project.
ExternalProject_Add(ClaraViz
    PREFIX claraviz
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
    BUILD_ALWAYS ON
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    CMAKE_CACHE_ARGS
        -DgRPC_DIR:PATH=${gRPC_DIR}
        -Dprotobuf-javascript_SOURCE_DIR:PATH=${protobuf-javascript_SOURCE_DIR}
        -Dabsl_DIR:PATH=${absl_DIR}
        -DZLIB_ROOT:PATH=${ZLIB_ROOT}
        -DProtobuf_DIR:PATH=${Protobuf_DIR}
        -DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}
        -Dgrpc-web_DIR:PATH=${grpc-web_DIR}
        -DCudaHelper_DIR:PATH=${CudaHelper_DIR}
        -DOpenH264_DIR:PATH=${OpenH264_DIR}
        -Dnlohmann_json_SOURCE_DIR:PATH=${nlohmann_json_SOURCE_DIR}
        -Dnlohmann_json_BINARY_DIR:PATH=${nlohmann_json_BINARY_DIR}
        -DCLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR:PATH=${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}
        -DCLARA_VIZ_CUDA_PROFILING:BOOL=${CLARA_VIZ_CUDA_PROFILING}
        -DCLARA_VIZ_CUDA_DEBUG:BOOL=${CLARA_VIZ_CUDA_DEBUG}
        -DCLARA_VIZ_WITH_GRPC:BOOL=${CLARA_VIZ_WITH_GRPC}
        -DCLARA_VIZ_WITH_OPENH264:BOOL=${CLARA_VIZ_WITH_OPENH264}
        -DCLARA_VIZ_WITH_EXAMPLES:BOOL=${CLARA_VIZ_WITH_EXAMPLES}
    DEPENDS ${_DEPENDENCIES}
)
