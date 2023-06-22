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

include(GNUInstallDirs)

set(PROJECT_NAME VolumeStreamRenderer)

find_package(gRPC REQUIRED CONFIG)
find_package(CudaHelper REQUIRED CONFIG)
find_package(grpc-web REQUIRED CONFIG)
find_package(clara_viz_renderer REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD INTERFACE)
find_package(CUDA REQUIRED)
enable_language(CUDA)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        cuda/BackProjectionConstruction.cu
        DataSourceCT.cpp
        DataSourceUS.cpp
        Main.cpp
        VolumeStreamRenderServer.cpp
        volume_stream_render_server.proto
        ../../protos/nvidia/claraviz/core/camera.proto
        ../../protos/nvidia/claraviz/core/types.proto
        ../../protos/nvidia/claraviz/video/v1/video.proto
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        .
    PUBLIC
        ../../protos
    )

# generate C++ sources for gRPC
grpc_generate(${PROJECT_NAME})

# generate sources for grpc-web
protobuf_grpc_generate(
    TARGET ${PROJECT_NAME}
    LANGUAGE grpc-web
    PROTOS
        volume_stream_render_server.proto
        ../../protos/nvidia/claraviz/core/camera.proto
        ../../protos/nvidia/claraviz/core/types.proto
        ../../protos/nvidia/claraviz/video/v1/video.proto
    IMPORT_DIRS
        ../../protos
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        clara::viz::core
        clara::viz::core::hw
        clara::viz::core::grpc
        clara::viz::renderer
        ${CUDA_CUDA_LIBRARY}
        # for std::experimental::filesystem on GCC, can be removed when we use GCC 8
        -lstdc++fs
    )

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_BINDIR}/${CMAKE_HOST_SYSTEM_PROCESSOR}
    )

# Note: use hard coded `lib` instead of `${CMAKE_INSTALL_LIBDIR}`, manylinux uses `lib64`, but we want to run on any system.
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib/${CMAKE_SYSTEM_PROCESSOR}/"
)
