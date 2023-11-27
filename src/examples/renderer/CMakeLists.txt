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

include(GNUInstallDirs)

set(PROJECT_NAME ExampleRenderServer)

find_package(gRPC REQUIRED CONFIG)
find_package(CudaHelper REQUIRED CONFIG)

set(CUDA_LINK_LIBRARIES_KEYWORD INTERFACE)
find_package(CUDA REQUIRED)
enable_language(CUDA)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        cuda/Renderer.cu
        interface/SliceInterface.cpp
        rpc/SliceRPC.cpp
        ExampleRenderServer.cpp
        Main.cpp
        example_render_server.proto
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

# generate sources for gRPC
grpc_generate(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        clara::viz::core
        clara::viz::core::grpc
        clara::viz::core::hw
        CudaHelper::cuda_helper
    )

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_BINDIR}/${CMAKE_HOST_SYSTEM_PROCESSOR}
    )
