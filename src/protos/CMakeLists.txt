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

set(PROJECT_NAME clara_viz_proto)

add_library(${PROJECT_NAME} STATIC)
add_library(clara::viz::proto ALIAS ${PROJECT_NAME})

find_package(gRPC REQUIRED CONFIG)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(${PROJECT_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE True
    )

target_sources(${PROJECT_NAME}
    PRIVATE
        nvidia/claraviz/cinematic/v1/render_server.proto
        nvidia/claraviz/core/camera.proto
        nvidia/claraviz/core/types.proto
        nvidia/claraviz/fetch/v1/fetch.proto
        nvidia/claraviz/image/v1/image.proto
        nvidia/claraviz/video/v1/video.proto
    )

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
        $<INSTALL_INTERFACE:src/protos>
    )

# generate sources for gRPC
grpc_generate(${PROJECT_NAME})
