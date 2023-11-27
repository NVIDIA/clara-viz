# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

set(PROJECT_NAME nv_rpc)

add_library(${PROJECT_NAME} STATIC)
add_library(nv::rpc ALIAS ${PROJECT_NAME})

find_package(gRPC REQUIRED CONFIG)

target_code_coverage(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE True
    )

target_sources(${PROJECT_NAME}
    PRIVATE
        Executor.cc
        Server.cc
        ThreadPool.cc
    )

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:src>
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        clara::viz::core
        gRPC::grpc++
    )
