# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

find_package(gRPC REQUIRED CONFIG)
find_package(grpc-web REQUIRED CONFIG)

set(PROJECT_NAME web_client)

# generate sources for grpc-web
protobuf_grpc_generate(
    OUT_VAR _GENERATED_SOURCES
    LANGUAGE grpc-web
    PROTOS
        ../../protos/nvidia/claraviz/cinematic/v1/render_server.proto
        ../../protos/nvidia/claraviz/core/camera.proto
        ../../protos/nvidia/claraviz/core/types.proto
        ../../protos/nvidia/claraviz/video/v1/video.proto
    IMPORT_DIRS
        ../../protos
    )

# create a target with dependencies on the generated sources
add_custom_target(${PROJECT_NAME} ALL
    DEPENDS ${_GENERATED_SOURCES}
    )

