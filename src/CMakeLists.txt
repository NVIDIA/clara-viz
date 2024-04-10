# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required(VERSION 3.24...3.26)

# set CMP0148 policy to OLD to allow FindCUDA
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

project(ClaraVizServer
    DESCRIPTION "ClaraVizServer"
    VERSION 0.4.0
    LANGUAGES CXX
)

include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Project.cmake")
setup_project()

# add the tools
include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Tools.cmake")

if (CLARA_VIZ_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()

if (CLARA_VIZ_WITH_GRPC)
    add_subdirectory(nvrpc)
    add_subdirectory(protos)
    add_subdirectory(python)
endif()

add_subdirectory(claraviz)
