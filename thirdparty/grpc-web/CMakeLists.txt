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

include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Tools.cmake")

prebuild_setup(VERSION_VAR grpc-web_VERSION)

ExternalProject_Add(grpc-web_src
    PREFIX grpc-web_src
    URL https://github.com/grpc/grpc-web/archive/${grpc-web_VERSION}.zip
    URL_MD5 cebda516f42ef75722b143b0aba75aaa
    PATCH_COMMAND patch -p1 -N < "${CMAKE_CURRENT_SOURCE_DIR}/CMakeBuild.patch" || true
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL ON
    )

ExternalProject_Get_Property(grpc-web_src SOURCE_DIR)
set(grpc-web_SOURCE_DIR ${SOURCE_DIR})

prebuild_package_name(PREFIX grpc-web
    VERSION ${grpc-web_VERSION}
    VAR PACKAGE_NAME
    )

# Builds gRPC-web based on locally checked-out sources and set arguments so that all the dependencies
# are correctly located.
ExternalProject_Add(grpc-web
    PREFIX grpc-web
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${grpc-web_SOURCE_DIR}"
    CMAKE_ARGS
        -DProtobuf_DIR:PATH=${Protobuf_DIR}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE:STRING=LAZY
    INSTALL_DIR ${PACKAGE_NAME}
    EXCLUDE_FROM_ALL ON
    DEPENDS grpc-web_src Protobuf
    )

ExternalProject_Get_Property(grpc-web INSTALL_DIR)

set(grpc-web_DIR "${INSTALL_DIR}/bin/cmake/grpc-web" CACHE INTERNAL "grpc-web install dir" FORCE)
