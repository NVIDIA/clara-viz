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

cmake_minimum_required(VERSION 3.13)

include(GNUInstallDirs)
include(FetchContent)
include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Tools.cmake")

prebuild_setup(VERSION_VAR gRPC_VERSION)

# silence a warning about 'ZLIB_ROOT' usage
cmake_policy(SET CMP0074 OLD)

ExternalProject_Add(gRPC_src
    PREFIX grpc
    GIT_REPOSITORY https://github.com/grpc/grpc.git
    GIT_TAG ${gRPC_VERSION}
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    GIT_SUBMODULES "third_party/abseil-cpp third_party/cares/cares third_party/protobuf third_party/re2 third_party/zlib"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL ON
    )

ExternalProject_Get_Property(gRPC_src SOURCE_DIR)
set(gRPC_SOURCE_DIR ${SOURCE_DIR})

# Build protobuf and zlib before because we also used these packages.

# Builds protobuf project from the git submodule.
prebuild_package_name(PREFIX grpc-protobuf
    VERSION ${gRPC_VERSION}
    VAR Protobuf_PACKAGE_NAME
    )

ExternalProject_Add(Protobuf
    PREFIX protobuf
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${gRPC_SOURCE_DIR}/third_party/protobuf/cmake"
    CMAKE_CACHE_ARGS
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -Dprotobuf_WITH_ZLIB:BOOL=OFF
        -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE:STRING=LAZY
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    INSTALL_DIR ${Protobuf_PACKAGE_NAME}
    EXCLUDE_FROM_ALL ON
    DEPENDS gRPC_src
    )

ExternalProject_Get_Property(Protobuf INSTALL_DIR)
set(Protobuf_DIR "${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/protobuf" CACHE INTERNAL "gRPC protobuf install dir" FORCE)

# Workaround for https://github.com/protocolbuffers/protobuf-javascript/issues/127
# Download protobuf-javascript
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(_arch "aarch_64")
    set(_pb_js_md5 f827c469ab2ed01aa49accb3739765cb)
else()
    set(_arch ${CMAKE_SYSTEM_PROCESSOR})
    set(_pb_js_md5 2d938b8f12ef298bd19838e68ef55d4f)
endif()
FetchContent_Declare(
  protobuf-javascript
  URL https://github.com/protocolbuffers/protobuf-javascript/releases/download/v3.21.2/protobuf-javascript-3.21.2-linux-${_arch}.tar.gz
  URL_MD5 ${_pb_js_md5}
)
FetchContent_MakeAvailable(protobuf-javascript)
set(protobuf-javascript_SOURCE_DIR "${protobuf-javascript_SOURCE_DIR}" CACHE INTERNAL "protobuf-javascript source dir" FORCE)

# Builds zlib project from the git submodule.
prebuild_package_name(PREFIX grpc-zlib
    VERSION ${gRPC_VERSION}
    VAR ZLIB_PACKAGE_NAME
    )

ExternalProject_Add(ZLIB
    PREFIX zlib
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${gRPC_SOURCE_DIR}/third_party/zlib"
    CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE:STRING=LAZY
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    INSTALL_DIR ${ZLIB_PACKAGE_NAME}
    EXCLUDE_FROM_ALL ON
    DEPENDS gRPC_src
    )

ExternalProject_Get_Property(ZLIB INSTALL_DIR)
set(ZLIB_ROOT "${INSTALL_DIR}" CACHE INTERNAL "gRPC zlib install dir" FORCE)

# Builds gRPC based on locally checked-out sources and set arguments so that all the dependencies
# are correctly located.
prebuild_package_name(PREFIX grpc
    VERSION ${gRPC_VERSION}
    VAR gRPC_PACKAGE_NAME
    )

ExternalProject_Add(gRPC
    PREFIX grpc
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${gRPC_SOURCE_DIR}"
    CMAKE_ARGS
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_BUILD_CODEGEN:BOOL=ON
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${Protobuf_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:STRING=${ZLIB_ROOT}
        -DgRPC_CARES_PROVIDER:STRING=module
        -DgRPC_SSL_PROVIDER:STRING=package
        -DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE:STRING=LAZY
    INSTALL_DIR ${gRPC_PACKAGE_NAME}
    EXCLUDE_FROM_ALL ON
    DEPENDS gRPC_src Protobuf ZLIB OpenSSL
    )

ExternalProject_Get_Property(gRPC INSTALL_DIR)

# we don't need absl internally but gRPC is using find_package to support externel absl builds
set(absl_DIR "${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/absl" CACHE INTERNAL "gRPC absl install dir" FORCE)

set(gRPC_DIR "${INSTALL_DIR}/lib/cmake/grpc" CACHE INTERNAL  "gRPC install dir" FORCE)
