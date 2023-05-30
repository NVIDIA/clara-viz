# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Tools.cmake")

prebuild_setup(VERSION_VAR OPENSSL_VERSION)

prebuild_package_name(PREFIX openssl
    VERSION ${OPENSSL_VERSION}
    VAR PACKAGE_NAME
    )

if (NOT CMAKE_GENERATOR STREQUAL "Ninja")
    set(JOBS "-j$(nproc)")
endif()

# build from source
ExternalProject_Add(OpenSSL
    PREFIX openssl
    URL https://github.com/openssl/openssl/archive/OpenSSL_${OPENSSL_VERSION}.zip
    URL_MD5 5c8e4598437641943bffaef0e10ba87d
    PATCH_COMMAND patch -p0 -N < "${CMAKE_CURRENT_SOURCE_DIR}/openssl_dont_install_docs.patch" || true
    CONFIGURE_COMMAND <SOURCE_DIR>/config --prefix=<INSTALL_DIR> no-asm no-shared
    BUILD_COMMAND make ${JOBS} build_libs
    INSTALL_DIR ${PACKAGE_NAME}
    INSTALL_COMMAND make install
    EXCLUDE_FROM_ALL ON
    )

ExternalProject_Get_Property(OpenSSL INSTALL_DIR)

set(OPENSSL_ROOT_DIR "${INSTALL_DIR}" CACHE INTERNAL "OpenSSL install dir" FORCE)
