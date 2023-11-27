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

prebuild_setup(VERSION_VAR OpenH264_VERSION)

prebuild_package_name(PREFIX openh264
    VERSION ${OpenH264_VERSION}
    VAR PACKAGE_NAME
    )

if (NOT CMAKE_GENERATOR STREQUAL "Ninja")
    set(JOBS "-j$(nproc)")
endif()

# build from source
ExternalProject_Add(OpenH264
    PREFIX openh264
    URL https://github.com/cisco/openh264/archive/${OpenH264_VERSION}.zip
    URL_MD5 0b7b0209b011865c22f8bd64d1701f8e
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make -C <SOURCE_DIR> CFLAGS='-fPIC' ${JOBS}
    INSTALL_DIR ${PACKAGE_NAME}
    INSTALL_COMMAND make -C <SOURCE_DIR> install-static-lib install-headers PREFIX=<INSTALL_DIR>
    EXCLUDE_FROM_ALL ON
    )

ExternalProject_Get_Property(OpenH264 INSTALL_DIR)

set(OpenH264_DIR "${INSTALL_DIR}" CACHE INTERNAL "OpenH264 install dir" FORCE)

# generate the files for find_package
include(CMakePackageConfigHelpers)
configure_package_config_file(OpenH264Config.cmake.in
    ${OpenH264_DIR}/OpenH264Config.cmake
    INSTALL_DESTINATION ${OpenH264_DIR}
    INSTALL_PREFIX ${OpenH264_DIR}
    PATH_VARS OpenH264_DIR
    )

write_basic_package_version_file(
    ${OpenH264_DIR}/OpenH264ConfigVersion.cmake
    VERSION {OpenH264_VERSION}
    COMPATIBILITY SameMajorVersion
    )

