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

include("${CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR}/Tools.cmake")
include(ExternalProject)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.txt CudaHelper_VERSION)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.txt)

prebuild_setup(VERSION_VAR CudaHelper_VERSION)

prebuild_package_name(PREFIX cuda_helper
    VERSION ${CudaHelper_VERSION}
    VAR CudaHelper_PACKAGE_NAME
    )

ExternalProject_Add(CudaHelper
    PREFIX cuda_helper
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include <INSTALL_DIR>/include
    EXCLUDE_FROM_ALL ON
    )

ExternalProject_Get_Property(CudaHelper INSTALL_DIR)

set(CudaHelper_DIR "${INSTALL_DIR}" CACHE INTERNAL "CudaHelper install dir" FORCE)

# generate the files for find_package
include(CMakePackageConfigHelpers)
configure_package_config_file(CudaHelperConfig.cmake.in
    ${CudaHelper_DIR}/CudaHelperConfig.cmake
    INSTALL_DESTINATION ${CudaHelper_DIR}
    INSTALL_PREFIX ${CudaHelper_DIR}
    PATH_VARS CudaHelper_DIR
    )

write_basic_package_version_file(
    ${CudaHelper_DIR}/CudaHelperConfigVersion.cmake
    VERSION {CudaHelper_VERSION}
    COMPATIBILITY SameMajorVersion
    )
