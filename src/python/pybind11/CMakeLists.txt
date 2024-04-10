# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

include(FetchContent)

find_package(gRPC REQUIRED CONFIG)
find_package(OpenH264 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(clara_viz_renderer REQUIRED)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  # add 'EXCLUDE_FROM_ALL' to avoid the install step
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# nlohmann json
add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)

# order is important, include pybind11 and json before pybind11_json

# pybind11 json
FetchContent_Declare(
    pybind11_json
    GIT_REPOSITORY https://github.com/pybind/pybind11_json.git
    GIT_TAG 0.2.11
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(pybind11_json)
if(NOT pybind11_json_POPULATED)
  FetchContent_Populate(pybind11_json)
  # add 'EXCLUDE_FROM_ALL' to avoid the install step
  add_subdirectory(${pybind11_json_SOURCE_DIR} ${pybind11_json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

pybind11_add_module(${PROJECT_NAME}
    MODULE
        clara-viz_py.cpp
        PythonRenderer.cpp
        VideoStream.cpp
    )

pybind11_extension(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        .
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        clara::viz::renderer
        clara::viz::core::hw
        clara::viz::core
        pybind11_json
    )

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    pybind11_strip(${PROJECT_NAME})
endif()

# cmake requires unique target names but the generated library should start with '_viz', the python suffix is added
# by cmake
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME "_viz"
    )

# add dependency to export file
# export only needed symbols
set(EXPORT_MAP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/export.map)
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--version-script=${EXPORT_MAP_FILE}")
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        LINK_DEPENDS ${EXPORT_MAP_FILE}
    )
