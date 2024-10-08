# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, s3oftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

######################
# core
######################
set(PROJECT_NAME clara_viz_core)

add_library(${PROJECT_NAME} STATIC)
add_library(clara::viz::core ALIAS ${PROJECT_NAME})

find_package(CUDAToolkit REQUIRED)
find_package(OpenH264 CONFIG)

if (NOT TARGET nlohmann_json)
    add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

target_code_coverage(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE True
    EXPORT_NAME clara::viz::core
    )

target_sources(${PROJECT_NAME}
    PRIVATE
        core/Video.cpp

        hardware/nvenc/NvEncService.cpp

        interface/CameraApertureInterface.cpp
        interface/CameraInterface.cpp
        interface/DataViewInterface.cpp
        interface/LightInterface.cpp
        interface/PostProcessDenoiseInterface.cpp
        interface/PostProcessTonemapInterface.cpp
        interface/RenderSettingsInterface.cpp
        interface/TransferFunctionInterface.cpp
        interface/VideoInterface.cpp
        interface/ViewInterface.cpp

        util/Log.cpp
        util/Message.cpp
        util/Observed.cpp
        util/Thread.cpp

        video/Mp4Wrapper.cpp
        video/NvEncVideoEncoder.cpp
        video/VideoEncoder.cpp
    )

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:src>
        ${CUDAToolkit_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        nlohmann_json::nlohmann_json
    )

if(OpenH264_FOUND)
    target_compile_definitions(${PROJECT_NAME}
        PRIVATE
            CLARA_VIZ_WITH_OPENH264
        )
    target_sources(${PROJECT_NAME}
        PRIVATE
            video/OpenH264VideoEncoder.cpp
        )
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            OpenH264::openh264
        )
endif()

######################
# core hw
######################
set(PROJECT_NAME clara_viz_core_hw)

add_library(${PROJECT_NAME} STATIC)
add_library(clara::viz::core::hw ALIAS ${PROJECT_NAME})

find_package(CUDA REQUIRED) # for find_cuda_helper_libs
find_package(CUDAToolkit REQUIRED)
find_package(nvsharedmemory CONFIG)
find_package(ZLIB)

# nvidia-docker is not mapping nvjpeg into the image, prefer to link statically
find_cuda_helper_libs(nvjpeg_static)
if(CUDA_nvjpeg_static_LIBRARY)
    message(STATUS "Found static nvjpeg lib")
    set(nvjpeg_LIB CUDA::nvjpeg_static)
    set(nvjpeg_FOUND TRUE)
else()
    find_cuda_helper_libs(nvjpeg)
    if(CUDA_nvjpeg_LIBRARY)
        message(STATUS "Found shared nvjpeg lib")
        set(nvjpeg_LIB CUDA::nvjpeg)
        set(nvjpeg_FOUND TRUE)
    else()
        message(NOTICE "Could not find nvjpeg lib (neither shared nor static), JPEG encdoing support disabled")
    endif()
endif()

enable_language(CUDA)

target_code_coverage(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE True
    EXPORT_NAME clara::viz::core::hw
    )

target_sources(${PROJECT_NAME}
    PRIVATE
        core/Image.cpp

        interface/ImageInterface.cpp

        hardware/cuda/Convert.cu

        image/JpegEncoder.cpp

        interface/DataInterface.cpp
        interface/JsonInterface.cpp

        util/Blob.cpp
    )

if(nvsharedmemory_FOUND)
    target_sources(${PROJECT_NAME}
        PRIVATE
            util/SharedMemoryBlob.cpp
        )
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            nvsharedmemory
        )
endif()

if(ZLIB_FOUND)
    target_sources(${PROJECT_NAME}
        PRIVATE
            util/MHDLoader.cpp
        )
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            ZLIB::ZLIB
        )
endif()

if(nvjpeg_FOUND)
    target_sources(${PROJECT_NAME}
        PRIVATE
            hardware/nvjpeg/NvJpegService.cpp
        )
    target_compile_definitions(${PROJECT_NAME}
        PRIVATE
            CLARA_VIZ_WITH_NVJPEG
        )
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            ${nvjpeg_LIB}
        )
endif()

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        clara::viz::core
        CUDA::cuda_driver
        CUDA::cudart_static
    )

if (CLARA_VIZ_WITH_GRPC)
    ######################
    # core grpc
    ######################
    set(PROJECT_NAME clara_viz_core_grpc)

    add_library(${PROJECT_NAME} STATIC)
    add_library(clara::viz::core::grpc ALIAS ${PROJECT_NAME})

    find_package(gRPC REQUIRED CONFIG)

    target_code_coverage(${PROJECT_NAME})

    set_target_properties(${PROJECT_NAME}
        PROPERTIES POSITION_INDEPENDENT_CODE True
        )

    target_sources(${PROJECT_NAME}
        PRIVATE
            core/RenderServerBase.cpp

            rpc/BackgroundLightRPC.cpp
            rpc/CameraApertureRPC.cpp
            rpc/CameraRPC.cpp
            rpc/DataCropRPC.cpp
            rpc/DataTransformRPC.cpp
            rpc/DataViewRPC.cpp
            rpc/LightRPC.cpp
            rpc/PostProcessTonemapRPC.cpp
            rpc/PostProcessDenoiseRPC.cpp
            rpc/RenderSettingsRPC.cpp
            rpc/ResetRPC.cpp
            rpc/ServerRPC.cpp
            rpc/TransferFunctionRPC.cpp
            rpc/VideoRPC.cpp
            rpc/ViewRPC.cpp
        )

    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            gRPC::grpc++
            nv::rpc
            clara::viz::core
            clara::viz::proto
        )

    ######################
    # core grpc hw
    ######################
    set(PROJECT_NAME clara_viz_core_grpc_hw)

    add_library(${PROJECT_NAME} STATIC)
    add_library(clara::viz::core::grpc::hw ALIAS ${PROJECT_NAME})

    find_package(gRPC REQUIRED CONFIG)

    target_code_coverage(${PROJECT_NAME})

    set_target_properties(${PROJECT_NAME}
        PROPERTIES POSITION_INDEPENDENT_CODE True
        )

    target_sources(${PROJECT_NAME}
        PRIVATE
            rpc/DataRPC.cpp
            rpc/ImageRPC.cpp
            rpc/QueryLimitsRPC.cpp
        )

    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            gRPC::grpc++
            clara::viz::proto
            clara::viz::core::hw
        )
endif()