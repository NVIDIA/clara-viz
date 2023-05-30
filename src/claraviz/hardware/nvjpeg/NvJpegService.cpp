/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "claraviz/hardware/nvjpeg/NvJpegService.h"

std::ostream &operator<<(std::ostream &os, const nvjpegStatus_t status)
{
    switch (status)
    {
    case NVJPEG_STATUS_SUCCESS:
        os << std::string("NVJPEG_STATUS_SUCCESS");
        break;
    case NVJPEG_STATUS_NOT_INITIALIZED:
        os << std::string("NVJPEG_STATUS_NOT_INITIALIZED");
        break;
    case NVJPEG_STATUS_INVALID_PARAMETER:
        os << std::string("NVJPEG_STATUS_INVALID_PARAMETER");
        break;
    case NVJPEG_STATUS_BAD_JPEG:
        os << std::string("NVJPEG_STATUS_BAD_JPEG");
        break;
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        os << std::string("NVJPEG_STATUS_JPEG_NOT_SUPPORTED");
        break;
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        os << std::string("NVJPEG_STATUS_ALLOCATOR_FAILURE");
        break;
    case NVJPEG_STATUS_EXECUTION_FAILED:
        os << std::string("NVJPEG_STATUS_EXECUTION_FAILED");
        break;
    case NVJPEG_STATUS_ARCH_MISMATCH:
        os << std::string("NVJPEG_STATUS_ARCH_MISMATCH");
        break;
    case NVJPEG_STATUS_INTERNAL_ERROR:
        os << std::string("NVJPEG_STATUS_INTERNAL_ERROR");
        break;
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        os << std::string("NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED");
        break;
    default:
        os << (int)status;
        break;
    }
    return os;
}
