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

#include "claraviz/hardware/cuda/Convert.h"

#include "claraviz/hardware/cuda/CudaService.h"

namespace clara::viz
{

/**
 * Convert from ABGR to YCbCr 4:4:4 (CCIR 601)
 */
__global__ void ConvertABGRToYCbCr444CCIR601(uint32_t width, uint32_t height, const uint8_t *src, size_t src_pitch,
                                             uint8_t *dst_y, uint8_t *dst_cb, uint8_t *dst_cr, size_t dst_pitch)
{
    const uint2 launch_index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if ((launch_index.x >= width) || (launch_index.y >= height))
    {
        return;
    }

    const size_t src_offset = launch_index.x * 4 + launch_index.y * src_pitch;
    const float r           = static_cast<float>(src[src_offset + 0]) / 255.f;
    const float g           = static_cast<float>(src[src_offset + 1]) / 255.f;
    const float b           = static_cast<float>(src[src_offset + 2]) / 255.f;

    const float y  = 0.299f * r + 0.578 * g + 0.114 * b;
    const float cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5f;
    const float cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5f;

    const size_t dst_offset = launch_index.x + launch_index.y * dst_pitch;

    dst_y[dst_offset]  = static_cast<uint8_t>(y * 255.f + 0.5f);
    dst_cb[dst_offset] = static_cast<uint8_t>(cb * 255.f + 0.5f);
    dst_cr[dst_offset] = static_cast<uint8_t>(cr * 255.f + 0.5f);
}

std::unique_ptr<CudaFunctionLauncher> GetConvertABGRToYCbCr444CCIR601Launcher()
{
    return std::unique_ptr<CudaFunctionLauncher>(new CudaFunctionLauncher(ConvertABGRToYCbCr444CCIR601));
}

/**
 * Convert from ABGR to YCbCr 4:2:0 (CCIR 601)
 */
__global__ void ConvertABGRToYCbCr420CCIR601(uint32_t width, uint32_t height, const uint8_t *src, size_t src_pitch,
                                             uint8_t *dst_y, size_t dst_y_pitch, uint8_t *dst_cb, uint8_t *dst_cr,
                                             size_t dst_cbcr_pitch)
{
    const uint2 launch_index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if ((launch_index.x >= width) || (launch_index.y >= height))
    {
        return;
    }

    const size_t src_offset = launch_index.x * 4 + launch_index.y * src_pitch;
    const float r           = static_cast<float>(src[src_offset + 0]) / 255.f;
    const float g           = static_cast<float>(src[src_offset + 1]) / 255.f;
    const float b           = static_cast<float>(src[src_offset + 2]) / 255.f;

    const float y = 0.299f * r + 0.578 * g + 0.114 * b;

    const size_t dst_y_offset = launch_index.x + launch_index.y * dst_y_pitch;
    dst_y[dst_y_offset]       = static_cast<uint8_t>(y * 255.f + 0.5f);
    if (((launch_index.x & 1) == 0) && ((launch_index.y & 1) == 0))
    {
        const float cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5f;
        const float cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5f;

        const size_t dst_cbcr_offset = (launch_index.x / 2) + (launch_index.y / 2) * dst_cbcr_pitch;
        dst_cb[dst_cbcr_offset]      = static_cast<uint8_t>(cb * 255.f + 0.5f);
        dst_cr[dst_cbcr_offset]      = static_cast<uint8_t>(cr * 255.f + 0.5f);
    }
}

std::unique_ptr<CudaFunctionLauncher> GetConvertABGRToYCbCr420CCIR601Launcher()
{
    return std::unique_ptr<CudaFunctionLauncher>(new CudaFunctionLauncher(ConvertABGRToYCbCr420CCIR601));
}

} // namespace clara::viz
