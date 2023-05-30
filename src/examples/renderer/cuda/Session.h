/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <cuda_helper/helper_math.h>

namespace clara::viz
{

/**
 * Helper class to access Cuda pitch memory from kernels.
 */
template<typename T>
struct BufferPitch
{
    CUdeviceptr data_;
    uint32_t pitch_;

    __device__ T &operator[](const uint2 index)
    {
        return *(reinterpret_cast<T *>(data_ + index.x * sizeof(T) + pitch_ * index.y));
    }

    void Update(CUdeviceptr data, uint32_t pitch)
    {
        data_  = data;
        pitch_ = pitch;
    }
};

/// Camera setup
struct CameraSetup
{
    void Update(const CameraInterface::DataOut::Camera &camera)
    {
        eye_ = static_cast<float3>(camera.eye);

        nz_ = normalize(static_cast<float3>(camera.look_at) - eye_);
        nx_ = normalize(cross(static_cast<float3>(camera.up), nz_));
        ny_ = normalize(cross(nz_, nx_));

        const float s = tanf(0.5f * camera.field_of_view * CUDART_PI_F / 180.f);

        if (camera.pixel_aspect_ratio > 1.0f)
        {
            viewport_offset_.x = -s;
            viewport_offset_.y = -s * camera.pixel_aspect_ratio;
        }
        else
        {
            viewport_offset_.x = -s / camera.pixel_aspect_ratio;
            viewport_offset_.y = -s;
        }

        viewport_inv_size_.x = -(viewport_offset_.x + viewport_offset_.x);
        viewport_inv_size_.y = -(viewport_offset_.y + viewport_offset_.y);
    }

    float3 eye_;
    float3 nx_;
    float3 ny_;
    float3 nz_;

    float2 viewport_offset_;
    float2 viewport_inv_size_;
};

/// The session holds all information used by the Cuda renderer
struct Session
{
    // camera setup
    CameraSetup camera_setup_;

    // display buffer to render into
    BufferPitch<uchar4> buffer_display_;

    // volume texture
    CUtexObject volume_texture_;

    /// minimum density value
    float density_min_;
    /// inverse of density range
    float inv_density_range_;

    // ray marching step value
    float step_;
};

} // namespace clara::viz
