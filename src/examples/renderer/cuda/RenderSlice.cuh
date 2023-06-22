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

#include "cuda/Session.h"

namespace clara::viz
{

/**
 * The Cuda slice render function.
 */
__global__ void RenderSlice(uint2 offset, uint2 size, const Matrix3x3 mat, float slice)
{
    const uint2 launch_index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if ((launch_index.x >= size.x) || (launch_index.y >= size.y))
    {
        return;
    }

    // get the in-slice location
    Vector3f loc(float(launch_index.x) / float(size.x - 1), float(launch_index.y) / float(size.y - 1), slice);
    // rotate to select the orientation
    loc = mat * loc;

    // get the density
    const float density = (tex3D<float>(g_session.volume_texture_, loc(0), loc(1), loc(2)) - g_session.density_min_) *
                          g_session.inv_density_range_;

    // output a grey level
    const uint8_t grey_level = uint8_t(std::max(std::min(density, 1.f), 0.f) * 255.f + 0.5f);

    g_session.buffer_display_[launch_index + offset] = make_uchar4(grey_level, grey_level, grey_level, 0);
}

} // namespace clara::viz
