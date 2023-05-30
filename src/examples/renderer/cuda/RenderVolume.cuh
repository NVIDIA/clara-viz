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
 * Ray
 */
struct Ray
{
    __device__ Ray(float3 origin_, float3 direction_, float tmin_, float tmax_)
        : origin(origin_)
        , direction(direction_)
        , tmin(tmin_)
        , tmax(tmax_)
    {
    }
    Ray() = default;

    // origin of the ray
    float3 origin;
    // direction of the ray
    float3 direction;
    // min extent associated with this ray
    float tmin;
    // max extent associated with this ray
    float tmax;
};

/**
 * Intersect a ray with the volume bounding box.
 */
__device__ inline bool IntersectBox(const Ray &ray, float *near, float *far)
{
    const float3 inv_ray = make_float3(1.0f, 1.0f, 1.0f) / ray.direction;

    // the volume is a unit bbox around the origin
    const float3 volume_min = make_float3(-0.5f, -0.5f, -0.5f);
    const float3 volume_max = make_float3(0.5f, 0.5f, 0.5f);

    const float3 bottom = inv_ray * (volume_min - ray.origin);
    const float3 top    = inv_ray * (volume_max - ray.origin);

    const float3 min = fminf(top, bottom);
    const float3 max = fmaxf(top, bottom);

    *near = fmaxf(fmaxf(min.x, min.y), fmaxf(min.z, ray.tmin));
    *far  = fminf(fminf(max.x, max.y), fminf(max.z, ray.tmax));

    return *far > *near;
}

/**
 * The Cuda volume render function.
 */
__global__ void RenderVolume(uint2 offset, uint2 size)
{
    const uint2 launch_index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if ((launch_index.x >= size.x) || (launch_index.y >= size.y))
    {
        return;
    }

    const float2 uv = make_float2(launch_index.x, launch_index.y) / make_float2(size.x - 1, size.y - 1);

    const float2 point =
        make_float2(g_session.camera_setup_.viewport_offset_.x + (g_session.camera_setup_.viewport_inv_size_.x * uv.x),
                    g_session.camera_setup_.viewport_offset_.y + (g_session.camera_setup_.viewport_inv_size_.y * uv.y));
    const float3 direction = normalize(g_session.camera_setup_.nz_ + (point.x * g_session.camera_setup_.nx_) +
                                       (point.y * g_session.camera_setup_.ny_));

    Ray ray(g_session.camera_setup_.eye_, direction, 0.0f, FLT_MAX);

    // start with background color
    const float3 top_color    = make_float3(0.1f, 0.4f, 1.f);
    const float3 bottom_color = make_float3(0.6f, 0.3f, 0.f);

    float3 color = lerp(top_color, bottom_color, uv.y);

    float t_min, t_max;
    if (IntersectBox(ray, &t_min, &t_max))
    {
        float sum = 0.f;
        float t   = t_min;
        while ((sum < 1.f) && (t <= t_max))
        {
            // calculate the volume location
            const float3 volume_min = make_float3(-0.5f, -0.5f, -0.5f);
            const float3 loc        = (ray.origin + t * ray.direction) - volume_min;

            // fetch the volume
            sum += (tex3D<float>(g_session.volume_texture_, loc.x, loc.y, loc.z) - g_session.density_min_) *
                   g_session.inv_density_range_ * g_session.step_ * 4.f;
            t += g_session.step_;
        }

        // clamp to 1
        sum = std::min(sum, 1.f);

        // mix with background color
        color = lerp(color, make_float3(1.f, 1.f, 1.f), sum);
    }

    // write the color to the display buffer
    g_session.buffer_display_[launch_index + offset] =
        make_uchar4(color.x * 255.f + 0.5f, color.y * 255.f + 0.5f, color.z * 255.f + 0.5f, 0);
}

} // namespace clara::viz
