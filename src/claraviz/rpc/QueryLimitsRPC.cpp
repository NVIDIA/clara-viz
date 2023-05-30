/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/rpc/QueryLimitsRPC.h"

#include <claraviz/hardware/cuda/CudaService.h>

#include "claraviz/interface/LightInterface.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

QueryLimitsResource::QueryLimitsResource(int cuda_device_ordinal)
    : max_lights_(LIGHTS_MAX)
{
    CUdevice cuda_device = 0;
    CudaCheck(cuDeviceGet(&cuda_device, cuda_device_ordinal));

    int value = 0;
    CudaCheck(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, cuda_device));
    max_image_width_ = value;

    value = 0;
    CudaCheck(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, cuda_device));
    max_image_height_ = value;

    value = 0;
    CudaCheck(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, cuda_device));
    max_volume_width_ = value;

    value = 0;
    CudaCheck(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, cuda_device));
    max_volume_height_ = value;

    value = 0;
    CudaCheck(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, cuda_device));
    max_volume_depth_ = value;
}

void QueryLimitsContext::ExecuteRPC(cinematic_v1::QueryLimitsRequest &request,
                                    cinematic_v1::QueryLimitsResponse &response)
{
    const std::shared_ptr<QueryLimitsResource> &resource = GetResources();

    uint32_t dims = 0;
    if (request.dimension_order().find('X') != std::string::npos)
    {
        ++dims;
    }
    if (request.dimension_order().find('Y') != std::string::npos)
    {
        ++dims;
    }
    if (request.dimension_order().find('Z') != std::string::npos)
    {
        ++dims;
    }

    response.set_max_lights(resource->max_lights_);
    if (dims == 3)
    {
        response.mutable_max_data_size()->Add(resource->max_volume_width_);
        response.mutable_max_data_size()->Add(resource->max_volume_height_);
        response.mutable_max_data_size()->Add(resource->max_volume_depth_);
    }
    else if (dims == 2)
    {
        response.mutable_max_data_size()->Add(resource->max_image_width_);
        response.mutable_max_data_size()->Add(resource->max_image_height_);
    }
}

} // namespace detail

} // namespace clara::viz
