/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/PostProcessDenoiseInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(PostProcessDenoiseInterface::Message);

template<>
PostProcessDenoiseInterface::DataIn::PostProcessDenoiseInterfaceData()
    : method(DenoiseMethod::OFF)
    , radius(3, [](const uint32_t value) { ValidatorMinExclusive(value, 0u, "Radius"); })
    , spatial_weight(0.05f, [](const float value) { ValidatorMinInclusive(value, 0.f, "Spatial Weight"); })
    , depth_weight(3.f, [](const float value) { ValidatorMinInclusive(value, 0.f, "Depth Weight"); })
    , noise_threshold(0.2f, [](const float value) { ValidatorMinMaxInclusive(value, 0.f, 1.f, "Noise Threshold"); })
    , enable_iteration_limit(false)
    , iteration_limit(100, [](const uint32_t value) { ValidatorMinExclusive(value, 0u, "Iteration Limit"); })
{
}

template<>
PostProcessDenoiseInterface::DataOut::PostProcessDenoiseInterfaceData()
{
}

/**
 * Copy a post process Denoise config interface structure to a post process Denoise config POD structure.
 */
template<>
PostProcessDenoiseInterface::DataOut PostProcessDenoiseInterface::Get()
{
    AccessGuardConst access(this);

    PostProcessDenoiseInterface::DataOut data_out;
    data_out.method                 = access->method;
    data_out.radius                 = access->radius.Get();
    data_out.spatial_weight         = access->spatial_weight.Get();
    data_out.depth_weight           = access->depth_weight.Get();
    data_out.noise_threshold        = access->noise_threshold.Get();
    data_out.enable_iteration_limit = access->enable_iteration_limit;
    data_out.iteration_limit        = access->iteration_limit.Get();

    return data_out;
}

} // namespace clara::viz
