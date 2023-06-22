/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/RenderSettingsInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(RenderSettingsInterface::Message);

template<>
RenderSettingsInterface::DataIn::RenderSettingsInterfaceData()
    : interpolation_mode(InterpolationMode::BSPLINE)
    , step_size(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Step size"); })
    , shadow_step_size(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Shadow step size"); })
    , max_iterations(10000, [](const uint32_t value) { ValidatorMinInclusive(value, 1u, "Maximum iterations"); })
    , time_slot(0.f, [](const float value) { ValidatorMinInclusive(value, 0.f, "Time slot"); })
    , enable_warp(false)
    , warp_resolution_scale(
          1.f, [](const float value) { ValidatorMinExclusiveMaxInclusive(value, 0.f, 1.f, "Warp resolution scale"); })
    , warp_full_resolution_size(
          1.f,
          [](const float value) { ValidatorMinExclusiveMaxInclusive(value, 0.f, 1.f, "Warp full resolution size"); })
    , enable_foveation(false)
    , enable_reproject(false)
    , enable_separate_depth(false)
{
}

template<>
RenderSettingsInterface::DataOut::RenderSettingsInterfaceData()
{
}

/**
 * Copy a post render settings config interface structure to a post render settings config POD structure.
 */
template<>
RenderSettingsInterface::DataOut RenderSettingsInterface::Get()
{
    AccessGuardConst access(this);

    RenderSettingsInterface::DataOut data_out;
    data_out.interpolation_mode        = access->interpolation_mode;
    data_out.step_size                 = access->step_size.Get();
    data_out.shadow_step_size          = access->shadow_step_size.Get();
    data_out.max_iterations            = access->max_iterations.Get();
    data_out.time_slot                 = access->time_slot.Get();
    data_out.enable_warp               = access->enable_warp;
    data_out.warp_resolution_scale     = access->warp_resolution_scale.Get();
    data_out.warp_full_resolution_size = access->warp_full_resolution_size.Get();
    data_out.enable_foveation          = access->enable_foveation;
    data_out.enable_reproject          = access->enable_reproject;
    data_out.enable_separate_depth     = access->enable_separate_depth;

    return data_out;
}

} // namespace clara::viz
