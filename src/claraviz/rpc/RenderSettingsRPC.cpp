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

#include "claraviz/rpc/RenderSettingsRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void RenderSettingsContext::ExecuteRPC(cinematic_v1::RenderSettingsRequest &request,
                                       cinematic_v1::RenderSettingsResponse &response)
{
    RenderSettingsInterface::AccessGuard access(GetResources()->render_settings_);

    switch (request.interpolation_mode())
    {
    case cinematic_v1::RenderSettingsRequest::LINEAR:
        access->interpolation_mode = InterpolationMode::LINEAR;
        break;
    case cinematic_v1::RenderSettingsRequest::BSPLINE:
        access->interpolation_mode = InterpolationMode::BSPLINE;
        break;
    case cinematic_v1::RenderSettingsRequest::CATMULLROM:
        access->interpolation_mode = InterpolationMode::CATMULLROM;
        break;
    case cinematic_v1::RenderSettingsRequest::MODE_UNKNOWN:
        break;
    default:
        Log(LogLevel::Warning) << "Unhandled render settings interpolation mode " << request.interpolation_mode();
        break;
    }

    if (request.step_size())
    {
        access->step_size.Set(request.step_size());
    }
    if (request.shadow_step_size())
    {
        access->shadow_step_size.Set(request.shadow_step_size());
    }
    if (request.max_iterations())
    {
        access->max_iterations.Set(request.max_iterations());
    }
    if (request.time_slot())
    {
        access->time_slot.Set(request.time_slot());
    }
    switch (request.enable_warp())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable_warp = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable_warp = false;
        break;
    }
    if (request.warp_resolution_scale())
    {
        access->warp_resolution_scale.Set(request.warp_resolution_scale());
    }
    if (request.warp_full_resolution_size())
    {
        access->warp_full_resolution_size.Set(request.warp_full_resolution_size());
    }
    switch (request.enable_foveation())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable_foveation = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable_foveation = false;
        break;
    }
    switch (request.enable_reproject())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable_reproject = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable_reproject = false;
        break;
    }
    switch (request.enable_separate_depth())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable_separate_depth = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable_separate_depth = false;
        break;
    }
}

} // namespace detail

} // namespace clara::viz
