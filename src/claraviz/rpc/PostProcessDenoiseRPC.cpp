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

#include "claraviz/rpc/PostProcessDenoiseRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void PostProcessDenoiseContext::ExecuteRPC(cinematic_v1::PostProcessDenoiseRequest &request,
                                           cinematic_v1::PostProcessDenoiseResponse &response)
{
    PostProcessDenoiseInterface::AccessGuard access(GetResources()->denoise_);

    switch (request.method())
    {
    case cinematic_v1::PostProcessDenoiseRequest::OFF:
        access->method = DenoiseMethod::OFF;
        break;
    case cinematic_v1::PostProcessDenoiseRequest::KNN:
        access->method = DenoiseMethod::KNN;
        break;
    case cinematic_v1::PostProcessDenoiseRequest::AI:
        access->method = DenoiseMethod::AI;
        break;
    case cinematic_v1::PostProcessDenoiseRequest::METHOD_UNKNOWN:
        break;
    default:
        Log(LogLevel::Warning) << "Unhandled post process denoise method " << request.method();
        break;
    }

    if (request.radius())
    {
        access->radius.Set(request.radius());
    }
    if (request.spatial_weight())
    {
        access->spatial_weight.Set(request.spatial_weight());
    }
    if (request.depth_weight())
    {
        access->depth_weight.Set(request.depth_weight());
    }
    if (request.noise_threshold())
    {
        access->noise_threshold.Set(request.noise_threshold());
    }

    switch (request.enable_iteration_limit())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable_iteration_limit = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable_iteration_limit = false;
        break;
    }
    if (request.iteration_limit())
    {
        access->iteration_limit.Set(request.iteration_limit());
    }
}

} // namespace detail

} // namespace clara::viz
