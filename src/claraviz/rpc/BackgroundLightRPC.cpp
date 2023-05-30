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

#include "claraviz/rpc/BackgroundLightRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void BackgroundLightContext::ExecuteRPC(cinematic_v1::BackgroundLightRequest &request,
                                        cinematic_v1::BackgroundLightResponse &response)
{
    BackgroundLightInterface::AccessGuard access(GetResources()->light_);

    if (request.intensity() != 0.f)
    {
        access->intensity.Set(request.intensity());
    }

    if (request.has_top_color())
    {
        access->top_color.Set(MakeVector3f(request.top_color()));
    }

    if (request.has_horizon_color())
    {
        access->horizon_color.Set(MakeVector3f(request.horizon_color()));
    }

    if (request.has_bottom_color())
    {
        access->bottom_color.Set(MakeVector3f(request.bottom_color()));
    }

    switch (request.enable())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable = false;
        break;
    }

    switch (request.show())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->show = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->show = false;
        break;
    }

    switch (request.cast_light())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->cast_light = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->cast_light = false;
        break;
    }
}

} // namespace detail

} // namespace clara::viz
