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

#include "claraviz/rpc/LightRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void LightContext::ExecuteRPC(cinematic_v1::LightRequest &request, cinematic_v1::LightResponse &response)
{
    LightInterface::AccessGuard access(GetResources()->light_);

    if ((request.index() < 0) || (request.index() >= access->lights.size()))
    {
        throw InvalidArgument("Light index")
            << "expected to be >= 0 and <= " << access->lights.size() - 1 << " but is " << request.index();
    }

    LightInterface::DataIn::Light &light = access->lights[request.index()];
    if (request.has_position())
    {
        light.position = MakeVector3f(request.position());
    }
    if (request.has_direction())
    {
        light.direction.Set(MakeVector3f(request.direction()));
    }
    if (request.size())
    {
        light.size.Set(request.size());
    }
    if (request.intensity())
    {
        light.intensity.Set(request.intensity());
    }

    if (request.has_color())
    {
        light.color.Set(MakeVector3f(request.color()));
    }

    switch (request.enable())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        light.enable = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        light.enable = false;
        break;
    }

    switch (request.show())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        light.show = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        light.show = false;
        break;
    }
}

} // namespace detail

} // namespace clara::viz
