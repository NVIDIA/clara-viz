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

#include "claraviz/rpc/CameraApertureRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void CameraApertureContext::ExecuteRPC(cinematic_v1::CameraApertureRequest &request,
                                       cinematic_v1::CameraApertureResponse &response)
{
    CameraApertureInterface::AccessGuard access(GetResources()->aperture_);

    switch (request.enable())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable = false;
        break;
    }

    if (request.aperture())
    {
        access->aperture.Set(request.aperture());
    }

    switch (request.auto_focus())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->auto_focus = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->auto_focus = false;
        break;
    }

    if (request.focus_distance())
    {
        access->focus_distance.Set(request.focus_distance());
    }
}

} // namespace detail

} // namespace clara::viz
