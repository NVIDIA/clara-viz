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

#include "claraviz/interface/CameraApertureInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(CameraApertureInterface::Message);

template<>
CameraApertureInterface::DataIn::CameraApertureInterfaceData()
    : enable(false)
    , aperture(0.1f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Aperture"); })
    , auto_focus(false)
    , focus_distance(1.0f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Focus distance"); })
{
}

template<>
CameraApertureInterface::DataOut::CameraApertureInterfaceData()
{
}

/**
 * Copy a post process tonemap config interface structure to a post process tonemap config POD structure.
 */
template<>
CameraApertureInterface::DataOut CameraApertureInterface::Get()
{
    AccessGuardConst access(this);

    CameraApertureInterface::DataOut data_out;
    data_out.enable         = access->enable;
    data_out.aperture       = access->aperture.Get();
    data_out.auto_focus     = access->auto_focus;
    data_out.focus_distance = access->focus_distance.Get();

    return data_out;
}

} // namespace clara::viz
