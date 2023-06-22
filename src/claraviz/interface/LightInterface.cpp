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

#include "claraviz/interface/LightInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(LightInterface::Message);
template<>
DEFINE_CLASS_MESSAGEID(BackgroundLightInterface::Message);

template<>
LightInterface::DataIn::Light::Light()
    : position(0.f, 0.f, -1.f)
    , direction(Vector3f(0.f, 0.f, 1.f), [](const Vector3f &value) { ValidatorUnitVector(value, "Direction"); })
    , size(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Size"); })
    , intensity(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Intensity"); })
    , color(Vector3f(1.f, 1.f, 1.f),
            [](const Vector3f &value) { ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Color"); })
    , enable(false)
    , show(true)
{
}

template<>
LightInterface::DataIn::LightInterfaceData()
{
}

template<>
LightInterface::DataOut::Light::Light()
{
}

template<>
LightInterface::DataOut::LightInterfaceData()
{
}

template<>
BackgroundLightInterface::DataIn::BackgroundLightInterfaceData()
    : intensity(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Intensity"); })
    , top_color(Vector3f(1.f, 1.f, 1.f),
                [](const Vector3f &value) {
                    ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Top color (+y direction)");
                })
    , horizon_color(Vector3f(1.f, 1.f, 1.f),
                    [](const Vector3f &value) {
                        ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Horizon color (x-z plane)");
                    })
    , bottom_color(Vector3f(1.f, 1.f, 1.f),
                   [](const Vector3f &value) {
                       ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Bottom color (-y direction)");
                   })
    , enable(false)
    , show(true)
    , cast_light(true)
{
}

template<>
BackgroundLightInterface::DataOut::BackgroundLightInterfaceData()
{
}

/**
 * Copy a light config interface structure to a light config POD structure.
 */
template<>
LightInterface::DataOut LightInterface::Get()
{
    AccessGuardConst access(this);

    LightInterface::DataOut data_out;
    for (size_t index = 0; index < access->lights.size(); ++index)
    {
        const LightInterface::DataIn::Light &light_in = access->lights[index];
        LightInterface::DataOut::Light &light_out     = data_out.lights[index];

        light_out.position  = light_in.position;
        light_out.direction = light_in.direction.Get();
        light_out.size      = light_in.size.Get();
        light_out.intensity = light_in.intensity.Get();
        light_out.color     = light_in.color.Get();
        light_out.enable    = light_in.enable;
        light_out.show      = light_in.show;
    }

    return data_out;
}

/**
 * Copy a background light data interface structure to a background light data POD structure.
 */
template<>
BackgroundLightInterface::DataOut BackgroundLightInterface::Get()
{
    AccessGuardConst access(this);

    BackgroundLightInterface::DataOut data_out;
    data_out.intensity     = access->intensity.Get();
    data_out.top_color     = access->top_color.Get();
    data_out.horizon_color = access->horizon_color.Get();
    data_out.bottom_color  = access->bottom_color.Get();
    data_out.enable        = access->enable;
    data_out.show          = access->show;
    data_out.cast_light    = access->cast_light;

    return data_out;
}

} // namespace clara::viz
