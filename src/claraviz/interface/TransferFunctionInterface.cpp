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

#include "claraviz/interface/TransferFunctionInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(TransferFunctionInterface::Message);

template<>
TransferFunctionInterface::DataIn::TransferFunctionInterfaceData()
    : shading_profile(TransferFunctionShadingProfile::HYBRID)
    , blending_profile(TransferFunctionBlendingProfile::MAXIMUM_OPACITY)
    , global_opacity(1.f, [](const float value) { ValidatorMinMaxInclusive(value, 0.f, 1.f, "Global opacity"); })
    , density_scale(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Density scale"); })
    , gradient_scale(1.f, [](const float value) { ValidatorMinExclusive(value, 0.f, "Gradient scale"); })
{
}

template<>
TransferFunctionInterface::DataIn::Component::Component()
    : range(Vector2f(0.f, 1.f), [](const Vector2f value) { ValidatorRange(value, 0.f, 1.f, "Range"); })
    , opacity_profile(TransferFunctionOpacityProfile::SQUARE)
    , opacity_transition(0.2f,
                         [](const float value) { ValidatorMinMaxInclusive(value, 0.f, 1.f, "Opacity transition"); })
    , opacity(0.5f, [](const float value) { ValidatorMinMaxInclusive(value, 0.f, 1.f, "Opacity"); })
    , roughness(0.f, [](const float value) { ValidatorMinInclusive(value, 0.f, "Roughness"); })
    , emissive_strength(0.f, [](const float value) { ValidatorMinInclusive(value, 0.f, "Emissive strength"); })
    , diffuse_start(
          Vector3f(1.f, 1.f, 1.f),
          [](const Vector3f value) { ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Diffuse start"); })
    , diffuse_end(
          Vector3f(1.f, 1.f, 1.f),
          [](const Vector3f value) { ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Diffuse end"); })
    , specular_start(
          Vector3f(1.f, 1.f, 1.f),
          [](const Vector3f value) { ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Specular start"); })
    , specular_end(
          Vector3f(1.f, 1.f, 1.f),
          [](const Vector3f value) { ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Specular end"); })
    , emissive_start(
          Vector3f(1.f, 1.f, 1.f),
          [](const Vector3f value) { ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Emissive start"); })
    , emissive_end(Vector3f(1.f, 1.f, 1.f), [](const Vector3f value) {
        ValidatorMinMaxInclusive(value, Vector3f(0.f), Vector3f(1.f), "Emissive end");
    })
{
}

template<>
TransferFunctionInterface::DataOut::TransferFunctionInterfaceData()
{
}

template<>
TransferFunctionInterface::DataOut::Component::Component()
{
}

/**
 * Copy a post process Denoise config interface structure to a post process Denoise config POD structure.
 */
template<>
TransferFunctionInterface::DataOut TransferFunctionInterface::Get()
{
    AccessGuardConst access(this);

    TransferFunctionInterface::DataOut data_out;

    data_out.shading_profile  = access->shading_profile;
    data_out.blending_profile = access->blending_profile;
    data_out.global_opacity   = access->global_opacity.Get();
    data_out.density_scale    = access->density_scale.Get();
    data_out.gradient_scale   = access->gradient_scale.Get();
    data_out.hidden_regions   = access->hidden_regions;

    data_out.components.clear();
    for (auto &&comp_in : access->components)
    {
        data_out.components.emplace_back();
        TransferFunctionInterface::DataOut::Component &comp_out = data_out.components.back();

        comp_out.range              = comp_in.range.Get();
        comp_out.active_regions     = comp_in.active_regions;
        comp_out.opacity_profile    = comp_in.opacity_profile;
        comp_out.opacity_transition = comp_in.opacity_transition.Get();
        comp_out.opacity            = comp_in.opacity.Get();
        comp_out.roughness          = comp_in.roughness.Get();
        comp_out.emissive_strength  = comp_in.emissive_strength.Get();
        comp_out.diffuse_start      = comp_in.diffuse_start.Get();
        comp_out.diffuse_end        = comp_in.diffuse_end.Get();
        comp_out.specular_start     = comp_in.specular_start.Get();
        comp_out.specular_end       = comp_in.specular_end.Get();
        comp_out.emissive_start     = comp_in.emissive_start.Get();
        comp_out.emissive_end       = comp_in.emissive_end.Get();
    }

    return data_out;
}

} // namespace clara::viz
