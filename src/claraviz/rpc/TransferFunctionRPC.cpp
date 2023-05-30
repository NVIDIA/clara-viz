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

#include "claraviz/rpc/TransferFunctionRPC.h"
#include "claraviz/rpc/CameraRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void TransferFunctionContext::ExecuteRPC(cinematic_v1::TransferFunctionRequest &request,
                                         cinematic_v1::TransferFunctionResponse &response)
{
    TransferFunctionInterface::AccessGuard access(GetResources()->transfer_function_);

    switch (request.shading_profile())
    {
    case cinematic_v1::TransferFunctionRequest::BRDF_ONLY:
        access->shading_profile = TransferFunctionShadingProfile::BRDF_ONLY;
        break;
    case cinematic_v1::TransferFunctionRequest::PHASE_ONLY:
        access->shading_profile = TransferFunctionShadingProfile::PHASE_ONLY;
        break;
    case cinematic_v1::TransferFunctionRequest::HYBRID:
        access->shading_profile = TransferFunctionShadingProfile::HYBRID;
        break;
    case cinematic_v1::TransferFunctionRequest::SHADING_PROFILE_UNKNOWN:
        break;
    default:
        Log(LogLevel::Warning) << "Unhandled transfer function shading profile " << request.shading_profile();
        break;
    }

    switch (request.blending_profile())
    {
    case cinematic_v1::TransferFunctionRequest::MAXIMUM_OPACITY:
        access->blending_profile = TransferFunctionBlendingProfile::MAXIMUM_OPACITY;
        break;
    case cinematic_v1::TransferFunctionRequest::BLENDED_OPACITY:
        access->blending_profile = TransferFunctionBlendingProfile::BLENDED_OPACITY;
        break;
    case cinematic_v1::TransferFunctionRequest::BLENDING_PROFILE_UNKNOWN:
        break;
    default:
        Log(LogLevel::Warning) << "Unhandled transfer function blending profile " << request.blending_profile();
        break;
    }

    if (request.global_opacity())
    {
        access->global_opacity.Set(request.global_opacity());
    }

    if (request.density_scale())
    {
        access->density_scale.Set(request.density_scale());
    }

    if (request.gradient_scale())
    {
        access->gradient_scale.Set(request.gradient_scale());
    }

    access->hidden_regions.resize(request.hidden_regions_size());
    for (size_t index = 0; index < request.hidden_regions_size(); ++index)
    {
        access->hidden_regions[index] = request.hidden_regions(index);
    }

    access->components.clear();
    for (size_t index = 0; index < request.components_size(); ++index)
    {
        const cinematic_v1::TransferFunctionRequest::Component &comp_in = request.components(index);

        access->components.emplace_back();
        TransferFunctionInterface::DataIn::Component &comp_out = access->components.back();

        if (comp_in.has_range())
        {
            comp_out.range.Set(MakeVector2f(comp_in.range()));
        }

        comp_out.active_regions.resize(comp_in.active_regions_size());
        for (size_t index = 0; index < comp_in.active_regions_size(); ++index)
        {
            comp_out.active_regions[index] = comp_in.active_regions(index);
        }

        switch (comp_in.opacity_profile())
        {
        case cinematic_v1::TransferFunctionRequest::Component::SQUARE:
            comp_out.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
            break;
        case cinematic_v1::TransferFunctionRequest::Component::TRIANGLE:
            comp_out.opacity_profile = TransferFunctionOpacityProfile::TRIANGLE;
            break;
        case cinematic_v1::TransferFunctionRequest::Component::SINE:
            comp_out.opacity_profile = TransferFunctionOpacityProfile::SINE;
            break;
        case cinematic_v1::TransferFunctionRequest::Component::TRAPEZIOD:
            comp_out.opacity_profile = TransferFunctionOpacityProfile::TRAPEZIOD;
            break;
        case cinematic_v1::TransferFunctionRequest::Component::OPACITY_PROFILE_UNKNOWN:
            break;
        default:
            Log(LogLevel::Warning) << "Unhandled transfer function opacity profile " << comp_in.opacity_profile();
            break;
        }

        if (comp_in.opacity_transition())
        {
            comp_out.opacity_transition.Set(comp_in.opacity_transition());
        }

        if (comp_in.opacity())
        {
            comp_out.opacity.Set(comp_in.opacity());
        }
        if (comp_in.roughness())
        {
            comp_out.roughness.Set(comp_in.roughness());
        }
        if (comp_in.emissive_strength())
        {
            comp_out.emissive_strength.Set(comp_in.emissive_strength());
        }

        if (comp_in.has_diffuse_start())
        {
            comp_out.diffuse_start.Set(MakeVector3f(comp_in.diffuse_start()));
        }
        if (comp_in.has_diffuse_end())
        {
            comp_out.diffuse_end.Set(MakeVector3f(comp_in.diffuse_end()));
        }
        if (comp_in.has_specular_start())
        {
            comp_out.specular_start.Set(MakeVector3f(comp_in.specular_start()));
        }
        if (comp_in.has_specular_end())
        {
            comp_out.specular_end.Set(MakeVector3f(comp_in.specular_end()));
        }
        if (comp_in.has_emissive_start())
        {
            comp_out.emissive_start.Set(MakeVector3f(comp_in.emissive_start()));
        }
        if (comp_in.has_emissive_end())
        {
            comp_out.emissive_end.Set(MakeVector3f(comp_in.emissive_end()));
        }
    }
}

} // namespace detail

} // namespace clara::viz
