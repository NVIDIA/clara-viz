/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/JsonInterface.h"

namespace clara::viz
{

/// json conversion helper functions

class Range2f : public Vector2f
{
public:
    Range2f() = default;
    Range2f(const Vector2f &v)
        : Vector2f(v)
    {
    }
};

// Helper class to handle converting from bool or gRPC Switch to bool.
class Switch
{
public:
    Switch(bool initial_value = false)
        : value(initial_value)
    {
    }

    operator bool() const
    {
        return value;
    }

    bool value;
};

// don't define a 'to_json' function for switch, default should be the json boolean, we handle
// SWITCH_? only to easily use gRPC JSON files
//void to_json(nlohmann::json &j, Switch &v)
//{
//    j = nlohmann::json{v ? "SWITCH_ENABLE" : "SWITCH_DISABLE"};
//}

void from_json(const nlohmann::json &j, Switch &v)
{
    if (j.type() == nlohmann::json::value_t::boolean)
    {
        v.value = j.get<bool>();
    }
    else if (j.type() == nlohmann::json::value_t::string)
    {
        if (j.get<std::string>() == "SWITCH_ENABLE")
        {
            v.value = true;
        }
        else if (j.get<std::string>() == "SWITCH_DISABLE")
        {
            v.value = false;
        }
        else
        {
            throw RuntimeError() << "Cannot convert " << j.get<std::string>() << " to bool";
        }
    }
    else
    {
        throw RuntimeError() << "Cannot convert to bool";
    }
}

void to_json(nlohmann::json &j, const Range2f &v)
{
    j = nlohmann::json{{"min", v(0)}, {"max", v(1)}};
}

void from_json(const nlohmann::json &j, Range2f &v)
{
    v(0) = j.value("min", 0.f);
    v(1) = j.value("max", 0.f);
}

void to_json(nlohmann::json &j, const Vector2f &v)
{
    j = nlohmann::json{{"x", v(0)}, {"y", v(1)}};
}

void from_json(const nlohmann::json &j, Vector2f &v)
{
    v(0) = j.value("x", 0.f);
    v(1) = j.value("y", 0.f);
}

void to_json(nlohmann::json &j, const Vector3f &v)
{
    j = nlohmann::json{{"x", v(0)}, {"y", v(1)}, {"z", v(2)}};
}

void from_json(const nlohmann::json &j, Vector3f &v)
{
    v(0) = j.value("x", 0.f);
    v(1) = j.value("y", 0.f);
    v(2) = j.value("z", 0.f);
}

void to_json(nlohmann::json &j, const Matrix4x4 &v)
{
    j = nlohmann::json();
    for (uint32_t row = 0; row < 4; ++row)
    {
        for (uint32_t col = 0; col < 4; ++col)
        {
            j.push_back(v(row, col));
        }
    }
}

void from_json(const nlohmann::json &j, Matrix4x4 &v)
{
    v.Identity();
    uint32_t row = 0, col = 0;
    for (auto it = j.begin(); it != j.end(); ++it)
    {
        v(row, col) = it.value();
        ++col;
        if (col == 4)
        {
            col = 0;
            ++row;
            if (row == 4)
            {
                break;
            }
        }
    }
}

void to_json(nlohmann::json &j, const BackgroundLightInterface::AccessGuardConst &background_light)
{
    j = nlohmann::json{{"intensity", background_light->intensity.Get()},
                       {"topColor", background_light->top_color.Get()},
                       {"horizonColor", background_light->horizon_color.Get()},
                       {"bottomColor", background_light->bottom_color.Get()},
                       {"enable", background_light->enable},
                       {"show", background_light->show},
                       {"castLight", background_light->cast_light}};
}

void from_json(const nlohmann::json &j, BackgroundLightInterface::AccessGuard &background_light)
{
    background_light->intensity.Set(j.value("intensity", background_light->intensity.Get()));
    background_light->top_color.Set(j.value("topColor", background_light->top_color.Get()));
    background_light->horizon_color.Set(j.value("horizonColor", background_light->horizon_color.Get()));
    background_light->bottom_color.Set(j.value("color", background_light->bottom_color.Get()));
    background_light->enable     = j.value("enable", Switch(background_light->enable));
    background_light->show       = j.value("show", Switch(background_light->show));
    background_light->cast_light = j.value("castLight", Switch(background_light->cast_light));
}

void to_json(nlohmann::json &j, const CameraApertureInterface::AccessGuardConst &camera_aperture)
{
    j = nlohmann::json{{"enable", camera_aperture->enable},
                       {"aperture", camera_aperture->aperture.Get()},
                       {"autoFocus", camera_aperture->auto_focus},
                       {"focusDistance", camera_aperture->focus_distance.Get()}};
}

void from_json(const nlohmann::json &j, CameraApertureInterface::AccessGuard &camera_aperture)
{
    camera_aperture->enable = j.value("enable", Switch(camera_aperture->enable));
    camera_aperture->aperture.Set(j.value("aperture", camera_aperture->aperture.Get()));
    camera_aperture->auto_focus = j.value("autoFocus", Switch(camera_aperture->auto_focus));
    camera_aperture->focus_distance.Set(j.value("focusDistance", camera_aperture->focus_distance.Get()));
}

void to_json(nlohmann::json &j, const CameraInterface::DataIn::Camera &camera)
{
    j = nlohmann::json{
        {"name", camera.name},
        {"enablePose", camera.enable_pose},
        {"eye", camera.eye.Get()},
        {"lookAt", camera.look_at.Get()},
        {"up", camera.up.Get()},
        {"pose", camera.pose},
        {"fieldOfView", camera.field_of_view.Get()},
        {"pixelAspectRatio", camera.pixel_aspect_ratio.Get()},
        {"enableStereo", camera.enable_stereo},
        {"leftEyePose", camera.left_eye_pose},
        {"rightEyePose", camera.right_eye_pose},
        {"leftGazeDirection", camera.left_gaze_direction.Get()},
        {"rightGazeDirection", camera.right_gaze_direction.Get()},
        {"leftTangentX", camera.left_tangent_x},
        {"leftTangentY", camera.left_tangent_y},
        {"rightTangentX", camera.right_tangent_x},
        {"rightTangentY", camera.right_tangent_y},
    };

    // json is using ranges with "min" and "max", but render settings interface is using Vector2F, manually cast
    j["depthClip"]  = static_cast<Range2f>(camera.depth_clip.Get());
    j["depthRange"] = static_cast<Range2f>(camera.depth_range.Get());
}

void from_json(const nlohmann::json &j, CameraInterface::DataIn::Camera &camera)
{
    camera.name        = j.value("name", camera.name);
    camera.enable_pose = j.value("enablePose", camera.enable_pose);
    camera.eye.Set(j.value("eye", camera.eye.Get()));
    camera.look_at.Set(j.value("lookAt", camera.look_at.Get()));
    camera.up.Set(j.value("up", camera.up.Get()));
    camera.pose = j.value("leftEyePose", camera.pose);
    camera.field_of_view.Set(j.value("fieldOfView", camera.field_of_view.Get()));
    camera.pixel_aspect_ratio.Set(j.value("pixelAspectRatio", camera.pixel_aspect_ratio.Get()));
    camera.enable_stereo  = j.value("enableStereo", camera.enable_stereo);
    camera.left_eye_pose  = j.value("leftEyePose", camera.left_eye_pose);
    camera.right_eye_pose = j.value("rightEyePose", camera.right_eye_pose);
    camera.left_gaze_direction.Set(j.value("leftGazeDirection", camera.left_gaze_direction.Get()));
    camera.right_gaze_direction.Set(j.value("rightGazeDirection", camera.right_gaze_direction.Get()));
    camera.left_tangent_x  = j.value("leftTangentX", camera.left_tangent_x);
    camera.left_tangent_y  = j.value("leftTangentY", camera.left_tangent_y);
    camera.right_tangent_x = j.value("rightTangentX", camera.right_tangent_x);
    camera.right_tangent_y = j.value("rightTangentY", camera.right_tangent_y);

    // json is using ranges with "min" and "max", but render settings interface is using Vector2F, manually cast
    if (j.contains("depth_clip"))
    {
        Range2f r;
        j["depth_clip"].get_to(r);
        camera.depth_clip.Set(static_cast<Vector2f>(r));
    }
    if (j.contains("depth_range"))
    {
        Range2f r;
        j["depth_range"].get_to(r);
        camera.depth_range.Set(static_cast<Vector2f>(r));
    }
}

void to_json(nlohmann::json &j, const DataCropInterface::AccessGuardConst &data_crop)
{
    // json is using ranges with "min" and "max", but data crop interface is using Vector2f, manually cast
    for (auto &&limit : data_crop->limits.Get())
    {
        j["limits"].push_back(static_cast<Range2f>(limit));
    }
}

void from_json(const nlohmann::json &j, DataCropInterface::AccessGuard &data_crop)
{
    // json is using ranges with "min" and "max", but data crop interface is using Vector2F, manually cast
    if (j.contains("limits"))
    {
        std::vector<Vector2f> limits;
        for (auto &&limit : j["limits"])
        {
            Range2f l;
            limit.get_to(l);
            limits.push_back(static_cast<Vector2f>(l));
        }
        data_crop->limits.Set(limits);
    }
}

void to_json(nlohmann::json &j, const DataTransformInterface::AccessGuardConst &data_transform)
{
    j = nlohmann::json{{"matrix", data_transform->matrix}};
}

void from_json(const nlohmann::json &j, DataTransformInterface::AccessGuard &data_transform)
{
    data_transform->matrix = j.value("matrix", data_transform->matrix);
}

void to_json(nlohmann::json &j, const DataViewInterface::DataIn::DataView &data_view)
{
    j = nlohmann::json{{"name", data_view.name},
                       {"zoomFactor", data_view.zoom_factor.Get()},
                       {"viewOffset", data_view.view_offset},
                       {"pixelAspectRatio", data_view.pixel_aspect_ratio.Get()}};
}

void from_json(const nlohmann::json &j, DataViewInterface::DataIn::DataView &data_view)
{
    data_view.name = j.value("name", data_view.name);
    data_view.zoom_factor.Set(j.value("zoomFactor", data_view.zoom_factor.Get()));
    data_view.view_offset = j.value("viewOffset", data_view.view_offset);
    data_view.pixel_aspect_ratio.Set(j.value("pixelAspectRatio", data_view.pixel_aspect_ratio.Get()));
}

void to_json(nlohmann::json &j, const LightInterface::DataIn::Light &light)
{
    j = nlohmann::json{{"position", light.position}, {"direction", light.direction.Get()},
                       {"size", light.size.Get()},   {"intensity", light.intensity.Get()},
                       {"color", light.color.Get()}, {"enable", light.enable},
                       {"show", light.show}};
}

void from_json(const nlohmann::json &j, LightInterface::DataIn::Light &light)
{
    light.position = j.value("position", light.position);
    light.direction.Set(j.value("direction", light.direction.Get()));
    light.size.Set(j.value("size", light.size.Get()));
    light.intensity.Set(j.value("intensity", light.intensity.Get()));
    light.color.Set(j.value("color", light.color.Get()));
    light.enable = j.value("enable", Switch(light.enable));
    light.show   = j.value("show", Switch(light.show));
}

void to_json(nlohmann::json &j, const PostProcessDenoiseInterface::AccessGuardConst &post_process_denoise)
{
    j = nlohmann::json{{"method", post_process_denoise->method},
                       {"radius", post_process_denoise->radius.Get()},
                       {"spatialWeight", post_process_denoise->spatial_weight.Get()},
                       {"depthWeight", post_process_denoise->depth_weight.Get()},
                       {"noiseThreshold", post_process_denoise->noise_threshold.Get()},
                       {"enableIterationLimit", post_process_denoise->enable_iteration_limit},
                       {"iterationLimit", post_process_denoise->iteration_limit.Get()}};
}

void from_json(const nlohmann::json &j, PostProcessDenoiseInterface::AccessGuard &post_process_denoise)
{
    post_process_denoise->method = j.value("method", post_process_denoise->method);
    post_process_denoise->radius.Set(j.value("radius", post_process_denoise->radius.Get()));
    post_process_denoise->spatial_weight.Set(j.value("spatialWeight", post_process_denoise->spatial_weight.Get()));
    post_process_denoise->depth_weight.Set(j.value("depthWeight", post_process_denoise->depth_weight.Get()));
    post_process_denoise->noise_threshold.Set(j.value("noiseThreshold", post_process_denoise->noise_threshold.Get()));
    post_process_denoise->enable_iteration_limit =
        j.value("enableIterationLimit", Switch(post_process_denoise->enable_iteration_limit));
    post_process_denoise->iteration_limit.Set(j.value("iterationLimit", post_process_denoise->iteration_limit.Get()));
}

void to_json(nlohmann::json &j, const PostProcessTonemapInterface::AccessGuardConst &post_process_tonemap)
{
    j = nlohmann::json{{"enable", post_process_tonemap->enable}, {"exposure", post_process_tonemap->exposure.Get()}};
}

void from_json(const nlohmann::json &j, PostProcessTonemapInterface::AccessGuard &post_process_tonemap)
{
    post_process_tonemap->enable = j.value("enable", Switch(post_process_tonemap->enable));
    post_process_tonemap->exposure.Set(j.value("exposure", post_process_tonemap->exposure.Get()));
}

void to_json(nlohmann::json &j, const RenderSettingsInterface::AccessGuardConst &render_settings)
{
    j = nlohmann::json{{"interpolationMode", render_settings->interpolation_mode},
                       {"stepSize", render_settings->step_size.Get()},
                       {"shadowStepSize", render_settings->shadow_step_size.Get()},
                       {"maxIterations", render_settings->max_iterations.Get()},
                       {"timeSlot", render_settings->time_slot.Get()},
                       {"enableWarp", render_settings->enable_warp},
                       {"warpResolutionScale", render_settings->warp_resolution_scale.Get()},
                       {"warpFullResolutionSize", render_settings->warp_full_resolution_size.Get()},
                       {"enableFoveation", render_settings->enable_foveation},
                       {"enableReproject", render_settings->enable_reproject},
                       {"enableSeparateDepth", render_settings->enable_separate_depth}};
}

void from_json(const nlohmann::json &j, RenderSettingsInterface::AccessGuard &render_settings)
{
    render_settings->interpolation_mode = j.value("interpolationMode", render_settings->interpolation_mode);
    render_settings->step_size.Set(j.value("stepSize", render_settings->step_size.Get()));
    render_settings->shadow_step_size.Set(j.value("shadowStepSize", render_settings->shadow_step_size.Get()));
    render_settings->max_iterations.Set(j.value("maxIterations", render_settings->max_iterations.Get()));
    render_settings->time_slot.Set(j.value("timeSlot", render_settings->time_slot.Get()));
    render_settings->enable_warp = j.value("enableWarp", Switch(render_settings->enable_warp));
    render_settings->warp_resolution_scale.Set(
        j.value("warpResolutionScale", render_settings->warp_resolution_scale.Get()));
    render_settings->warp_full_resolution_size.Set(
        j.value("warpFullResolutionSize", render_settings->warp_full_resolution_size.Get()));
    render_settings->enable_foveation = j.value("enableFoveation", Switch(render_settings->enable_foveation));
    render_settings->enable_reproject = j.value("enableReproject", Switch(render_settings->enable_reproject));
    render_settings->enable_separate_depth =
        j.value("enableSeparateDepth", Switch(render_settings->enable_separate_depth));
}

void to_json(nlohmann::json &j, const TransferFunctionInterface::DataIn::Component &component)
{
    j = nlohmann::json{{"activeRegions", component.active_regions},
                       {"opacityProfile", component.opacity_profile},
                       {"opacityTransition", component.opacity_transition.Get()},
                       {"opacity", component.opacity.Get()},
                       {"roughness", component.roughness.Get()},
                       {"emissiveStrength", component.emissive_strength.Get()},
                       {"diffuseStart", component.diffuse_start.Get()},
                       {"diffuseEnd", component.diffuse_end.Get()},
                       {"specularStart", component.specular_start.Get()},
                       {"specularEnd", component.specular_end.Get()},
                       {"emissiveStart", component.emissive_start.Get()},
                       {"emissiveEnd", component.emissive_end.Get()}};

    // json is using ranges with "min" and "max", but transfer function component interface is using Vector2F, manually cast
    j["range"] = static_cast<Range2f>(component.range.Get());
}

void from_json(const nlohmann::json &j, TransferFunctionInterface::DataIn::Component &component)
{
    // json is using ranges with "min" and "max", but transfer function component interface is using Vector2F, manually cast
    if (j.contains("range"))
    {
        Range2f r;
        j["range"].get_to(r);
        component.range.Set(static_cast<Vector2f>(r));
    }

    component.active_regions  = j.value("activeRegions", component.active_regions);
    component.opacity_profile = j.value("opacityProfile", component.opacity_profile);
    component.opacity_transition.Set(j.value("opacityTransition", component.opacity_transition.Get()));
    component.opacity.Set(j.value("opacity", component.opacity.Get()));
    component.roughness.Set(j.value("roughness", component.roughness.Get()));
    component.emissive_strength.Set(j.value("emissiveStrength", component.emissive_strength.Get()));
    component.diffuse_start.Set(j.value("diffuseStart", component.diffuse_start.Get()));
    component.diffuse_end.Set(j.value("diffuseEnd", component.diffuse_end.Get()));
    component.specular_start.Set(j.value("specularStart", component.specular_start.Get()));
    component.specular_end.Set(j.value("specularEnd", component.specular_end.Get()));
    component.emissive_start.Set(j.value("emissiveStart", component.emissive_start.Get()));
    component.emissive_end.Set(j.value("emissiveEnd", component.emissive_end.Get()));
}

void to_json(nlohmann::json &j, const TransferFunctionInterface::AccessGuardConst &transfer_function)
{
    j = nlohmann::json{{"shadingProfile", transfer_function->shading_profile},
                       {"blendingProfile", transfer_function->blending_profile},
                       {"globalOpacity", transfer_function->global_opacity.Get()},
                       {"densityScale", transfer_function->density_scale.Get()},
                       {"gradientScale", transfer_function->gradient_scale.Get()},
                       {"hiddenRegions", transfer_function->hidden_regions}};

    for (auto &&component : transfer_function->components)
    {
        j["components"].push_back(component);
    }
}

void from_json(const nlohmann::json &j, TransferFunctionInterface::AccessGuard &transfer_function)
{
    transfer_function->shading_profile  = j.value("shadingProfile", transfer_function->shading_profile);
    transfer_function->blending_profile = j.value("blendingProfile", transfer_function->blending_profile);
    transfer_function->global_opacity.Set(j.value("globalOpacity", transfer_function->global_opacity.Get()));
    transfer_function->density_scale.Set(j.value("densityScale", transfer_function->density_scale.Get()));
    transfer_function->gradient_scale.Set(j.value("gradientScale", transfer_function->gradient_scale.Get()));
    transfer_function->hidden_regions = j.value("hiddenRegions", transfer_function->hidden_regions);
    transfer_function->components.clear();
    if (j.contains("components"))
    {
        for (auto &&component : j["components"])
        {
            transfer_function->components.emplace_back();
            std::list<TransferFunctionInterface::DataIn::Component>::iterator it = transfer_function->components.end();
            --it;
            component.get_to(*it);
        }
    }
}

void to_json(nlohmann::json &j, const ViewInterface::DataIn::View &view)
{
    j = nlohmann::json{{"name", view.name},
                       {"streamName", view.stream_name},
                       {"mode", view.mode},
                       {"cameraName", view.camera_name},
                       {"dataViewName", view.data_view_name},
                       {"stereoMode", view.stereo_mode}};
}

void from_json(const nlohmann::json &j, ViewInterface::DataIn::View &view)
{
    view.name           = j.value("name", view.name);
    view.stream_name    = j.value("streamName", view.stream_name);
    view.mode           = j.value("mode", view.mode);
    view.camera_name    = j.value("cameraName", view.camera_name);
    view.data_view_name = j.value("dataViewName", view.data_view_name);
    view.stereo_mode    = j.value("stereoMode", view.stereo_mode);
}

/// map enum values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ViewMode, {{ViewMode::CINEMATIC, "CINEMATIC"},
                                        {ViewMode::SLICE, "SLICE"},
                                        {ViewMode::SLICE_SEGMENTATION, "SLICE_SEGMENTATION"},
                                        {ViewMode::TWOD, "TWOD"}})
NLOHMANN_JSON_SERIALIZE_ENUM(DenoiseMethod,
                             {{DenoiseMethod::OFF, "OFF"}, {DenoiseMethod::KNN, "KNN"}, {DenoiseMethod::AI, "AI"}})
NLOHMANN_JSON_SERIALIZE_ENUM(TransferFunctionShadingProfile,
                             {{TransferFunctionShadingProfile::BRDF_ONLY, "BRDF_ONLY"},
                              {TransferFunctionShadingProfile::PHASE_ONLY, "PHASE_ONLY"},
                              {TransferFunctionShadingProfile::HYBRID, "HYBRID"}})
NLOHMANN_JSON_SERIALIZE_ENUM(TransferFunctionBlendingProfile,
                             {{TransferFunctionBlendingProfile::MAXIMUM_OPACITY, "MAXIMUM_OPACITY"},
                              {TransferFunctionBlendingProfile::BLENDED_OPACITY, "BLENDED_OPACITY"}})
NLOHMANN_JSON_SERIALIZE_ENUM(TransferFunctionOpacityProfile, {{TransferFunctionOpacityProfile::SQUARE, "SQUARE"},
                                                              {TransferFunctionOpacityProfile::TRIANGLE, "TRIANGLE"},
                                                              {TransferFunctionOpacityProfile::SINE, "SINE"},
                                                              {TransferFunctionOpacityProfile::TRAPEZIOD, "TRAPEZIOD"}})
NLOHMANN_JSON_SERIALIZE_ENUM(InterpolationMode, {{InterpolationMode::LINEAR, "LINEAR"},
                                                 {InterpolationMode::BSPLINE, "BSPLINE"},
                                                 {InterpolationMode::CATMULLROM, "CATMULLROM"}})
NLOHMANN_JSON_SERIALIZE_ENUM(StereoMode, {{StereoMode::OFF, "OFF"},
                                          {StereoMode::LEFT, "LEFT"},
                                          {StereoMode::RIGHT, "RIGHT"},
                                          {StereoMode::TOP_BOTTOM, "TOP_BOTTOM"}})

/**
 * Set settings for a section and corresponding interface
 *
 * @param settings [in] json with current settings
 * @param new_settings [in] json with new settings
 * @param interface [in] interface
 * @param name [in] name of the section
 */
template<typename INTERFACE_TYPE>
void SetSettingsTemplate(const nlohmann::json &settings, const nlohmann::json &new_settings, INTERFACE_TYPE *interface,
                         const char *name)
{
    if (new_settings.contains(name))
    {
        auto sub_settings = new_settings.at(name);
        if (!settings.contains(name) || !(settings.at(name) == sub_settings))
        {
            typename INTERFACE_TYPE::AccessGuard access(*interface);
            sub_settings.get_to(access);
        }
    }
}

JsonInterface::JsonInterface(BackgroundLightInterface *background_light_interface, CameraInterface *camera_interface,
                             CameraApertureInterface *camera_aperture_interface,
                             DataConfigInterface *data_config_interface,
                             DataHistogramInterface *data_histogram_interface, DataCropInterface *data_crop_interface,
                             DataTransformInterface *data_transform_interface, DataViewInterface *data_view_interface,
                             LightInterface *light_interface,
                             PostProcessDenoiseInterface *post_process_denoise_interface,
                             PostProcessTonemapInterface *post_process_tonemap_interface,
                             RenderSettingsInterface *render_settings_interface,
                             TransferFunctionInterface *transfer_function_interface, ViewInterface *view_interface)
    : background_light_interface_(background_light_interface)
    , camera_interface_(camera_interface)
    , camera_aperture_interface_(camera_aperture_interface)
    , data_config_interface_(data_config_interface)
    , data_histogram_interface_(data_histogram_interface)
    , data_crop_interface_(data_crop_interface)
    , data_transform_interface_(data_transform_interface)
    , data_view_interface_(data_view_interface)
    , light_interface_(light_interface)
    , post_process_denoise_interface_(post_process_denoise_interface)
    , post_process_tonemap_interface_(post_process_tonemap_interface)
    , render_settings_interface_(render_settings_interface)
    , transfer_function_interface_(transfer_function_interface)
    , view_interface_(view_interface)
{
}

void JsonInterface::InitSettings()
{
    settings_.clear();

    settings_["BackgroundLight"] = BackgroundLightInterface::AccessGuardConst(background_light_interface_);
    settings_["CameraAperture"]  = CameraApertureInterface::AccessGuardConst(camera_aperture_interface_);
    {
        CameraInterface::AccessGuardConst access(camera_interface_);

        for (auto &&camera : access->cameras)
        {
            settings_["Cameras"].push_back(camera);
        }
    }
    settings_["DataCrop"]      = DataCropInterface::AccessGuardConst(data_crop_interface_);
    settings_["DataTransform"] = DataTransformInterface::AccessGuardConst(data_transform_interface_);
    {
        DataViewInterface::AccessGuardConst access(data_view_interface_);

        for (auto &&data_view : access->data_views)
        {
            settings_["DataViews"].push_back(data_view);
        }
    }
    {
        LightInterface::AccessGuardConst access(light_interface_);

        for (auto &&light : access->lights)
        {
            settings_["Lights"].push_back(light);
        }
    }
    settings_["PostProcessDenoise"] = PostProcessDenoiseInterface::AccessGuardConst(post_process_denoise_interface_);
    settings_["PostProcessTonemap"] = PostProcessTonemapInterface::AccessGuardConst(post_process_tonemap_interface_);
    settings_["RenderSettings"]     = RenderSettingsInterface::AccessGuardConst(render_settings_interface_);
    settings_["TransferFunction"]   = TransferFunctionInterface::AccessGuardConst(transfer_function_interface_);
    {
        ViewInterface::AccessGuardConst access(view_interface_);

        for (auto &&view : access->views)
        {
            settings_["Views"].push_back(view);
        }
    }
}

void JsonInterface::DeduceSettings(ViewMode view_mode)
{
    {
        DataConfigInterface::AccessGuardConst access(data_config_interface_);

        for (auto &&array : access->arrays)
        {
            switch (array.dimension_order.Get().size())
            {
            case 4:
                if ((view_mode != ViewMode::CINEMATIC) && (view_mode != ViewMode::SLICE_SEGMENTATION) &&
                    (view_mode != ViewMode::SLICE))
                {
                    Log(LogLevel::Warning) << "Detected volume arrays, switching view mode to Cinematic";
                    view_mode = ViewMode::CINEMATIC;
                }
                break;
            case 3:
                if (view_mode != ViewMode::TWOD)
                {
                    Log(LogLevel::Warning) << "Detected 2D arrays, switching view mode to TwoD";
                    view_mode = ViewMode::TWOD;
                }
                break;
            }
        }
    }

    if (view_mode == ViewMode::TWOD)
    {
        {
            DataConfigInterface::AccessGuardConst access(data_config_interface_);

            std::list<DataConfigInterface::DataIn::Array>::const_iterator array = std::find_if(
                access->arrays.begin(), access->arrays.end(),
                [](const DataConfigInterface::DataIn::Array &array) { return array.dimension_order.Get() == "CXY"; });
            if (array == access->arrays.end())
            {
                throw RuntimeError()
                    << "There is no 2D color array defined, but one is required to deduce settings for "
                       "viewing of 2D data";
            }
        }

        // add a data view
        {
            DataViewInterface::AccessGuard access(*data_view_interface_);

            DataViewInterface::DataIn::DataView *data_view;

            data_view = access->GetOrAddDataView("DataView");
            data_view->zoom_factor.Set(1.0f);
            data_view->view_offset = Vector2f(0.0f, 0.0f);
            data_view->pixel_aspect_ratio.Set(1.f);
        }
    }
    else
    {
        std::string density_array_id;
        std::vector<uint32_t> sizes;
        std::vector<float> element_sizes;

        {
            DataConfigInterface::AccessGuardConst access(data_config_interface_);

            std::list<DataConfigInterface::DataIn::Array>::const_iterator array = std::find_if(
                access->arrays.begin(), access->arrays.end(),
                [](const DataConfigInterface::DataIn::Array &array) { return array.dimension_order.Get() == "DXYZ"; });
            if (array == access->arrays.end())
            {
                throw RuntimeError() << "There is no density array defined, but one is required to deduce settings for "
                                        "viewing of volumes";
            }

            density_array_id = array->id;
            sizes            = array->levels.begin()->size.Get();
            element_sizes    = array->levels.begin()->element_size.Get();
            // extend element sizes array to sizes array
            element_sizes.resize(sizes.size(), 1.f);
        }

        // calculate the volume physical size, element spacing is in mm, physical size is in meters
        std::vector<float> physical_sizes(sizes.size());
        for (int index = 0; index < sizes.size(); ++index)
            physical_sizes[index] = sizes[index] * element_sizes[index] / 1000.f;

        // set the cameras
        {
            CameraInterface::AccessGuard access(*camera_interface_);

            const float PI            = std::acos(-1);
            const float field_of_view = 30.0f;
            auto toRadians            = [PI](float degree) { return degree * PI / 180.0f; };
            // calculate the camera distance for a given volume physical size and a field of view
            const float distance =
                (std::max(physical_sizes[1], physical_sizes[2]) * 0.5f) / std::tan(toRadians(field_of_view * 0.5f));

            CameraInterface::DataIn::Camera *camera;

            camera = access->GetOrAddCamera("Perspective");
            camera->eye.Set(Vector3f(0.f, 0.f, -(distance + physical_sizes[3] * 0.5f)));
            camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
            camera->up.Set(Vector3f(0.f, 1.f, 0.f));
            camera->field_of_view.Set(field_of_view);
            camera->pixel_aspect_ratio.Set(1.f);

            camera = access->GetOrAddCamera("Top");
            camera->eye.Set(Vector3f(0.f, 1.f, 0.f));
            camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
            camera->up.Set(Vector3f(0.f, 0.f, 1.f));
            camera->field_of_view.Set(30.f);
            camera->pixel_aspect_ratio.Set(1.f);

            camera = access->GetOrAddCamera("Front");
            camera->eye.Set(Vector3f(0.f, 0.f, 1.f));
            camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
            camera->up.Set(Vector3f(0.f, 1.f, 0.f));
            camera->field_of_view.Set(30.f);
            camera->pixel_aspect_ratio.Set(1.f);

            camera = access->GetOrAddCamera("Right");
            camera->eye.Set(Vector3f(1.f, 0.f, 0.f));
            camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
            camera->up.Set(Vector3f(0.f, 1.f, 0.f));
            camera->field_of_view.Set(30.f);
            camera->pixel_aspect_ratio.Set(1.f);

            camera = access->GetOrAddCamera("Oblique");
            camera->eye.Set(Vector3f(0.f, 1.f, 0.f));
            camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
            camera->up.Set(Vector3f(0.f, 0.f, 1.f));
            camera->field_of_view.Set(30.f);
            camera->pixel_aspect_ratio.Set(1.f);
        }

        // place a light
        {
            LightInterface::AccessGuard access(*light_interface_);

            const float light_distance_squared = (physical_sizes[1] * physical_sizes[1]) +
                                                 (physical_sizes[2] * physical_sizes[2]) +
                                                 (physical_sizes[3] * physical_sizes[3]);

            access->lights[0].position =
                Vector3f(-2.f * physical_sizes[1], 2.f * physical_sizes[2], -2.f * physical_sizes[3]);
            access->lights[0].direction.Set(Vector3f(std::sqrt(1.f / 3.f), std::sqrt(1.f / 3.f), std::sqrt(1.f / 3.f)));
            access->lights[0].size.Set(std::max(physical_sizes[1], std::max(physical_sizes[2], physical_sizes[3])) /
                                       2.f);
            access->lights[0].intensity.Set(light_distance_squared * 10.f);
            access->lights[0].color.Set(Vector3f(1.f, 1.f, 1.f));
            access->lights[0].enable = true;
        }

        // set the background light
        {
            BackgroundLightInterface::AccessGuard access(*background_light_interface_);

            access->intensity.Set(0.5f);
            access->top_color.Set(Vector3f(1.f, 1.f, 1.f));
            access->horizon_color.Set(Vector3f(1.f, 1.f, 1.f));
            access->bottom_color.Set(Vector3f(1.f, 1.f, 1.f));
            access->enable     = true;
            access->cast_light = true;
            access->show       = true;
        }

        // define a transfer function
        {
            TransferFunctionInterface::AccessGuard access(*transfer_function_interface_);

            access->blending_profile = TransferFunctionBlendingProfile::MAXIMUM_OPACITY;
            access->density_scale.Set(100.f);
            access->global_opacity.Set(1.f);
            access->gradient_scale.Set(10.f);
            access->shading_profile = TransferFunctionShadingProfile::HYBRID;

            // get a histogram of the density data, find the maximum and define a transform function
            // around that maximum
            std::vector<float> histogram;
            data_histogram_interface_->GetHistogram(density_array_id, histogram);
            std::vector<float>::iterator start = histogram.begin();
            // skip the first few elements of the histogram, this is usual the air around a scan
            std::advance(start, 5);
            std::vector<float>::iterator max_element = std::max_element(start, histogram.end());
            const float max_pos_normalized = static_cast<float>(std::distance(histogram.begin(), max_element)) /
                                             static_cast<float>(histogram.size() - 1);

            access->components.clear();
            {
                access->components.emplace_back();
                TransferFunctionInterface::DataIn::Component &component = access->components.back();
                component.range.Set(
                    Vector2f(std::max(0.f, max_pos_normalized - 0.025f), std::min(1.f, max_pos_normalized + 0.025f)));
                component.active_regions  = std::vector<uint32_t>({0});
                component.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
                component.opacity_transition.Set(0.2f);
                component.opacity.Set(1.f);
                component.roughness.Set(0.5f);
                component.emissive_strength.Set(0.f);
                component.diffuse_start.Set(Vector3f(.7f, .7f, 1.f));
                component.diffuse_end.Set(Vector3f(.7f, .7f, 1.f));
                component.specular_start.Set(Vector3f(1.f, 1.f, 1.f));
                component.specular_end.Set(Vector3f(1.f, 1.f, 1.f));
                component.emissive_start.Set(Vector3f(0.f, 0.f, 0.f));
                component.emissive_end.Set(Vector3f(0.f, 0.f, 0.f));
            }
        }

        // configure the render settings
        {
            RenderSettingsInterface::AccessGuard access(*render_settings_interface_);

            // set the interation count, the higher the better the image quality but this also increases render time
            access->max_iterations.Set(1000);
        }

        // configure post process denoise settings
        {
            PostProcessDenoiseInterface::AccessGuard access(*post_process_denoise_interface_);

            access->method = DenoiseMethod::AI;
            access->iteration_limit.Set(500);
            access->enable_iteration_limit = true;
        }
    }

    // set the view
    {
        ViewInterface::AccessGuard access(*view_interface_);

        ViewInterface::DataIn::View *view = access->GetOrAddView("");
        switch (view_mode)
        {
        case ViewMode::CINEMATIC:
            view->camera_name = "Perspective";
            break;
        case ViewMode::SLICE:
        case ViewMode::SLICE_SEGMENTATION:
            view->camera_name = "Top";
            break;
        case ViewMode::TWOD:
            view->data_view_name = "DataView";
            break;
        }
        view->mode = view_mode;
    }

    // build json settings
    InitSettings();
}

void JsonInterface::SetSettings(const nlohmann::json &new_settings)
{
    SetSettingsTemplate(settings_, new_settings, background_light_interface_, "BackgroundLight");
    if (new_settings.contains("Cameras"))
    {
        auto cameras_settings = new_settings.at("Cameras");
        if (!settings_.contains("Cameras") || !(settings_.at("Cameras") == cameras_settings))
        {
            CameraInterface::AccessGuard access(*camera_interface_);
            for (auto &&camera : cameras_settings)
            {
                camera.get_to(*access->GetOrAddCamera(camera.value("name", "")));
            }
        }
    }
    // backward compatibility with Clara Deploy
    if (new_settings.contains("Camera"))
    {
        auto camera_settings = new_settings.at("Camera");
        if (!settings_.contains("Camera") || !(settings_.at("Camera") == camera_settings))
        {
            CameraInterface::AccessGuard access(*camera_interface_);
            camera_settings.get_to(*access->GetOrAddCamera(camera_settings.value("name", "")));
        }
    }

    SetSettingsTemplate(settings_, new_settings, camera_aperture_interface_, "CameraAperture");
    SetSettingsTemplate(settings_, new_settings, data_crop_interface_, "DataCrop");
    SetSettingsTemplate(settings_, new_settings, data_transform_interface_, "DataTransform");
    if (new_settings.contains("DataViews"))
    {
        auto data_views_settings = new_settings.at("DataViews");
        if (!settings_.contains("DataViews") || !(settings_.at("DataViews") == data_views_settings))
        {
            DataViewInterface::AccessGuard access(*data_view_interface_);
            for (auto &&data_view : data_views_settings)
            {
                data_view.get_to(*access->GetOrAddDataView(data_view.value("name", "")));
            }
        }
    }
    // backward compatibility with Clara Deploy
    if (new_settings.contains("DataView"))
    {
        auto data_view_settings = new_settings.at("DataView");
        if (!settings_.contains("DataView") || !(settings_.at("DataView") == data_view_settings))
        {
            DataViewInterface::AccessGuard access(*data_view_interface_);
            data_view_settings.get_to(*access->GetOrAddDataView(data_view_settings.value("name", "")));
        }
    }

    if (new_settings.contains("Lights"))
    {
        auto lights_settings = new_settings.at("Lights");
        if (!settings_.contains("Lights") || !(settings_.at("Lights") == lights_settings))
        {
            LightInterface::AccessGuard access(*light_interface_);
            for (size_t index = 0; index < lights_settings.size(); ++index)
            {
                lights_settings[index].get_to(access->lights[index]);
            }
        }
    }
    // backward compatibility with Clara Deploy
    if (new_settings.contains("Light"))
    {
        auto light_settings = new_settings.at("Light");
        if (!settings_.contains("Light") || !(settings_.at("Light") == light_settings))
        {
            LightInterface::AccessGuard access(*light_interface_);
            for (size_t index = 0; index < light_settings.size(); ++index)
            {
                light_settings[index].get_to(access->lights[index]);
            }
        }
    }

    SetSettingsTemplate(settings_, new_settings, post_process_denoise_interface_, "PostProcessDenoise");
    SetSettingsTemplate(settings_, new_settings, post_process_tonemap_interface_, "PostProcessTonemap");
    SetSettingsTemplate(settings_, new_settings, render_settings_interface_, "RenderSettings");
    SetSettingsTemplate(settings_, new_settings, transfer_function_interface_, "TransferFunction");
    if (new_settings.contains("Views"))
    {
        auto views_settings = new_settings.at("Views");
        if (!settings_.contains("Views") || !(settings_.at("Views") == views_settings))
        {
            ViewInterface::AccessGuard access(*view_interface_);
            for (auto &&view : views_settings)
            {
                view.get_to(*access->GetOrAddView(view.value("name", "")));
            }
        }
    }
    // backward compatibility with Clara Deploy
    if (new_settings.contains("View"))
    {
        auto view_settings = new_settings.at("View");
        if (!settings_.contains("View") || !(settings_.at("View") == view_settings))
        {
            ViewInterface::AccessGuard access(*view_interface_);
            view_settings.get_to(*access->GetOrAddView(view_settings.value("name", "")));
        }
    }

    settings_ = new_settings;

    // backward compatibility with Clara Deploy, if the settings contain old fields, rebuild from renderer settings.
    if (settings_.contains("Camera") || settings_.contains("DataView") || settings_.contains("Light") ||
        settings_.contains("Light"))
    {
        InitSettings();
    }
}

void JsonInterface::MergeSettings(const nlohmann::json &new_settings)
{
    nlohmann::json merged = settings_;

    merged.merge_patch(new_settings);

    SetSettings(merged);
}

nlohmann::json JsonInterface::GetSettings() const
{
    return settings_;
}

} // namespace clara::viz