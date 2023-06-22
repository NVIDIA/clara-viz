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

#pragma once

#include <nlohmann/json.hpp>

#include "claraviz/interface/CameraInterface.h"
#include "claraviz/interface/CameraApertureInterface.h"
#include "claraviz/interface/DataInterface.h"
#include "claraviz/interface/DataViewInterface.h"
#include "claraviz/interface/ImageInterface.h"
#include "claraviz/interface/VideoInterface.h"
#include "claraviz/interface/LightInterface.h"
#include "claraviz/interface/PostProcessDenoiseInterface.h"
#include "claraviz/interface/PostProcessTonemapInterface.h"
#include "claraviz/interface/RenderSettingsInterface.h"
#include "claraviz/interface/TransferFunctionInterface.h"
#include "claraviz/interface/ViewInterface.h"

namespace clara::viz
{

class JsonInterface
{
public:
    /**
     * Initialize Json settings with interfaces
     *
     * @param background_light_interface [in]
     * @param camera_interface [in]
     * @param camera_aperture_interface [in]
     * @param data_config_interface [in]
     * @param data_histogram_interface [in]
     * @param data_crop_interface [in]
     * @param data_transform_interface [in]
     * @param data_view_interface [in]
     * @param light_interface [in]
     * @param post_process_denoise_interface [in]
     * @param post_process_tonemap_interface [in]
     * @param render_settings_interface [in]
     * @param transfer_function_interface [in]
     * @param view_interface [in]
     */
    JsonInterface(BackgroundLightInterface *background_light_interface, CameraInterface *camera_interface,
                  CameraApertureInterface *camera_aperture_interface, DataConfigInterface *data_config_interface,
                  DataHistogramInterface *data_histogram_interface, DataCropInterface *data_crop_interface,
                  DataTransformInterface *data_transform_interface, DataViewInterface *data_view_interface,
                  LightInterface *light_interface, PostProcessDenoiseInterface *post_process_denoise_interface,
                  PostProcessTonemapInterface *post_process_tonemap_interface,
                  RenderSettingsInterface *render_settings_interface,
                  TransferFunctionInterface *transfer_function_interface, ViewInterface *view_interface);
    JsonInterface() = delete;

    /**
     * Initialize Json settings from current renderer settings
     *
     */
    void InitSettings();

    /**
     * Deduce settings from configured data (data needs to be configured by DataConfigInterface).
     * Make the whole dataset visible. Set a light in correct distance. Set a transfer function
     * using the histogram of the data.
     *
     * @param view_mode [in] view mode
     */
    void DeduceSettings(ViewMode view_mode);

    /**
     * Get current renderer settings as Json
     *
     * @returns json with settings
     */
    nlohmann::json GetSettings() const;

    /**
     * Set settings
     *
     * @param new_settings [in] json with new settings
     */
    void SetSettings(const nlohmann::json &new_settings);

    /**
     * Merge settings
     *
     * @param new_settings [in] json with settings to merge in
     */
    void MergeSettings(const nlohmann::json &new_settings);

private:
    nlohmann::json settings_;

    BackgroundLightInterface *const background_light_interface_;
    CameraInterface *const camera_interface_;
    CameraApertureInterface *const camera_aperture_interface_;
    DataConfigInterface *const data_config_interface_;
    DataHistogramInterface *const data_histogram_interface_;
    DataCropInterface *const data_crop_interface_;
    DataTransformInterface *const data_transform_interface_;
    DataViewInterface *const data_view_interface_;
    LightInterface *const light_interface_;
    PostProcessDenoiseInterface *const post_process_denoise_interface_;
    PostProcessTonemapInterface *const post_process_tonemap_interface_;
    RenderSettingsInterface *const render_settings_interface_;
    TransferFunctionInterface *const transfer_function_interface_;
    ViewInterface *const view_interface_;
};

} // namespace clara::viz