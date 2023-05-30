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

#include "claraviz/rpc/ResetRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void ResetContext::ExecuteRPC(cinematic_v1::ResetRequest &request, cinematic_v1::ResetResponse &response)
{
    for (size_t index = 0; index < request.interfaces_size(); ++index)
    {
        switch (request.interfaces(index))
        {
        case cinematic_v1::ResetRequest::ALL:
            GetResources()->background_light_interface_.Reset();
            GetResources()->camera_interface_.Reset();
            GetResources()->camera_aperture_interface_.Reset();
            GetResources()->data_config_interface_.Reset();
            GetResources()->data_crop_interface_.Reset();
            GetResources()->data_transform_interface_.Reset();
            GetResources()->data_interface_->Reset();
            GetResources()->data_view_interface_.Reset();
            GetResources()->light_interface_.Reset();
            GetResources()->post_process_denoise_interface_.Reset();
            GetResources()->post_process_tonemap_interface_.Reset();
            GetResources()->render_settings_interface_.Reset();
            GetResources()->transfer_function_interface_.Reset();
            GetResources()->view_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::BACKGROUND_LIGHT:
            GetResources()->background_light_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::CAMERA:
            GetResources()->camera_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::CAMERA_APERTURE:
            GetResources()->camera_aperture_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::DATA_CONFIG:
            GetResources()->data_config_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::DATA_CROP:
            GetResources()->data_crop_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::DATA_TRANSFORM:
            GetResources()->data_transform_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::DATA:
            GetResources()->data_interface_->Reset();
            break;
        case cinematic_v1::ResetRequest::DATA_VIEW:
            GetResources()->data_view_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::LIGHT:
            GetResources()->light_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::POST_PROCESS_DENOISE:
            GetResources()->post_process_denoise_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::POST_PROCESS_TONEMAP:
            GetResources()->post_process_tonemap_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::RENDER_SETTINGS:
            GetResources()->render_settings_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::TRANSFER_FUNCTION:
            GetResources()->transfer_function_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::VIEW:
            GetResources()->view_interface_.Reset();
            break;
        case cinematic_v1::ResetRequest::INTERFACE_UNKNOWN:
            break;
        default:
            Log(LogLevel::Warning) << "Unhandled reset interface " << request.interfaces(index);
            break;
        }
    }
}

} // namespace detail

} // namespace clara::viz
