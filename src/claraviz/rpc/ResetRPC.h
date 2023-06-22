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

#pragma once

#include <memory>

#include <nvrpc/Context.h>
#include <nvrpc/Resources.h>
#include <claraviz/rpc/ServerRPC.h>

#include <nvidia/claraviz/cinematic/v1/render_server.grpc.pb.h>

#include <claraviz/interface/CameraInterface.h>

#include "claraviz/interface/CameraApertureInterface.h"
#include "claraviz/interface/DataInterface.h"
#include "claraviz/interface/DataViewInterface.h"
#include "claraviz/interface/LightInterface.h"
#include "claraviz/interface/PostProcessDenoiseInterface.h"
#include "claraviz/interface/PostProcessTonemapInterface.h"
#include "claraviz/interface/RenderSettingsInterface.h"
#include "claraviz/interface/TransferFunctionInterface.h"
#include "claraviz/interface/ViewInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class ResetResource : public nvrpc::Resources
{
public:
    ResetResource(BackgroundLightInterface &background_light_interface, CameraInterface &camera_interface,
                  CameraApertureInterface &camera_aperture_interface, DataConfigInterface &data_config_interface,
                  DataCropInterface &data_crop_interface, DataTransformInterface &data_transform_interface,
                  const std::shared_ptr<DataInterface> &data_interface, DataViewInterface &data_view_interface,
                  LightInterface &light_interface, PostProcessDenoiseInterface &post_process_denoise_interface,
                  PostProcessTonemapInterface &post_process_tonemap_interface,
                  RenderSettingsInterface &render_settings_interface,
                  TransferFunctionInterface &transfer_function_interface, ViewInterface &view_interface)
        : background_light_interface_(background_light_interface)
        , camera_interface_(camera_interface)
        , camera_aperture_interface_(camera_aperture_interface)
        , data_config_interface_(data_config_interface)
        , data_crop_interface_(data_crop_interface)
        , data_transform_interface_(data_transform_interface)
        , data_interface_(data_interface)
        , data_view_interface_(data_view_interface)
        , light_interface_(light_interface)
        , post_process_denoise_interface_(post_process_denoise_interface)
        , post_process_tonemap_interface_(post_process_tonemap_interface)
        , render_settings_interface_(render_settings_interface)
        , transfer_function_interface_(transfer_function_interface)
        , view_interface_(view_interface)
    {
    }
    ResetResource() = delete;

    BackgroundLightInterface &background_light_interface_;
    CameraInterface &camera_interface_;
    CameraApertureInterface &camera_aperture_interface_;
    DataConfigInterface &data_config_interface_;
    DataCropInterface &data_crop_interface_;
    DataTransformInterface &data_transform_interface_;
    const std::shared_ptr<DataInterface> &data_interface_;
    DataViewInterface &data_view_interface_;
    LightInterface &light_interface_;
    PostProcessDenoiseInterface &post_process_denoise_interface_;
    PostProcessTonemapInterface &post_process_tonemap_interface_;
    RenderSettingsInterface &render_settings_interface_;
    TransferFunctionInterface &transfer_function_interface_;
    ViewInterface &view_interface_;
};

/**
 * RPC call context
 */
class ResetContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::ResetRequest,
                                 nvidia::claraviz::cinematic::v1::ResetResponse, ResetResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::ResetRequest &request,
                    nvidia::claraviz::cinematic::v1::ResetResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the Reset class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param reset [in] Reset interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterResetRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                      BackgroundLightInterface &background_light_interface, CameraInterface &camera_interface,
                      CameraApertureInterface &camera_aperture_interface, DataConfigInterface &data_config_interface,
                      DataCropInterface &data_crop_interface, DataTransformInterface &data_transform_interface,
                      const std::shared_ptr<DataInterface> &data_interface, DataViewInterface &data_view_interface,
                      LightInterface &light_interface, PostProcessDenoiseInterface &post_process_denoise_interface,
                      PostProcessTonemapInterface &post_process_tonemap_interface,
                      RenderSettingsInterface &render_settings_interface,
                      TransferFunctionInterface &transfer_function_interface, ViewInterface &view_interface)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::ResetContext>(
        service,
        std::make_shared<detail::ResetResource>(background_light_interface, camera_interface, camera_aperture_interface,
                                                data_config_interface, data_crop_interface, data_transform_interface,
                                                data_interface, data_view_interface, light_interface,
                                                post_process_denoise_interface, post_process_tonemap_interface,
                                                render_settings_interface, transfer_function_interface, view_interface),
        &SERVICE_TYPE::RequestReset);
}

} // namespace clara::viz
