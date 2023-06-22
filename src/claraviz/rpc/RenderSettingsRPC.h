/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/RenderSettingsInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class RenderSettingsResource : public nvrpc::Resources
{
public:
    RenderSettingsResource(RenderSettingsInterface &render_settings)
        : render_settings_(render_settings)
    {
    }
    RenderSettingsResource() = delete;

    RenderSettingsInterface &render_settings_;
};

/**
 * RPC call context
 */
class RenderSettingsContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::RenderSettingsRequest,
                                 nvidia::claraviz::cinematic::v1::RenderSettingsResponse, RenderSettingsResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::RenderSettingsRequest &request,
                    nvidia::claraviz::cinematic::v1::RenderSettingsResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the RenderSettings class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param render_settings [in] RenderSettings interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                 RenderSettingsInterface &render_settings)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::RenderSettingsContext>(
        service, std::make_shared<detail::RenderSettingsResource>(render_settings),
        &SERVICE_TYPE::RequestRenderSettings);
}

} // namespace clara::viz
