/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/LightInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class LightResource : public nvrpc::Resources
{
public:
    LightResource(LightInterface &light)
        : light_(light)
    {
    }
    LightResource() = delete;

    LightInterface &light_;
};

/**
 * RPC call context
 */
class LightContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::LightRequest, nvidia::claraviz::cinematic::v1::LightResponse,
                                 LightResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::LightRequest &request,
                    nvidia::claraviz::cinematic::v1::LightResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the Light class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param light [in] Light interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service, LightInterface &light)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::LightContext>(service, std::make_shared<detail::LightResource>(light),
                                                                &SERVICE_TYPE::RequestLight);
}

} // namespace clara::viz
