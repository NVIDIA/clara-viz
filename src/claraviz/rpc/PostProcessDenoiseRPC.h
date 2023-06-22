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

#include "claraviz/interface/PostProcessDenoiseInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class PostProcessDenoiseResource : public nvrpc::Resources
{
public:
    PostProcessDenoiseResource(PostProcessDenoiseInterface &denoise)
        : denoise_(denoise)
    {
    }
    PostProcessDenoiseResource() = delete;

    PostProcessDenoiseInterface &denoise_;
};

/**
 * RPC call context
 */
class PostProcessDenoiseContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::PostProcessDenoiseRequest,
                                 nvidia::claraviz::cinematic::v1::PostProcessDenoiseResponse, PostProcessDenoiseResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::PostProcessDenoiseRequest &request,
                    nvidia::claraviz::cinematic::v1::PostProcessDenoiseResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the PostProcessDenoise class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param denoise [in] PostProcessDenoise interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                 PostProcessDenoiseInterface &denoise)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::PostProcessDenoiseContext>(
        service, std::make_shared<detail::PostProcessDenoiseResource>(denoise),
        &SERVICE_TYPE::RequestPostProcessDenoise);
}

} // namespace clara::viz
