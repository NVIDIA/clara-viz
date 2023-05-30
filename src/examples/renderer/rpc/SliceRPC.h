/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "interface/SliceInterface.h"

#include "example_render_server.grpc.pb.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class SliceResource : public nvrpc::Resources
{
public:
    SliceResource(SliceInterface &slice)
        : slice_(slice)
    {
    }
    SliceResource() = delete;

    SliceInterface &slice_;
};

/**
 * RPC call context
 */
class SliceContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::example::SliceRequest, nvidia::claraviz::example::SliceResponse, SliceResource>
{
    void ExecuteRPC(nvidia::claraviz::example::SliceRequest &request, nvidia::claraviz::example::SliceResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the Slice class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param slice [in] Slice interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service, SliceInterface &slice)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::SliceContext>(service, std::make_shared<detail::SliceResource>(slice),
                                                                &SERVICE_TYPE::RequestSlice);
}

} // namespace clara::viz
