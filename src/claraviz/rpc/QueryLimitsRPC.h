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

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class QueryLimitsResource : public nvrpc::Resources
{
public:
    QueryLimitsResource(int cuda_device_ordinal);
    QueryLimitsResource() = delete;

    uint32_t max_image_width_;
    uint32_t max_image_height_;

    uint32_t max_volume_width_;
    uint32_t max_volume_height_;
    uint32_t max_volume_depth_;

    uint32_t max_lights_;
};

/**
 * RPC call context
 */
class QueryLimitsContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::QueryLimitsRequest,
                                 nvidia::claraviz::cinematic::v1::QueryLimitsResponse, QueryLimitsResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::QueryLimitsRequest &request,
                    nvidia::claraviz::cinematic::v1::QueryLimitsResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the QueryLimits class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param cuda_device_ordinal [in] Cuda device ordinal used by the renderer
 */
template<typename SERVICE_TYPE>
void RegisterQueryLimitsRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                            int cuda_device_ordinal)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::QueryLimitsContext>(
        service, std::make_shared<detail::QueryLimitsResource>(cuda_device_ordinal), &SERVICE_TYPE::RequestQueryLimits);
}

} // namespace clara::viz
