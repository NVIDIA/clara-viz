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

#include <memory>

#include <nvrpc/Context.h>
#include <nvrpc/Resources.h>
#include <claraviz/rpc/ServerRPC.h>

#include <nvidia/claraviz/cinematic/v1/render_server.grpc.pb.h>

#include "claraviz/interface/DataInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class DataTransformResource : public nvrpc::Resources
{
public:
    DataTransformResource(DataTransformInterface &data_transform)
        : data_transform_(data_transform)
    {
    }
    DataTransformResource() = delete;

    DataTransformInterface &data_transform_;
};

/**
 * RPC call context
 */
class DataTransformContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::DataTransformRequest,
                                 nvidia::claraviz::cinematic::v1::DataTransformResponse, DataTransformResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::DataTransformRequest &request,
                    nvidia::claraviz::cinematic::v1::DataTransformResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the DataTransform class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param data_transform [in] DataTransform interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                 DataTransformInterface &data_transform)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::DataTransformContext>(
        service, std::make_shared<detail::DataTransformResource>(data_transform), &SERVICE_TYPE::RequestDataTransform);
}

} // namespace clara::viz
