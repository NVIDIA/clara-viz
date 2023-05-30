/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/DataViewInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class DataViewResource : public nvrpc::Resources
{
public:
    DataViewResource(DataViewInterface &data_view)
        : data_view_(data_view)
    {
    }
    DataViewResource() = delete;

    DataViewInterface &data_view_;
};

/**
 * RPC call context
 */
class DataViewContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::DataViewRequest,
                                 nvidia::claraviz::cinematic::v1::DataViewResponse, DataViewResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::DataViewRequest &request,
                    nvidia::claraviz::cinematic::v1::DataViewResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the DataView class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param data_view [in] DataView interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service, DataViewInterface &data_view)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::DataViewContext>(
        service, std::make_shared<detail::DataViewResource>(data_view), &SERVICE_TYPE::RequestDataView);
}

} // namespace clara::viz
