/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <nvidia/claraviz/core/camera.grpc.pb.h>

#include "claraviz/interface/CameraInterface.h"
#include "claraviz/rpc/ServerRPC.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class CameraResource : public nvrpc::Resources
{
public:
    CameraResource(CameraInterface &camera)
        : camera_(camera)
    {
    }
    CameraResource() = delete;

    CameraInterface &camera_;
};

/**
 * RPC call context
 */
class CameraContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::core::CameraRequest, nvidia::claraviz::core::CameraResponse, CameraResource>
{
    void ExecuteRPC(nvidia::claraviz::core::CameraRequest &request, nvidia::claraviz::core::CameraResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the Camera class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param camera [in] Camera interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service, CameraInterface &camera)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::CameraContext>(
        service, std::make_shared<detail::CameraResource>(camera), &SERVICE_TYPE::RequestCamera);
}

} // namespace clara::viz
