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

#include "claraviz/interface/CameraApertureInterface.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class CameraApertureResource : public nvrpc::Resources
{
public:
    CameraApertureResource(CameraApertureInterface &aperture)
        : aperture_(aperture)
    {
    }
    CameraApertureResource() = delete;

    CameraApertureInterface &aperture_;
};

/**
 * RPC call context
 */
class CameraApertureContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::cinematic::v1::CameraApertureRequest,
                                 nvidia::claraviz::cinematic::v1::CameraApertureResponse, CameraApertureResource>
{
    void ExecuteRPC(nvidia::claraviz::cinematic::v1::CameraApertureRequest &request,
                    nvidia::claraviz::cinematic::v1::CameraApertureResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the CameraAperture class
 *
 * @tparam SERVICE_TYPE        gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param aperture [in] CameraAperture interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                 CameraApertureInterface &aperture)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::CameraApertureContext>(
        service, std::make_shared<detail::CameraApertureResource>(aperture), &SERVICE_TYPE::RequestCameraAperture);
}

} // namespace clara::viz
