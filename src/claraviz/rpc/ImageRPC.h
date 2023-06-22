/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <nvidia/claraviz/image/v1/image.grpc.pb.h>

#include "claraviz/interface/ImageInterface.h"
#include "claraviz/rpc/ServerRPC.h"

namespace clara::viz
{

namespace detail
{

/**
 * RPC resource
 */
class ImageGenerateResource : public nvrpc::Resources
{
public:
    ImageGenerateResource(ImageInterface &image_interface,
                          const std::shared_ptr<ImageInterfaceOutput> &image_interface_output)
        : image_interface_(image_interface)
        , image_interface_output_(image_interface_output)
    {
    }
    ImageGenerateResource() = delete;

    ImageInterface &image_interface_;
    std::shared_ptr<ImageInterfaceOutput> image_interface_output_;
};

/**
 * RPC resource
 */
class ImageQueryLimitsResource : public nvrpc::Resources
{
public:
    ImageQueryLimitsResource() {}
};

/**
 * RPC call context
 */
class ImageGenerateContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::image::v1::GenerateRequest, nvidia::claraviz::image::v1::GenerateResponse,
                                 ImageGenerateResource>
{
    void ExecuteRPC(nvidia::claraviz::image::v1::GenerateRequest &request,
                    nvidia::claraviz::image::v1::GenerateResponse &response) final;
};

/**
 * RPC call context
 */
class ImageQueryLimitsContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::image::v1::QueryLimitsRequest, nvidia::claraviz::image::v1::QueryLimitsResponse,
                                 ImageQueryLimitsResource>
{
    void ExecuteRPC(nvidia::claraviz::image::v1::QueryLimitsRequest &request,
                    nvidia::claraviz::image::v1::QueryLimitsResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the Image class
 *
 * @tparam SERVICE_TYPE gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param image_interface [in] Image interface class object used by the RPC
 * @param image_interface_output [in] Image interface output class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                 ImageInterface &image_interface, const std::shared_ptr<ImageInterfaceOutput> &image_interface_output)
{
    // since this RPC is returning the image and could take some time to execute we need to run it on its
    // own executor, else it would block the default executor serving other requests
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::ImageGenerateContext>(
        service, std::make_shared<detail::ImageGenerateResource>(image_interface, image_interface_output),
        &SERVICE_TYPE::RequestGenerate, rpc_server->CreateExecutor());

    rpc_server->RegisterRPC<SERVICE_TYPE, detail::ImageQueryLimitsContext>(
        service, std::make_shared<detail::ImageQueryLimitsResource>(), &SERVICE_TYPE::RequestQueryLimits);
}

} // namespace clara::viz
