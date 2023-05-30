/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <nvidia/claraviz/video/v1/video.grpc.pb.h>

#include "claraviz/interface/VideoInterface.h"
#include "claraviz/rpc/ServerRPC.h"

namespace clara::viz
{

/// The maximum amount of similtaneous but the grpc interface supported video streams
constexpr int MAX_VIDEO_STREAMS = 4;

namespace detail
{

/**
 * RPC resource
 */
class VideoResource : public nvrpc::Resources
{
public:
    /**
     * Construct
     *
     * @param video_interface [in] video interface
     */
    VideoResource(const std::shared_ptr<VideoInterface> &video_interface)
        : video_interface_(*video_interface.get())
    {
    }
    VideoResource() = delete;

    VideoInterface &video_interface_; ///< video interface
};

/**
 * RPC call context
 */
class VideoConfigContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::video::v1::ConfigRequest, nvidia::claraviz::video::v1::ConfigResponse,
                                 VideoResource>
{
    void ExecuteRPC(nvidia::claraviz::video::v1::ConfigRequest &request,
                    nvidia::claraviz::video::v1::ConfigResponse &response) final;
};

/**
 * RPC call context
 */
class VideoStreamContext final
    : public nvrpc::ContextServerStreaming<nvidia::claraviz::video::v1::StreamRequest, nvidia::claraviz::video::v1::StreamResponse,
                                           VideoResource>
{
public:
    ~VideoStreamContext();

    /**
     * Write response to the stream.
     *
     * @param response [in] response to write
     * @param id [in] stream id
     *
     * @returns false if the write failed
     */
    bool Write(const ResponseType &response, const std::string &id);

    /**
     * Stop the stream
     */
    void Stop();

    using CallbackHandle =
        nvrpc::ContextServerStreaming<nvidia::claraviz::video::v1::StreamRequest, nvidia::claraviz::video::v1::StreamResponse,
                                      VideoResource>::CallbackHandle;

    /**
     * Register a callback which is called when the gRPC stream is closing,
     * either regularily or on error.
     *
     * @param callback [in] callback function
     */
    CallbackHandle RegisterFinishCallback(const std::function<void()> &callback);

private:
    void OnRequestReceived(const nvidia::claraviz::video::v1::StreamRequest &request) final;
};

/**
 * RPC call context
 */
class VideoControlContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::video::v1::ControlRequest, nvidia::claraviz::video::v1::ControlResponse,
                                 VideoResource>
{
    void ExecuteRPC(nvidia::claraviz::video::v1::ControlRequest &request,
                    nvidia::claraviz::video::v1::ControlResponse &response) final;
};

/**
 * RPC call context
 */
class VideoQueryLimitsContext final
    : public nvrpc::ContextUnary<nvidia::claraviz::video::v1::QueryLimitsRequest, nvidia::claraviz::video::v1::QueryLimitsResponse,
                                 VideoResource>
{
    void ExecuteRPC(nvidia::claraviz::video::v1::QueryLimitsRequest &request,
                    nvidia::claraviz::video::v1::QueryLimitsResponse &response) final;
};

} // namespace detail

/**
 * Register the RPC for the Video class
 *
 * @tparam SERVICE_TYPE gRPC service type (class type from 'service SomeService' defined in the proto file)
 *
 * @param rpc_server [in] server to register the RPC with
 * @param service [in] service to register the RPC with
 * @param video_interface [in] Video interface class object used by the RPC
 */
template<typename SERVICE_TYPE>
void RegisterRPC(const std::shared_ptr<ServerRPC> &rpc_server, nvrpc::IService *service,
                 const std::shared_ptr<VideoInterface> &video_interface)
{
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::VideoConfigContext>(
        service, std::make_shared<detail::VideoResource>(video_interface), &SERVICE_TYPE::RequestConfig);
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::VideoControlContext>(
        service, std::make_shared<detail::VideoResource>(video_interface), &SERVICE_TYPE::RequestControl);
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::VideoQueryLimitsContext>(
        service, std::make_shared<detail::VideoResource>(video_interface), &SERVICE_TYPE::RequestQueryLimits);

    // since this is a streaming RPC we need to run it on its own executor, else it would block the default executor
    rpc_server->RegisterRPC<SERVICE_TYPE, detail::VideoStreamContext>(
        service, std::make_shared<detail::VideoResource>(video_interface), &SERVICE_TYPE::RequestStream,
        rpc_server->CreateExecutor(MAX_VIDEO_STREAMS));
}

} // namespace clara::viz
