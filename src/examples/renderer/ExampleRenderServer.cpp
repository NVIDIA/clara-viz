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

#include "ExampleRenderServer.h"

#include <claraviz/core/Video.h>
#include <claraviz/interface/CameraInterface.h>
#include <claraviz/rpc/CameraRPC.h>
#include <claraviz/rpc/VideoRPC.h>

#include <nvidia/claraviz/video/v1/video.grpc.pb.h>

#include "cuda/Renderer.h"
#include "interface/SliceInterface.h"
#include "rpc/SliceRPC.h"

#include "example_render_server.grpc.pb.h"

namespace clara::viz
{

using RenderServerGRPC = nvidia::claraviz::example::RenderServer;
using VideoGRPC        = nvidia::claraviz::video::v1::Video;

/**
 * Server implementation data
 */
class ExampleRenderServer::Impl : public MessageReceiver
{
public:
    /**
     * Construct
     *
     * @param server_rpc [in] gRPC server
     * @param cuda_device_ordinal [in] Cuda device to render on
     * @param mhd_file_name [in] Name of MHD file to load
     */
    Impl(const std::shared_ptr<ServerRPC> &server_rpc, uint32_t cuda_device_ordinal, const std::string &mhd_file_name);

    /**
     * Destruct
     */
    ~Impl();

    /**
     * Run the server.
     */
    void Run();

    /**
     * Shutdown the server, wait for all running tasks to finish.
     */
    void Shutdown();

private:
    /// Server RPC
    std::shared_ptr<ServerRPC> server_rpc_;

    /// video service
    std::shared_ptr<Video> video_;

    /// renderer
    std::unique_ptr<Renderer> renderer_;

    /// interface data
    CameraInterface camera_interface_;
    SliceInterface slice_interface_;
    std::shared_ptr<VideoInterface> video_interface_;
};

ExampleRenderServer::ExampleRenderServer(uint32_t port, uint32_t cuda_device_ordinal, const std::string &mhd_file_name)
    : RenderServerBase("0.0.0.0:" + std::to_string(port))
    , impl_(new Impl(GetServerRPC(), cuda_device_ordinal, mhd_file_name))
{
}

ExampleRenderServer::~ExampleRenderServer()
{
    Shutdown();
}

void ExampleRenderServer::Run()
{
    // call the base class
    RenderServerBase::Run();

    impl_->Run();
}

bool ExampleRenderServer::Shutdown()
{
    if (impl_)
    {
        impl_->Shutdown();
    }

    // shutdown the base class
    return RenderServerBase::Shutdown();
}

ExampleRenderServer::Impl::Impl(const std::shared_ptr<ServerRPC> &server_rpc, uint32_t cuda_device_ordinal,
                                const std::string &mhd_file_name)
    : server_rpc_(server_rpc)
{
    if (!server_rpc)
    {
        throw InvalidArgument("server_rpc") << "is nullptr";
    }

    // Overview
    // The RPC server manages client connections and requests. Once a client request is received it's handled
    // by registered RPC. The RPC execution function forwards the RPC data to the interface. The interface
    // does parameter validation (e.g. range checks) and emits a message.
    // The renderer is receiving the messages from the interface and takes the appropriate actions.

    // register the RPC service, this service handles all RPC's from the nvidia.claraviz.example.RenderServer proto service.
    nvrpc::IService *render_server_service =
        server_rpc_->RegisterService(std::make_unique<nvrpc::AsyncService<RenderServerGRPC::AsyncService>>());
    // register the Video RPC service, this service handles all RPC's from the nvidia::claraviz::video::v1::Video proto service.
    nvrpc::IService *video_service =
        server_rpc_->RegisterService(std::make_unique<nvrpc::AsyncService<VideoGRPC::AsyncService>>());

    // create the video service
    video_ = Video::Create(cuda_device_ordinal);

    // create the renderer, will receive message from the interfaces and send messages to the video service.
    renderer_ = std::make_unique<Renderer>(std::static_pointer_cast<MessageReceiver>(video_), cuda_device_ordinal,
                                           mhd_file_name);

    // register message receivers which will be triggered on changes
    const std::shared_ptr<MessageReceiver> &receiver = renderer_->GetReceiver();
    camera_interface_.RegisterReceiver(receiver);
    slice_interface_.RegisterReceiver(receiver);

    // create the video interface, it needs the encoder limits, query them from the encoder
    {
        const std::unique_ptr<IVideoEncoder> &encoder = video_->GetEncoder();
        const uint32_t min_width                      = encoder->Query(IVideoEncoder::Capability::MIN_WIDTH);
        const uint32_t min_height                     = encoder->Query(IVideoEncoder::Capability::MIN_HEIGHT);
        const uint32_t max_width                      = encoder->Query(IVideoEncoder::Capability::MAX_WIDTH);
        const uint32_t max_height                     = encoder->Query(IVideoEncoder::Capability::MAX_HEIGHT);
        video_interface_.reset(new VideoInterface(min_width, min_height, max_width, max_height));
    }
    // the video service will receive message from the video interface
    video_interface_->RegisterReceiver(video_);

    // renderer will also receive messages from the video service
    video_->RegisterReceiver(receiver);

    // register the RPC's
    RegisterRPC<RenderServerGRPC::AsyncService>(server_rpc_, render_server_service, camera_interface_);
    RegisterRPC<RenderServerGRPC::AsyncService>(server_rpc_, render_server_service, slice_interface_);

    RegisterRPC<VideoGRPC::AsyncService>(server_rpc_, video_service, video_interface_);
}

ExampleRenderServer::Impl::~Impl()
{
    // unregister the receivers
    const std::shared_ptr<MessageReceiver> &receiver = renderer_->GetReceiver();
    camera_interface_.UnregisterReceiver(receiver);
    slice_interface_.UnregisterReceiver(receiver);

    video_interface_->UnregisterReceiver(video_);

    video_->UnregisterReceiver(receiver);
}

void ExampleRenderServer::Impl::Run()
{
    renderer_->Run();
    video_->Run();
}

void ExampleRenderServer::Impl::Shutdown()
{
    // shutdown the video server first, it will return the resources which are currently encoded
    video_->Shutdown();

    // then shutdown the renderer
    renderer_->Shutdown();
}

} // namespace clara::viz
