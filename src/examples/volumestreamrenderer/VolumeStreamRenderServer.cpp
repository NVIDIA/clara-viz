/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "VolumeStreamRenderServer.h"

#include <claraviz/core/Video.h>
#include <claraviz/hardware/cuda/CudaService.h>
#include <claraviz/rpc/CameraRPC.h>
#include <claraviz/rpc/VideoRPC.h>
#include <claraviz/util/Blob.h>
#include <claraviz/util/Thread.h>

#include <nvidia/claraviz/video/v1/video.grpc.pb.h>

#include <ClaraVizRenderer.h>

#include <claraviz/interface/CameraApertureInterface.h>
#include <claraviz/interface/CameraInterface.h>
#include <claraviz/interface/DataInterface.h>
#include <claraviz/interface/DataViewInterface.h>
#include <claraviz/interface/LightInterface.h>
#include <claraviz/interface/PostProcessDenoiseInterface.h>
#include <claraviz/interface/PostProcessTonemapInterface.h>
#include <claraviz/interface/RenderSettingsInterface.h>
#include <claraviz/interface/TransferFunctionInterface.h>
#include <claraviz/interface/ViewInterface.h>

#include "volume_stream_render_server.grpc.pb.h"
#include "DataSourceCT.h"
#include "DataSourceUS.h"

namespace clara::viz
{

namespace
{

/**
 * Shutdown message ID
 */
DEFINE_MESSAGEID(IDMESSAGE_SHUTDOWN);

/**
 * Next frame message ID
 */
DEFINE_MESSAGEID(IDMESSAGE_NEXT_FRAME);

/**
 * Helper class to send a message if a video frame had been encoded.
 */
class VideoStream : public IVideoStream
{
public:
    explicit VideoStream(MessageReceiver *receiver)
        : receiver_(receiver)
    {
    }
    VideoStream() = delete;

    virtual void NewStream() final {}

    bool Write(const char *data, size_t count) final
    {
        // signal the thread that the frame arrived
        receiver_->EnqueueMessage(std::make_shared<Message>(IDMESSAGE_NEXT_FRAME));
        return true;
    }

    bool Failed() const final
    {
        return false;
    }

private:
    MessageReceiver *const receiver_;
};

/**
 * Boomerang blob message
 */
class BoomerangBlobMessage : public Message
{
public:
    BoomerangBlobMessage()
        : Message(id_)
    {
    }

    std::shared_ptr<IBlob> blob_;

    /// message id
    static const MessageID id_;
};

/**
 * Boomerang blob message ID
 */
DEFINE_CLASS_MESSAGEID(BoomerangBlobMessage);

/**
 * A blob container which on destruction sends a message to a recevier with the blob it contained.
 * Used to send a blob of data to the renderer and get it back once the renderer is done with it.
 */
class BoomerangBlob : public IBlob
{
public:
    BoomerangBlob(const std::shared_ptr<MessageReceiver> &receiver, const std::shared_ptr<IBlob> &blob)
        : receiver_(receiver)
        , blob_(blob)
    {
    }
    BoomerangBlob() = delete;

    ~BoomerangBlob()
    {
        // if the receiver is still alive send a message
        std::shared_ptr<MessageReceiver> receiver = receiver_.lock();
        if (receiver)
        {
            auto message   = std::make_shared<BoomerangBlobMessage>();
            message->blob_ = blob_;
            receiver->EnqueueMessage(message);
        }
    }

    std::unique_ptr<IBlob::AccessGuard> Access() override final
    {
        return blob_->Access();
    }

    std::unique_ptr<IBlob::AccessGuard> Access(CUstream stream) override final
    {
        return blob_->Access(stream);
    }

    std::unique_ptr<IBlob::AccessGuardConst> AccessConst() override final
    {
        return blob_->AccessConst();
    }

    std::unique_ptr<IBlob::AccessGuardConst> AccessConst(CUstream stream) override final
    {
        return blob_->AccessConst(stream);
    }

    size_t GetSize() const override final
    {
        return blob_->GetSize();
    }

private:
    const std::weak_ptr<MessageReceiver> receiver_;
    const std::shared_ptr<IBlob> blob_;
};

} // anonymous namespace

using RenderServerGRPC = nvidia::claraviz::volumestream::RenderServer;
using VideoGRPC        = nvidia::claraviz::video::v1::Video;

/**
 * Server implementation data
 */
class VolumeStreamRenderServer::Impl
    : public std::enable_shared_from_this<VolumeStreamRenderServer::Impl>
    , public MessageReceiver
{
public:
    /**
     * Construct
     *
     * @param input_dir [in] source data input directory, if empty generate synthetic data
     * @param scenario [in] scenario to execute
     * @param benchmark_duration [in] benchmark duration in seconds, if 0 run in interactive mode
     * @param stream_from_cpu [in] if set, stream from CPU memory else from GPU memory
     * @param server_rpc [in] gRPC server
     * @param cuda_device_ordinals [in] Cuda device to render on
     */
    Impl(const std::string &input_dir, const std::string &scenario, const std::chrono::seconds &benchmark_duration,
         bool stream_from_cpu, const std::shared_ptr<ServerRPC> &server_rpc,
         const std::vector<uint32_t> &cuda_device_ordinals);

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
    /**
     * Thread function.
     */
    void ThreadFunction(std::function<void()> ready);

    /// CUDA devices to use
    const std::vector<uint32_t> cuda_device_ordinals_;

    /// input directory
    const std::string input_dir_;

    /// scenario
    const std::string scenario_;

    /// benchmark duration
    const std::chrono::seconds benchmark_duration_;

    /// stream from CPU memory
    const bool stream_from_cpu_;

    /// stream thread
    std::unique_ptr<Thread> thread_;

    /// Server RPC
    std::shared_ptr<ServerRPC> server_rpc_;

    /// image service
    std::shared_ptr<Video> video_;
    std::shared_ptr<MessageReceiver> image_;

    /// renderer
    std::unique_ptr<Renderer> renderer_;

    /// interface data
    CameraInterface camera_interface_;
    CameraApertureInterface camera_aperture_interface_;
    std::shared_ptr<DataInterface> data_interface_;
    DataConfigInterface data_config_interface_;
    DataCropInterface data_crop_interface_;
    DataViewInterface data_view_interface_;
    BackgroundLightInterface background_light_interface_;
    LightInterface light_interface_;
    PostProcessDenoiseInterface post_process_denoise_interface_;
    PostProcessTonemapInterface post_process_tonemap_interface_;
    RenderSettingsInterface render_settings_interface_;
    TransferFunctionInterface transfer_function_interface_;
    std::shared_ptr<VideoInterface> video_interface_;
    ViewInterface view_interface_;
};

VolumeStreamRenderServer::VolumeStreamRenderServer(const std::string &input_dir, const std::string &scenario,
                                                   const std::chrono::seconds &benchmark_duration, bool stream_from_cpu,
                                                   uint32_t port, const std::vector<uint32_t> &cuda_device_ordinals)
    : RenderServerBase("0.0.0.0:" + std::to_string(port))
    , impl_(new Impl(input_dir, scenario, benchmark_duration, stream_from_cpu, GetServerRPC(), cuda_device_ordinals))
{
}

VolumeStreamRenderServer::~VolumeStreamRenderServer()
{
    Shutdown();
}

void VolumeStreamRenderServer::Run()
{
    // call the base class
    RenderServerBase::Run();

    impl_->Run();
}

bool VolumeStreamRenderServer::Shutdown()
{
    if (impl_)
    {
        impl_->Shutdown();
    }

    // shutdown the base class
    return RenderServerBase::Shutdown();
}

VolumeStreamRenderServer::Impl::Impl(const std::string &input_dir, const std::string &scenario,
                                     const std::chrono::seconds &benchmark_duration, bool stream_from_cpu,
                                     const std::shared_ptr<ServerRPC> &server_rpc,
                                     const std::vector<uint32_t> &cuda_device_ordinals)
    : cuda_device_ordinals_(cuda_device_ordinals)
    , input_dir_(input_dir)
    , scenario_(scenario)
    , benchmark_duration_(benchmark_duration)
    , stream_from_cpu_(stream_from_cpu)
    , server_rpc_(server_rpc)
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
    video_ = Video::Create(cuda_device_ordinals[0]);

    // create the renderer, will receive message from the interfaces and send messages to the video service.
    renderer_ =
        std::make_unique<Renderer>(std::static_pointer_cast<MessageReceiver>(video_), image_, cuda_device_ordinals);

    // register message receivers which will be triggered on changes
    const std::shared_ptr<MessageReceiver> &receiver = renderer_->GetReceiver();
    camera_interface_.RegisterReceiver(receiver);
    camera_aperture_interface_.RegisterReceiver(receiver);
    data_interface_ = std::make_shared<DataInterface>();
    data_interface_->RegisterReceiver(receiver);
    data_config_interface_.RegisterReceiver(receiver);
    // the data interface needs to get updates from the data config interface to do proper parameter validation
    data_config_interface_.RegisterReceiver(data_interface_);
    data_crop_interface_.RegisterReceiver(receiver);
    data_view_interface_.RegisterReceiver(receiver);
    background_light_interface_.RegisterReceiver(receiver);
    light_interface_.RegisterReceiver(receiver);
    post_process_denoise_interface_.RegisterReceiver(receiver);
    post_process_tonemap_interface_.RegisterReceiver(receiver);
    render_settings_interface_.RegisterReceiver(receiver);
    transfer_function_interface_.RegisterReceiver(receiver);
    view_interface_.RegisterReceiver(receiver);

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

    if (!benchmark_duration_.count())
    {
        RegisterRPC<VideoGRPC::AsyncService>(server_rpc_, video_service, video_interface_);
    }
}

VolumeStreamRenderServer::Impl::~Impl()
{
    // unregister the receivers
    const std::shared_ptr<MessageReceiver> &receiver = renderer_->GetReceiver();
    background_light_interface_.UnregisterReceiver(receiver);
    camera_interface_.UnregisterReceiver(receiver);
    camera_aperture_interface_.UnregisterReceiver(receiver);
    data_interface_->UnregisterReceiver(receiver);
    data_config_interface_.UnregisterReceiver(receiver);
    data_config_interface_.UnregisterReceiver(data_interface_);
    data_crop_interface_.UnregisterReceiver(receiver);
    data_view_interface_.UnregisterReceiver(receiver);
    light_interface_.UnregisterReceiver(receiver);
    post_process_denoise_interface_.UnregisterReceiver(receiver);
    post_process_tonemap_interface_.UnregisterReceiver(receiver);
    render_settings_interface_.UnregisterReceiver(receiver);
    transfer_function_interface_.UnregisterReceiver(receiver);
    view_interface_.UnregisterReceiver(receiver);

    video_interface_->UnregisterReceiver(video_);

    video_->UnregisterReceiver(receiver);
}

void VolumeStreamRenderServer::Impl::Run()
{
    // start the renderer
    renderer_->Run();
    video_->Run();

    // run the stream thread
    thread_.reset(new Thread("Stream thread", [this](std::function<void()> ready) { ThreadFunction(ready); }));
}

void VolumeStreamRenderServer::Impl::Shutdown()
{
    if (thread_)
    {
        // shutdown the thread
        EnqueueMessage(std::make_shared<Message>(IDMESSAGE_SHUTDOWN));

        // destroy the thread
        thread_.reset();
    }

    // shutdown the video server first, it will return the resources which are currently encoded
    video_->Shutdown();

    // then shutdown the renderer
    renderer_->Shutdown();
}

void VolumeStreamRenderServer::Impl::ThreadFunction(std::function<void()> ready)
{
    CudaPrimaryContext cuda_context(cuda_device_ordinals_[0]);

    // thread is ready now
    ready();

    const std::string array_id = "density";

    std::unique_ptr<DataSource> data_source;
    if (scenario_ == "CT")
    {
        data_source.reset(new DataSourceCT(stream_from_cpu_, input_dir_));
    }
    else if (scenario_ == "US")
    {
        data_source.reset(new DataSourceUS(input_dir_));
    }
    else
    {
        throw RuntimeError() << "Unsupported scenario " << scenario_;
    }

    // configure render settings
    {
        RenderSettingsInterface::AccessGuard access(render_settings_interface_);

        // switch to linear interpolation for speed, the quality difference is minimal compared to BSPLINE
        access->interpolation_mode = InterpolationMode::LINEAR;
        // in benchmark mode set iterations to minimum, we want to measure transfer time, not rendering
        access->max_iterations.Set(benchmark_duration_.count() ? 1 : 500);
    }

    // configure the data
    {
        DataConfigInterface::AccessGuard access(data_config_interface_);

        // enable the streaming use case
        access->streaming = true;

        access->arrays.emplace_back();
        const auto array = access->arrays.begin();
        array->id        = array_id;
        array->dimension_order.Set("DXYZ");
        array->element_type.Set(data_source->volume_type_);
        array->permute_axis.Set(std::vector<uint32_t>({0, 1, 3, 2}));
        array->flip_axes.Set(std::vector<bool>({false, false, false, true}));

        array->levels.emplace_back();
        const auto level = array->levels.begin();
        level->size.Set({1, data_source->volume_size_(0), data_source->volume_size_(1), data_source->volume_size_(2)});
        level->element_size.Set({1.0f, data_source->volume_element_spacing_(0), data_source->volume_element_spacing_(1),
                                 data_source->volume_element_spacing_(2)});
        level->element_range.Set({data_source->volume_element_range_});
    }

    data_source->Light(light_interface_);
    data_source->BackgroundLight(background_light_interface_);

    {
        ViewInterface::AccessGuard access(view_interface_);

        ViewInterface::DataIn::View *view = access->GetView();

        view->camera_name = "Cinematic";
        view->mode        = ViewMode::CINEMATIC;
    }

    data_source->Camera(camera_interface_);
    data_source->TransferFunction(transfer_function_interface_);

    {
        PostProcessDenoiseInterface::AccessGuard access(post_process_denoise_interface_);

        // disable denoising in benchmark mode
        if (benchmark_duration_.count())
        {
            access->method = DenoiseMethod::OFF;
        }
        else
        {
            access->method = DenoiseMethod::AI;
        }
        access->enable_iteration_limit = true;
        access->iteration_limit.Set(1000);
    }

    if (benchmark_duration_.count())
    {
        // configure video
        VideoInterface::AccessGuard access(*video_interface_);

        VideoInterface::DataIn::Video *video = access->GetVideo();

        video->width.Set(1024);
        video->height.Set(768);
        video->bit_rate.Set(3 * 1024 * 1024);
        // a frame rate of zero indicates "as fast as possible"
        video->frame_rate.Set(0);
        video->stream.reset(new VideoStream(this));
        video->state = VideoInterfaceState::PLAY;
    }

    const std::chrono::duration<float, std::milli> frame_duration(
        benchmark_duration_.count() > 0 ? 0.f : 1000.f / data_source->FrameRate());

    uint32_t volumes         = 0;
    uint32_t pending_volumes = 0;
    bool warm_up             = true;
    // debug do one single upload only
    bool single_upload = false;

    std::chrono::steady_clock::time_point start                    = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> target_time = start;
    while (true)
    {
        try
        {
            // if the previous volume is received back and the target time elapsed, send a new volume
            if (!single_upload && (pending_volumes < 2) &&
                ((target_time - std::chrono::steady_clock::now()).count() <= 0.f))
            {
                // get the next blob from the source and send
                const std::shared_ptr<IBlob> blob = data_source->NextBlob(benchmark_duration_.count());

                if (!blob)
                {
                    Log(LogLevel::Warning) << "Something went wrong, no blobs available!";
                }

                DataInterface::AccessGuard access(*data_interface_.get());
                access->array_id.Set(array_id);
                access->level.Set(0);
                access->offset.Set({0, 0, 0, 0});
                access->size.Set(
                    {1, data_source->volume_size_(0), data_source->volume_size_(1), data_source->volume_size_(2)});
                access->blob = std::make_shared<BoomerangBlob>(this->shared_from_this(), blob);
                access->sharedmemory_allocation_id.Set("");

                target_time = std::chrono::steady_clock::now() +
                              std::chrono::duration_cast<std::chrono::milliseconds>(frame_duration);
                ++pending_volumes;
                single_upload = false;
            }

            if (benchmark_duration_.count())
            {
                if (warm_up && (volumes > 0))
                {
                    start   = std::chrono::steady_clock::now();
                    warm_up = false;
                    volumes = 0;
                }

                const std::chrono::duration<float, std::milli> elapsed_time = std::chrono::steady_clock::now() - start;

                if (elapsed_time > benchmark_duration_)
                {
                    const size_t volume_size_in_bytes = data_source->volume_bytes_per_element_ *
                                                        data_source->volume_size_(0) * data_source->volume_size_(1) *
                                                        data_source->volume_size_(2);
                    Log(LogLevel::Warning)
                        << (volumes / elapsed_time.count()) * 1000.f << " volumes/s "
                        << (volumes / elapsed_time.count()) * 1000.f * volume_size_in_bytes / 1024.f / 1024.f
                        << " MByte/s" << std::flush;

                    // that's it
                    exit(0);
                }

                // wait for new messages
                Wait();
            }
            else
            {
                // wait until the next message arrives or the next volume needs to be sent
                std::chrono::duration<float, std::milli> wait_time = target_time - std::chrono::steady_clock::now();
                if (wait_time.count() > 0.f)
                {
                    WaitFor(wait_time);
                }
                else
                {
                    // If there are two pending volumes, wait for one to be returned. Else continue
                    // and upload a new volume.
                    if (pending_volumes == 2)
                    {
                        Wait();
                    }
                }
            }

            // if set then shutdown
            bool shutdown = false;

            std::shared_ptr<const Message> message;
            while ((message = DequeueMessage()))
            {
                Log(LogLevel::Debug) << "Volume stream received " << message->GetID().GetName();

                if (message->GetID() == IDMESSAGE_NEXT_FRAME)
                {
                    ++volumes;
                }
                else if (message->GetID() == BoomerangBlobMessage::id_)
                {
                    const auto &boomerang_blob_message = std::static_pointer_cast<const BoomerangBlobMessage>(message);
                    data_source->ReturnBlob(boomerang_blob_message->blob_);
                    --pending_volumes;
                }
                else if (message->GetID() == IDMESSAGE_SHUTDOWN)
                {
                    shutdown = true;
                }
                else
                {
                    throw InvalidState() << "Unhandled message Id " << message->GetID().GetName();
                }
            }

            if (shutdown)
            {
                break;
            }
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Stream thread threw exception " << e.what();
        }
        catch (...)
        {
            Log(LogLevel::Error) << "Stream thread unknown exception";
        }
    }
}

} // namespace clara::viz
