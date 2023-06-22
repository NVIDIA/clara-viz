/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "Renderer.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <cfloat>
#include <list>
#include <map>

#include <claraviz/core/Video.h>
#include <claraviz/hardware/cuda/CudaService.h>
#include <claraviz/interface/CameraInterface.h>
#include <claraviz/interface/VideoInterface.h>
#include <claraviz/util/CudaMemoryBlob.h>
#include <claraviz/util/MatrixT.h>
#include <claraviz/util/Message.h>
#include <claraviz/util/Thread.h>
#include <claraviz/util/VectorT.h>
#include <claraviz/util/MHDLoader.h>

#include "cuda/Session.h"
#include "interface/SliceInterface.h"

namespace clara::viz
{

namespace
{

/// global session data for all Cuda kernels
__constant__ Session g_session;

} // anonymous namespace

} // namespace clara::viz

#include "cuda/MinMax.cuh"
#include "cuda/RenderVolume.cuh"
#include "cuda/RenderSlice.cuh"

namespace clara::viz
{

namespace
{

/**
 * Shutdown message ID
 */
DEFINE_MESSAGEID(IDMESSAGE_SHUTDOWN);

/// the maximum amount of outstanding display buffers, if more buffers are
/// in the encoder queue the renderer will wait for buffers to return.
constexpr uint32_t MAX_OUTSTANDING_DISPLAY_BUFFERS = 3u;

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
 * Used to send a blob of data to the video encoder and get it back once the encoder is done with it.
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

/**
 * Renderer implementation data.
 */
class Renderer::Impl
    : public std::enable_shared_from_this<Renderer::Impl>
    , public MessageReceiver
{
public:
    /**
     * Construct.
     *
     * @param video_msg_receiver [in] video message receiver
     * @param cuda_device_ordinal [in] Cuda device to render on
     * @param mhd_file_name [in] Name of MHD file to load
     */
    explicit Impl(const std::shared_ptr<MessageReceiver> &video_msg_receiver, uint32_t cuda_device_ordinal,
                  const std::string &mhd_file_name);
    Impl() = delete;

    /**
     * Run the renderer.
     */
    void Run();

    /**
     * Shutdown the renderer, wait for all running tasks to finish.
     */
    void Shutdown();

private:
    /**
     * Server thread function.
     */
    void ThreadFunction(std::function<void()> ready);

    /**
     * Render!
     *
     * @param executor [in] the executor responsible for calculating the iterations
     */
    void Render(const CudaAdaptiveExecutor *executor);

    /// The cuda device to render on
    const uint32_t cuda_device_ordinal_;

    /**
     * Cuda context.
     * Note that the context has to be the first member interacting with Cuda.
     * The context will be destroyed last and all other Cuda objects should be already
     * be destroyed at this point.
     */
    std::unique_ptr<CudaPrimaryContext> cuda_context_;

    /// volume cuda array
    std::shared_ptr<CudaArray> volume_array_;

    /// volume cuda texture
    std::unique_ptr<CudaTexture> volume_texture_;

    /// render thread
    std::unique_ptr<Thread> thread_;

    /// camera setup
    CameraInterface::DataOut::Camera camera_;
    /// slice setup
    SliceInterface::DataOut slice_;

    /// video output state
    std::shared_ptr<const VideoMessage> video_state_;
    /// video message receiver
    std::shared_ptr<MessageReceiver> video_msg_receiver_;

    /// data required by the Cuda functions
    Session session_{};

    /// Cuda function launchers
    std::unique_ptr<CudaFunctionLauncher> render_volume_;
    std::unique_ptr<CudaFunctionLauncher> render_slice_;

    /// required buffer size, changes if video size is changed
    size_t required_size_ = 0;
    /// available display buffers
    std::list<std::shared_ptr<IBlob>> display_buffers_;
    /// number of outstanding display buffers (sent to encoder and not received back yet)
    uint32_t outstanding_display_buffers_ = 0;
};

Renderer::Impl::Impl(const std::shared_ptr<MessageReceiver> &video_msg_receiver, uint32_t cuda_device_ordinal,
                     const std::string &mhd_file_name)
    : video_msg_receiver_(video_msg_receiver)
    , cuda_device_ordinal_(cuda_device_ordinal)
{
    // we are copying the Session structure data to the device therefore it needs to be trivially copyable
    static_assert(std::is_trivially_copyable<Session>::value, "Session needs to be trivially copyable");

    // initialize Cuda
    CudaCheck(cuInit(0));
    cuda_context_.reset(new CudaPrimaryContext(cuda_device_ordinal_));

    {
        CUdevice cuda_device = 0;
        CudaCheck(cuDeviceGet(&cuda_device, cuda_device_ordinal));

        char device_name[64];
        CudaCheck(cuDeviceGetName(device_name, CountOf(device_name), cuda_device));
        Log(LogLevel::Info) << "Creating renderer on device " << cuda_device_ordinal << " (" << device_name << ")";
    }

    // load the MHD file
    Log(LogLevel::Info) << "Loading MHD file " << mhd_file_name;

    MHDLoader mhd_loader;
    mhd_loader.Load(mhd_file_name);

    Log(LogLevel::Info) << "Volume size: " << mhd_loader.GetSize()(0) << ", " << mhd_loader.GetSize()(1) << ", "
                        << mhd_loader.GetSize()(2);
    Log(LogLevel::Info) << "Volume element type: " << mhd_loader.GetElementType();

    // create a Cuda array using the volume
    CUarray_format array_format;
    switch (mhd_loader.GetElementType())
    {
    case MHDLoader::ElementType::INT8:
        array_format = CU_AD_FORMAT_SIGNED_INT8;
        break;
    case MHDLoader::ElementType::UINT8:
        array_format = CU_AD_FORMAT_UNSIGNED_INT8;
        break;
    case MHDLoader::ElementType::INT16:
        array_format = CU_AD_FORMAT_SIGNED_INT16;
        break;
    case MHDLoader::ElementType::UINT16:
        array_format = CU_AD_FORMAT_UNSIGNED_INT16;
        break;
    case MHDLoader::ElementType::INT32:
        array_format = CU_AD_FORMAT_SIGNED_INT32;
        break;
    case MHDLoader::ElementType::UINT32:
        array_format = CU_AD_FORMAT_UNSIGNED_INT32;
        break;
    case MHDLoader::ElementType::FLOAT:
        array_format = CU_AD_FORMAT_FLOAT;
        break;
    default:
        throw InvalidState() << "Unhandled element type "
                             << static_cast<std::underlying_type<MHDLoader::ElementType>::type>(
                                    mhd_loader.GetElementType());
    }

    const Vector3ui size = mhd_loader.GetSize();
    volume_array_.reset(new CudaArray(size, array_format));

    // Create the Cuda texture from the array
    volume_texture_.reset(new CudaTexture(volume_array_, CU_TR_FILTER_MODE_LINEAR, CU_TRSF_NORMALIZED_COORDINATES));
    // update the session with the Cuda texture
    session_.volume_texture_ = volume_texture_->GetTexture().get();
    // step size depends on the max resolution the volume
    session_.step_ = 1.f / std::max(size(0), std::max(size(1), size(2)));

    // Upload volume data to the Cuda array
    {
        std::unique_ptr<IBlob::AccessGuard> access = mhd_loader.GetData()->Access(CU_STREAM_PER_THREAD);

        CUDA_MEMCPY3D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost       = access->GetData();
        copy.srcPitch      = size(0) * mhd_loader.GetBytesPerElement();
        copy.srcHeight     = size(1);

        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray      = volume_array_->GetArray().get();

        copy.WidthInBytes = copy.srcPitch;
        copy.Height       = copy.srcHeight;
        copy.Depth        = size(2);

        CudaCheck(cuMemcpy3DAsync(&copy, CU_STREAM_PER_THREAD));
    }

    // get the min/max values from the volume
    {
        std::unique_ptr<CudaFunctionLauncher> min_max_launcher;

        // number of threads per row
        constexpr auto threads = 32;
        min_max_launcher.reset(new CudaFunctionLauncher(&MinMax<float, threads>, threads * 2 * sizeof(float)));

        // the min/max kernel needs special block dimensions and a special launch grid calculation function
        dim3 block_dim;
        block_dim.x = threads;
        block_dim.y = 1;
        block_dim.z = 1;
        while (block_dim.x * block_dim.y * block_dim.z < min_max_launcher->GetOptimalBlockSize())
        {
            if (block_dim.y > block_dim.z)
            {
                block_dim.z *= 2;
            }
            else
            {
                block_dim.y *= 2;
            }
        }
        min_max_launcher->SetBlockDim(block_dim);

        const std::function<dim3(const Vector3ui &grid)> calc_launch_grid([block_dim](const Vector3ui &grid) -> dim3 {
            dim3 launch_grid;
            launch_grid.x = 1;
            launch_grid.y = std::min(grid(1), (grid(1) + (block_dim.y - 1)) / block_dim.y);
            launch_grid.z = std::min(grid(2), (grid(2) + (block_dim.z - 1)) / block_dim.z);
            return launch_grid;
        });
        min_max_launcher->SetCalcLaunchGrid(calc_launch_grid);

        // buffer to receive the volume min/max values
        std::unique_ptr<CudaMemory> memory_min_max(new CudaMemory(2 * sizeof(float)));

        // initialize min/max values and upload to device
        float min_max[2]{std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
        CudaCheck(
            cuMemcpyHtoDAsync(memory_min_max->GetMemory().get(), &min_max[0], sizeof(min_max), CU_STREAM_PER_THREAD));

        // get the density min/max values
        CudaTexture volume_texture(volume_array_, CU_TR_FILTER_MODE_POINT, /*flags*/ 0);
        min_max_launcher->Launch(size, volume_texture.GetTexture().get(), uint3(size),
                                 reinterpret_cast<float *>(memory_min_max->GetMemory().get()));

        // read to host and update the session parameter
        CudaCheck(
            cuMemcpyDtoHAsync(&min_max[0], memory_min_max->GetMemory().get(), sizeof(min_max), CU_STREAM_PER_THREAD));
        CudaCheck(cuStreamSynchronize(CU_STREAM_PER_THREAD));

        Log(LogLevel::Info) << "Volume element min/max: " << min_max[0] << ", " << min_max[1];

        session_.density_min_ = min_max[0];
        if (min_max[0] != min_max[1])
        {
            session_.inv_density_range_ = 1.f / (min_max[1] - min_max[0]);
        }
        else
        {
            session_.inv_density_range_ = 1.f;
        }
    }

    // initialize function launcher, launchers are used to call cuda functions
    render_volume_.reset(new CudaFunctionLauncher(&RenderVolume));
    render_slice_.reset(new CudaFunctionLauncher(&RenderSlice));
}

void Renderer::Impl::Run()
{
    // run the render thread
    thread_.reset(new Thread("Render thread", [this](std::function<void()> ready) { ThreadFunction(ready); }));
}

void Renderer::Impl::Shutdown()
{
    // shutdown the thread
    EnqueueMessage(std::make_shared<Message>(IDMESSAGE_SHUTDOWN));

    // destroy the thread
    thread_.reset();
}

void Renderer::Impl::ThreadFunction(std::function<void()> ready)
{
    CudaPrimaryContext cuda_context(cuda_device_ordinal_);

    // thread is ready now
    ready();

    // frame render start time
    std::chrono::time_point<std::chrono::steady_clock> frame_start;
    // if set then state had changed
    bool state_changed = false;

    while (true)
    {
        try
        {
            // if set then shutdown
            bool shutdown = false;

            // If
            // - no frame had been rendered yet
            // - or there are too many outstanding display buffers
            // - there is no video stream in play state
            // then wait for new messages
            if ((frame_start.time_since_epoch().count() == 0) ||
                (outstanding_display_buffers_ >= MAX_OUTSTANDING_DISPLAY_BUFFERS) || !video_state_ ||
                (video_state_->streams_.front().state_ != VideoMessage::State::PLAY))
            {
                Wait();
            }
            else
            {
                // Else check the time required for the last frame and wait enough to match the desired
                // output frame rate.
                const auto video_frame_duration =
                    std::chrono::duration<float, std::milli>(1000.f / video_state_->streams_.front().frame_rate_);
                const auto last_frame_duration = std::chrono::steady_clock::now() - frame_start;
                if (last_frame_duration < video_frame_duration)
                {
                    WaitFor(video_frame_duration - last_frame_duration);
                }
            }

            // check for messages
            std::shared_ptr<const Message> message;
            while ((message = DequeueMessage()))
            {
                if (message->GetID() == CameraInterface::Message::id_)
                {
                    camera_ =
                        *(std::static_pointer_cast<const CameraInterface::Message>(message)->data_out_.cameras.begin());
                    session_.camera_setup_.Update(camera_);
                    state_changed = true;
                }
                else if (message->GetID() == SliceInterface::Message::id_)
                {
                    slice_        = std::static_pointer_cast<const SliceInterface::Message>(message)->data_out_;
                    state_changed = true;
                }
                else if (message->GetID() == VideoMessage::id_)
                {
                    const std::shared_ptr<const VideoMessage> &video_message =
                        std::static_pointer_cast<const VideoMessage>(message);

                    // if the display buffer size changed, throw away the display and render buffers
                    if (video_state_)
                    {
                        const size_t new_required_size = video_message->streams_.front().width_ *
                                                         video_message->streams_.front().height_ * sizeof(uchar4);
                        if (required_size_ != new_required_size)
                        {
                            display_buffers_.clear();
                            required_size_ = new_required_size;
                        }
                    }

                    // now set the new video state and update the session with it
                    video_state_  = video_message;
                    state_changed = true;
                }
                else if (message->GetID() == BoomerangBlobMessage::id_)
                {
                    const auto &boomerang_blob_message = std::static_pointer_cast<const BoomerangBlobMessage>(message);

                    // decrement the outstanding display buffers counter, will be used to limit the inflight buffers
                    assert(outstanding_display_buffers_ > 0);
                    --outstanding_display_buffers_;

                    // if the video size did not change, add the memory back to the pool, else it will be freed when the
                    // BoomerangBlob message is destroyed.
                    if (boomerang_blob_message->blob_->GetSize() == required_size_)
                    {
                        // add to the pool of display buffers so that the memory can be reused
                        display_buffers_.emplace_back(boomerang_blob_message->blob_);
                    }
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

            // check if we should render if
            // - state changed
            // - and the video is in `play` state
            // - display buffers are available
            if (state_changed &&
                (video_state_ && (video_state_->streams_.front().state_ == VideoMessage::State::PLAY)) &&
                (outstanding_display_buffers_ < MAX_OUTSTANDING_DISPLAY_BUFFERS))
            {
                // check if a new frame needs to be drawn, this is the case if the time since the last
                // frame is longer then the video frame duration (or this is the first frame)
                bool redraw = false;
                if (frame_start.time_since_epoch().count() != 0)
                {
                    const auto video_frame_duration =
                        std::chrono::duration<float, std::milli>(1000.f / video_state_->streams_.front().frame_rate_);
                    const auto since_last_frame = std::chrono::steady_clock::now() - frame_start;
                    if (since_last_frame >= video_frame_duration)
                    {
                        redraw = true;
                    }
                }
                else
                {
                    // first frame
                    redraw = true;
                }

                if (redraw)
                {
                    // start a new frame
                    frame_start = std::chrono::steady_clock::now();

                    // get the display buffer, either use a previous buffer returned by the encoder or allocate
                    // a new one
                    std::shared_ptr<IBlob> display_buffer;
                    if (display_buffers_.empty())
                    {
                        // allocate display buffers in video context
                        CudaContext::ScopedPush context(video_state_->streams_.front().cuda_context_);
                        display_buffer = std::make_shared<CudaMemoryBlob>(std::make_unique<CudaMemory>(required_size_));
                    }
                    else
                    {
                        display_buffer = std::move(display_buffers_.front());
                        display_buffers_.pop_front();
                    }

                    // another display buffer is outstanding
                    assert(outstanding_display_buffers_ < MAX_OUTSTANDING_DISPLAY_BUFFERS);
                    ++outstanding_display_buffers_;

                    const uint2 size =
                        make_uint2(video_state_->streams_.front().width_, video_state_->streams_.front().height_);

                    {
                        // start access to the display buffer
                        const std::unique_ptr<IBlob::AccessGuard> access = display_buffer->Access(CU_STREAM_PER_THREAD);

                        // update the session with the new display buffer
                        session_.buffer_display_.Update(reinterpret_cast<CUdeviceptr>(access->GetData()),
                                                        size.x * sizeof(uchar4));

                        // copy the session data to the GPU
                        CudaRTCheck(cudaMemcpyToSymbolAsync(g_session, &session_, sizeof(Session), 0 /*offset*/,
                                                            cudaMemcpyHostToDevice, cudaStreamPerThread));
                        // Need to wait for the upload to finish since this is using host memory which can be overwritten
                        // when new messages arrive.
                        // This could be done asynchronously, f.e. by creating an upload queue class which uses events
                        // to determine when host memory can be released.
                        CudaCheck(cuStreamSynchronize(CU_STREAM_PER_THREAD));

                        // render the frame!

                        // render the volume (top-left)
                        const uint2 top_left_size   = make_uint2(size.x / 2, size.y / 2);
                        const uint2 top_left_offset = make_uint2(0, 0);
                        render_volume_->Launch(Vector2ui(top_left_size.x, top_left_size.y), top_left_offset,
                                               top_left_size);

                        // render the slices
                        const uint2 top_right_size   = make_uint2(size.x - top_left_size.x, top_left_size.y);
                        const uint2 top_right_offset = make_uint2(top_left_size.x, 0);
                        render_slice_->Launch(Vector2ui(top_right_size.x, top_right_size.y), top_right_offset,
                                              top_right_size, Matrix3x3(), slice_.slice(1));
                        const uint2 bottom_left_size   = make_uint2(top_left_size.x, size.y - top_left_size.y);
                        const uint2 bottom_left_offset = make_uint2(0, top_left_size.y);
                        render_slice_->Launch(
                            Vector2ui(bottom_left_size.x, bottom_left_size.y), bottom_left_offset, bottom_left_size,
                            Matrix3x3({{{{0.f, 0.f, 1.f}}, {{1.f, 0.f, 0.f}}, {{0.f, 1.f, 0.f}}}}), slice_.slice(0));
                        const uint2 bottom_right_size = make_uint2(size.x - top_left_size.x, size.y - top_left_size.y);
                        const uint2 bottom_right_offset = make_uint2(top_left_size.x, top_left_size.y);
                        render_slice_->Launch(
                            Vector2ui(bottom_right_size.x, bottom_right_size.y), bottom_right_offset, bottom_right_size,
                            Matrix3x3({{{{1.f, 0.f, 0.f}}, {{0.f, 0.f, 1.f}}, {{0.f, 1.f, 0.f}}}}), slice_.slice(2));
                    }

                    // send a message to the encoder thread to encode the resource to the stream
                    auto video_encode_message = std::make_shared<VideoEncodeMessage>();

                    video_encode_message->width_  = size.x;
                    video_encode_message->height_ = size.y;
                    video_encode_message->memory_ =
                        std::make_shared<BoomerangBlob>(this->shared_from_this(), std::move(display_buffer));
                    video_encode_message->format_ = IVideoEncoder::Format::ABGR;

                    video_msg_receiver_->EnqueueMessage(video_encode_message);

                    // settings are now consumed
                    state_changed = false;
                }
            }
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Render thread threw exception " << e.what();
        }
        catch (...)
        {
            Log(LogLevel::Error) << "Render thread threw unknown exception";
        }
    }
} // namespace clara::viz

Renderer::Renderer(const std::shared_ptr<MessageReceiver> &video_msg_receiver, uint32_t cuda_device_ordinal,
                   const std::string &mhd_file_name)
    : impl_(new Impl(video_msg_receiver, cuda_device_ordinal, mhd_file_name))
{
}

Renderer::~Renderer() {}

void Renderer::Run()
{
    impl_->Run();
}

void Renderer::Shutdown()
{
    impl_->Shutdown();
}

std::shared_ptr<MessageReceiver> Renderer::GetReceiver() const
{
    return impl_;
}

} // namespace clara::viz
