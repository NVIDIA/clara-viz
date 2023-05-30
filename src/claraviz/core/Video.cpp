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

#include "claraviz/core/Video.h"

#include "claraviz/interface/VideoInterface.h"
#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/util/Thread.h"
#include "claraviz/util/Blob.h"
#include "claraviz/video/NvEncVideoEncoder.h"
#ifdef CLARA_VIZ_WITH_OPENH264
#include "claraviz/video/OpenH264VideoEncoder.h"
#endif

namespace clara::viz
{

DEFINE_CLASS_MESSAGEID(VideoMessage);
DEFINE_CLASS_MESSAGEID(VideoEncodeMessage);

namespace
{

/**
 * Shutdown message ID
 */
DEFINE_MESSAGEID(IDMESSAGE_SHUTDOWN);

} // anonymous namespace

/**
 * Server implementation data
 */
class Video::Impl
{
public:
    explicit Impl(const std::shared_ptr<Video> &base, uint32_t cuda_device_ordinal);
    Impl() = delete;

    void Run();
    void Shutdown();
    std::unique_ptr<IVideoEncoder> CreateEncoder() const;

private:
    class EncodeTarget;
    class Stream;

    /**
     * Interface thread function.
     */
    void ThreadFunction(std::function<void()> ready);

    /**
     * Send to encoder.
     *
     * @param width [in]
     * @param height [in]
     * @param memory [in]
     * @param format [in]
     * @param stream [in]
     */
    void Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory, IVideoEncoder::Format format,
                Stream *const stream);

    /**
     * Send a video message to the client with the current state.
     */
    void SendVideoMessage();

    /**
     * Get a stream for an stream name
     *
     * @param name [in]
     *
     * @returns the stream for the name or streams_::end() if the stream is not found
     */
    std::list<Stream>::iterator GetStream(const std::string &name);

    /**
     * Describes a frame which has to be encoded at a given target time.
     */
    class EncodeTarget
    {
    public:
        /**
         * Construct
         *
         * @param width [in] width of the resource
         * @param height [in] height of the resource
         * @param memory [in] the resource to encode
         * @param format [in] pixel format of the resource
         * @param target_time [in] point in time the frame should be encoded
         */
        EncodeTarget(uint32_t width, uint32_t height, std::shared_ptr<IBlob> &memory, IVideoEncoder::Format format,
                     const std::chrono::time_point<std::chrono::steady_clock> &target_time)
            : width_(width)
            , height_(height)
            , memory_(std::move(memory))
            , format_(format)
            , target_time_(target_time)
        {
        }

        /// width of the resource
        const uint32_t width_;
        /// height of the resource
        const uint32_t height_;
        /// the resource to encode
        const std::shared_ptr<IBlob> memory_;
        /// pixel format of the resource
        const IVideoEncoder::Format format_;

        /// target time
        const std::chrono::time_point<std::chrono::steady_clock> target_time_;
    };

    /// data for one stream
    struct Stream
    {
        Stream(std::unique_ptr<IVideoEncoder> &encoder)
            : encoder_(std::move(encoder))
        {
        }

        /// interface data
        VideoInterface::DataOut::Video video_;

        /// video encoder
        std::unique_ptr<IVideoEncoder> encoder_;

        /// time the last frame had been encoded
        std::chrono::time_point<std::chrono::steady_clock> last_frame_time_;

        /// the current frame waiting to be encoded
        std::unique_ptr<EncodeTarget> encode_target_;

        /// current frame duration
        std::chrono::duration<float, std::milli> frame_duration_;
    };

    /// base class (needs to be a weak_ptr to break circular dependency)
    const std::weak_ptr<Video> base_;

    // Cuda device to use for encoding
    const uint32_t cuda_device_ordinal_;

    /// interface thread
    std::unique_ptr<Thread> thread_;

    /// the video streams
    std::list<Stream> streams_;
};

std::shared_ptr<Video> Video::Create(uint32_t cuda_device_ordinal)
{
    auto video = std::make_shared<Video>();

    video->impl_.reset(new Video::Impl(video, cuda_device_ordinal));

    return video;
}

Video::~Video()
{
    Shutdown();
}

void Video::Run()
{
    impl_->Run();
}

void Video::Shutdown()
{
    if (impl_)
    {
        impl_->Shutdown();
    }
}

std::unique_ptr<IVideoEncoder> Video::GetEncoder() const
{
    return impl_->CreateEncoder();
}

Video::Impl::Impl(const std::shared_ptr<Video> &base, uint32_t cuda_device_ordinal)
    : base_(base)
    , cuda_device_ordinal_(cuda_device_ordinal)
{
    if (!base)
    {
        throw InvalidArgument("base") << "is nullptr";
    }
}

void Video::Impl::Run()
{
    // run the render thread
    thread_.reset(new Thread("Encoder thread", [this](std::function<void()> ready) { ThreadFunction(ready); }));
}

void Video::Impl::Shutdown()
{
    if (thread_)
    {
        // might be called when the base class is already destroyed
        const std::shared_ptr<Video> base = base_.lock();
        if (base)
        {
            base->EnqueueMessage(std::make_shared<Message>(IDMESSAGE_SHUTDOWN));
        }

        // destroy the thread
        thread_.reset();
    }
}

std::unique_ptr<IVideoEncoder> Video::Impl::CreateEncoder() const
{
    std::unique_ptr<IVideoEncoder> encoder;

    encoder.reset(new NvEncVideoEncoder(cuda_device_ordinal_));
    if (!encoder->Query(IVideoEncoder::Capability::IS_SUPPORTED))
    {
#ifdef CLARA_VIZ_WITH_OPENH264
        encoder.reset(new OpenH264VideoEncoder(cuda_device_ordinal_));
        if (!encoder->Query(IVideoEncoder::Capability::IS_SUPPORTED))
#endif
        {
            throw RuntimeError() << "Video encoding not supported";
        }
    }

    return encoder;
}

std::list<Video::Impl::Stream>::iterator Video::Impl::GetStream(const std::string &name)
{
    return std::find_if(streams_.begin(), streams_.end(),
                        [name](const Stream &stream) { return stream.video_.name == name; });
}

void Video::Impl::ThreadFunction(std::function<void()> ready)
{
    CudaCheck(cuInit(0));

    // thread is ready now
    ready();

    while (true)
    {
        try
        {
            auto base = base_.lock();

            // check if there is a pending frame to encode
            bool had_pending_frame = false;
            for (auto &&stream : streams_)
            {
                if (stream.encode_target_)
                {
                    // check if time is left until the frame should be encoded, if this is the case
                    // wait until the target time has elapsed
                    std::chrono::duration<float, std::milli> time_to_target =
                        stream.encode_target_->target_time_ - std::chrono::steady_clock::now();
                    if (time_to_target.count() > 0)
                    {
                        base->WaitFor(time_to_target + std::chrono::milliseconds(1));
                        // calculate the new time to target after waiting
                        time_to_target = stream.encode_target_->target_time_ - std::chrono::steady_clock::now();
                    }

                    // if the target time had been reached, encode the frame
                    if (time_to_target <= std::chrono::milliseconds(1))
                    {
                        Encode(stream.encode_target_->width_, stream.encode_target_->height_,
                               stream.encode_target_->memory_, stream.encode_target_->format_, &stream);

                        // we are now done with this frame
                        stream.encode_target_.reset();
                    }

                    had_pending_frame = true;
                }
            }
            if (!had_pending_frame)
            {
                // nothing to encode, wait for the next message
                base->Wait();
            }

            bool shutdown = false;
            std::shared_ptr<const Message> message;
            while ((message = base->DequeueMessage()))
            {
                Log(LogLevel::Debug) << "Video received " << message->GetID().GetName();

                if (message->GetID() == VideoInterface::Message::id_)
                {
                    // interface message
                    const VideoInterface::DataOut video_data =
                        std::static_pointer_cast<const VideoInterface::Message>(message)->data_out_;

                    // check if streams had been deleted
                    for (auto stream_it = streams_.begin(); stream_it != streams_.end(); ++stream_it)
                    {
                        auto it = std::find_if(video_data.videos.begin(), video_data.videos.end(),
                                               [&stream_it](const VideoInterface::DataOut::Video &video) {
                                                   return video.name == stream_it->video_.name;
                                               });
                        if (it == video_data.videos.end())
                        {
                            // stream had been deleted, remove
                            stream_it = streams_.erase(stream_it);
                        }
                    }

                    // update stream data and check if there are new ones
                    for (auto &&video : video_data.videos)
                    {
                        auto stream = GetStream(video.name);
                        if (stream == streams_.end())
                        {
                            // new stream, create one
                            std::unique_ptr<IVideoEncoder> encoder = CreateEncoder();
                            streams_.emplace_back(encoder);
                            stream = streams_.end();
                            --stream;
                        }
                        // update the data
                        stream->video_ = video;

                        // update frame duration
                        stream->frame_duration_ = std::chrono::duration<float, std::milli>(
                            (video.frame_rate != 0.f) ? (1000.f / video.frame_rate) : 0.f);
                    }

                    // send the video message to the client
                    SendVideoMessage();
                }
                else if (message->GetID() == VideoEncodeMessage::id_)
                {
                    // client message, encode a frame
                    auto encode_message = std::static_pointer_cast<const VideoEncodeMessage>(message);
                    auto stream_name    = encode_message->stream_name_;
                    auto width          = encode_message->width_;
                    auto height         = encode_message->height_;
                    auto memory         = encode_message->memory_;
                    auto format         = encode_message->format_;

                    auto stream = GetStream(stream_name);
                    if (stream == streams_.end())
                    {
                        throw InvalidState() << "Stream with name " << stream_name << " not found";
                    }

                    // calculate the target time if a target frame rate is given, we want to reach the target
                    // frame rate but never exceed it
                    if ((stream->video_.frame_rate != 0.f) && stream->last_frame_time_.time_since_epoch().count())
                    {
                        if (stream->encode_target_)
                        {
                            // drop the existing encode target and replace with a new one with the same time
                            // and the new memory to encode
                            stream->encode_target_.reset(
                                new EncodeTarget(width, height, memory, format, stream->encode_target_->target_time_));
                        }
                        else
                        {
                            const auto time_since_last_frame =
                                std::chrono::steady_clock::now() - stream->last_frame_time_;

                            // check the time since the last frame, encode at next frame time point if less
                            // than one frame had elapsed. Else fall through and encode immediately.
                            if (time_since_last_frame < stream->frame_duration_)
                            {
                                const auto target_time =
                                    stream->last_frame_time_ +
                                    std::chrono::duration_cast<std::chrono::microseconds>(stream->frame_duration_);

                                stream->encode_target_.reset(
                                    new EncodeTarget(width, height, memory, format, target_time));
                            }
                        }
                    }

                    // no target time because more than one frame elapsed since the last encoding, encode immediately
                    if (!stream->encode_target_)
                    {
                        Encode(width, height, memory, format, &*stream);

                        if (stream->frame_duration_.count() != 0.f)
                        {
                            // Workaround for streaming to browsers
                            // Background: Chrome (and Firefox) do not update the display if just a single frame is streamed,
                            //             maybe there is some internal logic which needs at least two available frames to start
                            //             playback. Sending single IDR frames did not trigger playback either.
                            // Solution: Enqueue the last encoded frame for repetition after two frame periods. That means if no
                            //           additional frame is send by the renderer in time we encode and stream the last frame again
                            //           to trigger playback in the browser.
                            const auto target_time =
                                stream->last_frame_time_ +
                                std::chrono::duration_cast<std::chrono::microseconds>(stream->frame_duration_) * 2;
                            stream->encode_target_.reset(new EncodeTarget(width, height, memory, format, target_time));
                        }
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
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Video thread threw exception " << e.what();
        }
        catch (...)
        {
            Log(LogLevel::Error) << "Video thread threw unknown exception";
        }
    }
}

void Video::Impl::SendVideoMessage()
{
    auto video_message = std::make_shared<VideoMessage>();

    for (auto &&stream : streams_)
    {
        // add a stream to the video message
        video_message->streams_.emplace_back();
        VideoMessage::Stream &out_stream = video_message->streams_.back();

        switch (stream.video_.state)
        {
        case VideoInterfaceState::PLAY:
            out_stream.state_ = VideoMessage::State::PLAY;
            break;
        case VideoInterfaceState::PAUSE:
            out_stream.state_ = VideoMessage::State::PAUSE;
            break;
        case VideoInterfaceState::STOP:
            out_stream.state_ = VideoMessage::State::STOP;
            break;
        }
        out_stream.name_         = stream.video_.name;
        out_stream.width_        = stream.video_.width;
        out_stream.height_       = stream.video_.height;
        out_stream.frame_rate_   = stream.video_.frame_rate;
        out_stream.cuda_context_ = stream.encoder_->GetCudaContext()->GetContext();
    }

    // forward the data to the client
    base_.lock()->SendMessage(video_message);
}

void Video::Impl::Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory,
                         IVideoEncoder::Format format, Stream *const stream)
{
    // record the time the frame had been encoded
    stream->last_frame_time_ = std::chrono::steady_clock::now();

    if (!stream->video_.stream || stream->video_.stream->Failed())
    {
        // if the stream is not active or has the fail flag set, stop the stream and update the client
        Log(LogLevel::Warning) << (!stream->video_.stream ? "Video stream is not active"
                                                          : "Video stream had been closed on failure");

        stream->video_.state = VideoInterfaceState::STOP;

        SendVideoMessage();
    }
    else
    {
        // update the encoder with the properties received from the interface
        stream->encoder_->SetStream(stream->video_.stream);
        stream->encoder_->SetFrameRate(stream->video_.frame_rate);
        stream->encoder_->SetBitRate(stream->video_.bit_rate);

        // encode the frame in playing state only, else ignore
        if (stream->video_.state == VideoInterfaceState::PLAY)
        {
            stream->encoder_->Encode(width, height, memory, format);
        }
    }
}

} // namespace clara::viz
