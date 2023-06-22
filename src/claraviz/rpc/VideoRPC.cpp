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

#include "claraviz/rpc/VideoRPC.h"

#include "claraviz/video/VideoEncoder.h"
#include "claraviz/util/Synchronized.h"

namespace clara::viz
{

namespace video_v1 = nvidia::claraviz::video::v1;

namespace detail
{

class Stream : public IVideoStream
{
public:
    explicit Stream(VideoStreamContext *context, const std::string &name)
        : name_(name)
    {
        if (!context)
        {
            throw InvalidArgument("context") << "is nullptr";
        }

        Synchronized<Data>::AccessGuard access(data_);
        access->context = context;

        callback_handle_ = access->context->RegisterFinishCallback([this] {
            Synchronized<Data>::AccessGuard access(data_);
            access->context = nullptr;
        });
    }

    Stream() = delete;

    void NewStream()
    {
        Synchronized<Data>::AccessGuard access(data_);

        access->new_stream = true;
    }

    bool Write(const char *data, size_t count) final
    {
        Synchronized<Data>::AccessGuard access(data_);

        if (!access->context)
        {
            return false;
        }

        video_v1::StreamResponse response;

        response.set_data(data, count);
        // set the new stream flag and reset it since it's only true for the first
        // frame of a new stream
        response.set_new_stream(access->new_stream);
        access->new_stream = false;

        return access->context->Write(response, name_);
    }

    bool Failed() const final
    {
        Synchronized<Data>::AccessGuardConst access(data_);

        return (access->context == nullptr);
    }

    void Stop()
    {
        Synchronized<Data>::AccessGuard access(data_);

        if (access->context)
        {
            access->context->Stop();
        }
    }

private:
    struct Data
    {
        Data()
            : new_stream(true)
            , context(nullptr)
        {
        }
        bool new_stream;
        VideoStreamContext *context;
    };
    const std::string name_;
    Synchronized<Data> data_;
    VideoStreamContext::CallbackHandle callback_handle_;
};

void VideoConfigContext::ExecuteRPC(video_v1::ConfigRequest &request, video_v1::ConfigResponse &response)
{
    VideoInterface::AccessGuard access(GetResources()->video_interface_);

    VideoInterface::DataIn::Video *video = access->GetOrAddVideo(request.name());

    if (request.width())
    {
        video->width.Set(request.width());
    }
    if (request.height())
    {
        video->height.Set(request.height());
    }
    if (request.bit_rate())
    {
        video->bit_rate.Set(request.bit_rate());
    }

    video->frame_rate.Set(request.frame_rate());
}

VideoStreamContext::~VideoStreamContext()
{
    VideoInterface::AccessGuard access(GetResources()->video_interface_);

    for (auto &&video : access->videos)
    {
        if (video.stream)
        {
            std::static_pointer_cast<Stream>(video.stream)->Stop();
            video.stream.reset();
        }
    }
}

void VideoStreamContext::OnRequestReceived(const video_v1::StreamRequest &request)
{
    VideoInterface::AccessGuard access(GetResources()->video_interface_);

    VideoInterface::DataIn::Video *video = access->GetOrAddVideo(request.name());

    if (video->stream)
    {
        // if the current stream has failed, delete it
        if (video->stream->Failed())
        {
            video->stream.reset();
        }
        else
        {
            throw InvalidState() << "There is already an active stream for the stream " << request.name();
        }
    }

    video->stream.reset(new detail::Stream(this, video->name));
}

bool VideoStreamContext::Write(const ResponseType &response, const std::string &name)
{
    VideoInterface::AccessGuardConst access(&(GetResources()->video_interface_));

    const VideoInterface::DataIn::Video *video = access->GetVideo(name);

    bool result;
    if (video->stream)
    {
        result = nvrpc::ContextServerStreaming<video_v1::StreamRequest, video_v1::StreamResponse, VideoResource>::Write(
            response);
    }
    else
    {
        Log(LogLevel::Warning) << "Trying to write a video frame but there is no active stream";
        result = false;
    }
    return result;
}

void VideoStreamContext::Stop()
{
    FinishResponse();
}

VideoStreamContext::CallbackHandle VideoStreamContext::RegisterFinishCallback(const std::function<void()> &callback)
{
    return nvrpc::ContextServerStreaming<video_v1::StreamRequest, video_v1::StreamResponse,
                                         VideoResource>::RegisterFinishCallback(callback);
}

void VideoControlContext::ExecuteRPC(video_v1::ControlRequest &request, video_v1::ControlResponse &response)
{
    VideoInterface::AccessGuard access(GetResources()->video_interface_);

    VideoInterface::DataIn::Video *video = access->GetOrAddVideo(request.name());

    switch (request.state())
    {
    case video_v1::ControlRequest::PLAY:
        video->state = VideoInterfaceState::PLAY;
        break;
    case video_v1::ControlRequest::PAUSE:
        video->state = VideoInterfaceState::PAUSE;
        break;
    case video_v1::ControlRequest::STOP:
        // stop and free the stream
        if (video->stream)
        {
            std::static_pointer_cast<Stream>(video->stream)->Stop();
            video->stream.reset();
        }
        video->state = VideoInterfaceState::STOP;
        break;
    }
}

void VideoQueryLimitsContext::ExecuteRPC(video_v1::QueryLimitsRequest &request, video_v1::QueryLimitsResponse &response)
{
    VideoInterface::AccessGuard access(GetResources()->video_interface_);

    response.set_min_video_width(access->min_width_);
    response.set_min_video_height(access->min_height_);
    response.set_max_video_width(access->max_width_);
    response.set_max_video_height(access->max_height_);
}

} // namespace detail

} // namespace clara::viz
