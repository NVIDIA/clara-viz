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

#include "VideoStream.h"

#include <claraviz/interface/VideoInterface.h>

#include <pybind11/pybind11.h>

#include <claraviz/util/Synchronized.h>
#include <claraviz/video/VideoEncoder.h>

namespace py = pybind11;

namespace clara::viz
{

namespace detail
{

class Stream : public IVideoStream
{
public:
    explicit Stream(const std::function<void(py::object, bool)> &callback)
        : callback_(callback)
    {
    }

    Stream() = delete;

    void NewStream() final
    {
        Synchronized<Data>::AccessGuard access(data_);

        access->new_stream = true;
    }

    bool Write(const char *data, size_t count) final
    {
        Synchronized<Data>::AccessGuard access(data_);

        // see https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
        py::gil_scoped_acquire acquire;
        try
        {
            callback_(py::memoryview::from_memory(data, count), access->new_stream);
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Warning) << "Video callback failed with " << e.what();
        }

        // reset the new_stream flag since it's only true for the first
        // frame of a new stream
        access->new_stream = false;

        return true;
    }

    bool Failed() const final
    {
        return false;
    }

private:
    struct Data
    {
        Data()
            : new_stream(true)
        {
        }
        bool new_stream;
    };
    Synchronized<Data> data_;

    const std::function<void(py::object, bool)> callback_;
};

} // namespace detail

class VideoStream::Impl
{
public:
    /**
     * Construct
     */
    Impl(const std::shared_ptr<clara::viz::VideoInterface> &video_interface,
         const std::function<void(py::object, bool)> &callback);
    Impl() = delete;

    void Configure(uint32_t width, uint32_t height, float frame_rate, uint32_t bit_rate);

    void Play();
    void Pause();
    void Stop();

private:
    const std::shared_ptr<clara::viz::VideoInterface> video_interface_;
};

void VideoStream::ImplDeleter::operator()(VideoStream::Impl *p) const
{
    delete p;
}

VideoStream::Impl::Impl(const std::shared_ptr<clara::viz::VideoInterface> &video_interface,
                        const std::function<void(py::object, bool)> &callback)
    : video_interface_(video_interface)
{
    VideoInterface::AccessGuard access(*video_interface_.get());

    VideoInterface::DataIn::Video *video = access->GetOrAddVideo("");

    video->stream.reset(new detail::Stream(callback));
}

void VideoStream::Impl::Configure(uint32_t width, uint32_t height, float frame_rate, uint32_t bit_rate)
{
    VideoInterface::AccessGuard access(*video_interface_.get());

    VideoInterface::DataIn::Video *video = access->GetVideo();

    video->width.Set(width);
    video->height.Set(height);
    video->frame_rate.Set(frame_rate);
    video->bit_rate.Set(bit_rate);
}

void VideoStream::Impl::Play()
{
    VideoInterface::AccessGuard access(*video_interface_.get());

    VideoInterface::DataIn::Video *video = access->GetVideo();

    video->state = VideoInterfaceState::PLAY;
}

void VideoStream::Impl::Pause()
{
    VideoInterface::AccessGuard access(*video_interface_.get());

    VideoInterface::DataIn::Video *video = access->GetVideo();

    video->state = VideoInterfaceState::PAUSE;
}

void VideoStream::Impl::Stop()
{
    VideoInterface::AccessGuard access(*video_interface_.get());

    VideoInterface::DataIn::Video *video = access->GetVideo();

    video->state = VideoInterfaceState::STOP;
}

VideoStream::VideoStream(const std::shared_ptr<clara::viz::VideoInterface> &video_interface,
                         const std::function<void(py::object, bool)> &callback)
    : impl_(new Impl(video_interface, callback))
{
}

void VideoStream::Configure(uint32_t width, uint32_t height, float frame_rate, uint32_t bit_rate)
{
    impl_->Configure(width, height, frame_rate, bit_rate);
}

void VideoStream::Play()
{
    impl_->Play();
}

void VideoStream::Pause()
{
    impl_->Pause();
}

void VideoStream::Stop()
{
    impl_->Stop();
}

} // namespace clara::viz
