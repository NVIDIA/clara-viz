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

#include "claraviz/interface/VideoInterface.h"

#include "claraviz/util/Validator.h"
#include "claraviz/video/VideoEncoder.h"

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(VideoInterface::Message);

template<>
template<>
VideoInterface::DataIn::Video::Video(uint32_t min_width, uint32_t max_width, uint32_t min_height, uint32_t max_height,
                                     const std::string &name)
    : name(name)
    , width(min_width,
            [min_width, max_width](const uint32_t &value) {
                if ((value < min_width) || (value & 1) || (value > max_width))
                    throw InvalidArgument("Video width") << "expected to be >= " << min_width << ", <= " << max_width
                                                         << " and evenly divisible by 2 but is " << value;
            })
    , height(min_height,
             [min_height, max_height](const uint32_t &value) {
                 if ((value < min_height) || (value & 1) || (value > max_height))
                     throw InvalidArgument("Video height")
                         << "expected to be >= " << min_height << ", <= " << max_height
                         << " and evenly divisible by 2 but is " << value;
             })
    , frame_rate(30.f, [](const float &value) { ValidatorMinInclusive(value, 0.f, "Video frame rate"); })
    , bit_rate(1 * 1024 * 1024, [](const uint32_t &value) { ValidatorMinExclusive(value, 0u, "Video bit rate"); })
    , state(VideoInterfaceState::STOP)
{
}

template<>
VideoInterface::DataIn::VideoInterfaceData()
{
}

template<>
template<>
VideoInterface::DataIn::Video *VideoInterface::DataIn::GetOrAddVideo(const std::string &name)
{
    std::list<Video>::iterator it =
        std::find_if(videos.begin(), videos.end(), [name](const Video &video) { return video.name == name; });
    if (it == videos.end())
    {
        videos.emplace_back(min_width_, max_width_, min_height_, max_height_, name);
        it = videos.end();
        --it;
        it->name = name;
    }
    return &*it;
}

namespace detail
{

template<typename T>
typename T::Video *GetVideo(std::list<typename T::Video> &videos, const std::string &name)
{
    typename std::list<typename T::Video>::iterator it = std::find_if(
        videos.begin(), videos.end(), [name](const typename T::Video &video) { return video.name == name; });
    if (it == videos.end())
    {
        throw InvalidArgument("name") << "Video with name '" << name << "' not found";
    }
    return &*it;
}

template<typename T>
const typename T::Video *GetVideo(const std::list<typename T::Video> &videos, const std::string &name)
{
    typename std::list<typename T::Video>::const_iterator it = std::find_if(
        videos.cbegin(), videos.cend(), [name](const typename T::Video &video) { return video.name == name; });
    if (it == videos.end())
    {
        throw InvalidArgument("name") << "Video with name '" << name << "' not found";
    }
    return &*it;
}

} // namespace detail

template<>
template<>
VideoInterface::DataIn::Video *VideoInterface::DataIn::GetVideo(const std::string &name)
{
    return detail::GetVideo<VideoInterface::DataIn>(videos, name);
}

template<>
const VideoInterface::DataIn::Video *VideoInterface::DataIn::GetVideo(const std::string &name) const
{
    return detail::GetVideo<const VideoInterface::DataIn>(videos, name);
}

template<>
const VideoInterface::DataOut::Video *VideoInterface::DataOut::GetVideo(const std::string &name) const
{
    return detail::GetVideo<const VideoInterface::DataOut>(videos, name);
}

template<>
VideoInterface::DataOut::VideoInterfaceData()
{
}

VideoInterface::VideoInterface(uint32_t min_width, uint32_t min_height, uint32_t max_width, uint32_t max_height)
    : min_width_(min_width)
    , min_height_(min_height)
    , max_width_(max_width)
    , max_height_(max_height)
{
    Reset();
}

void VideoInterface::Reset()
{
    detail::VideoInterfaceBase::Reset();

    AccessGuard access(*this);

    access->min_width_  = min_width_;
    access->min_height_ = min_height_;
    access->max_width_  = max_width_;
    access->max_height_ = max_height_;

    // create the default stream
    access->GetOrAddVideo("");
}

/**
 * Copy a video interface structure to a video POD structure.
 */
template<>
VideoInterface::DataOut detail::VideoInterfaceBase::Get()
{
    AccessGuardConst access(this);

    VideoInterface::DataOut data_out;
    data_out.videos.clear();
    for (auto &&video_in : access->videos)
    {
        data_out.videos.emplace_back();
        VideoInterface::DataOut::Video &video_out = data_out.videos.back();

        video_out.name       = video_in.name;
        video_out.width      = video_in.width.Get();
        video_out.height     = video_in.height.Get();
        video_out.frame_rate = video_in.frame_rate.Get();
        video_out.bit_rate   = video_in.bit_rate.Get();
        video_out.stream     = video_in.stream;
        video_out.state      = video_in.state;
    }

    return data_out;
}

} // namespace clara::viz
