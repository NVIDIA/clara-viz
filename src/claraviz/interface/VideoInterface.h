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

#include <type_traits>
#include <memory>
#include <list>

#include "claraviz/interface/InterfaceData.h"

namespace clara::viz
{

class IVideoStream;

/// video state enum
enum class VideoInterfaceState
{
    PLAY,  // video is playing
    PAUSE, // video is paused
    STOP   // video is stopped
};

namespace detail
{

template<template<typename> typename V>
class VideoInterfaceDataBase
{
public:
    virtual ~VideoInterfaceDataBase() = default;
};

template<>
class VideoInterfaceDataBase<InterfaceValueT>
{
public:
    uint32_t min_width_;  ///< minimum output width supported
    uint32_t max_width_;  ///< maximum output width supported
    uint32_t min_height_; ///< minimum output height supported
    uint32_t max_height_; ///< maximum output height supported
};

} // namespace detail

/**
 * Video interface parameter definition.
 */
template<template<typename> typename V>
struct VideoInterfaceData : public detail::VideoInterfaceDataBase<V>
{
    VideoInterfaceData();

    struct Video
    {
        /**
         * Constructor for InterfaceValueT
         */
        template<typename U = V<int>,
                 class      = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
        Video(uint32_t min_width, uint32_t max_width, uint32_t min_height, uint32_t max_height,
              const std::string &name);

        /**
         * Constructor for InterfaceDirectT
         */
        template<typename U = V<int>,
                 class      = typename std::enable_if<std::is_same<U, InterfaceDirectT<int>>::value>::type>
        Video()
        {
        }

        /**
         * Name for the stream.
         */
        std::string name;

        /**
         * Width of the video, value needs to be evenly divisible by 2
         *
         * Default: min_width_
         *
         * Range: ]min_width_, max_width_]
         */
        V<uint32_t> width;

        /**
         * Height of the video, value needs to be evenly divisible by 2
         *
         * Default: min_height_
         *
         * Range: ]min_height_, max_height_]
         */
        V<uint32_t> height;

        /**
         * Target framerate of the video
         * If set to 0.0 the frames will delivered when rendering is done. Converging renderers
         * will deliver the final frame only.
         *
         * Default: 30.0
         *
         * Range: [0.0, inf]
         */
        V<float> frame_rate;

        /**
         * Target bitrate of the video
         *
         * Default: 1 * 1024 * 1024
         *
         * Range: ]0.0, UINT32_MAX]
         */
        V<uint32_t> bit_rate;

        /**
         * Video output stream. Responsible for writing the output of the video stream.
         * Create a class derived from IVideoStream to handle the video stream output.
         */
        std::shared_ptr<IVideoStream> stream;

        /**
         * Video state
         *
         * Default: VideoInterfaceState::STOP
         */
        VideoInterfaceState state;
    };

    /**
     * Video streams
     */
    std::list<Video> videos;

    /**
     * Get the video stream with the given name, add it if it does not exist already
     *
     * @param name [in] stream name, default empty
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    Video *GetOrAddVideo(const std::string &name);

    /**
     * Get the video stream with the given name
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    Video *GetVideo(const std::string &name = std::string());

    /**
     * Get the const video stream with the given name
     *
     * @param name [in]
     */
    const Video *GetVideo(const std::string &name = std::string()) const;
};

namespace detail
{

using VideoInterfaceDataIn  = VideoInterfaceData<InterfaceValueT>;
using VideoInterfaceDataOut = VideoInterfaceData<InterfaceDirectT>;

struct VideoInterfaceDataPrivate
{
};

using VideoInterfaceBase = InterfaceData<VideoInterfaceDataIn, VideoInterfaceDataOut, VideoInterfaceDataPrivate>;

} // namespace detail

/**
 * Video interface, see @ref VideoInterfaceData for the interface properties.
 */
class VideoInterface : public detail::VideoInterfaceBase
{
public:
    /**
     * Construct
     *
     * @param min_width [in] minimum video width
     * @param min_height [in] minimum video height
     * @param max_width [in] maximum video width
     * @param max_height [in] maximum video height
     **/
    explicit VideoInterface(uint32_t min_width, uint32_t min_height, uint32_t max_width, uint32_t max_height);
    VideoInterface() = delete;

    void Reset() final;

private:
    const uint32_t min_width_;
    const uint32_t max_width_;
    const uint32_t min_height_;
    const uint32_t max_height_;
};

} // namespace clara::viz
