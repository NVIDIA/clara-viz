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

#pragma once

#include <memory>
#include <list>

#include "claraviz/util/Message.h"
#include "claraviz/video/VideoEncoder.h"

typedef struct CUctx_st *CUcontext;

namespace clara::viz
{

class VideoInterface;
class IBlob;

/**
 * Video output server.
 * Exposes the 'VideoInterface'.
 * Communicates to the server through messages, once the video stream is established
 * a `VideoMessage` is emitted. To encode frames enqueue a `VideoEncodeMessage`.
 */
class Video
    : public std::enable_shared_from_this<Video>
    , public MessageProvider
    , public MessageReceiver
{
public:
    /**
     * Create
     *
     * @param cuda_device_ordinal [in] Cuda device to use for encoding
     *
     * @returns created instance
     */
    static std::shared_ptr<Video> Create(uint32_t cuda_device_ordinal);
    virtual ~Video();

    /**
     * Start the server
     */
    void Run();

    /**
     * Shutdown
     */
    void Shutdown();

    /**
     * @returns an encoder
     */
    std::unique_ptr<IVideoEncoder> GetEncoder() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Emitted by the Video class indicating that video is active and containing the video
 * stream parameters such as width and height.
 */
class VideoMessage : public Message
{
public:
    VideoMessage()
        : Message(id_)
    {
    }

    /// video stream state enum
    enum class State
    {
        PLAY,  // video is playing
        PAUSE, // video is paused
        STOP   // video is stopped
    };

    struct Stream
    {
        /// Unique video stream name
        std::string name_;

        /// video stream state
        State state_;

        /// Width of the video
        uint32_t width_;
        /// Height of the video
        uint32_t height_;
        /// Target framerate of the video
        float frame_rate_;
        /**
         * Cuda context which should be used to allocate the memory to be send to the encoder for encoding.
         * It's required that the memory send to the encoder is allocated within the video context.
         */
        CUcontext cuda_context_;
    };

    /// video streams
    std::list<Stream> streams_;

    /// message id
    static const MessageID id_;
};

/**
 * Encode a resource.
 * Send by the client to the Video service.
 */
class VideoEncodeMessage : public Message
{
public:
    VideoEncodeMessage()
        : Message(id_)
    {
    }

    /// Unique video stream name
    std::string stream_name_;

    /// width of the resource
    uint32_t width_;
    /// height of the resource
    uint32_t height_;
    /// the resource to encode
    std::shared_ptr<IBlob> memory_;
    /// pixel format of the resource
    IVideoEncoder::Format format_;

    /// message id
    static const MessageID id_;
};

} // namespace clara::viz
