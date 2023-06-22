/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <ostream>

#include "claraviz/util/Types.h"

namespace clara::viz
{

class IBlob;
class CudaContext;

/**
 * Video stream interface.
 * The encoder writes encoded data out to this stream. Could be a gRPC service or a file.
 */
class IVideoStream
{
public:
    virtual ~IVideoStream() {}

    /**
     * Indicates that a new stream had been started because e.g. the resolution changed.
     * When writing to a file this can be used to close the current file and start writing to
     * a new file. When streaming to a browser then a new MediaSource SourceBuffer object needs
     * to be created.
     *
     * Will also be called before the first write happens.
     */
    virtual void NewStream() = 0;

    /**
     * Write data to the stream.
     *
     * @param data [in] pointer to the data to write
     * @param count [in] amount of bytes to write;
     *
     * @returns false if the write failed
     */
    virtual bool Write(const char *data, size_t count) = 0;

    /**
     * @returns true if an error had occured
     */
    virtual bool Failed() const = 0;
};

/**
 * Video encoder interface
 * Encoded data will be output to the video stream.
 * The resolution of the video stream is set to the resolution of the encoded
 * Cuda memory.
 */
class IVideoEncoder : public NonCopyable
{
public:
    virtual ~IVideoEncoder() = default;

    /**
     * Supported capabilities
     */
    enum class Capability
    {
        /// Check if encoder is supported
        IS_SUPPORTED,
        /// Minimum supported output width
        MIN_WIDTH,
        /// Minimum supported output height
        MIN_HEIGHT,
        /// Maximum supported output width
        MAX_WIDTH,
        /// Maximum supported output height
        MAX_HEIGHT
    };

    /**
     * Query a capability
     *
     * @param capability [in] capability to query
     *
     * @returns query result
     */
    virtual int32_t Query(Capability capability) = 0;

    /**
     * Supported pixel formats
     */
    enum class Format
    {
        /**
         * 8 bit Packed A8R8G8B8. This is a word-ordered format
         * where a pixel is represented by a 32-bit word with B
         * in the lowest 8 bits, G in the next 8 bits, R in the
         * 8 bits after that and A in the highest 8 bits. */
        ARGB,
        /**
         * 8 bit Packed A8B8G8R8. This is a word-ordered format
         * where a pixel is represented by a 32-bit word with R
         * in the lowest 8 bits, G in the next 8 bits, B in the
         * 8 bits after that and A in the highest 8 bits. */
        ABGR
    };

    /**
     * Set the output stream.
     *
     * @param stream [in] output stream for video data
     */
    virtual void SetStream(const std::shared_ptr<IVideoStream> &stream) = 0;

    /**
     * Set the target frame rate of the video
     *
     * @param frame_rate [in] new target frame rate
     */
    virtual void SetFrameRate(float frame_rate) = 0;

    /**
     * Set the bit rate of the video
     *
     * @param bit_rate [in] new bit rate
     */
    virtual void SetBitRate(uint32_t bit_rate) = 0;

    /**
     * @return the cuda context used for encoding
     */
    virtual const std::shared_ptr<CudaContext> &GetCudaContext() = 0;

    /**
     * Encode an image from a memory blob.
     *
     * @param width [in] width
     * @param height [in] height
     * @param memory [in] memory blob
     * @param format [in] format of the data
     */
    virtual void Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory, Format format) = 0;
};

/**
 * Operator that appends string representation of a video encoder format to a stream.
 */
std::ostream &operator<<(std::ostream &os, const IVideoEncoder::Format &format);

} // namespace clara::viz
