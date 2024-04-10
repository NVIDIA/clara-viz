/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>

#include "claraviz/util/Message.h"

namespace clara::viz
{

class IBlob;

/// color image type enum
enum class ColorImageType
{
    /**
     * unknown (invalid)
     */
    UNKNOWN,
    /**
     * RAW uncompressed RGBA unsigned int 8-bit
     * This is a word-ordered format where a pixel is represented by a 32-bit word with R
     * in the lowest 8 bits, G in the next 8 bits, B in the
     * 8 bits after that and A in the highest 8 bits.
     */
    RAW_RGBA_U8,
    /**
     * JPEG
     */
    JPEG,
};

/// depth image type enum
enum class DepthImageType
{
    /**
     * unknown (invalid)
     */
    UNKNOWN,
    /**
     * RAW uncompressed depth 32-bit float.
     */
    RAW_DEPTH_F32,
};

/**
 * Image output server.
 * Communicates to the server through messages, once an image is requested
 * a `ImageRequestMessage` is emitted. To deliver an image enqueue a `ImageEncodeMessage`.
 */
class Image
    : public std::enable_shared_from_this<Image>
    , public MessageProvider
    , public MessageReceiver
{
public:
    /**
     * Create
     *
     * @param image_interface_output [in] The message receiver for the encoded data
     * @param cuda_device_ordinal [in] Cuda device to use for encoding
     *
     * @returns created instance
     */
    static std::shared_ptr<Image> Create(const std::shared_ptr<MessageReceiver> &image_interface_output,
                                         uint32_t cuda_device_ordinal);
    virtual ~Image();

    /**
     * Start the server
     */
    void Run();

    /**
     * Shutdown
     */
    void Shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// parameters (depends on type)
union EncodeParameters
{
    /// Jpeg parameters
    struct
    {
        // quality
        uint32_t quality_;
    } jpeg_;
};

/**
 * Emitted by the Image class indicating that a new image is requested to be delivered.
 */
class ImageMessage : public Message
{
public:
    ImageMessage()
        : Message(id_)
    {
    }

    /// name of the view to render
    std::string view_name_;
    /// width of the image
    uint32_t width_;
    /// height of the image
    uint32_t height_;

    /// type of the color image
    ColorImageType color_type_;
    /// Pre-allocated CUDA memory blob to write color data into. The allocated memory must be able to store width by
    /// height elements of type 'color_type_'. If this member is empty, memory will be allocated by the renderer.
    std::shared_ptr<IBlob> color_memory_;

    /// type of the color image
    DepthImageType depth_type_;
    /// Pre-allocated CUDA memory blob to write depth data into. The allocated memory must be able to store width by
    /// height elements of type 'depth_type_'. If this member is empty, memory will be allocated by the renderer.
    std::shared_ptr<IBlob> depth_memory_;

    /// parameters (depends on type)
    EncodeParameters parameters_;

    /// message id
    static const MessageID id_;
};

/**
 * Encode a resource.
 * Send by the client to the Image service.
 */
class ImageEncodeMessage : public Message
{
public:
    ImageEncodeMessage()
        : Message(id_)
    {
    }

    /// width in pixels of the resources
    uint32_t width_;
    /// height in pixels of the resources
    uint32_t height_;

    /**
     * Supported color pixel formats
     */
    enum class ColorFormat
    {
        /**
         * 8 bit Packed A8B8G8R8. This is a word-ordered format
         * where a pixel is represented by a 32-bit word with R
         * in the lowest 8 bits, G in the next 8 bits, B in the
         * 8 bits after that and A in the highest 8 bits. */
        ABGR_U8
    };

    /// the color resource to encode
    std::shared_ptr<IBlob> color_memory_;
    /// pixel format of the color resource
    ColorFormat color_format_;
    /// type of the encoded color image
    ColorImageType color_type_;

    /**
     * Supported depth pixel formats
     */
    enum class DepthFormat
    {
        /**
         * 32-bit float depth.
         */
        D_F32,
    };

    /// the depth resource to encode
    std::shared_ptr<IBlob> depth_memory_;
    /// pixel format of the depth resource
    DepthFormat depth_format_;
    /// type of the encoded depth image
    DepthImageType depth_type_;

    /// parameters (depends on type)
    EncodeParameters parameters_;

    /// message id
    static const MessageID id_;
};

/**
 * Image rendering failed message.
 * Send by the client to the Image service.
 */
class ImageRenderFailedMessage : public Message
{
public:
    ImageRenderFailedMessage()
        : Message(id_)
    {
    }

    /// The reason why the rendering failed
    std::string reason_;

    /// message id
    static const MessageID id_;
};

/**
 * Message containing the image data
 * Emitted by the Image service.
 */
class ImageEncodedDataMessage : public Message
{
public:
    ImageEncodedDataMessage()
        : Message(id_)
    {
    }

    /// type of the color data
    ColorImageType color_type_;
    /// the color memory
    std::shared_ptr<IBlob> color_memory_;

    /// type of the depth data
    DepthImageType depth_type_;
    /// the depth memory
    std::shared_ptr<IBlob> depth_memory_;

    /// message id
    static const MessageID id_;
};

} // namespace clara::viz
