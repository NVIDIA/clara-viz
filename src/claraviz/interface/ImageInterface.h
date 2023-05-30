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

#include <type_traits>
#include <memory>
#include <string>

#include "claraviz/core/Image.h"
#include "claraviz/interface/InterfaceData.h"

namespace clara::viz
{

/**
 * Image interface data definition
 */
template<template<typename> typename V>
struct ImageInterfaceData
{
    ImageInterfaceData();

    /**
     * Name of the view to render, optional.
     * If no name is given the default view is rendered.
     *
     * Default: ""
     */
    std::string view_name;

    /**
     * Width of the image
     *
     * Default: 64
     *
     * Range: [64, INT32_MAX]
     */
    V<uint32_t> width;

    /**
     * Height of the image
     *
     * Default: 64
     *
     * Range: ]64, INT32_MAX]
     */
    V<uint32_t> height;

    /**
     * The type of color data requested
     */
    ColorImageType color_type;

    /**
     * Pre-allocated CUDA memory blob to write color data into. The allocated memory must be able to store width by
     * height elements of type 'color_type'. If this member is empty, memory will be allocated by the renderer.
     */
    std::shared_ptr<IBlob> color_memory;

    /**
     * The type of depth data requested
     */
    DepthImageType depth_type;

    /**
     * Pre-allocated CUDA memory blob to write depth data into. The allocated memory must be able to store width by
     * height elements of type 'depth_type'. If this member is empty, memory will be allocated by the renderer.
     */
    std::shared_ptr<IBlob> depth_memory;

    /**
     * Encode quality if 'color_type' is ColorImageType:JPEG
     *
     * Default: 75
     *
     * Range: [1, 100]
     */
    V<uint32_t> jpeg_quality;
};

namespace detail
{

using ImageInterfaceDataIn = ImageInterfaceData<InterfaceValueT>;

using ImageInterfaceDataOut = ImageInterfaceData<InterfaceDirectT>;

struct ImageInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::ImageInterface ImageInterface.h
 * Image interface, see @ref ImageInterfaceData for the interface properties.
 */
using ImageInterface =
    InterfaceData<detail::ImageInterfaceDataIn, detail::ImageInterfaceDataOut, detail::ImageInterfaceDataPrivate>;

/**
 * Backchannel for encoded image data
 **/
class ImageInterfaceOutput
    : public MessageReceiver
    , public MessageProvider
{
public:
    ImageInterfaceOutput() = default;
    virtual ~ImageInterfaceOutput(){};

    /**
     * Wait for the encoded data to be returned by the renderer and return it.
     *
     * @returns the encode data
     **/
    std::shared_ptr<const ImageEncodedDataMessage> WaitForEncodedData();
};

} // namespace clara::viz
