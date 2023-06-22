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
#include <string>

#include <claraviz/interface/InterfaceData.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

/// Interpolation mode
enum class InterpolationMode
{
    /// Linear interpolation
    LINEAR,
    /// B-spline interpolation
    BSPLINE,
    /// Catmull-Rom spline interpolation
    CATMULLROM
};

/**
 * Render settings interface data definition
 */
template<template<typename> typename V>
struct RenderSettingsInterfaceData
{
    RenderSettingsInterfaceData();

    /**
     * Interpolation mode
     *
     * Default: InterpolationMode::BSPLINE
     */
    InterpolationMode interpolation_mode;

    /**
     * Ray marching step size in voxels
     *
     * Default: 1.0
     *
     * Range: ]0, FLOAT_MAX]
     */
    V<float> step_size;

    /**
     * Ray marching step size in voxels when doing shadow rays
     *
     * Default: 1.0
     *
     * Range: ]0, FLOAT_MAX]
     */
    V<float> shadow_step_size;

    /**
     * Maximum iterations, the renderer stops after this limit is reached
     *
     * Default: 10000
     *
     * Range: [1, UINT32_MAX]
     */
    V<uint32_t> max_iterations;

    /**
     * Time slot for a frame in ms. If rendering single images, the renderer returns after this time.
     * If rendering a video stream the time is clamped to the frame time.
     * If the value is 0.0 the time slot is ignored.
     *
     * Default: 0.0
     *
     * Range: [0.0, FLOAT_MAX]
     */
    V<float> time_slot;

    /**
     * Enable warped rendering.
     * If enabled the image is rendered in high resolution in the center. The resolution is reduced linearly to
     * the edges.
     * @sa warp_resolution_scale, warp_full_resolution_diameter
     *
     * Default: false
     */
    bool enable_warp;

    /**
     * If warped rendering is enabled this sets the ratio between the warp resolution and the output resolution.
     * @sa enable_warp
     *
     * Default: 1.0
     *
     * Range: ]0.0, 1.0]
     */
    V<float> warp_resolution_scale;

    /**
     * If warped rendering is enable this set size of the center full resolution area.
     * @sa enable_warp
     *
     * Default: 1.0
     *
     * Range: ]0.0, 1.0]
     */
    V<float> warp_full_resolution_size;

    /**
     * Enable to render more samples the central foveated area than in the border areas. Else all pixels are sampled equally.
     *
     * Default: false
     */
    bool enable_foveation;

    /**
     * If enabled the color and depth information of previous frames is stored. That information is reprojected to the
     * current frame thus improving quality and reducing noise.
     *
     * Default: false
     */
    bool enable_reproject;

    /**
     * Enable to store image depth information to a separate high precision depth buffer. Else depth is stored in the
     * color image alpha channel.
     *
     * Default: false
     */
    bool enable_separate_depth;
};

namespace detail
{

using RenderSettingsInterfaceDataIn = RenderSettingsInterfaceData<InterfaceValueT>;

using RenderSettingsInterfaceDataOut = RenderSettingsInterfaceData<InterfaceDirectT>;

struct RenderSettingsInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::RenderSettingsInterface RenderSettingsInterface.h
 * Render settings interface, see @ref RenderSettingsInterfaceData for the interface properties.
 */
using RenderSettingsInterface =
    InterfaceData<detail::RenderSettingsInterfaceDataIn, detail::RenderSettingsInterfaceDataOut,
                  detail::RenderSettingsInterfaceDataPrivate>;

} // namespace clara::viz
