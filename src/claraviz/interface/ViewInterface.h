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
#include <string>
#include <list>

#include <claraviz/interface/InterfaceData.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

/// View mode
enum class ViewMode
{
    /// 3D Cinematic render view
    CINEMATIC,
    /// 3D Slice view
    SLICE,
    /// 3D Slice with segmenation view
    SLICE_SEGMENTATION,
    /// 2D n-dimensional data view
    TWOD
};

/// Stereo mode
enum class StereoMode
{
    /// No stereo rendering
    OFF,
    /// Render left eye
    LEFT,
    /// Render right eye
    RIGHT,
    /// Render left eye in top half and right eye in bottom half
    TOP_BOTTOM
};

/**
 * Operator that appends the string representation of a ViewMode to a stream.
 */
std::ostream &operator<<(std::ostream &os, const ViewMode &view_mode);

/**
 * Operator that appends the string representation of a StereoMode to a stream.
 */
std::ostream &operator<<(std::ostream &os, const StereoMode &stereo_mode);

/**
 * View interface data definition
 */
template<template<typename> typename V>
struct ViewInterfaceData
{
    /**
     * Construct
     */
    ViewInterfaceData();

    struct View
    {
        /**
         * Construct
         */
        View();

        /**
         * View name.
         *
         * When using multiple views then give each of them a unique name.
         * If no name is given this references the default view.
         *
         * Default: ""
         */
        std::string name;

        /**
         * Name of the stream to render to, optional. If not specified render to the default stream.
         *
         * Default: ""
         */
        std::string stream_name;

        /**
         * View mode
         *
         * Default: ViewMode::CINEMATIC
         */
        ViewMode mode;

        /**
         * Name of the camera to use for 3D views
         *
         * Default: ""
         */
        std::string camera_name;

        /**
         * Name of the data view to use for the 2D data view mode
         *
         * Default: ""
         */
        std::string data_view_name;

        /**
         * Stereo mode (supported by 3D Cinematic renderer only)
         *
         * Default: StereoMode::OFF
         */
        StereoMode stereo_mode;
    };

    /**
     * Views
     */
    std::list<View> views;

    /**
     * Get the view with the given name, add it if it does not exist already
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    View *GetOrAddView(const std::string &name);

    /**
     * Get the view with the given name
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    View *GetView(const std::string &name = std::string());

    /**
     * Get the view with the given name
     *
     * @param name [in]
     */
    const View *GetView(const std::string &name = std::string()) const;
};

namespace detail
{

using ViewInterfaceDataIn = ViewInterfaceData<InterfaceValueT>;

using ViewInterfaceDataOut = ViewInterfaceData<InterfaceDirectT>;

struct ViewInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::ViewInterface ViewInterface.h
 * View interface, see @ref ViewInterfaceData for the interface properties.
 */
using ViewInterface =
    InterfaceData<detail::ViewInterfaceDataIn, detail::ViewInterfaceDataOut, detail::ViewInterfaceDataPrivate>;

} // namespace clara::viz
