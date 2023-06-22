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

#include <list>
#include <string>

#include "claraviz/interface/InterfaceData.h"
#include "claraviz/util/MatrixT.h"
#include "claraviz/util/VectorT.h"

namespace clara::viz
{

/**
 * Camera interface data definition.
 *
 * Defines the 3D viewing transformation.
 * The the head pose is set by the 'eye', 'look_at' and 'up' parameters.
 * The ViewInterfaceData::stereo_mode setting determines the camera setup.
 * If stereo mode is 'OFF' then the image is rendered from the head pose and field of view is set using 'field_of_view'.
 * If stereo mode is 'LEFT' or 'RIGHT' then the the left or right eye image is rendered as set by 'left_eye_pose'
 * or 'right_eye_pose'.
 * If stereo mode is 'TOP_BOTTOM' then he left eye is rendered to the top half of the image and the right eye to the
 * bottom half of the image.
 * If stereo mode is 'LEFT', 'RIGHT' or TOP_BOTTOM', the field of view is set using 'left_tangent_x', 'left_tangent_y',
 * 'right_tangent_x' and 'right_tangent_y'. The 'field_of_view` parameter is ignored.
 * Also when stereo rendering is enabled, `left_gaze_direction` and `right_gaze_direction` are used the determine
 * the position of the high resolution area when using warped rendering. @sa RenderSettingsInterfaceData::enable_warp.
 */
template<template<typename> typename V>
struct CameraInterfaceData
{
    CameraInterfaceData();

    /**
     * Defines a camera
     */
    struct Camera
    {
        Camera();

        /**
         * Name
         *
         * Default: ""
         */
        std::string name;

        /**
         * Position of the eye point, has to be different from 'look_at'
         *
         * Default: (0.0, 0.0, -1.0)
         */
        V<Vector3f> eye;

        /**
         * Position of the reference point, has to be different from 'eye'
         *
         * Default: (0.0, 0.0, 0.0)
         */
        V<Vector3f> look_at;

        /**
         * Direction of the up vector, has to be a unit vector
         *
         * Default: (0.0, 1.0, 0.0)
         */
        V<Vector3f> up;

        /**
         * Field of view angle in degrees, in x direction
         *
         * Default: 30.0
         *
         * Range: ]0.0, 180.0[
         */
        V<float> field_of_view;

        /**
         * Pixel aspect ratio, describes how the width of a pixel compares to the height
         *
         * Default: 1
         *
         * Range ]0.0, FLOAT_MAX]
         */
        V<float> pixel_aspect_ratio;

        /**
         * Enable stereo rendering.
         *
         * Default: false
         */
        bool enable_stereo;

        /**
         * Left eye pose for stereo rendering.
         *
         * Default: identity matrix
         */
        Matrix4x4 left_eye_pose;

        /**
         * Right eye pose for stereo rendering.
         *
         * Default: identity matrix
         */
        Matrix4x4 right_eye_pose;

        /**
         * Left eye gaze direction for stereo rendering, has to be a unit vector.
         * Determines the position of the high resolution area when using warped rendering.
         * @sa RenderSettingsInterfaceData::enable_warp.
         *
         * Default: (0.0, 0.0, 1.0)
         */
        V<Vector3f> left_gaze_direction;

        /**
         * Right eye gaze direction for stereo rendering, has to be a unit vector.
         * Determines the position of the high resolution area when using warped rendering.
         * @sa RenderSettingsInterfaceData::enable_warp.
         *
         * Default: (0.0, 0.0, 1.0)
         */
        V<Vector3f> right_gaze_direction;

        /**
         * Left eye view tangent in x direction for stereo rendering.
         *
         * Default: (-1.0, 1.0)
         */
        Vector2f left_tangent_x;

        /**
         * Left eye view tangent in y direction for stereo rendering.
         *
         * Default: (1.0, -1.0)
         */
        Vector2f left_tangent_y;

        /**
         * Right eye view tangent in x direction for stereo rendering
         *
         * Default: (-1.0, 1.0)
         */
        Vector2f right_tangent_x;

        /**
         * Right eye view tangent in y direction for stereo rendering
         *
         * Default: (1.0, -1.0)
         */
        Vector2f right_tangent_y;

        /**
         * Distance to the near and far frustum planes.
         *
         * Default: (0, FLT_MAX)
         * Range: ([-FLT_MAX, depth_clip.max[, ]depth_clip.min, FLT_MAX]
         */
        V<Vector2f> depth_clip;

        /**
         * Mapping from near and far depth clipping plane to output depth values.
         *
         * Default: (0, FLT_MAX)
         */
        V<Vector2f> depth_range;
    };

    /**
     * Named cameras
     */
    std::list<Camera> cameras;

    /**
     * Get the camera with the given name, add it if it does not exist already
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    Camera *GetOrAddCamera(const std::string &name);

    /**
     * Get the camera with the given name
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    Camera *GetCamera(const std::string &name = std::string());

    /**
     * Get the camera with the given name (const)
     *
     * @param name [in]
     */
    const Camera *GetCamera(const std::string &name = std::string()) const;
};

namespace detail
{

using CameraInterfaceDataIn = CameraInterfaceData<InterfaceValueT>;

using CameraInterfaceDataOut = CameraInterfaceData<InterfaceDirectT>;

struct CameraInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::CameraInterface CameraInterface.h
 * Camera interface, see @ref CameraInterfaceData for the interface properties.
 */
using CameraInterface =
    InterfaceData<detail::CameraInterfaceDataIn, detail::CameraInterfaceDataOut, detail::CameraInterfaceDataPrivate>;

} // namespace clara::viz
