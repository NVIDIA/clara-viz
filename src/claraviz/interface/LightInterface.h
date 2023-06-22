/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
#include <array>

#include <claraviz/interface/InterfaceData.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

/// maximum number of supported lights
constexpr uint32_t LIGHTS_MAX = 4u;

/**
 * Light interface data definition.
 *
 * Configures a light. Lights are rectangular area lights located with its center at the given position.
 * The rectangle is facing the given direction, light rays emitted perpendicular from the plane.
 */
template<template<typename> typename V>
struct LightInterfaceData
{
    LightInterfaceData();

    /**
     * Defines a light
     */
    struct Light
    {
        Light();

        /**
         * Light position
         *
         * Default: (0.0, 0.0, -1.0)
         */
        Vector3f position;

        /**
         * Light direction, has to be a unit vector
         *
         * Default: (0.0, 0.0, 1.0)
         */
        V<Vector3f> direction;

        /**
         * Size
         *
         * Default: 1.0
         *
         * Range: ]0.0, FLOAT_MAX]
         */
        V<float> size;

        /**
         * Intensity
         *
         * Default: 1.0
         *
         * Range: ]0.0, FLOAT_MAX]
         */
        V<float> intensity;

        /**
         * Color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> color;

        /**
         * Enable/Disable
         *
         * Default: false
         */
        bool enable;

        /**
         * Show the light
         *
         * Default: true
         */
        bool show;
    };

    /**
     * Named cameras
     */
    std::array<Light, LIGHTS_MAX> lights;
};

namespace detail
{

using LightInterfaceDataIn = LightInterfaceData<InterfaceValueT>;

using LightInterfaceDataOut = LightInterfaceData<InterfaceDirectT>;

struct LightInterfaceDataPrivate
{
};

} // namespace detail

/**
 * Background light interface data definition.
 *
 * Configures the background light. The background light is a sphere around the origin emitting light from its surface.
 */
template<template<typename> typename V>
struct BackgroundLightInterfaceData
{
    BackgroundLightInterfaceData();

    /**
     * Intensity
     *
     * Default: 1.0
     *
     * Range: ]0.0, FLOAT_MAX]
     */
    V<float> intensity;

    /**
     * Top color (+y direction)
     *
     * Default: (1.0, 1.0, 1.0)
     *
     * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
     */
    V<Vector3f> top_color;

    /**
     * Horizon color (x-z plane)
     *
     * Default: (1.0, 1.0, 1.0)
     *
     * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
     */
    V<Vector3f> horizon_color;

    /**
     * Bottom color (-y direction)
     *
     * Default: (1.0, 1.0, 1.0)
     *
     * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
     */
    V<Vector3f> bottom_color;

    /**
     * Enable/Disable
     *
     * Default: false
     */
    bool enable;

    /**
     * Show the background
     *
     * Default: true
     */
    bool show;

    /**
     * The background light should cast light
     *
     * Default: true
     */
    bool cast_light;
};

namespace detail
{

using BackgroundLightInterfaceDataIn = BackgroundLightInterfaceData<InterfaceValueT>;

using BackgroundLightInterfaceDataOut = BackgroundLightInterfaceData<InterfaceDirectT>;

struct BackgroundLightInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::LightInterface LightInterface.h
 * Light interface, see @ref LightInterfaceData for the interface properties.
 */
using LightInterface =
    InterfaceData<detail::LightInterfaceDataIn, detail::LightInterfaceDataOut, detail::LightInterfaceDataPrivate>;

/**
 * @class clara::viz::BackgroundLightInterface LightInterface.h
 * Background light interface, see @ref BackgroundLightInterfaceData for the interface properties.
 */
using BackgroundLightInterface =
    InterfaceData<detail::BackgroundLightInterfaceDataIn, detail::BackgroundLightInterfaceDataOut,
                  detail::BackgroundLightInterfaceDataPrivate>;

} // namespace clara::viz
