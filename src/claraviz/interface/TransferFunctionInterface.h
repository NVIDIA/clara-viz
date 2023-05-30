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

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <claraviz/interface/InterfaceData.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

/// Shading profile enum
enum class TransferFunctionShadingProfile
{
    /// BRDF only
    BRDF_ONLY,
    /// PHASE only
    PHASE_ONLY,
    /// Hybrid of the above
    HYBRID
};

/// Blending profile enum
enum class TransferFunctionBlendingProfile
{
    /// The component with the maximum opacity is used
    MAXIMUM_OPACITY,
    /// Components are linearily combined by their opacity
    BLENDED_OPACITY
};

/// Opacity profile enum
enum class TransferFunctionOpacityProfile
{
    /// identity, opacity unchanged
    SQUARE,
    /// opacity output linearly increases until half range then linearly decreases
    TRIANGLE,
    /**
     * opacity output increase follows sine until `opacity_transition * range`, stays at
     * opacity input until `1 - opacity_transition * range' then decrease follows sine
     */
    SINE,
    /**
     * opacity output increase linearly until `opacity_transition * range`, stays at
     * opacity input until `1 - opacity_transition * range' then decrease linearly
     */
    TRAPEZIOD
};

/**
 * Transfer function interface data definition
 */
template<template<typename> typename V>
struct TransferFunctionInterfaceData
{
    TransferFunctionInterfaceData();

    /**
     * Shading profile
     *
     * Default: TransferFunctionShadingProfile::HYBRID
     */
    TransferFunctionShadingProfile shading_profile;

    /**
     * Blending profile, defines how components are blended together
     *
     * Default: TransferFunctionBlendingProfile::MAXIMUM_OPACITY
     */
    TransferFunctionBlendingProfile blending_profile;

    /**
     * Global opacity scale factor
     *
     * Default: 1.0
     *
     * Range: ]0.0, FLOAT_MAX]
     */
    V<float> global_opacity;

    /**
     * Global density scale factor
     *
     * Default: 1.0
     *
     * Range: ]0.0, FLOAT_MAX]
     */
    V<float> density_scale;

    /**
     * Global gradient scale factor
     *
     * Default: 1.0
     *
     * Range: ]0.0, FLOAT_MAX]
     */
    V<float> gradient_scale;

    /**
     * Segmentation regions which should be hidden
     *
     * Default: empty
     */
    std::vector<uint32_t> hidden_regions;

    /// Transfer function component
    struct Component
    {
        Component();

        /**
         * Density range this component is defined for
         *
         * Default: (0.0, 1.0)
         *
         * Range: ([0.0, range.max[, ]range.min, 1.0]
         */
        V<Vector2f> range;

        /**
         * Segmentation regions the component is active in
         *
         * Default: empty
         */
        std::vector<uint32_t> active_regions;

        /**
         * Opacity profile
         *
         * Default: SQUARE
         */
        TransferFunctionOpacityProfile opacity_profile;
        /**
         * Opacity transition value
         *
         * Default: 0.2
         *
         * Range: [0.0, 1.0]
         */
        V<float> opacity_transition;

        /**
         * Opacity input, output determined by opacity profile above
         *
         * Default: 0.5
         *
         * Range: [0.0, 1.0]
         */
        V<float> opacity;
        /**
         * Roughness
         *
         * Default: 0.0
         *
         * Range: [0.0, FLOAT_MAX]
         */
        V<float> roughness;
        /**
         * Emissive strength
         *
         * Default: 0.0
         *
         * Range: [0.0, FLOAT_MAX]
         */
        V<float> emissive_strength;

        // Each color has a start and end component for the range start and end

        /**
         * Diffuse start color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> diffuse_start;
        /**
         * Diffuse end color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> diffuse_end;
        /**
         * Specular start color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> specular_start;
        /**
         * Specular end color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> specular_end;
        /**
         * Emissive start color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> emissive_start;
        /**
         * Emissive end color
         *
         * Default: (1.0, 1.0, 1.0)
         *
         * Range: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
         */
        V<Vector3f> emissive_end;
    };

    /**
     * Transfer function components
     */
    std::list<Component> components;
};

namespace detail
{

using TransferFunctionInterfaceDataIn = TransferFunctionInterfaceData<InterfaceValueT>;

using TransferFunctionInterfaceDataOut = TransferFunctionInterfaceData<InterfaceDirectT>;

struct TransferFunctionInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::TransferFunctionInterface TransferFunctionInterface.h
 * Transfer function interface, see @ref TransferFunctionInterfaceData for the interface properties.
 */
using TransferFunctionInterface =
    InterfaceData<detail::TransferFunctionInterfaceDataIn, detail::TransferFunctionInterfaceDataOut,
                  detail::TransferFunctionInterfaceDataPrivate>;

} // namespace clara::viz
