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

#include <claraviz/interface/InterfaceData.h>

namespace clara::viz
{

/// Denoise method
enum class DenoiseMethod
{
    /// off
    OFF,
    /// KNN (K Nearest Neighbors) filter including depth
    KNN,
    /// Optix AI-Accelerated denoiser
    AI
};

/**
 * Post process tonemap interface data definition
 */
template<template<typename> typename V>
struct PostProcessDenoiseInterfaceData
{
    PostProcessDenoiseInterfaceData();

    /**
     * Denoise method
     *
     * Default: DenoiseMethod::OFF
     */
    DenoiseMethod method;

    /**
     * Filter radius in pixels (KNN filter only)
     *
     * Default: 3
     *
     * Range: [1, INT32_MAX]
     */
    V<uint32_t> radius;

    /**
     *  Spatial weight (KNN filter only)
     *
     * Default: 0.05
     *
     * Range: [0.0, FLOAT_MAX]
     */
    V<float> spatial_weight;

    /**
     * Depth weight (KNN filter only)
     *
     * Default: 3.0
     *
     * Range: [0.0, FLOAT_MAX]
     */
    V<float> depth_weight;

    /**
     * Noise threshold (KNN filter only)
     *
     * Default: 0.2
     *
     * Range: [0.0, 1.0]
     */
    V<float> noise_threshold;

    /**
     * Enable/Disable iteration limit
     *
     * Default: false
     */
    bool enable_iteration_limit;

    /**
     * Apply denoise for iterations below that limit only
     *
     * Default: 100
     *
     * Range: [1, UINT32_MAX]
     */
    V<uint32_t> iteration_limit;
};

namespace detail
{

using PostProcessDenoiseInterfaceDataIn = PostProcessDenoiseInterfaceData<InterfaceValueT>;

using PostProcessDenoiseInterfaceDataOut = PostProcessDenoiseInterfaceData<InterfaceDirectT>;

struct PostProcessDenoiseInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::PostProcessDenoiseInterface PostProcessDenoiseInterface.h
 * Post process denoise interface, see @ref PostProcessDenoiseInterfaceData for the interface properties.
 */
using PostProcessDenoiseInterface =
    InterfaceData<detail::PostProcessDenoiseInterfaceDataIn, detail::PostProcessDenoiseInterfaceDataOut,
                  detail::PostProcessDenoiseInterfaceDataPrivate>;

} // namespace clara::viz
