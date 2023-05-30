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

/**
 * Post process tonemap interface data definition
 */
template<template<typename> typename V>
struct PostProcessTonemapInterfaceData
{
    PostProcessTonemapInterfaceData();

    /**
     * Enable/Disable
     *
     * Default: false
     */
    bool enable;

    /**
     * Exposure
     *
     * Default: 0.5
     *
     * Range: [0.0, FLOAT_MAX]
     */
    V<float> exposure;
};

namespace detail
{

using PostProcessTonemapInterfaceDataIn = PostProcessTonemapInterfaceData<InterfaceValueT>;

using PostProcessTonemapInterfaceDataOut = PostProcessTonemapInterfaceData<InterfaceDirectT>;

struct PostProcessTonemapInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::PostProcessTonemapInterface PostProcessTonemapInterface.h
 * Post process tonemap interface, see @ref PostProcessTonemapInterfaceData for the interface properties.
 */
using PostProcessTonemapInterface =
    InterfaceData<detail::PostProcessTonemapInterfaceDataIn, detail::PostProcessTonemapInterfaceDataOut,
                  detail::PostProcessTonemapInterfaceDataPrivate>;

} // namespace clara::viz
