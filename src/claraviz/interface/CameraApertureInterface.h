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
 * Camera aperture interface data definition
 */
template<template<typename> typename V>
struct CameraApertureInterfaceData
{
    CameraApertureInterfaceData();

    /**
     * Enable/Disable
     *
     * Default: false
     */
    bool enable;

    /**
     * Aperture
     *
     * Default: 0.1
     *
     * Range: ]0.0, FLOAT_MAX]
     */
    V<float> aperture;

    /**
     * Enable/Disable auto focus
     *
     * Default: false
     */
    bool auto_focus;

    /**
     * Focus distance
     *
     * Default: 1.0
     *
     * Range: ]0.0, FLOAT_MAX]
     */
    V<float> focus_distance;
};

namespace detail
{

using CameraApertureInterfaceDataIn = CameraApertureInterfaceData<InterfaceValueT>;

using CameraApertureInterfaceDataOut = CameraApertureInterfaceData<InterfaceDirectT>;

struct CameraApertureInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::CameraApertureInterface CameraApertureInterface.h
 * Camera aperture interface, see @ref CameraApertureInterfaceData for the interface properties.
 */
using CameraApertureInterface =
    InterfaceData<detail::CameraApertureInterfaceDataIn, detail::CameraApertureInterfaceDataOut,
                  detail::CameraApertureInterfaceDataPrivate>;

} // namespace clara::viz
