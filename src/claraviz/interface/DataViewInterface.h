/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <string>

#include <claraviz/interface/InterfaceData.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

/**
 * Data view interface data definition.
 *
 * Defines the section of the n-dimensional data to display when in 'TWOD' view mode.
 */
template<template<typename> typename V>
struct DataViewInterfaceData
{
    DataViewInterfaceData();

    /**
     * Defines a data view
     */
    struct DataView
    {
        DataView();

        /**
         * Name
         *
         * Default: ""
         */
        std::string name;

        /**
         * Zoom factor.
         *
         * Default: 1.0
         *
         * Range: [1.0, FLOAT_MAX]
         */
        V<float> zoom_factor;

        /**
         * View offset in mm. If the view offset is zero, the viewed data is centered
         * to the view.
         *
         * Default (0.0, 0.0)
         */
        Vector2f view_offset;

        /**
         * Pixel aspect ratio, describes how the width of a pixel compares to the height.
         *
         * Default: 1
         *
         * Range ]0.0, FLOAT_MAX]
         */
        V<float> pixel_aspect_ratio;
    };

    /**
     * Named data views
     */
    std::list<DataView> data_views;

    /**
     * Get the data view with the given name, add it if it does not exist already
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    DataView *GetOrAddDataView(const std::string &name);

    /**
     * Get the data view with the given name
     *
     * @param name [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    DataView *GetDataView(const std::string &name = std::string());

    /**
     * Get the data view with the given name (const)
     *
     * @param name [in]
     */
    const DataView *GetDataView(const std::string &name = std::string()) const;
};

namespace detail
{

using DataViewInterfaceDataIn = DataViewInterfaceData<InterfaceValueT>;

using DataViewInterfaceDataOut = DataViewInterfaceData<InterfaceDirectT>;

struct DataViewInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::DataViewInterface DataViewInterface.h
 * Data view interface, see @ref DataViewInterfaceData for the interface properties.
 */
using DataViewInterface = InterfaceData<detail::DataViewInterfaceDataIn, detail::DataViewInterfaceDataOut,
                                        detail::DataViewInterfaceDataPrivate>;

} // namespace clara::viz
