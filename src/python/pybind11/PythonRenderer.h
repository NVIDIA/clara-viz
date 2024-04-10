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

#include <claraviz/core/Image.h>
#include <claraviz/interface/DataInterface.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <nlohmann/json.hpp>

#include <stdint.h>

#include <functional>
#include <vector>
#include <string>
#include <memory>

namespace clara::viz
{

enum class ViewMode;
class ImageCapsule;
class VideoStream;

class PythonRenderer
{
public:
    /**
     * Construct
     *
     * @param cuda_device_ordinals [in] Cuda devices to render on
     */
    PythonRenderer(const std::vector<uint32_t> &cuda_device_ordinals);
    PythonRenderer() = delete;

    /**
     * Defines a data array
     */
    class Array
    {
    public:
        /**
         * Constructor
         */
        Array(pybind11::array array, const std::string &dimension_order = std::string(),
              const std::vector<uint32_t> &permute_axes = std::vector<uint32_t>(),
              const std::vector<bool> &flip_axes        = std::vector<bool>(),
              const std::vector<float> &element_size    = std::vector<float>())
            : levels_({array})
            , dimension_order_(dimension_order)
            , permute_axes_(permute_axes)
            , flip_axes_(flip_axes)
            , element_sizes_({element_size})
        {
        }

        /**
         * Constructor
         */
        Array(const std::vector<pybind11::array> &levels, const std::string &dimension_order = std::string(),
              const std::vector<uint32_t> &permute_axes            = std::vector<uint32_t>(),
              const std::vector<bool> &flip_axes                   = std::vector<bool>(),
              const std::vector<std::vector<float>> &element_sizes = std::vector<std::vector<float>>())
            : levels_(levels)
            , dimension_order_(dimension_order)
            , permute_axes_(permute_axes)
            , flip_axes_(flip_axes)
            , element_sizes_(element_sizes)
        {
        }
        Array() = default;

        /// Array of arrays with the data for each level for multi-dimensional data. An array
        /// with a single array for other (e.g. volume) data.
        std::vector<pybind11::array> levels_;
        /// A string defining the data organization and format
        std::string dimension_order_;
        /// Permutes the given data axes, e.g. to swap x and y of a 3-dimensional density array specify (0, 2, 1, 3)
        std::vector<uint32_t> permute_axes_;
        /// Flips the given axes
        std::vector<bool> flip_axes_;
        /// Physical size of an element for each level. the order is defined by the 'dimension_order' field. For
        /// elements which have no physical size like 'M' or 'T' the corresponding value is 1.0.
        /// For multi-dimensional data this is an array of element sizes, else an array with a single
        /// element.
        std::vector<std::vector<float>> element_sizes_;
    };

    /// Python tailored fetch callback function similar to the one defined in the DataInterface, this one just
    /// takes an pybind11::array for the data
    using FetchCallbackFunc =
        std::function<bool(uintptr_t context, uint32_t level_index, const std::vector<uint32_t> &offset,
                           const std::vector<uint32_t> &size, pybind11::array data)>;

    /// Python tailored fetch function similar to the one defined in the DataInterface, this one just takes
    /// the python tailored fetch callback function
    using FetchFunc = std::function<bool(uintptr_t context, const std::string &array_id, uint32_t level_index,
                                         const std::vector<uint32_t> &offset, const std::vector<uint32_t> &size,
                                         const FetchCallbackFunc &fetch_callback_func)>;

    /**
     * Set data arrays
     *
     * @param arrays [in] arrays
     * @param fetch_func [in] A function to be called on demand data fetches
     */
    void SetArrays(const std::vector<Array> &arrays, const FetchFunc &fetch_func = FetchFunc());

    /**
     * Get data arrays
     *
     * @returns arrays
     */
    std::vector<Array> GetArrays();

    /**
     * Deduce settings from configured data (data needs to be configered by SetArrays()).
     * Make the whole dataset visible. Set a light in correct distance. Set a transfer function
     * using the histogram of the data.
     *
     * @param view_mode [in] view mode
     */
    void DeduceSettings(clara::viz::ViewMode view_mode);

    /**
     * Set settings
     *
     * @param new_settings [in] json with new settings
     */
    void SetSettings(const nlohmann::json &new_settings);

    /**
     * Merge settings
     *
     * @param new_settings [in] json with settings to merge in
     */
    void MergeSettings(const nlohmann::json &new_settings);

    /**
     * Get settings
     *
     * @returns json with settings
     */
    nlohmann::json GetSettings();

    /**
     * Create a video stream.
     *
     * @param callback [in] video data callback
     *
     * @returns video stream object
     */
    std::unique_ptr<VideoStream> CreateVideoStream(const std::function<void(pybind11::object, bool)> &callback);

    /**
     * Render a image.
     *
     * @param width [in]
     * @param height [in]
     * @param image_type [in]
     */
    std::unique_ptr<ImageCapsule> RenderImage(
        uint32_t width, uint32_t height,
        clara::viz::ColorImageType image_type = clara::viz::ColorImageType::RAW_RGBA_U8);

    // Interface selector
    enum class InterfaceSelector
    {
        ALL, //< reset all interfaces

        CAMERA,
        CAMERA_APERTURE,
        DATA,
        DATA_CONFIG,
        DATA_CROP,
        DATA_TRANSFORM,
        DATA_VIEW,
        LIGHT,
        BACKGROUND_LIGHT,
        POST_PROCESS_DENOISE,
        POST_PROCESS_TONEMAP,
        RENDER_SETTINGS,
        TRANSFER_FUNCTION,
        VIEW
    };

    /**
     * Reset selected interfaces to default
     *
     * @param selectors [in] vector of interfaces to reset to defaults
     */
    void Reset(const std::vector<InterfaceSelector> &selectors);

private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl *) const;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

} // namespace clara::viz
