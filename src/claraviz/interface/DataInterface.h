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
#include <memory>
#include <string>
#include <vector>

#include <claraviz/interface/InterfaceData.h>
#include "claraviz/util/MatrixT.h"
#include <claraviz/util/VectorT.h>

/// forward declaration of SharedMemory classes
namespace nvidia
{

namespace sharedmemory
{

class Driver;
class Context;
class Allocation;
class AllocationLock;
class AllocationMapping;

} // namespace sharedmemory
} // namespace nvidia

namespace clara::viz
{

/// forward declaration
class IBlob;

/// data element type enum
enum class DataElementType
{
    UNKNOWN,    /// unknown (invalid)
    INT8,       /// 8 bit signed
    UINT8,      /// 8 bit unsigned
    INT16,      /// 16 bit signed
    UINT16,     /// 16 bit unsigned
    INT32,      /// 32 bit signed
    UINT32,     /// 32 bit unsigned
    HALF_FLOAT, /// 16 bit floating point
    FLOAT,      /// 32 bit floating point
};

/**
 * Operator that appends the string representation of a DataElementType to a stream.
 */
std::ostream &operator<<(std::ostream &os, const DataElementType &data_element_type);

/**
 * Get the size of an element type in bytes
 *
 * @param type [in] element type
 *
 * @return size in bytes
 */
size_t BytesPerElement(DataElementType type);

namespace detail
{

template<template<typename> typename V>
struct DataConfigInterfaceLevel
{
    DataConfigInterfaceLevel();

    /// Number of elements in each dimension, the order is defined by 'dimension_order' field.
    /// For example a 2D RGB color image with a width of 1024 and a height of 768 pixels has a size
    /// of (3, 1024, 768). A 30 frame time sequence of a density volume with with 128, height 256
    /// and depth 64 has a size of (1, 128, 256, 64, 30).
    V<std::vector<uint32_t>> size;

    /// Physical size of each element, the order is defined by the 'dimension_order' field. For
    /// elements which have no physical size like 'M' or 'T' the corresponding value is 1.0.
    V<std::vector<float>> element_size;

    /// Optional range of the values contained in the level, if this is not set then the range is calculated form the data.
    /// One min/max pair for each element component, e.g. for RGB data where 'size[0]' is '3', 'element_range' contains
    /// threa 'Vector2f' values.
    V<std::vector<Vector2f>> element_range;
};

template<template<typename> typename V>
struct DataConfigInterfaceArray
{
    DataConfigInterfaceArray();

    /// Unique identifier
    std::string id;

    /// A string defining the data organization and format.
    /// Each character defines a dimension starting with fastest varying axis
    /// and ending with the slowest varying axis. For example a 2D color image
    /// is defined as 'CXY', a time sequence of density volumes is defined as
    /// 'DCYZ'.
    /// Each character can occur only once. Either one of the data element definition
    /// characters 'C' or 'D' and the 'X' axis definition has to be present.
    /// - 'X': width
    /// - 'Y': height
    /// - 'Z': depth
    /// - 'T': time
    /// - 'I': sequence
    /// - 'C': RGB(A) color
    /// - 'D': density
    /// - 'M': mask
    V<std::string> dimension_order;

    /// Defines the element type of the data.
    V<DataElementType> element_type;

    /// Permutes the given data axes, e.g. to swap x and y of a 3-dimensional
    /// density array specify (0, 2, 1, 3)
    V<std::vector<uint32_t>> permute_axis;

    /// Flips the given axes
    V<std::vector<bool>> flip_axes;

    using Level = DataConfigInterfaceLevel<V>;

    /// Array levels. Most arrays have only one level, for multi-resolution arrays
    /// there can be multiple levels defining down-scaled representations
    /// of the data.
    std::list<Level> levels;
};

template<template<typename> typename V>
struct DataConfigInterfaceData
{
    DataConfigInterfaceData();

    using Array = DataConfigInterfaceArray<V>;

    /// Array of data arrays
    std::list<Array> arrays;

    /**
     * Get the data array with the given id, add it if it does not exist already
     *
     * @param id [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    Array *GetOrAddArray(const std::string &id);

    /**
     * Get the data array with the given id
     *
     * @param id [in]
     */
    template<typename U = V<int>, class = typename std::enable_if<std::is_same<U, InterfaceValueT<int>>::value>::type>
    Array *GetArray(const std::string &id = std::string());

    /**
     * Get the data array with the given id (const)
     *
     * @param id [in]
     */
    const Array *GetArray(const std::string &id = std::string()) const;

    /**
     * This function has to be called when fetched data had been received. For on fetch call the fetch
     * callback can be called multiple times (e.g. when loading parts of the data in multiple threads).
     * When all data is fetched the fetch callback function has to be called with a nullptr data
     * pointer to indicate that fetching of all data is done.
     *
     * @param context [in] internal context of the fetch call
     * @param level_index [in] index of the level the data had been fetched for
     * @param offset [in] offset into the level the data had been fetched for
     * @param size [in] size of the data which had been fetched
     * @param data [in] fetched data, this is nullptr for the final call of the sequence
     * @param data_size [in] size of the fetched data
     *
     * @returns false if fetching should be stopped
     */
    using FetchCallbackFunc =
        std::function<bool(uintptr_t context, uint32_t level_index, const std::vector<uint32_t> &offset,
                           const std::vector<uint32_t> &size, const void *data, size_t data_size)>;

    /**
     * This function is called to trigger on demand data fetches
     *
     * @param context [in] internal context of the fetch call, pass to fetch callback function
     * @param array_id [in] id of the array to fetch data from
     * @param level_index [in] index of the level to fetch data for
     * @param offset [in] offset into the level to fetch data for
     * @param size [in] size of the data to fetch
     * @param fetch_callback_func [in] callback function, called when data is received
     *
     * @returns false if fetching failed
     */
    using FetchFunc = std::function<bool(uintptr_t context, const std::string &array_id, uint32_t level_index,
                                         const std::vector<uint32_t> &offset, const std::vector<uint32_t> &size,
                                         const FetchCallbackFunc &fetch_callback_func)>;

    /// This function is called to fetch data on demand for 2D multi-level images
    FetchFunc fetch_func;

    /// SharedMemory driver UUID
    V<std::string> sharedmemory_driver_id;

    /**
     * If set then performance is optimized for the streaming data use case, which means that data is
     * constantly updated. This might result in higher memory usage.
     * Else it's expected that data is only specified once or rarely.
     */
    bool streaming;
};

using DataConfigInterfaceDataIn = DataConfigInterfaceData<InterfaceValueT>;

using DataConfigInterfaceDataOut = DataConfigInterfaceData<InterfaceDirectT>;

struct DataConfigInterfaceDataPrivate
{
};

} // namespace detail

using DataConfigInterface = InterfaceData<detail::DataConfigInterfaceDataIn, detail::DataConfigInterfaceDataOut,
                                          detail::DataConfigInterfaceDataPrivate>;

namespace detail
{

template<template<typename> typename V>
struct DataCropInterfaceData
{
    DataCropInterfaceData();

    /// Array of limits for each dimension. Dimension order as defined by the DataConfig interface.
    V<std::vector<Vector2f>> limits;
};

using DataCropInterfaceDataIn = DataCropInterfaceData<InterfaceValueT>;

using DataCropInterfaceDataOut = DataCropInterfaceData<InterfaceDirectT>;

struct DataCropInterfaceDataPrivate
{
};

} // namespace detail

using DataCropInterface = InterfaceData<detail::DataCropInterfaceDataIn, detail::DataCropInterfaceDataOut,
                                        detail::DataCropInterfaceDataPrivate>;
class DataInterface;

namespace detail
{

struct DataInterfaceDataPrivate;

template<template<typename> typename V>
class DataInterfaceDataBase
{
};

/**
 * DataInterfaceDataBase specialization for input data of the Data interface.
 * The data can either be provided as a 'Blob' or a shared memory allocation (by the ID).
 */
template<>
class DataInterfaceDataBase<InterfaceValueT>
{
public:
    DataInterfaceDataBase();
    virtual ~DataInterfaceDataBase() = default;

    /// Element data, 'size * elements_size' bytes
    std::shared_ptr<IBlob> blob;

    /// SharedMemory allocation UUID
    InterfaceValueT<std::string> sharedmemory_allocation_id;

    /// Internal function. Don't use.
    void SetGetDataConfig(const std::function<const DataConfigInterfaceDataOut &()> &get_data_config);
    std::function<const DataConfigInterfaceDataOut &()> get_data_config_;
};

/**
 * Common input/output data of the Data interface.
 */
template<template<typename> typename V>
class DataInterfaceData : public DataInterfaceDataBase<V>
{
public:
    /**
     * Constructor
     **/
    DataInterfaceData();

    /// Unique identifier of the array
    V<std::string> array_id;

    /// Level of the data array to store the elements to
    V<uint32_t> level;

    /**
     * Offset in the data array to store the elements to. If the data array has more dimensions than
     * offset values specified, the missing values are assumed to be zero.
     */
    V<std::vector<uint32_t>> offset;

    /**
     * Number of elements to store, required. If the data array has more dimensions than
     * size values specified, the missing values are assumed to be zero.
     **/
    V<std::vector<uint32_t>> size;
};

using DataInterfaceDataIn = DataInterfaceData<InterfaceValueT>;

using DataInterfaceDataOut = DataInterfaceData<InterfaceDirectT>;

/**
 * Private data of the data interface.
 */
class DataInterfaceDataPrivate
{
public:
    /**
     * Lock a shared memory allocation for read.
     *
     * @param sharedmemory_allocation_id [in] UUID of shared memory allocation
     *
     * @returns blob
     */
    std::shared_ptr<IBlob> LockForRead(const std::string &sharedmemory_allocation_id);

    /**
     * Reset the data to defaults.
     */
    void Reset();

    /// Data configuration received from the data config interface through messages
    DataConfigInterfaceDataOut data_config_;

private:
    /// Shared memory driver
    std::shared_ptr<nvidia::sharedmemory::Driver> sharedmemory_driver_;
    /// Shared memory context
    std::shared_ptr<nvidia::sharedmemory::Context> sharedmemory_context_;
};

using DataInterfaceBase = InterfaceData<DataInterfaceDataIn, DataInterfaceDataOut, DataInterfaceDataPrivate>;

/**
 * DataInterfaceDataBase specialization for output data of the Data interface.
 */
template<>
class DataInterfaceDataBase<InterfaceDirectT>
{
public:
    virtual ~DataInterfaceDataBase() = default;

    /// Data blob
    std::shared_ptr<IBlob> blob;
};

} // namespace detail

/**
 * Data interface. Receives message from the data config interface to validate correctness of parameters.
 */
class DataInterface
    : public MessageReceiver
    , public detail::DataInterfaceBase
{
public:
    /**
     * Construct
     */
    DataInterface();

    void Reset() final;
};

class DataHistogramInterface;

namespace detail
{

struct DataHistogramInterfaceDataIn
{
    /// Unique identifier of the array to get the histogram data from
    std::string array_id;
};

struct DataHistogramInterfaceDataOut
{
    /// Unique identifier of the array to get the histogram data from
    std::string array_id;

    std::shared_ptr<DataHistogramInterface> receiver;
};

struct DataHistogramInterfaceDataPrivate
{
};

using DataHistogramInterfaceBase =
    InterfaceData<DataHistogramInterfaceDataIn, DataHistogramInterfaceDataOut, DataHistogramInterfaceDataPrivate>;

} // namespace detail

/**
 * Data histogram interface. Requests histogram data from the renderer.
 */
class DataHistogramInterface
    : public std::enable_shared_from_this<DataHistogramInterface>
    , public MessageReceiver
    , public detail::DataHistogramInterfaceBase
{
public:
    /**
     * Construct
     */
    DataHistogramInterface();

    /**
     * Get the histogram data for an array
     *
     * @param array_id [in] id of array to get histogram from
     * @param histogram [out] histogram data
     */
    void GetHistogram(const std::string &array_id, std::vector<float> &histogram);

    /**
     * This message is send back from the renderer and contains the histogram data
     */
    class MessageHistogram : public ::clara::viz::Message
    {
    public:
        /**
         * Construct
         *
         * @param histogram [in] histogram data
         */
        MessageHistogram(const std::vector<float> &histogram)
            : ::clara::viz::Message(id_)
            , histogram_(histogram)
        {
        }

        const std::vector<float> histogram_; ///< histogram data

        static const MessageID id_; ///< message id
    };

private:
    std::mutex mutex_;
};

namespace detail
{

/**
 * Data transform interface data definition.
 *
 * Defines the transformation of the spatial dimensions ('X', 'Y', 'Z') of data.
 */
template<template<typename> typename V>
struct DataTransformInterfaceData
{
    DataTransformInterfaceData();

    /**
     * Transform matrix (row major).
     */
    Matrix4x4 matrix;
};

using DataTransformInterfaceDataIn = DataTransformInterfaceData<InterfaceValueT>;

using DataTransformInterfaceDataOut = DataTransformInterfaceData<InterfaceDirectT>;

struct DataTransformInterfaceDataPrivate
{
};

} // namespace detail

/**
 * @class clara::viz::DataTransformInterface DataTransformInterface.h
 * Data transform interface, see @ref DataTransformInterfaceData for the interface properties.
 */
using DataTransformInterface =
    InterfaceData<detail::DataTransformInterfaceDataIn, detail::DataTransformInterfaceDataOut,
                  detail::DataTransformInterfaceDataPrivate>;

} // namespace clara::viz
