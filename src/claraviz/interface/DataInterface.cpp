/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/DataInterface.h"

#include <set>
#include <type_traits>

#include <claraviz/util/SharedMemoryBlob.h>
#include <claraviz/util/Validator.h>
#include <claraviz/util/Log.h>

#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
#include <nvsharedmemory/SharedMemory.h>
namespace nvsharedmemory = nvidia::sharedmemory;
#endif

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(DataConfigInterface::Message);
template<>
DEFINE_CLASS_MESSAGEID(DataCropInterface::Message);
template<>
DEFINE_CLASS_MESSAGEID(DataInterface::Message);
template<>
DEFINE_CLASS_MESSAGEID(DataHistogramInterface::Message);
DEFINE_CLASS_MESSAGEID(DataHistogramInterface::MessageHistogram);
template<>
DEFINE_CLASS_MESSAGEID(DataTransformInterface::Message);

std::ostream &operator<<(std::ostream &os, const DataElementType &data_element_type)
{
    switch (data_element_type)
    {
    case DataElementType::UNKNOWN:
        os << std::string("UNKNOWN");
        break;
    case DataElementType::INT8:
        os << std::string("INT8");
        break;
    case DataElementType::UINT8:
        os << std::string("UINT8");
        break;
    case DataElementType::INT16:
        os << std::string("INT16");
        break;
    case DataElementType::UINT16:
        os << std::string("UINT16");
        break;
    case DataElementType::INT32:
        os << std::string("INT32");
        break;
    case DataElementType::UINT32:
        os << std::string("UINT32");
        break;
    case DataElementType::HALF_FLOAT:
        os << std::string("HALF_FLOAT");
        break;
    case DataElementType::FLOAT:
        os << std::string("FLOAT");
        break;
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

size_t BytesPerElement(DataElementType type)
{
    switch (type)
    {
    case DataElementType::INT8:
        return sizeof(int8_t);
    case DataElementType::UINT8:
        return sizeof(uint8_t);
    case DataElementType::INT16:
        return sizeof(int16_t);
    case DataElementType::UINT16:
        return sizeof(uint16_t);
    case DataElementType::INT32:
        return sizeof(int32_t);
    case DataElementType::UINT32:
        return sizeof(uint32_t);
    case DataElementType::HALF_FLOAT:
        return sizeof(float) / 2;
    case DataElementType::FLOAT:
        return sizeof(float);
    }

    throw InvalidState() << "Unhandled element type " << static_cast<std::underlying_type<DataElementType>::type>(type);
}

namespace detail
{

template<>
DataConfigInterfaceLevel<InterfaceValueT>::DataConfigInterfaceLevel()
    : size({},
           [](const std::vector<uint32_t> &value) {
               if (value.empty())
               {
                   throw InvalidArgument("Size") << "needs to have at least one element";
               }
               for (auto &&item : value)
               {
                   if (item == 0)
                   {
                       throw InvalidArgument("Size") << "every element is expected to be > 0";
                   }
               }
           })
    , element_size({},
                   [](const std::vector<float> &value) {
                       for (auto &&item : value)
                       {
                           if (item <= 0.f)
                           {
                               throw InvalidArgument("Element Size") << "every element is expected to be > 0.f";
                           }
                       }
                   })
    , element_range({}, [](const std::vector<Vector2f> &value) {
        for (auto &&item : value)
        {
            if (item(0) > item(1))
            {
                throw InvalidArgument("Element Range") << "not a valid range";
            }
        }
    })
{
}

template<>
DataConfigInterfaceLevel<InterfaceDirectT>::DataConfigInterfaceLevel()
{
}

template<>
DataConfigInterfaceArray<InterfaceValueT>::DataConfigInterfaceArray()
    : id("")
    , dimension_order("",
                      [](const std::string &value) {
                          std::set<char> set;
                          for (auto &&item : value)
                          {
                              if ((item != 'X') && (item != 'Y') && (item != 'Z') && (item != 'I') && (item != 'T') &&
                                  (item != 'D') && (item != 'M') && (item != 'C'))
                              {
                                  throw InvalidArgument("Dimension Order")
                                      << "has an invalid element '" << item << "': '" << value << "'";
                              }
                              if (!set.insert(item).second)
                              {
                                  throw InvalidArgument("Dimension Order")
                                      << "has a duplicate element '" << item << "': '" << value << "'";
                              }
                          }
                          if (value.find('X') == std::string::npos)
                          {
                              throw InvalidArgument("Dimension Order")
                                  << "at least the 'X' dimension has to be present";
                          }
                          if ((value.find('D') == std::string::npos) && (value.find('M') == std::string::npos) &&
                              (value.find('C') == std::string::npos))
                          {
                              throw InvalidArgument("Dimension Order")
                                  << "could not find one of the data element definitions 'D', 'M' or 'C'";
                          }
                      })
    , element_type(DataElementType::UNKNOWN,
                   [this](const DataElementType value) {
                       if (dimension_order.Get().find('M') != std::string::npos)
                       {
                           if ((value != DataElementType::UINT8) && (value != DataElementType::UINT16) &&
                               (value != DataElementType::UINT32))
                           {
                               throw InvalidArgument("Element type")
                                   << "segmentation volumes support unsigned integer types only";
                           }
                       }
                   })
    , permute_axis({},
                   [this](const std::vector<uint32_t> &value) {
                       if (value.size() > dimension_order.Get().size())
                       {
                           throw InvalidArgument("Permute Axis")
                               << "element count has to be lower than dimension order element count";
                       }
                       std::set<uint32_t> set;
                       for (auto &&item : value)
                       {
                           if (item > dimension_order.Get().size() - 1)
                           {
                               throw InvalidArgument("Permute Axis")
                                   << "has an invalid element '" << item << "', max dimensions are "
                                   << dimension_order.Get().size() - 1;
                           }
                           if (!set.insert(item).second)
                           {
                               throw InvalidArgument("Permute Axis") << "has a duplicate element '" << item << "'";
                           }
                       }
                   })
    , flip_axes({}, [this](const std::vector<bool> &value) {
        if (value.size() > dimension_order.Get().size())
        {
            throw InvalidArgument("Flip Axes") << "element count has to be lower than dimension order element count";
        }
    })
{
}

template<>
DataConfigInterfaceArray<InterfaceDirectT>::DataConfigInterfaceArray()
{
}

template<>
template<>
DataConfigInterface::DataIn::Array *DataConfigInterface::DataIn::GetOrAddArray(const std::string &id)
{
    std::list<Array>::iterator it =
        std::find_if(arrays.begin(), arrays.end(), [id](const Array &array) { return array.id == id; });
    if (it == arrays.end())
    {
        arrays.emplace_back();
        it = arrays.end();
        --it;
        it->id = id;
    }
    return &*it;
}

template<typename T>
typename T::Array *GetArray(std::list<typename T::Array> &arrays, const std::string &id)
{
    typename std::list<typename T::Array>::iterator it =
        std::find_if(arrays.begin(), arrays.end(), [id](const typename T::Array &array) { return array.id == id; });
    if (it == arrays.end())
    {
        throw InvalidArgument("id") << "Array with id '" << id << "' not found";
    }
    return &*it;
}

template<typename T>
const typename T::Array *GetArray(const std::list<typename T::Array> &arrays, const std::string &id)
{
    typename std::list<typename T::Array>::const_iterator it =
        std::find_if(arrays.cbegin(), arrays.cend(), [id](const typename T::Array &array) { return array.id == id; });
    if (it == arrays.end())
    {
        throw InvalidArgument("id") << "Array with id '" << id << "' not found";
    }
    return &*it;
}

} // namespace detail

template<>
template<>
DataConfigInterface::DataIn::Array *DataConfigInterface::DataIn::GetArray(const std::string &name)
{
    return detail::GetArray<DataConfigInterface::DataIn>(arrays, name);
}

template<>
const DataConfigInterface::DataIn::Array *DataConfigInterface::DataIn::GetArray(const std::string &name) const
{
    return detail::GetArray<const DataConfigInterface::DataIn>(arrays, name);
}

template<>
const DataConfigInterface::DataOut::Array *DataConfigInterface::DataOut::GetArray(const std::string &name) const
{
    return detail::GetArray<const DataConfigInterface::DataOut>(arrays, name);
}

namespace detail
{

template<>
DataConfigInterfaceData<InterfaceValueT>::DataConfigInterfaceData()
    : sharedmemory_driver_id("",
                             [](const std::string &value) {
                                 if (!value.empty())
                                 {
#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
                                     nvsharedmemory::UUID uuid;
                                     if (nvsharedmemory::UUID::fromString(value, &uuid) != nvsharedmemory::STATUS_OK)
                                     {
                                         throw InvalidArgument("sharedmemory_driver_id")
                                             << value << " is not a valid UUID";
                                     }
#else  // CLARA_VIZ_USE_NVSHAREDMEMORY
                                    Log(LogLevel::Error) << "DataInterface invalid argument 'sharedmemory_driver_id' "
                                        << value << ", shared memory is not supported";
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY
                                 }
                             })
    , streaming(false)
{
}

template<>
DataConfigInterfaceData<InterfaceDirectT>::DataConfigInterfaceData()
{
}

template<>
DataCropInterface::DataIn::DataCropInterfaceData()
    : limits({}, [](const std::vector<Vector2f> &value) {
        for (auto &&limit : value)
        {
            ValidatorRange(limit, 0.f, 1.f, "limit");
        }
    })
{
}

template<>
DataCropInterface::DataOut::DataCropInterfaceData()
{
}

template<>
DataInterfaceData<InterfaceValueT>::DataInterfaceData()
    : array_id("",
               [this](const std::string &value) {
                   const DataConfigInterfaceDataOut &data_config = get_data_config_();

                   // only id's in the data configuration are allowed
                   if (!data_config.GetArray(value))
                   {
                       throw InvalidArgument("array_id")
                           << "array with id '" << value << "' is not configured, can't set data";
                   }
               })
    , level(0,
            [this](const uint32_t value) {
                const DataConfigInterfaceDataOut &data_config = get_data_config_();

                const DataConfigInterface::DataOut::Array *array = data_config.GetArray(array_id.Get());
                if (!array)
                {
                    throw InvalidArgument("level")
                        << "array with id '" << array_id.Get() << "' is not configured, can't set data";
                }

                size_t index = 0;
                if (value >= array->levels.size())
                {
                    throw InvalidArgument("level") << "is invalid, array " << array_id.Get() << " has "
                                                   << array->levels.size() << " levels, but requested level " << value;
                }
            })
    , offset({},
             [this](const std::vector<uint32_t> &value) {
                 const DataConfigInterfaceDataOut &data_config = get_data_config_();

                 const DataConfigInterface::DataOut::Array *array = data_config.GetArray(array_id.Get());
                 if (!array)
                 {
                     throw InvalidArgument("offset")
                         << "array with id '" << array_id.Get() << "' is not configured, can't set data";
                 }

                 if (value.size() > array->dimension_order.size())
                 {
                     throw InvalidArgument("offset")
                         << "element count has to be lower than dimension order element count";
                 }
                 if (level.Get() >= array->levels.size())
                 {
                     throw InvalidArgument("offset")
                         << "can't set, array " << array_id.Get() << " has " << array->levels.size()
                         << " levels, but requested level " << level.Get();
                 }
                 const auto &cur_level = std::next(array->levels.cbegin(), level.Get());
                 for (size_t index = 0; index < value.size(); ++index)
                 {
                     if (value[index] > cur_level->size[index] - (index >= size.Get().size() ? 1 : size.Get()[index]))
                     {
                         throw InvalidArgument("offset") << "offset at index " << index << " is too big";
                     }
                 }
             })
    , size({}, [this](const std::vector<uint32_t> &value) {
        const DataConfigInterfaceDataOut &data_config = get_data_config_();

        for (size_t index = 0; index < value.size(); ++index)
        {
            if (value[index] == 0)
            {
                throw InvalidArgument("size") << "size at index " << index << " is zero";
            }
        }

        const DataConfigInterface::DataOut::Array *array = data_config.GetArray(array_id.Get());
        if (!array)
        {
            throw InvalidArgument("size")
                << "array with id '" << array_id.Get() << "' is not configured, can't set data";
        }

        if (value.size() > array->dimension_order.size())
        {
            throw InvalidArgument("size") << "element count has to be lower than dimension order element count";
        }
        if (level.Get() >= array->levels.size())
        {
            throw InvalidArgument("size") << "can't set, array " << array_id.Get() << " has " << array->levels.size()
                                          << " levels, but requested level " << level.Get();
        }
        const auto &cur_level = std::next(array->levels.cbegin(), level.Get());
        for (size_t index = 0; index < value.size(); ++index)
        {
            if (value[index] > cur_level->size[index] - (index >= offset.Get().size() ? 0 : offset.Get()[index]))
            {
                throw InvalidArgument("size") << "size at index " << index << " is too big";
            }
        }
    })
{
}

DataInterfaceDataBase<InterfaceValueT>::DataInterfaceDataBase()
    : sharedmemory_allocation_id("", [](const std::string &value) {
        if (!value.empty())
        {
#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
            nvsharedmemory::UUID uuid;
            if (nvsharedmemory::UUID::fromString(value, &uuid) != nvsharedmemory::STATUS_OK)
            {
                throw InvalidArgument("sharedmemory_allocation_id") << value << " is not a valid UUID";
            }
#else  // CLARA_VIZ_USE_NVSHAREDMEMORY
            Log(LogLevel::Error) << "DataInterface invalid argument 'sharedmemory_driver_id' "
                << value << ", shared memory is not supported";
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY
        }
    })
{
}

void DataInterfaceDataBase<InterfaceValueT>::SetGetDataConfig(
    const std::function<const DataConfigInterfaceDataOut &()> &get_data_config)
{
    get_data_config_ = get_data_config;
}

template<>
DataInterfaceBase::DataOut::DataInterfaceData()
{
}

} // namespace detail

/**
 * Copy a data config interface structure to a data config POD structure.
 */
template<>
DataConfigInterface::DataOut DataConfigInterface::Get()
{
    AccessGuardConst access(this);

    DataConfigInterface::DataOut data_out;

    data_out.arrays.clear();
    for (auto &&array : access->arrays)
    {
        data_out.arrays.emplace_back();

        auto &out_array = data_out.arrays.back();

        out_array.id              = array.id;
        out_array.dimension_order = array.dimension_order.Get();
        out_array.element_type    = array.element_type.Get();
        out_array.permute_axis    = array.permute_axis.Get();
        out_array.flip_axes       = array.flip_axes.Get();

        out_array.levels.clear();
        for (auto &&level : array.levels)
        {
            out_array.levels.emplace_back();

            auto &out_level = out_array.levels.back();

            out_level.size          = level.size.Get();
            out_level.element_size  = level.element_size.Get();
            out_level.element_range = level.element_range.Get();
        }
    }
    data_out.fetch_func             = access->fetch_func;
    data_out.sharedmemory_driver_id = access->sharedmemory_driver_id.Get();
    data_out.streaming              = access->streaming;

    return data_out;
}

/**
 * Copy a data crop interface structure to a data crop POD structure.
 */
template<>
DataCropInterface::DataOut DataCropInterface::Get()
{
    AccessGuardConst access(this);

    DataCropInterface::DataOut data_out;
    data_out.limits = access->limits.Get();

    return data_out;
}

DataInterface::DataInterface()
{
    Reset();
}

void DataInterface::Reset()
{
    detail::DataInterfaceBase::Reset();
    data_.Reset();

    AccessGuard access(*this);

    // set the data config get function for the input data, it reads messages delivered by the data config
    // interface and updates our private copy of the data configuration
    access->SetGetDataConfig([this]() -> const detail::DataConfigInterfaceDataOut & {
        std::shared_ptr<const clara::viz::Message> message;
        while ((message = DequeueMessage()))
        {
            if (message->GetID() == DataConfigInterface::Message::id_)
            {
                data_.data_config_ = std::static_pointer_cast<const DataConfigInterface::Message>(message)->data_out_;
            }
            else
            {
                throw InvalidState() << "Unhandled message Id " << message->GetID().GetName();
            }
        }
        return data_.data_config_;
    });
}

void detail::DataInterfaceDataPrivate::Reset()
{
    sharedmemory_context_.reset();
    sharedmemory_driver_.reset();
}

std::shared_ptr<IBlob> detail::DataInterfaceDataPrivate::LockForRead(const std::string &sharedmemory_allocation_id)
{
#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
    if (data_config_.sharedmemory_driver_id.empty())
    {
        throw InvalidState() << "The shared memory driver ID is not set";
    }

    if (sharedmemory_driver_)
    {
        // check if the driver UUID changed
        if (nvsharedmemory::interface_cast<nvsharedmemory::IDriver>(sharedmemory_driver_)->getId().toString() !=
            data_config_.sharedmemory_driver_id)
        {
            // close the old context and driver
            sharedmemory_context_.reset();
            sharedmemory_driver_.reset();
        }
    }

    if (!sharedmemory_driver_)
    {
        // open the shared memory driver
        nvsharedmemory::UUID driver_uuid;
        if (nvsharedmemory::UUID::fromString(data_config_.sharedmemory_driver_id, &driver_uuid) !=
            nvsharedmemory::STATUS_OK)
        {
            throw InvalidState() << "Invalid shared memory driver ID " << data_config_.sharedmemory_driver_id;
        }

        std::shared_ptr<nvsharedmemory::Driver> local_driver(nvsharedmemory::Driver::open(driver_uuid));
        nvsharedmemory::IDriver *iDriver = nvsharedmemory::interface_cast<nvsharedmemory::IDriver>(local_driver);
        if (!iDriver)
        {
            throw InvalidState() << "Could not open shared memory driver with ID "
                                 << data_config_.sharedmemory_driver_id;
        }
        // make sure drivers are freed on release
        iDriver->freeOnRelease(true);

        std::shared_ptr<nvsharedmemory::Context> local_context(iDriver->createContext());
        nvsharedmemory::IContext *iContext = nvsharedmemory::interface_cast<nvsharedmemory::IContext>(local_context);
        if (!iContext)
        {
            throw InvalidState() << "Could not create shared memory context";
        }
        // make sure contexts are freed on release
        iContext->freeOnRelease(true);

        sharedmemory_driver_  = local_driver;
        sharedmemory_context_ = local_context;
    }

    // get the allocation UUID from the string
    nvsharedmemory::UUID uuid;
    if (nvsharedmemory::UUID::fromString(sharedmemory_allocation_id, &uuid) != nvsharedmemory::STATUS_OK)
    {
        throw InvalidState() << "Invalid shared memory allocation ID " << sharedmemory_allocation_id;
    }

    // import the allocation
    std::shared_ptr<nvsharedmemory::Allocation> allocation(
        nvsharedmemory::interface_cast<nvsharedmemory::IContext>(sharedmemory_context_)->importFromAllocId(uuid));
    nvsharedmemory::IAllocation *iAllocation = nvsharedmemory::interface_cast<nvsharedmemory::IAllocation>(allocation);
    if (!iAllocation)
    {
        throw InvalidState() << "Can't import allocation with ID " << sharedmemory_allocation_id;
    }

    // make sure the allocation is freed when it is no longer used
    iAllocation->freeOnRelease(true);

    return std::shared_ptr<IBlob>(new SharedMemoryBlob(sharedmemory_context_, allocation));
#else  // CLARA_VIZ_USE_NVSHAREDMEMORY
    throw RuntimeError() << "Shared memory is not supported";
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY
}

/**
 * Copy a data interface structure to a data POD structure.
 */
template<>
detail::DataInterfaceBase::DataOut detail::DataInterfaceBase::Get()
{
    AccessGuardConst access(this);

    DataInterface::DataOut data_out;
    data_out.array_id = access->array_id.Get();
    data_out.level    = access->level.Get();
    data_out.offset   = access->offset.Get();
    data_out.size     = access->size.Get();

    if (!access->sharedmemory_allocation_id.Get().empty())
    {
        // update the configuration to fetch the driver UUID
        access->get_data_config_();
        // lock the data for read
        data_out.blob = data_.LockForRead(access->sharedmemory_allocation_id.Get());
    }
    else if (access->blob)
    {
        data_out.blob = access->blob;
    }

    return data_out;
}

DataHistogramInterface::DataHistogramInterface() {}

/**
 * Copy a data histogram interface structure to a data histogram POD structure.
 */
template<>
DataHistogramInterface::DataOut detail::DataHistogramInterfaceBase::Get()
{
    AccessGuardConst access(this);

    DataHistogramInterface::DataOut data_out;
    data_out.array_id = access->array_id;
    data_out.receiver = static_cast<DataHistogramInterface *>(this)->shared_from_this();

    return data_out;
}

void DataHistogramInterface::GetHistogram(const std::string &array_id, std::vector<float> &histogram)
{
    // lock to make sure only one histogram request is in flight, else we can't be sure that
    // the message received is triggered by this call or by another one
    std::unique_lock<std::mutex> lock(mutex_);

    // Send the message to the renderer to query histogram data
    {
        AccessGuard access(*this);

        access->array_id = array_id;
    }

    // wait for the renderer message containing the histogram data
    Wait();

    std::shared_ptr<const clara::viz::Message> message = DequeueMessage();
    if (message->GetID() != DataHistogramInterface::MessageHistogram::id_)
    {
        throw InvalidState() << "Unexpected message Id " << message->GetID().GetName();
    }
    histogram = std::static_pointer_cast<const DataHistogramInterface::MessageHistogram>(message)->histogram_;
}

namespace detail
{

template<>
DataTransformInterface::DataIn::DataTransformInterfaceData()
{
}

template<>
DataTransformInterface::DataOut::DataTransformInterfaceData()
{
}

} // namespace detail

/**
 * Copy a data transform interface structure to a data transform POD structure.
 */
template<>
DataTransformInterface::DataOut DataTransformInterface::Get()
{
    AccessGuardConst access(this);

    DataTransformInterface::DataOut data_out;
    data_out.matrix = access->matrix;

    return data_out;
}

} // namespace clara::viz
