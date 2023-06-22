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

#include "claraviz/rpc/DataRPC.h"
#include "claraviz/rpc/TypesRPC.h"

#include <chrono>

#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
#include <nvsharedmemory/SharedMemory.h>
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY

#include <claraviz/util/Message.h>
#include <claraviz/util/Thread.h>
#include <claraviz/util/StdContainerBlob.h>
#include <claraviz/util/UniqueObj.h>

#include <grpcpp/grpcpp.h>

#include <nvidia/claraviz/fetch/v1/fetch.grpc.pb.h>

#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
namespace nvsharedmemory = nvidia::sharedmemory;
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY

namespace clara::viz
{

namespace detail
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace
{

/**
 *
 */
class FetchMessage : public Message
{
public:
    FetchMessage()
        : Message(id_)
    {
    }

    uintptr_t context_;
    std::string array_id_;
    uint32_t level_;
    std::vector<uint32_t> offset_;
    std::vector<uint32_t> size_;
    DataConfigInterface::DataOut::FetchCallbackFunc fetch_callback_func_;

    /// message id
    static const MessageID id_;
};

DEFINE_CLASS_MESSAGEID(FetchMessage);

/**
 * Shutdown message ID
 */
DEFINE_MESSAGEID(IDMESSAGE_SHUTDOWN);

} // anonymous namespace

class DataConfigResource::Impl : public MessageReceiver
{
public:
    Impl(DataConfigInterface &data_config);
    Impl() = delete;
    virtual ~Impl();

    /**
     * @brief Configure the fetch service
     *
     * @param fetch_uri [in] gRPC server to fetch data from, the server needs to support the 'FetchData' method
     * @param sharedmemory_driver_id [in] SharedMemory driver UUID
     */
    void Config(const std::string &fetch_uri, const std::string &sharedmemory_driver_id);

    /**
     * Fetch data.
     *
     * @param context [in] internal context of the fetch call, pass to fetch callback function
     * @param array_id [in] id of the array to fetch data from
     * @param level_index [in] index of the level to fetch data for
     * @param offset [in] offset into the level to fetch data for
     * @param size [in] size of the data to fetch
     * @param fetch_callback_func [in] callback function, called when data is received
     */
    bool Fetch(uintptr_t context, const std::string &array_id, uint32_t level_index,
               const std::vector<uint32_t> &offset, const std::vector<uint32_t> &size,
               const DataConfigInterface::DataOut::FetchCallbackFunc &fetch_callback_func);

    /// Data config interface
    DataConfigInterface &data_config_;

private:
    std::string fetch_uri_;
    std::string sharedmemory_driver_id_;

#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
    /// Shared memory driver
    std::shared_ptr<nvidia::sharedmemory::Driver> sharedmemory_driver_;
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY

    /**
     * Fetch thread function.
     */
    void ThreadFunction(std::function<void()> ready);

    /// fetch thread
    std::unique_ptr<Thread> thread_;
};

DataConfigResource::Impl::Impl(DataConfigInterface &data_config)
    : data_config_(data_config)
{
}

void DataConfigResource::Impl::Config(const std::string &fetch_uri, const std::string &sharedmemory_driver_id)
{
    if (sharedmemory_driver_id_ != sharedmemory_driver_id)
    {
        sharedmemory_driver_id_ = sharedmemory_driver_id;

        if (sharedmemory_driver_id_.empty())
        {
#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
            sharedmemory_driver_.reset();
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY
        }
        else
        {
#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
            // open the shared memory driver
            nvsharedmemory::UUID driver_uuid;
            if (nvsharedmemory::UUID::fromString(sharedmemory_driver_id, &driver_uuid) != nvsharedmemory::STATUS_OK)
            {
                throw InvalidState() << "Invalid shared memory driver ID " << sharedmemory_driver_id;
            }

            sharedmemory_driver_.reset(nvsharedmemory::Driver::open(driver_uuid));
            nvsharedmemory::IDriver *iDriver =
                nvsharedmemory::interface_cast<nvsharedmemory::IDriver>(sharedmemory_driver_);
            if (!iDriver)
            {
                throw InvalidState() << "Could not open shared memory driver with ID " << sharedmemory_driver_id;
            }
            // make sure contexts are freed on release
            iDriver->freeOnRelease(true);
#else  // CLARA_VIZ_USE_NVSHAREDMEMORY
            Log(LogLevel::Error) << "Shared memory not supported";
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY
        }
    }

    fetch_uri_ = fetch_uri;
    if (!fetch_uri_.empty() && !thread_)
    {
        // run the fetch thread
        thread_.reset(new Thread("Fetch thread", [this](std::function<void()> ready) { ThreadFunction(ready); }));
    }
}

DataConfigResource::Impl::~Impl()
{
    if (thread_)
    {
        // shutdown the thread
        EnqueueMessage(std::make_shared<Message>(IDMESSAGE_SHUTDOWN));

        // destroy the thread
        thread_.reset();
    }
}

void DataConfigResource::Impl::ThreadFunction(std::function<void()> ready)
{
    // thread is ready now
    ready();

    while (true)
    {
        try
        {
            Wait();

            // if set then shutdown
            bool shutdown = false;

            // check for messages
            std::shared_ptr<const Message> message;
            while ((message = DequeueMessage()))
            {
                Log(LogLevel::Debug) << "Fetch received " << message->GetID().GetName();

                if (message->GetID() == FetchMessage::id_)
                {
                    const auto &fetch_message = std::static_pointer_cast<const FetchMessage>(message);

                    // create a guard to automatically do the final call on exit so the thread which started
                    // the fetch will continue even on error
                    Guard final_call([fetch_message] {
                        fetch_message->fetch_callback_func_(fetch_message->context_, fetch_message->level_,
                                                            fetch_message->offset_, fetch_message->size_, nullptr, 0);
                    });

                    // connect to the fetch server
                    std::shared_ptr<grpc::Channel> channel =
                        grpc::CreateChannel(fetch_uri_, grpc::InsecureChannelCredentials());

                    const auto timeout = std::chrono::seconds(10);
                    if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout))
                    {
                        throw RuntimeError() << "Can't connect to fetch server at " << fetch_uri_;
                    }
                    std::unique_ptr<nvidia::claraviz::fetch::v1::Fetch::Stub> stub =
                        nvidia::claraviz::fetch::v1::Fetch::NewStub(channel);

                    // send the request
                    nvidia::claraviz::fetch::v1::FetchDataRequest request;

                    request.mutable_array_id()->set_value(fetch_message->array_id_);
                    request.set_level(fetch_message->level_);
                    for (auto &&offset : fetch_message->offset_)
                    {
                        request.mutable_offset()->Add(offset);
                    }
                    for (auto &&size : fetch_message->size_)
                    {
                        request.mutable_size()->Add(size);
                    }

                    // wait for the response
                    grpc::ClientContext context;
                    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
                    std::unique_ptr<grpc::ClientReader<nvidia::claraviz::fetch::v1::FetchDataResponse>> reader =
                        stub->FetchData(&context, request);
                    if (!reader)
                    {
                        throw RuntimeError() << "Failed to send FetchData request";
                    }

                    // read the responses
                    nvidia::claraviz::fetch::v1::FetchDataResponse response;
                    while (reader->Read(&response))
                    {
                        std::vector<uint32_t> offset(response.offset_size());
                        for (size_t index = 0; index < response.offset_size(); ++index)
                        {
                            offset[index] = response.offset(index);
                        }

                        std::vector<uint32_t> size(response.size_size());
                        for (size_t index = 0; index < response.size_size(); ++index)
                        {
                            size[index] = response.size(index);
                        }

                        const void *data = nullptr;
                        size_t data_size = 0;
#ifdef CLARA_VIZ_USE_NVSHAREDMEMORY
                        std::shared_ptr<nvsharedmemory::Context> sharedmemory_context;
                        std::shared_ptr<nvsharedmemory::Allocation> sharedmemory_allocation;
                        std::shared_ptr<nvsharedmemory::AllocationLock> sharedmemory_lock;
                        std::shared_ptr<nvsharedmemory::AllocationMapping> sharedmemory_mapping;
                        if (response.has_sharedmemory())
                        {
                            // get the context UUID from the string
                            nvsharedmemory::UUID context_uuid;
                            if (nvsharedmemory::UUID::fromString(response.sharedmemory().context_id().value(),
                                                                 &context_uuid) != nvsharedmemory::STATUS_OK)
                            {
                                throw InvalidState() << "Invalid shared memory context ID "
                                                     << response.sharedmemory().context_id().value();
                            }
                            sharedmemory_context.reset(
                                nvsharedmemory::interface_cast<nvsharedmemory::IDriver>(sharedmemory_driver_)
                                    ->openContext(context_uuid));
                            nvsharedmemory::IContext *iContext =
                                nvsharedmemory::interface_cast<nvsharedmemory::IContext>(sharedmemory_context);
                            if (!iContext)
                            {
                                throw InvalidState() << "Could not create shared memory context";
                            }
                            // make sure allocations are freed on release
                            iContext->freeOnRelease(true);

                            // get the allocation UUID from the string
                            nvsharedmemory::UUID allocation_uuid;
                            if (nvsharedmemory::UUID::fromString(response.sharedmemory().allocation_id().value(),
                                                                 &allocation_uuid) != nvsharedmemory::STATUS_OK)
                            {
                                throw InvalidState() << "Invalid shared memory allocation ID "
                                                     << response.sharedmemory().allocation_id().value();
                            }

                            // import the allocation
                            sharedmemory_allocation.reset(iContext->createFromAllocId(allocation_uuid));
                            nvsharedmemory::IAllocation *iAllocation =
                                nvsharedmemory::interface_cast<nvsharedmemory::IAllocation>(sharedmemory_allocation);
                            if (!iAllocation)
                            {
                                throw InvalidState() << "Can't import allocation with ID "
                                                     << response.sharedmemory().allocation_id().value();
                            }

                            // make sure the allocation is freed when it is no longer used
                            iAllocation->freeOnRelease(true);

                            // lock for read
                            sharedmemory_lock.reset(iAllocation->lock(nvsharedmemory::LOCK_TYPE_READONLY));
                            nvsharedmemory::IAllocationMapper *iAllocationMapper =
                                nvsharedmemory::interface_cast<nvsharedmemory::IAllocationMapper>(sharedmemory_lock);
                            if (!iAllocationMapper)
                            {
                                throw InvalidState()
                                    << "Failed to lock the allocation with ID "
                                    << response.sharedmemory().allocation_id().value() << " for read access";
                            }

                            // and map into the process
                            sharedmemory_mapping.reset(iAllocationMapper->map());
                            if (!sharedmemory_mapping)
                            {
                                throw InvalidState() << "Failed to map the allocation with ID "
                                                     << response.sharedmemory().allocation_id().value();
                            }

                            data =
                                nvsharedmemory::interface_cast<nvsharedmemory::IAllocationMapping>(sharedmemory_mapping)
                                    ->getPtr();
                            data_size = iAllocation->getSize();
                        }
                        else
#endif // CLARA_VIZ_USE_NVSHAREDMEMORY
                        {
                            data      = reinterpret_cast<const void *>(response.data().data());
                            data_size = response.data().size();
                        }

                        // call the callback with the received data
                        if (!fetch_message->fetch_callback_func_(fetch_message->context_, response.level(), offset,
                                                                 size, data, data_size))
                        {
                            context.TryCancel();
                        }
                    }

                    if (reader->Finish().error_code() == grpc::StatusCode::CANCELLED)
                    {
                        Log(LogLevel::Debug) << "Cancelled the FetchData request";
                    }
                }
                else if (message->GetID() == IDMESSAGE_SHUTDOWN)
                {
                    shutdown = true;
                }
                else
                {
                    throw InvalidState() << "Unhandled message Id " << message->GetID().GetName();
                }
            }
            if (shutdown)
            {
                break;
            }
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Fetch thread threw exception " << e.what();
        }
        catch (...)
        {
            Log(LogLevel::Error) << "Fetch thread threw unknown exception";
        }
    }
}

bool DataConfigResource::Impl::Fetch(uintptr_t context, const std::string &array_id, uint32_t level_index,
                                     const std::vector<uint32_t> &offset, const std::vector<uint32_t> &size,
                                     const DataConfigInterface::DataOut::FetchCallbackFunc &fetch_callback_func)
{
    if (fetch_uri_.empty())
    {
        return false;
    }

    // send the fetch message to the fetch thread
    auto fetch_message                  = std::make_shared<FetchMessage>();
    fetch_message->context_             = context;
    fetch_message->array_id_            = array_id;
    fetch_message->level_               = level_index;
    fetch_message->offset_              = offset;
    fetch_message->size_                = size;
    fetch_message->fetch_callback_func_ = fetch_callback_func;

    EnqueueMessage(fetch_message);

    return true;
}

DataConfigResource::DataConfigResource(DataConfigInterface &data_config)
    : impl_(new Impl(data_config))
{
}

void DataConfigContext::ExecuteRPC(cinematic_v1::DataConfigRequest &request, cinematic_v1::DataConfigResponse &response)
{
    if (request.arrays_size() == 0)
    {
        throw InvalidState() << "arrays is required";
    }

    const std::shared_ptr<DataConfigResource> &resource = GetResources();
    DataConfigInterface::AccessGuard access(resource->impl_->data_config_);

    // copy the arrays
    access->arrays.resize(request.arrays_size());
    auto src_array = request.arrays().cbegin();
    auto dst_array = access->arrays.begin();
    while (src_array < request.arrays().cend())
    {
        dst_array->id = src_array->id().value();
        dst_array->dimension_order.Set(src_array->dimension_order());

        switch (src_array->element_type())
        {
        case cinematic_v1::DataConfigRequest::Array::INT8:
            dst_array->element_type.Set(DataElementType::INT8);
            break;
        case cinematic_v1::DataConfigRequest::Array::UINT8:
            dst_array->element_type.Set(DataElementType::UINT8);
            break;
        case cinematic_v1::DataConfigRequest::Array::INT16:
            dst_array->element_type.Set(DataElementType::INT16);
            break;
        case cinematic_v1::DataConfigRequest::Array::UINT16:
            dst_array->element_type.Set(DataElementType::UINT16);
            break;
        case cinematic_v1::DataConfigRequest::Array::INT32:
            dst_array->element_type.Set(DataElementType::INT32);
            break;
        case cinematic_v1::DataConfigRequest::Array::UINT32:
            dst_array->element_type.Set(DataElementType::UINT32);
            break;
        case cinematic_v1::DataConfigRequest::Array::HALF_FLOAT:
            dst_array->element_type.Set(DataElementType::HALF_FLOAT);
            break;
        case cinematic_v1::DataConfigRequest::Array::FLOAT:
            dst_array->element_type.Set(DataElementType::FLOAT);
            break;
        case cinematic_v1::DataConfigRequest::Array::ELEMENT_TYPE_UNKNOWN:
            throw InvalidState() << "element type is required";
            break;
        default:
            throw InvalidState() << "Unhandled data element type " << src_array->element_type();
            break;
        }

        std::vector<uint32_t> permute_axis(src_array->permute_axes_size());
        for (size_t permute_index = 0; permute_index < src_array->permute_axes_size(); ++permute_index)
        {
            permute_axis[permute_index] = src_array->permute_axes(permute_index);
        }
        dst_array->permute_axis.Set(permute_axis);

        std::vector<bool> flip_axes(src_array->flip_axes_size());
        for (size_t flip_index = 0; flip_index < src_array->flip_axes_size(); ++flip_index)
        {
            flip_axes[flip_index] = src_array->flip_axes(flip_index);
        }
        dst_array->flip_axes.Set(flip_axes);

        // copy the levels
        if (src_array->levels_size() == 0)
        {
            throw InvalidState() << "levels is required";
        }

        dst_array->levels.resize(src_array->levels_size());
        auto src_level = src_array->levels().cbegin();
        auto dst_level = dst_array->levels.begin();
        while (src_level < src_array->levels().cend())
        {
            std::vector<uint32_t> size(src_level->size_size());
            for (size_t size_index = 0; size_index < src_level->size_size(); ++size_index)
            {
                size[size_index] = src_level->size(size_index);
            }
            dst_level->size.Set(size);

            std::vector<float> element_size(src_level->element_size_size());
            for (size_t element_index = 0; element_index < src_level->element_size_size(); ++element_index)
            {
                element_size[element_index] = src_level->element_size(element_index);
            }
            dst_level->element_size.Set(element_size);

            std::vector<Vector2f> element_range(src_level->element_range_size());
            for (size_t element_index = 0; element_index < src_level->element_range_size(); ++element_index)
            {
                element_range[element_index] = MakeVector2f(src_level->element_range(element_index));
            }
            dst_level->element_range.Set(element_range);

            ++src_level;
            ++dst_level;
        }

        ++src_array;
        ++dst_array;
    }
    access->fetch_func = [resource](
                             uintptr_t context, const std::string &array_id, uint32_t level_index,
                             const std::vector<uint32_t> &offset, const std::vector<uint32_t> &size,
                             const DataConfigInterface::DataOut::FetchCallbackFunc &fetch_callback_func) -> bool {
        return resource->impl_->Fetch(context, array_id, level_index, offset, size, fetch_callback_func);
    };
    access->sharedmemory_driver_id.Set(request.sharedmemory_driver_id().value());

    // configure the fetch service
    resource->impl_->Config(request.fetch_uri(), access->sharedmemory_driver_id.Get());

    switch (request.streaming())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->streaming = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->streaming = false;
        break;
    }
}

void DataContext::ExecuteRPC(cinematic_v1::DataRequest &request, cinematic_v1::DataResponse &response)
{
    if (!request.has_array_id())
    {
        throw InvalidState() << "array_id is required";
    }
    if (request.size_size() == 0)
    {
        throw InvalidState() << "size is required";
    }

    std::shared_ptr<DataInterface> data(GetResources()->data_);
    DataInterface::AccessGuard access(*data.get());

    access->array_id.Set(request.array_id().value());
    access->level.Set(request.level());

    // reset the offset before setting the size, the interface validates offset and size against the array config size. Keeping the old
    // offset might produce wrong validation errors.
    std::vector<uint32_t> offset(request.offset_size());
    access->offset.Set(offset);

    std::vector<uint32_t> size(request.size_size());
    for (size_t index = 0; index < request.size_size(); ++index)
    {
        size[index] = request.size(index);
    }
    access->size.Set(size);

    for (size_t index = 0; index < request.offset_size(); ++index)
    {
        offset[index] = request.offset(index);
    }
    access->offset.Set(offset);

    access->blob.reset();
    if (request.has_sharedmemory_allocation_id())
    {
        access->sharedmemory_allocation_id.Set(request.sharedmemory_allocation_id().value());
        access->blob.reset();
    }
    else
    {
        access->blob.reset(new StdContainerBlob<std::string>(std::unique_ptr<std::string>(request.release_data())));
        access->sharedmemory_allocation_id.Set("");
    }
}

void DataHistogramContext::ExecuteRPC(cinematic_v1::DataHistogramRequest &request,
                                      cinematic_v1::DataHistogramResponse &response)
{
    if (!request.has_array_id())
    {
        throw InvalidState() << "array_id is required";
    }

    std::vector<float> histogram;
    GetResources()->histogram_.lock()->GetHistogram(request.array_id().value(), histogram);

    // copy histogram data to response
    google::protobuf::RepeatedField<float> *response_histogram = response.mutable_data();
    response_histogram->Reserve(histogram.size());
    for (auto it = histogram.cbegin(); it != histogram.cend(); ++it)
    {
        response_histogram->Add(*it);
    }
}

} // namespace detail

} // namespace clara::viz
