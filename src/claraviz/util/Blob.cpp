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

#include "Blob.h"

#include <list>
#include <shared_mutex>

#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/util/Synchronized.h"

namespace clara::viz
{

class Blob::Impl
{
public:
    ~Impl()
    {
        // don't need to sync events here, CUDA is handling synchronization of freed memory for us.
    }

    class SharedData
    {
    public:
        void SyncAndClearWriteEvents()
        {
            for_each(write_events_.cbegin(), write_events_.cend(),
                     [](const UniqueCudaEvent &event) { CudaCheck(cuEventSynchronize(event.get())); });
            write_events_.clear();
        }

        void SyncWriteEvents(CUstream stream)
        {
            for_each(write_events_.cbegin(), write_events_.cend(),
                     [stream](const UniqueCudaEvent &event) { CudaCheck(cuStreamWaitEvent(stream, event.get(), 0)); });
        }
        void SyncAndClearWriteEvents(CUstream stream)
        {
            SyncWriteEvents(stream);
            write_events_.clear();
        }

        void SyncAndClearReadEvents()
        {
            for_each(read_events_.cbegin(), read_events_.cend(),
                     [](const UniqueCudaEvent &event) { CudaCheck(cuEventSynchronize(event.get())); });
            read_events_.clear();
        }

        void SyncAndClearReadEvents(CUstream stream)
        {
            for_each(read_events_.cbegin(), read_events_.cend(),
                     [stream](const UniqueCudaEvent &event) { CudaCheck(cuStreamWaitEvent(stream, event.get(), 0)); });
            read_events_.clear();
        }

        /// a list of write events to be synced with when reading or writing
        std::list<UniqueCudaEvent> write_events_;
        /// a list of read events to be synced with when writing
        std::list<UniqueCudaEvent> read_events_;
    };
    /// protects shared data
    Synchronized<SharedData> shared_;

    /// protects access to the memory blob, allows single writer but multiple readers
    std::shared_mutex mutex_;
};

Blob::Blob()
    : impl_(new Impl)
{
}

void Blob::SyncAccess(Blob::AccessGuard *guard)
{
    guard->lock_ = std::unique_lock(impl_->mutex_);

    Synchronized<Impl::SharedData>::AccessGuard access(impl_->shared_);

    // wait for events, we need to wait for both read and write events
    access->SyncAndClearReadEvents();
    access->SyncAndClearWriteEvents();
}

void Blob::SyncAccess(Blob::AccessGuard *guard, CUstream stream)
{
    guard->lock_         = std::unique_lock(impl_->mutex_);
    guard->stream_       = stream;
    guard->valid_stream_ = true;

    Synchronized<Impl::SharedData>::AccessGuard access(impl_->shared_);

    // wait for events, we need to wait for both read and write events
    access->SyncAndClearReadEvents(stream);
    access->SyncAndClearWriteEvents(stream);
}

void Blob::SyncAccessConst(Blob::AccessGuardConst *guard)
{
    guard->lock_ = std::shared_lock(impl_->mutex_);

    Synchronized<Impl::SharedData>::AccessGuard access(impl_->shared_);

    // wait for write events only
    access->SyncAndClearWriteEvents();
}

void Blob::SyncAccessConst(Blob::AccessGuardConst *guard, CUstream stream)
{
    guard->lock_         = std::shared_lock(impl_->mutex_);
    guard->stream_       = stream;
    guard->valid_stream_ = true;

    Synchronized<Impl::SharedData>::AccessGuard access(impl_->shared_);

    // wait for write events only, but don't clear them. Each reading stream
    // needs to wait for the write events individually.
    access->SyncWriteEvents(stream);
}

Blob::AccessGuard::~AccessGuard()
{
    // when releasing the access guard and a stream had been specified create an
    // event and add it to the blob
    if (valid_stream_)
    {
        try
        {
            Synchronized<Impl::SharedData>::AccessGuard access(blob_->impl_->shared_);

            UniqueCudaEvent event;
            event.reset([] {
                CUevent event;
                CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
                return event;
            }());
            CudaCheck(cuEventRecord(event.get(), stream_));
            access->write_events_.emplace_back(std::move(event));
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Blob::AccessGuard destructor failed with " << e.what();
        }
    }
}

Blob::AccessGuardConst::~AccessGuardConst()
{
    // when releasing the access guard and a stream had been specified create an
    // event and add it to the blob
    if (valid_stream_)
    {
        try
        {
            Synchronized<Impl::SharedData>::AccessGuard access(blob_->impl_->shared_);

            UniqueCudaEvent event;
            event.reset([] {
                CUevent event;
                CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
                return event;
            }());
            CudaCheck(cuEventRecord(event.get(), stream_));
            access->read_events_.emplace_back(std::move(event));
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Blob::AccessGuardConst destructor failed " << e.what();
        }
    }
}

} // namespace clara::viz