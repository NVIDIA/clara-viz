/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "SharedMemoryBlob.h"

#include <nvsharedmemory/SharedMemory.h>

namespace clara::viz
{

SharedMemoryBlob::SharedMemoryBlob(const std::shared_ptr<nvidia::sharedmemory::Context> &context,
                                   const std::shared_ptr<nvidia::sharedmemory::Allocation> &allocation)
    : context_(context)
    , allocation_(allocation)
{
}

std::unique_ptr<IBlob::AccessGuard> SharedMemoryBlob::Access()
{
    std::unique_ptr<Blob::AccessGuard> guard(new AccessGuard(this));
    Blob::SyncAccess(guard.get());
    return guard;
}

std::unique_ptr<IBlob::AccessGuard> SharedMemoryBlob::Access(CUstream stream)
{
    std::unique_ptr<Blob::AccessGuard> guard(new AccessGuard(this));
    Blob::SyncAccess(guard.get(), stream);
    return guard;
}

std::unique_ptr<IBlob::AccessGuardConst> SharedMemoryBlob::AccessConst()
{
    std::unique_ptr<Blob::AccessGuardConst> guard(new AccessGuardConst(this));
    Blob::SyncAccessConst(guard.get());
    return guard;
}

std::unique_ptr<IBlob::AccessGuardConst> SharedMemoryBlob::AccessConst(CUstream stream)
{
    std::unique_ptr<Blob::AccessGuardConst> guard(new AccessGuardConst(this));
    Blob::SyncAccessConst(guard.get(), stream);
    return guard;
}

size_t SharedMemoryBlob::GetSize() const
{
    return nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocation>(allocation_)->getSize();
}

SharedMemoryBlob::AccessGuard::AccessGuard(Blob *blob)
    : Blob::AccessGuard(blob)
{
    nvidia::sharedmemory::IAllocation *iAllocation =
        nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocation>(
            static_cast<SharedMemoryBlob *>(blob_)->allocation_);
    if (!iAllocation)
    {
        throw InvalidState() << "Failed to get shared memory allocation interface";
    }

    // lock for read/write
    lock_.reset(iAllocation->lock(nvidia::sharedmemory::LOCK_TYPE_READWRITE));
    nvidia::sharedmemory::IAllocationMapper *iAllocationMapper =
        nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocationMapper>(lock_);
    if (!iAllocationMapper)
    {
        throw InvalidState() << "Failed to lock the shared memory allocation for read/write access";
    }

    // and map into the process
    mapping_.reset(iAllocationMapper->map());
    if (!mapping_)
    {
        throw InvalidState() << "Failed to map the shared memory allocation";
    }
}

void *SharedMemoryBlob::AccessGuard::GetData()
{
    return reinterpret_cast<void *>(
        nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocationMapping>(mapping_)->getPtr());
}

SharedMemoryBlob::AccessGuardConst::AccessGuardConst(Blob *blob)
    : Blob::AccessGuardConst(blob)
{
    nvidia::sharedmemory::IAllocation *iAllocation =
        nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocation>(
            static_cast<SharedMemoryBlob *>(blob_)->allocation_);
    if (!iAllocation)
    {
        throw InvalidState() << "Failed to get shared memory allocation interface";
    }

    // lock for read
    lock_.reset(iAllocation->lock(nvidia::sharedmemory::LOCK_TYPE_READONLY));
    nvidia::sharedmemory::IAllocationMapper *iAllocationMapper =
        nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocationMapper>(lock_);
    if (!iAllocationMapper)
    {
        throw InvalidState() << "Failed to lock the shared memory allocation for read access";
    }

    // and map into the process
    mapping_.reset(iAllocationMapper->map());
    if (!mapping_)
    {
        throw InvalidState() << "Failed to map the shared memory allocation";
    }
}

const void *SharedMemoryBlob::AccessGuardConst::GetData()
{
    return reinterpret_cast<const void *>(
        nvidia::sharedmemory::interface_cast<nvidia::sharedmemory::IAllocationMapping>(mapping_)->getPtr());
}

} // namespace clara::viz
