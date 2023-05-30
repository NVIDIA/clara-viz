/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/Blob.h"

/// forward declaration
namespace nvidia
{
namespace sharedmemory
{
class Context;
class Allocation;
class AllocationLock;
class AllocationMapping;
} // namespace sharedmemory
} // namespace nvidia

namespace clara::viz
{

/**
 * Blob to be used by shared memory allocations.
 * Note that this blob does not support recursive locking (multiple calls to Access() or AccessConst() at one time
 * from one thread).
 */
class SharedMemoryBlob : public Blob
{
public:
    /**
     * Construct
     *
     * @param context [in] shared memory context
     * @param allocation [in] shared memory allocation
     */
    explicit SharedMemoryBlob(const std::shared_ptr<nvidia::sharedmemory::Context> &context,
                              const std::shared_ptr<nvidia::sharedmemory::Allocation> &allocation);
    SharedMemoryBlob() = delete;

    /// IBlob virtual members
    ///@{
    std::unique_ptr<IBlob::AccessGuard> Access() override;
    std::unique_ptr<IBlob::AccessGuard> Access(CUstream stream) override;
    std::unique_ptr<IBlob::AccessGuardConst> AccessConst() override;
    std::unique_ptr<IBlob::AccessGuardConst> AccessConst(CUstream stream) override;
    size_t GetSize() const override;
    ///@}

private:
    class AccessGuard : public Blob::AccessGuard
    {
    public:
        AccessGuard(Blob *blob);

        void *GetData() override;

    private:
        std::shared_ptr<nvidia::sharedmemory::AllocationLock> lock_;
        std::shared_ptr<nvidia::sharedmemory::AllocationMapping> mapping_;
    };

    class AccessGuardConst : public Blob::AccessGuardConst
    {
    public:
        AccessGuardConst(Blob *blob);

        const void *GetData() override;

    private:
        std::shared_ptr<nvidia::sharedmemory::AllocationLock> lock_;
        std::shared_ptr<nvidia::sharedmemory::AllocationMapping> mapping_;
    };

    const std::shared_ptr<nvidia::sharedmemory::Context> context_;
    const std::shared_ptr<nvidia::sharedmemory::Allocation> allocation_;
};

} // namespace clara::viz
