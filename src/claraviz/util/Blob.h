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
#include <mutex>
#include <shared_mutex>

#include <claraviz/util/Exception.h>
#include <claraviz/util/Types.h>

/// forward declaration of Cuda stream to avoid including cuda.h
typedef struct CUstream_st *CUstream;

namespace clara::viz
{

/**
 * Interface class for BLOB objects (Binary Large OBject)
 */
class IBlob
{
public:
    IBlob()          = default;
    virtual ~IBlob() = default;

    /**
     * Access guard for a blob for reading and writing. The data is only guaranteed to be accessible
     * while the guard exists.
     */
    class AccessGuard : public NonCopyable
    {
    public:
        AccessGuard()          = default;
        virtual ~AccessGuard() = default;

        /**
         * @returns a pointer to the data
         */
        virtual void *GetData() = 0;
    };

    /**
     * Access the data from host (CPU) for reading and writing. Only one client at a time is allowed to access
     * the memory for writing. Access is synchronized with device (GPU) accesses issued before.
     */
    virtual std::unique_ptr<AccessGuard> Access() = 0;

    /**
     * Access the data from device (GPU) for reading and writing. Only one client at a time is allowed to access
     * the memory for writing. Access is synchronized with both host (CPU) and device (GPU) accesses issued before.
     *
     * @param stream [in] the Cuda stream which will be used to access the data, has to be valid as long as the access
     *                    guard exists
     */
    virtual std::unique_ptr<AccessGuard> Access(CUstream stream) = 0;

    /**
     * Read only access guard for a blob. The data is only guaranteed to be accessible
     * while the guard exists.
     */
    class AccessGuardConst : public NonCopyable
    {
    public:
        AccessGuardConst()          = default;
        virtual ~AccessGuardConst() = default;

        /**
         * @returns a const pointer to the data
         */
        virtual const void *GetData() = 0;
    };

    /**
     * Access the data from host (CPU) for reading only. Multiple clients are allowed to access
     * the memory for reading. Access is synchronized with device (GPU) accesses issued before.
     */
    virtual std::unique_ptr<AccessGuardConst> AccessConst() = 0;

    /**
     * Access the data from device (GPU) for reading only. Multiple clients are allowed to access
     * the memory for reading. Access is synchronized with device (GPU) accesses issued before.
     *
     * @param stream [in] the Cuda stream which will be used to access the data, has to be valid as long as the access
     *                    guard exists
     */
    virtual std::unique_ptr<AccessGuardConst> AccessConst(CUstream stream) = 0;

    /**
     * @returns the size of the stored data in bytes
     */
    virtual size_t GetSize() const = 0;
};

/**
 * Base class for BLOB objects (Binary Large OBject).
 * Handles host (CPU) and device (GPU) synchronization.
 */
class Blob : public IBlob
{
public:
    Blob();
    virtual ~Blob() = default;

    /**
     * Access guard for a blob for reading and writing. The data is only guaranteed to be accessible
     * while the guard exists.
     */
    class AccessGuard : public IBlob::AccessGuard
    {
    public:
        /**
         * Construct
         *
         * @param blob [in] blob to access
         */
        AccessGuard(Blob *blob)
            : blob_(blob)
            , valid_stream_(false)
            , stream_(0)
        {
        }
        AccessGuard() = delete;
        virtual ~AccessGuard();

    protected:
        Blob *blob_; ///< blob which is accessed

    private:
        friend Blob;

        std::unique_lock<std::shared_mutex> lock_;
        bool valid_stream_;
        CUstream stream_;
    };

    /**
     * Read only access guard for a blob. The data is only guaranteed to be accessible
     * while the guard exists.
     */
    class AccessGuardConst : public IBlob::AccessGuardConst
    {
    public:
        /**
         * Construct
         *
         * @param blob [in] blob to access
         */
        AccessGuardConst(Blob *blob)
            : blob_(blob)
            , valid_stream_(false)
            , stream_(0)
        {
        }
        AccessGuardConst() = delete;
        virtual ~AccessGuardConst();

        virtual const void *GetData() = 0;

    protected:
        Blob *blob_; ///< blob which is accessed

    private:
        friend Blob;

        std::shared_lock<std::shared_mutex> lock_;
        bool valid_stream_;
        CUstream stream_;
    };

protected:
    /**
     * Start synchronized read-write CPU access.
     * This takes the lock and synchronizes with any previous
     * CUDA accesses.
     *
     * @param guard [in] guard to use
     */
    void SyncAccess(AccessGuard *guard);

    /**
     * Start synchronized read-write GPU access with 'stream'.
     * This takes the lock and synchronizes with any previous
     * CUDA accesses.
     *
     * @param guard [in] guard to use
     * @param stream [in] Cuda stream
     */
    void SyncAccess(AccessGuard *guard, CUstream stream);

    /**
     * Start synchronized read-only CPU access.
     * This takes the lock and synchronizes with any previous
     * CUDA accesses.
     *
     * @param guard [in] guard to use
     */
    void SyncAccessConst(AccessGuardConst *guard);

    /**
     * Start synchronized read-only GPU access with 'stream'.
     * This takes the lock and synchronizes with any previous
     * CUDA accesses.
     *
     * @param guard [in] guard to use
     * @param stream [in] Cuda stream
     */
    void SyncAccessConst(AccessGuardConst *guard, CUstream stream);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::viz