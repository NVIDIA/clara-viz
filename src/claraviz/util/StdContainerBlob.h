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
#include <type_traits>

#include "claraviz/util/Blob.h"
#include "claraviz/util/Exception.h"

namespace clara::viz
{

/**
 * Blob to be used by std containers (e.g. std::vector, std::string)
 */
template<typename T>
class StdContainerBlob : public Blob
{
public:
    /**
     * Construct
     *
     * @param memory [in] std container to store, object is moved to class
     */
    explicit StdContainerBlob(std::unique_ptr<T> &&memory)
        : memory_(std::move(memory))
    {
    }
    StdContainerBlob() = delete;

    std::unique_ptr<IBlob::AccessGuard> Access() override
    {
        if (std::is_const<T>::value)
        {
            throw RuntimeError() << "Write access not allowed on Blob holding type " << typeid(T).name();
        }
        else
        {
            std::unique_ptr<Blob::AccessGuard> guard(new AccessGuard(this));
            Blob::SyncAccess(guard.get());
            return guard;
        }
    }

    std::unique_ptr<IBlob::AccessGuard> Access(CUstream stream) override
    {
        if (std::is_const<T>::value)
        {
            throw RuntimeError() << "Write access not allowed on Blob holding type " << typeid(T).name();
        }
        else
        {
            std::unique_ptr<Blob::AccessGuard> guard(new AccessGuard(this));
            Blob::SyncAccess(guard.get(), stream);
            return guard;
        }
    }

    std::unique_ptr<IBlob::AccessGuardConst> AccessConst() override
    {
        std::unique_ptr<Blob::AccessGuardConst> guard(new AccessGuardConst(this));
        Blob::SyncAccessConst(guard.get());
        return guard;
    }

    std::unique_ptr<IBlob::AccessGuardConst> AccessConst(CUstream stream) override
    {
        std::unique_ptr<Blob::AccessGuardConst> guard(new AccessGuardConst(this));
        Blob::SyncAccessConst(guard.get(), stream);
        return guard;
    }

    size_t GetSize() const override
    {
        return memory_->size() * sizeof(typename T::value_type);
    }

private:
    class AccessGuard : public Blob::AccessGuard
    {
    public:
        AccessGuard(Blob *blob)
            : Blob::AccessGuard(blob)
        {
        }

        void *GetData() override
        {
            return reinterpret_cast<void *>(
                const_cast<typename T::value_type *>(static_cast<StdContainerBlob<T> *>(blob_)->memory_->data()));
        }
    };

    class AccessGuardConst : public Blob::AccessGuardConst
    {
    public:
        AccessGuardConst(Blob *blob)
            : Blob::AccessGuardConst(blob)
        {
        }

        const void *GetData() override
        {
            return reinterpret_cast<const void *>(static_cast<StdContainerBlob<T> *>(blob_)->memory_->data());
        }
    };

    const std::unique_ptr<T> memory_;
};

} // namespace clara::viz
