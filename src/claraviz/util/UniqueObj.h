/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <functional>

#include "claraviz/util/Types.h"

namespace clara::viz
{

/**
 * A RAII-style class holding an pointer to an object and calling a function when control flow leaves the scope.
 * Typically the function to be called is to free the object.
 * Supports std::unique_ptr functionality.
 *
 * @tparam T type to be used
 * @tparam TF signature of the function to be called
 * @tparam F function to be called
 */
template<typename T, typename TF, TF F>
class UniqueObj : public std::unique_ptr<T, TF>
{
public:
    /**
     * Construct
     */
    UniqueObj()
        : std::unique_ptr<T, TF>(nullptr, F)
    {
    }

    /**
     * Construct from object
     *
     * @param obj [in] object
     */
    explicit UniqueObj(T *obj)
        : std::unique_ptr<T, TF>(obj, F)
    {
    }
};

/**
 * Similar to UniqueObj but stores a value (e.g. handle) and not a pointer.
 * Supports std::unique_ptr functionality.
 *
 * @tparam T type to be used
 * @tparam TF signature of the function to be called
 * @tparam F function to be called
 */
template<typename T, typename TF, TF F>
class UniqueValue : public NonCopyable
{
public:
    /**
     * Construct
     */
    UniqueValue()
        : value_(T())
    {
    }
    /**
     * Construct from value
     *
     * @param value [in] value
     */
    explicit UniqueValue(T value)
        : value_(value)
    {
    }
    ~UniqueValue()
    {
        reset();
    }

    /**
     * Release the value
     *
     * @returns value
     */
    T release() noexcept
    {
        T value = value_;
        value_  = T();
        return value;
    }

    /**
     * Reset with new value. Previous will be destroyed.
     *
     * @param value [in] new value
     */
    void reset(T value = T()) noexcept
    {
        T old_value = value_;
        value_      = value;
        if (old_value != T())
        {
            F(old_value);
        }
    }

    /**
     * Swap
     */
    void swap(UniqueValue &other) noexcept
    {
        std::swap(value_, other.value_);
    }

    /**
     * @return the value
     */
    T get() const noexcept
    {
        return value_;
    }

    /**
     * @returns true if the value is set
     */
    explicit operator bool() const noexcept
    {
        return (value_ != T());
    }

    /**
     * @returns reference to value
     */
    T &operator*() const
    {
        return value_;
    }

    /**
     * @returns value
     */
    T operator->() const noexcept
    {
        return value_;
    }

    /**
     * @returns true if equal
     */
    bool operator==(const UniqueValue &other) const
    {
        return (value_ == other.value_);
    }

    /**
     * @returns true if not equal
     */
    bool operator!=(const UniqueValue &other) const
    {
        return !(operator==(other));
    }

private:
    T value_;
};

/**
 * The Guard class calls a function when it goes out of scope. Typically the function to be called is
 * to undo a previous action in case of an error. Useful when e.g. mapping/unmapping memory where
 * the memory should just be unmapped and not be deleted.
 */
class Guard : public NonCopyable
{
public:
    /**
     * Construct
     *
     * @param function [in] function to call when going out of scope
     */
    Guard(std::function<void()> function)
        : function_(function)
    {
    }
    ~Guard()
    {
        function_();
    }

private:
    std::function<void()> function_;
};

} // namespace clara::viz
