/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/Exception.h"
#include "claraviz/util/Observed.h"
#include "claraviz/util/Types.h"

namespace clara::viz
{

namespace detail
{

/**
 * This class used when Observed is disabled
 */
class NotObserved : public NonCopyable
{
public:
    NotObserved() = default;
    virtual ~NotObserved() {}

protected:
    virtual void NotifyObservers() const
    {
        throw RuntimeError();
    }
};

} // namespace detail

/**
 * Value class.
 * Holds a value. Can check if the value is valid when it's set.
 * If enabled notify observers when the value change.
 *
 * @tparam VALUE_TYPE type of the value to hold
 * @tparam OBSERVED if `Observed` the value can be observed
 */
template<typename VALUE_TYPE, typename OBSERVED = detail::NotObserved>
class Value : public OBSERVED
{
public:
    /**
     * Construct, validation is disabled
     *
     * @param value [in] initial value
     */
    explicit Value(const VALUE_TYPE &value)
        : value_(value)
        , validator_([](const VALUE_TYPE &) {})
    {
    }
    /**
     * Construct with a validation function.
     *
     * @param value [in] initial value
     * @param validator [in] validation function, should throw if the value is invalid
     */
    Value(const VALUE_TYPE &value, std::function<void(const VALUE_TYPE &)> validator)
        : value_(value)
        , validator_(validator)
    {
    }
    Value() = default;

    /**
     * Set the value. Check if the new value is valid, notify observers if the new value is
     * different from the old value.
     *
     * @param value [in] new value
     */
    template<typename U = VALUE_TYPE,
             class      = typename std::enable_if<std::is_same<OBSERVED, detail::NotObserved>::value, U>::type>
    void Set(const VALUE_TYPE &value)
    {
        validator_(value);
        value_ = value;
    }

    /**
     * Set the value. Check if the new value is valid, notify observers if the new value is
     * different from the old value. Specialization when Value is observed.
     *
     * @param value [in] new value
     * @param force_notify [in] notify observers even if the value has not changed
     */
    template<typename U = VALUE_TYPE, class = typename std::enable_if<std::is_same<OBSERVED, Observed>::value, U>::type>
    void Set(const VALUE_TYPE &value, bool force_notify = false)
    {
        validator_(value);
        if (!(value == value_) || force_notify)
        {
            value_ = value;
            OBSERVED::NotifyObservers();
        }
    }

    /**
     * Get the value.
     */
    const VALUE_TYPE &Get() const
    {
        return value_;
    }

private:
    VALUE_TYPE value_;
    std::function<void(const VALUE_TYPE &)> validator_;
};

/**
 * A value which can be observed.
 *
 * @tparam VALUE_TYPE type of the value to hold
 */
template<typename VALUE_TYPE>
using ValueObserved = Value<VALUE_TYPE, Observed>;

} // namespace clara::viz
