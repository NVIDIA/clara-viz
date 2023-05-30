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

#include <cmath>
#include <limits>

#include "claraviz/util/Exception.h"
#include "claraviz/util/VectorT.h"

namespace clara::viz
{

/**
 * Enforce that a value is between 'min' and 'max' (exclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (exclusive)
 * @param max [in] maximum value (exclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMinMaxExclusive(const TYPE &value, const TYPE &min, const TYPE &max, const char *name)
{
    if (!((value > min) && (value < max)))
    {
        throw InvalidArgument(name) << "expected to be > " << min << " and < " << max << " but is " << value;
    }
}

/**
 * Enforce that a value is between 'min' and 'max' (inclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (inclusive)
 * @param max [in] maximum value (inclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMinMaxInclusive(const TYPE &value, const TYPE &min, const TYPE &max, const char *name)
{
    if (!((value >= min) && (value <= max)))
    {
        throw InvalidArgument(name) << "expected to be >= " << min << " and <= " << max << " but is " << value;
    }
}

/**
 * Enforce that a value is between 'min' (exclusive) and 'max' (inclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (exclusive)
 * @param max [in] maximum value (inclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMinExclusiveMaxInclusive(const TYPE &value, const TYPE &min, const TYPE &max, const char *name)
{
    if (!((value > min) && (value <= max)))
    {
        throw InvalidArgument(name) << "expected to be > " << min << " and <= " << max << " but is " << value;
    }
}

/**
 * Enforce that a value is between 'min' (inclusive) and 'max' (exclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (inclusive)
 * @param max [in] maximum value (exclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMinInclusiveMaxExclusive(const TYPE &value, const TYPE &min, const TYPE &max, const char *name)
{
    if (!((value >= min) && (value < max)))
    {
        throw InvalidArgument(name) << "expected to be >= " << min << " and < " << max << " but is " << value;
    }
}

/**
 * Enforce that a value is higher than 'min' (exclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (exclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMinExclusive(const TYPE &value, const TYPE &min, const char *name)
{
    if (!(value > min))
    {
        throw InvalidArgument(name) << "is expected to be > " << min << " but is " << value;
    }
}

/**
 * Enforce that a value is higher than 'min' (inclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (inclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMinInclusive(const TYPE &value, const TYPE &min, const char *name)
{
    if (!(value >= min))
    {
        throw InvalidArgument(name) << "is expected to be >= " << min << " but is " << value;
    }
}

/**
 * Enforce that a value is lower than 'max' (exclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param max [in] minimum value (exclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorMaxExclusive(const TYPE &value, const TYPE &max, const char *name)
{
    if (!(value < max))
    {
        throw InvalidArgument(name) << "is expected to be < " << max << " but is " << value;
    }
}

/**
 * Enforce that a vector is a unit vector
 *
 * @tparam VECTOR value type
 *
 * @param vector [in] vector to check
 * @param name [in] value name
 */
template<typename VECTOR>
void ValidatorUnitVector(const VECTOR &vector, const char *name)
{
    uint32_t index = 0;
    auto sum       = vector(index) * vector(index);
    ++index;
    while (index < VECTOR::kComponents)
    {
        sum += vector(index) * vector(index);
        ++index;
    }

    // compare to float epsilon
    if (std::abs(sum - 1.f) > std::numeric_limits<float>::epsilon() * 10.f)
    {
        throw InvalidArgument(name) << "is expected to be an unit vector";
    }
}

/**
 * Enforce that two values are not identical
 *
 * @tparam TYPE value type
 *
 * @param first [in] first value
 * @param second [in] second value
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorDifferent(const TYPE &first, const TYPE &second, const char *name)
{
    if (first == second)
    {
        throw InvalidArgument(name) << "are expected to be not identical";
    }
}

/**
 * Enforce that a range is valid. That is 'min' < 'max' and range values are between 'min' and 'max' (inclusive)
 *
 * @tparam TYPE value type
 *
 * @param value [in] value to check
 * @param min [in] minimum value (inclusive)
 * @param max [in] maximum value (inclusive)
 * @param name [in] value name
 */
template<typename TYPE>
void ValidatorRange(const VectorT<TYPE, 2> &value, const TYPE &min, const TYPE &max, const char *name)
{
    if (!((value(0) < value(1)) && (value(0) >= min) && (value(1) <= max)))
    {
        throw InvalidArgument(name) << "minimum expected to be >= " << min << " and maximum <= " << max
                                    << ", invalid range is " << value;
    }
}

} // namespace clara::viz
