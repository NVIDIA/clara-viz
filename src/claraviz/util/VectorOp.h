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

#pragma once

#include <ostream>
#include <iterator>
#include <vector>

namespace clara::viz
{

/**
 * Output vector
 **/
template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
    return os;
}

/**
 * Add vectors
 */
template<typename T>
std::vector<T> operator+(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if (v1.size() != v2.size())
    {
        throw InvalidArgument("v2") << "vector size mismatch";
    }

    std::vector<T> result(v1.size());

    for (size_t index = 0; index < v1.size(); ++index)
    {
        result[index] = v1[index] + v2[index];
    }

    return result;
}

/**
 * Subtract vectors
 */
template<typename T>
std::vector<T> operator-(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if (v1.size() != v2.size())
    {
        throw InvalidArgument("v2") << "vector size mismatch";
    }

    std::vector<T> result(v1.size());

    for (size_t index = 0; index < v1.size(); ++index)
    {
        result[index] = v1[index] - v2[index];
    }

    return result;
}

/**
 * Add to vector
 */
template<typename T>
std::vector<T> operator+=(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if (v1.size() != v2.size())
    {
        throw InvalidArgument("v2") << "vector size mismatch";
    }

    std::vector<T> result;

    result.reserve(v1.size());

    for (size_t index = 0; index < v1.size(); ++index)
    {
        result[index] = v1[index] + v2[index];
    }

    return result;
}

} // namespace clara::viz