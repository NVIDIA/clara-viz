/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <cstdint>
#include <type_traits>

#ifndef __CUDACC__
#include <iostream>
#include <string>
#endif
#include <vector_functions.h>

namespace clara::viz
{

#ifdef __CUDACC__
#define CUDA_FUNC __host__ __device__
#else
#define CUDA_FUNC
#endif

/**
 * Templated vector class
 */
template<typename T, uint32_t COMPONENTS>
class VectorT
{
public:
    /**
     * Construct (not initialized)
     */
    VectorT() = default;

    /**
     * Construct and set to values, two component vector
     */
    template<typename U = T, class = typename std::enable_if<COMPONENTS == 2, U>::type>
    CUDA_FUNC explicit VectorT(T x, T y)
    {
        values_[0] = x;
        values_[1] = y;
    }

    /**
     * Construct and set to values, three component vector
     */
    template<typename U = T, class = typename std::enable_if<COMPONENTS == 3, U>::type>
    CUDA_FUNC explicit VectorT(T x, T y, T z)
    {
        values_[0] = x;
        values_[1] = y;
        values_[2] = z;
    }

    /**
     * Construct and set to values, four component vector
     */
    template<typename U = T, class = typename std::enable_if<COMPONENTS == 4, U>::type>
    CUDA_FUNC explicit VectorT(T x, T y, T z, T w)
    {
        values_[0] = x;
        values_[1] = y;
        values_[2] = z;
        values_[3] = w;
    }
    /**
     * Construct from different vector type
     */
    template<typename U>
    CUDA_FUNC explicit VectorT(const VectorT<U, COMPONENTS> &other)
    {
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            values_[col] = static_cast<T>(other(col));
        }
    }

    /**
     * Construct from Cuda type 'uint3'
     */
    template<typename U = T, class = typename std::enable_if<(COMPONENTS == 3) && std::is_same<U, uint32_t>(), U>::type>
    explicit VectorT(const uint3 &from)
    {
        values_[0] = from.x;
        values_[1] = from.y;
        values_[2] = from.z;
    }

    /**
     * Construct and set to value
     */
    CUDA_FUNC explicit VectorT(T value)
    {
        set(value);
    }

    /**
     * Convert to Cuda type 'uint2'
     */
    template<typename U = T, class = typename std::enable_if<(COMPONENTS == 2) && std::is_same<U, uint32_t>(), U>::type>
    explicit operator uint2() const
    {
        return make_uint2(values_[0], values_[1]);
    }

    /**
     * Convert to Cuda type 'uint3'
     */
    template<typename U = T, class = typename std::enable_if<(COMPONENTS == 3) && std::is_same<U, uint32_t>(), U>::type>
    explicit operator uint3() const
    {
        return make_uint3(values_[0], values_[1], values_[2]);
    }

    /**
     * Convert to Cuda type 'float2'
     */
    template<typename U = T, class = typename std::enable_if<(COMPONENTS == 2) && std::is_same<U, float>(), U>::type>
    explicit operator float2() const
    {
        return make_float2(values_[0], values_[1]);
    }

    /**
     * Convert to Cuda type 'float3'
     */
    template<typename U = T, class = typename std::enable_if<(COMPONENTS == 3) && std::is_same<U, float>(), U>::type>
    explicit operator float3() const
    {
        return make_float3(values_[0], values_[1], values_[2]);
    }

    /**
     * Convert to Cuda type 'float4'
     */
    template<typename U = T, class = typename std::enable_if<(COMPONENTS == 4) && std::is_same<U, float>(), U>::type>
    explicit operator float4() const
    {
        return make_float4(values_[0], values_[1], values_[2], values_[3]);
    }

    /**
     * Set to value
     */
    CUDA_FUNC void set(T value)
    {
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            values_[col] = value;
        }
    }

    /**
     * Access operator
     */
    CUDA_FUNC T &operator()(uint32_t col)
    {
        assert(col < COMPONENTS);
        return values_[col];
    }

    /**
     * Access operator (const)
     */
    CUDA_FUNC const T &operator()(uint32_t col) const
    {
        assert(col < COMPONENTS);
        return values_[col];
    }

    /**
     * @returns true if all components of this == rhs
     */
    bool operator==(const VectorT &rhs) const
    {
        bool result = true;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result &= (values_[col] == rhs.values_[col]);
        }
        return result;
    }

    /**
     * @returns true if all components of this != rhs
     */
    bool operator!=(const VectorT &rhs) const
    {
        bool result = true;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result &= (values_[col] != rhs.values_[col]);
        }
        return result;
    }

    /**
     * @returns true if all components of this < rhs
     */
    bool operator<(const VectorT &rhs) const
    {
        bool result = true;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result &= (values_[col] < rhs.values_[col]);
        }
        return result;
    }

    /**
     * @returns true if all components of this >= rhs
     */
    bool operator>=(const VectorT &rhs) const
    {
        bool result = true;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result &= (values_[col] >= rhs.values_[col]);
        }
        return result;
    }

    /**
     * @returns true if all components of this > rhs
     */
    bool operator>(const VectorT &rhs) const
    {
        bool result = true;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result &= (values_[col] > rhs.values_[col]);
        }
        return result;
    }

    /**
     * @returns true if all components of this <= rhs
     */
    bool operator<=(const VectorT &rhs) const
    {
        bool result = true;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result &= (values_[col] <= rhs.values_[col]);
        }
        return result;
    }

    /**
     * Add
     */
    VectorT operator+(const VectorT &b) const
    {
        VectorT result;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result.values_[col] = values_[col] + b.values_[col];
        }
        return result;
    }

    /**
     * Add
     */
    void operator+=(const VectorT &b)
    {
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            values_[col] += b.values_[col];
        }
    }

    /**
     * Subtract
     */
    VectorT operator-(const VectorT &b) const
    {
        VectorT result;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result.values_[col] = values_[col] - b.values_[col];
        }
        return result;
    }

    /**
     * Subtract
     */
    void operator-=(const VectorT &b)
    {
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            values_[col] -= b.values_[col];
        }
    }

    /**
     * Multiply
     */
    VectorT operator*(const VectorT &b) const
    {
        VectorT result;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result.values_[col] = values_[col] * b.values_[col];
        }
        return result;
    }

    /**
     * Multiply
     */
    void operator*=(const VectorT &b)
    {
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            values_[col] *= b.values_[col];
        }
    }

    /**
     * Divide
     */
    VectorT operator/(const VectorT &b) const
    {
        VectorT result;
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            result.values_[col] = values_[col] / b.values_[col];
        }
        return result;
    }

    /**
     * Divide
     */
    void operator/=(const VectorT &b)
    {
        for (uint32_t col = 0; col < COMPONENTS; ++col)
        {
            values_[col] /= b.values_[col];
        }
    }

    static const uint32_t kComponents = COMPONENTS; ///< component count

private:
    T values_[COMPONENTS];
};

/**
 * Multiply single value with vector.
 */
template<typename T, uint32_t COMPONENTS>
inline VectorT<T, COMPONENTS> operator*(T a, const VectorT<T, COMPONENTS> &b)
{
    VectorT<T, COMPONENTS> result;
    for (uint32_t col = 0; col < COMPONENTS; ++col)
    {
        result(col) = a * b(col);
    }
    return result;
}

#ifndef __CUDACC__
/**
 * Operator that appends the string representation of a Vector to a stream.
 */
template<typename T, uint32_t COMPONENTS>
std::ostream &operator<<(std::ostream &os, const VectorT<T, COMPONENTS> &v)
{
    std::string s("(");
    for (uint32_t col = 0; col < COMPONENTS; ++col)
    {
        s += std::to_string(v(col));
        if (col == COMPONENTS - 1)
            s += ")";
        else
            s += ", ";
    }
    os << s;
    return os;
}
#endif // !__CUDACC__

// define some commonly used vector types
typedef VectorT<float, 2> Vector2f;
typedef VectorT<float, 3> Vector3f;
typedef VectorT<float, 4> Vector4f;

typedef VectorT<int8_t, 2> Vector2c;
typedef VectorT<int8_t, 3> Vector3c;
typedef VectorT<int8_t, 4> Vector4c;

typedef VectorT<uint8_t, 2> Vector2uc;
typedef VectorT<uint8_t, 3> Vector3uc;
typedef VectorT<uint8_t, 4> Vector4uc;

typedef VectorT<int16_t, 2> Vector2s;
typedef VectorT<int16_t, 3> Vector3s;
typedef VectorT<int16_t, 4> Vector4s;

typedef VectorT<uint16_t, 2> Vector2us;
typedef VectorT<uint16_t, 3> Vector3us;
typedef VectorT<uint16_t, 4> Vector4us;

typedef VectorT<int32_t, 2> Vector2i;
typedef VectorT<int32_t, 3> Vector3i;
typedef VectorT<int32_t, 4> Vector4i;

typedef VectorT<uint32_t, 2> Vector2ui;
typedef VectorT<uint32_t, 3> Vector3ui;
typedef VectorT<uint32_t, 4> Vector4ui;

#undef CUDA_FUNC

} // namespace clara::viz
