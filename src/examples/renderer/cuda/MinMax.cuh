/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <limits>

#include <cooperative_groups.h>

namespace clara::viz
{

namespace
{

/**
 * Utility class used to avoid linker errors with extern
 * unsized shared memory arrays with templated type
 */
template<class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

/**
 * @returns the minimum of the given values
 */
template<class T>
__device__ inline T Min(const T &lhs, const T &rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

/**
 * @returns the maximum of the given values
 */
template<class T>
__device__ inline T Max(const T &lhs, const T &rhs)
{
    return (lhs < rhs) ? rhs : lhs;
}

/**
 * Atomically update the content of address with the minimum of the content and the given value.
 */
template<typename T>
__device__ inline void AtomicMin(T *address, T value)
{
    atomicMin(address, value);
}

/**
 * Atomically update the content of address with the minimum of the content and the given value, float specialization.
 */
template<>
__device__ inline void AtomicMin<float>(float *address, float value)
{
    // we have integer atomic operations only, for float have to do the trick to use the opposite
    // operation for negative values
    if (value >= 0.f)
    {
        atomicMin(reinterpret_cast<unsigned int *>(address), __float_as_int(value));
    }
    else
    {
        atomicMax(reinterpret_cast<unsigned int *>(address), __float_as_uint(value));
    }
}

/**
 * Atomically update the content of address with the minimum of the content and the given value.
 */
template<typename T>
__device__ inline void AtomicMax(T *address, T value)
{
    atomicMax(address, value);
}

/**
 * Atomically update the content of address with the minimum of the content and the given value, float specialization.
 */
template<>
__device__ inline void AtomicMax<float>(float *address, float value)
{
    // we have integer atomic operations only, for float have to do the trick to use the opposite
    // operation for negative values
    if (value >= 0.f)
    {
        atomicMax(reinterpret_cast<unsigned int *>(address), __float_as_int(value));
    }
    else
    {
        atomicMin(reinterpret_cast<unsigned int *>(address), __float_as_uint(value));
    }
}

} // anonymous namespace

/**
 * Determine the minimum and maximum value of a volume.
 *
 * Based on Cuda reduce sample.
 *
 * First works on multiple elements per thread sequentially, uses shuffle to propagate the values
 * within a warp and finally atomics to get the result.
 *
 * Note, this kernel needs a minimum of 32 * sizeof(T) bytes of shared memory.
 * In other words if threads <= 32, allocate 32 * sizeof(T) bytes.
 * If threads > 32, allocate threads * sizeof(T) bytes.
 *
 * @tparam T data type
 * @tprarm threads threads per row
 *
 * @param texture [in] volume texture
 * @param size [in] size of the volume texture
 * @param buffer_min_max [in]
 */
template<class T, unsigned int threads>
__global__ void MinMax(CUtexObject texture, const int3 size, T *buffer_min_max)
{
    const uint3 index = make_uint3(blockIdx.x * threads + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
                                   blockIdx.z * blockDim.z + threadIdx.z);

    // Handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    const unsigned int tid       = threadIdx.x;
    const unsigned int grid_size = threads * gridDim.x;

    T minimum = std::numeric_limits<T>::max();
    T maximum = std::numeric_limits<T>::min();

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim). More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    uint3 cur_index = index;
    while (cur_index.x < size.x)
    {
        const T value = tex3D<T>(texture, cur_index.x, cur_index.y, cur_index.z);
        cur_index.x += grid_size;

        minimum = Min(minimum, value);
        maximum = Max(maximum, value);
    }

    // each thread puts its local sum into shared memory
    sdata[tid * 2 + 0] = minimum;
    sdata[tid * 2 + 1] = maximum;

    cooperative_groups::sync(cta);

    constexpr auto threads_per_warp = 32u;
    if (cta.thread_rank() < threads_per_warp)
    {
        cooperative_groups::thread_block_tile<threads_per_warp> tile =
            cooperative_groups::tiled_partition<threads_per_warp>(cta);

        // Reduce final warp using shuffle
        for (int offset = tile.size() / 2; offset > 0; offset /= 2)
        {
            minimum = Min(minimum, tile.shfl_down(minimum, offset));
            maximum = Max(maximum, tile.shfl_down(maximum, offset));
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0)
    {
        AtomicMin(&buffer_min_max[0], minimum);
        AtomicMax(&buffer_min_max[1], maximum);
    }
}

} // namespace clara::viz
