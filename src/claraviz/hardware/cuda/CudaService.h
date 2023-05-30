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

#include <algorithm>
#include <chrono>
#include <memory>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_functions.h>
#include <math_constants.h>

#include "claraviz/util/Exception.h"
#include "claraviz/util/Log.h"
#include "claraviz/util/UniqueObj.h"
#include "claraviz/util/VectorT.h"

namespace clara::viz
{

/**
 * Cuda driver API error check helper
 */
#define CudaCheck(FUNC)                                                                         \
    {                                                                                           \
        const CUresult result = FUNC;                                                           \
        if (result != CUDA_SUCCESS)                                                             \
        {                                                                                       \
            const char *error_name = "";                                                        \
            cuGetErrorName(result, &error_name);                                                \
            const char *error_string = "";                                                      \
            cuGetErrorString(result, &error_string);                                            \
            throw RuntimeError() << "Cuda driver error " << error_name << ": " << error_string; \
        }                                                                                       \
    }

/**
 * Cuda runtime API error check helper
 */
#define CudaRTCheck(FUNC)                                                                     \
    {                                                                                         \
        const cudaError_t result = FUNC;                                                      \
        if (result != cudaSuccess)                                                            \
        {                                                                                     \
            throw RuntimeError() << "Cuda runtime error " << cudaGetErrorName(result) << ": " \
                                 << cudaGetErrorString(result);                               \
        }                                                                                     \
    }

/**
 * UniqueObj's for a Cuda objects
 */
/**@{*/
using UniqueCudaArray   = UniqueObj<std::remove_pointer<CUarray>::type, decltype(&cuArrayDestroy), &cuArrayDestroy>;
using UniqueCudaContext = UniqueObj<std::remove_pointer<CUcontext>::type, decltype(&cuCtxDestroy), &cuCtxDestroy>;
using UniqueCudaEvent   = UniqueObj<std::remove_pointer<CUevent>::type, decltype(&cuEventDestroy), &cuEventDestroy>;
using UniqueCudaMemory  = UniqueValue<CUdeviceptr, decltype(&cuMemFree), &cuMemFree>;
using UniqueCudaSurface = UniqueValue<CUsurfObject, decltype(&cuSurfObjectDestroy), &cuSurfObjectDestroy>;
using UniqueCudaTexture = UniqueValue<CUtexObject, decltype(&cuTexObjectDestroy), &cuTexObjectDestroy>;
/**@}*/

/**
 * Get the primary context and make it current
 */
class CudaPrimaryContext : public NonCopyable
{
public:
    /**
     * Construct
     *
     * @param device_ordinal [in] device to create the context on
     */
    explicit CudaPrimaryContext(uint32_t device_ordinal = 0)
        : device_(0)
    {
        CudaCheck(cuDeviceGet(&device_, device_ordinal));
        CudaCheck(cuDevicePrimaryCtxRetain(&context_, device_));
        CudaCheck(cuCtxPushCurrent(context_));
    }
    CudaPrimaryContext() = delete;

    ~CudaPrimaryContext()
    {
        try
        {
            CUcontext popped_context;
            CudaCheck(cuCtxPopCurrent(&popped_context));
            if (popped_context != context_)
            {
                Log(LogLevel::Error) << "Cuda: Unexpected context popped";
            }
            CudaCheck(cuDevicePrimaryCtxRelease(device_));
        }
        catch (...)
        {
            Log(LogLevel::Error) << "CudaPrimaryContext destructor failed";
        }
    }

    /**
     * @returns the Cuda context
     */
    CUcontext GetContext() const
    {
        return context_;
    }

private:
    CUdevice device_;
    CUcontext context_;
};

/**
 * Cuda context (not made current on creation)
 */
class CudaContext : public NonCopyable
{
public:
    /**
     * Construct
     *
     * @param device_ordinal [in] device to create the context on
     */
    explicit CudaContext(uint32_t device_ordinal = 0)
    {
        context_.reset([device_ordinal] {
            CUdevice device;
            CudaCheck(cuDeviceGet(&device, device_ordinal));
            CUcontext context;
            CudaCheck(cuCtxCreate(&context, 0, device));
            CUcontext popped_context;
            CudaCheck(cuCtxPopCurrent(&popped_context));
            if (popped_context != context)
            {
                throw InvalidState() << "Unexpected context popped";
            }
            return context;
        }());
    }
    CudaContext() = delete;

    /**
     * RAII type class to push a CUDA context.
     */
    class ScopedPush : public NonCopyable
    {
    public:
        /**
         * Construct
         *
         * @param cuda_context [in] CUDA context
         */
        ScopedPush(CudaContext &cuda_context)
            : context_(cuda_context.context_.get())
        {
            CudaCheck(cuCtxPushCurrent(context_));
        }

        /**
         * Construct
         *
         * @param cuda_context [in] CUDA context handle
         */
        ScopedPush(CUcontext cuda_context)
            : context_(cuda_context)
        {
            CudaCheck(cuCtxPushCurrent(context_));
        }

        ScopedPush() = delete;

        /**
         * Destruct. Context is popped from the stack.
         */
        ~ScopedPush()
        {
            try
            {
                CUcontext popped_context;
                CudaCheck(cuCtxPopCurrent(&popped_context));
                if (popped_context != context_)
                {
                    Log(LogLevel::Error) << "Cuda: Unexpected context popped";
                }
            }
            catch (...)
            {
                Log(LogLevel::Error) << "CudaPrimaryContext destructor failed";
            }
        }

    private:
        CUcontext context_;
    };

    /**
     * @returns the Cuda context
     */
    CUcontext GetContext() const
    {
        return context_.get();
    }

private:
    UniqueCudaContext context_;
};

/**
 * Cuda memory.
 */
class CudaMemory
{
public:
    /**
     * Construct
     *
     * @param size [in] size in bytes
     */
    explicit CudaMemory(size_t size)
        : size_(size)
    {
        memory_.reset([this] {
            CUdeviceptr memory;
            CudaCheck(cuMemAlloc(&memory, size_));
            return memory;
        }());
    }
    CudaMemory() = delete;

    /**
     * @returns the size
     */
    size_t GetSize() const
    {
        return size_;
    }

    /**
     * @returns the CUDA memory handle
     */
    const UniqueCudaMemory &GetMemory() const
    {
        return memory_;
    }

private:
    const size_t size_;

    UniqueCudaMemory memory_;
};

/**
 * Cuda 2D memory.
 * Has an embedded Cuda event to support recording and synchronizing access to the memory.
 */
class CudaMemory2D
{
public:
    /**
     * Construct
     *
     * @param width [in] width of the memory in elements
     * @param height [in] height of the memory
     * @param element_size [in] size of one element in bytes
     */
    explicit CudaMemory2D(uint32_t width, uint32_t height, uint32_t element_size)
        : CudaMemory2D(width, height, element_size, [this] {
            CUdeviceptr memory;
            pitch_ = width_ * element_size_;
            CudaCheck(cuMemAlloc(&memory, height_ * pitch_));
            memory_.reset(memory);
        })
    {
    }
    virtual ~CudaMemory2D()
    {
        // if this is an externally managed event avoid deleting it
        if (external_event_)
        {
            event_.release();
        }
    }

    /**
     * @returns the width in elements
     */
    uint32_t GetWidth() const
    {
        return width_;
    }

    /**
     * @returns the height
     */
    uint32_t GetHeight() const
    {
        return height_;
    }

    /**
     * @returns the element size in bytes
     */
    uint32_t GetElementSize() const
    {
        return element_size_;
    }

    /**
     * @returns the pitch of one line in bytes (distance between the start of one line and the subsequent line)
     */
    size_t GetPitch() const
    {
        return pitch_;
    }

    /**
     * @returns the memory
     */
    const UniqueCudaMemory &GetMemory() const
    {
        return memory_;
    }

    /**
     * Records an event, typically called when the memory had been accessed.
     *
     * @param stream [in] stream to record the event for
     */
    void EventRecord(CUstream stream)
    {
        // if there is no event yet, create one
        if (!event_)
        {
            event_.reset([this] {
                CUevent event;
                CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
                return event;
            }());
            external_event_ = false;
        }
        CudaCheck(cuEventRecord(event_.get(), stream));
    }

    /**
     * Wait on the recorded event to complete. This is a no-op if no event had been recorded.
     */
    void EventSynchronize()
    {
        // if no event had been recorded then there is nothing to do
        if (!event_)
        {
            return;
        }
        CudaCheck(cuEventSynchronize(event_.get()));
    }

    /**
     * Make a stream wait on the recorded event. This is a no-op if no event had been recorded.
     *
     * @param stream [in] stream to wait
     */
    void StreamWaitEvent(CUstream stream)
    {
        // if no event had been recorded then there is nothing to do
        if (!event_)
        {
            return;
        }

        CudaCheck(cuStreamWaitEvent(stream, event_.get(), /*flags*/ 0));
    }

    /**
     * Set the event to an external managed event. The life-time of that event is also managed externally.
     */
    void SetEvent(CUevent event)
    {
        if (external_event_)
        {
            event_.release();
        }
        event_.reset(event);
        external_event_ = true;
    }

protected:
    /**
     * Construct
     *
     * @param width [in] width of the memory in elements
     * @param height [in] height of the memory
     * @param element_size [in] size of one element in bytes
     * @param alloc [in] allocation function
     */
    CudaMemory2D(uint32_t width, uint32_t height, uint32_t element_size, std::function<void()> alloc)
        : width_(width)
        , height_(height)
        , element_size_(element_size)
        , pitch_(0)
        , external_event_(false)
    {
        alloc();
    }

    /// width in elements
    const uint32_t width_;
    /// height
    const uint32_t height_;
    /// element size in bytes
    const uint32_t element_size_;

    /// pitch of one line in bytes (distance between the start of one line and the subsequent line)
    size_t pitch_;

    /// CUDA memory handle
    UniqueCudaMemory memory_;

    /// optional event to synchronize to before accessing the resource
    UniqueCudaEvent event_;
    /// set if using an externally allocated event
    bool external_event_;
};

/**
 * Cuda pitch memory.
 */
class CudaMemoryPitch : public CudaMemory2D
{
public:
    /**
     * Construct
     *
     * @param width [in] width of the memory in elements
     * @param height [in] height of the memory
     * @param element_size [in] size of one element in bytes
     */
    explicit CudaMemoryPitch(uint32_t width, uint32_t height, uint32_t element_size)
        : CudaMemory2D(width, height, element_size, [this] {
            CUdeviceptr memory;
            CudaCheck(cuMemAllocPitch(&memory, &pitch_, width_ * element_size_, height_, element_size_));
            memory_.reset(memory);
        })
    {
    }
    CudaMemoryPitch() = delete;
};

/**
 * Cuda array.
 */
class CudaArray
{
public:
    /**
     * Construct 2D array
     *
     * @param size [in] array size
     * @param format [in] array format
     * @param channels [in] number of packed elements per array element
     */
    explicit CudaArray(const Vector2ui &size, CUarray_format format, uint32_t channels = 1)
        : size_(Vector3ui(size(0), size(1), 0))
        , format_(format)
    {
        array_.reset([this, channels] {
            CUarray array;
            CUDA_ARRAY_DESCRIPTOR desc{};
            desc.Width       = size_(0);
            desc.Height      = size_(1);
            desc.Format      = format_;
            desc.NumChannels = channels;
            CudaCheck(cuArrayCreate(&array, &desc));
            return array;
        }());
    }

    /**
     * Construct 3D array
     * An 1d layered array is allocated if height is zero. In this case size.z determines
     * the number of layers.
     *
     * @param size [in] array size
     * @param format [in] array format
     * @param channels [in] number of packed elements per array element
     * @param flags [in] flags
     */
    explicit CudaArray(const Vector3ui &size, CUarray_format format, uint32_t channels = 1, uint32_t flags = 0)
        : size_(size)
        , format_(format)
    {
        array_.reset([this, channels, flags] {
            CUarray array;
            CUDA_ARRAY3D_DESCRIPTOR desc{};
            desc.Width       = size_(0);
            desc.Height      = size_(1);
            desc.Depth       = size_(2);
            desc.Format      = format_;
            desc.NumChannels = channels;
            desc.Flags       = flags;
            if ((desc.Height == 0) && (desc.Depth != 0))
            {
                desc.Flags |= CUDA_ARRAY3D_LAYERED;
            }
            CudaCheck(cuArray3DCreate(&array, &desc));
            return array;
        }());
    }
    CudaArray() = delete;

    /**
     * @returns the array size
     */
    Vector3ui GetSize() const
    {
        return size_;
    }

    /**
     * @returns the array format
     */
    CUarray_format GetFormat() const
    {
        return format_;
    }

    /**
     * @returns the CUDA array handle
     */
    const UniqueCudaArray &GetArray() const
    {
        return array_;
    }

private:
    const Vector3ui size_;
    const CUarray_format format_;

    UniqueCudaArray array_;
};

/**
 * Cuda texture.
 */
class CudaTexture
{
public:
    /**
     * Construct from Cuda array
     *
     * @param array [in] Cuda array
     * @param filter_mode [in] texture filter mode
     * @param flags [in] texture flags (CU_TRSF_*)
     */
    explicit CudaTexture(const std::shared_ptr<CudaArray> &array, CUfilter_mode filter_mode = CU_TR_FILTER_MODE_POINT,
                         uint32_t flags = 0)
        : array_(array)
    {
        if (!array)
        {
            throw InvalidArgument("array") << "is nullptr";
        }
        texture_.reset([this, filter_mode, flags] {
            CUtexObject texture;
            CUDA_RESOURCE_DESC res_desc{};
            res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
            res_desc.res.array.hArray = array_->GetArray().get();

            CUDA_TEXTURE_DESC tex_desc{};
            tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.filterMode     = filter_mode;
            tex_desc.flags          = flags;

            CudaCheck(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
            return texture;
        }());
    }
    CudaTexture() = delete;

    /**
     * @returns the CUDA texture handle
     */
    const UniqueCudaTexture &GetTexture() const
    {
        return texture_;
    }

private:
    const std::shared_ptr<CudaArray> array_;
    UniqueCudaTexture texture_;
};

/**
 * Cuda surface.
 */
class CudaSurface
{
public:
    /**
     * Construct from Cuda array
     *
     * @param array [in] Cuda array
     */
    explicit CudaSurface(const std::shared_ptr<CudaArray> &array)
        : array_(array)
    {
        if (!array)
        {
            throw InvalidArgument("array") << "is nullptr";
        }
        surface_.reset([this] {
            CUsurfObject surface;
            CUDA_RESOURCE_DESC res_desc{};
            res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
            res_desc.res.array.hArray = array_->GetArray().get();

            CudaCheck(cuSurfObjectCreate(&surface, &res_desc));
            return surface;
        }());
    }
    CudaSurface() = delete;

    /**
     * @returns the CUDA surface handle
     */
    const UniqueCudaSurface &GetSurface() const
    {
        return surface_;
    }

private:
    const std::shared_ptr<CudaArray> array_;
    UniqueCudaSurface surface_;
};

/**
 * Launch a Cuda function. Calculates the optimal block size for max occupancy.
 */
class CudaFunctionLauncher
{
public:
    /**
     * Construct.
     *
     * @param function [in] cuda function
     * @param shared_mem_size [in] per-block dynamic shared memory usage intended, in bytes
     */
    template<class T>
    explicit CudaFunctionLauncher(T function, size_t shared_mem_size = 0)
        : function_(reinterpret_cast<CUfunction>(function))
        , shared_mem_size_(shared_mem_size)
        , calc_launch_grid_([this](const Vector3ui &grid) -> dim3 {
            dim3 launch_grid;
            // calculate the launch rid size
            launch_grid.x = (grid(0) + (block_dim_.x - 1)) / block_dim_.x;
            launch_grid.y = (grid(1) + (block_dim_.y - 1)) / block_dim_.y;
            launch_grid.z = (grid(2) + (block_dim_.z - 1)) / block_dim_.z;
            return launch_grid;
        })
        , optimal_block_size_(0)
    {
        // calculate the optimal block size for max occupancy
        int min_grid_size      = 0;
        int optimal_block_size = 0;
        CudaRTCheck(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size, function, shared_mem_size));
        SetOptimalBlockSize(optimal_block_size);
    }

    CudaFunctionLauncher() = delete;

    /**
     * Set the launch grid calculation function. This overwrites the default launch grid calculation function.
     *
     * @param calc_launch_grid [in] launch grid calculation function.
     */
    void SetCalcLaunchGrid(const std::function<dim3(const Vector3ui &grid)> &calc_launch_grid)
    {
        calc_launch_grid_ = calc_launch_grid;
    }

    /**
     * Set the optimal block size, this overwrites the optimal block dim calculated by the constructor
     *
     * @param block_size [in] block size
     */
    void SetOptimalBlockSize(int block_size)
    {
        if (block_size != optimal_block_size_)
        {
            optimal_block_size_ = block_size;
            // get a 2D block size from the optimal block size
            block_dim_.x = 1;
            block_dim_.y = 1;
            block_dim_.z = 1;
            while (static_cast<int>(block_dim_.x * block_dim_.y * 2) <= optimal_block_size_)
            {
                if (block_dim_.x > block_dim_.y)
                {
                    block_dim_.y *= 2;
                }
                else
                {
                    block_dim_.x *= 2;
                }
            }
        }
    }

    /**
     * @returns the optimal block size
     */
    int GetOptimalBlockSize() const
    {
        return optimal_block_size_;
    }

    /**
     * Set the block dim, this overwrites the optimal block dim calculated by the constructor
     *
     * @param block_dim [in] block dim
     */
    void SetBlockDim(const dim3 &block_dim)
    {
        block_dim_ = block_dim;
    }

    /**
     * Launch the kernel on a grid.
     *
     * @param grid [in] grid size
     * @param args [in] kernel arguments (optional)
     */
    template<class... TYPES>
    void Launch(const Vector2ui &grid, TYPES... args) const
    {
        const Vector3ui grid3(grid(0), grid(1), 1);
        const dim3 launch_grid = calc_launch_grid_(grid3);

        void *args_array[] = {reinterpret_cast<void *>(&args)...};
        CudaRTCheck(
            cudaLaunchKernel(function_, launch_grid, block_dim_, args_array, shared_mem_size_, cudaStreamPerThread));
#ifndef NDEBUG
        // check kernel runtime errors - note this is a blocking operation
        CudaRTCheck(cudaStreamSynchronize(cudaStreamPerThread));
#endif // !NDEBUG
    }

    /**
     * Launch the kernel on a grid with a stream.
     *
     * @param grid [in] grid size
     * @param stream [in] stream
     * @param args [in] kernel arguments (optional)
     */
    template<class... TYPES>
    void LaunchWithStream(const Vector2ui &grid, CUstream stream, TYPES... args) const
    {
        const Vector3ui grid3(grid(0), grid(1), 1);
        const dim3 launch_grid = calc_launch_grid_(grid3);

        void *args_array[] = {reinterpret_cast<void *>(&args)...};
        CudaRTCheck(cudaLaunchKernel(function_, launch_grid, block_dim_, args_array, shared_mem_size_, stream));
#ifndef NDEBUG
        // check kernel runtime errors - note this is a blocking operation
        CudaRTCheck(cudaStreamSynchronize(stream));
#endif // !NDEBUG
    }

    /**
     * Launch the kernel on a grid.
     *
     * @param grid [in] grid size
     * @param args [in] kernel arguments (optional)
     */
    template<class... TYPES>
    void Launch(const Vector3ui &grid, TYPES... args) const
    {
        const dim3 launch_grid = calc_launch_grid_(grid);

        void *args_array[] = {reinterpret_cast<void *>(&args)...};
        CudaRTCheck(
            cudaLaunchKernel(function_, launch_grid, block_dim_, args_array, shared_mem_size_, cudaStreamPerThread));
#ifndef NDEBUG
        // check kernel runtime errors - note this is a blocking operation
        CudaRTCheck(cudaStreamSynchronize(cudaStreamPerThread));
#endif // !NDEBUG
    }

private:
    const CUfunction function_;
    const size_t shared_mem_size_;

    std::function<dim3(const Vector3ui &grid)> calc_launch_grid_;
    int optimal_block_size_;
    dim3 block_dim_;
};

/**
 * The adaptive executor scales the iteration count of a function launching Cuda kernels so that as many iterations as possible
 * can be executed within the given duration.
 * This requires that the function records four events to measure the time in the pre, main and post Cuda kernels.
 * The function gets a pointer to the adaptive executor to get the events and the current iteration count. The runtime of the
 * function should be directly proportional to the iteration count calculated.
 */
class CudaAdaptiveExecutor
{
public:
    /**
     * Construct
     *
     * @param func [in] the function to execute
     */
    explicit CudaAdaptiveExecutor(const std::function<void(const CudaAdaptiveExecutor *executor)> &func)
        : func_(func)
        , target_duration_(0.f)
    {
        // create iteration timing events
        start_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
            return event;
        }());
        start_iterations_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
            return event;
        }());
        end_iterations_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
            return event;
        }());
        end_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
            return event;
        }());
    }
    CudaAdaptiveExecutor() = delete;

    /**
     * Reset, call that when the duration spend by one iteration will change.
     */
    void Reset()
    {
        previous_iterations_ = 0;
    }

    /**
     * Set the target duration. A target duration of 0.0 means to disable adaptive execution.
     *
     * @param target_duration [in] target duration
     *
     */
    void SetTargetDuration(const std::chrono::duration<float, std::milli> &target_duration)
    {
        if (target_duration.count() < 0.f)
        {
            throw InvalidArgument("target_duration") << "is negative";
        }
        target_duration_ = target_duration;
    }

    /**
     * Records the start event.
     *
     * @param stream [in] stream to record the event for
     */
    void RecordStartEvent(CUstream stream) const
    {
        CudaCheck(cuEventRecord(start_event_.get(), stream));
    }

    /**
     * Records the start iterations event.
     *
     * @param stream [in] stream to record the event for
     */
    void RecordStartIterationsEvent(CUstream stream) const
    {
        CudaCheck(cuEventRecord(start_iterations_event_.get(), stream));
    }

    /**
     * Records the end iterations event.
     *
     * @param stream [in] stream to record the event for
     */
    void RecordEndIterationsEvent(CUstream stream) const
    {
        CudaCheck(cuEventRecord(end_iterations_event_.get(), stream));
    }

    /**
     * Records the end event.
     *
     * @param stream [in] stream to record the event for
     */
    void RecordEndEvent(CUstream stream) const
    {
        CudaCheck(cuEventRecord(end_event_.get(), stream));
    }

    /**
     * @returns the amount of iterations which should be executed next.
     */
    uint32_t GetIterations() const
    {
        return iterations_;
    }

    /**
     * Execute.
     * Gets the run times of the previous execution and calculates the amount of iterations which should be executed.
     * Then calls the function to be executed.
     */
    void Execute()
    {
        // if target duration is set then time the execution and use dynamic scaling, else set iterations to max
        if (target_duration_.count() != 0.f)
        {
            if (previous_iterations_ == 0)
            {
                // if this is the initial run, call the function once to get the first measurement
                iterations_ = 1;
                func_(this);
                previous_iterations_ = iterations_;
            }

            // the total time spend
            CudaCheck(cuEventSynchronize(start_event_.get()));
            CudaCheck(cuEventSynchronize(end_event_.get()));
            const std::chrono::duration<float, std::milli> total_duration([this] {
                float milliseconds = 0.f;
                CudaCheck(cuEventElapsedTime(&milliseconds, start_event_.get(), end_event_.get()));
                return milliseconds;
            }());

            // the time spend in the iterating kernel
            CudaCheck(cuEventSynchronize(start_iterations_event_.get()));
            CudaCheck(cuEventSynchronize(end_iterations_event_.get()));
            const std::chrono::duration<float, std::milli> all_iterations_duration([this] {
                float milliseconds = 0.f;
                CudaCheck(
                    cuEventElapsedTime(&milliseconds, start_iterations_event_.get(), end_iterations_event_.get()));
                return milliseconds;
            }());

            // the time one iteration of the scattering kernel took
            const std::chrono::duration<float, std::milli> iteration_duration =
                all_iterations_duration / previous_iterations_;

            // the time we have for the scattering iterations is the target time minus the time spend for pre- and post-processing
            // calculate the estimated iterations for the next frame
            iterations_ =
                std::max(static_cast<int32_t>((target_duration_ - (total_duration - all_iterations_duration)) /
                                              iteration_duration),
                         1);
        }
        else
        {
            iterations_ = std::numeric_limits<uint32_t>::max();
        }

        // now run with the calculated iteration count
        func_(this);

        previous_iterations_ = iterations_;
    }

private:
    const std::function<void(const CudaAdaptiveExecutor *executor)> func_;

    UniqueCudaEvent start_event_;
    UniqueCudaEvent start_iterations_event_;
    UniqueCudaEvent end_iterations_event_;
    UniqueCudaEvent end_event_;

    std::chrono::duration<float, std::milli> target_duration_;

    uint32_t iterations_          = 0;
    uint32_t previous_iterations_ = 0;
};

/**
 * RAII class to time the execution of a asyn CUDA function or kernel call.
 *
 * Usage:
 * @code{.cpp}
 * {
 *     CudaTiming timing("kernel_to_time");
 *     kernel_to_time<<<>>>();
 * }
 * @endcode
 */
class CudaTiming
{
public:
    /**
     * Construct. Starts the timing
     *
     * @param msg [in] message to when timing is done
     */
    explicit CudaTiming(const std::string &msg)
        : msg_(msg)
    {
        start_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
            return event;
        }());
        end_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
            return event;
        }());
        CudaCheck(cuEventRecord(start_event_.get(), CU_STREAM_PER_THREAD));
    }
    CudaTiming() = delete;

    /**
     * Destruct. Stops the timing and prints a message.
     */
    ~CudaTiming()
    {
        try
        {
            CudaCheck(cuEventRecord(end_event_.get(), CU_STREAM_PER_THREAD));
            CudaCheck(cuEventSynchronize(start_event_.get()));
            CudaCheck(cuEventSynchronize(end_event_.get()));
            const std::chrono::duration<float, std::milli> duration([this] {
                float milliseconds = 0.f;
                CudaCheck(cuEventElapsedTime(&milliseconds, start_event_.get(), end_event_.get()));
                return milliseconds;
            }());
            Log(LogLevel::Info) << "Timing " << msg_ << " took " << duration.count() << " ms";
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "CudaTiming destructor failed with " << e.what();
        }
    }

private:
    std::string msg_;
    UniqueCudaEvent start_event_;
    UniqueCudaEvent end_event_;
};

} // namespace clara::viz
