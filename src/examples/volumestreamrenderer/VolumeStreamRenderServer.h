/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <claraviz/core/RenderServerBase.h>

namespace clara::viz
{

/**
 * Volume stream render server.
 * Provides a gRPC interface, renders images.
 */
class VolumeStreamRenderServer : public RenderServerBase
{
public:
    /**
     * Construct.
     *
     * @param input_dir [in] source data input directory, if empty generate synthetic data
     * @param scenario [in] scenario to execute
     * @param benchmark_duration [in] benchmark duration in seconds, if 0 run in interactive mode
     * @param stream_from_cpu [in] if set, stream from CPU memory else from GPU memory
     * @param port [in] gRPC port
     * @param cuda_device_ordinals [in] Cuda devices to render on
     */
    explicit VolumeStreamRenderServer(const std::string &input_dir, const std::string &scenario,
                                      const std::chrono::seconds &benchmark_duration, bool stream_from_cpu,
                                      uint32_t port, const std::vector<uint32_t> &cuda_device_ordinals);
    VolumeStreamRenderServer() = delete;

    /**
     * Destruct.
     */
    ~VolumeStreamRenderServer();

    /** @name RenderServerBase methods */
    /**@{*/
    void Run() final;
    bool Shutdown() final;
    /**@}*/

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::viz
