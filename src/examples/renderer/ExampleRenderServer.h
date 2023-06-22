/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
 * Example render server.
 * Provides a gRPC interface, renders images.
 */
class ExampleRenderServer : public RenderServerBase
{
public:
    /**
     * Construct.
     *
     * @param port [in] gRPC port
     * @param cuda_device_ordinal [in] Cuda device to render on
     * @param mhd_file_name [in] Name of MHD file to load
     */
    explicit ExampleRenderServer(uint32_t port, uint32_t cuda_device_ordinal, const std::string &mhd_file_name);
    ExampleRenderServer() = delete;

    /**
     * Destruct.
     */
    ~ExampleRenderServer();

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
