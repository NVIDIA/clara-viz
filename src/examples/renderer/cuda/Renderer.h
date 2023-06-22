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

namespace clara::viz
{

class MessageReceiver;

/**
 * Renderer.
 * Receives messages from the interfaces, renders images and pass them to the video message
 * receiver.
 */
class Renderer
{
public:
    /**
     * Construct
     *
     * @param video_msg_receiver [in] video message receiver
     * @param cuda_device_ordinal [in] Cuda device to render on
     * @param mhd_file_name [in] Name of MHD file to load
     */
    explicit Renderer(const std::shared_ptr<MessageReceiver> &video_msg_receiver, uint32_t cuda_device_ordinal,
                      const std::string &mhd_file_name);
    Renderer() = delete;

    /**
     * Destruct.
     */
    ~Renderer();

    /**
     * Run the renderer.
     */
    void Run();

    /**
     * Shutdown the renderer, wait for all running tasks to finish.
     */
    void Shutdown();

    /**
     * @returns the message receiver of the renderer
     */
    std::shared_ptr<MessageReceiver> GetReceiver() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::viz
