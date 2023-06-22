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
#include <vector>
#include <ostream>

#include <claraviz/util/Log.h>

namespace clara::viz
{

class MessageReceiver;

/// Render backend for volumes
enum VolumeRenderBackend
{
    NvRTVol,   ///< NvRTVol renderer
#ifdef CLARA_VIZ_WITH_OMNIVERSE
    Omniverse, ///< Omniverse RTX renderer
#endif // CLARA_VIZ_WITH_OMNIVERSE
    Default = NvRTVol
};

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
     * @param image_msg_receiver [in] image message receiver
     * @param cuda_device_ordinals [in] Cuda devices to render on
     * @param volume_render_backend [in] render backend for volumes
     * @param log_level [in] log level
     * @param log_stream [in] log stream
     */
    explicit Renderer(const std::shared_ptr<MessageReceiver> &video_msg_receiver,
                      const std::shared_ptr<MessageReceiver> &image_msg_receiver,
                      const std::vector<uint32_t> &cuda_device_ordinals,
                      VolumeRenderBackend volume_render_backend = VolumeRenderBackend::Default,
                      LogLevel log_level = LogLevel::Info,
                      std::ostream *log_stream = &std::cout);
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
