/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <functional>
#include <memory>

namespace pybind11
{
class object;
} // namespace pybind11

namespace clara::viz
{

class VideoInterface;

class VideoStream
{
public:
    VideoStream(const std::shared_ptr<clara::viz::VideoInterface> &video_interface,
                const std::function<void(pybind11::object, bool)> &callback);

    /**
     * Configure the video stream.
     * 
     * @param width [in] Width of the video, value needs to evenly divisible by 2
     * @param height [in] Height of the video, value needs to evenly divisible by 2
     * @param frame_rate [in] Target framerate of the video. If set to 0.0 the frames will delivered when rendering is done. Converging renderers
     *                        will deliver the final frame only.
     * @param bit_rate [in] Target bitrate of the video
     */
    void Configure(uint32_t width, uint32_t height, float frame_rate, uint32_t bit_rate);

    /**
     * Play video
     */
    void Play();

    /**
     * Pause video
     */
    void Pause();

    /**
     * Stop video. Video stream is closed if it had been open.
     */
    void Stop();

private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl *) const;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

} // namespace clara::viz
