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

#include <memory>
#include <vector>

namespace clara::viz
{

/**
 * Create a MP4 container stream from a H264/HEVC stream.
 */
class MP4Wrapper
{
public:
    MP4Wrapper();
    ~MP4Wrapper();

    /**
     * Reset the stream, this will write the initializations segment again.
     */
    void ResetStream();

    /**
     * Input frame type
     */
    enum class Type
    {
        H264,
        HEVC
    };

    /**
     * Wrap a H264 frame.
     *
     * @param width [in] stream width
     * @param height [in] stream height
     * @param fps [in] stream frames per second
     * @param type [in] input frame type
     * @param input_frame [in] input frame
     * @param output_buffer [in] stream output buffer (data is appended)
     */
    void Wrap(uint32_t width, uint32_t height, float fps, Type type, const std::vector<uint8_t> &input_frame,
              std::vector<uint8_t> &output_buffer);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clara::viz
