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

#include <memory>
#include <vector>

#include "claraviz/util/Types.h"

namespace clara::viz
{

class IBlob;

/**
 * JPEG encoder class
 */
class JpegEncoder : public NonCopyable
{
public:
    /**
     * Construct
     */
    JpegEncoder();
    ~JpegEncoder() final;

    /**
     * Supported pixel formats
     */
    enum class Format
    {
        /**
         * 8 bit Packed A8R8G8B8. This is a word-ordered format
         * where a pixel is represented by a 32-bit word with B
         * in the lowest 8 bits, G in the next 8 bits, R in the
         * 8 bits after that and A in the highest 8 bits. */
        ARGB,
        /**
         * 8 bit Packed A8B8G8R8. This is a word-ordered format
         * where a pixel is represented by a 32-bit word with R
         * in the lowest 8 bits, G in the next 8 bits, B in the
         * 8 bits after that and A in the highest 8 bits. */
        ABGR
    };

    /**
     * Set the quality
     *
     * @param quality [in]
     **/
    void SetQuality(uint32_t quality);

    /**
     * Encode an image represented by IBlob.
     *
     * @param width [in] image width
     * @param height [in] image height
     * @param format [in] image format
     * @param memory [in] memory to encode
     * @param bitstream [out] output bitstream
     */
    void Encode(uint32_t width, uint32_t height, Format format, const std::shared_ptr<IBlob> &memory,
                std::vector<uint8_t> &bitstream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clara::viz
