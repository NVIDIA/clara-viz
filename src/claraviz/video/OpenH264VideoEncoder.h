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

#include "claraviz/video/VideoEncoder.h"

namespace clara::viz
{

/**
 * OpenH264 video encoder
 */
class OpenH264VideoEncoder : public IVideoEncoder
{
public:
    /**
     * Construct
     *
     * @param cuda_device_ordinal [in] Cuda device to use for encoding
     */
    explicit OpenH264VideoEncoder(uint32_t cuda_device_ordinal);
    OpenH264VideoEncoder() = delete;

    /**
     * Destruct
     */
    virtual ~OpenH264VideoEncoder();

    /// clara::viz::IVideoEncoder virtual members
    ///@{
    int32_t Query(IVideoEncoder::Capability capability) final override;
    void SetStream(const std::shared_ptr<IVideoStream> &stream) final override;
    void SetFrameRate(float frame_rate) final override;
    void SetBitRate(uint32_t bit_rate) final override;
    const std::shared_ptr<CudaContext> &GetCudaContext() final override;
    void Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory,
                IVideoEncoder::Format format) final override;
    ///@}

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clara::viz
