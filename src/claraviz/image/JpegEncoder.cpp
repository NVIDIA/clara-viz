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

#include "claraviz/image/JpegEncoder.h"

#include <list>
#include <cstring>

#include "claraviz/hardware/cuda/Convert.h"
#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/hardware/nvjpeg/NvJpegService.h"
#include "claraviz/util/Blob.h"

namespace clara::viz
{

class JpegEncoder::Impl
{
public:
    /**
     * Construct
     */
    explicit Impl()
        : quality_(75)
    {
    }

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
    uint32_t quality_;

    std::unique_ptr<CudaFunctionLauncher> convert_ABGR_to_YCbCr444CCIR601_;
    std::unique_ptr<CudaMemory2D> buffer_ycbcr_;

    UniqueNvJpegInstance instance_;    ///< encoder instance
    UniqueNvJpegEncoderState state_;   ///< encoder state
    UniqueNvJpegEncoderParams params_; ///< encoder params
};

JpegEncoder::JpegEncoder()
    : impl_(new Impl())
{
}

JpegEncoder::~JpegEncoder() {}

void JpegEncoder::SetQuality(uint32_t quality)
{
    impl_->SetQuality(quality);
}

void JpegEncoder::Encode(uint32_t width, uint32_t height, Format format, const std::shared_ptr<IBlob> &memory,
                         std::vector<uint8_t> &bitstream)
{
    impl_->Encode(width, height, format, memory, bitstream);
}

void JpegEncoder::Impl::SetQuality(uint32_t quality)
{
    quality_ = quality;
}

void JpegEncoder::Impl::Encode(uint32_t width, uint32_t height, Format format, const std::shared_ptr<IBlob> &memory,
                               std::vector<uint8_t> &bitstream)
{
    if (!memory)
    {
        throw InvalidArgument("memory") << "is a nullptr";
    }

    if (!convert_ABGR_to_YCbCr444CCIR601_)
    {
        convert_ABGR_to_YCbCr444CCIR601_ = GetConvertABGRToYCbCr444CCIR601Launcher();
    }
    if (!instance_)
    {
        nvjpegHandle_t instance = nullptr;
        NvJpegCheck(nvjpegCreateSimple(&instance));
        instance_.reset(instance);
    }
    if (!state_)
    {
        nvjpegEncoderState_t state = nullptr;
        NvJpegCheck(nvjpegEncoderStateCreate(instance_.get(), &state, CU_STREAM_PER_THREAD));
        state_.reset(state);
    }
    if (!params_)
    {
        nvjpegEncoderParams_t params = nullptr;
        NvJpegCheck(nvjpegEncoderParamsCreate(instance_.get(), &params, CU_STREAM_PER_THREAD));
        params_.reset(params);

        NvJpegCheck(nvjpegEncoderParamsSetOptimizedHuffman(params_.get(), 1, CU_STREAM_PER_THREAD));
        NvJpegCheck(nvjpegEncoderParamsSetSamplingFactors(params_.get(), NVJPEG_CSS_444, CU_STREAM_PER_THREAD));
    }
    NvJpegCheck(nvjpegEncoderParamsSetQuality(params_.get(), quality_, CU_STREAM_PER_THREAD));

    // allocate YCbCr buffer
    if (!buffer_ycbcr_ || (buffer_ycbcr_->GetWidth() != width) || (buffer_ycbcr_->GetHeight() != height * 3))
    {
        buffer_ycbcr_.reset(new CudaMemory2D(width, height * 3, sizeof(uint8_t)));
    }

    // data is organized in planes
    nvjpegImage_t image{};

    image.pitch[0] = buffer_ycbcr_->GetPitch();
    image.pitch[1] = buffer_ycbcr_->GetPitch();
    image.pitch[2] = buffer_ycbcr_->GetPitch();

    image.channel[0] = reinterpret_cast<uint8_t *>(buffer_ycbcr_->GetMemory().get());
    image.channel[1] = image.channel[0] + image.pitch[0] * height;
    image.channel[2] = image.channel[1] + image.pitch[1] * height;

    {
        std::unique_ptr<IBlob::AccessGuardConst> access = memory->AccessConst(CU_STREAM_PER_THREAD);

        convert_ABGR_to_YCbCr444CCIR601_->Launch(
            Vector2ui(width, height), width, height, reinterpret_cast<const uint8_t *>(access->GetData()),
            static_cast<size_t>(width * 4), image.channel[0], image.channel[1], image.channel[2], buffer_ycbcr_->GetPitch());
    }

    // Compress image
    NvJpegCheck(nvjpegEncodeYUV(instance_.get(), state_.get(), params_.get(), &image, NVJPEG_CSS_444, width, height,
                                CU_STREAM_PER_THREAD));

    // get the bitstream size
    size_t length = 0;
    NvJpegCheck(nvjpegEncodeRetrieveBitstream(instance_.get(), state_.get(), nullptr, &length, CU_STREAM_PER_THREAD));

    // allocate the bitstream
    bitstream.resize(length);

    // retrieve the data
    NvJpegCheck(
        nvjpegEncodeRetrieveBitstream(instance_.get(), state_.get(), bitstream.data(), &length, CU_STREAM_PER_THREAD));
}

} // namespace clara::viz
