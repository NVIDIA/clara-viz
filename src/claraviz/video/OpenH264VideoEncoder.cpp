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

#include "claraviz/video/OpenH264VideoEncoder.h"

#include <list>
#include <vector>

#include <wels/codec_api.h>

#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/hardware/cuda/Convert.h"
#include "claraviz/util/Blob.h"
#include "claraviz/video/Mp4Wrapper.h"

namespace clara::viz
{

namespace
{

/// use min/max values to match NvEnc
constexpr uint32_t OPENH264_MIN_WIDTH  = 160;
constexpr uint32_t OPENH264_MIN_HEIGHT = 64;
constexpr uint32_t OPENH264_MAX_WIDTH  = 4096;
constexpr uint32_t OPENH264_MAX_HEIGHT = 4096;

/**
 * Operator that appends string representation of OpenH264 error.
 */
std::ostream &operator<<(std::ostream &os, const CM_RETURN ret)
{
    switch (ret)
    {
    case cmResultSuccess:
        os << std::string("cmResultSuccess");
        break;
    case cmInitParaError:
        os << std::string("cmInitParaError");
        break;
    case cmUnknownReason:
        os << std::string("cmUnknownReason");
        break;
    case cmMallocMemeError:
        os << std::string("cmMallocMemeError");
        break;
    case cmInitExpected:
        os << std::string("cmInitExpected");
        break;
    case cmUnsupportedData:
        os << std::string("cmUnsupportedData");
        break;
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

/**
 * OpenH264 error check helper
 */
#define OpenH264Check(FUNC)                                       \
    {                                                             \
        const CM_RETURN ret = static_cast<CM_RETURN>(FUNC);       \
        if (ret != cmResultSuccess)                               \
        {                                                         \
            throw RuntimeError() << "OpenH264 API error " << ret; \
        }                                                         \
    }

} // anonymous namespace

class OpenH264VideoEncoder::Impl
{
public:
    /**
     * Construct
     *
     * @param cuda_device_ordinal [in] Cuda device to use for encoding
     */
    explicit Impl(uint32_t cuda_device_ordinal)
        : frame_index_(0)
        , new_stream_(false)
        , initialized_(false)
        , encoder_(nullptr)
    {
        params_                = {};
        params_.iUsageType     = CAMERA_VIDEO_REAL_TIME;
        params_.iRCMode        = RC_BITRATE_MODE;
        params_.iPicWidth      = OPENH264_MIN_WIDTH;
        params_.iPicHeight     = OPENH264_MIN_HEIGHT;
        params_.fMaxFrameRate  = 30.f;
        params_.iTargetBitrate = 1 * 1024 * 1024;

        CudaCheck(cuInit(0));
        cuda_context_.reset(new CudaContext(cuda_device_ordinal));
    }

    ~Impl()
    {
        if (encoder_)
        {
            encoder_->Uninitialize();
            WelsDestroySVCEncoder(encoder_);
        }
    }

    /**
     * Set the output stream.
     *
     * @param stream [in] output stream for video data
     */
    void SetStream(const std::shared_ptr<IVideoStream> &stream);

    /**
     * Set the target frame rate of the video
     *
     * @param frame_rate [in] new target frame rate
     */
    void SetFrameRate(float frame_rate);

    /**
     * Set the bit rate of the video
     *
     * @param bit_rate [in] new bit rate
     */
    void SetBitRate(uint32_t bit_rate);

    /**
     * @return the cuda context used for encoding
     */
    const std::shared_ptr<CudaContext> &GetCudaContext();

    /**
     * Encode a memory blob.
     *
     * @param width [in] width
     * @param height [in] height
     * @param memory [in] Cuda 2D memory
     * @param format [in] format of the data
     */
    void Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory, IVideoEncoder::Format format);

private:
    /**
     * Initialize the encoder if needed.
     */
    void Initialize();

    /**
     * Uninitialize the encoder when e.g. a parameter such as the resolution changed.
     */
    void Uninitialize();

    /**
     * Set the video resolution.
     *
     * @param width [in] new width
     * @param height [in] new height
     */
    void SetResolution(uint32_t width, uint32_t height);

    /// Cuda context used by the encoder
    std::shared_ptr<CudaContext> cuda_context_;

    uint32_t frame_index_;

    std::shared_ptr<IVideoStream> stream_; ///< output stream
    bool new_stream_;                      ///< set if the stream has changed

    std::unique_ptr<CudaFunctionLauncher> convert_ABGR_to_YCbCr420CCIR601_;
    std::unique_ptr<CudaMemory2D> buffer_ycbcr_;

    bool initialized_;     ///< encoder is initialized
    SEncParamBase params_; ///< encoder init parameters
    /// @todo should be smart ptr
    ISVCEncoder *encoder_;

    MP4Wrapper mp4_wrapper_; ///< MP4 wrapper
};

OpenH264VideoEncoder::OpenH264VideoEncoder(uint32_t cuda_device_ordinal)
    : impl_(new Impl(cuda_device_ordinal))
{
}

OpenH264VideoEncoder::~OpenH264VideoEncoder() {}

int OpenH264VideoEncoder::Query(IVideoEncoder::Capability capability)
{
    switch (capability)
    {
    case IVideoEncoder::Capability::IS_SUPPORTED:
        return true;
    case IVideoEncoder::Capability::MIN_WIDTH:
        return OPENH264_MIN_WIDTH;
    case IVideoEncoder::Capability::MIN_HEIGHT:
        return OPENH264_MIN_HEIGHT;
    case IVideoEncoder::Capability::MAX_WIDTH:
        return OPENH264_MAX_WIDTH;
    case IVideoEncoder::Capability::MAX_HEIGHT:
        return OPENH264_MAX_HEIGHT;
    }
    throw InvalidState() << "Unhandled capability";
}

void OpenH264VideoEncoder::SetStream(const std::shared_ptr<IVideoStream> &stream)
{
    impl_->SetStream(stream);
}

void OpenH264VideoEncoder::SetFrameRate(float frame_rate)
{
    impl_->SetFrameRate(frame_rate);
}

void OpenH264VideoEncoder::SetBitRate(uint32_t bit_rate)
{
    impl_->SetBitRate(bit_rate);
}

const std::shared_ptr<CudaContext> &OpenH264VideoEncoder::GetCudaContext()
{
    return impl_->GetCudaContext();
}

void OpenH264VideoEncoder::Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory,
                                  IVideoEncoder::Format format)
{
    impl_->Encode(width, height, memory, format);
}

void OpenH264VideoEncoder::Impl::SetStream(const std::shared_ptr<IVideoStream> &stream)
{
    if (stream_ != stream)
    {
        new_stream_ = true;
        stream_     = stream;
    }
}

void OpenH264VideoEncoder::Impl::SetResolution(uint32_t width, uint32_t height)
{
    // check for min width
    assert(width >= OPENH264_MIN_WIDTH);
    assert(height >= OPENH264_MIN_HEIGHT);
    // OpenH264 can encode frames with even width/height only
    assert((width & 1) == 0);
    assert((height & 1) == 0);

    if ((width > OPENH264_MAX_WIDTH) || (height > OPENH264_MAX_HEIGHT))
    {
        throw InvalidState() << "The video resolution of (" << width << ", " << height
                             << ") exceeds the maximum supported resolution of (" << OPENH264_MAX_WIDTH << ", "
                             << OPENH264_MAX_HEIGHT << ")";
    }

    // if the resolution changed reconfigure the encoder and the MP4 wrapper
    if ((params_.iPicWidth != width) || (params_.iPicHeight != height))
    {
        params_.iPicWidth  = width;
        params_.iPicHeight = height;
        Uninitialize();

        // a change in resolution forces a new stream
        new_stream_ = true;
    }
}

void OpenH264VideoEncoder::Impl::SetFrameRate(float frame_rate)
{
    if (frame_rate < 0.f)
    {
        throw InvalidArgument("frame_rate") << "is negative";
    }

    // if the frame rate changed reconfigure the encoder
    if (params_.fMaxFrameRate != frame_rate)
    {
        params_.fMaxFrameRate = frame_rate;
        Uninitialize();

        // a change in frame rate forces a new stream
        new_stream_ = true;
    }
}

void OpenH264VideoEncoder::Impl::SetBitRate(uint32_t bit_rate)
{
    if (bit_rate == 0)
    {
        throw InvalidArgument("bit_rate") << "is zero";
    }

    // if the bit rate changed reconfigure the encoder
    if (params_.iTargetBitrate != bit_rate)
    {
        params_.iTargetBitrate = bit_rate;
        Uninitialize();
    }
}

const std::shared_ptr<CudaContext> &OpenH264VideoEncoder::Impl::GetCudaContext()
{
    return cuda_context_;
}

namespace
{

/**
 * Encoder trace callback function
 **/
void EncodeTraceCallback(void *context, int level, const char *message)
{
    LogLevel logLevel;
    switch (level)
    {
    case WELS_LOG_ERROR:
        logLevel = LogLevel::Error;
        break;
    case WELS_LOG_WARNING:
        logLevel = LogLevel::Warning;
        break;
    case WELS_LOG_INFO:
        logLevel = LogLevel::Info;
        break;
    case WELS_LOG_DEBUG:
        logLevel = LogLevel::Debug;
        break;
    default:
        logLevel = LogLevel::Info;
        break;
    }
    Log(logLevel) << message;
}

} // namespace

void OpenH264VideoEncoder::Impl::Initialize()
{
    if (!encoder_)
    {
        OpenH264Check(WelsCreateSVCEncoder(&encoder_));
        void *func = reinterpret_cast<void *>(&EncodeTraceCallback);
        encoder_->SetOption(ENCODER_OPTION_TRACE_CALLBACK, &func);

        int32_t trace_level;
        encoder_->GetOption(ENCODER_OPTION_TRACE_LEVEL, &trace_level);
        switch (Log::g_log_level)
        {
        case LogLevel::Debug:
            trace_level = WELS_LOG_DEBUG;
            break;
        case LogLevel::Info:
            // OpenH264 'info' level is more verbose than what we want, display warnings only
            trace_level = WELS_LOG_WARNING;
            break;
        case LogLevel::Warning:
            trace_level = WELS_LOG_WARNING;
            break;
        case LogLevel::Error:
            trace_level = WELS_LOG_ERROR;
            break;
        }
        encoder_->SetOption(ENCODER_OPTION_TRACE_LEVEL, &trace_level);

        Log(LogLevel::Info) << "OpenH264 video encoder max resolution " << OPENH264_MAX_WIDTH << "x"
                            << OPENH264_MAX_HEIGHT;
    }
    if (!initialized_)
    {
        OpenH264Check(encoder_->Initialize(&params_));
        SProfileInfo profileInfo = {};
        profileInfo.iLayer       = 0;
        profileInfo.uiProfileIdc = PRO_MAIN;
        OpenH264Check(encoder_->SetOption(ENCODER_OPTION_PROFILE, &profileInfo));
        initialized_ = true;
    }
}

void OpenH264VideoEncoder::Impl::Uninitialize()
{
    if (encoder_ && initialized_)
    {
        OpenH264Check(encoder_->Uninitialize());
    }
    initialized_ = false;
}

void OpenH264VideoEncoder::Impl::Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory,
                                        IVideoEncoder::Format format)
{
    if (!memory)
    {
        throw InvalidArgument("memory") << "is a nullptr";
    }
    if (!stream_)
    {
        throw InvalidState() << "output stream is not set";
    }

    if (format != IVideoEncoder::Format::ABGR)
    {
        throw InvalidArgument("format") << "only ABGR is supported, but format is " << format;
    }
    // check for even width/height
    if ((width & 1 != 0) || (height & 1 != 0))
    {
        throw InvalidState() << "Resource width and height need to be evenly divisible by two, resource size is ("
                             << width << ", " << height << ")";
    }
    // check minimum size
    if ((width < OPENH264_MIN_WIDTH) || (height < OPENH264_MIN_HEIGHT))
    {
        throw InvalidState() << "Resource width and height has to be at least " << OPENH264_MIN_WIDTH << ", "
                             << OPENH264_MIN_HEIGHT << " resource size is (" << width << ", " << height << ")";
    }

    // set the resolution
    SetResolution(width, height);

    // make the cuda context current
    CudaContext::ScopedPush scoped_cuda_context(*cuda_context_.get());

    if (!convert_ABGR_to_YCbCr420CCIR601_)
    {
        convert_ABGR_to_YCbCr420CCIR601_ = GetConvertABGRToYCbCr420CCIR601Launcher();
    }

    // allocate YCbCr buffer
    if (!buffer_ycbcr_ || (buffer_ycbcr_->GetWidth() != width) || (buffer_ycbcr_->GetHeight() != height + height / 2))
    {
        buffer_ycbcr_.reset(new CudaMemory2D(width, height + height / 2, sizeof(uint8_t)));
    }

    {
        std::unique_ptr<IBlob::AccessGuardConst> access = memory->AccessConst(CU_STREAM_PER_THREAD);

        convert_ABGR_to_YCbCr420CCIR601_->Launch(
            Vector2ui(width, height), width, height, reinterpret_cast<const uint8_t *>(access->GetData()),
            static_cast<size_t>(width * 4), reinterpret_cast<uint8_t *>(buffer_ycbcr_->GetMemory().get()),
            buffer_ycbcr_->GetPitch(),
            reinterpret_cast<uint8_t *>(buffer_ycbcr_->GetMemory().get()) + buffer_ycbcr_->GetPitch() * height,
            reinterpret_cast<uint8_t *>(buffer_ycbcr_->GetMemory().get()) + buffer_ycbcr_->GetPitch() * height +
                width / 2,
            buffer_ycbcr_->GetPitch());
    }

    // copy the image to host memory
    std::vector<uint8_t> buf(buffer_ycbcr_->GetWidth() * buffer_ycbcr_->GetHeight() * buffer_ycbcr_->GetElementSize());
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice     = buffer_ycbcr_->GetMemory().get();
    copy.srcPitch      = buffer_ycbcr_->GetPitch();
    copy.WidthInBytes  = buffer_ycbcr_->GetWidth() * buffer_ycbcr_->GetElementSize();
    copy.Height        = buffer_ycbcr_->GetHeight();
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstPitch      = copy.WidthInBytes;
    copy.dstHost       = buf.data();
    CudaCheck(cuMemcpy2DAsync(&copy, CU_STREAM_PER_THREAD));
    // wait for the transfer to finish
    buffer_ycbcr_->EventRecord(CU_STREAM_PER_THREAD);
    buffer_ycbcr_->EventSynchronize();

    // initialize the encoder
    Initialize();

    SSourcePicture pic = {};
    pic.iPicWidth      = width;
    pic.iPicHeight     = height;
    pic.iColorFormat   = videoFormatI420;
    pic.iStride[0]     = buffer_ycbcr_->GetWidth() * buffer_ycbcr_->GetElementSize();
    pic.iStride[1]     = pic.iStride[0];
    pic.iStride[2]     = pic.iStride[0];
    pic.pData[0]       = buf.data();
    pic.pData[1]       = pic.pData[0] + pic.iStride[0] * pic.iPicHeight;
    pic.pData[2]       = pic.pData[1] + pic.iStride[0] / 2;

    // if the stream has changed then force an IDR picture and reset the MP4 stream so
    // that the header is re-written for the new stream
    if (new_stream_)
    {
        encoder_->ForceIntraFrame(true);

        mp4_wrapper_.ResetStream();
        new_stream_  = false;
        frame_index_ = 0;

        // tell the stream that a new stream begins
        stream_->NewStream();
    }

    // force an IDR picture every second
    if (frame_index_ > params_.fMaxFrameRate)
    {
        encoder_->ForceIntraFrame(true);
        frame_index_ = 0;
    }
    else
    {
        ++frame_index_;
    }

    SFrameBSInfo info = {};
    OpenH264Check(encoder_->EncodeFrame(&pic, &info));

    if (info.eFrameType == videoFrameTypeSkip)
    {
        Log(LogLevel::Debug) << "Encoder skipped frame";
        return;
    }

    // write the video frame
    std::vector<uint8_t> frame_buffer(info.iFrameSizeInBytes);
    int writeIndex = 0;
    for (int layer = 0; layer < info.iLayerNum; ++layer)
    {
        SLayerBSInfo *layerInfo = &info.sLayerInfo[layer];
        if (layerInfo != NULL)
        {
            int layerSize = 0;
            for (int nalIdx = 0; nalIdx < layerInfo->iNalCount; ++nalIdx)
            {
                layerSize += layerInfo->pNalLengthInByte[nalIdx];
            }
            if (writeIndex + layerSize > info.iFrameSizeInBytes)
            {
                throw RuntimeError() << "Layer buffer size mismatch";
            }
            std::memcpy(&frame_buffer[writeIndex], layerInfo->pBsBuf, layerSize);
            writeIndex += layerSize;
        }
    }
    if (writeIndex != info.iFrameSizeInBytes)
    {
        throw RuntimeError() << "Frame buffer size mismatch";
    }

    // wrap into MP4 stream
    std::vector<uint8_t> mp4_buffer;
    mp4_wrapper_.Wrap(params_.iPicWidth, params_.iPicHeight, params_.fMaxFrameRate, MP4Wrapper::Type::H264,
                      frame_buffer, mp4_buffer);

    // write the data out
    stream_->Write(reinterpret_cast<char *>(mp4_buffer.data()), mp4_buffer.size());
}

} // namespace clara::viz
