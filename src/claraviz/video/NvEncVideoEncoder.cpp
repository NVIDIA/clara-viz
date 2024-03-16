/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/video/NvEncVideoEncoder.h"

#include <list>
#include <cstring>

#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/hardware/nvenc/NvEncService.h"
#include "claraviz/util/Blob.h"
#include "claraviz/video/Mp4Wrapper.h"

namespace clara::viz
{

namespace
{

/// Frame rate is passed to NVENC in a numerator/denominator pair, we use a fixed denominator
constexpr float FRAME_RATE_DENOMINATOR = 1000.f;

/**
 * Minimum width/height (currently this is fixed, the capability to query will be added with NvEncAPI 9.1 but we want
 * to stay at 9.0 because it works with driver version 418, 9.1 needs driver 430+)
 * According to the driver the values should be:
 *                   min width    min height
 * Turing H264       145          49
 * Turing HEVC       129          33
 * pre-Turing H264    33          17
 * pre-Turing HEVC    65          33
 * But these do not work, so I rounded up to multiples of 16.
 */
constexpr uint32_t VIDEO_ENCODER_MIN_WIDTH  = 160;
constexpr uint32_t VIDEO_ENCODER_MIN_HEIGHT = 64;

class VideoResource
{
public:
    VideoResource(const UniqueNvEncSession &session, const NV_ENC_REGISTER_RESOURCE &resource)
        : session_(session)
        , resource_(resource)
    {
    }
    ~VideoResource()
    {
        NvEncUnregisterResource(session_.get(), resource_.registeredResource);
    }

    const NV_ENC_REGISTER_RESOURCE resource_;

private:
    const UniqueNvEncSession &session_;
};

/**
 * Converts a IVideoEncoder format to a NvENC format
 *
 * @param format [in] IVideoEncoder format
 *
 * @return NvENC format
 */
NV_ENC_BUFFER_FORMAT ToNvEncFormat(IVideoEncoder::Format format)
{

    switch (format)
    {
    case IVideoEncoder::Format::ARGB:
        return NV_ENC_BUFFER_FORMAT_ARGB;
    case IVideoEncoder::Format::ABGR:
        return NV_ENC_BUFFER_FORMAT_ABGR;
    default:
        throw InvalidState() << "Unhandled format " << format;
    }
}

/**
 * Get the bytes per pixel for a IVideoEncoder format
 *
 * @param format [in] IVideoEncoder format
 *
 * @return bytes per pixel
 */
uint32_t BytesPerPixel(IVideoEncoder::Format format)
{

    switch (format)
    {
    case IVideoEncoder::Format::ARGB:
    case IVideoEncoder::Format::ABGR:
        return 4;
    default:
        throw InvalidState() << "Unhandled format " << format;
    }
}

} // anonymous namespace

class NvEncVideoEncoder::Impl
{
private:
    struct BitstreamBuffer
    {
        BitstreamBuffer(const UniqueNvEncSession &session, NV_ENC_OUTPUT_PTR bitstream_buffer)
            : session_(session)
            , bitstream_buffer_(bitstream_buffer)
        {
        }
        ~BitstreamBuffer()
        {
            NvEncDestroyBitstreamBuffer(session_.get(), bitstream_buffer_);
        }

        const UniqueNvEncSession &session_;
        NV_ENC_OUTPUT_PTR bitstream_buffer_;
    };

public:
    /**
     * Construct
     *
     * @param cuda_device_ordinal [in] Cuda device to use for encoding
     */
    explicit Impl(uint32_t cuda_device_ordinal)
        : width_(VIDEO_ENCODER_MIN_WIDTH)
        , height_(VIDEO_ENCODER_MIN_HEIGHT)
        , frame_rate_(30.f)
        , bit_rate_(1 * 1024 * 1024)
        , use_hevc_(false)
        , frame_index_(0)
        , new_stream_(false)
    {
        init_params_         = {};
        init_params_.version = NV_ENC_INITIALIZE_PARAMS_VER;

        CudaCheck(cuInit(0));
        cuda_context_.reset(new CudaContext(cuda_device_ordinal));
    }

    /**
     * Query a capability value from the encoder
     *
     * @param cap [in] capability value enum
     *
     * @returns queried value
     */
    int QueryEncodeCap(NV_ENC_CAPS cap);

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
     * Get the encoder session, create one if there is none.
     *
     * @returns encoder session
     */
    const UniqueNvEncSession &GetSession();

    /**
     * Register a memory blob to be used by the encoder
     *
     * @param width [in] width
     * @param height [in] height
     * @param memory [in] memory blob
     * @param format [in] format of the data
     *
     * @returns video resource pointer
     */
    std::shared_ptr<VideoResource> RegisterResource(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory, IVideoEncoder::Format format);

    /**
     * @return the cuda context used for encoding
     */
    const std::shared_ptr<CudaContext> &GetCudaContext();

    /**
     * Encode a memory blob.
     *
     * @param width [in] width
     * @param height [in] height
     * @param memory [in] memory blob
     * @param format [in] format of the data
     */
    void Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory, IVideoEncoder::Format format);

private:
    /**
     * Set the video resolution.
     *
     * @param width [in] new width
     * @param height [in] new height
     */
    void SetResolution(uint32_t width, uint32_t height);

    /**
     * Check if the given encode guid is supported
     *
     * @param encode_guid [in]
     *
     * @returns false if the guid is not supported
     */
    bool CheckEncodeGUID(const GUID &encode_guid);

    /**
     * Check if the given encoder preset guid is supported
     *
     * @param preset_guid [in]
     *
     * @returns false if the guid is not supported
     */
    bool CheckPresetGUID(const GUID &preset_guid);

    /**
     * Check if the given encoder profile guid is supported
     *
     * @param profile_guid [in]
     *
     * @returns false if the guid is not supported
     */
    bool CheckProfileGUID(const GUID &profile_guid);

    /**
     * Check if the given rate control mode is supported
     *
     * @param rate_control_mode [in]
     *
     * @returns false if the mode is not supported
     */
    bool CheckRateControlMode(NV_ENC_PARAMS_RC_MODE rate_control_mode);

    /// Cuda context used by the encoder
    std::shared_ptr<CudaContext> cuda_context_;

    uint32_t width_;
    uint32_t height_;
    float frame_rate_;
    uint32_t bit_rate_;
    bool use_hevc_;

    uint32_t frame_index_;

    std::shared_ptr<IVideoStream> stream_; ///< output stream
    bool new_stream_;                      ///< set if the stream has changed

    NV_ENC_INITIALIZE_PARAMS init_params_; ///< encoder init parameters
    NV_ENC_CONFIG config_;                 ///< encoder config

    /// @todo should be shared_ptr
    UniqueNvEncSession session_; ///< encoder session

    std::list<std::shared_ptr<VideoResource>> registered_; ///< registered resources
    std::unique_ptr<BitstreamBuffer> bitstream_buffer_;    ///< bitstream output buffer

    MP4Wrapper mp4_wrapper_; ///< MP4 wrapper
};

NvEncVideoEncoder::NvEncVideoEncoder(uint32_t cuda_device_ordinal)
    : impl_(new Impl(cuda_device_ordinal))
{
}

NvEncVideoEncoder::~NvEncVideoEncoder() {}

int NvEncVideoEncoder::Query(Capability capability)
{
    switch (capability)
    {
    case IVideoEncoder::Capability::IS_SUPPORTED:
        return impl_->GetSession() ? 1 : 0;
    case IVideoEncoder::Capability::MIN_WIDTH:
        return VIDEO_ENCODER_MIN_WIDTH;
    case IVideoEncoder::Capability::MIN_HEIGHT:
        return VIDEO_ENCODER_MIN_HEIGHT;
    case IVideoEncoder::Capability::MAX_WIDTH:
        return impl_->QueryEncodeCap(NV_ENC_CAPS_WIDTH_MAX);
    case IVideoEncoder::Capability::MAX_HEIGHT:
        return impl_->QueryEncodeCap(NV_ENC_CAPS_HEIGHT_MAX);
    }
    throw InvalidState() << "Unhandled capability";
}

void NvEncVideoEncoder::SetStream(const std::shared_ptr<IVideoStream> &stream)
{
    impl_->SetStream(stream);
}

void NvEncVideoEncoder::SetFrameRate(float frame_rate)
{
    impl_->SetFrameRate(frame_rate);
}

void NvEncVideoEncoder::SetBitRate(uint32_t bit_rate)
{
    impl_->SetBitRate(bit_rate);
}

const std::shared_ptr<CudaContext> &NvEncVideoEncoder::GetCudaContext()
{
    return impl_->GetCudaContext();
}

void NvEncVideoEncoder::Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory,
                               IVideoEncoder::Format format)
{
    impl_->Encode(width, height, memory, format);
}

void NvEncVideoEncoder::Impl::SetStream(const std::shared_ptr<IVideoStream> &stream)
{
    if (stream_ != stream)
    {
        new_stream_ = true;
        stream_     = stream;
    }
}

void NvEncVideoEncoder::Impl::SetResolution(uint32_t width, uint32_t height)
{
    // NvENC can encode frames with even width/height only, therefore the resource size has to be even
    if ((width & 1 != 0) || (height & 1 != 0))
    {
        throw InvalidState() << "Resolution width and height need to be evenly divisible by two, resoltion is ("
                             << width << ", " << height << ")";
    }
    // NvEnc has a minimum size (unfortunately that value can't be queried).
    if ((width < VIDEO_ENCODER_MIN_WIDTH) || (height < VIDEO_ENCODER_MIN_HEIGHT))
    {
        throw InvalidState() << "Resolution width and height has to be at least " << VIDEO_ENCODER_MIN_WIDTH << ", "
                             << VIDEO_ENCODER_MIN_HEIGHT << " resolution is (" << width << ", " << height << ")";
    }

    if ((width > QueryEncodeCap(NV_ENC_CAPS_WIDTH_MAX)) || (height > QueryEncodeCap(NV_ENC_CAPS_HEIGHT_MAX)))
    {
        throw InvalidState() << "The video resolution of (" << width << ", " << height
                             << ") exceeds the maximum supported resolution of ("
                             << QueryEncodeCap(NV_ENC_CAPS_WIDTH_MAX) << ", " << QueryEncodeCap(NV_ENC_CAPS_HEIGHT_MAX)
                             << ")";
    }

    width_  = width;
    height_ = height;

    // if the resolution changed recreate the encoder session and the start a new stream in the MP4 wrapper
    if (session_ && ((init_params_.encodeWidth != width_) || (init_params_.encodeHeight != height_)))
    {
        session_.reset();

        // a change in resolution forces a new stream
        new_stream_ = true;
    }
}

void NvEncVideoEncoder::Impl::SetFrameRate(float frame_rate)
{
    if (frame_rate < 0.f)
    {
        throw InvalidArgument("frame_rate") << "is negative";
    }

    frame_rate_ = frame_rate;

    // if the frame rate changed reconfigure the encoder
    const uint32_t new_frame_rate_num = static_cast<uint32_t>((frame_rate * FRAME_RATE_DENOMINATOR) + 0.5f);
    if (session_ && (init_params_.frameRateNum != new_frame_rate_num))
    {
        init_params_.frameRateNum = new_frame_rate_num;

        NV_ENC_RECONFIGURE_PARAMS reconfigure_params = {};

        reconfigure_params.version            = NV_ENC_RECONFIGURE_PARAMS_VER;
        reconfigure_params.reInitEncodeParams = init_params_;
        reconfigure_params.resetEncoder       = 1;
        reconfigure_params.forceIDR           = 1;

        NvEncCheck(NvEncReconfigureEncoder(session_.get(), &reconfigure_params));

        // a change in frame rate forces a new stream
        new_stream_ = true;
    }
}

void NvEncVideoEncoder::Impl::SetBitRate(uint32_t bit_rate)
{
    if (bit_rate == 0)
    {
        throw InvalidArgument("bit_rate") << "is zero";
    }

    bit_rate_ = bit_rate;

    // if the bit rate changed reconfigure the encoder
    if (session_ && (config_.rcParams.maxBitRate != bit_rate_))
    {
        config_.rcParams.maxBitRate     = bit_rate_;
        config_.rcParams.averageBitRate = bit_rate_;

        NV_ENC_RECONFIGURE_PARAMS reconfigure_params = {};

        reconfigure_params.version            = NV_ENC_RECONFIGURE_PARAMS_VER;
        reconfigure_params.reInitEncodeParams = init_params_;
        reconfigure_params.resetEncoder       = 1;
        reconfigure_params.forceIDR           = 1;

        NvEncCheck(NvEncReconfigureEncoder(session_.get(), &reconfigure_params));
    }
}

const std::shared_ptr<CudaContext> &NvEncVideoEncoder::Impl::GetCudaContext()
{
    return cuda_context_;
}

bool NvEncVideoEncoder::Impl::CheckEncodeGUID(const GUID &encode_guid)
{
    if (!session_)
    {
        throw InvalidState() << "There is no active encoder session";
    }

    // get the number of GUIDs supported by the interface
    uint32_t count = 0;
    NvEncCheck(NvEncGetEncodeGUIDCount(session_.get(), &count));

    // then get the GUIDs supported by the currently installed HW
    std::vector<GUID> guids(count);
    uint32_t valid_count = 0;
    NvEncCheck(NvEncGetEncodeGUIDs(session_.get(), guids.data(), count, &valid_count));
    // resize the array to contain the valid GUIDs only
    guids.resize(valid_count);

    Log(LogLevel::Debug) << "Supported encode codecs";
    for (auto &&guid : guids)
    {
        Log(LogLevel::Debug) << " " << guid;
    }

    auto found_guid = std::find(guids.begin(), guids.end(), encode_guid);
    return (found_guid != guids.end());
}

bool NvEncVideoEncoder::Impl::CheckPresetGUID(const GUID &preset_guid)
{
    if (!session_)
    {
        throw InvalidState() << "There is no active encoder session";
    }

    // get the number of GUIDs supported by the interface
    uint32_t count = 0;
    NvEncCheck(NvEncGetEncodePresetCount(session_.get(), init_params_.encodeGUID, &count));

    // then get the GUIDs supported by the currently installed HW
    std::vector<GUID> guids(count);
    uint32_t valid_count = 0;
    NvEncCheck(NvEncGetEncodePresetGUIDs(session_.get(), init_params_.encodeGUID, guids.data(), count, &valid_count));
    // resize the array to contain the valid GUIDs only
    guids.resize(valid_count);

    Log(LogLevel::Debug) << "Supported encoder presets";
    for (auto &&guid : guids)
    {
        Log(LogLevel::Debug) << " " << guid;
    }

    auto found_guid = std::find(guids.begin(), guids.end(), preset_guid);
    return (found_guid != guids.end());
}

bool NvEncVideoEncoder::Impl::CheckProfileGUID(const GUID &profile_guid)
{
    if (!session_)
    {
        throw InvalidState() << "There is no active encoder session";
    }

    // get the number of GUIDs supported by the interface
    uint32_t count = 0;
    NvEncCheck(NvEncGetEncodeProfileGUIDCount(session_.get(), init_params_.encodeGUID, &count));

    // then get the GUIDs supported by the currently installed HW
    std::vector<GUID> guids(count);
    uint32_t valid_count = 0;
    NvEncCheck(NvEncGetEncodeProfileGUIDs(session_.get(), init_params_.encodeGUID, guids.data(), count, &valid_count));
    // resize the array to contain the valid GUIDs only
    guids.resize(valid_count);

    Log(LogLevel::Debug) << "Supported encoder profiles";
    for (auto &&guid : guids)
    {
        Log(LogLevel::Debug) << " " << guid;
    }

    return std::binary_search(guids.begin(), guids.end(), profile_guid);
}

bool NvEncVideoEncoder::Impl::CheckRateControlMode(NV_ENC_PARAMS_RC_MODE rate_control_mode)
{
    const int supported_rc_mode_mask = QueryEncodeCap(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES);

    Log(LogLevel::Debug) << "Supported rate control modes";
    Log(LogLevel::Debug) << " NV_ENC_PARAMS_RC_CONSTQP";
    for (int index = 0; index < 31; ++index)
    {
        const NV_ENC_PARAMS_RC_MODE mode = (NV_ENC_PARAMS_RC_MODE)(1 << index);
        if (supported_rc_mode_mask & mode)
        {
            Log(LogLevel::Debug) << " " << mode;
        }
    }

    // NV_ENC_PARAMS_RC_CONSTQP is defined as 0 which can't be represented by a bitmask so it must be always supported
    return ((rate_control_mode == NV_ENC_PARAMS_RC_CONSTQP) || ((supported_rc_mode_mask & rate_control_mode) != 0));
}

const UniqueNvEncSession &NvEncVideoEncoder::Impl::GetSession()
{
    if (!session_)
    {
        // registered resources and the bitstream buffer depend on the session, if we have create a new session
        // all registrations are void
        registered_.clear();
        bitstream_buffer_.reset();

        // create the session
        session_.reset([this] {
            NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encode_session_ex_params = {};

            encode_session_ex_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;

            encode_session_ex_params.device     = cuda_context_->GetContext();
            encode_session_ex_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
            encode_session_ex_params.apiVersion = NVENCAPI_VERSION;

            void *session            = nullptr;
            const NVENCSTATUS status = NvEncOpenEncodeSessionEx(&encode_session_ex_params, &session);
            if (status != NV_ENC_SUCCESS)
            {
                Log(LogLevel::Debug) << "NvEnc not supported, reason " << status;
            }
            return session;
        }());

        // If NvEnc failed to initialize (e.g. because it's not supported on A100) return immediately.
        if (!session_)
        {
            return session_;
        }

        // Setup the encoder for low-latency use cases like game-streaming, video conferencing, etc.
        // according to the NvEnc manual:
        // * Low-latency high quality preset
        // * Rate control mode = CBR
        // * No B Frames
        // * Infinite GOP length
        // * Adaptive quantization (AQ) enabled
        // Not done although mentioned in the manual
        // * Very low VBV buffer size (single frame) -> results in frame to frame variations (noisy)

        // initialize the encoder
        std::list<GUID> encode_guids{NV_ENC_CODEC_H264_GUID};
        if (use_hevc_)
        {
            encode_guids.push_front(NV_ENC_CODEC_HEVC_GUID);
        }
        bool found_encode_id = false;
        for (auto &&encode_guid : encode_guids)
        {
            if (CheckEncodeGUID(encode_guid))
            {
                init_params_.encodeGUID = encode_guid;
                found_encode_id         = true;
                break;
            }
        }
        if (!found_encode_id)
        {
            std::stringstream msg;
            msg << "None of the encode codecs";
            for (auto &&guid : encode_guids)
            {
                msg << " " << guid;
            }
            msg << "is supported.";
            throw InvalidState() << msg.str();
        }

        Log(LogLevel::Debug) << "Selecting encode codec " << init_params_.encodeGUID;

        init_params_.presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
        if (!CheckPresetGUID(init_params_.presetGUID))
        {
            throw InvalidState() << "Encoder preset " << init_params_.presetGUID << " is not supported";
        }
        else
        {
            Log(LogLevel::Debug) << "Selecting encoder preset " << init_params_.presetGUID;
        }
        init_params_.encodeWidth  = width_;
        init_params_.encodeHeight = height_;
        init_params_.darWidth     = width_;
        init_params_.darHeight    = height_;
        init_params_.frameRateNum = static_cast<uint32_t>((frame_rate_ * FRAME_RATE_DENOMINATOR) + 0.5f);
        init_params_.frameRateDen = static_cast<uint32_t>(FRAME_RATE_DENOMINATOR + 0.5f);
        init_params_.enablePTD    = 1;
        init_params_.encodeConfig = &config_;

        Log(LogLevel::Info) << "NvEnc video encoder max resolution " << QueryEncodeCap(NV_ENC_CAPS_WIDTH_MAX) << "x"
                            << QueryEncodeCap(NV_ENC_CAPS_HEIGHT_MAX);

        // configure the preset
        NV_ENC_PRESET_CONFIG preset_config = {};

        preset_config.version           = NV_ENC_PRESET_CONFIG_VER;
        preset_config.presetCfg.version = NV_ENC_CONFIG_VER;

        NvEncCheck(NvEncGetEncodePresetConfig(session_.get(), init_params_.encodeGUID, init_params_.presetGUID,
                                              &preset_config));

        config_ = preset_config.presetCfg;

        if (init_params_.encodeGUID == NV_ENC_CODEC_H264_GUID)
        {
            config_.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
        }
        else if (init_params_.encodeGUID == NV_ENC_CODEC_HEVC_GUID)
        {
            config_.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
        }
        else
        {
            throw InvalidState() << "Unhandled encode coded " << init_params_.encodeGUID;
        }
        if (!CheckProfileGUID(config_.profileGUID))
        {
            throw InvalidState() << "Encoder profile " << config_.profileGUID << " is not supported";
        }
        Log(LogLevel::Debug) << "Selecting encoder profile " << config_.profileGUID;

        config_.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ;
        if (!CheckRateControlMode(config_.rcParams.rateControlMode))
        {
            throw InvalidState() << "Encoder rate control mode " << config_.rcParams.rateControlMode
                                 << " is not supported";
        }
        else
        {
            Log(LogLevel::Debug) << "Selecting encoder rate control mode " << config_.rcParams.rateControlMode;
        }
        config_.rcParams.maxBitRate     = bit_rate_;
        config_.rcParams.averageBitRate = bit_rate_;
        config_.rcParams.enableAQ       = 1;

        config_.gopLength      = NVENC_INFINITE_GOPLENGTH;
        config_.frameIntervalP = 1;

        if (init_params_.encodeGUID == NV_ENC_CODEC_H264_GUID)
        {
            config_.encodeCodecConfig.h264Config.repeatSPSPPS    = 1;
            config_.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
            config_.encodeCodecConfig.h264Config.idrPeriod       = NVENC_INFINITE_GOPLENGTH;
        }
        else if (init_params_.encodeGUID == NV_ENC_CODEC_HEVC_GUID)
        {
            config_.encodeCodecConfig.hevcConfig.repeatSPSPPS    = 1;
            config_.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1;
            config_.encodeCodecConfig.hevcConfig.idrPeriod       = NVENC_INFINITE_GOPLENGTH;
        }
        else
        {
            throw InvalidState() << "Unhandled encode coded " << init_params_.encodeGUID;
        }

        NvEncCheck(NvEncInitializeEncoder(session_.get(), &init_params_));

        // allocate the bit-stream buffer
        NV_ENC_CREATE_BITSTREAM_BUFFER create_bitstream_buffer = {};

        create_bitstream_buffer.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

        NvEncCheck(NvEncCreateBitstreamBuffer(session_.get(), &create_bitstream_buffer));
        bitstream_buffer_.reset(new BitstreamBuffer(session_, create_bitstream_buffer.bitstreamBuffer));
    }
    return session_;
}

int NvEncVideoEncoder::Impl::QueryEncodeCap(NV_ENC_CAPS cap)
{
    NV_ENC_CAPS_PARAM encodeCapsParam{};

    encodeCapsParam.version     = NV_ENC_CAPS_PARAM_VER;
    encodeCapsParam.capsToQuery = cap;

    int value = 0;
    NvEncCheck(NvEncGetEncodeCaps(GetSession().get(), init_params_.encodeGUID, &encodeCapsParam, &value));
    return value;
}

std::shared_ptr<VideoResource> NvEncVideoEncoder::Impl::RegisterResource(uint32_t width, uint32_t height,
                                                                         const std::shared_ptr<IBlob> &memory,
                                                                         IVideoEncoder::Format format)
{
    // NvENC can encode frames with even width/height only, therefore the resource size has to be even
    if ((width & 1 != 0) || (height & 1 != 0))
    {
        throw InvalidState() << "Resource width and height need to be evenly divisible by two, resource size is ("
                             << width << ", " << height << ")";
    }
    // NvEnc has a minimum size (unfortunately that value can't be queried).
    if ((width < VIDEO_ENCODER_MIN_WIDTH) || (height < VIDEO_ENCODER_MIN_HEIGHT))
    {
        throw InvalidState() << "Resource width and height has to be at least " << VIDEO_ENCODER_MIN_WIDTH << ", "
                             << VIDEO_ENCODER_MIN_HEIGHT << " resource size is (" << width << ", "
                             << height << ")";
    }

    NV_ENC_REGISTER_RESOURCE resource = {NV_ENC_REGISTER_RESOURCE_VER};

    std::unique_ptr<IBlob::AccessGuardConst> access = memory->AccessConst();

    resource.resourceType       = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    resource.resourceToRegister = const_cast<void *>(access->GetData());
    resource.width              = width;
    resource.height             = height;
    resource.pitch              = width * BytesPerPixel(format);
    resource.bufferFormat       = ToNvEncFormat(format);

    // check if it is already registered
    auto it = std::find_if(registered_.begin(), registered_.end(),
                           [&resource](const std::shared_ptr<VideoResource> &video_resource) {
                               return (resource.resourceToRegister == video_resource->resource_.resourceToRegister) &&
                                      (resource.resourceType == video_resource->resource_.resourceType) &&
                                      (resource.width == video_resource->resource_.width) &&
                                      (resource.height == video_resource->resource_.height) &&
                                      (resource.pitch == video_resource->resource_.pitch) &&
                                      (resource.bufferFormat == video_resource->resource_.bufferFormat);
                           });

    if (it == registered_.end())
    {
        // new resource, register
        auto &session = GetSession();
        NvEncCheck(NvEncRegisterResource(session.get(), &resource));

        auto video_resource = std::make_shared<VideoResource>(session, resource);
        registered_.emplace_back(video_resource);
        return video_resource;
    }
    else
    {
        return *it;
    }
}

void NvEncVideoEncoder::Impl::Encode(uint32_t width, uint32_t height, const std::shared_ptr<IBlob> &memory, IVideoEncoder::Format format)
{
    if (!memory)
    {
        throw InvalidArgument("memory") << "is a nullptr";
    }
    if (!stream_)
    {
        throw InvalidState() << "output stream is not set";
    }

    SetResolution(width, height);

    const std::shared_ptr<VideoResource> enc_resource = RegisterResource(width, height, memory, format);

    // map the input resource
    NV_ENC_MAP_INPUT_RESOURCE map_input_resource = {};

    map_input_resource.version            = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map_input_resource.registeredResource = enc_resource->resource_.registeredResource;
    NvEncCheck(NvEncMapInputResource(GetSession().get(), &map_input_resource));
    // create a guard to automatically unmap the input resource on exit
    Guard unmap_input_resource([session = GetSession().get(), mapped_resource = map_input_resource.mappedResource] {
        NvEncCheck(NvEncUnmapInputResource(session, mapped_resource));
    });

    // do the encoding
    NV_ENC_PIC_PARAMS pic_params = {};

    pic_params.version         = NV_ENC_PIC_PARAMS_VER;
    pic_params.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;
    pic_params.inputBuffer     = map_input_resource.mappedResource;
    pic_params.bufferFmt       = map_input_resource.mappedBufferFmt;
    pic_params.inputWidth      = enc_resource->resource_.width;
    pic_params.inputHeight     = enc_resource->resource_.height;
    pic_params.outputBitstream = bitstream_buffer_->bitstream_buffer_;

    // if the stream has changed then force an IDR picture and reset the MP4 stream so
    // that the header is re-written for the new stream
    if (new_stream_)
    {
        mp4_wrapper_.ResetStream();
        pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        new_stream_  = false;
        frame_index_ = 0;

        // tell the stream that a new stream begins
        stream_->NewStream();
    }
    // force an IDR picture every second
    if (frame_index_ > frame_rate_)
    {
        pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
        frame_index_ = 0;
    }
    else
    {
        ++frame_index_;
    }

    NvEncCheck(NvEncEncodePicture(GetSession().get(), &pic_params));

    NV_ENC_LOCK_BITSTREAM lock_bitstream_data = {};

    lock_bitstream_data.version         = NV_ENC_LOCK_BITSTREAM_VER;
    lock_bitstream_data.outputBitstream = bitstream_buffer_->bitstream_buffer_;
    lock_bitstream_data.doNotWait       = false;

    NvEncCheck(NvEncLockBitstream(GetSession().get(), &lock_bitstream_data));
    // create a guard to automatically unlock the bitstream buffer on exit
    Guard unlock_bitstream([session = GetSession().get(), bitstream_buffer = bitstream_buffer_->bitstream_buffer_] {
        NvEncCheck(NvEncUnlockBitstream(session, bitstream_buffer));
    });

    // wrap into MP4 stream
    std::vector<uint8_t> mp4_buffer;

    // write the video frame
    std::vector<uint8_t> host_buffer(lock_bitstream_data.bitstreamSizeInBytes);
    std::memcpy(host_buffer.data(), lock_bitstream_data.bitstreamBufferPtr, lock_bitstream_data.bitstreamSizeInBytes);

    MP4Wrapper::Type type;
    if (init_params_.encodeGUID == NV_ENC_CODEC_H264_GUID)
    {
        type = MP4Wrapper::Type::H264;
    }
    else if (init_params_.encodeGUID == NV_ENC_CODEC_HEVC_GUID)
    {
        type = MP4Wrapper::Type::HEVC;
    }
    else
    {
        throw InvalidState() << "Unhandled encode coded " << init_params_.encodeGUID;
    }

    mp4_wrapper_.Wrap(width_, height_, frame_rate_, type, host_buffer, mp4_buffer);

    // write the data out
    stream_->Write(reinterpret_cast<char *>(mp4_buffer.data()), mp4_buffer.size());
}

} // namespace clara::viz
