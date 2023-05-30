/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/hardware/nvenc/NvEncService.h"

#include <algorithm>
#include <cstring>
#include <iomanip>

#include <dlfcn.h>

#include "claraviz/util/Log.h"
#include "claraviz/util/UniqueObj.h"

bool operator==(const GUID &lhs, const GUID &rhs) noexcept
{
    return (std::memcmp(&lhs, &rhs, sizeof(GUID)) == 0);
}

std::ostream &operator<<(std::ostream &os, const GUID &guid)
{
    // encode codec
    if (guid == NV_ENC_CODEC_H264_GUID)
    {
        os << std::string("NV_ENC_CODEC_H264_GUID");
    }
    else if (guid == NV_ENC_CODEC_HEVC_GUID)
    {
        os << std::string("NV_ENC_CODEC_HEVC_GUID");
    }
    // encoder presets
    else if (guid == NV_ENC_PRESET_DEFAULT_GUID)
    {
        os << std::string("NV_ENC_PRESET_DEFAULT_GUID");
    }
    else if (guid == NV_ENC_PRESET_HP_GUID)
    {
        os << std::string("NV_ENC_PRESET_HP_GUID");
    }
    else if (guid == NV_ENC_PRESET_HQ_GUID)
    {
        os << std::string("NV_ENC_PRESET_HQ_GUID");
    }
    else if (guid == NV_ENC_PRESET_BD_GUID)
    {
        os << std::string("NV_ENC_PRESET_BD_GUID");
    }
    else if (guid == NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID)
    {
        os << std::string("NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID");
    }
    else if (guid == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID)
    {
        os << std::string("NV_ENC_PRESET_LOW_LATENCY_HQ_GUID");
    }
    else if (guid == NV_ENC_PRESET_LOW_LATENCY_HP_GUID)
    {
        os << std::string("NV_ENC_PRESET_LOW_LATENCY_HP_GUID");
    }
    else if (guid == NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID)
    {
        os << std::string("NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID");
    }
    else if (guid == NV_ENC_PRESET_LOSSLESS_HP_GUID)
    {
        os << std::string("NV_ENC_PRESET_LOSSLESS_HP_GUID");
    }
    // encoder profiles
    else if (guid == NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID)
    {
        os << std::string("NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_BASELINE_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_BASELINE_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_MAIN_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_MAIN_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_HIGH_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_HIGH_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_HIGH_444_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_HIGH_444_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_STEREO_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_STEREO_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_SVC_TEMPORAL_SCALABILTY)
    {
        os << std::string("NV_ENC_H264_PROFILE_SVC_TEMPORAL_SCALABILTY");
    }
    else if (guid == NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID");
    }
    else if (guid == NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID)
    {
        os << std::string("NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID");
    }
    else if (guid == NV_ENC_HEVC_PROFILE_MAIN_GUID)
    {
        os << std::string("NV_ENC_HEVC_PROFILE_MAIN_GUID");
    }
    else if (guid == NV_ENC_HEVC_PROFILE_MAIN10_GUID)
    {
        os << std::string("NV_ENC_HEVC_PROFILE_MAIN10_GUID");
    }
    else if (guid == NV_ENC_HEVC_PROFILE_FREXT_GUID)
    {
        os << std::string("NV_ENC_HEVC_PROFILE_FREXT_GUID");
    }
    else
    {
        os << std::hex << std::setfill('0') << std::setw(8) << guid.Data1 << "-" << std::setw(4) << guid.Data2 << "-"
           << guid.Data3 << "-" << std::setw(16) << *reinterpret_cast<const uint64_t *>(&guid.Data4[0]);
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const NV_ENC_PARAMS_RC_MODE rc_mode)
{
    switch (rc_mode)
    {
    case NV_ENC_PARAMS_RC_CONSTQP:
        os << std::string("NV_ENC_PARAMS_RC_CONSTQP");
        break;
    case NV_ENC_PARAMS_RC_VBR:
        os << std::string("NV_ENC_PARAMS_RC_VBR");
        break;
    case NV_ENC_PARAMS_RC_CBR:
        os << std::string("NV_ENC_PARAMS_RC_CBR");
        break;
    case NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ:
        os << std::string("NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ");
        break;
    case NV_ENC_PARAMS_RC_CBR_HQ:
        os << std::string("NV_ENC_PARAMS_RC_CBR_HQ");
        break;
    case NV_ENC_PARAMS_RC_VBR_HQ:
        os << std::string("NV_ENC_PARAMS_RC_VBR_HQ");
        break;
    default:
        os << (int)rc_mode;
        break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const NVENCSTATUS status)
{
    switch (status)
    {
    case NV_ENC_SUCCESS:
        os << std::string("NV_ENC_SUCCESS");
        break;
    case NV_ENC_ERR_NO_ENCODE_DEVICE:
        os << std::string("NV_ENC_ERR_NO_ENCODE_DEVICE");
        break;
    case NV_ENC_ERR_UNSUPPORTED_DEVICE:
        os << std::string("NV_ENC_ERR_UNSUPPORTED_DEVICE");
        break;
    case NV_ENC_ERR_INVALID_ENCODERDEVICE:
        os << std::string("NV_ENC_ERR_INVALID_ENCODERDEVICE");
        break;
    case NV_ENC_ERR_INVALID_DEVICE:
        os << std::string("NV_ENC_ERR_INVALID_DEVICE");
        break;
    case NV_ENC_ERR_DEVICE_NOT_EXIST:
        os << std::string("NV_ENC_ERR_DEVICE_NOT_EXIST");
        break;
    case NV_ENC_ERR_INVALID_PTR:
        os << std::string("NV_ENC_ERR_INVALID_PTR");
        break;
    case NV_ENC_ERR_INVALID_EVENT:
        os << std::string("NV_ENC_ERR_INVALID_EVENT");
        break;
    case NV_ENC_ERR_INVALID_PARAM:
        os << std::string("NV_ENC_ERR_INVALID_PARAM");
        break;
    case NV_ENC_ERR_INVALID_CALL:
        os << std::string("NV_ENC_ERR_INVALID_CALL");
        break;
    case NV_ENC_ERR_OUT_OF_MEMORY:
        os << std::string("NV_ENC_ERR_OUT_OF_MEMORY");
        break;
    case NV_ENC_ERR_ENCODER_NOT_INITIALIZED:
        os << std::string("NV_ENC_ERR_ENCODER_NOT_INITIALIZED");
        break;
    case NV_ENC_ERR_UNSUPPORTED_PARAM:
        os << std::string("NV_ENC_ERR_UNSUPPORTED_PARAM");
        break;
    case NV_ENC_ERR_LOCK_BUSY:
        os << std::string("NV_ENC_ERR_LOCK_BUSY");
        break;
    case NV_ENC_ERR_NOT_ENOUGH_BUFFER:
        os << std::string("NV_ENC_ERR_NOT_ENOUGH_BUFFER");
        break;
    case NV_ENC_ERR_INVALID_VERSION:
        os << std::string("NV_ENC_ERR_INVALID_VERSION");
        break;
    case NV_ENC_ERR_MAP_FAILED:
        os << std::string("NV_ENC_ERR_MAP_FAILED");
        break;
    case NV_ENC_ERR_NEED_MORE_INPUT:
        os << std::string("NV_ENC_ERR_NEED_MORE_INPUT");
        break;
    case NV_ENC_ERR_ENCODER_BUSY:
        os << std::string("NV_ENC_ERR_ENCODER_BUSY");
        break;
    case NV_ENC_ERR_EVENT_NOT_REGISTERD:
        os << std::string("NV_ENC_ERR_EVENT_NOT_REGISTERD");
        break;
    case NV_ENC_ERR_GENERIC:
        os << std::string("NV_ENC_ERR_GENERIC");
        break;
    case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY:
        os << std::string("NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY");
        break;
    case NV_ENC_ERR_UNIMPLEMENTED:
        os << std::string("NV_ENC_ERR_UNIMPLEMENTED");
        break;
    case NV_ENC_ERR_RESOURCE_REGISTER_FAILED:
        os << std::string("NV_ENC_ERR_RESOURCE_REGISTER_FAILED");
        break;
    case NV_ENC_ERR_RESOURCE_NOT_REGISTERED:
        os << std::string("NV_ENC_ERR_RESOURCE_NOT_REGISTERED");
        break;
    case NV_ENC_ERR_RESOURCE_NOT_MAPPED:
        os << std::string("NV_ENC_ERR_RESOURCE_NOT_MAPPED");
        break;
    default:
        os << (int)status;
        break;
    }
    return os;
}

namespace clara::viz
{

struct NvEncService::Impl
{
    ~Impl() = default;

    UniqueObj<void, decltype(&dlclose), &dlclose> nvenc_lib; ///< NvEnc lib handle

    NV_ENCODE_API_FUNCTION_LIST nvenc; ///< NvEnc API
};

NvEncService::NvEncService()
    : impl_(new Impl)
{
    // open the encode lib
    impl_->nvenc_lib.reset(dlopen("libnvidia-encode.so.1", RTLD_LAZY));
    if (!impl_->nvenc_lib)
    {
        throw RuntimeError() << "Could not open nvEncodeAPI library";
    }

    // check the maximum version supported by the encoder
    using NvEncodeAPIGetMaxSupportedVersionType = NVENCSTATUS(NVENCAPI *)(uint32_t *);
    NvEncodeAPIGetMaxSupportedVersionType NvEncodeAPIGetMaxSupportedVersion =
        reinterpret_cast<NvEncodeAPIGetMaxSupportedVersionType>(
            dlsym(impl_->nvenc_lib.get(), "NvEncodeAPIGetMaxSupportedVersion"));
    if (!NvEncodeAPIGetMaxSupportedVersion)
    {
        throw RuntimeError() << "Cannot find NvEncodeAPIGetMaxSupportedVersion() entry in NVENC library";
    }

    uint32_t version = 0;
    NvEncCheck(NvEncodeAPIGetMaxSupportedVersion(&version));
    if (((NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION) > version)
    {
        throw RuntimeError() << "Required NvEncode API version is " << NVENCAPI_MAJOR_VERSION << "."
                             << NVENCAPI_MINOR_VERSION << "but found " << (version >> 4) << "." << (version & 0xF);
    }
    Log(LogLevel::Info) << "Using NvEncode API " << (version >> 4) << "." << (version & 0xF);

    // create the NVENC instance
    using NvEncodeAPICreateInstanceType = NVENCSTATUS(NVENCAPI *)(NV_ENCODE_API_FUNCTION_LIST *);
    NvEncodeAPICreateInstanceType NvEncodeAPICreateInstance =
        reinterpret_cast<NvEncodeAPICreateInstanceType>(dlsym(impl_->nvenc_lib.get(), "NvEncodeAPICreateInstance"));
    if (!NvEncodeAPICreateInstance)
    {
        throw RuntimeError() << "Cannot find NvEncodeAPICreateInstance() entry in NVENC library";
    }

    impl_->nvenc = {NV_ENCODE_API_FUNCTION_LIST_VER};
    NvEncCheck(NvEncodeAPICreateInstance(&impl_->nvenc));
}

NvEncService::~NvEncService() {}

const NV_ENCODE_API_FUNCTION_LIST &NvEncService::GetApi() const
{
    return impl_->nvenc;
}

NVENCSTATUS NvEncDestroyEncoder(void *encoder)
{
    return NvEncService::GetInstance().GetApi().nvEncDestroyEncoder(encoder);
}

} // namespace clara::viz
