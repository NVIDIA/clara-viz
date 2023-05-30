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

#include "claraviz/hardware/nvenc/nvEncodeAPI.h"
#include "claraviz/util/Exception.h"
#include "claraviz/util/Singleton.h"
#include "claraviz/util/UniqueObj.h"

/**
 * Compares NvEnc GUID's
 */
bool operator==(const GUID &lhs, const GUID &rhs) noexcept;

/**
 * Operator that appends the string representation of a GUID to a stream.
 */
std::ostream &operator<<(std::ostream &os, const GUID &guid);

/**
 * Operator that appends the string representation of a NV_ENC_PARAMS_RC_MODE to a stream.
 */
std::ostream &operator<<(std::ostream &os, const NV_ENC_PARAMS_RC_MODE rc_mode);

/**
 * Operator that appends the string representation of a NVENCSTATUS to a stream.
 */
std::ostream &operator<<(std::ostream &os, const NVENCSTATUS status);

namespace clara::viz
{

/**
 * NVENC error check helper
 */
#define NvEncCheck(FUNC)                                             \
    {                                                                \
        const NVENCSTATUS status = FUNC;                             \
        if (status != NV_ENC_SUCCESS)                                \
        {                                                            \
            throw RuntimeError() << "NvEncode API error " << status; \
        }                                                            \
    }

/**
 * Call helpers for the NvEncode API
 */
/**@{*/
#define NvEncGetEncodeGUIDCount (NvEncService::GetInstance().GetApi().nvEncGetEncodeGUIDCount)
#define NvEncGetEncodeProfileGUIDCount (NvEncService::GetInstance().GetApi().nvEncGetEncodeProfileGUIDCount)
#define NvEncGetEncodeProfileGUIDs (NvEncService::GetInstance().GetApi().nvEncGetEncodeProfileGUIDs)
#define NvEncGetEncodeGUIDs (NvEncService::GetInstance().GetApi().nvEncGetEncodeGUIDs)
#define NvEncGetInputFormatCount (NvEncService::GetInstance().GetApi().nvEncGetInputFormatCount)
#define NvEncGetInputFormats (NvEncService::GetInstance().GetApi().nvEncGetInputFormats)
#define NvEncGetEncodeCaps (NvEncService::GetInstance().GetApi().nvEncGetEncodeCaps)
#define NvEncGetEncodePresetCount (NvEncService::GetInstance().GetApi().nvEncGetEncodePresetCount)
#define NvEncGetEncodePresetGUIDs (NvEncService::GetInstance().GetApi().nvEncGetEncodePresetGUIDs)
#define NvEncGetEncodePresetConfig (NvEncService::GetInstance().GetApi().nvEncGetEncodePresetConfig)
#define NvEncInitializeEncoder (NvEncService::GetInstance().GetApi().nvEncInitializeEncoder)
#define NvEncCreateInputBuffer (NvEncService::GetInstance().GetApi().nvEncCreateInputBuffer)
#define NvEncDestroyInputBuffer (NvEncService::GetInstance().GetApi().nvEncDestroyInputBuffer)
#define NvEncCreateBitstreamBuffer (NvEncService::GetInstance().GetApi().nvEncCreateBitstreamBuffer)
#define NvEncDestroyBitstreamBuffer (NvEncService::GetInstance().GetApi().nvEncDestroyBitstreamBuffer)
#define NvEncEncodePicture (NvEncService::GetInstance().GetApi().nvEncEncodePicture)
#define NvEncLockBitstream (NvEncService::GetInstance().GetApi().nvEncLockBitstream)
#define NvEncUnlockBitstream (NvEncService::GetInstance().GetApi().nvEncUnlockBitstream)
#define NvEncLockInputBuffer (NvEncService::GetInstance().GetApi().nvEncLockInputBuffer)
#define NvEncUnlockInputBuffer (NvEncService::GetInstance().GetApi().nvEncUnlockInputBuffer)
#define NvEncGetEncodeStats (NvEncService::GetInstance().GetApi().nvEncGetEncodeStats)
#define NvEncGetSequenceParams (NvEncService::GetInstance().GetApi().nvEncGetSequenceParams)
#define NvEncRegisterAsyncEvent (NvEncService::GetInstance().GetApi().nvEncRegisterAsyncEvent)
#define NvEncUnregisterAsyncEvent (NvEncService::GetInstance().GetApi().nvEncUnregisterAsyncEvent)
#define NvEncMapInputResource (NvEncService::GetInstance().GetApi().nvEncMapInputResource)
#define NvEncUnmapInputResource (NvEncService::GetInstance().GetApi().nvEncUnmapInputResource)
NVENCSTATUS NvEncDestroyEncoder(void *encoder);
#define NvEncInvalidateRefFrames (NvEncService::GetInstance().GetApi().nvEncInvalidateRefFrames)
#define NvEncOpenEncodeSessionEx (NvEncService::GetInstance().GetApi().nvEncOpenEncodeSessionEx)
#define NvEncRegisterResource (NvEncService::GetInstance().GetApi().nvEncRegisterResource)
#define NvEncUnregisterResource (NvEncService::GetInstance().GetApi().nvEncUnregisterResource)
#define NvEncReconfigureEncoder (NvEncService::GetInstance().GetApi().nvEncReconfigureEncoder)
/**@}*/

/**
 * UniqueObj's for a NvEnc objects
 */
/**@{*/
using UniqueNvEncSession = UniqueObj<void, decltype(&NvEncDestroyEncoder), &NvEncDestroyEncoder>;
/**@}*/

/**
 * Global NvENC service class.
 */
class NvEncService : public Singleton<NvEncService>
{
public:
    NvEncService();
    ~NvEncService();

    /**
     * @returns the NvENC api function list
     */
    const NV_ENCODE_API_FUNCTION_LIST &GetApi() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clara::viz
