/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <nvjpeg.h>

#include "claraviz/util/Exception.h"
#include "claraviz/util/Singleton.h"
#include "claraviz/util/UniqueObj.h"

/**
 * Operator that appends the string representation of a nvjpegStatus_t to a stream.
 */
std::ostream &operator<<(std::ostream &os, const nvjpegStatus_t status);

namespace clara::viz
{

/**
 * NvJpeg error check helper
 */
#define NvJpegCheck(FUNC)                                          \
    {                                                              \
        const nvjpegStatus_t status = FUNC;                        \
        if (status != NVJPEG_STATUS_SUCCESS)                       \
        {                                                          \
            throw RuntimeError() << "NvJpeg API error " << status; \
        }                                                          \
    }

/**
 * UniqueObj's for NvJpeg objects
 */
/**@{*/
using UniqueNvJpegInstance = UniqueObj<nvjpegHandle, decltype(&nvjpegDestroy), &nvjpegDestroy>;
using UniqueNvJpegEncoderState =
    UniqueObj<nvjpegEncoderState, decltype(&nvjpegEncoderStateDestroy), &nvjpegEncoderStateDestroy>;
using UniqueNvJpegEncoderParams =
    UniqueObj<nvjpegEncoderParams, decltype(&nvjpegEncoderParamsDestroy), &nvjpegEncoderParamsDestroy>;
/**@}*/

} // namespace clara::viz
