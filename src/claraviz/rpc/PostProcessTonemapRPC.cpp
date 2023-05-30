/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/rpc/PostProcessTonemapRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void PostProcessTonemapContext::ExecuteRPC(cinematic_v1::PostProcessTonemapRequest &request,
                                           cinematic_v1::PostProcessTonemapResponse &response)
{
    PostProcessTonemapInterface::AccessGuard access(GetResources()->tonemap_);

    switch (request.enable())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        access->enable = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        access->enable = false;
        break;
    }

    access->exposure.Set(request.exposure());
}

} // namespace detail

} // namespace clara::viz
