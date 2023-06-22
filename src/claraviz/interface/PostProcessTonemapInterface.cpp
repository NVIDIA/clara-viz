/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/PostProcessTonemapInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(PostProcessTonemapInterface::Message);

template<>
PostProcessTonemapInterface::DataIn::PostProcessTonemapInterfaceData()
    : enable(false)
    , exposure(0.5f, [](const float value) { ValidatorMinInclusive(value, 0.f, "Exposure"); })
{
}

template<>
PostProcessTonemapInterface::DataOut::PostProcessTonemapInterfaceData()
{
}

/**
 * Copy a post process tonemap config interface structure to a post process tonemap config POD structure.
 */
template<>
PostProcessTonemapInterface::DataOut PostProcessTonemapInterface::Get()
{
    AccessGuardConst access(this);

    PostProcessTonemapInterface::DataOut data_out;
    data_out.enable   = access->enable;
    data_out.exposure = access->exposure.Get();

    return data_out;
}

} // namespace clara::viz
