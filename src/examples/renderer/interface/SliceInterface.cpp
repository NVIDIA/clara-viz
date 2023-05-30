/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "interface/SliceInterface.h"

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(SliceInterface::Message);

namespace detail
{

template<>
SliceInterface::DataIn::SliceInterfaceData()
    : slice(Vector3f(0.f, 0.f, 0.f), [this](const Vector3f &value) {
        ValidatorMinMaxInclusive(value, Vector3f(0.f, 0.f, 0.f), Vector3f(1.f, 1.f, 1.f), "slice location");
    })
{
}

template<>
SliceInterface::DataOut::SliceInterfaceData()
{
}

} // namespace detail

/**
 * Copy a slice interface structure to a slice POD structure.
 */
template<>
SliceInterface::DataOut SliceInterface::Get()
{
    AccessGuardConst access(this);

    SliceInterface::DataOut data_out;
    data_out.slice = access->slice.Get();

    return data_out;
}

} // namespace clara::viz
