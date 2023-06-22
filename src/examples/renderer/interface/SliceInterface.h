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

#pragma once

#include <claraviz/interface/InterfaceData.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

namespace detail
{

template<template<typename> typename V>
struct SliceInterfaceData
{
    SliceInterfaceData();

    /// Slice location
    V<Vector3f> slice;
};

using SliceInterfaceDataIn = SliceInterfaceData<InterfaceValueT>;

using SliceInterfaceDataOut = SliceInterfaceData<InterfaceDirectT>;

struct SliceInterfaceDataPrivate
{
};

} // namespace detail

using SliceInterface =
    InterfaceData<detail::SliceInterfaceDataIn, detail::SliceInterfaceDataOut, detail::SliceInterfaceDataPrivate>;

} // namespace clara::viz
