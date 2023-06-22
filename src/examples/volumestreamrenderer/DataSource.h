/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <claraviz/util/VectorT.h>
#include <claraviz/interface/DataInterface.h>
#include <claraviz/interface/CameraInterface.h>
#include <claraviz/interface/LightInterface.h>
#include <claraviz/interface/TransferFunctionInterface.h>

namespace clara::viz
{

class DataSource
{
public:
    DataSource()          = default;
    virtual ~DataSource() = default;

    virtual float FrameRate() = 0;

    virtual std::shared_ptr<IBlob> NextBlob(bool benchmark_mode) = 0;
    virtual void ReturnBlob(const std::shared_ptr<IBlob> &blob)  = 0;

    virtual void Light(LightInterface &interface) {}
    virtual void BackgroundLight(BackgroundLightInterface &interface) {}
    virtual void Camera(CameraInterface &interface) {}
    virtual void TransferFunction(TransferFunctionInterface &interface) {}

    Vector3ui volume_size_;
    Vector3f volume_element_spacing_;
    DataElementType volume_type_;
    uint32_t volume_bytes_per_element_;
    Vector2f volume_element_range_;
};

} // namespace clara::viz