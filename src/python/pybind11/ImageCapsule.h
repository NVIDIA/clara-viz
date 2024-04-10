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

#pragma once

#include <stdint.h>

#include <memory>

#include <claraviz/util/Blob.h>

namespace clara::viz
{

class ImageEncodedDataMessage;
class ImageCapsule
{
public:
    std::shared_ptr<const ImageEncodedDataMessage> image;
    uint32_t width;
    uint32_t height;
    std::unique_ptr<IBlob::AccessGuardConst> access;
};

} // namespace clara::viz
