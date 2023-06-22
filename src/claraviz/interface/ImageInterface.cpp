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

#include "claraviz/interface/ImageInterface.h"

#include <limits>

#include "claraviz/util/Validator.h"

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(ImageInterface::Message);

template<>
ImageInterface::DataIn::ImageInterfaceData()
    : width(64,
            [](const uint32_t value) {
                ValidatorMinMaxInclusive(value, 64u, static_cast<uint32_t>(std::numeric_limits<int>::max()),
                                         "Image width");
            })
    , height(64,
             [](const uint32_t value) {
                 ValidatorMinMaxInclusive(value, 64u, static_cast<uint32_t>(std::numeric_limits<int>::max()),
                                          "Image height");
             })
    , color_type(ColorImageType::UNKNOWN)
    , depth_type(DepthImageType::UNKNOWN)
    , jpeg_quality(75, [](const uint32_t value) { ValidatorMinMaxInclusive(value, 1u, 100u, "Jpeg quality"); })
{
}

template<>
ImageInterface::DataOut::ImageInterfaceData()
{
}

/**
 * Copy a image interface structure to a image POD structure.
 */
template<>
ImageInterface::DataOut ImageInterface::Get()
{
    AccessGuardConst access(this);

    ImageInterface::DataOut data_out;
    data_out.view_name    = access->view_name;
    data_out.width        = access->width.Get();
    data_out.height       = access->height.Get();
    data_out.color_type   = access->color_type;
    data_out.color_memory = access->color_memory;
    data_out.depth_type   = access->depth_type;
    data_out.depth_memory = access->depth_memory;
    data_out.jpeg_quality = access->jpeg_quality.Get();

    return data_out;
}

std::shared_ptr<const ImageEncodedDataMessage> ImageInterfaceOutput::WaitForEncodedData()
{
    Wait();
    auto message = DequeueMessage();
    if (message->GetID() != ImageEncodedDataMessage::id_)
    {
        throw InvalidState() << "Unexpected message: " << message->GetID().GetName();
    }

    return std::static_pointer_cast<const ImageEncodedDataMessage>(message);
}

} // namespace clara::viz
