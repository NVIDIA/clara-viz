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

#include "claraviz/rpc/ImageRPC.h"

#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/util/Blob.h"

#include <limits>

namespace clara::viz
{

namespace image_v1 = nvidia::claraviz::image::v1;

namespace detail
{

void ImageGenerateContext::ExecuteRPC(image_v1::GenerateRequest &request, image_v1::GenerateResponse &response)
{
    // send the request to generate an image
    {
        ImageInterface::AccessGuard access(GetResources()->image_interface_);

        if (request.width())
        {
            access->width.Set(request.width());
        }
        if (request.height())
        {
            access->height.Set(request.height());
        }
        switch (request.type())
        {
        case image_v1::IMAGE_TYPE_RAW_RGBA_U8:
            access->color_type = ColorImageType::RAW_RGBA_U8;
            break;
        case image_v1::IMAGE_TYPE_JPEG:
            access->color_type = ColorImageType::JPEG;
            if (request.has_jpeg())
            {
                if (request.jpeg().quality())
                {
                    access->jpeg_quality.Set(request.jpeg().quality());
                }
            }
            break;
        case image_v1::IMAGE_TYPE_UNKNOWN:
            break;
        default:
            throw InvalidArgument("type") << "Unhandled image type " << static_cast<int>(request.type());
        }

        // the message is sent when the AccessGuard gets out of scope
    }

    // then wait for the message with the encoded data
    std::shared_ptr<const ImageEncodedDataMessage> encoded_data_message =
        GetResources()->image_interface_output_->WaitForEncodedData();

    if (encoded_data_message->color_memory_)
    {
        // and copy the encoded data to the response
        std::unique_ptr<IBlob::AccessGuardConst> access = encoded_data_message->color_memory_->AccessConst();
        const size_t size = encoded_data_message->color_memory_->GetSize();
        if (request.type() != image_v1::IMAGE_TYPE_JPEG)
        {
            // copy to host
            /// @todo add support copy to host to image interface
            std::unique_ptr<uint8_t> data(new uint8_t[size]);
            CudaCheck(cuInit(0));
            CudaPrimaryContext cuda_context(0);
            CudaCheck(cuMemcpy(reinterpret_cast<CUdeviceptr>(data.get()),
                               reinterpret_cast<CUdeviceptr>(access->GetData()),
                               size));
            response.set_data(data.get(), size);
        }
        else
        {
            response.set_data(access->GetData(), size);
        }
    }
    switch (encoded_data_message->color_type_)
    {
    case ColorImageType::RAW_RGBA_U8:
        response.set_type(image_v1::IMAGE_TYPE_RAW_RGBA_U8);
        break;
    case ColorImageType::JPEG:
        response.set_type(image_v1::IMAGE_TYPE_JPEG);
        break;
    default:
        throw InvalidState() << "Unhandled image type " << static_cast<int>(encoded_data_message->color_type_);
    }
}

void ImageQueryLimitsContext::ExecuteRPC(image_v1::QueryLimitsRequest &request, image_v1::QueryLimitsResponse &response)
{
    /// @todo NvRTVol does not support images smaller than 64x64
    response.set_min_image_width(64);
    response.set_min_image_height(64);
    response.set_max_image_width(std::numeric_limits<int>::max());
    response.set_max_image_height(std::numeric_limits<int>::max());
}

} // namespace detail

} // namespace clara::viz
