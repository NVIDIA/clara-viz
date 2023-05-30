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

#include "DataSourceCT.h"

#include <claraviz/util/CudaMemoryBlob.h>
#include <claraviz/util/MHDLoader.h>

#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <future>
#include <queue>

namespace clara::viz
{

class DataSourceCT::Impl
{
public:
    std::queue<std::shared_ptr<IBlob>> blobs_;
};

DataSourceCT::DataSourceCT(bool stream_from_cpu, const std::string input_dir)
    : impl_(new Impl)
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    if (!input_dir.empty())
    {
        // load the files
        std::list<std::string> file_names;

        for (auto &entry :
             std::experimental::filesystem::directory_iterator(std::experimental::filesystem::path(input_dir)))
            if (entry.path().extension().string() == ".mhd")
            {
                file_names.push_back(entry.path().string());
            }

        file_names.sort([](const std::string &a, const std::string &b) { return a < b; });

        Log(LogLevel::Info) << "Loading " << file_names.size() << " volumes ...";

        std::list<std::future<void>> tasks;
        std::list<std::unique_ptr<MHDLoader>> files;
        for (auto it = file_names.begin(); it != file_names.end(); ++it)
        {
            files.emplace_back(new MHDLoader());
            tasks.push_back(std::async(
                std::launch::async,
                [this](auto loader, const std::string &file) {
                    (*loader)->Load(file);
                    Log(LogLevel::Info) << "Loaded " << file;
                },
                --files.end(), *it));
        }

        std::for_each(tasks.begin(), tasks.end(), [](std::future<void> &future) { future.get(); });

        volume_size_            = files.front()->GetSize();
        volume_element_spacing_ = files.front()->GetElementSpacing();
        switch (files.front()->GetElementType())
        {
        case MHDLoader::ElementType::INT8:
            volume_type_ = DataElementType::INT8;
            break;
        case MHDLoader::ElementType::UINT8:
            volume_type_ = DataElementType::UINT8;
            break;
        case MHDLoader::ElementType::INT16:
            volume_type_ = DataElementType::INT16;
            break;
        case MHDLoader::ElementType::UINT16:
            volume_type_ = DataElementType::UINT16;
            break;
        case MHDLoader::ElementType::INT32:
            volume_type_ = DataElementType::INT32;
            break;
        case MHDLoader::ElementType::UINT32:
            volume_type_ = DataElementType::UINT32;
            break;
        case MHDLoader::ElementType::FLOAT:
            volume_type_ = DataElementType::FLOAT;
            break;
        default:
            throw RuntimeError() << "Unhandled element type " << files.front()->GetElementType();
        }
        volume_bytes_per_element_ = files.front()->GetBytesPerElement();
        volume_element_range_     = Vector2f(-1000.f, 2976.f);

        // validate data
        for (auto it = files.cbegin(); it != files.cend(); ++it)
        {
            if ((*it)->GetSize() != volume_size_)
            {
                throw RuntimeError() << "Volume size mismatch";
            }
            if ((*it)->GetElementType() != files.front()->GetElementType())
            {
                throw RuntimeError() << "Volume element type mismatch";
            }
            if (stream_from_cpu)
            {
                impl_->blobs_.push((*it)->GetData());
            }
            else
            {
                std::shared_ptr<CudaMemoryBlob> gpu_blob =
                    std::make_shared<CudaMemoryBlob>(std::make_unique<CudaMemory>((*it)->GetData()->GetSize()));
                {
                    std::unique_ptr<IBlob::AccessGuard> access_cpu = (*it)->GetData()->Access();
                    std::unique_ptr<IBlob::AccessGuard> access_gpu = gpu_blob->Access();
                    CudaCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(access_gpu->GetData()), access_cpu->GetData(),
                                           gpu_blob->GetSize()));
                }
                impl_->blobs_.push(gpu_blob);
            }
        }
    }
    else
    {
        const uint32_t volumes = 10;
        Log(LogLevel::Info) << "Generating " << volumes << " volumes ...";

        volume_size_              = Vector3ui(512, 512, 128);
        volume_element_spacing_   = Vector3f(1.0f, 1.0f, 4.f);
        volume_type_              = DataElementType::UINT8;
        volume_bytes_per_element_ = sizeof(uint8_t);
        volume_element_range_     = Vector2f(0.f, 255.f);

        for (uint32_t v = 0; v < volumes; ++v)
        {
            std::unique_ptr<std::vector<uint8_t>> data =
                std::make_unique<std::vector<uint8_t>>(volume_size_(0) * volume_size_(1) * volume_size_(2));

            uint8_t *dst         = data->data();
            const float PI       = std::acos(-1.f);
            const float sequence = 0.5f + static_cast<float>(v + 1) / volumes;
            for (uint32_t z = 0; z < volume_size_(2); ++z)
            {
                const float fz = std::sin(((static_cast<float>(z) / static_cast<float>(volume_size_(2))) - 0.5f) *
                                          2.3f * PI * sequence);
                for (uint32_t y = 0; y < volume_size_(1); ++y)
                {
                    const float fy = std::sin(((static_cast<float>(y) / static_cast<float>(volume_size_(1))) - 0.5f) *
                                              4.7f * PI * sequence);
                    for (uint32_t x = 0; x < volume_size_(0); ++x)
                    {
                        const float fx =
                            std::sin(((static_cast<float>(x) / static_cast<float>(volume_size_(0))) - 0.5f) * 6.2f *
                                     PI * sequence);
                        *dst = static_cast<uint8_t>(std::abs(fx * fy * fz) * 255.f + 0.5f);
                        ++dst;
                    }
                }
            }
            if (stream_from_cpu)
            {
                impl_->blobs_.push(std::make_shared<StdContainerBlob<std::vector<uint8_t>>>(std::move(data)));
            }
            else
            {
                std::shared_ptr<CudaMemoryBlob> gpu_blob =
                    std::make_shared<CudaMemoryBlob>(std::make_unique<CudaMemory>(data->size()));
                {
                    std::unique_ptr<IBlob::AccessGuard> access_gpu = gpu_blob->Access();
                    CudaCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(access_gpu->GetData()), data->data(),
                                           gpu_blob->GetSize()));
                }
                impl_->blobs_.push(gpu_blob);
            }
        }
    }
    Log(LogLevel::Info)
        << "... in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() /
               1000.f
        << " s, size " << volume_size_ << ", type " << volume_type_;
}

float DataSourceCT::FrameRate()
{
    // 10 frames in 4 seconds
    return 10.f / 4.f;
}

std::shared_ptr<IBlob> DataSourceCT::NextBlob(bool benchmark_mode)
{
    std::shared_ptr<IBlob> blob;
    if (!impl_->blobs_.empty())
    {
        blob = impl_->blobs_.front();
        impl_->blobs_.pop();
    }
    return blob;
}

void DataSourceCT::ReturnBlob(const std::shared_ptr<IBlob> &blob)
{
    impl_->blobs_.push(blob);
}

void DataSourceCT::Light(LightInterface &interface)
{
    LightInterface::AccessGuard access(interface);

    LightInterface::DataIn::Light &light = access->lights[0];

    light.position = Vector3f(1.1f, 0.89f, 0.69f);
    light.direction.Set(Vector3f(-0.6984941632629672f, -0.5656294206902576f, -0.4383711467890774f));
    light.size.Set(1.f);
    light.intensity.Set(29.46f);
    light.color.Set(Vector3f(1.f, 1.f, 1.f));
    light.enable = true;
}

void DataSourceCT::BackgroundLight(BackgroundLightInterface &interface)
{
    BackgroundLightInterface::AccessGuard access(interface);

    access->intensity.Set(0.055f);
    access->top_color.Set(Vector3f(0.39f, 0.58f, 0.f));
    access->horizon_color.Set(Vector3f(1.f, 1.f, 0.39f));
    access->bottom_color.Set(Vector3f(1.f, 0.78f, 0.39f));
    access->enable = true;
}

void DataSourceCT::Camera(CameraInterface &interface)
{
    CameraInterface::AccessGuard access(interface);

    auto camera  = access->cameras.begin();
    camera->name = "Cinematic";
    camera->eye.Set(Vector3f(0.438f, 0.193f, 0.940f));
    camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
    camera->up.Set(Vector3f(0.f, 1.f, 0.f));
    camera->field_of_view.Set(30.f);
    camera->pixel_aspect_ratio.Set(1.f);
}

void DataSourceCT::TransferFunction(TransferFunctionInterface &interface)
{
    TransferFunctionInterface::AccessGuard access(interface);

    access->blending_profile = TransferFunctionBlendingProfile::MAXIMUM_OPACITY;
    access->density_scale.Set(1000);
    access->global_opacity.Set(1);
    access->gradient_scale.Set(10);
    access->shading_profile = TransferFunctionShadingProfile::HYBRID;

    access->components.clear();
    {
        access->components.emplace_back();
        TransferFunctionInterface::DataIn::Component &component = access->components.back();
        component.range.Set(Vector2f(0.252f, 0.272f));
        component.active_regions  = std::vector<uint32_t>({0});
        component.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
        component.opacity_transition.Set(0.2f);
        component.opacity.Set(0.51f);
        component.roughness.Set(57.7f);
        component.emissive_strength.Set(0);
        component.diffuse_start.Set(Vector3f(0.796f, 0.502f, 0.388f));
        component.diffuse_end.Set(Vector3f(0.796f, 0.502f, 0.388f));
        component.specular_start.Set(Vector3f(1.f, 1.f, 1.f));
        component.specular_end.Set(Vector3f(1.f, 1.f, 1.f));
        component.emissive_start.Set(Vector3f(0.f, 0.f, 0.f));
        component.emissive_end.Set(Vector3f(0.f, 0.f, 0.f));
    }

    {
        access->components.emplace_back();
        TransferFunctionInterface::DataIn::Component &component = access->components.back();
        component.range.Set(Vector2f(0.287f, 0.37f));
        component.active_regions  = std::vector<uint32_t>({0});
        component.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
        component.opacity_transition.Set(0.f);
        component.opacity.Set(1.f);
        component.roughness.Set(0.f);
        component.emissive_strength.Set(0);
        component.diffuse_start.Set(Vector3f(0.902f, 0.855f, 0.718f));
        component.diffuse_end.Set(Vector3f(0.902f, 0.855f, 0.718f));
        component.specular_start.Set(Vector3f(1.f, 1.f, 1.f));
        component.specular_end.Set(Vector3f(1.f, 1.f, 1.f));
        component.emissive_start.Set(Vector3f(0.f, 0.f, 0.f));
        component.emissive_end.Set(Vector3f(0.f, 0.f, 0.f));
    }
}

} // namespace clara::viz