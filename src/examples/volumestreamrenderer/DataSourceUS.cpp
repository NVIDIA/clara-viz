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

#include "DataSourceUS.h"

#include <claraviz/util/CudaMemoryBlob.h>
#include <claraviz/util/MHDLoader.h>

#include <cuda/BackProjectionConstruction.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <list>
#include <queue>
#include <thread>

namespace clara::viz
{

class DataSourceUS::Impl
{
public:
    /// source images
    std::list<std::vector<uint8_t>> source_;

    /// information structure for source images
    struct Header
    {
        float phi;
        std::string time;
        uint32_t depth, width, scan_depth, origin_offset_x, origin_offset_y;
    };
    /// headers for each source image
    std::list<Header> header_;

    /// source image dimensions
    uint32_t image_depth_;
    uint32_t image_width_;

    std::shared_ptr<CudaMemoryBlob> phi_sorted_;
    std::shared_ptr<CudaMemoryBlob> input_;

    std::unique_ptr<CudaFunctionLauncher> back_project_;

    std::queue<std::shared_ptr<IBlob>> blobs_;

    uint32_t frame_index_ = 0;
    uint32_t frame_count_ = 1;
};

DataSourceUS::DataSourceUS(const std::string input_dir)
    : impl_(new Impl)
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    constexpr uint32_t down_scale                     = 2;

    if (!input_dir.empty())
    {
        // read the header file
        std::ifstream header(input_dir + "PhiData.txt");

        std::string line;
        while (std::getline(header, line))
        {
            Impl::Header &header = impl_->header_.emplace_back();

            std::stringstream str(line);
            str >> header.phi >> header.time >> header.depth >> header.width >> header.scan_depth >>
                header.origin_offset_x >> header.origin_offset_y;
        }

        // sort for phi
        impl_->header_.sort([](const Impl::Header &a, const Impl::Header &b) { return a.phi < b.phi; });

        // assume all input images have the same size
        impl_->image_depth_ = impl_->header_.front().depth / down_scale;
        impl_->image_width_ = impl_->header_.front().width / down_scale;

        // read the files
        Log(LogLevel::Info) << "Loading " << impl_->header_.size() << " slices ...";
        std::list<std::future<void>> tasks;
        for (auto it = impl_->header_.cbegin(); it != impl_->header_.cend(); ++it)
        {
            std::vector<uint8_t> &data = impl_->source_.emplace_back(impl_->image_depth_ * impl_->image_width_);
            tasks.push_back(std::async(std::launch::async, [this, &data, input_dir, it]() {
                // read whole file into temp array
                std::ifstream file;
                file.exceptions(std::ios::failbit | std::ios::badbit);
                file.open(input_dir + it->time + ".bin", std::ios::binary | std::ios::ate);
                std::vector<uint8_t> tmp(file.tellg());
                file.seekg(0, std::ios::beg);
                file.read(reinterpret_cast<char *>(tmp.data()), tmp.size());

                // copy first component of gray-scale RGB data to destination
                for (uint32_t z = 0; z < impl_->image_depth_; ++z)
                {
                    for (uint32_t x = 0; x < impl_->image_width_; ++x)
                    {
                        data[x + z * impl_->image_width_] = tmp[(x * down_scale + z * down_scale * it->width) * 3];
                    }
                }
            }));
        }

        std::for_each(tasks.begin(), tasks.end(), [](std::future<void> &future) { future.get(); });
    }
    else
    {
        const uint32_t slices = 500;
        Log(LogLevel::Info) << "Generating " << slices << " slices ...";

        constexpr uint32_t depth           = 512;
        constexpr uint32_t width           = 1024;
        constexpr uint32_t scan_depth      = 100;
        constexpr uint32_t origin_offset_x = 0;
        constexpr uint32_t origin_offset_y = 0;
        const float PI                     = std::acos(-1.f);

        // assume all input images have the same size
        impl_->image_depth_ = depth / down_scale;
        impl_->image_width_ = width / down_scale;

        for (uint32_t s = 0; s < slices; ++s)
        {
            Impl::Header &header = impl_->header_.emplace_back();

            header.phi             = (static_cast<float>(s) / static_cast<float>(slices - 1) - 0.5f) * PI * 0.5f;
            header.time            = s;
            header.depth           = depth;
            header.width           = width;
            header.scan_depth      = scan_depth;
            header.origin_offset_x = 0;
            header.origin_offset_y = 0;

            std::vector<uint8_t> &data = impl_->source_.emplace_back(impl_->image_depth_ * impl_->image_width_);

            uint8_t *dst         = data.data();
            const float sequence = 0.5f + static_cast<float>(s + 1) / slices;
            for (uint32_t z = 0; z < impl_->image_depth_; ++z)
            {
                const float fz = std::sin(((static_cast<float>(z) / static_cast<float>(impl_->image_depth_)) - 0.5f) *
                                          2.3f * PI * sequence);
                for (uint32_t x = 0; x < impl_->image_width_; ++x)
                {
                    const float fx =
                        std::sin(((static_cast<float>(x) / static_cast<float>(impl_->image_width_)) - 0.5f) * 6.2f *
                                 PI * sequence);
                    *dst = static_cast<uint8_t>(std::abs(fx * fz) * 255.f + 0.5f);
                    ++dst;
                }
            }
        }
    }

    const uint32_t depth_mm_to_pixel_scale = impl_->image_depth_ / impl_->header_.front().scan_depth;
    volume_size_(0) =
        impl_->image_depth_ * (std::sin(impl_->header_.back().phi) - std::sin(impl_->header_.front().phi)) +
        100; // extra room
    volume_size_(1) = impl_->image_width_ + impl_->header_.front().origin_offset_y * depth_mm_to_pixel_scale;
    volume_size_(2) = impl_->image_depth_ + impl_->header_.front().origin_offset_x * depth_mm_to_pixel_scale;

    volume_element_spacing_ =
        Vector3f(1.f / depth_mm_to_pixel_scale, 1.f / depth_mm_to_pixel_scale, 1.f / depth_mm_to_pixel_scale);
    volume_type_              = DataElementType::UINT8;
    volume_bytes_per_element_ = sizeof(uint8_t);
    volume_element_range_     = Vector2f(0.f, 255.f);

    Log(LogLevel::Info)
        << "... in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() /
               1000.f
        << " s, final volume size will be " << volume_size_ << " (downscaled by factor " << down_scale
        << " in each direction)";

    // copy sorted PHI values to GPU
    impl_->phi_sorted_ =
        std::make_shared<CudaMemoryBlob>(std::make_unique<CudaMemory>(impl_->header_.size() * sizeof(float)));
    {
        std::vector<float> phi_sorted(impl_->header_.size());
        uint32_t index = 0;
        for (auto it = impl_->header_.cbegin(); it != impl_->header_.cend(); ++it, ++index)
        {
            phi_sorted[index] = it->phi;
        }
        std::unique_ptr<IBlob::AccessGuard> access_phi_sorted = impl_->phi_sorted_->Access();
        CudaCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(access_phi_sorted->GetData()), phi_sorted.data(),
                               impl_->phi_sorted_->GetSize()));
    }

    impl_->input_ = std::make_shared<CudaMemoryBlob>(std::make_unique<CudaMemory>(
        impl_->image_depth_ * impl_->image_width_ * impl_->source_.size() * sizeof(uint8_t)));

    // copy source images to input
    {
        std::unique_ptr<IBlob::AccessGuard> access_input = impl_->input_->Access(CU_STREAM_PER_THREAD);
        CUdeviceptr dst                                  = reinterpret_cast<CUdeviceptr>(access_input->GetData());
        for (auto it = impl_->source_.cbegin(); it != impl_->source_.cend(); ++it)
        {
            CudaCheck(cuMemcpyHtoDAsync(dst, it->data(), it->size(), CU_STREAM_PER_THREAD));
            dst += it->size();
        }
    }

    impl_->back_project_ = GetBackProjectionLauncher(impl_->phi_sorted_->GetSize());
    impl_->back_project_->SetCalcLaunchGrid([](const Vector3ui grid) -> dim3 {
        dim3 launch_grid;
        launch_grid.x = grid(0);
        launch_grid.y = grid(1);
        launch_grid.z = grid(2);
        return launch_grid;
    });

    // start with a few slices
    impl_->frame_index_ = 0;
    impl_->frame_count_ = 5 * impl_->header_.size() / 100;
}

float DataSourceUS::FrameRate()
{
    return 7.f;
}

std::shared_ptr<IBlob> DataSourceUS::NextBlob(bool benchmark_mode)
{
    if (benchmark_mode)
    {
        // all slices if in benchmark mode
        impl_->frame_index_ = 0;
        impl_->frame_count_ = impl_->header_.size();
    }
    else
    {
        // animate if not in benchmark mode
        impl_->frame_count_ += impl_->header_.size() / 100;
        if (impl_->frame_index_ + impl_->frame_count_ > impl_->header_.size())
        {
            impl_->frame_index_ = 0;
            impl_->frame_count_ = 5 * impl_->header_.size() / 100;
        }
    }

    // get an output memory blob
    std::shared_ptr<IBlob> blob;
    while (!blob)
    {
        if (!impl_->blobs_.empty())
        {
            blob = impl_->blobs_.front();
            impl_->blobs_.pop();
        }
        else
        {
            try
            {
                // allocate a new one
                blob = std::make_shared<CudaMemoryBlob>(std::make_unique<CudaMemory>(
                    volume_size_(2) * volume_size_(1) * volume_size_(0) * sizeof(uint8_t)));
            }
            catch (...)
            {
                Log(LogLevel::Warning) << "Failed to allocate blob, waiting...";
                std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(1000.f / FrameRate()));
            }
        }
    }

    // launch the back projection kernel
    {
        constexpr uint32_t threads_per_block = 128u;
        const uint16_t write_phi_data_times =
            static_cast<uint16_t>((impl_->frame_count_ + (threads_per_block - 1)) / threads_per_block);
        constexpr uint16_t subtract_bottom_lines = 16u;
        const uint32_t temp = static_cast<uint32_t>((volume_size_(1) + (threads_per_block - 1)) / threads_per_block);
        const uint32_t grid_size_y = volume_size_(2) * temp;

        std::unique_ptr<IBlob::AccessGuard> access_output     = blob->Access(CU_STREAM_PER_THREAD);
        std::unique_ptr<IBlob::AccessGuardConst> access_input = impl_->input_->AccessConst(CU_STREAM_PER_THREAD);
        std::unique_ptr<IBlob::AccessGuardConst> access_phi_sorted =
            impl_->phi_sorted_->AccessConst(CU_STREAM_PER_THREAD);

        dim3 block_dim;
        block_dim.x = threads_per_block;
        block_dim.y = 1;
        block_dim.z = 1;
        impl_->back_project_->SetBlockDim(block_dim);

        impl_->back_project_->Launch(
            Vector2ui(volume_size_(0), grid_size_y), reinterpret_cast<uint8_t *>(access_output->GetData()),
            reinterpret_cast<const uint8_t *>(access_input->GetData()) +
                impl_->frame_index_ * impl_->source_.front().size(),
            reinterpret_cast<const float *>(access_phi_sorted->GetData()) + impl_->frame_index_,
            make_ushort3(volume_size_(2), volume_size_(1), volume_size_(0)),
            make_ushort2(impl_->image_depth_, impl_->image_width_), impl_->frame_count_, write_phi_data_times,
            subtract_bottom_lines);

#if 0
        // write volume out
        std::vector<uint8_t> host(blob->GetSize());
        CudaCheck(cuMemcpyDtoHAsync(host.data(), reinterpret_cast<CUdeviceptr>(access_output->GetData()),
            blob->GetSize(), CU_STREAM_PER_THREAD));
        CudaCheck(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        std::ofstream file("volume.raw", std::ios::out | std::ofstream::binary);
        std::copy(host.begin(), host.end(), std::ostreambuf_iterator<char>(file));
        exit(0);
#endif
    }

    return blob;
}

void DataSourceUS::ReturnBlob(const std::shared_ptr<IBlob> &blob)
{
    impl_->blobs_.push(blob);
}

void DataSourceUS::Light(LightInterface &interface)
{
    LightInterface::AccessGuard access(interface);

    LightInterface::DataIn::Light &light = access->lights[0];

    light.position = Vector3f(1.1070386443292992f, 0.8964622183077132f, 0.6947714463173809f);
    light.direction.Set(Vector3f(-0.6984941632629672f, -0.5656294206902576f, -0.4383711467890774f));
    light.size.Set(1.f);
    light.intensity.Set(29.46f);
    light.color.Set(Vector3f(1.f, 1.f, 1.f));
    light.enable = true;
}

void DataSourceUS::BackgroundLight(BackgroundLightInterface &interface)
{
    BackgroundLightInterface::AccessGuard access(interface);

    access->intensity.Set(0.5f);
    access->top_color.Set(Vector3f(0.39f, 0.58f, 0.f));
    access->horizon_color.Set(Vector3f(1.f, 1.f, 0.39f));
    access->bottom_color.Set(Vector3f(1.f, 0.78f, 0.39f));
    access->enable = true;
}

void DataSourceUS::Camera(CameraInterface &interface)
{
    CameraInterface::AccessGuard access(interface);

    auto camera  = access->cameras.begin();
    camera->name = "Cinematic";
    camera->eye.Set(Vector3f(-0.11101658252063473f, -0.28485389832476954f, 0.3641088056102614f));
    camera->look_at.Set(Vector3f(0.008982176673583302f, 0.009773539189614539f, 0.006764344392308056f));
    camera->up.Set(Vector3f(-0.5759427670052438f, 0.7150792748929987f, 0.39617112433011076f));
    camera->field_of_view.Set(30.f);
    camera->pixel_aspect_ratio.Set(1.f);
}

void DataSourceUS::TransferFunction(TransferFunctionInterface &interface)
{
    TransferFunctionInterface::AccessGuard access(interface);

    access->blending_profile = TransferFunctionBlendingProfile::MAXIMUM_OPACITY;
    access->density_scale.Set(100);
    access->global_opacity.Set(1);
    access->gradient_scale.Set(10);
    access->shading_profile = TransferFunctionShadingProfile::HYBRID;

    access->components.clear();
    {
        TransferFunctionInterface::DataIn::Component &component = access->components.emplace_back();
        component.range.Set(Vector2f(0.134f, 0.169f));
        component.active_regions  = std::vector<uint32_t>({0});
        component.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
        component.opacity_transition.Set(0.2f);
        component.opacity.Set(0.26f);
        component.roughness.Set(8.f);
        component.emissive_strength.Set(0);
        component.diffuse_start.Set(Vector3f(0.4823529411764706f, 0.7215686274509804f, 0.5803921568627451f));
        component.diffuse_end.Set(Vector3f(0.4823529411764706f, 0.7215686274509804f, 0.5803921568627451f));
        component.specular_start.Set(Vector3f(0.43529411764705883f, 0.6313725490196078f, 0.3568627450980392f));
        component.specular_end.Set(Vector3f(0.43529411764705883f, 0.6313725490196078f, 0.3568627450980392f));
        component.emissive_start.Set(Vector3f(0.f, 0.f, 0.f));
        component.emissive_end.Set(Vector3f(0.f, 0.f, 0.f));
    }

    {
        TransferFunctionInterface::DataIn::Component &component = access->components.emplace_back();
        component.range.Set(Vector2f(0.38f, 0.441f));
        component.active_regions  = std::vector<uint32_t>({0});
        component.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
        component.opacity_transition.Set(0.f);
        component.opacity.Set(1.f);
        component.roughness.Set(0.f);
        component.emissive_strength.Set(0);
        component.diffuse_start.Set(Vector3f(1.f, 0.29411764705882354f, 0.f));
        component.diffuse_end.Set(Vector3f(1.f, 0.29411764705882354f, 0.f));
        component.specular_start.Set(Vector3f(1.f, 0.3254901960784314f, 0.f));
        component.specular_end.Set(Vector3f(1.f, 0.3254901960784314f, 0.f));
        component.emissive_start.Set(Vector3f(0.f, 0.f, 0.f));
        component.emissive_end.Set(Vector3f(0.f, 0.f, 0.f));
    }
}

} // namespace clara::viz
