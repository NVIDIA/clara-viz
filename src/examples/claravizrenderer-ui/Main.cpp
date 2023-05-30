/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "UI.h"

#include <ClaraVizRenderer.h>

#include <claraviz/util/Log.h>
#include <claraviz/util/MHDLoader.h>
#include <claraviz/util/StdContainerBlob.h>

#include <claraviz/core/Image.h>

#include <claraviz/interface/CameraInterface.h>
#include <claraviz/interface/DataInterface.h>
#include <claraviz/interface/DataViewInterface.h>
#include <claraviz/interface/ImageInterface.h>
#include <claraviz/interface/LightInterface.h>
#include <claraviz/interface/PostProcessDenoiseInterface.h>
#include <claraviz/interface/PostProcessTonemapInterface.h>
#include <claraviz/interface/RenderSettingsInterface.h>
#include <claraviz/interface/TransferFunctionInterface.h>
#include <claraviz/interface/ViewInterface.h>

#include <claraviz/hardware/cuda/CudaService.h>

#include <chrono>
#include <cmath>

#include <getopt.h>

using namespace clara::viz;

class RendererState
{
public:
    RendererState();
    ~RendererState();

    void setup(MHDLoader &mhd_loader);
    std::shared_ptr<const ImageEncodedDataMessage> render(const std::string &view_name, uint32_t width,
                                                          uint32_t height);

private:
    // defines on which cuda device(s) to render, we use the first device
    const std::vector<uint32_t> cuda_device_ordinals = {0};

    std::shared_ptr<ImageInterfaceOutput> image_interface_output;
    std::shared_ptr<Image> image;
    ImageInterface image_interface;

    std::unique_ptr<Renderer> renderer;

    std::shared_ptr<DataInterface> data_interface;
    DataConfigInterface data_config_interface;
    std::shared_ptr<DataHistogramInterface> data_histogram_interface;

    CameraInterface camera_interface;
    DataCropInterface data_crop_interface;
    DataViewInterface data_view_interface;
    BackgroundLightInterface background_light_interface;
    LightInterface light_interface;
    PostProcessDenoiseInterface post_process_denoise_interface;
    PostProcessTonemapInterface post_process_tonemap_interface;
    RenderSettingsInterface render_settings_interface;
    TransferFunctionInterface transfer_function_interface;
    ViewInterface view_interface;
};

RendererState::RendererState()
{
    // create the image encoding service, it receives the image data from the renderer and outputs the final image. There
    // are several image format options, e.g. JPEG-encoding
    image_interface_output.reset(new ImageInterfaceOutput);
    image = Image::Create(image_interface_output, cuda_device_ordinals[0]);
    // the image service will receive message from the image interface
    image_interface.RegisterReceiver(image);

    // create the renderer
    renderer = std::make_unique<Renderer>(std::shared_ptr<MessageReceiver>(),
                                          std::static_pointer_cast<MessageReceiver>(image), cuda_device_ordinals);

    // create the interfaces, these will generate messages which are then handled by the renderer
    const std::shared_ptr<MessageReceiver> &receiver = renderer->GetReceiver();

    camera_interface.RegisterReceiver(receiver);

    data_interface = std::make_shared<DataInterface>();
    data_interface->RegisterReceiver(receiver);
    data_config_interface.RegisterReceiver(receiver);
    // the data interface needs to get updates from the data config interface to do proper parameter validation
    data_config_interface.RegisterReceiver(data_interface);

    data_histogram_interface = std::make_shared<DataHistogramInterface>();
    data_histogram_interface->RegisterReceiver(receiver);

    data_crop_interface.RegisterReceiver(receiver);
    data_view_interface.RegisterReceiver(receiver);
    background_light_interface.RegisterReceiver(receiver);
    light_interface.RegisterReceiver(receiver);
    post_process_denoise_interface.RegisterReceiver(receiver);
    post_process_tonemap_interface.RegisterReceiver(receiver);
    render_settings_interface.RegisterReceiver(receiver);
    transfer_function_interface.RegisterReceiver(receiver);
    view_interface.RegisterReceiver(receiver);

    // renderer will also receive messages from the image service
    image->RegisterReceiver(receiver);

    // start the renderer thread
    renderer->Run();
    // start the image encoder thread
    image->Run();
}

RendererState::~RendererState()
{
    // shutdown the image servers first, they will return the resource which had been about to be encoded
    image->Shutdown();

    // then shutdown the renderer
    renderer->Shutdown();

    // unregister the receivers
    const std::shared_ptr<MessageReceiver> &receiver = renderer->GetReceiver();
    background_light_interface.UnregisterReceiver(receiver);
    camera_interface.UnregisterReceiver(receiver);
    data_interface->UnregisterReceiver(receiver);
    data_config_interface.UnregisterReceiver(receiver);
    data_config_interface.UnregisterReceiver(data_interface);
    data_crop_interface.UnregisterReceiver(receiver);
    data_view_interface.UnregisterReceiver(receiver);
    data_histogram_interface->UnregisterReceiver(receiver);
    light_interface.UnregisterReceiver(receiver);
    post_process_denoise_interface.UnregisterReceiver(receiver);
    post_process_tonemap_interface.UnregisterReceiver(receiver);
    render_settings_interface.UnregisterReceiver(receiver);
    transfer_function_interface.UnregisterReceiver(receiver);
    view_interface.UnregisterReceiver(receiver);

    image_interface.UnregisterReceiver(image);

    image->UnregisterReceiver(receiver);
}

void RendererState::setup(MHDLoader &mhd_loader)
{
    // configure the data
    Log(LogLevel::Info) << "Configuring and setting volume data";
    const std::string array_id = "density";
    {
        DataConfigInterface::AccessGuard access(data_config_interface);

        access->arrays.emplace_back();
        const auto array = access->arrays.begin();
        array->id        = array_id;
        array->dimension_order.Set("DXYZ");
        switch (mhd_loader.GetElementType())
        {
        case MHDLoader::ElementType::UINT8:
            array->element_type.Set(DataElementType::UINT8);
            break;
        case MHDLoader::ElementType::INT8:
            array->element_type.Set(DataElementType::INT8);
            break;
        case MHDLoader::ElementType::UINT16:
            array->element_type.Set(DataElementType::UINT16);
            break;
        case MHDLoader::ElementType::INT16:
            array->element_type.Set(DataElementType::INT16);
            break;
        case MHDLoader::ElementType::UINT32:
            array->element_type.Set(DataElementType::UINT32);
            break;
        case MHDLoader::ElementType::INT32:
            array->element_type.Set(DataElementType::INT32);
            break;
        case MHDLoader::ElementType::FLOAT:
            array->element_type.Set(DataElementType::FLOAT);
            break;
        default:
            throw std::runtime_error("Unhandled element type");
        }

        array->levels.emplace_back();
        const auto level = array->levels.begin();
        level->size.Set({1, mhd_loader.GetSize()(0), mhd_loader.GetSize()(1), mhd_loader.GetSize()(2)});
        level->element_size.Set({1.0f, mhd_loader.GetElementSpacing()(0), mhd_loader.GetElementSpacing()(1),
                                 mhd_loader.GetElementSpacing()(2)});
    }
    // set the data
    {
        DataInterface::AccessGuard access(*data_interface.get());
        access->array_id.Set(array_id);
        access->level.Set(0);
        access->offset.Set({0, 0, 0, 0});
        access->size.Set({1, mhd_loader.GetSize()(0), mhd_loader.GetSize()(1), mhd_loader.GetSize()(2)});
        access->blob = mhd_loader.GetData();
    }

    // calculate the volume physical size, element spacing is in mm, physical size is in meters
    Vector3f size;
    for (int index = 0; index < 3; ++index)
        size(index) = mhd_loader.GetSize()(index) * mhd_loader.GetElementSpacing()(index) / 1000.f;

    // set the cameras
    const std::string front_camera_name("Front");
    const std::string right_camera_name("Right");
    {
        CameraInterface::AccessGuard access(camera_interface);

        const float PI            = std::acos(-1);
        const float field_of_view = 30.0f;
        auto toRadians            = [PI](float degree) { return degree * PI / 180.0f; };
        // calculate the camera distance for a given volume physical size and a field of view
        const float front_distance = (std::max(size(0), size(1)) * 0.5f) / std::tan(toRadians(field_of_view * 0.5f));
        const float right_distance = (std::max(size(1), size(2)) * 0.5f) / std::tan(toRadians(field_of_view * 0.5f));

        CameraInterface::DataIn::Camera *front_camera = access->GetOrAddCamera(front_camera_name);
        front_camera->eye.Set(Vector3f(0.f, 0.f, -(front_distance + size(2) * 0.5f)));
        front_camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
        front_camera->up.Set(Vector3f(0.f, 1.f, 0.f));
        front_camera->field_of_view.Set(field_of_view);
        front_camera->pixel_aspect_ratio.Set(1.f);
        front_camera->depth_clip.Set(Vector2f(front_distance, front_distance + size(2)));
        front_camera->depth_range.Set(Vector2f(0.f, 1.f));

        CameraInterface::DataIn::Camera *right_camera = access->GetOrAddCamera(right_camera_name);
        right_camera->eye.Set(Vector3f(right_distance + size(0) * 0.5f, 0.f, 0.f));
        right_camera->look_at.Set(Vector3f(0.f, 0.f, 0.f));
        right_camera->up.Set(Vector3f(0.f, 1.f, 0.f));
        right_camera->field_of_view.Set(field_of_view);
        right_camera->pixel_aspect_ratio.Set(1.f);
        right_camera->depth_clip.Set(Vector2f(right_distance, right_distance + size(0)));
        right_camera->depth_range.Set(Vector2f(0.f, 1.f));
    }

    // set the views
    {
        ViewInterface::AccessGuard access(view_interface);

        ViewInterface::DataIn::View *front_view = access->GetOrAddView("Front");
        front_view->camera_name                 = front_camera_name;
        front_view->mode                        = ViewMode::CINEMATIC;

        ViewInterface::DataIn::View *right_view = access->GetOrAddView("Right");
        right_view->camera_name                 = right_camera_name;
        right_view->mode                        = ViewMode::CINEMATIC;
    }

    // place a light
    {
        LightInterface::AccessGuard access(light_interface);

        const float light_distance_squared = (size(0) * size(0)) + (size(1) * size(1)) + (size(2) * size(2));

        LightInterface::DataIn::Light &light = access->lights[0];

        light.position = Vector3f(-2.f * size(0), 2.f * size(1), -2.f * size(2));
        light.direction.Set(Vector3f(std::sqrt(1.f / 3.f), std::sqrt(1.f / 3.f), std::sqrt(1.f / 3.f)));
        light.size.Set(std::max(size(0), std::max(size(1), size(2))) / 2.f);
        light.intensity.Set(light_distance_squared * 10.f);
        light.color.Set(Vector3f(1.f, 1.f, 1.f));
        light.enable = true;
    }

    // define a transfer function
    {
        TransferFunctionInterface::AccessGuard access(transfer_function_interface);

        access->blending_profile = TransferFunctionBlendingProfile::MAXIMUM_OPACITY;
        access->density_scale.Set(100.f);
        access->global_opacity.Set(1.f);
        access->gradient_scale.Set(10.f);
        access->shading_profile = TransferFunctionShadingProfile::HYBRID;

        // get a histogram of the density data, find the maximum and define a transform function
        // around that maximum
        Log(LogLevel::Info) << "Getting data histogram";
        std::vector<float> histogram;
        data_histogram_interface->GetHistogram(array_id, histogram);
        std::vector<float>::iterator start = histogram.begin();
        // skip the first few elements of the histogram, this is usual the air around a scan
        std::advance(start, 5);
        std::vector<float>::iterator max_element = std::max_element(start, histogram.end());
        const float max_pos_normalized           = static_cast<float>(std::distance(histogram.begin(), max_element)) /
                                         static_cast<float>(histogram.size() - 1);
        Log(LogLevel::Info) << "Data histogram max is at " << max_pos_normalized;

        access->components.clear();
        {
            access->components.emplace_back();
            TransferFunctionInterface::DataIn::Component &component = access->components.back();
            component.range.Set(
                Vector2f(std::max(0.f, max_pos_normalized - 0.025f), std::min(1.f, max_pos_normalized + 0.025f)));
            component.active_regions  = std::vector<uint32_t>({});
            component.opacity_profile = TransferFunctionOpacityProfile::SQUARE;
            component.opacity_transition.Set(0.2f);
            component.opacity.Set(1.f);
            component.roughness.Set(0.5f);
            component.emissive_strength.Set(0.f);
            component.diffuse_start.Set(Vector3f(.7f, 1.f, .7f));
            component.diffuse_end.Set(Vector3f(.7f, 1.f, .7f));
            component.specular_start.Set(Vector3f(1.f, 1.f, 1.f));
            component.specular_end.Set(Vector3f(1.f, 1.f, 1.f));
            component.emissive_start.Set(Vector3f(0.f, 0.f, 0.f));
            component.emissive_end.Set(Vector3f(0.f, 0.f, 0.f));
        }
    }

    // configure the render settings
    {
        RenderSettingsInterface::AccessGuard access(render_settings_interface);

        // set the interation count, the higher the better the image quality but this also increases render time
        access->max_iterations.Set(200);

        access->enable_separate_depth = true;
    }
}

std::shared_ptr<const ImageEncodedDataMessage> RendererState::render(const std::string &view_name, uint32_t width,
                                                                     uint32_t height)
{
    // trigger the rendering of the image
    {
        ImageInterface::AccessGuard access(image_interface);

        access->view_name = view_name;
        access->width.Set(width);
        access->height.Set(height);
        access->color_type = ColorImageType::RAW_RGBA_U8;
        access->depth_type = DepthImageType::RAW_DEPTH_F32;
    }

    // then wait for the message with the encoded data
    return image_interface_output->WaitForEncodedData();
}

bool parseOptions(int argc, char **argv, std::string &mhd_file_name, bool &multi_view)
{
    struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                    {"file", required_argument, 0, 'f'},
                                    {"multiview", no_argument, 0, 'm'},
                                    {"loglevel", required_argument, 0, 'l'},
                                    {0, 0, 0, 0}};

    // parse options
    while (true)
    {
        int option_index = 0;

        const int c = getopt_long(argc, argv, "hf:ml:", long_options, &option_index);

        if (c == -1)
        {
            break;
        }

        const std::string argument(optarg ? optarg : "");
        switch (c)
        {
        case 'h':
            std::cout << "ClaraVizRenderServer example. Loading of MHD/MHA files "
                         "(https://itk.org/Wiki/ITK/MetaIO/Documentation) is "
                      << "supported."
                      << "Usage: " << argv[0] << " [options]" << std::endl
                      << "Options:" << std::endl
                      << "  -h, --help                            Display this information" << std::endl
                      << "  -f <FILENAME>, --file <FILENAME>      Set the name of the MHD/MHA file for load"
                      << std::endl
                      << "  -m, --multiview                       Use multiple viewports" << std::endl
                      << "  -l <LOGLEVEL>, --loglevel <LOGLEVEL>  Set the loglevel to <LOGLEVEL>, available levels "
                         "'debug', 'info', 'warning' and 'error'; (default 'info')"
                      << std::endl;
            return false;

        case 'f':
            mhd_file_name = argument;
            break;

        case 'm':
            multi_view = true;
            break;

        case 'l': {
            // convert to lower case
            std::string lower_case_argument;
            std::transform(argument.begin(), argument.end(), std::back_inserter(lower_case_argument),
                           [](unsigned char c) -> unsigned char { return std::tolower(c); });
            if (std::string(lower_case_argument) == "debug")
            {
                Log::g_log_level = LogLevel::Debug;
            }
            else if (std::string(lower_case_argument) == "info")
            {
                Log::g_log_level = LogLevel::Info;
            }
            else if (std::string(lower_case_argument) == "warning")
            {
                Log::g_log_level = LogLevel::Warning;
            }
            else if (std::string(lower_case_argument) == "error")
            {
                Log::g_log_level = LogLevel::Error;
            }
            else
            {
                throw InvalidArgument("loglevel") << "Invalid log level '" << argument << "'";
            }
            break;
        }
        case '?':
            // unknown option, error already printed by getop_long
            break;
        default:
            throw InvalidState() << "Unhandled option " << c;
        }
    }

    if (mhd_file_name.empty())
    {
        throw InvalidState() << "The name of the MHD/MHA file to load is required";
    }

    return true;
}

int main(int argc, char **argv)
{
    try
    {
        std::string mhd_file_name;
        bool multi_view = false;

        if (!parseOptions(argc, argv, mhd_file_name, multi_view))
            return EXIT_SUCCESS;

        const uint32_t width = 512, height = 512;

        // load the MHD file
        Log(LogLevel::Info) << "Loading MHD file " << mhd_file_name;

        MHDLoader mhd_loader;
        mhd_loader.Load(mhd_file_name);

        Log(LogLevel::Info) << "Volume size: " << mhd_loader.GetSize()(0) << ", " << mhd_loader.GetSize()(1) << ", "
                            << mhd_loader.GetSize()(2);
        Log(LogLevel::Info) << "Volume spacing: " << mhd_loader.GetElementSpacing()(0) << ", "
                            << mhd_loader.GetElementSpacing()(1) << ", " << mhd_loader.GetElementSpacing()(2);
        Log(LogLevel::Info) << "Volume element type: " << mhd_loader.GetElementType();

        std::vector<std::string> views{"Front"};
        if (multi_view)
        {
            views.push_back("Right");
        }
        RendererState renderer_state;
        renderer_state.setup(mhd_loader);

        // init CUDA
        CudaCheck(cuInit(0));
        CudaPrimaryContext cuda_context(0);

        std::list<std::shared_ptr<const ImageEncodedDataMessage>> encoded_datas;
        std::list<std::vector<uint8_t>> datas;
        std::list<std::vector<float>> depth_datas;
        for (auto &&view : views)
        {

            Log(LogLevel::Info) << "Rendering " << view << " ...";
            const auto start = std::chrono::steady_clock::now();

            std::shared_ptr<const ImageEncodedDataMessage> encoded_data = renderer_state.render(view, width, height);
            encoded_datas.push_back(encoded_data);

            // copy the data to host memory
            /// @todo add support copy to host to image interface
            {
                std::unique_ptr<IBlob::AccessGuardConst> access = encoded_data->color_memory_->AccessConst();
                datas.push_back(std::vector<uint8_t>(encoded_data->color_memory_->GetSize()));
                CudaCheck(cuMemcpy(reinterpret_cast<CUdeviceptr>(datas.back().data()),
                                   reinterpret_cast<CUdeviceptr>(access->GetData()),
                                   encoded_data->color_memory_->GetSize()));
            }
            {
                std::unique_ptr<IBlob::AccessGuardConst> access = encoded_data->depth_memory_->AccessConst();
                depth_datas.push_back(std::vector<float>(encoded_data->depth_memory_->GetSize() / sizeof(float)));
                CudaCheck(cuMemcpy(reinterpret_cast<CUdeviceptr>(depth_datas.back().data()),
                                   reinterpret_cast<CUdeviceptr>(access->GetData()),
                                   encoded_data->depth_memory_->GetSize()));
            }

            const std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
            Log(LogLevel::Info) << "... done in " << elapsed_seconds.count() << "s";
        }
        show(width, height, datas, depth_datas);
    }
    catch (std::exception &er)
    {
        Log(LogLevel::Error) << "Error: " << er.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
