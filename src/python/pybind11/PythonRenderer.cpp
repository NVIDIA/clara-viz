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

#include "PythonRenderer.h"
#include "ImageCapsule.h"
#include "VideoStream.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

#include <ClaraVizRenderer.h>

#include <claraviz/hardware/cuda/CudaService.h>

#include <claraviz/interface/CameraInterface.h>
#include <claraviz/interface/CameraApertureInterface.h>
#include <claraviz/interface/DataInterface.h>
#include <claraviz/interface/DataViewInterface.h>
#include <claraviz/interface/ImageInterface.h>
#include <claraviz/interface/VideoInterface.h>
#include <claraviz/interface/LightInterface.h>
#include <claraviz/interface/PostProcessDenoiseInterface.h>
#include <claraviz/interface/PostProcessTonemapInterface.h>
#include <claraviz/interface/RenderSettingsInterface.h>
#include <claraviz/interface/TransferFunctionInterface.h>
#include <claraviz/interface/ViewInterface.h>

#include <claraviz/interface/JsonInterface.h>

#include <claraviz/util/StdContainerBlob.h>
#include <claraviz/core/Video.h>

using namespace clara::viz;

namespace py = pybind11;

/// json conversion helper functions
namespace clara::viz
{

/// PythonRenderer implementation
/// had to explicitly hide visibility to suppress
///   warning: 'clara::viz::PythonRenderer::Impl' declared with greater visibility than the type of its field 'clara::viz::PythonRenderer::Impl::python_stdout_buf_' [-Wattributes]
class __attribute__((visibility("hidden"))) PythonRenderer::Impl
{
public:
    /**
     * Construct
     *
     * @param cuda_device_ordinals [in] Cuda devices to render on
     */
    Impl(const std::vector<uint32_t> &cuda_device_ordinals);
    Impl() = delete;

    /**
     * Destruct
     */
    ~Impl();

    /**
     * Set a data arrays
     *
     * @param arrays [in] arrays
     * @param fetch_func [in] A function to be called on demand data fetches
     */
    void SetArrays(const std::vector<Array> &arrays, const FetchFunc &fetch_func);

    /**
     * Get data arrays
     *
     * @returns arrays
     */
    std::vector<Array> GetArrays();

    /**
     * Create a video stream.
     *
     * @param callback [in] video data callback
     *
     * @returns video stream object
     */
    std::unique_ptr<VideoStream> CreateVideoStream(const std::function<void(py::object, bool)> &callback);

    /**
     * Render a image.
     *
     * @param width [in]
     * @param height [in]
     * @param image_type [in]
     */
    std::unique_ptr<ImageCapsule> RenderImage(uint32_t width, uint32_t height,
                                              ColorImageType image_type = ColorImageType::RAW_RGBA_U8);

    /**
     * Deduce settings from configured data (data needs to be configered by SetArrays()).
     * Make the whole dataset visible. Set a light in correct distance. Set a transfer function
     * using the histogram of the data.
     *
     * @param view_mode [in] view mode
     */
    void DeduceSettings(ViewMode view_mode);

    /**
     * Set settings
     *
     * @param new_settings [in] json with new settings
     */
    void SetSettings(const nlohmann::json &new_settings);

    /**
     * Merge settings
     *
     * @param new_settings [in] json with settings to merge in
     */
    void MergeSettings(const nlohmann::json &new_settings);

    /**
     * Get settings
     *
     * @returns json with settings
     */
    nlohmann::json GetSettings();

    /**
     * Reset selected interfaces to default
     *
     * @param selectors [in] vector of interfaces to reset to defaults
     */
    void Reset(const std::vector<InterfaceSelector> &selectors);

private:
    /// Python stdout buffer, used to redirect log to Python
    std::unique_ptr<py::detail::pythonbuf> python_stdout_buf_;
    /// Log ostream using the Python stdout buffer above, assigned to global log
    std::unique_ptr<std::ostream> log_ostream_;
    std::ostream *prev_log_ostream_;

    /// video service
    std::shared_ptr<Video> video_;
    std::shared_ptr<VideoInterface> video_interface_;

    /// image service
    std::shared_ptr<ImageInterfaceOutput> image_interface_output_;
    std::shared_ptr<Image> image_;
    ImageInterface image_interface_;

    std::unique_ptr<clara::viz::Renderer> renderer_;

    CameraInterface camera_interface_;
    CameraApertureInterface camera_aperture_interface_;

    std::shared_ptr<DataInterface> data_interface_;
    DataConfigInterface data_config_interface_;
    std::shared_ptr<DataHistogramInterface> data_histogram_interface_;

    DataCropInterface data_crop_interface_;
    DataTransformInterface data_transform_interface_;
    DataViewInterface data_view_interface_;
    BackgroundLightInterface background_light_interface_;
    LightInterface light_interface_;
    PostProcessDenoiseInterface post_process_denoise_interface_;
    PostProcessTonemapInterface post_process_tonemap_interface_;
    RenderSettingsInterface render_settings_interface_;
    TransferFunctionInterface transfer_function_interface_;
    ViewInterface view_interface_;

    std::unique_ptr<JsonInterface> json_settings_;
};

void PythonRenderer::ImplDeleter::operator()(PythonRenderer::Impl *p) const
{
    delete p;
}

PythonRenderer::Impl::Impl(const std::vector<uint32_t> &cuda_device_ordinals)
{
    // set log level to error to avoid too many messages
    Log::g_log_level = LogLevel::Error;
    // create the Python stream for redirecting
    python_stdout_buf_ = std::make_unique<py::detail::pythonbuf>(py::module_::import("sys").attr("stdout"));
    log_ostream_       = std::make_unique<std::ostream>(python_stdout_buf_.get());
    // redirected log stream to Python
    prev_log_ostream_ = Log::SetLogStream(log_ostream_.get());

    // create the video service
    video_ = Video::Create(cuda_device_ordinals[0]);

    // create the image encoding service, it receives the image data from the renderer and outputs the final image. There
    // are several image format options, e.g. JPEG-encoding
    image_interface_output_.reset(new ImageInterfaceOutput);
    image_ = Image::Create(image_interface_output_, cuda_device_ordinals[0]);
    // the image service will receive message from the image interface
    image_interface_.RegisterReceiver(image_);

    // create the renderer
    renderer_ = std::make_unique<clara::viz::Renderer>(
        std::static_pointer_cast<MessageReceiver>(video_), std::static_pointer_cast<MessageReceiver>(image_),
        cuda_device_ordinals, VolumeRenderBackend::Default, LogLevel::Error, log_ostream_.get());

    // create the interfaces, these will generate messages which are then handled by the renderer
    const std::shared_ptr<MessageReceiver> &receiver = renderer_->GetReceiver();

    camera_interface_.RegisterReceiver(receiver);
    camera_aperture_interface_.RegisterReceiver(receiver);

    data_interface_ = std::make_shared<DataInterface>();
    data_interface_->RegisterReceiver(receiver);
    data_config_interface_.RegisterReceiver(receiver);
    // the data interface needs to get updates from the data config interface to do proper parameter validation
    data_config_interface_.RegisterReceiver(data_interface_);

    data_histogram_interface_ = std::make_shared<DataHistogramInterface>();
    data_histogram_interface_->RegisterReceiver(receiver);

    data_crop_interface_.RegisterReceiver(receiver);
    data_transform_interface_.RegisterReceiver(receiver);
    data_view_interface_.RegisterReceiver(receiver);
    background_light_interface_.RegisterReceiver(receiver);
    light_interface_.RegisterReceiver(receiver);
    post_process_denoise_interface_.RegisterReceiver(receiver);
    post_process_tonemap_interface_.RegisterReceiver(receiver);
    render_settings_interface_.RegisterReceiver(receiver);
    transfer_function_interface_.RegisterReceiver(receiver);
    view_interface_.RegisterReceiver(receiver);

    // create the video interface, it needs the encoder limits, query them from the encoder
    {
        const std::unique_ptr<IVideoEncoder> &encoder = video_->GetEncoder();
        const uint32_t min_width                      = encoder->Query(IVideoEncoder::Capability::MIN_WIDTH);
        const uint32_t min_height                     = encoder->Query(IVideoEncoder::Capability::MIN_HEIGHT);
        const uint32_t max_width                      = encoder->Query(IVideoEncoder::Capability::MAX_WIDTH);
        const uint32_t max_height                     = encoder->Query(IVideoEncoder::Capability::MAX_HEIGHT);
        video_interface_.reset(new VideoInterface(min_width, min_height, max_width, max_height));
    }
    // the video service will receive message from the image and video interfaces
    video_interface_->RegisterReceiver(video_);

    // renderer will also receive messages from the image and video service
    video_->RegisterReceiver(receiver);
    image_->RegisterReceiver(receiver);

    // start the renderer thread
    renderer_->Run();
    // start the image encoder thread
    image_->Run();
    // start the video thread
    video_->Run();

    // init json settings
    json_settings_ = std::make_unique<JsonInterface>(
        &background_light_interface_, &camera_interface_, &camera_aperture_interface_, &data_config_interface_,
        data_histogram_interface_.get(), &data_crop_interface_, &data_transform_interface_, &data_view_interface_,
        &light_interface_, &post_process_denoise_interface_, &post_process_tonemap_interface_,
        &render_settings_interface_, &transfer_function_interface_, &view_interface_);
    json_settings_->InitSettings();
}

PythonRenderer::Impl::~Impl()
{
    // shutdown the video and image servers first, they will return the resource which had been about to be encoded
    image_->Shutdown();
    video_->Shutdown();

    // then shutdown the renderer
    renderer_->Shutdown();

    // undo log stream redirection
    Log::SetLogStream(prev_log_ostream_);

    // unregister the receivers
    const std::shared_ptr<MessageReceiver> &receiver = renderer_->GetReceiver();
    background_light_interface_.UnregisterReceiver(receiver);
    camera_interface_.UnregisterReceiver(receiver);
    camera_aperture_interface_.UnregisterReceiver(receiver);
    data_interface_->UnregisterReceiver(receiver);
    data_config_interface_.UnregisterReceiver(receiver);
    data_config_interface_.UnregisterReceiver(data_interface_);
    data_crop_interface_.UnregisterReceiver(receiver);
    data_transform_interface_.UnregisterReceiver(receiver);
    data_view_interface_.UnregisterReceiver(receiver);
    data_histogram_interface_->UnregisterReceiver(receiver);
    light_interface_.UnregisterReceiver(receiver);
    post_process_denoise_interface_.UnregisterReceiver(receiver);
    post_process_tonemap_interface_.UnregisterReceiver(receiver);
    render_settings_interface_.UnregisterReceiver(receiver);
    transfer_function_interface_.UnregisterReceiver(receiver);
    view_interface_.UnregisterReceiver(receiver);

    video_interface_->UnregisterReceiver(video_);
    image_interface_.UnregisterReceiver(image_);

    video_->UnregisterReceiver(receiver);
    image_->UnregisterReceiver(receiver);
}

void PythonRenderer::Impl::SetArrays(const std::vector<Array> &arrays, const FetchFunc &fetch_func)
{
    // configure the data
    {
        DataConfigInterface::AccessGuard access(data_config_interface_);

        if (fetch_func)
        {
            // for python we need a special fetch callback functions which takes an py::array for the data.
            access->fetch_func =
                [fetch_func](uintptr_t context, const std::string &array_id, uint32_t level_index,
                             const std::vector<uint32_t> &offset, const std::vector<uint32_t> &size,
                             const DataConfigInterface::DataOut::FetchCallbackFunc &fetch_callback_func) -> bool {
                // see https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
                py::gil_scoped_acquire acquire;
                try
                {
                    return fetch_func(context, array_id, level_index, offset, size,
                                      [fetch_callback_func](uintptr_t context, uint32_t level_index,
                                                            const std::vector<uint32_t> &offset,
                                                            const std::vector<uint32_t> &size, py::array data) -> bool {
                                          const py::buffer_info info = data.request();

                                          const bool result = fetch_callback_func(context, level_index, offset, size,
                                                                                  info.ptr, info.size);
                                          if (result)
                                          {
                                              // indicate that the fetch had been fullfilled
                                              fetch_callback_func(context, level_index, offset, size, nullptr, 0);
                                          }
                                          return result;
                                      });
                }
                catch (const std::exception &e)
                {
                    Log(LogLevel::Warning) << "Fetch function failed with " << e.what();
                }
                return false;
            };
        }

        for (auto &&input_array : arrays)
        {
            // add the array, use the dimension_order as array ID expecting that the order is unique
            const std::string array_id(input_array.dimension_order_);
            DataConfigInterface::DataIn::Array *array = access->GetOrAddArray(array_id);
            array->dimension_order.Set(input_array.dimension_order_);
            array->permute_axis.Set(input_array.permute_axes_);
            array->flip_axes.Set(input_array.flip_axes_);

            const uint32_t dimensions = input_array.dimension_order_.size();

            for (auto &&input_level : input_array.levels_)
            {
                array->levels.emplace_back();
                DataConfigInterface::DataIn::Array::Level &level = array->levels.back();

                const py::buffer_info info = input_level.request();

                // shape is reverse to expected size
                std::vector<uint32_t> sizes(info.ndim);
                for (int index = 0; index < info.ndim; ++index)
                {
                    sizes[info.ndim - index - 1] = info.shape[index];
                }

                if (info.ndim == dimensions - 1)
                {
                    // for single component data the buffer_info does not have an axis, add it
                    sizes.insert(sizes.begin(), 1);
                }

                level.size.Set(sizes);

                if (info.format == py::format_descriptor<uint8_t>::format())
                {
                    array->element_type.Set(DataElementType::UINT8);
                }
                else if (info.format == py::format_descriptor<int8_t>::format())
                {
                    array->element_type.Set(DataElementType::INT8);
                }
                else if (info.format == py::format_descriptor<uint16_t>::format())
                {
                    array->element_type.Set(DataElementType::UINT16);
                }
                else if (info.format == py::format_descriptor<int16_t>::format())
                {
                    array->element_type.Set(DataElementType::INT16);
                }
                else if (info.format == py::format_descriptor<uint32_t>::format())
                {
                    array->element_type.Set(DataElementType::UINT32);
                }
                else if (info.format == py::format_descriptor<int32_t>::format())
                {
                    array->element_type.Set(DataElementType::INT32);
                }
                else if (info.format == py::format_descriptor<float>::format())
                {
                    array->element_type.Set(DataElementType::FLOAT);
                }
                else
                {
                    throw RuntimeError() << "Unhandled buffer format " << info.format;
                }
            }

            auto level = array->levels.begin();
            for (auto &&input_element_size : input_array.element_sizes_)
            {
                if (level == array->levels.end())
                {
                    throw RuntimeError() << "Too many elements in element_sizes_";
                }
                level->element_size.Set(input_element_size);
                ++level;
            }
        }
    }

    // set the data
    {
        for (auto &&input_array : arrays)
        {
            const std::string array_id(input_array.dimension_order_);

            for (auto &&input_level : input_array.levels_)
            {
                const py::buffer_info info = input_level.request();

                // If no data is provided, then skip. Data will be fetched on demand later.
                if (!info.ptr)
                {
                    continue;
                }

                uint32_t dimensions = info.ndim;

                // shape is reverse to expected size
                std::vector<uint32_t> sizes(dimensions), strides(dimensions);
                for (int index = 0; index < dimensions; ++index)
                {
                    sizes[dimensions - index - 1]   = info.shape[index];
                    strides[dimensions - index - 1] = info.strides[index];
                }

                if (dimensions == array_id.length() - 1)
                {
                    // for single component data the buffer_info does not have an axis, add it
                    ++dimensions;
                    sizes.insert(sizes.begin(), 1);
                    strides.insert(strides.begin(), info.itemsize);
                }

                DataInterface::AccessGuard access(*data_interface_.get());

                access->array_id.Set(array_id);
                access->level.Set(0);
                access->size.Set(sizes);
                std::vector<uint32_t> offset;
                for (int index = 0; index < dimensions; ++index)
                {
                    offset.push_back(0);
                }
                access->offset.Set(offset);

                // calculate the size of each dimension
                std::vector<size_t> dim_size(dimensions);
                dim_size[0] = info.itemsize * sizes[0];
                for (int index = 1; index < dimensions; ++index)
                {
                    dim_size[index] = sizes[index] * dim_size[index - 1];
                }

                // check for contiguous regions to copy
                uint32_t contiguous_index = 0;
                for (int index = 0; index < dimensions; ++index)
                {
                    if (sizes[index] * strides[index] != dim_size[index])
                    {
                        break;
                    }
                    ++contiguous_index;
                }

                // allocate the destination data
                std::unique_ptr<std::vector<uint8_t>> data =
                    std::make_unique<std::vector<uint8_t>>(dim_size[dimensions - 1]);

                if (contiguous_index == dimensions)
                {
                    // full copy, all data contiguous
                    std::memcpy(data->data(), info.ptr, dim_size[dimensions - 1]);
                }
                else
                {
                    // partial copies
                    uintptr_t dst_data           = reinterpret_cast<uintptr_t>(data->data());
                    const uintptr_t dst_data_end = dst_data + dim_size[dimensions - 1];
                    size_t dst_increment_bytes =
                        (contiguous_index > 0) ? dim_size[contiguous_index - 1] : info.itemsize;

                    std::vector<uint32_t> src_index(dimensions);
                    for (int index = 0; index < dimensions; ++index)
                    {
                        src_index[index] = 0;
                    }

                    while (dst_data != dst_data_end)
                    {
                        // calculate src data pointer
                        uintptr_t src_data = reinterpret_cast<uintptr_t>(info.ptr);
                        for (int index = 0; index < dimensions; ++index)
                        {
                            src_data += src_index[index] * strides[index];
                        }

                        // copy
                        std::memcpy(reinterpret_cast<void *>(dst_data), reinterpret_cast<const void *>(src_data),
                                    dst_increment_bytes);

                        // increment source index
                        for (int index = contiguous_index; index < dimensions; ++index)
                        {
                            ++src_index[index];
                            if (src_index[index] == sizes[index])
                            {
                                src_index[index] = 0;
                            }
                            else
                            {
                                break;
                            }
                        }

                        dst_data += dst_increment_bytes;
                    }
                }

                access->blob.reset(new StdContainerBlob<std::vector<uint8_t>>(std::move(data)));
            }
        }
    }
}

std::vector<PythonRenderer::Array> PythonRenderer::Impl::GetArrays()
{
    std::vector<Array> output_arrays;

    // Get the configuration
    {
        DataConfigInterface::AccessGuardConst access(&data_config_interface_);

        for (auto &&array : access->arrays)
        {
            const uint32_t dimensions = array.dimension_order.Get().size();

            output_arrays.emplace_back();
            Array &output_array = output_arrays.back();

            output_array.dimension_order_ = array.dimension_order.Get();
            output_array.permute_axes_    = array.permute_axis.Get();
            output_array.flip_axes_       = array.flip_axes.Get();

            // extend arrays if needed
            while (output_array.permute_axes_.size() < dimensions)
            {
                output_array.permute_axes_.push_back(output_array.permute_axes_.size());
            }
            output_array.flip_axes_.resize(dimensions, false);

            for (auto &&level : array.levels)
            {
                const auto &size = level.size.Get();

                std::vector<size_t> shape(dimensions);
                // shape is reverse to size
                for (int index = 0; index < dimensions; ++index)
                {
                    shape[index] = size[size.size() - index - 1];
                }

                switch (array.element_type.Get())
                {
                case DataElementType::INT8:
                    output_array.levels_.emplace_back(pybind11::dtype::of<int8_t>(), shape);
                    break;
                case DataElementType::UINT8:
                    output_array.levels_.emplace_back(pybind11::dtype::of<uint8_t>(), shape);
                    break;
                case DataElementType::INT16:
                    output_array.levels_.emplace_back(pybind11::dtype::of<int16_t>(), shape);
                    break;
                case DataElementType::UINT16:
                    output_array.levels_.emplace_back(pybind11::dtype::of<uint16_t>(), shape);
                    break;
                case DataElementType::INT32:
                    output_array.levels_.emplace_back(pybind11::dtype::of<int32_t>(), shape);
                    break;
                case DataElementType::UINT32:
                    output_array.levels_.emplace_back(pybind11::dtype::of<uint32_t>(), shape);
                    break;
                case DataElementType::FLOAT:
                    output_array.levels_.emplace_back(pybind11::dtype::of<float>(), shape);
                    break;
                default:
                    throw RuntimeError() << "Unhandled data type";
                }

                output_array.element_sizes_.push_back(level.element_size.Get());
                // extend coordinate array if needed
                output_array.element_sizes_.back().resize(dimensions, 1.f);
            }
        }
    }

    return output_arrays;
}

void PythonRenderer::Impl::DeduceSettings(ViewMode view_mode)
{
    json_settings_->DeduceSettings(view_mode);
}

void PythonRenderer::Impl::SetSettings(const nlohmann::json &new_settings)
{
    json_settings_->SetSettings(new_settings);
}

void PythonRenderer::Impl::MergeSettings(const nlohmann::json &new_settings)
{
    json_settings_->MergeSettings(new_settings);
}

nlohmann::json PythonRenderer::Impl::GetSettings()
{
    return json_settings_->GetSettings();
}

std::unique_ptr<VideoStream> PythonRenderer::Impl::CreateVideoStream(
    const std::function<void(py::object, bool)> &callback)
{
    return std::make_unique<VideoStream>(video_interface_, callback);
}

std::unique_ptr<ImageCapsule> PythonRenderer::Impl::RenderImage(uint32_t width, uint32_t height,
                                                                ColorImageType image_type)
{
    // trigger the rendering of the image
    {
        ImageInterface::AccessGuard access(image_interface_);

        access->width.Set(width);
        access->height.Set(height);
        access->color_type = image_type;
    }

    std::unique_ptr<ImageCapsule> image_capsule = std::make_unique<ImageCapsule>();

    image_capsule->width  = width;
    image_capsule->height = height;

    // then wait for the message with the encoded data
    image_capsule->image = image_interface_output_->WaitForEncodedData();

    if (image_capsule->image->color_type_ == ColorImageType::UNKNOWN)
    {
        // rendering failed, return empty object
        Log(LogLevel::Warning) << "Rendering failed";
        return std::unique_ptr<ImageCapsule>();
    }
    else if (image_capsule->image->color_type_ != ColorImageType::JPEG)
    {
        // data is on device memory, copy to host
        auto host_message = std::make_shared<ImageEncodedDataMessage>();

        host_message->color_type_ = image_capsule->image->color_type_;
        host_message->color_memory_.reset(new StdContainerBlob<std::vector<uint8_t>>(
            std::make_unique<std::vector<uint8_t>>(image_capsule->image->color_memory_->GetSize())));

        {
            auto access_src = image_capsule->image->color_memory_->AccessConst(CU_STREAM_PER_THREAD);
            auto access_dst = host_message->color_memory_->Access(CU_STREAM_PER_THREAD);
            CudaCheck(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(access_dst->GetData()),
                                    reinterpret_cast<CUdeviceptr>(access_src->GetData()),
                                    image_capsule->image->color_memory_->GetSize(), CU_STREAM_PER_THREAD));
            CudaCheck(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        }

        image_capsule->image = host_message;
    }

    return image_capsule;
}

void PythonRenderer::Impl::Reset(const std::vector<InterfaceSelector> &selectors)
{
    for (auto &&selector : selectors)
    {
        switch (selector)
        {
        case InterfaceSelector::ALL:
            background_light_interface_.Reset();
            camera_interface_.Reset();
            camera_aperture_interface_.Reset();
            data_config_interface_.Reset();
            data_crop_interface_.Reset();
            data_transform_interface_.Reset();
            data_interface_->Reset();
            data_view_interface_.Reset();
            light_interface_.Reset();
            post_process_denoise_interface_.Reset();
            post_process_tonemap_interface_.Reset();
            render_settings_interface_.Reset();
            transfer_function_interface_.Reset();
            view_interface_.Reset();
            break;
        case InterfaceSelector::BACKGROUND_LIGHT:
            background_light_interface_.Reset();
            break;
        case InterfaceSelector::CAMERA:
            camera_interface_.Reset();
            break;
        case InterfaceSelector::CAMERA_APERTURE:
            camera_aperture_interface_.Reset();
            break;
        case InterfaceSelector::DATA_CONFIG:
            data_config_interface_.Reset();
            break;
        case InterfaceSelector::DATA_CROP:
            data_crop_interface_.Reset();
            break;
        case InterfaceSelector::DATA_TRANSFORM:
            data_transform_interface_.Reset();
            break;
        case InterfaceSelector::DATA:
            data_interface_->Reset();
            break;
        case InterfaceSelector::DATA_VIEW:
            data_view_interface_.Reset();
            break;
        case InterfaceSelector::LIGHT:
            light_interface_.Reset();
            break;
        case InterfaceSelector::POST_PROCESS_DENOISE:
            post_process_denoise_interface_.Reset();
            break;
        case InterfaceSelector::POST_PROCESS_TONEMAP:
            post_process_tonemap_interface_.Reset();
            break;
        case InterfaceSelector::RENDER_SETTINGS:
            render_settings_interface_.Reset();
            break;
        case InterfaceSelector::TRANSFER_FUNCTION:
            transfer_function_interface_.Reset();
            break;
        case InterfaceSelector::VIEW:
            view_interface_.Reset();
            break;
        default:
            Log(LogLevel::Warning) << "Unhandled reset interface "
                                   << static_cast<std::underlying_type_t<InterfaceSelector>>(selector);
            break;
        }
    }
}

PythonRenderer::PythonRenderer(const std::vector<uint32_t> &cuda_device_ordinals)
    : impl_(new Impl(cuda_device_ordinals))
{
}

void PythonRenderer::SetArrays(const std::vector<Array> &arrays, const FetchFunc &fetch_func)
{
    impl_->SetArrays(arrays, fetch_func);
}

std::vector<PythonRenderer::Array> PythonRenderer::GetArrays()
{
    return impl_->GetArrays();
}

void PythonRenderer::DeduceSettings(ViewMode view_mode)
{
    impl_->DeduceSettings(view_mode);
}

void PythonRenderer::SetSettings(const nlohmann::json &new_settings)
{
    impl_->SetSettings(new_settings);
}

void PythonRenderer::MergeSettings(const nlohmann::json &new_settings)
{
    impl_->MergeSettings(new_settings);
}

nlohmann::json PythonRenderer::GetSettings()
{
    return impl_->GetSettings();
}

std::unique_ptr<VideoStream> PythonRenderer::CreateVideoStream(const std::function<void(py::object, bool)> &callback)
{
    return impl_->CreateVideoStream(callback);
}

std::unique_ptr<ImageCapsule> PythonRenderer::RenderImage(uint32_t width, uint32_t height, ColorImageType image_type)
{
    return impl_->RenderImage(width, height, image_type);
}

void PythonRenderer::Reset(const std::vector<InterfaceSelector> &selectors)
{
    impl_->Reset(selectors);
}

} // namespace clara::viz
