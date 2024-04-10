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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <pybind11_json/pybind11_json.hpp>

#include "PythonRenderer.h"
#include "ImageCapsule.h"
#include "VideoStream.h"

#include <claraviz/interface/ViewInterface.h>
#include <claraviz/interface/DataInterface.h>
#include <claraviz/util/Blob.h>

#include <memory>
#include <optional>
#include <vector>

namespace py = pybind11;

namespace clara::viz
{

// These classes are used to export enums to Python as 'Type.VALUE', e.g.
// 'ColorImageType.JPEG'
class ColorImageTypeExport
{
public:
    using EnumType = clara::viz::ColorImageType;
};

class InterfaceSelectorExport
{
public:
    using EnumType = PythonRenderer::InterfaceSelector;
};

class ViewModeExport
{
public:
    using EnumType = clara::viz::ViewMode;
};

// Call the module '_viz' since we need to export both the C++ bindings and
// the python code for e.g. the Jupyter widget. The python code will import the
// '_viz' module and export all symbols
PYBIND11_MODULE(_viz, m)
{
    // export enums, see
    // https://pybind11.readthedocs.io/en/stable/classes.html?highlight=enum_#enumerations-and-internal-types
    py::class_<ColorImageTypeExport> image_type(m, "ColorImageType");

    py::enum_<ColorImageTypeExport::EnumType>(image_type, "ColorImageType")
        .value("RAW_RGBA_U8", clara::viz::ColorImageType::RAW_RGBA_U8,
               R"pbdoc(RAW uncompressed RGBA unsigned int 8-bit)pbdoc")
        .value("JPEG", clara::viz::ColorImageType::JPEG, R"pbdoc(JPEG)pbdoc")
        .export_values();

    py::class_<InterfaceSelectorExport> interface_selector(m, "InterfaceSelector");

    py::enum_<InterfaceSelectorExport::EnumType>(interface_selector, "InterfaceSelector")
        .value("ALL", PythonRenderer::InterfaceSelector::ALL)
        .value("CAMERA", PythonRenderer::InterfaceSelector::CAMERA)
        .value("CAMERA_APERTURE", PythonRenderer::InterfaceSelector::CAMERA_APERTURE)
        .value("DATA", PythonRenderer::InterfaceSelector::DATA)
        .value("DATA_CONFIG", PythonRenderer::InterfaceSelector::DATA_CONFIG)
        .value("DATA_CROP", PythonRenderer::InterfaceSelector::DATA_CROP)
        .value("DATA_TRANSFORM", PythonRenderer::InterfaceSelector::DATA_TRANSFORM)
        .value("DATA_VIEW", PythonRenderer::InterfaceSelector::DATA_VIEW)
        .value("LIGHT", PythonRenderer::InterfaceSelector::LIGHT)
        .value("BACKGROUND_LIGHT", PythonRenderer::InterfaceSelector::BACKGROUND_LIGHT)
        .value("POST_PROCESS_DENOISE", PythonRenderer::InterfaceSelector::POST_PROCESS_DENOISE)
        .value("POST_PROCESS_TONEMAP", PythonRenderer::InterfaceSelector::POST_PROCESS_TONEMAP)
        .value("RENDER_SETTINGS", PythonRenderer::InterfaceSelector::RENDER_SETTINGS)
        .value("TRANSFER_FUNCTION", PythonRenderer::InterfaceSelector::TRANSFER_FUNCTION)
        .value("VIEW", PythonRenderer::InterfaceSelector::VIEW)
        .export_values();

    py::class_<ViewModeExport> view_mode(m, "ViewMode");

    py::enum_<ViewModeExport::EnumType>(view_mode, "ViewMode")
        .value("CINEMATIC", clara::viz::ViewMode::CINEMATIC,
               R"pbdoc(
    3D Cinematic render view

    The view is using a perspective projection. Data is displayed using realistic lighting and shadows. Transfer
    functions are used to map from input data to material properties.
)pbdoc")
        .value("SLICE", clara::viz::ViewMode::SLICE,
               R"pbdoc(
    3D Slice view

    The view is using an orthographic projection. The vector between the camera 'look_at' and 'eye' points define
    the view direction. The 'eye' point of the camera defines the slice to display within the volumetric data.
    The size of the data is defined by the data array level configuration 'size' and 'element_size' parameters, the
    data is also limited to the data crop settings.
    The 'fov' camera parameter defines the width of the viewing frustum.
)pbdoc")
        .value("SLICE_SEGMENTATION", clara::viz::ViewMode::SLICE_SEGMENTATION,
               R"pbdoc(
    3D Slice with segmenation view

    Same as the 'SLICE' mode above but when a segmentation mask is specified the segments are colored with the transfer
    function emissive and diffuse color blended with the density of the data.
)pbdoc")
        .value("TWOD", clara::viz::ViewMode::TWOD,
               R"pbdoc(
    2D n-dimensional data view

    The view is displaying generic n-dimensional data. The section of the data to display is defined by 'data_view_name'.
)pbdoc")
        .export_values();

    py::class_<PythonRenderer::Array>(m, "Array")
        .def(py::init<py::array &, const std::string &, const std::vector<uint32_t> &, const std::vector<bool> &,
                      const std::vector<float> &>(),
             py::arg("array"), py::arg("dimension_order") = std::string(),
             py::arg("permute_axes") = std::vector<uint32_t>(), py::arg("flip_axes") = std::vector<bool>(),
             py::arg("element_size") = std::vector<float>(),
             R"pbdoc(
Construct an array.

Args:
    array: numpy array
    dimension_order: a string defining the data organization and format.
        Each character defines a dimension starting with the fastest varying axis
        and ending with the slowest varying axis. For example a 2D color image
        is defined as 'CXY', a time sequence of density volumes is defined as
        'DXYZT'.
        Each character can occur only once. Either one of the data element definition
        characters 'C', 'D' or 'M' and the 'X' axis definition has to be present.
        - 'X': width
        - 'Y': height
        - 'Z': depth
        - 'T': time
        - 'I': sequence
        - 'C': RGB(A) color
        - 'D': density
        - 'M': mask
    permute_axes: Permutes the given data axes, e.g. to swap x and y of a 3-dimensional
        density array specify (0, 2, 1, 3)
    flip_axes: flips the given axes
    element_size: Physical size of each element, the order is defined by the 'dimension_order' field. For
        elements which have no physical size like 'M' or 'T' the corresponding value is 1.0.
                )pbdoc")
        .def(py::init<const std::vector<py::array> &, const std::string &, const std::vector<uint32_t> &,
                      const std::vector<bool> &, const std::vector<std::vector<float>> &>(),
             py::arg("levels"), py::arg("dimension_order") = std::string(),
             py::arg("permute_axes") = std::vector<uint32_t>(), py::arg("flip_axes") = std::vector<bool>(),
             py::arg("element_sizes") = std::vector<std::vector<float>>(),
             R"pbdoc(
Construct an array holding multi-dimensional data.

Args:
    levels: array of numpy arrays with the data for each level for multi-dimensional data, an array
            with a single numpy array for other (e.g. volume) data
    dimension_order: a string defining the data organization and format.
        Each character defines a dimension starting with the fastest varying axis
        and ending with the slowest varying axis. For example a 2D color image
        is defined as 'CXY', a time sequence of density volumes is defined as
        'DXYZT'.
        Each character can occur only once. Either one of the data element definition
        characters 'C', 'D' or 'M' and the 'X' axis definition has to be present.
        - 'X': width
        - 'Y': height
        - 'Z': depth
        - 'T': time
        - 'I': sequence
        - 'C': RGB(A) color
        - 'D': density
        - 'M': mask
    permute_axes: Permutes the given data axes, e.g. to swap x and y of a 3-dimensional
        density array specify (0, 2, 1, 3)
    flip_axes: flips the given axes
    element_sizes: Physical size of an element for each level. The order is defined by the 'dimension_order' field. For
        elements which have no physical size like 'M' or 'T' the corresponding value is 1.0.
        For multi-dimensional data this is an array of element sizes, else an array with a single
        element.
                )pbdoc")
        .def_readwrite("levels", &PythonRenderer::Array::levels_, R"pbdoc(numpy array for each level)pbdoc")
        .def_readwrite("dimension_order", &PythonRenderer::Array::dimension_order_, R"pbdoc(data organization)pbdoc")
        .def_readwrite("permute_axes", &PythonRenderer::Array::permute_axes_, R"pbdoc(permutes the given axes)pbdoc")
        .def_readwrite("flip_axes", &PythonRenderer::Array::flip_axes_, R"pbdoc(flips the given axes)pbdoc")
        .def_readwrite("element_sizes", &PythonRenderer::Array::element_sizes_,
                       R"pbdoc(physical size of each element for each level)pbdoc");

    // video stream class
    py::class_<VideoStream>(m, "VideoStream")
        .def("configure", &VideoStream::Configure, py::call_guard<py::gil_scoped_release>(), py::arg("width"),
             py::arg("height"), py::arg("frame_rate"), py::arg("bit_rate"),
             R"pbdoc(
Configure the video stream.

Args:
    width: Width of the video, value needs to evenly divisible by 2
    height: Height of the video, value needs to evenly divisible by 2
    frame_rate: Target framerate of the video. If set to 0.0 the frames will delivered when rendering is done. Converging renderers
        will deliver the final frame only.
    bit_rate: Target bitrate of the video
             )pbdoc")
        .def("play", &VideoStream::Play, py::call_guard<py::gil_scoped_release>(), R"pbdoc(Play video.)pbdoc")
        .def("pause", &VideoStream::Pause, py::call_guard<py::gil_scoped_release>(), R"pbdoc(Pause video.)pbdoc")
        .def("stop", &VideoStream::Stop, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(Stop video. Video stream is closed if it had been open.)pbdoc");

    // image capsule class
    // used to provide buffer protocol access to rendered image data
    py::class_<ImageCapsule>(m, "ImageCapsule", py::buffer_protocol())
        .def_buffer([](ImageCapsule &i) -> py::buffer_info {
            if (i.image->color_type_ == ColorImageTypeExport::EnumType::JPEG)
            {
                i.access = i.image->color_memory_->AccessConst();
                return py::buffer_info(
                    const_cast<uint8_t *>(static_cast<const uint8_t *>(i.access->GetData())), // Pointer to buffer
                    i.image->color_memory_->GetSize(),                                        // Total number of entries
                    true                                                                      // readonly
                );
            }
            else if (i.image->color_type_ == ColorImageTypeExport::EnumType::RAW_RGBA_U8)
            {
                i.access = i.image->color_memory_->AccessConst();
                return py::buffer_info(
                    const_cast<uint8_t *>(static_cast<const uint8_t *>(i.access->GetData())), // Pointer to buffer
                    {i.height, i.width, uint32_t(4)},                                         // Buffer dimensions
                    {sizeof(uint8_t) * 4 * i.width, // Strides (in bytes) for each index
                     sizeof(uint8_t) * 4, sizeof(uint8_t)},
                    true // readonly
                );
            }
            else
            {
                return py::buffer_info();
            }
        });

    // main class
    // NOTE: functions not using Python objects releases the GIL
    py::class_<PythonRenderer>(m, "Renderer")
        .def(py::init([]() {
                 /// @todo cuda device support
                 std::vector<uint32_t> cuda_device_ordinals{0};
                 return new PythonRenderer(cuda_device_ordinals);
             }),
             R"pbdoc(Constructor of the Clara Viz renderer)pbdoc")
        .def(py::init([](py::array density, std::optional<py::array> mask, clara::viz::ViewMode view_mode,
                         const nlohmann::json &settings) {
                 /// @todo cuda device support
                 std::vector<uint32_t> cuda_device_ordinals{0};
                 std::unique_ptr<PythonRenderer> renderer = std::make_unique<PythonRenderer>(cuda_device_ordinals);

                 // add the arrays
                 std::vector<PythonRenderer::Array> arrays;

                 arrays.emplace_back(density, "DXYZ");
                 if (mask.has_value())
                 {
                     arrays.emplace_back(mask.value(), "MXYZ");
                 }
                 renderer->SetArrays(arrays);

                 if (!settings.empty())
                 {
                     renderer->SetSettings(settings);
                 }
                 else
                 {
                     // if no settings are given try to deduce the optimal settings
                     renderer->DeduceSettings(view_mode);
                 }

                 return renderer.release();
             }),
             py::arg("density"), py::arg("mask") = py::none(),
             py::arg("view_mode") = ViewModeExport::EnumType::CINEMATIC, py::arg("settings") = nlohmann::json(),
             R"pbdoc(
Constructor of the Clara Viz renderer to render a density volume with an optional segmentation mask volume

Args:
    density: numpy array with density values
    mask: numpy array with mask values
    view_mode: initial view mode
    settings: initial settings in JSON format
                )pbdoc")
        .def(py::init([](py::object data_definition, clara::viz::ViewMode view_mode) {
                 /// @todo cuda device support
                 std::vector<uint32_t> cuda_device_ordinals{0};
                 std::unique_ptr<PythonRenderer> renderer = std::make_unique<PythonRenderer>(cuda_device_ordinals);

                 // add the arrays
                 std::vector<PythonRenderer::Array> arrays;
                 auto source_arrays = data_definition.attr("arrays");
                 for (auto &&array : source_arrays)
                 {
                     const std::string dimension_order = array.attr("dimension_order").cast<std::string>();

                     // switch to TWOD view mode when 2D data is detected
                     if (dimension_order.size() == 3)
                     {
                         view_mode = clara::viz::ViewMode::TWOD;
                     }

                     arrays.emplace_back(array.attr("levels").cast<std::vector<py::array>>(), dimension_order,
                                         array.attr("permute_axes").cast<std::vector<uint32_t>>(),
                                         array.attr("flip_axes").cast<std::vector<bool>>(),
                                         array.attr("element_sizes").cast<std::vector<std::vector<float>>>());
                 }

                 renderer->SetArrays(arrays, data_definition.attr("fetch_func").cast<PythonRenderer::FetchFunc>());

                 auto settings = data_definition.attr("settings").cast<nlohmann::json>();
                 if (!settings.empty())
                 {
                     renderer->SetSettings(settings);
                 }
                 else
                 {
                     // if no settings are given try to deduce the optimal settings
                     renderer->DeduceSettings(view_mode);
                 }

                 return renderer.release();
             }),
             py::arg("data_definition"), py::arg("view_mode") = ViewModeExport::EnumType::CINEMATIC,
             R"pbdoc(
Constructor of the Clara Viz renderer with a DataDefinition object

Args:
    data_definition: a DataDefinition object
    view_mode: initial view mode
                )pbdoc")
        .def("set_arrays", &PythonRenderer::SetArrays, py::arg("arrays"),
             py::arg("fetch_func") = PythonRenderer::FetchFunc(),
             R"pbdoc(
Set data arrays

Args:
    array: list of clara.viz.Array objects
    fetch_func: function to be called on demand data fetches
            )pbdoc")
        .def("get_arrays", &PythonRenderer::GetArrays,
             R"pbdoc(
Get data arrays

Returns:
    list of clara.viz.Array objects
            )pbdoc")
        .def("set_settings", &PythonRenderer::SetSettings, py::call_guard<py::gil_scoped_release>(),
             py::arg("new_settings"),
             R"pbdoc(
Set settings

Args:
    new_settings: json with new settings
            )pbdoc")
        .def("merge_settings", &PythonRenderer::MergeSettings, py::call_guard<py::gil_scoped_release>(),
             py::arg("new_settings"),
             R"pbdoc(
Merge settings

Args:
    new_settings: json with settings to merge in
            )pbdoc")
        .def("get_settings", &PythonRenderer::GetSettings, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
Get settings

Returns:
    json with settings
            )pbdoc")
        .def("deduce_settings", &PythonRenderer::DeduceSettings, py::call_guard<py::gil_scoped_release>(),
             py::arg("view_mode") = ViewModeExport::EnumType::CINEMATIC,
             R"pbdoc(
Deduce settings from configured data (data needs to be configered by SetArray()).
Make the whole dataset visible. Set a light in correct distance. Set a transfer function
using the histogram of the data.

Args:
    view_mode: view mode
            )pbdoc")
        .def("create_video_stream", &PythonRenderer::CreateVideoStream, py::call_guard<py::gil_scoped_release>(),
             py::arg("callback"),
             R"pbdoc(
Create a video stream.

Args:
    callback: video data callback function. The function receives the video stream data and a boolean flag.
        If this is true then this is the first data segment of a new stream. This can be triggered
        when e.g. the resolution changed.
        When writing to a file this can be used to close the current file and start writing to
        a new file. When streaming to a browser then a new MediaSource SourceBuffer object needs
        to be created.
Returns:
    Video stream object
            )pbdoc")
        .def("render_image", &PythonRenderer::RenderImage, py::call_guard<py::gil_scoped_release>(), py::arg("width"),
             py::arg("height"), py::arg("image_type") = ColorImageTypeExport::EnumType::JPEG,
             R"pbdoc(
Render an image.

Args:
    width: width of the image to render
    height: width of the image to render
    image_type: type of the image to render
Returns:
    Image data
            )pbdoc")
        .def("reset", &PythonRenderer::Reset, py::call_guard<py::gil_scoped_release>(),
             py::arg("selector") = std::vector{PythonRenderer::InterfaceSelector::ALL},
             R"pbdoc(
Reset selected interfaces to default.

Args:
    selectors: list of interfaces to reset to defaults
            )pbdoc");
}

} // namespace clara::viz
