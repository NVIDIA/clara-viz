/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/CameraInterface.h"

#include <algorithm>

#include "claraviz/util/Validator.h"

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(CameraInterface::Message);

template<>
CameraInterface::DataIn::Camera::Camera()
    : enable_pose(false)
    , eye(Vector3f(0.f, 0.f, -1.f),
          [this](const Vector3f &value) { ValidatorDifferent(value, look_at.Get(), "Eye position and look at point"); })
    , look_at(Vector3f(0.f, 0.f, 0.f),
              [this](const Vector3f &value) { ValidatorDifferent(value, eye.Get(), "Look at point and eye position"); })
    , up(Vector3f(0.f, 1.f, 0.f), [](const Vector3f &value) { ValidatorUnitVector(value, "Up direction"); })
    , field_of_view(30.f, [](const float value) { ValidatorMinMaxExclusive(value, 0.0f, 360.f, "Field of view"); })
    , pixel_aspect_ratio(1.f, [](const float value) { ValidatorMinExclusive(value, 0.0f, "Pixel aspect ratio"); })
    , enable_stereo(false)
    , left_gaze_direction(Vector3f(0.f, 0.f, 1.f),
                          [](const Vector3f &value) { ValidatorUnitVector(value, "Left gaze direction"); })
    , right_gaze_direction(Vector3f(0.f, 0.f, 1.f),
                           [](const Vector3f &value) { ValidatorUnitVector(value, "Right gaze direction"); })
    , left_tangent_x(-1.0, 1.0)
    , left_tangent_y(1.0, -1.0)
    , right_tangent_x(-1.0, 1.0)
    , right_tangent_y(1.0, -1.0)
    , depth_clip(Vector2f(0.f, std::numeric_limits<float>::max()),
                 [](const Vector2f value) {
                     ValidatorRange(value, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(),
                                    "Depth clip");
                 })
    , depth_range(Vector2f(0.f, std::numeric_limits<float>::max()),
                  [](const Vector2f value) { ValidatorDifferent(value(0), value(1), "Depth range"); })
{
}

template<>
CameraInterface::DataIn::CameraInterfaceData()
{
    cameras.emplace_back();
}

template<>
CameraInterface::DataOut::CameraInterfaceData()
{
}

template<>
template<>
CameraInterface::DataIn::Camera *CameraInterface::DataIn::GetOrAddCamera(const std::string &name)
{
    std::list<Camera>::iterator it =
        std::find_if(cameras.begin(), cameras.end(), [name](const Camera &camera) { return camera.name == name; });
    if (it == cameras.end())
    {
        cameras.emplace_back();
        it = cameras.end();
        --it;
        it->name = name;
    }
    return &*it;
}

namespace detail
{

template<typename T>
typename T::Camera *GetCamera(std::list<typename T::Camera> &cameras, const std::string &name)
{
    typename std::list<typename T::Camera>::iterator it = std::find_if(
        cameras.begin(), cameras.end(), [name](const typename T::Camera &camera) { return camera.name == name; });
    if (it == cameras.end())
    {
        throw InvalidArgument("name") << "Camera with name '" << name << "' not found";
    }
    return &*it;
}

template<typename T>
const typename T::Camera *GetCamera(const std::list<typename T::Camera> &cameras, const std::string &name)
{
    typename std::list<typename T::Camera>::const_iterator it = std::find_if(
        cameras.cbegin(), cameras.cend(), [name](const typename T::Camera &camera) { return camera.name == name; });
    if (it == cameras.end())
    {
        throw InvalidArgument("name") << "Camera with name '" << name << "' not found";
    }
    return &*it;
}

} // namespace detail

template<>
template<>
CameraInterface::DataIn::Camera *CameraInterface::DataIn::GetCamera(const std::string &name)
{
    return detail::GetCamera<CameraInterface::DataIn>(cameras, name);
}

template<>
const CameraInterface::DataIn::Camera *CameraInterface::DataIn::GetCamera(const std::string &name) const
{
    return detail::GetCamera<const CameraInterface::DataIn>(cameras, name);
}

template<>
const CameraInterface::DataOut::Camera *CameraInterface::DataOut::GetCamera(const std::string &name) const
{
    return detail::GetCamera<const CameraInterface::DataOut>(cameras, name);
}

template<>
CameraInterface::DataOut::Camera::Camera()
{
}

/**
 * Copy a camera interface structure to a camera POD structure.
 */
template<>
CameraInterface::DataOut CameraInterface::Get()
{
    AccessGuardConst access(this);

    CameraInterface::DataOut data_out;

    data_out.cameras.clear();
    for (auto &&camera_in : access->cameras)
    {
        data_out.cameras.emplace_back();
        CameraInterface::DataOut::Camera &camera_out = data_out.cameras.back();

        camera_out.name                 = camera_in.name;
        camera_out.enable_pose          = camera_in.enable_pose;
        camera_out.eye                  = camera_in.eye.Get();
        camera_out.look_at              = camera_in.look_at.Get();
        camera_out.up                   = camera_in.up.Get();
        camera_out.field_of_view        = camera_in.field_of_view.Get();
        camera_out.pose                 = camera_in.pose;
        camera_out.pixel_aspect_ratio   = camera_in.pixel_aspect_ratio.Get();
        camera_out.enable_stereo        = camera_in.enable_stereo;
        camera_out.left_eye_pose        = camera_in.left_eye_pose;
        camera_out.right_eye_pose       = camera_in.right_eye_pose;
        camera_out.left_gaze_direction  = camera_in.left_gaze_direction.Get();
        camera_out.right_gaze_direction = camera_in.right_gaze_direction.Get();
        camera_out.left_tangent_x       = camera_in.left_tangent_x;
        camera_out.left_tangent_y       = camera_in.left_tangent_y;
        camera_out.right_tangent_x      = camera_in.right_tangent_x;
        camera_out.right_tangent_y      = camera_in.right_tangent_y;
        camera_out.depth_clip           = camera_in.depth_clip.Get();
        camera_out.depth_range          = camera_in.depth_range.Get();
    }

    return data_out;
}

} // namespace clara::viz
