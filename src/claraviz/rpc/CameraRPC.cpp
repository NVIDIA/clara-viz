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

#include "claraviz/rpc/CameraRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace detail
{

void CameraContext::ExecuteRPC(nvidia::claraviz::core::CameraRequest &request,
                               nvidia::claraviz::core::CameraResponse &response)
{
    CameraInterface::AccessGuard access(GetResources()->camera_);

    CameraInterface::DataIn::Camera *camera = access->GetOrAddCamera(request.name());

    switch (request.enable_pose())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        camera->enable_pose = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        camera->enable_pose = false;
        break;
    }

    if (request.has_eye())
    {
        camera->eye.Set(MakeVector3f(request.eye()));
    }
    if (request.has_look_at())
    {
        camera->look_at.Set(MakeVector3f(request.look_at()));
    }
    if (request.has_up())
    {
        camera->up.Set(MakeVector3f(request.up()));
    }

    if (request.has_pose())
    {
        camera->pose = MakeMatrix4x4(request.pose());
    }

    if (request.field_of_view() != 0.f)
    {
        camera->field_of_view.Set(request.field_of_view());
    }
    if (request.pixel_aspect_ratio() != 0.f)
    {
        camera->pixel_aspect_ratio.Set(request.pixel_aspect_ratio());
    }

    switch (request.enable_stereo())
    {
    case nvidia::claraviz::core::SWITCH_ENABLE:
        camera->enable_stereo = true;
        break;
    case nvidia::claraviz::core::SWITCH_DISABLE:
        camera->enable_stereo = false;
        break;
    }

    if (request.has_left_eye_pose())
    {
        camera->left_eye_pose = MakeMatrix4x4(request.left_eye_pose());
    }
    if (request.has_right_eye_pose())
    {
        camera->right_eye_pose = MakeMatrix4x4(request.right_eye_pose());
    }

    if (request.has_left_gaze_direction())
    {
        camera->left_gaze_direction.Set(MakeVector3f(request.left_gaze_direction()));
    }
    if (request.has_right_gaze_direction())
    {
        camera->right_gaze_direction.Set(MakeVector3f(request.right_gaze_direction()));
    }

    if (request.has_left_tangent_x())
    {
        camera->left_tangent_x = MakeVector2f(request.left_tangent_x());
    }
    if (request.has_left_tangent_y())
    {
        camera->left_tangent_y = MakeVector2f(request.left_tangent_y());
    }
    if (request.has_right_tangent_x())
    {
        camera->right_tangent_x = MakeVector2f(request.right_tangent_x());
    }
    if (request.has_right_tangent_y())
    {
        camera->right_tangent_y = MakeVector2f(request.right_tangent_y());
    }

    if (request.has_depth_clip())
    {
        camera->depth_clip.Set(MakeVector2f(request.depth_clip()));
    }

    if (request.has_depth_range())
    {
        camera->depth_range.Set(MakeVector2f(request.depth_range()));
    }
}

} // namespace detail

} // namespace clara::viz
