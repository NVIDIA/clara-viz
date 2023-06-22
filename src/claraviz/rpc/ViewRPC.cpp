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

#include "claraviz/rpc/ViewRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void ViewContext::ExecuteRPC(cinematic_v1::ViewRequest &request, cinematic_v1::ViewResponse &response)
{
    ViewInterface::AccessGuard access(GetResources()->view_);

    ViewInterface::DataIn::View *view = access->GetOrAddView(request.name());

    view->name        = request.name();
    view->stream_name = request.stream_name();

    switch (request.mode())
    {
    case cinematic_v1::ViewRequest::MODE_UNKNOWN:
    case cinematic_v1::ViewRequest::CINEMATIC:
        view->mode = ViewMode::CINEMATIC;
        break;
    case cinematic_v1::ViewRequest::SLICE:
        view->mode = ViewMode::SLICE;
        break;
    case cinematic_v1::ViewRequest::SLICE_SEGMENTATION:
        view->mode = ViewMode::SLICE_SEGMENTATION;
        break;
    case cinematic_v1::ViewRequest::TWOD:
        view->mode = ViewMode::TWOD;
        break;
    }

    view->camera_name    = request.camera_name();
    view->data_view_name = request.data_view_name();

    switch (request.stereo_mode())
    {
    case cinematic_v1::ViewRequest::STEREO_MODE_OFF:
        view->stereo_mode = StereoMode::OFF;
        break;
    case cinematic_v1::ViewRequest::STEREO_MODE_LEFT:
        view->stereo_mode = StereoMode::LEFT;
        break;
    case cinematic_v1::ViewRequest::STEREO_MODE_RIGHT:
        view->stereo_mode = StereoMode::RIGHT;
        break;
    case cinematic_v1::ViewRequest::STEREO_MODE_TOP_BOTTOM:
        view->stereo_mode = StereoMode::TOP_BOTTOM;
        break;
    }
}

} // namespace detail

} // namespace clara::viz
