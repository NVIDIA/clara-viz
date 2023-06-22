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

#include "claraviz/rpc/DataViewRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace detail
{

void DataViewContext::ExecuteRPC(nvidia::claraviz::cinematic::v1::DataViewRequest &request,
                                 nvidia::claraviz::cinematic::v1::DataViewResponse &response)
{
    DataViewInterface::AccessGuard access(GetResources()->data_view_);

    auto data_view = std::find_if(
        access->data_views.begin(), access->data_views.end(),
        [request](const DataViewInterface::DataIn::DataView &element) { return element.name == request.name(); });
    if (data_view == access->data_views.end())
    {
        access->data_views.emplace_back();
        data_view       = --(access->data_views.end());
        data_view->name = request.name();
    }

    if (request.zoom_factor() != 0.f)
    {
        data_view->zoom_factor.Set(request.zoom_factor());
    }
    if (request.has_view_offset())
    {
        data_view->view_offset = MakeVector2f(request.view_offset());
    }

    if (request.pixel_aspect_ratio() != 0.f)
    {
        data_view->pixel_aspect_ratio.Set(request.pixel_aspect_ratio());
    }
}

} // namespace detail

} // namespace clara::viz
