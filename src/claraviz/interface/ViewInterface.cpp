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
#include "claraviz/interface/ViewInterface.h"

#include <type_traits>

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(ViewInterface::Message);

std::ostream &operator<<(std::ostream &os, const ViewMode &view_mode)
{
    switch (view_mode)
    {
    case ViewMode::CINEMATIC:
        os << std::string("CINEMATIC");
        break;
    case ViewMode::SLICE:
        os << std::string("SLICE");
        break;
    case ViewMode::SLICE_SEGMENTATION:
        os << std::string("SLICE_SEGMENTATION");
        break;
    case ViewMode::TWOD:
        os << std::string("TWOD");
        break;
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const StereoMode &stereo_mode)
{
    switch (stereo_mode)
    {
    case StereoMode::OFF:
        os << std::string("OFF");
        break;
    case StereoMode::LEFT:
        os << std::string("LEFT");
        break;
    case StereoMode::RIGHT:
        os << std::string("RIGHT");
        break;
    case StereoMode::TOP_BOTTOM:
        os << std::string("TOP_BOTTOM");
        break;
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

template<>
ViewInterface::DataIn::View::View()
    : mode(ViewMode::CINEMATIC)
    , stereo_mode(StereoMode::OFF)
{
}

template<>
ViewInterface::DataOut::View::View()
{
}

template<>
ViewInterface::DataIn::ViewInterfaceData()
{
    // create the default view
    views.emplace_back();
}

template<>
ViewInterface::DataOut::ViewInterfaceData()
{
}

template<>
template<>
ViewInterface::DataIn::View *ViewInterface::DataIn::GetOrAddView(const std::string &name)
{
    std::list<View>::iterator it =
        std::find_if(views.begin(), views.end(), [name](const View &view) { return view.name == name; });
    if (it == views.end())
    {
        views.emplace_back();
        it = views.end();
        --it;
        it->name = name;
    }
    return &*it;
}

namespace detail
{

template<typename T>
typename T::View *GetView(std::list<typename T::View> &views, const std::string &name)
{
    typename std::list<typename T::View>::iterator it =
        std::find_if(views.begin(), views.end(), [name](const typename T::View &view) { return view.name == name; });
    if (it == views.end())
    {
        throw InvalidArgument("name") << "View with name '" << name << "' not found";
    }
    return &*it;
}

template<typename T>
const typename T::View *GetView(const std::list<typename T::View> &views, const std::string &name)
{
    typename std::list<typename T::View>::const_iterator it =
        std::find_if(views.cbegin(), views.cend(), [name](const typename T::View &view) { return view.name == name; });
    if (it == views.end())
    {
        throw InvalidArgument("name") << "View with name '" << name << "' not found";
    }
    return &*it;
}

} // namespace detail

template<>
template<>
ViewInterface::DataIn::View *ViewInterface::DataIn::GetView(const std::string &name)
{
    return detail::GetView<ViewInterface::DataIn>(views, name);
}

template<>
const ViewInterface::DataIn::View *ViewInterface::DataIn::GetView(const std::string &name) const
{
    return detail::GetView<const ViewInterface::DataIn>(views, name);
}

template<>
const ViewInterface::DataOut::View *ViewInterface::DataOut::GetView(const std::string &name) const
{
    return detail::GetView<const ViewInterface::DataOut>(views, name);
}

/**
 * Copy a view interface structure to a view POD structure.
 */
template<>
ViewInterface::DataOut ViewInterface::Get()
{
    AccessGuardConst access(this);

    ViewInterface::DataOut data_out;

    data_out.views.clear();
    for (auto &&view_in : access->views)
    {
        data_out.views.emplace_back();
        ViewInterface::DataOut::View &view_out = data_out.views.back();

        view_out.name           = view_in.name;
        view_out.stream_name    = view_in.stream_name;
        view_out.mode           = view_in.mode;
        view_out.camera_name    = view_in.camera_name;
        view_out.data_view_name = view_in.data_view_name;
        view_out.stereo_mode    = view_in.stereo_mode;
    }

    return data_out;
}

} // namespace clara::viz
