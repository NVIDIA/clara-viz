/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/interface/DataViewInterface.h"

#include <claraviz/util/Validator.h>

namespace clara::viz
{

template<>
DEFINE_CLASS_MESSAGEID(DataViewInterface::Message);

template<>
DataViewInterface::DataIn::DataView::DataView()
    : zoom_factor(1.f, [](const float value) { ValidatorMinInclusive(value, 1.0f, "Zoom factor"); })
    , view_offset(0.f, 0.f)
    , pixel_aspect_ratio(1.f, [](const float value) { ValidatorMinExclusive(value, 0.0f, "Pixel aspect ratio"); })
{
}

template<>
DataViewInterface::DataIn::DataViewInterfaceData()
{
    data_views.emplace_back();
}

template<>
DataViewInterface::DataOut::DataViewInterfaceData()
{
}

template<>
template<>
DataViewInterface::DataIn::DataView *DataViewInterface::DataIn::GetOrAddDataView(const std::string &name)
{
    std::list<DataView>::iterator it = std::find_if(
        data_views.begin(), data_views.end(), [name](const DataView &data_view) { return data_view.name == name; });
    if (it == data_views.end())
    {
        data_views.emplace_back();
        it = data_views.end();
        --it;
        it->name = name;
    }
    return &*it;
}

namespace detail
{

template<typename T>
typename T::DataView *GetDataView(std::list<typename T::DataView> &data_views, const std::string &name)
{
    typename std::list<typename T::DataView>::iterator it =
        std::find_if(data_views.begin(), data_views.end(),
                     [name](const typename T::DataView &data_view) { return data_view.name == name; });
    if (it == data_views.end())
    {
        throw InvalidArgument("name") << "DataView with name '" << name << "' not found";
    }
    return &*it;
}

template<typename T>
const typename T::DataView *GetDataView(const std::list<typename T::DataView> &data_views, const std::string &name)
{
    typename std::list<typename T::DataView>::const_iterator it =
        std::find_if(data_views.cbegin(), data_views.cend(),
                     [name](const typename T::DataView &data_view) { return data_view.name == name; });
    if (it == data_views.end())
    {
        throw InvalidArgument("name") << "DataView with name '" << name << "' not found";
    }
    return &*it;
}

} // namespace detail

template<>
template<>
DataViewInterface::DataIn::DataView *DataViewInterface::DataIn::GetDataView(const std::string &name)
{
    return detail::GetDataView<DataViewInterface::DataIn>(data_views, name);
}

template<>
const DataViewInterface::DataIn::DataView *DataViewInterface::DataIn::GetDataView(const std::string &name) const
{
    return detail::GetDataView<const DataViewInterface::DataIn>(data_views, name);
}

template<>
const DataViewInterface::DataOut::DataView *DataViewInterface::DataOut::GetDataView(const std::string &name) const
{
    return detail::GetDataView<const DataViewInterface::DataOut>(data_views, name);
}

template<>
DataViewInterface::DataOut::DataView::DataView()
{
}

/**
 * Copy a data view interface structure to a data view POD structure.
 */
template<>
DataViewInterface::DataOut DataViewInterface::Get()
{
    AccessGuardConst access(this);

    DataViewInterface::DataOut data_out;

    data_out.data_views.clear();
    for (auto &&data_view_in : access->data_views)
    {
        data_out.data_views.emplace_back();
        DataViewInterface::DataOut::DataView &data_view_out = data_out.data_views.back();

        data_view_out.name               = data_view_in.name;
        data_view_out.zoom_factor        = data_view_in.zoom_factor.Get();
        data_view_out.view_offset        = data_view_in.view_offset;
        data_view_out.pixel_aspect_ratio = data_view_in.pixel_aspect_ratio.Get();
    }

    return data_out;
}

} // namespace clara::viz
