/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/Observed.h"

#include <algorithm>

#include "claraviz/util/Exception.h"

namespace clara::viz
{

Observed::CallbackHandle Observed::RegisterCallback(const CallbackFunction &&callback)
{
    // add to the observer list
    callbacks_.push_front(callback);

    // initial call once to feed with current value
    callback(*this);

    return callbacks_.cbegin();
}

void Observed::UnregisterCallback(const CallbackHandle &callback)
{
    auto it = callbacks_.cbegin();
    while ((it != callbacks_.cend()) && (it != callback))
    {
        ++it;
    }
    if (it == callbacks_.cend())
    {
        throw InvalidState() << "Invalid callback handle";
    }

    callbacks_.erase(callback);
}

void Observed::NotifyObservers() const
{
    // iterate through the registered observers, and trigger callback on observers
    std::for_each(callbacks_.cbegin(), callbacks_.cend(),
                  [this](const CallbackFunction &callback) { callback(*this); });
}

} // namespace clara::viz
