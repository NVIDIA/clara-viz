/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <list>
#include <functional>

#include "claraviz/util/Types.h"

namespace clara::viz
{

/**
 * Base class for observed classes. Observers register callback's and get called
 * through a callback function.
 */
class Observed : public NonCopyable
{
public:
    virtual ~Observed() = default;

    /**
     * Callback function type.
     */
    using CallbackFunction = std::function<void(const Observed &source)>;

    /**
     * Callback handle type.
     */
    using CallbackHandle = std::list<CallbackFunction>::const_iterator;

    /**
     * Register an callback which is executed when the observed object has changed.
     *
     * @param callback [in] callback to register
     *
     * @returns a handle to the registered callback
     */
    CallbackHandle RegisterCallback(const CallbackFunction &&callback);

    /**
     * Unregister an callback
     *
     * @param callbackHandle [in] handle of the callback to unregister
     */
    void UnregisterCallback(const CallbackHandle &callbackHandle);

protected:
    /**
     * Notify all observers of this object.
     */
    void NotifyObservers() const;

private:
    std::list<CallbackFunction> callbacks_;
};

} // namespace clara::viz
