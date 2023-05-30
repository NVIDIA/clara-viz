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

#pragma once

#include <future>
#include <thread>

#include "claraviz/util/Types.h"

namespace clara::viz
{

/**
 * A thread class.
 */
class Thread : public NonCopyable
{
public:
    /**
     * Construct a Thread instance, the thread is started immediately. The function to be called
     * can be a lambda, e.g. '[this](std::function<void()> ready) { memberFunc(ready); }' for a member
     * of the current class or '[&obj](std::function<void()> ready) { obj->memberFunc(ready); }' for
     * members of an object.
     * The thread function needs to call the supplied 'ready()' function when the thread is ready.
     *
     * @param name [in] the name of the thread
     * @param thread_func [in] the function to call
     */
    Thread(const std::string &name, std::function<void(std::function<void()> ready)> thread_func);
    Thread() = delete;
    ~Thread();

private:
    /**
     * Thread startup function
     */
    void ThreadStartupFunc();

    std::function<void(std::function<void()> ready)> thread_func_; ///< the function to call
    std::promise<bool> thread_ready_;                              ///< set by the worker thread if ready
    std::promise<bool> thread_result_;                             ///< worker thread result, also transports exceptions
    std::thread thread_;                                           ///< the worker thread
};

} // namespace clara::viz
