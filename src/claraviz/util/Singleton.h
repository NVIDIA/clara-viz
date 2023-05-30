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

#include "claraviz/util/Types.h"

namespace clara::viz
{

/**
 * Restricts the instantiation of a class to one object.
 */
template<class T>
class Singleton : public NonCopyable
{
public:
    /**
     * @returns the instance
     */
    static T &GetInstance()
    {
        // since C++11 static variables a thread-safe
        static T instance;

        return instance;
    }
};

} // namespace clara::viz
