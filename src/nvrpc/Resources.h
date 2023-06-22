/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>

namespace nvrpc
{

struct Resources : public std::enable_shared_from_this<Resources>
{
    virtual ~Resources() {}

    template<class Target>
    std::shared_ptr<Target> casted_shared_from_this()
    {
        return std::dynamic_pointer_cast<Target>(Resources::shared_from_this());
    }
};

// credit:
// https://stackoverflow.com/questions/16082785/use-of-enable-shared-from-this-with-multiple-inheritance
template<class T>
class InheritableResources : virtual public Resources
{
public:
    std::shared_ptr<T> shared_from_this()
    {
        return std::dynamic_pointer_cast<T>(Resources::shared_from_this());
    }
};

} // namespace nvrpc
