/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/rpc/DataTransformRPC.h"
#include "claraviz/rpc/TypesRPC.h"

namespace clara::viz
{

namespace cinematic_v1 = nvidia::claraviz::cinematic::v1;

namespace detail
{

void DataTransformContext::ExecuteRPC(cinematic_v1::DataTransformRequest &request,
                                      cinematic_v1::DataTransformResponse &response)
{
    DataTransformInterface::AccessGuard access(GetResources()->data_transform_);

    if (request.has_matrix())
    {
        access->matrix = MakeMatrix4x4(request.matrix());
    }
}

} // namespace detail

} // namespace clara::viz
