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

#pragma once

#include "claraviz/util/MatrixT.h"
#include "claraviz/util/VectorT.h"

#include <nvidia/claraviz/core/types.pb.h>

namespace clara::viz
{

/**
 * Construct Vector2f from gRPC type 'Range'
 */
static inline Vector2f MakeVector2f(const nvidia::claraviz::core::Range &from)
{
    return Vector2f(from.min(), from.max());
}

/**
 * Construct Vector2f from gRPC type 'Float2'
 */
static inline Vector2f MakeVector2f(const nvidia::claraviz::core::Float2 &from)
{
    return Vector2f(from.x(), from.y());
}

/**
 * Construct from gRPC type 'Float3'
 */
static inline Vector3f MakeVector3f(const nvidia::claraviz::core::Float3 &from)
{
    return Vector3f(from.x(), from.y(), from.z());
}

/**
 * Construct from gRPC type 'Matrix4x4'
 */
static inline Matrix4x4 MakeMatrix4x4(const nvidia::claraviz::core::Matrix4x4 &from)
{
    return Matrix4x4({{{{from.m00(), from.m01(), from.m02(), from.m03()}},
                       {{from.m10(), from.m11(), from.m12(), from.m13()}},
                       {{from.m20(), from.m21(), from.m22(), from.m23()}},
                       {{from.m30(), from.m31(), from.m32(), from.m33()}}}});
}

} // namespace clara::viz
