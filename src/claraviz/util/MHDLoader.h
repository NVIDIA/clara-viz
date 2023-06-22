/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <claraviz/util/StdContainerBlob.h>
#include <claraviz/util/VectorT.h>

namespace clara::viz
{

/**
 * Load volume data described by an 'mhd'.
 * See https://itk.org/Wiki/ITK/MetaIO/Documentation.
 */
class MHDLoader
{
public:
    /**
     * Construct
     */
    MHDLoader();
    ~MHDLoader();

    /**
     * Volume element type enum
     */
    enum class ElementType
    {
        INVALID, /// invalid
        INT8,    /// 8 bit signed
        UINT8,   /// 8 bit unsigned
        INT16,   /// 16 bit signed
        UINT16,  /// 16 bit unsigned
        INT32,   /// 32 bit signed
        UINT32,  /// 32 bit unsigned
        FLOAT,   /// 32 bit floating point
    };

    /**
     * Load a 'mhd' file
     *
     * @param file_name [in] name of the file to load
     */
    void Load(const std::string &file_name);

    /**
     * @returns the volume size
     */
    Vector3ui GetSize() const;

    /**
     * @returns the element type
     */
    ElementType GetElementType() const;

    /**
     * @returns the size of an element in bytes
     */
    uint32_t GetBytesPerElement() const;

    /**
     * @returns the element spacing
     */
    Vector3f GetElementSpacing() const;

    /**
     * @returns the volume data
     */
    std::shared_ptr<IBlob> GetData() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Operator that appends string representation of a element type to a stream.
 */
std::ostream &operator<<(std::ostream &os, const MHDLoader::ElementType &element_type);

} // namespace clara::viz
