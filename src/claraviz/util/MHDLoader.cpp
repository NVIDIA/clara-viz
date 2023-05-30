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

#include "MHDLoader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

#include <zlib.h>

#include <claraviz/util/Exception.h>

namespace clara::viz
{

std::ostream &operator<<(std::ostream &os, const MHDLoader::ElementType &element_type)
{
    switch (element_type)
    {
    case MHDLoader::ElementType::INVALID:
        os << std::string("INVALID");
        break;
    case MHDLoader::ElementType::INT8:
        os << std::string("INT8");
        break;
    case MHDLoader::ElementType::UINT8:
        os << std::string("UINT8");
        break;
    case MHDLoader::ElementType::INT16:
        os << std::string("INT16");
        break;
    case MHDLoader::ElementType::UINT16:
        os << std::string("UINT16");
        break;
    case MHDLoader::ElementType::INT32:
        os << std::string("INT32");
        break;
    case MHDLoader::ElementType::UINT32:
        os << std::string("UINT32");
        break;
    case MHDLoader::ElementType::FLOAT:
        os << std::string("FLOAT");
        break;
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

class MHDLoader::Impl
{
public:
    Impl()
        : element_type_(ElementType::INVALID)
        , size_(0)
    {
    }

    ElementType element_type_;
    Vector3ui size_;
    Vector3f element_spacing_;
    std::shared_ptr<IBlob> blob_;
};

MHDLoader::MHDLoader()
    : impl_(new Impl)
{
}

MHDLoader::~MHDLoader() {}

void MHDLoader::Load(const std::string &file_name)
{
    bool compressed = false;
    std::string data_file_name;

    {
        std::stringstream meta_header;
        {
            std::ifstream file;
            file.open(file_name, std::ios::in);
            if (!file.is_open())
            {
                throw RuntimeError() << "Could not open " << file_name;
            }
            meta_header << file.rdbuf();
        }
        // get the parameters
        std::string parameter;
        while (std::getline(meta_header, parameter, '='))
        {
            // remove spaces
            parameter.erase(
                std::remove_if(parameter.begin(), parameter.end(), [](unsigned char x) { return std::isspace(x); }),
                parameter.end());

            // get the value
            std::string value;
            std::getline(meta_header, value);
            // remove leading spaces
            auto it = value.begin();
            while ((it != value.end()) && (std::isspace(*it)))
            {
                it = value.erase(it);
            }

            if (parameter == "NDims")
            {
                int dims = std::stoi(value);
                if (dims != 3)
                {
                    throw RuntimeError() << "Expected a three dimensional input, instead NDims is " << dims;
                }
            }
            else if (parameter == "CompressedData")
            {
                if (value == "True")
                {
                    compressed = true;
                }
                else if (value == "False")
                {
                    compressed = false;
                }
                else
                {
                    throw RuntimeError() << "Unexpected value for " << parameter << ": " << value;
                }
            }
            else if (parameter == "DimSize")
            {
                std::stringstream value_stream(value);
                std::string value;
                for (int index = 0; std::getline(value_stream, value, ' '); ++index)
                {
                    impl_->size_(index) = std::stoi(value);
                }
            }
            else if (parameter == "ElementSpacing")
            {
                std::stringstream value_stream(value);
                std::string value;
                for (int index = 0; std::getline(value_stream, value, ' '); ++index)
                {
                    impl_->element_spacing_(index) = std::stof(value);
                }
            }
            else if (parameter == "ElementType")
            {
                if (value == "MET_CHAR")
                {
                    impl_->element_type_ = ElementType::INT8;
                }
                else if (value == "MET_UCHAR")
                {
                    impl_->element_type_ = ElementType::UINT8;
                }
                else if (value == "MET_SHORT")
                {
                    impl_->element_type_ = ElementType::INT16;
                }
                else if (value == "MET_USHORT")
                {
                    impl_->element_type_ = ElementType::UINT16;
                }
                else if (value == "MET_INT")
                {
                    impl_->element_type_ = ElementType::INT32;
                }
                else if (value == "MET_UINT")
                {
                    impl_->element_type_ = ElementType::UINT32;
                }
                else if (value == "MET_FLOAT")
                {
                    impl_->element_type_ = ElementType::FLOAT;
                }
                else
                {
                    throw RuntimeError() << "Unexpected value for " << parameter << ": " << value;
                }
            }
            else if (parameter == "ElementDataFile")
            {
                const std::string path = file_name.substr(0, file_name.find_last_of("/\\") + 1);
                data_file_name         = path + value;
            }
        }
    }

    const size_t data_size = impl_->size_(0) * impl_->size_(1) * impl_->size_(2) * GetBytesPerElement();
    std::unique_ptr<std::vector<uint8_t>> data = std::make_unique<std::vector<uint8_t>>(data_size);

    std::ifstream file;

    file.open(data_file_name, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw RuntimeError() << "Could not open " << data_file_name;
    }
    const std::streampos file_size = file.tellg();
    file.seekg(0, std::ios_base::beg);

    if (compressed)
    {
        // need to uncompress, first read to 'compressed_data' vector and then uncompress to 'data'
        std::vector<uint8_t> compressed_data(file_size);

        // read
        file.read(reinterpret_cast<char *>(compressed_data.data()), compressed_data.size());

        // uncompress
        z_stream strm{};
        int result = inflateInit2(&strm, 32 + MAX_WBITS);
        if (result != Z_OK)
        {
            throw RuntimeError() << "Failed to uncompress " << data_file_name
                                 << ", inflateInit2 failed with error code " << result;
        }

        strm.next_in   = compressed_data.data();
        strm.avail_in  = compressed_data.size();
        strm.next_out  = data->data();
        strm.avail_out = data->size();

        result = inflate(&strm, Z_FINISH);
        inflateEnd(&strm);
        if (result != Z_STREAM_END)
        {
            throw RuntimeError() << "Failed to uncompress " << data_file_name << ", inflate failed with error code "
                                 << result;
        }
    }
    else
    {
        file.read(reinterpret_cast<char *>(data->data()), data->size());
    }

    impl_->blob_.reset(new StdContainerBlob<std::vector<uint8_t>>(std::move(data)));
}

Vector3ui MHDLoader::GetSize() const
{
    return impl_->size_;
}

MHDLoader::ElementType MHDLoader::GetElementType() const
{
    return impl_->element_type_;
}

uint32_t MHDLoader::GetBytesPerElement() const
{
    switch (impl_->element_type_)
    {
    case ElementType::INT8:
        return sizeof(int8_t);
    case ElementType::UINT8:
        return sizeof(uint8_t);
    case ElementType::INT16:
        return sizeof(int16_t);
    case ElementType::UINT16:
        return sizeof(uint16_t);
    case ElementType::INT32:
        return sizeof(int32_t);
    case ElementType::UINT32:
        return sizeof(uint32_t);
    case ElementType::FLOAT:
        return sizeof(float);
    default:
        break;
    }
    throw InvalidState() << "Unhandled element type " << impl_->element_type_;
}

Vector3f MHDLoader::GetElementSpacing() const
{
    return impl_->element_spacing_;
}

std::shared_ptr<IBlob> MHDLoader::GetData() const
{
    return impl_->blob_;
}

} // namespace clara::viz
