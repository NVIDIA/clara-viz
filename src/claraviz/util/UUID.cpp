/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/UUID.h"

#include "claraviz/util/Exception.h"

namespace clara::viz
{

UUID::UUID()
{
    uuid_generate(_uuid);
}

UUID::UUID(const std::string &str)
{
    if (uuid_parse(str.c_str(), _uuid) != 0)
    {
        throw InvalidArgument("str") << "Failed to parse UUID: " << str;
    }
}

std::string UUID::toString() const
{
    char str[UUID_STR_LEN + 1];
    uuid_unparse(_uuid, str);
    return std::string(str);
}

bool UUID::operator==(const UUID &r) const
{
    return uuid_compare(_uuid, r._uuid) == 0;
}

bool UUID::operator!=(const UUID &r) const
{
    return uuid_compare(_uuid, r._uuid) != 0;
}

bool UUID::operator<(const UUID &r) const
{
    return uuid_compare(_uuid, r._uuid) < 0;
}

bool UUID::operator>(const UUID &r) const
{
    return uuid_compare(_uuid, r._uuid) > 0;
}

std::ostream &operator<<(std::ostream &s, const UUID &id)
{
    s << id.toString();
    return s;
}

} // namespace clara::viz
