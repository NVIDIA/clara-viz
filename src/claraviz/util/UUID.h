/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <string>
#include <iostream>

#include <uuid/uuid.h>

namespace clara::viz
{

/**
 * A 128-bit Universally Unique Identifier.
 */
class UUID
{
public:
    /**
     * Constructs a new, randomly generated UUID.
     */
    UUID();

    /**
     * Constructs a new UUID from a string representation.
     *
     * @param str [in] The string representation of the UUID.
     */
    UUID(const std::string &str);

    /**
     * Returns a string representation of the UUID.
     */
    std::string toString() const;

    /**
     * @returns true if equal
     */
    bool operator==(const UUID &r) const;

    /**
     * @returns true if not equal
     */
    bool operator!=(const UUID &r) const;

    /**
     * @returns true if less
     */
    bool operator<(const UUID &r) const;

    /**
     * @returns true if greater
     */
    bool operator>(const UUID &r) const;

private:
    uuid_t _uuid;
};

/**
 * Outputs a UUID string to an ostream.
 */
std::ostream &operator<<(std::ostream &s, const UUID &id);

} // namespace clara::viz
