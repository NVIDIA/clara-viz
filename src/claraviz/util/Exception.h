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

#include <sstream>
#include <stdexcept>

namespace clara::viz
{

/**
 * Templated exception class to extend std exceptions.
 * 
 * Supports building a message using '<<' operators, e.g.
 * @code{.cpp}
 * throw Exception() << Object " << object << "unexpected";
 * @endcode
 * 
 * @todo Very similar to Log class, define common string stream class? Or inherit
 *       from std::ostream?
 * @todo Would like to use std::ostringstream but this returns a temporary string and what()
 *       is const and does not allow allocations. Used a std::string but exceptions should
 *       not allocate. Have no better solution for now.
 */
template<class StdExceptionClass>
class Exception : public StdExceptionClass
{
public:
    /**
     * Construct
     * 
     * @param filename [in] name of the file where the exception had been thrown
     * @param line [in] line number where the exception had been thrown
     */
    template<typename U = StdExceptionClass,
             typename std::enable_if<!std::is_same<U, std::invalid_argument>::value &&
                                     !std::is_same<U, std::runtime_error>::value>::type...>
    Exception(const char *filename, uint32_t line)
    {
        operator<<("(");
        operator<<(filename);
        operator<<(":");
        operator<<(line);
        operator<<(") ");
    }

    /**
     * Construct with message
     * 
     * @param filename [in] name of the file where the exception had been thrown
     * @param line [in] line number where the exception had been thrown
     * @param which [in] message to output
     */
    template<typename U = StdExceptionClass,
             typename std::enable_if<std::is_same<U, std::invalid_argument>::value ||
                                     std::is_same<U, std::runtime_error>::value>::type...>
    Exception(const char *filename, uint32_t line, const char *which)
        : StdExceptionClass("")
    {
        operator<<("(");
        operator<<(filename);
        operator<<(":");
        operator<<(line);
        operator<<(") '");
        operator<<(which);
        operator<<("' ");
    }
    virtual ~Exception() {}

    /**
     * Operator template that adds content to the stream.
     */
    template<class Type>
    Exception &operator<<(const Type &msg)
    {
        what_ += static_cast<std::ostringstream &>(std::ostringstream().flush() << msg).str();
        return *this;
    }

    /**
     * Operator that handles std::ostream specific IO manipulators such as
     * std::endl.
     */
    Exception &operator<<(std::ostream &(*func)(std::ostream &))
    {
        std::ostringstream tmp;
        func(tmp);
        what_ += tmp.str();
        return *this;
    }

    /** @name std::exception methods */
    /**@{*/
    virtual const char *what() const noexcept
    {
        return what_.c_str();
    }
    /**@}*/

private:
    std::string what_; //!< the message
};

#define BadAlloc() clara::viz::Exception<std::bad_alloc>(__FILE__, __LINE__)
#define InvalidArgument(WHICH) clara::viz::Exception<std::invalid_argument>(__FILE__, __LINE__, WHICH)
#define Unimplemented() clara::viz::Exception<std::exception>(__FILE__, __LINE__) << "Unimplemented "
#define InvalidState() clara::viz::Exception<std::exception>(__FILE__, __LINE__) << "Invalid state "
#define RuntimeError() clara::viz::Exception<std::runtime_error>(__FILE__, __LINE__, "") << "Runtime error "

} // namespace clara::viz
