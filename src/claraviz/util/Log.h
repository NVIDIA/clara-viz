/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <mutex>
#include <sstream>

namespace clara::viz
{

/**
 * Log level
 */
enum class LogLevel
{
    Debug,
    Info,
    Warning,
    Error
};

/**
 * Operator that appends string representation of a log level to a stream.
 */
std::ostream &operator<<(std::ostream &os, const LogLevel &logLevel);

/**
 * Logging class.
 * Supports building a message using '<<' operators, e.g. 'Log(LogLevel::Debug) <<
 * "Object " << object << " created";'
 * @todo Very similar to Exception class, define common string stream class? Or
 *       inherit from std::ostream?
 *
 * Example output
 *  [DEBUG] 2017-03-25 19:22:51 file.cpp:123 Object foo created
 */
class Log
{
public:
    /**
     * Construct
     *
     * @param level [in] log level
     */
    Log(LogLevel level);
    ~Log();

    /**
     * Operator template that adds content to the stream.
     */
    template<class T>
    Log &operator<<(const T &msg)
    {
        if (log_level_ >= g_log_level)
        {
            msg_ << msg;
        }
        return *this;
    }

    /**
     * Operator that handles std::ostream specific IO manipulators such as
     * std::endl
     */
    Log &operator<<(std::ostream &(*func)(std::ostream &));

    /**
     * Set the log stream object
     *
     * @param stream stream object
     *
     * @returns previous stream object
     */
    static std::ostream* SetLogStream(std::ostream *stream)
    {
        std::unique_lock<std::mutex> lock(g_mutex_);
        std::ostream *prev = g_stream;
        g_stream = stream;
        return prev;
    }

    static LogLevel g_log_level; ///< Current log level

private:
    Log() = delete;

    static std::mutex g_mutex_;
    static std::ostream *g_stream; ///< Log stream

    LogLevel log_level_;
    /// @todo this is using dynamic allocations, need to move this to static allocation
    std::ostringstream preamble_; ///< Log level & timestamp buffer; flushed to g_stream in dtor
    std::ostringstream msg_;      ///< Log message buffer; flushed to g_stream in dtor
};

} // namespace clara::viz
