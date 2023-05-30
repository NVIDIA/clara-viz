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

#include "claraviz/util/Log.h"

#include <iomanip>
#include <chrono>

namespace clara::viz
{

/*static*/ LogLevel Log::g_log_level   = LogLevel::Info;
/*static*/ std::mutex Log::g_mutex_;
/*static*/ std::ostream *Log::g_stream = &std::cout;

Log::Log(LogLevel level)
    : log_level_(level)
{
    if (log_level_ >= g_log_level)
    {
        preamble_ << ("[");
        preamble_ << (log_level_);
        preamble_ << ("] ");

        const auto now      = std::chrono::system_clock::now();
        const auto ms       = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm          = *std::localtime(&t);

        preamble_ << std::put_time(&tm, "%F %T") << '.' << std::setfill('0') << std::setw(3) << ms.count() << " ";
    }
}

Log::~Log()
{
    if (log_level_ >= g_log_level)
    {
        std::unique_lock<std::mutex> lock(g_mutex_);
        (*g_stream) << preamble_.str() << msg_.str() << std::endl;
    }
}

Log &Log::operator<<(std::ostream &(*func)(std::ostream &))
{
    if (log_level_ >= g_log_level)
    {
        func(msg_);
    }
    return *this;
}

std::ostream &operator<<(std::ostream &os, const LogLevel &logLevel)
{
    switch (logLevel)
    {
    case LogLevel::Debug:
        os << std::string("DEBUG");
        break;
    case LogLevel::Info:
        os << std::string("INFO ");
        break;
    case LogLevel::Warning:
        os << std::string("WARN ");
        break;
    case LogLevel::Error:
        os << std::string("ERROR");
        break;
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

} // namespace clara::viz
