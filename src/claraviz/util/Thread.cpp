/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/Thread.h"

#include <pthread.h>
#include <cstring>

#include "claraviz/util/Exception.h"
#include "claraviz/util/Log.h"

namespace clara::viz
{

namespace
{

void SetThreadName(std::thread::native_handle_type handle, const std::string &name)
{
    // pthread thread names are limited to 16 chars, including null char
    if (name.size() > 15)
    {
        Log(LogLevel::Warning) << "Thread name " << name << " exceeds pthread name length, will"
                               << " be truncated.";
    }

    const int err = pthread_setname_np(handle, name.substr(0, 15).c_str());
    if (err)
    {
        Log(LogLevel::Warning) << "Could not set thread name for " << name << ": " << std::strerror(err);
    }
}

} // anonymous namespace

Thread::Thread(const std::string &name, std::function<void(std::function<void()> ready)> thread_func)
    : thread_func_(thread_func)
{
    if (!thread_func_)
    {
        throw InvalidArgument("thread_func_");
    }

    thread_ = std::thread([this] { ThreadStartupFunc(); });

    // wait for the thread to be ready
    std::future<bool> future = thread_ready_.get_future();
    bool thread_result       = false;
    try
    {
        if (future.wait_for(std::chrono::seconds(60)) != std::future_status::ready)
        {
            // the thread failed to start
            throw RuntimeError() << "Thread failed to start";
        }

        thread_result = future.get();
    }
    catch (...)
    {
        // wait for the thread to finish
        if (thread_.joinable())
        {
            thread_.join();
        }

        // rethrow
        throw;
    }
    if (thread_result == false)
    {
        // the thread started but returned false
        throw RuntimeError() << "Thread function failed";
    }

    SetThreadName(thread_.native_handle(), name);
}

Thread::~Thread()
{
    // check if the thread responded to the exit request, if not the thread hangs
    std::future<bool> future = thread_result_.get_future();
    if (future.wait_for(std::chrono::seconds(60)) != std::future_status::ready)
    {
        Log(LogLevel::Error) << "The thread is not responding";
        // This is bad, if an joinable std::thread object is destroyed the program is
        // terminated and we don't want this to happen.
        // Therefore separate the thread of execution from the thread object. Any allocated
        // resources will be freed once the thread exits.
        thread_.detach();
    }
    else
    {
        // wait for the thread to finish
        if (thread_.joinable())
        {
            thread_.join();
        }

        // get the thread result and the exception if the thread threw one
        bool threadResult = false;
        try
        {
            threadResult = future.get();
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "The thread threw the exception: " << e.what();
        }

        if (threadResult == false)
        {
            Log(LogLevel::Error) << "The thread failed and returned false";
        }
    }
}

void Thread::ThreadStartupFunc()
{
    const char *error = nullptr;

    try
    {
        // call the thread function
        thread_func_([this] { thread_ready_.set_value(true); });

        thread_result_.set_value(true);
    }

    // pass the exception to the main thread
    catch (const std::exception &e)
    {
        error = e.what();
        thread_result_.set_exception(std::current_exception());
    }
    catch (...)
    {
        error = "unknown exception";
        thread_result_.set_exception(std::current_exception());
    }

    if (error)
    {
        Log(LogLevel::Error) << "The thread failed and returned " << error;
    }
}

} // namespace clara::viz
