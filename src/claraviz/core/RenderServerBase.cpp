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

#include "claraviz/core/RenderServerBase.h"

#include <signal.h>

#include <thread>

#include "claraviz/util/Log.h"

namespace clara::viz
{

namespace
{

/**
 * Called by the signal handler to shutdown the server
 */
std::function<void()> shutdown_server;

void SignalHandler(int signum)
{
    // Don't need a mutex here since signals should be disabled while in
    // the handler.
    Log(LogLevel::Info) << "Interrupt signal (" << signum << ") received.";

    static std::once_flag flag;
    std::call_once(flag, [] {
        std::unique_ptr<std::thread> thread;

        thread.reset(new std::thread([] { shutdown_server(); }));

        thread->detach();
    });
}

} // anonymous namespace

RenderServerBase::RenderServerBase(const std::string &address)
    : server_rpc_(new ServerRPC(address))
{
    // setup the global shutdown function used by the signal handler to
    // shutdown the server
    shutdown_server = [this] {
        if (!Shutdown())
        {
            Log(LogLevel::Warning) << "Failed to shutdown render server";
        }
    };

    // Trap SIGINT and SIGTERM to allow server to exit gracefully
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);
}

const std::shared_ptr<ServerRPC> &RenderServerBase::GetServerRPC()
{
    return server_rpc_;
}

void RenderServerBase::Run()
{
    // run the RPC server
    server_rpc_->Run();
}

void RenderServerBase::Wait()
{
    // wait for the RPC server
    server_rpc_->Wait();
}

bool RenderServerBase::Shutdown()
{
    // shutdown the RPC server
    return server_rpc_->Shutdown();
}

} // namespace clara::viz
