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

#include "claraviz/rpc/ServerRPC.h"

#include <functional>
#include <future>
#include <limits>
#include <thread>

#include "claraviz/util/Log.h"

#include <nvrpc/Server.h>

namespace clara::viz
{

/**
 * Server implementation data
 */
struct ServerRPC::Impl
{
    Impl()
        : ready_state(State::INVALID)
    {
    }

    State ready_state;

    std::unique_ptr<nvrpc::Server> grpc_server;

    std::list<std::shared_ptr<nvrpc::Resources>> resources;

    std::promise<void> is_shutdown;

    nvrpc::IExecutor *default_executor;
};

ServerRPC::ServerRPC(const std::string &address)
    : impl_(new Impl)
{
    // create the gRPC server
    Log(LogLevel::Info) << "Creating gPRC service on " << address;
    impl_->grpc_server = std::make_unique<nvrpc::Server>(address);

    // set the message size to the maximum, 3D data packages might be big
    impl_->grpc_server->GetBuilder().SetMaxMessageSize(std::numeric_limits<int32_t>::max());

    // create the default executor
    impl_->default_executor = impl_->grpc_server->CreateExecutor();

    impl_->ready_state = State::INITIALIZED;
}

ServerRPC::~ServerRPC()
{
    try
    {
        Shutdown();
    }
    catch (...)
    {
        Log(LogLevel::Error) << "Shutdown failed";
    }
}

nvrpc::IService *ServerRPC::RegisterService(std::unique_ptr<nvrpc::IService> &service)
{
    return impl_->grpc_server->RegisterAsyncService(service);
}

void ServerRPC::Run()
{
    if (impl_->ready_state != State::INITIALIZED)
    {
        throw std::runtime_error("Server not in INITIALIZED state, cannot start");
    }

    // start the gRPC service
    Log(LogLevel::Info) << "Starting gPRC service";
    impl_->grpc_server->AsyncRun();

    impl_->ready_state = State::READY;
}

bool ServerRPC::Shutdown()
{
    if (impl_->ready_state == State::READY)
    {
        Log(LogLevel::Info) << "Shutting down gPRC service";
        impl_->ready_state = State::EXITING;

        impl_->grpc_server->Shutdown();

        impl_->ready_state = State::SHUTDOWN;

        impl_->is_shutdown.set_value();
    }

    return true;
}

void ServerRPC::Wait()
{
    // if the server is ready wait until it is shut down
    if (impl_->ready_state == State::READY)
    {
        std::future<void> future = impl_->is_shutdown.get_future();
        future.wait();
    }
    else
    {
        Log(LogLevel::Error) << "Server is not running, can't wait";
    }
}

ServerRPC::State ServerRPC::ReadyState() const
{
    return impl_->ready_state;
}

nvrpc::IExecutor *ServerRPC::CreateExecutor(int numThreads)
{
    return impl_->grpc_server->CreateExecutor(numThreads);
}

void ServerRPC::RegisterRPC(nvrpc::IRPC *rpc, std::shared_ptr<nvrpc::Resources> &resource, nvrpc::IExecutor *executor)
{
    if (!executor)
    {
        executor = impl_->default_executor;
    }

    executor->RegisterContexts(rpc, resource, 1);
    impl_->resources.emplace_back(std::move(resource));
}
} // namespace clara::viz
