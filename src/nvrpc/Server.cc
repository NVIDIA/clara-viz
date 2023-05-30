/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "nvrpc/Server.h"
#include "nvrpc/Executor.h"

#include <chrono>
#include <thread>

namespace nvrpc
{

Server::Server(std::string server_address)
    : m_ServerAddress(server_address)
    , m_Running(false)
{
    m_Builder.AddListeningPort(m_ServerAddress, ::grpc::InsecureServerCredentials());
}

Server::~Server()
{
    Shutdown();
}

IService *Server::RegisterAsyncService(std::unique_ptr<IService> &service)
{
    if (m_Running)
    {
        throw std::runtime_error("Error: cannot register service on a running server");
    }
    IService *iService = service.get();
    m_Services.push_back(std::move(service));
    iService->Initialize(m_Builder);
    return iService;
}

IExecutor *Server::CreateExecutor(int numThreads)
{
    auto executor        = std::make_unique<Executor>(numThreads);
    IExecutor *iExecutor = static_cast<IExecutor *>(executor.get());
    m_Executors.push_back(std::move(executor));
    iExecutor->Initialize(m_Builder);
    return iExecutor;
}

::grpc::ServerBuilder &Server::GetBuilder()
{
    if (m_Running)
    {
        throw std::runtime_error("Unable to access Builder after the Server is running.");
    }
    return m_Builder;
}

void Server::Run()
{
    Run(std::chrono::milliseconds(5000), [] {});
}

void Server::Run(std::chrono::milliseconds timeout, std::function<void()> control_fn)
{
    AsyncRun();
    for (;;)
    {
        control_fn();
        std::this_thread::sleep_for(timeout);
    }
    // TODO: gracefully shutdown each service and join threads
}

void Server::AsyncRun()
{
    m_Running = true;
    m_Server  = m_Builder.BuildAndStart();
    for (int i = 0; i < m_Executors.size(); i++)
    {
        m_Executors[i]->Run();
    }
}

void Server::Shutdown()
{
    // shutdown the server first
    if (m_Server)
    {
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1000);
        m_Server->Shutdown(deadline);
    }

    // then all executors
    for (auto &&executor : m_Executors)
    {
        executor->Shutdown();
    }

    // and finally all services
    m_Services.clear();

    // This should cause enforce a join on all async Executor threads
    m_Executors.clear();
}

} // end namespace nvrpc
