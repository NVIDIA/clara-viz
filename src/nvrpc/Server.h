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

#pragma once

#include <chrono>

#include "nvrpc/Service.h"

namespace nvrpc
{

using std::chrono::milliseconds;

class Server
{
public:
    Server(std::string server_address);
    ~Server();

    IService *RegisterAsyncService(std::unique_ptr<IService> &service);

    IExecutor *CreateExecutor(int numThreads = 1);

    void Run();
    void Run(milliseconds timeout, std::function<void()> control_fn);
    void AsyncRun();

    void Shutdown();

    ::grpc::ServerBuilder &GetBuilder();

private:
    bool m_Running;
    std::string m_ServerAddress;
    ::grpc::ServerBuilder m_Builder;
    std::unique_ptr<::grpc::Server> m_Server;
    std::vector<std::unique_ptr<IExecutor>> m_Executors;
    std::vector<std::unique_ptr<IService>> m_Services;
};

} // end namespace nvrpc
