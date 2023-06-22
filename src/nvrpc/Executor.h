/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>

#include "nvrpc/Interfaces.h"

namespace nvrpc
{

class Resources;
class ThreadPool;

class Executor : public IExecutor
{
public:
    Executor();
    Executor(int numThreads);
    Executor(std::unique_ptr<ThreadPool> threadpool);
    ~Executor() override;

    void Initialize(::grpc::ServerBuilder &builder) final override;

    void RegisterContexts(IRPC *rpc, std::shared_ptr<Resources> resources, int numContextsPerThread) final override;

    void Run() final override;
    void Shutdown() final override;

private:
    void ProgressEngine(int thread_id);

    std::unique_ptr<ThreadPool> m_ThreadPool;
    std::vector<std::unique_ptr<IContext>> m_Contexts;
    std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>> m_ServerCompletionQueues;
};

} // end namespace nvrpc
