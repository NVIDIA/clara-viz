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

#include "nvrpc/Executor.h"

#include "nvrpc/Resources.h"
#include "nvrpc/ThreadPool.h"

namespace nvrpc
{

Executor::Executor()
    : Executor(2)
{
}

Executor::Executor(int numThreads)
    : Executor(std::make_unique<ThreadPool>(numThreads))
{
}

Executor::Executor(std::unique_ptr<ThreadPool> threadpool)
    : IExecutor()
    , m_ThreadPool(std::move(threadpool))
{
}

Executor::~Executor() {}

void Executor::Initialize(::grpc::ServerBuilder &builder)
{
    for (int i = 0; i < m_ThreadPool->Size(); i++)
    {
        m_ServerCompletionQueues.emplace_back(builder.AddCompletionQueue());
    }
}

void Executor::RegisterContexts(IRPC *rpc, std::shared_ptr<Resources> resources, int numContextsPerThread)
{
    // CHECK_EQ(m_ThreadPool->Size(), m_ServerCompletionQueues.size()) <<
    // "Incorrect number of CQs";
    for (int i = 0; i < m_ThreadPool->Size(); i++)
    {
        auto cq = m_ServerCompletionQueues[i].get();
        for (int j = 0; j < numContextsPerThread; j++)
        {
            m_Contexts.emplace_back(this->CreateContext(rpc, cq, resources));
        }
    }
}

void Executor::Run()
{
    // Launch the threads polling on their CQs
    for (int i = 0; i < m_ThreadPool->Size(); i++)
    {
        m_ThreadPool->enqueue([this, i] { ProgressEngine(i); });
    }
    // Queue the Execution Contexts in the receive queue
    for (size_t i = 0; i < m_Contexts.size(); i++)
    {
        // Reseting the context decrements the gauge
        ResetContext(m_Contexts[i].get());
    }
}

void Executor::Shutdown()
{
    for (auto &context : m_Contexts)
    {
        context->Shutdown();
    }
    for (auto &cq : m_ServerCompletionQueues)
    {
        cq->Shutdown();
    }

    // wait for all threads to finish
    m_ThreadPool.reset();
}

void Executor::ProgressEngine(int thread_id)
{
    bool ok;
    void *tag;
    auto myCQ = m_ServerCompletionQueues[thread_id].get();

    while (true)
    {
        auto status = myCQ->Next(&tag, &ok);
        if (!status)
        {
            return; // Shutdown
        }
        auto ctx = IContext::Detag(tag);
        if (!RunContext(ctx, ok))
        {
            ResetContext(ctx);
        }
    }
}

} // namespace nvrpc
