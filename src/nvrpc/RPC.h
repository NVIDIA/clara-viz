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

#include "nvrpc/Context.h"

namespace nvrpc
{

template<class ContextType, class ServiceType>
class AsyncRPC : public IRPC
{
public:
    using ContextType_t         = ContextType;
    using ServiceQueueFuncType  = typename ContextType::LifeCycleType::ServiceQueueFuncType;
    using ExecutorQueueFuncType = typename ContextType::LifeCycleType::ExecutorQueueFuncType;

    AsyncRPC(ServiceQueueFuncType);
    ~AsyncRPC() override {}

protected:
    std::unique_ptr<IContext> CreateContext(::grpc::ServerCompletionQueue *, std::shared_ptr<Resources>) final override;

private:
    ServiceQueueFuncType m_RequestFunc;
};

template<class ContextType, class ServiceType>
AsyncRPC<ContextType, ServiceType>::AsyncRPC(ServiceQueueFuncType req_fn)
    : m_RequestFunc(req_fn)
{
}

template<class ContextType, class ServiceType>
std::unique_ptr<IContext> AsyncRPC<ContextType, ServiceType>::CreateContext(::grpc::ServerCompletionQueue *cq,
                                                                            std::shared_ptr<nvrpc::Resources> r)
{
    auto ctx_resources = std::dynamic_pointer_cast<typename ContextType::ResourcesType::element_type>(r);
    if (!ctx_resources)
    {
        throw std::runtime_error("Incompatible Resource object");
    }
    auto q_fn                     = ContextType::LifeCycleType::BindExecutorQueueFunc(m_RequestFunc, cq);
    std::unique_ptr<IContext> ctx = ContextFactory<ContextType>(q_fn, ctx_resources);
    return ctx;
}

} // end namespace nvrpc
