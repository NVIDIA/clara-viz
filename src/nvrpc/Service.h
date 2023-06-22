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

#include "nvrpc/Interfaces.h"
#include "nvrpc/RPC.h"

namespace nvrpc
{

template<class ServiceType>
class AsyncService : public IService
{
public:
    using ServiceType_t = ServiceType;

    AsyncService()
        : IService()
        , m_Service(std::make_unique<ServiceType>())
    {
    }
    ~AsyncService() override {}

    void Initialize(::grpc::ServerBuilder &builder) final override
    {
        builder.RegisterService(m_Service.get());
    }

    template<typename ContextType, typename RequestFuncType>
    IRPC *RegisterRPC(RequestFuncType req_fn)
    {
        auto q_fn = ContextType::LifeCycleType::BindServiceQueueFunc(req_fn, m_Service.get());
        auto rpc  = new AsyncRPC<ContextType, ServiceType>(q_fn);
        auto base = static_cast<IRPC *>(rpc);
        m_RPCs.emplace_back(base);
        return base;
    }

private:
    std::unique_ptr<ServiceType> m_Service;
    std::vector<std::unique_ptr<IRPC>> m_RPCs;
};

} // end namespace nvrpc
