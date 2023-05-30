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
#include "nvrpc/LifeCycleUnary.h"
#include "nvrpc/LifeCycleServerStreaming.h"

namespace nvrpc
{

template<class LifeCycle, class Resources>
class BaseContext;

template<class Request, class Response, class Resources>
using ContextUnary = BaseContext<LifeCycleUnary<Request, Response>, Resources>;

template<class Request, class Response, class Resources>
using ContextServerStreaming = BaseContext<LifeCycleServerStreaming<Request, Response>, Resources>;

template<class LifeCycle, class Resources>
class BaseContext : public LifeCycle
{
public:
    using RequestType   = typename LifeCycle::RequestType;
    using ResponseType  = typename LifeCycle::ResponseType;
    using ResourcesType = std::shared_ptr<Resources>;
    using QueueFuncType = typename LifeCycle::ExecutorQueueFuncType;
    using LifeCycleType = LifeCycle;

    virtual ~BaseContext() override {}

protected:
    const ResourcesType &GetResources() const
    {
        return m_Resources;
    }
    double Walltime() const;

    virtual void OnContextStart();
    virtual void OnContextReset();

private:
    virtual void OnLifeCycleStart() final override;
    virtual void OnLifeCycleReset() final override;

    ResourcesType m_Resources;
    std::chrono::high_resolution_clock::time_point m_StartTime;

    void FactoryInitializer(QueueFuncType, ResourcesType);

    // Factory function allowed to create unique pointers to context objects
    template<class ContextType>
    friend std::unique_ptr<ContextType> ContextFactory(typename ContextType::QueueFuncType q_fn,
                                                       typename ContextType::ResourcesType resources);

public:
    // Convenience method to acquire the Context base pointer from a derived class
    BaseContext<LifeCycle, Resources> *GetBase()
    {
        return dynamic_cast<BaseContext<LifeCycle, Resources> *>(this);
    }
};

// Implementations

/**
 * @brief Method invoked when a request is received and the per-call context
 * lifecycle begins.
 */
template<class LifeCycle, class Resources>
void BaseContext<LifeCycle, Resources>::OnLifeCycleStart()
{
    m_StartTime = std::chrono::high_resolution_clock::now();
    OnContextStart();
}

template<class LifeCycle, class Resources>
void BaseContext<LifeCycle, Resources>::OnContextStart()
{
}

/**
 * @brief Method invoked at the end of the per-call lifecycle just before the
 * context is reset.
 */
template<class LifeCycle, class Resources>
void BaseContext<LifeCycle, Resources>::OnLifeCycleReset()
{
    OnContextReset();
}

template<class LifeCycle, class Resources>
void BaseContext<LifeCycle, Resources>::OnContextReset()
{
}

/**
 * @brief Number of seconds since the start of the RPC
 */
template<class LifeCycle, class Resources>
double BaseContext<LifeCycle, Resources>::Walltime() const
{
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_StartTime).count();
}

/**
 * @brief Used by ContextFactory to initialize the Context
 */
template<class LifeCycle, class Resources>
void BaseContext<LifeCycle, Resources>::FactoryInitializer(QueueFuncType queue_fn, ResourcesType resources)
{
    this->SetQueueFunc(queue_fn);
    m_Resources = resources;
}

/**
 * @brief ContextFactory is the only function in the library allowed to create
 * an IContext object.
 */
template<class ContextType>
std::unique_ptr<ContextType> ContextFactory(typename ContextType::QueueFuncType queue_fn,
                                            typename ContextType::ResourcesType resources)
{
    auto ctx  = std::make_unique<ContextType>();
    auto base = ctx->GetBase();
    base->FactoryInitializer(queue_fn, resources);
    return ctx;
}

} // end namespace nvrpc
