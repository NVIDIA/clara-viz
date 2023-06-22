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

#include <typeinfo>

#include <grpcpp/grpcpp.h>
// for MessageToJsonString
#include <google/protobuf/util/json_util.h>

#include <claraviz/util/Log.h>

#include "nvrpc/Resources.h"

namespace nvrpc
{

class IContext;
class IExecutor;
class IContextLifeCycle;
class IRPC;
class IService;

/**
 * The IContext object and it's subsequent derivations are the single more
 * important class in this library. Contexts are responsible for maintaining the
 * state of a message and performing the custom code for an RPC invocation.
 */
class IContext
{
public:
    virtual ~IContext() {}

    static IContext *Detag(void *tag)
    {
        return static_cast<IContext *>(tag);
    }

    virtual void Shutdown() = 0;

protected:
    IContext() = default;
    void *Tag()
    {
        return reinterpret_cast<void *>(this);
    }

private:
    virtual bool RunNextState(bool ok) = 0;
    virtual void Reset()               = 0;

    friend class IRPC;
    friend class IExecutor;
};

class IContextLifeCycle : public IContext
{
public:
    ~IContextLifeCycle() override {}

protected:
    IContextLifeCycle() = default;

    virtual void OnLifeCycleStart() = 0;
    virtual void OnLifeCycleReset() = 0;

    virtual void FinishResponse() = 0;
    virtual void CancelResponse() = 0;

    /**
     * Dump a gRPC message
     *
     * @param
     */
    template<class Message>
    static void Dump(const Message &message)
    {
        // dump message when loglevel is debug
        if (clara::viz::Log::g_log_level == clara::viz::LogLevel::Debug)
        {
            google::protobuf::util::JsonOptions options{};
            options.add_whitespace                = true;
            options.always_print_primitive_fields = true;

            std::string message_string;
            MessageToJsonString(message, &message_string, options);
            // limit long messages (e.g. when the including large data fields)
            constexpr size_t MAX_MESSAGE_LENGTH = 1024;
            if (message_string.length() > MAX_MESSAGE_LENGTH)
            {
                message_string = message_string.substr(0, MAX_MESSAGE_LENGTH);
                message_string += "\n**** Message exceeded max length, truncated ****\n";
            }
            clara::viz::Log(clara::viz::LogLevel::Debug) << "RPC " << typeid(Message).name() << std::endl << message_string;
        }
    }
};

class IService
{
public:
    IService() = default;
    virtual ~IService() {}

    virtual void Initialize(::grpc::ServerBuilder &) = 0;
};

class IRPC
{
public:
    IRPC() = default;
    virtual ~IRPC() {}

protected:
    virtual std::unique_ptr<IContext> CreateContext(::grpc::ServerCompletionQueue *, std::shared_ptr<Resources>) = 0;

    friend class IExecutor;
};

class IExecutor
{
public:
    IExecutor() = default;
    virtual ~IExecutor() {}

    virtual void Initialize(::grpc::ServerBuilder &)                                                         = 0;
    virtual void Run()                                                                                       = 0;
    virtual void Shutdown()                                                                                  = 0;
    virtual void RegisterContexts(IRPC *rpc, std::shared_ptr<Resources> resources, int numContextsPerThread) = 0;

protected:
    inline bool RunContext(IContext *ctx, bool ok)
    {
        return ctx->RunNextState(ok);
    }
    inline void ResetContext(IContext *ctx)
    {
        ctx->Reset();
    }
    inline std::unique_ptr<IContext> CreateContext(IRPC *rpc, ::grpc::ServerCompletionQueue *cq,
                                                   std::shared_ptr<Resources> res)
    {
        return rpc->CreateContext(cq, res);
    }
};

} // end namespace nvrpc
