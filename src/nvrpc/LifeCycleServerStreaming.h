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

#include <mutex>
#include <list>
#include <functional>
#include <condition_variable>

#include "nvrpc/Interfaces.h"

// for SerializationTraits
#include <grpcpp/impl/codegen/proto_utils.h>

#include <claraviz/util/Log.h>

namespace nvrpc
{

template<class Request, class Response>
class LifeCycleServerStreaming : public IContextLifeCycle
{
public:
    using RequestType  = Request;
    using ResponseType = Response;
    using ServiceQueueFuncType =
        std::function<void(::grpc::ServerContext *, RequestType *, ::grpc::ServerAsyncWriter<ResponseType> *,
                           ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>;
    using ExecutorQueueFuncType =
        std::function<void(::grpc::ServerContext *, RequestType *, ::grpc::ServerAsyncWriter<ResponseType> *, void *)>;

    ~LifeCycleServerStreaming() override;

    void Shutdown() final override;

private:
    class Callback;

protected:
    LifeCycleServerStreaming();
    void SetQueueFunc(ExecutorQueueFuncType);

    virtual void OnRequestReceived(const RequestType &request) = 0;

    /**
     * Write a reponse
     *
     * @param response [in] response to write
     */
    bool Write(const ResponseType &response);

    using CallbackHandle = std::shared_ptr<Callback>;

    /**
     * Register a callback function which will be called when the RPC finished.
     *
     * @param callback [in]
     *
     * @returns a callback handle, if this goes out of scope the callback is unregistered automatically
     */
    CallbackHandle RegisterFinishCallback(const std::function<void()> &callback);

    void FinishResponse() final override;
    void CancelResponse() final override;

private:
    // IContext Methods
    bool RunNextState(bool ok) final override;
    void Reset() final override;

    // LifeCycleServerStreaming Specific Methods
    bool StateRequestDone(bool ok);
    bool StateWrite(bool ok);
    bool StateFinishedDone(bool ok);

    using CallbackList = std::list<CallbackHandle>;

    class Callback
    {
    public:
        Callback(CallbackList &list, const std::function<void()> &callback)
            : list_(list)
            , callback_(callback)
            , valid_(true)
        {
        }
        Callback() = delete;

        ~Callback()
        {
            if (valid_)
            {
                list_.remove_if([this](const CallbackHandle &callback) { return callback.get() == this; });
            }
        }

        void Invalidate()
        {
            valid_ = false;
        }

        void operator()()
        {
            callback_();
        }

    private:
        // not copy able
        Callback(const CallbackHandle &) = delete;
        Callback &operator=(const CallbackHandle &) = delete;

        CallbackList &list_;
        const std::function<void()> callback_;
        bool valid_;
    };

    // Function pointers
    ExecutorQueueFuncType m_QueuingFunc;
    bool (LifeCycleServerStreaming<RequestType, ResponseType>::*m_NextState)(bool ok);

    // Variables
    RequestType m_Request;
    std::unique_ptr<::grpc::ServerContext> m_Context;
    std::unique_ptr<::grpc::ServerAsyncWriter<ResponseType>> m_ResponseWriter;

    // shared variables
    std::recursive_mutex m_mutex;                  // protects access to shared variables
    std::condition_variable_any m_write_condition; // condition to wait on for new writes
    bool m_write_waiting;                          // if true waiting on write condition
    bool m_Finish;                                 // if true finish request
    bool m_Cancel;                                 // if true cancel request
    bool m_response_active;                        // if true there is a response in flight
    bool m_is_shutdown;                            // is shut down
    CallbackList m_finish_callbacks;               // callbacks to call on finish
    std::list<ResponseType> m_Responses;           // pending responses

public:
    template<class RequestFuncType, class ServiceType>
    static ServiceQueueFuncType BindServiceQueueFunc(
        /*
     std::function<void(
     ServiceType *, ::grpc::ServerContext *, RequestType *,
     ::grpc::ServerAsyncWriter<ResponseType> *,
     ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
     */
        RequestFuncType request_fn, ServiceType *service_type)
    {
        return std::bind(request_fn, service_type, std::placeholders::_1, // ServerContext*
                         std::placeholders::_2,                           // InputType
                         std::placeholders::_3,                           // AsyncResponseWriter<OutputType>
                         std::placeholders::_4,                           // CQ
                         std::placeholders::_5,                           // ServerCQ
                         std::placeholders::_6                            // Tag
        );
    }

    static ExecutorQueueFuncType BindExecutorQueueFunc(ServiceQueueFuncType service_q_fn,
                                                       ::grpc::ServerCompletionQueue *cq)
    {
        return std::bind(service_q_fn, std::placeholders::_1, // ServerContext*
                         std::placeholders::_2,               // Request *
                         std::placeholders::_3,               // AsyncResponseWriter<Response> *
                         cq, cq, std::placeholders::_4        // Tag
        );
    }
};

// Implementation

template<class Request, class Response>
LifeCycleServerStreaming<Request, Response>::LifeCycleServerStreaming()
    : m_QueuingFunc(nullptr)
    , m_NextState(nullptr)
    , m_write_waiting(false)
    , m_Finish(false)
    , m_Cancel(false)
    , m_response_active(false)
    , m_is_shutdown(false)
{
}

template<class Request, class Response>
LifeCycleServerStreaming<Request, Response>::~LifeCycleServerStreaming()
{
    Shutdown();
    for (auto &&callback : m_finish_callbacks)
    {
        callback->Invalidate();
    }
}

template<class Request, class Response>
void LifeCycleServerStreaming<Request, Response>::Shutdown()
{
    // if there is a pending Request cancel it
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    if (m_write_waiting)
    {
        CancelResponse();
    }
    m_is_shutdown = true;
}

template<class Request, class Response>
bool LifeCycleServerStreaming<Request, Response>::RunNextState(bool ok)
{
    return (this->*m_NextState)(ok);
}

template<class Request, class Response>
void LifeCycleServerStreaming<Request, Response>::Reset()
{
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    // already shutdown, do nothing
    if (m_is_shutdown)
    {
        return;
    }

    OnLifeCycleReset();

    // previous request should be done
    if (m_Context && (m_NextState == &LifeCycleServerStreaming<RequestType, ResponseType>::StateWrite))
    {
        throw std::runtime_error("Can't reset when previous request is not done");
    }

    m_Request.Clear();
    m_write_waiting   = false;
    m_Finish          = false;
    m_Cancel          = false;
    m_response_active = false;
    for (auto &&callback : m_finish_callbacks)
    {
        callback->Invalidate();
    }
    m_finish_callbacks.clear();
    m_Context.reset(new ::grpc::ServerContext);
    m_ResponseWriter.reset(new ::grpc::ServerAsyncWriter<ResponseType>(m_Context.get()));
    m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateRequestDone;
    m_QueuingFunc(m_Context.get(), &m_Request, m_ResponseWriter.get(), IContext::Tag());
}

template<class Request, class Response>
bool LifeCycleServerStreaming<Request, Response>::Write(const Response &response)
{
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    if (m_Finish || m_Cancel)
    {
        clara::viz::Log(clara::viz::LogLevel::Warning) << "RPC " << typeid(Request).name() << " stream cancelled, write is dropped";
        return false;
    }

    m_Responses.emplace_back(response);

    m_write_condition.notify_one();

    return true;
}

template<class Request, class Response>
typename LifeCycleServerStreaming<Request, Response>::CallbackHandle
    LifeCycleServerStreaming<Request, Response>::RegisterFinishCallback(const std::function<void()> &callback)
{
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    std::shared_ptr<Callback> callbackHandle = std::make_shared<Callback>(m_finish_callbacks, callback);
    m_finish_callbacks.push_back(callbackHandle);

    return callbackHandle;
}

template<class Request, class Response>
bool LifeCycleServerStreaming<Request, Response>::StateRequestDone(bool ok)
{
    if (!ok)
    {
        // the server has been Shutdown before this particular call got matched to an incoming RPC
        return false;
    }

    OnLifeCycleStart();

    ::grpc::Status status = ::grpc::Status::OK;
    try
    {
        Dump(m_Request);

        OnRequestReceived(m_Request);
    }
    catch (std::exception &e)
    {
        clara::viz::Log(clara::viz::LogLevel::Error) << "RPC " << typeid(Request).name() << " failed with " << e.what();
        status = ::grpc::Status(grpc::StatusCode::UNKNOWN, e.what());
    }
    catch (...)
    {
        clara::viz::Log(clara::viz::LogLevel::Error) << "RPC " << typeid(Request).name() << " failed with unknown exception";
        status = ::grpc::Status(grpc::StatusCode::UNKNOWN, "Unknown exception");
    }

    if (!status.ok())
    {
        m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateFinishedDone;
        m_ResponseWriter->Finish(status, IContext::Tag());
        m_response_active = true;
    }
    else
    {
        StateWrite(true);
    }
    return true;
}

template<class Request, class Response>
bool LifeCycleServerStreaming<Request, Response>::StateWrite(bool ok)
{
    if (!ok)
    {
        // Data did not go to the wire because the call is already dead (i.e., canceled, deadline expired,
        // other side dropped the channel, etc).
        clara::viz::Log(clara::viz::LogLevel::Warning) << "RPC " << typeid(Request).name()
                                           << " could not be executed (canceled, deadline expired, channel dropped)";
        m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateFinishedDone;
        m_ResponseWriter->Finish(::grpc::Status::CANCELLED, IContext::Tag());
        m_response_active = true;
        return true;
    }

    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    if (m_Responses.empty() && !m_Finish && !m_Cancel)
    {
        m_write_waiting = true;
        m_write_condition.wait(lock);
        m_write_waiting = false;
    }

    if (m_Cancel)
    {
        m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateFinishedDone;
        m_ResponseWriter->Finish(::grpc::Status::CANCELLED, IContext::Tag());
        m_response_active = true;
    }
    else
    {
        if (!m_Responses.empty())
        {
            Dump(m_Responses.front());

            m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateWrite;
            m_ResponseWriter->Write(m_Responses.front(), IContext::Tag());
            m_response_active = true;
            m_Responses.pop_front();
        }
        else if (m_Finish)
        {
            m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateFinishedDone;
            m_ResponseWriter->Finish(::grpc::Status::OK, IContext::Tag());
            m_response_active = true;
        }
    }

    return true;
}

template<class Request, class Response>
bool LifeCycleServerStreaming<Request, Response>::StateFinishedDone(bool ok)
{
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    m_response_active = false;

    // execute callbacks
    for (auto &&callback : m_finish_callbacks)
    {
        (*callback)();
        callback->Invalidate();
    }
    m_finish_callbacks.clear();

    return false;
}

template<class Request, class Response>
void LifeCycleServerStreaming<Request, Response>::FinishResponse()
{
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    m_Finish = true;

    // if the request is actively writing messages wake it up
    if (m_write_waiting || m_response_active || !m_Responses.empty())
    {
        m_write_condition.notify_one();
    }
    else
    {
        m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateFinishedDone;
        m_ResponseWriter->Finish(::grpc::Status::OK, IContext::Tag());
        m_response_active = true;
    }
}

template<class Request, class Response>
void LifeCycleServerStreaming<Request, Response>::CancelResponse()
{
    std::unique_lock<std::recursive_mutex> lock(m_mutex);

    m_Cancel = true;

    // if the request is actively writing messages wake it up
    if (m_write_waiting || m_response_active || !m_Responses.empty())
    {
        m_write_condition.notify_one();
    }
    else
    {
        m_NextState = &LifeCycleServerStreaming<RequestType, ResponseType>::StateFinishedDone;
        m_ResponseWriter->Finish(::grpc::Status::CANCELLED, IContext::Tag());
        m_response_active = true;
    }
}

template<class Request, class Response>
void LifeCycleServerStreaming<Request, Response>::SetQueueFunc(ExecutorQueueFuncType queue_fn)
{
    m_QueuingFunc = queue_fn;
}

} // namespace nvrpc
