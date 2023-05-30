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

#pragma once

#include <memory>
#include <string>

#include <nvrpc/Service.h>
#include <nvrpc/Executor.h>

namespace clara::viz
{

/**
 * RPC server
 */
class ServerRPC
{
public:
    /**
     * Construct
     *
     * @param address [in] address to start the gRPC service on
     */
    ServerRPC(const std::string &address);
    ServerRPC() = delete;

    /**
     * Destruct.
     */
    ~ServerRPC();

    /**
     * Register a RPC service.
     *
     * @tparam SERVICE_TYPE     RPC service type. This is the class type of the service defined in the proto file
     *
     * @param service [in] the async service to register with the server, will be moved to the server
     *
     * @returns the service interface
     */
    template<typename SERVICE_TYPE>
    nvrpc::IService *RegisterService(std::unique_ptr<nvrpc::AsyncService<SERVICE_TYPE>> &&service)
    {
        auto i_service = std::unique_ptr<nvrpc::IService>(static_cast<nvrpc::IService *>(service.release()));
        return RegisterService(i_service);
    }

    /**
     * Creates an executor.
     *
     * @param numThreads [in] maximum threads
     * 
     * @returns the created executor
     */
    nvrpc::IExecutor *CreateExecutor(int numThreads = 1);

    /**
     * Register a RPC.
     *
     * For a proto file like that:
     *     package a.b;
     *     service SomeService {
     *         rpc MyRPC(MyRPCRequest) returns(MyRPCResponse) { }
     *     }
     * 'SERVICE_TYPE' would be 'a::b::SomeService::AsyncService',
     * 'CONTEXT_TYPE' would be 'MyRPCContext'
     * 'REQUEST_FUNC_TYPE' would be 'a::b::SomeService::AsyncService::RequestMyRPC'
     *
     * @tparam SERVICE_TYPE     RPC service type. This is the class type of the service defined in the proto file
     * @tparam CONTEXT_TYPE     call context type
     * @tparam REQUEST_FUNC_TYPE    request function type.
     *
     * @param service [in] RPC service
     * @param resource [in] RPC resource
     * @param request_function [in] function to be called
     * @param executor [in] the executor to use, if nullptr use the default executor
     */
    template<typename SERVICE_TYPE, typename CONTEXT_TYPE, typename REQUEST_FUNC_TYPE>
    void RegisterRPC(nvrpc::IService *service, std::shared_ptr<nvrpc::Resources> resource,
                     REQUEST_FUNC_TYPE request_function, nvrpc::IExecutor *executor = nullptr)
    {
        RegisterRPC(static_cast<nvrpc::AsyncService<SERVICE_TYPE> *>(service)->template RegisterRPC<CONTEXT_TYPE>(
                        request_function),
                    resource, executor);
    }

    /**
     * Run the server.
     */
    void Run();

    /**
     * Shutdown the server, wait for all running tasks to finish.
     *
     * @returns false if a timeout occurred, else true
     */
    bool Shutdown();

    /**
     * Wait for the server, does not return until the server is shutdown.
     */
    void Wait();

    /**
     * Readiness status for the server.
     */
    enum class State
    {
        /**
         * The server is in an invalid state and will not respond to any requests.
         */
        INVALID,

        /**
         * The server is initialized.
         */
        INITIALIZED,

        /**
         * The server is ready and accepting requests.
         */
        READY,

        /**
         * The server is exiting and will not respond to requests.
         */
        EXITING,

        /**
         * The server is shut down.
         */
        SHUTDOWN
    };

    /**
     * Return the ready state for the server.
     */
    State ReadyState() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    nvrpc::IService *RegisterService(std::unique_ptr<nvrpc::IService> &service);
    void RegisterRPC(nvrpc::IRPC *rpc, std::shared_ptr<nvrpc::Resources> &resource, nvrpc::IExecutor *executor);
};

} // namespace clara::viz
