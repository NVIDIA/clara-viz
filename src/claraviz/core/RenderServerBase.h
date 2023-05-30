/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <memory>

#include <claraviz/rpc/ServerRPC.h>
#include <claraviz/util/Types.h>

namespace clara::viz
{

/**
 * Base class for all RenderServer's.
 * Contains a gRPC server and a signal handler to shutdown on Ctrl-C.
 */
class RenderServerBase : public NonCopyable
{
public:
    /**
     * Construct
     *
     * @param address [in] address to start the gRPC service on
     */
    RenderServerBase(const std::string &address);
    virtual ~RenderServerBase() = default;

    /**
     * @returns the RPC server
     */
    const std::shared_ptr<ServerRPC> &GetServerRPC();

    /**
     * Run the server.
     */
    virtual void Run();

    /**
     * Shutdown the server, wait for all running tasks to finish.
     *
     * @returns false if a timeout occurred, else true
     */
    virtual bool Shutdown();

    /**
     * Wait for the server, does not return until the server is shutdown.
     */
    virtual void Wait();

private:
    std::shared_ptr<ServerRPC> server_rpc_;
};

} // namespace clara::viz
