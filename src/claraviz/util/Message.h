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

#include <chrono>
#include <memory>
#include <string>

#include "claraviz/util/Types.h"

namespace clara::viz
{

class MessageProvider;

/**
 * Named unique message ID.
 */
class MessageID : public NonCopyable
{
public:
    /**
     * Construct
     *
     * @param name [in] unique name
     */
    explicit MessageID(const char *name)
        : name_(name)
    {
    }
    MessageID() = delete;

    /**
     * @returns true if equal
     */
    bool operator==(const MessageID &other) const
    {
        return (this->name_ == other.name_);
    }

    /**
     * @returns true if not equal
     */
    bool operator!=(const MessageID &other) const
    {
        return !operator==(other);
    }

    /**
     * @returns the message id name
     */
    const std::string &GetName() const
    {
        return name_;
    }

private:
    const std::string name_;
};

/**
 * Helper macro used to define message id values.
 */
#define DEFINE_MESSAGEID(NAME) const MessageID NAME(#NAME);
#define DEFINE_CLASS_MESSAGEID(CLASS) const MessageID CLASS::id_(#CLASS);

/**
 * Base class for messages.
 */
class Message
{
public:
    /**
     * Construct
     *
     * @param id [in] message id
     **/
    explicit Message(const MessageID &id)
        : id_(id)
    {
    }
    Message()          = delete;
    virtual ~Message() = default;

    /**
     * @returns the message Id
     */
    const MessageID &GetID() const
    {
        return id_;
    }

private:
    const MessageID &id_;
};

/**
 * Message receiver.
 */
class MessageReceiver
{
public:
    MessageReceiver();
    virtual ~MessageReceiver();

    /**
     * Enqueues a message in the queue.
     *
     * @param message [in] message to enqueue
     */
    void EnqueueMessage(const std::shared_ptr<const Message> &message);

protected:
    /**
     * Waits for new messages.
     */
    void Wait();

    /**
     * Describes whether a timeout occurred or not
     */
    enum class Status
    {
        NO_TIMEOUT,
        TIMEOUT
    };

    /**
     * Waits until new messages are available or a timeout occurs.
     *
     * @returns Status::TIMEOUT if the timeout expired, Status::NO_TIMEOUT else.
     */
    template<class Rep, class Period>
    Status WaitFor(const std::chrono::duration<Rep, Period> &timeout)
    {
        return WaitFor(std::chrono::duration_cast<std::chrono::nanoseconds>(timeout));
    }

    /**
     * Returns the next message in the queue. The returned event will be removed from the queue,
     * ownership is passed to the caller.
     *
     * @returns the next message, if the queue is empty, returns an empty pointer.
     */
    std::shared_ptr<const Message> DequeueMessage();

private:
    Status WaitFor(const std::chrono::nanoseconds &timeout);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * A message provider.
 */
class MessageProvider
{
public:
    MessageProvider();
    virtual ~MessageProvider();

    /**
     * Register a receiver to the provider. Each registered receiver will receive all messages.
     *
     * @param receiver [in] receiver to register
     */
    void RegisterReceiver(const std::shared_ptr<MessageReceiver> &receiver);

    /**
     * Unregister a receiver from the provider.
     *
     * @param receiver [in] receiver to unregister
     */
    void UnregisterReceiver(const std::shared_ptr<MessageReceiver> &receiver);

    /**
     * @returns true if any receivers are registered
     */
    bool HasReceivers();

protected:
    /**
     * Called when a new message receiver had been registered. The provider
     * can use that to e.g. send an initial state to newly registered receivers.
     *
     * @param receiver [in] receiver which had been registered
     */
    virtual void OnRegisterReceiver(const std::shared_ptr<MessageReceiver> &receiver) {}

    /**
     * Send a message.
     *
     * @param message [in] message to send
     */
    void SendMessage(const std::shared_ptr<const Message> &message);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clara::viz
