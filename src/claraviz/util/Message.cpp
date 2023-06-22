/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/Message.h"

#include <algorithm>
#include <list>
#include <queue>

#include "claraviz/util/Exception.h"
#include "claraviz/util/Synchronized.h"

namespace clara::viz
{

struct MessageProvider::Impl
{
    using ProtectedData = std::list<MessageReceiver *>;
    Synchronized<ProtectedData> receivers;
};

MessageProvider::MessageProvider()
    : impl_(new Impl)
{
}

MessageProvider::~MessageProvider() {}

void MessageProvider::SendMessage(const std::shared_ptr<const Message> &message)
{
    if (!message)
    {
        throw InvalidArgument("message") << "is a nullptr";
    }

    Synchronized<Impl::ProtectedData>::AccessGuard access(impl_->receivers);

    // iterate through the registered receivers, and send them the message
    std::for_each(access->cbegin(), access->cend(),
                  [&message](MessageReceiver *receiver) { receiver->EnqueueMessage(message); });
}

void MessageProvider::RegisterReceiver(const std::shared_ptr<MessageReceiver> &receiver)
{
    if (!receiver)
    {
        throw InvalidArgument("receiver") << "is a nullptr";
    }

    Synchronized<Impl::ProtectedData>::AccessGuard access(impl_->receivers);

    auto it = std::find(access->begin(), access->end(), receiver.get());
    if (it != access->end())
    {
        throw InvalidState() << "receiver is already registered";
    }

    // add to the receiver list
    access->push_back(receiver.get());

    // some providers need to know when new receivers are registered to e.g. send initial
    // state messages
    OnRegisterReceiver(receiver);
}

void MessageProvider::UnregisterReceiver(const std::shared_ptr<MessageReceiver> &receiver)
{
    if (!receiver)
    {
        throw InvalidArgument("receiver") << "is a nullptr";
    }

    Synchronized<Impl::ProtectedData>::AccessGuard access(impl_->receivers);

    auto it = std::find(access->begin(), access->end(), receiver.get());
    if (it == access->end())
    {
        throw InvalidState() << "receiver is not registered";
    }

    access->erase(it);
}

bool MessageProvider::HasReceivers()
{
    Synchronized<Impl::ProtectedData>::AccessGuard access(impl_->receivers);

    return (access->size() > 0);
}

struct MessageReceiver::Impl
{
    using ProtectedData = std::queue<std::shared_ptr<const Message>>;
    SynchronizedWait<ProtectedData> messages;
};

MessageReceiver::MessageReceiver()
    : impl_(new Impl)
{
}

MessageReceiver::~MessageReceiver() {}

void MessageReceiver::Wait()
{
    SynchronizedWait<Impl::ProtectedData>::AccessGuard access(impl_->messages);
    while (access->empty())
    {
        access.Wait();
    }
}

MessageReceiver::Status MessageReceiver::WaitFor(const std::chrono::nanoseconds &timeout)
{
    SynchronizedWait<Impl::ProtectedData>::AccessGuard access(impl_->messages);
    while (access->empty())
    {
        if (access.WaitFor(timeout) == std::cv_status::timeout)
        {
            return Status::TIMEOUT;
        }
    }
    return Status::NO_TIMEOUT;
}

void MessageReceiver::EnqueueMessage(const std::shared_ptr<const Message> &message)
{
    if (!message)
    {
        throw InvalidArgument("message") << "is a nullptr";
    }

    SynchronizedWait<Impl::ProtectedData>::AccessGuard access(impl_->messages);

    access->emplace(message);
    access.NotifyAll();
}

std::shared_ptr<const Message> MessageReceiver::DequeueMessage()
{
    SynchronizedWait<Impl::ProtectedData>::AccessGuard access(impl_->messages);
    if (access->empty())
    {
        return std::shared_ptr<const Message>();
    }

    auto message = access->front();
    access->pop();
    return message;
}

} // namespace clara::viz
