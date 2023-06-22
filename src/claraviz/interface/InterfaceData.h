/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/util/Synchronized.h"
#include "claraviz/util/Log.h"
#include "claraviz/util/Message.h"
#include "claraviz/util/Value.h"

namespace clara::viz
{

/**
 * Helper for defining types as is, e.g. 'InterfaceDirectT<int> v;' results in 'int v;'.
 */
template<typename T>
using InterfaceDirectT = T;

/**
 * Helper for defining types as Values, e.g. 'InterfaceValueT<int> v;' results in 'Value<int> v;'.
 */
template<typename T>
using InterfaceValueT = Value<T>;

/**
 * Interface data template.
 * Input interfaces write to 'DATA_IN' using 'AccessGuard' to access the input data
 * in a thread safe way.
 * The consumer will get output data which is a (sometimes partial) copy of the input
 * data.
 * Private data can be stored by implementers of the interface.
 * Messages are generated if data changed.
 *
 * @tparam DATA_IN input data to be stored type
 * @tparam DATA_OUT output data type
 * @tparam DATA_PRIVATE private data type
 */
template<typename DATA_IN, typename DATA_OUT, typename DATA_PRIVATE>
class InterfaceData : public MessageProvider
{
public:
    /**
     * Construct
     */
    InterfaceData()
    {
        Reset();
    }
    virtual ~InterfaceData() = default;

    /**
     * Input data
     */
    using DataIn = DATA_IN;

    /**
     * Output data
     */
    using DataOut = DATA_OUT;

    /**
     * Message which is generated when data had been changed.
     */
    class Message : public ::clara::viz::Message
    {
    public:
        /**
         * Construct
         *
         * @param data_out [in] message data
         */
        Message(const DataOut &data_out)
            : ::clara::viz::Message(id_)
            , data_out_(data_out)
        {
        }

        const DataOut data_out_; ///< message data

        static const MessageID id_; ///< message id
    };

    /**
     * Access the data in a thread safe way for writing.
     * When modification is done a message is sent.
     */
    class AccessGuard : public Synchronized<std::unique_ptr<DataIn>>::AccessGuard
    {
    public:
        /**
         * Construct
         *
         * @param interface_data [in]
         */
        AccessGuard(InterfaceData &interface_data)
            : Synchronized<std::unique_ptr<DataIn>>::AccessGuard(interface_data.data_in_)
            , uncaught_exceptions_(std::uncaught_exceptions())
            , interface_data_(interface_data)
        {
        }

        /**
         * Destruct. Sends a message with the output data.
         */
        ~AccessGuard()
        {
            // Only send the message if there is no new uncaught exception and there are receivers
            // If there is a new uncaught exception between construction and destruction of the AccessGuard
            // then the change to the interface data had not been finished successfully so no message
            // should be sent.
            if ((uncaught_exceptions_ == std::uncaught_exceptions()) && interface_data_.HasReceivers())
            {
                try
                {
                    // send the message
                    interface_data_.SendMessage(std::make_shared<Message>(interface_data_.Get()));
                }
                catch (const std::exception &e)
                {
                    Log(LogLevel::Error) << e.what();
                }
            }
        }

        /**
         * @returns a reference to the input data
         */
        DataIn &operator*()
        {
            return *(Get().get());
        }

        /**
         * @returns a pointer to the input data
         */
        DataIn *operator->()
        {
            return Get().get();
        }

    private:
        const int uncaught_exceptions_;

        friend InterfaceData;

        /**
         * @returns the unique pointer holding the input data
         **/
        std::unique_ptr<DataIn> &Get()
        {
            return (Synchronized<std::unique_ptr<DataIn>>::AccessGuard::operator*)();
        }

        InterfaceData &interface_data_;
    };

    /**
     * Access the data in a thread safe way for reading.
     */
    class AccessGuardConst : public Synchronized<std::unique_ptr<DataIn>>::AccessGuardConst
    {
    public:
        /**
         * Construct
         *
         * @param interface_data [in]
         */
        AccessGuardConst(InterfaceData *interface_data)
            : Synchronized<std::unique_ptr<DataIn>>::AccessGuardConst(interface_data->data_in_)
        {
        }

        /**
         * @returns a const reference to the input data
         */
        const DataIn &operator*() const
        {
            return *((Synchronized<std::unique_ptr<DataIn>>::AccessGuardConst::operator*)().get());
        }

        /**
         * @returns a const pointer to the input data
         */
        const DataIn *operator->() const
        {
            return (Synchronized<std::unique_ptr<DataIn>>::AccessGuardConst::operator*)().get();
        }
    };

    /**
     * Reset the data to defaults.
     */
    virtual void Reset()
    {
        AccessGuard access(*this);

        access.Get().reset(new DataIn);
    }

protected:
    /// since this input data is accessed concurrently by both the consumer (render thread) and the input
    /// service (gRPC), it has to be access protected by `Synchronized`
    Synchronized<std::unique_ptr<DataIn>> data_in_;

    /// private data
    DATA_PRIVATE data_;

private:
    /**
     * Called when a new message receiver had been registered. The interface
     * is using that to send an initial state to newly registered receivers.
     *
     * @param receiver [in] receiver which had been registered
     */
    void OnRegisterReceiver(const std::shared_ptr<MessageReceiver> &receiver) override
    {
        // only send the message if the input data is not empty since there is no initial state
        // in that case
        if (!std::is_empty<DataIn>::value)
        {
            receiver->EnqueueMessage(std::make_shared<Message>(Get()));
        }
    }

    /**
     * Get the output data.
     *
     * @returns output data
     */
    DataOut Get();
};

} // namespace clara::viz
