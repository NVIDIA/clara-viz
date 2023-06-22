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

#include <chrono>
#include <mutex>
#include <condition_variable>

#include "claraviz/util/Types.h"

namespace clara::viz
{

namespace detail
{

/**
 * Used to add a condition variable if enable is set
 */
template<bool enable>
class Condition
{
};

/**
 * Specialization adding a condition variable
 */
template<>
class Condition<true>
{
protected:
    std::condition_variable_any m_condition;
};

} // namespace detail

/**
 * A template to force synchronized access to a data structure in a multi-threaded environment.
 * Data can only be accessed while the lock is held by AccessGuard.
 *
 * Usage:
 * @code{.cpp}
 *   // define the data to protect
 *   struct ProtectedData
 *   {
 *      uint32_t value;
 *   };
 *   // encapsulate using the template
 *   Synchronized<ProtectedData> shared;
 *
 *   {
 *       // take the lock
 *       Synchronized<ProtectedData>::AccessGuard accessProtectedData(shared);
 *       // access the data
 *       accessProtectedData->value = 42;
 *       // the lock is released when 'accessProtectedData' goes out of scope
 *   }
 * @endcode
 */
template<typename T, bool CONDITION = false>
class Synchronized
    : public NonCopyable
    , public detail::Condition<CONDITION>
{
public:
    /**
     * The type protected by the class.
     */
    using Type = T;

    /**
     * Access the data in a thread safe way (read/write).
     */
    class AccessGuard : public NonCopyable
    {
    public:
        /**
         * Construct
         *
         * @param synchronized [in]
         */
        AccessGuard(Synchronized &synchronized)
            : m_lock(synchronized.m_mutex)
            , m_synchronized(synchronized)
        {
        }

        /**
         * @returns reference to value
         */
        T &operator*() const
        {
            return m_synchronized.m_data;
        }
        /**
         * @returns pointer to value
         */
        T *operator->() const
        {
            return &m_synchronized.m_data;
        }

        /**
         * Wait on the condition variable.
         */
        template<typename U = T, typename = typename std::enable_if<CONDITION, U>::type>
        void Wait()
        {
            m_synchronized.m_condition.wait(m_lock);
        }

        /**
         * Wait on the condition variable for a given time.
         */
        template<typename REP, typename PERIOD, typename U = T, typename = typename std::enable_if<CONDITION, U>::type>
        std::cv_status WaitFor(const std::chrono::duration<REP, PERIOD> &duration)
        {
            return m_synchronized.m_condition.wait_for(m_lock, duration);
        }

        /**
         * Notify all waiters for the condition variable.
         */
        template<typename U = T, typename = typename std::enable_if<CONDITION, U>::type>
        void NotifyAll()
        {
            m_synchronized.m_condition.notify_all();
        }

    private:
        std::unique_lock<std::recursive_mutex> m_lock;
        Synchronized &m_synchronized;
    };

    /**
     * Access the data in a thread safe way (read only).
     * @todo can extend with std::shared_timed_mutex which allows multiple readers and single
     *       writer if this is a bottle neck
     */
    class AccessGuardConst : public NonCopyable
    {
    public:
        /**
         * Construct
         *
         * @param synchronized [in]
         */
        AccessGuardConst(const Synchronized &synchronized)
            : m_lock(const_cast<std::recursive_mutex &>(synchronized.m_mutex))
            , m_synchronized(synchronized)
        {
        }

        /**
         * @returns reference to value
         */
        const T &operator*() const
        {
            return m_synchronized.m_data;
        }
        /**
         * @returns pointer to value
         */
        const T *operator->() const
        {
            return &m_synchronized.m_data;
        }

        /**
         * Wait on the condition variable.
         */
        template<typename U = T, typename = typename std::enable_if<CONDITION, U>::type>
        void Wait()
        {
            const_cast<std::condition_variable_any &>(m_synchronized.m_condition).wait(m_lock);
        }

        /**
         * Wait on the condition variable for a given time.
         */
        template<typename REP, typename PERIOD, typename U = T, typename = typename std::enable_if<CONDITION, U>::type>
        std::cv_status WaitFor(const std::chrono::duration<REP, PERIOD> &duration)
        {
            return const_cast<std::condition_variable_any &>(m_synchronized.m_condition).wait_for(m_lock, duration);
        }

        /**
         * Notify all waiters for the condition variable.
         */
        template<typename U = T, typename = typename std::enable_if<CONDITION, U>::type>
        void NotifyAll()
        {
            const_cast<std::condition_variable_any &>(m_synchronized.m_condition).notify_all();
        }

    private:
        std::unique_lock<std::recursive_mutex> m_lock;
        const Synchronized &m_synchronized;
    };

private:
    std::recursive_mutex m_mutex;
    T m_data;
};

/**
 * Same as Synchronized but includes a condition variable which can be notified/waited on.
 */
template<typename T>
using SynchronizedWait = Synchronized<T, true>;

} // namespace clara::viz
