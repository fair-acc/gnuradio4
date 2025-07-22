#ifndef CORE_TEST_UTILS_HPP
#define CORE_TEST_UTILS_HPP

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <expected>
#include <future>

/// the goal of utils.hpp is to reduce copy-paste between tests

namespace gr::testing {

namespace thread_pool {

/// Runs f() in a thread and returns a future you can join on
template<typename Func>
auto execute(std::string threadName, Func&& f) -> std::future<std::invoke_result_t<std::decay_t<Func>>> {
    using FuncResultType = std::invoke_result_t<std::decay_t<Func>>;
    std::promise<FuncResultType> promise;
    std::future<FuncResultType>  future = promise.get_future();

    auto lambda = [f, promise = std::move(promise), threadName] mutable {
        gr::thread_pool::thread::setThreadName(threadName);
        if constexpr (std::is_void_v<FuncResultType>) {
            f();
            promise.set_value();
        } else {
            promise.set_value(f());
        }
    };

    gr::thread_pool::Manager::defaultCpuPool()->execute(std::move(lambda));
    return future;
}

/// Runs the Scheduler's runAndWait() in a thread and returns a future you can join on
template<typename TScheduler>
std::future<std::expected<void, gr::Error>> executeScheduler(std::string threadName, TScheduler& sched) {
    auto f = [&sched] -> std::expected<void, Error> { return sched.runAndWait(); };
    return execute(threadName, std::move(f));
}

} // namespace thread_pool

} // namespace gr::testing

#endif
