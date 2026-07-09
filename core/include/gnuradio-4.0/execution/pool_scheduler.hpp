#ifndef GNURADIO_POOL_SCHEDULER_HPP
#define GNURADIO_POOL_SCHEDULER_HPP

#include <memory>

#include <gnuradio-4.0/execution/execution.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

namespace gr::execution {

/**
 * @brief P2300-compatible scheduler wrapping the existing GR4 thread pool.
 *
 * `schedule()` returns a sender that completes on the pool's threads. Use `cpuScheduler()`
 * or `ioScheduler()` for the default pools. `executor()` provides direct access to the
 * underlying TaskExecutor for features not exposed through the sender model.
 *
 * Usage:
 * @code
 * auto sched = gr::execution::cpuScheduler();
 * auto result = gr::execution::sync_wait(
 *     sched.schedule() | gr::execution::then([] { return computeOnPoolThread(); }));
 * @endcode
 */
class PoolScheduler {
    std::shared_ptr<thread_pool::TaskExecutor> _exec;

public:
    explicit PoolScheduler(std::shared_ptr<thread_pool::TaskExecutor> e) : _exec(std::move(e)) {}

    struct ScheduleSender {
        using completion_signatures_t = completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;
        using value_tuple_t           = std::tuple<>;

        std::shared_ptr<thread_pool::TaskExecutor> exec;

        template<receiver R>
        struct Op {
            std::shared_ptr<thread_pool::TaskExecutor> exec;
            std::shared_ptr<std::remove_cvref_t<R>>    recv; // heap-allocated to outlive Op

            void start() noexcept {
                try {
                    exec->execute([r = recv]() mutable { gr::execution::set_value(std::move(*r)); });
                } catch (...) {
                    gr::execution::set_error(std::move(*recv), std::current_exception());
                }
            }
        };

        template<receiver R>
        [[nodiscard]] auto connect(R&& r) const& {
            return Op<R>{exec, std::make_shared<std::remove_cvref_t<R>>(std::forward<R>(r))};
        }

        template<receiver R>
        [[nodiscard]] auto connect(R&& r) && {
            return Op<R>{std::move(exec), std::make_shared<std::remove_cvref_t<R>>(std::forward<R>(r))};
        }
    };

    [[nodiscard]] ScheduleSender schedule() const { return {_exec}; }

    thread_pool::TaskExecutor&       executor() noexcept { return *_exec; }
    const thread_pool::TaskExecutor& executor() const noexcept { return *_exec; }

    bool operator==(const PoolScheduler&) const = default;
};

[[nodiscard]] inline PoolScheduler cpuScheduler() { return PoolScheduler{thread_pool::Manager::defaultCpuPool()}; }
[[nodiscard]] inline PoolScheduler ioScheduler() { return PoolScheduler{thread_pool::Manager::defaultIoPool()}; }

} // namespace gr::execution

#endif // GNURADIO_POOL_SCHEDULER_HPP
