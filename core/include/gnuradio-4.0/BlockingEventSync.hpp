#ifndef GNURADIO_BLOCKINGEVENTSYNC_HPP
#define GNURADIO_BLOCKINGEVENTSYNC_HPP

#include <atomic>
#include <chrono>
#include <concepts>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

namespace gr {

/**
 * @brief CRTP mixin for event-driven blocks that need scheduler wakeups and optional background worker lifecycle.
 *
 * This mixin intentionally does not provide event queueing/buffering. Concrete blocks remain responsible
 * for event storage and backpressure policy.
 *
 * Provided functionality:
 * - Wake scheduler from any thread via eventNotify()
 * - Optional IO-pool worker via eventRunInIoPool()
 * - Stop token for worker loops via eventStopRequested()
 *
 * Usage example (event-driven source with worker thread):
 * @code
 * template<typename T>
 * struct DriverSource : gr::Block<DriverSource<T>>, gr::BlockingEventSync<DriverSource<T>> {
 *     gr::PortOut<T> out;
 *
 *     RingBuffer<T>      eventBuffer;
 *     std::atomic<bool>  dataReady{false};
 *     DriverTimingSource timing;
 *
 *     void start() {
 *         this->eventSyncStart();
 *         std::ignore = this->eventRunInIoPool("driver-worker", [this] {
 *             while (!this->eventStopRequested()) {
 *                 if (const bool signalsAvailable = timing.waitForSignal(5ms); !signalsAvailable) {
 *                     continue;
 *                 }
 *                 eventBuffer.push(timing->signals);
 *                 dataReady.store(true, std::memory_order_release);
 *                 this->eventNotify();
 *             }
 *         });
 *     }
 *
 *     void stop() {
 *         timing.stop();
 *         this->eventSyncStop();
 *     }
 *
 *     gr::work::Status processBulk(gr::OutputSpanLike auto& output) {
 *         if (!dataReady.exchange(false, std::memory_order_acq_rel)) {
 *             output.publish(0UZ);
 *             return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
 *         }
 *         const std::size_t n = eventBuffer.pop(output);
 *         output.publish(n);
 *         return n > 0UZ ? gr::work::Status::OK : gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
 *     }
 * };
 * @endcode
 */
template<typename Derived>
struct BlockingEventSync {
private:
    std::atomic<bool> _eventSync_stopRequested{false};
    std::atomic<bool> _eventSync_workerRunning{false};
    std::atomic<bool> _eventSync_workerDone{true};

    [[nodiscard]] constexpr Derived&       self() noexcept { return *static_cast<Derived*>(this); }
    [[nodiscard]] constexpr const Derived& self() const noexcept { return *static_cast<const Derived*>(this); }

    void stopWorker() {
        _eventSync_stopRequested.store(true, std::memory_order_release);
        if (_eventSync_workerRunning.load(std::memory_order_acquire)) {
            _eventSync_workerDone.wait(false);
        }
    }

public:
    void eventSyncStart() { _eventSync_stopRequested.store(false, std::memory_order_release); }

    void eventSyncStop() { stopWorker(); }

    ~BlockingEventSync() { stopWorker(); }

    [[nodiscard]] bool eventStopRequested() const noexcept { return _eventSync_stopRequested.load(std::memory_order_acquire); }
    [[nodiscard]] bool eventWorkerRunning() const noexcept { return _eventSync_workerRunning.load(std::memory_order_acquire); }

    void eventNotify() {
        self().progress->incrementAndGet();
        self().progress->notify_all();
    }

    template<typename Fn>
    requires std::invocable<Fn&>
    [[nodiscard]] bool eventRunInIoPool(std::string_view threadName, Fn&& fn) {
        bool expected = false;
        if (!_eventSync_workerRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return false;
        }

        _eventSync_workerDone.store(false, std::memory_order_release);
        _eventSync_stopRequested.store(false, std::memory_order_release);

        std::string name = threadName.empty() ? std::format("event:{}", self().name.value) : std::string(threadName);

        thread_pool::Manager::defaultIoPool()->execute([this, name = std::move(name), fn = std::forward<Fn>(fn)]() mutable {
            thread_pool::thread::setThreadName(name);
            try {
                std::invoke(fn);
            } catch (...) {
                // publish error message?
            }

            _eventSync_workerRunning.store(false, std::memory_order_release);
            _eventSync_workerDone.store(true, std::memory_order_release);
            _eventSync_workerDone.notify_all();
        });

        return true;
    }
};

} // namespace gr

#endif // GNURADIO_BLOCKINGEVENTSYNC_HPP
