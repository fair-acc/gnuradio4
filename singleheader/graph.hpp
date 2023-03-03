#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

// #include "circular_buffer.hpp"
#ifndef GNURADIO_CIRCULAR_BUFFER_HPP
#define GNURADIO_CIRCULAR_BUFFER_HPP

#if defined(_LIBCPP_VERSION) and _LIBCPP_VERSION < 16000
#include <experimental/memory_resource>

namespace std::pmr {
using memory_resource = std::experimental::pmr::memory_resource;
template<typename T>
using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;
} // namespace std::pmr
#else
#include <memory_resource>
#endif
#include <algorithm>
#include <bit>
#include <cassert> // to assert if compiled for debugging
#include <functional>
#include <numeric>
#include <ranges>
#include <span>

#include <fmt/format.h>

// header for creating/opening or POSIX shared memory objects
#include <cerrno>
#include <fcntl.h>
#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gr {
static constexpr bool has_posix_mmap_interface = true;
}

#define HAS_POSIX_MAP_INTERFACE
#else
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif
#else // #if defined __has_include -- required for portability
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif

// #include "claim_strategy.hpp"
#ifndef GNURADIO_CLAIM_STRATEGY_HPP
#define GNURADIO_CLAIM_STRATEGY_HPP

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

// #include "wait_strategy.hpp"
#ifndef GNURADIO_WAIT_STRATEGY_HPP
#define GNURADIO_WAIT_STRATEGY_HPP

#include <condition_variable>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// #include "sequence.hpp"
#ifndef GNURADIO_SEQUENCE_HPP
#define GNURADIO_SEQUENCE_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

namespace gr {

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

static constexpr const std::size_t kCacheLine
        = 64; // waiting for clang: std::hardware_destructive_interference_size
static constexpr const std::int64_t kInitialCursorValue = -1L;

/**
 * Concurrent sequence class used for tracking the progress of the ring buffer and event
 * processors. Support a number of concurrent operations including CAS and order writes.
 * Also attempts to be more efficient with regards to false sharing by adding padding
 * around the volatile field.
 */
// clang-format off
class Sequence
{
    alignas(kCacheLine) std::atomic<std::int64_t> _fieldsValue{};

public:
    Sequence(const Sequence&) = delete;
    Sequence(const Sequence&&) = delete;
    void operator=(const Sequence&) = delete;
    explicit Sequence(std::int64_t initialValue = kInitialCursorValue) noexcept
        : _fieldsValue(initialValue)
    {
    }

    [[nodiscard]] forceinline std::int64_t value() const noexcept
    {
        return std::atomic_load_explicit(&_fieldsValue, std::memory_order_acquire);
    }

    forceinline void setValue(const std::int64_t value) noexcept
    {
        std::atomic_store_explicit(&_fieldsValue, value, std::memory_order_release);
    }

    [[nodiscard]] forceinline bool compareAndSet(std::int64_t expectedSequence,
                                                 std::int64_t nextSequence) noexcept
    {
        // atomically set the value to the given updated value if the current value == the
        // expected value (true, otherwise folse).
        return std::atomic_compare_exchange_strong(
            &_fieldsValue, &expectedSequence, nextSequence);
    }

    [[nodiscard]] forceinline std::int64_t incrementAndGet() noexcept
    {
        return std::atomic_fetch_add(&_fieldsValue, 1L) + 1L;
    }

    [[nodiscard]] forceinline std::int64_t addAndGet(std::int64_t value) noexcept
    {
        return std::atomic_fetch_add(&_fieldsValue, value) + value;
    }
};

namespace detail {
/**
 * Get the minimum sequence from an array of Sequences.
 *
 * \param sequences sequences to compare.
 * \param minimum an initial default minimum.  If the array is empty this value will
 * returned. \returns the minimum sequence found or lon.MaxValue if the array is empty.
 */
inline std::int64_t getMinimumSequence(
    const std::vector<std::shared_ptr<Sequence>>& sequences,
    std::int64_t minimum = std::numeric_limits<std::int64_t>::max()) noexcept
{
    if (sequences.empty()) {
        return minimum;
    }
#if not defined(_LIBCPP_VERSION)
    return std::min(minimum, std::ranges::min(sequences, std::less{}, [](const auto &sequence) noexcept { return sequence->value(); })->value());
#else
    std::vector<int64_t> v(sequences.size());
    std::transform(sequences.cbegin(), sequences.cend(), v.begin(), [](auto val) { return val->value(); });
    auto min = std::min(v.begin(), v.end());
    return std::min(*min, minimum);
#endif
}

inline void addSequences(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences,
             const Sequence& cursor,
             const std::vector<std::shared_ptr<Sequence>>& sequencesToAdd)
{
    std::int64_t cursorSequence;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> updatedSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> currentSequences;

    do {
        currentSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
        updatedSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(currentSequences->size() + sequencesToAdd.size());

#if not defined(_LIBCPP_VERSION)
        std::ranges::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#else
        std::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#endif

        cursorSequence = cursor.value();

        auto index = currentSequences->size();
        for (auto&& sequence : sequencesToAdd) {
            sequence->setValue(cursorSequence);
            (*updatedSequences)[index] = sequence;
            index++;
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &currentSequences, updatedSequences)); // xTODO: explicit memory order

    cursorSequence = cursor.value();

    for (auto&& sequence : sequencesToAdd) {
        sequence->setValue(cursorSequence);
    }
}

inline bool removeSequence(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences, const std::shared_ptr<Sequence>& sequence)
{
    std::uint32_t numToRemove;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> oldSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> newSequences;

    do {
        oldSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
#if not defined(_LIBCPP_VERSION)
        numToRemove = static_cast<std::uint32_t>(std::ranges::count(*oldSequences, sequence)); // specifically uses identity
#else
        numToRemove = static_cast<std::uint32_t>(std::count((*oldSequences).begin(), (*oldSequences).end(), sequence)); // specifically uses identity
#endif
        if (numToRemove == 0) {
            break;
        }

        auto oldSize = static_cast<std::uint32_t>(oldSequences->size());
        newSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(
            oldSize - numToRemove);

        for (auto i = 0U, pos = 0U; i < oldSize; ++i) {
            const auto& testSequence = (*oldSequences)[i];
            if (sequence != testSequence) {
                (*newSequences)[pos] = testSequence;
                pos++;
            }
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &oldSequences, newSequences));

    return numToRemove != 0;
}

// clang-format on

} // namespace detail

} // namespace gr

#ifdef FMT_FORMAT_H_
#include <fmt/core.h>
#include <fmt/ostream.h>

template<>
struct fmt::formatter<gr::Sequence> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto
    format(gr::Sequence const &value, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "{}", value.value());
    }
};

namespace gr {
inline std::ostream &
operator<<(std::ostream &os, const Sequence &v) {
    return os << fmt::format("{}", v);
}
} // namespace gr

#endif // FMT_FORMAT_H_

#endif // GNURADIO_SEQUENCE_HPP


namespace gr {
// clang-format off
/**
 * Wait for the given sequence to be available.  It is possible for this method to return a value less than the sequence number supplied depending on the implementation of the WaitStrategy.
 * A common use for this is to signal a timeout.Any EventProcessor that is using a WaitStrategy to get notifications about message becoming available should remember to handle this case.
 * The BatchEventProcessor<T> explicitly handles this case and will signal a timeout if required.
 *
 * \param sequence sequence to be waited on.
 * \param cursor Ring buffer cursor on which to wait.
 * \param dependentSequence on which to wait.
 * \param barrier barrier the IEventProcessor is waiting on.
 * \returns the sequence that is available which may be greater than the requested sequence.
 */
template<typename T>
inline constexpr bool isWaitStrategy = requires(T /*const*/ t, const std::int64_t sequence, const Sequence &cursor, std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
    { t.waitFor(sequence, cursor, dependentSequences) } -> std::same_as<std::int64_t>;
};
static_assert(!isWaitStrategy<int>);

/**
 * signal those waiting that the cursor has advanced.
 */
template<typename T>
inline constexpr bool hasSignalAllWhenBlocking = requires(T /*const*/ t) {
    { t.signalAllWhenBlocking() } -> std::same_as<void>;
};
static_assert(!hasSignalAllWhenBlocking<int>);

template<typename T>
concept WaitStrategy = isWaitStrategy<T>;



/**
 * Blocking strategy that uses a lock and condition variable for IEventProcessor's waiting on a barrier.
 * This strategy should be used when performance and low-latency are not as important as CPU resource.
 */
class BlockingWaitStrategy {
    std::recursive_mutex        _gate;
    std::condition_variable_any _conditionVariable;

public:
    std::int64_t waitFor(const std::int64_t sequence, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
        if (cursor.value() < sequence) {
            std::unique_lock uniqueLock(_gate);

            while (cursor.value() < sequence) {
                // optional: barrier check alert
                _conditionVariable.wait(uniqueLock);
            }
        }

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }

        return availableSequence;
    }

    void signalAllWhenBlocking() {
        std::unique_lock uniqueLock(_gate);
        _conditionVariable.notify_all();
    }
};
static_assert(WaitStrategy<BlockingWaitStrategy>);

/**
 * Busy Spin strategy that uses a busy spin loop for IEventProcessor's waiting on a barrier.
 * This strategy will use CPU resource to avoid syscalls which can introduce latency jitter.
 * It is best used when threads can be bound to specific CPU cores.
 */
struct BusySpinWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }
        return availableSequence;
    }
};
static_assert(WaitStrategy<BusySpinWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<BusySpinWaitStrategy>);

/**
 * Sleeping strategy that initially spins, then uses a std::this_thread::yield(), and eventually sleep. T
 * his strategy is a good compromise between performance and CPU resource.
 * Latency spikes can occur after quiet periods.
 */
class SleepingWaitStrategy {
    static const std::int32_t _defaultRetries = 200;
    std::int32_t              _retries        = 0;

public:
    explicit SleepingWaitStrategy(std::int32_t retries = _defaultRetries)
        : _retries(retries) {
    }

    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        auto       counter    = _retries;
        const auto waitMethod = [&counter]() {
            // optional: barrier check alert

            if (counter > 100) {
                --counter;
            } else if (counter > 0) {
                --counter;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(0));
            }
        };

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            waitMethod();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<SleepingWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<SleepingWaitStrategy>);

struct TimeoutException : public std::runtime_error {
    TimeoutException() : std::runtime_error("TimeoutException") {}
};

class TimeoutBlockingWaitStrategy {
    using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    Clock::duration             _timeout;
    std::recursive_mutex        _gate;
    std::condition_variable_any _conditionVariable;

public:
    explicit TimeoutBlockingWaitStrategy(Clock::duration timeout)
        : _timeout(timeout) {}

    std::int64_t waitFor(const std::int64_t sequence, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
        auto timeSpan = std::chrono::duration_cast<std::chrono::microseconds>(_timeout);

        if (cursor.value() < sequence) {
            std::unique_lock uniqueLock(_gate);

            while (cursor.value() < sequence) {
                // optional: barrier check alert

                if (_conditionVariable.wait_for(uniqueLock, timeSpan) == std::cv_status::timeout) {
                    throw TimeoutException();
                }
            }
        }

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }

        return availableSequence;
    }

    void signalAllWhenBlocking() {
        std::unique_lock uniqueLock(_gate);
        _conditionVariable.notify_all();
    }
};
static_assert(WaitStrategy<TimeoutBlockingWaitStrategy>);
static_assert(hasSignalAllWhenBlocking<TimeoutBlockingWaitStrategy>);

/**
 * Yielding strategy that uses a Thread.Yield() for IEventProcessors waiting on a barrier after an initially spinning.
 * This strategy is a good compromise between performance and CPU resource without incurring significant latency spikes.
 */
class YieldingWaitStrategy {
    const std::size_t _spinTries = 100;

public:
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        auto       counter    = _spinTries;
        const auto waitMethod = [&counter]() {
            // optional: barrier check alert

            if (counter == 0) {
                std::this_thread::yield();
            } else {
                --counter;
            }
        };

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            waitMethod();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<YieldingWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<YieldingWaitStrategy>);

struct NoWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> & /*dependentSequences*/) const {
        // wait for nothing
        return sequence;
    }
};
static_assert(WaitStrategy<NoWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<NoWaitStrategy>);


/**
 *
 * SpinWait is meant to be used as a tool for waiting in situations where the thread is not allowed to block.
 *
 * In order to get the maximum performance, the implementation first spins for `YIELD_THRESHOLD` times, and then
 * alternates between yielding, spinning and putting the thread to sleep, to allow other threads to be scheduled
 * by the kernel to avoid potential CPU contention.
 *
 * The number of spins, yielding, and sleeping for either '0 ms' or '1 ms' is controlled by the NTTP constants
 * @tparam YIELD_THRESHOLD
 * @tparam SLEEP_0_EVERY_HOW_MANY_TIMES
 * @tparam SLEEP_1_EVERY_HOW_MANY_TIMES
 */
template<std::int32_t YIELD_THRESHOLD = 10, std::int32_t SLEEP_0_EVERY_HOW_MANY_TIMES = 5, std::int32_t SLEEP_1_EVERY_HOW_MANY_TIMES = 20>
class SpinWait {
    using Clock         = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    std::int32_t _count = 0;
    static void  spinWaitInternal(std::int32_t iterationCount) noexcept {
        for (auto i = 0; i < iterationCount; i++) {
            yieldProcessor();
        }
    }
#ifndef __EMSCRIPTEN__
    static void yieldProcessor() noexcept { asm volatile("rep\nnop"); }
#else
    static void yieldProcessor() noexcept { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
#endif

public:
    SpinWait() = default;

    [[nodiscard]] std::int32_t count() const noexcept { return _count; }
    [[nodiscard]] bool         nextSpinWillYield() const noexcept { return _count > YIELD_THRESHOLD; }

    void                       spinOnce() {
        if (nextSpinWillYield()) {
            auto num = _count >= YIELD_THRESHOLD ? _count - 10 : _count;
            if (num % SLEEP_1_EVERY_HOW_MANY_TIMES == SLEEP_1_EVERY_HOW_MANY_TIMES - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                if (num % SLEEP_0_EVERY_HOW_MANY_TIMES == SLEEP_0_EVERY_HOW_MANY_TIMES - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(0));
                } else {
                    std::this_thread::yield();
                }
            }
        } else {
            spinWaitInternal(4 << _count);
        }

        if (_count == std::numeric_limits<std::int32_t>::max()) {
            _count = YIELD_THRESHOLD;
        } else {
            ++_count;
        }
    }

    void reset() noexcept { _count = 0; }

    template<typename T>
    requires std::is_nothrow_invocable_r_v<bool, T>
    bool
    spinUntil(const T &condition) const { return spinUntil(condition, -1); }

    template<typename T>
    requires std::is_nothrow_invocable_r_v<bool, T>
    bool
    spinUntil(const T &condition, std::int64_t millisecondsTimeout) const {
        if (millisecondsTimeout < -1) {
            throw std::out_of_range("Timeout value is out of range");
        }

        std::int64_t num = 0;
        if (millisecondsTimeout != 0 && millisecondsTimeout != -1) {
            num = getTickCount();
        }

        SpinWait spinWait;
        while (!condition()) {
            if (millisecondsTimeout == 0) {
                return false;
            }

            spinWait.spinOnce();

            if (millisecondsTimeout != 1 && spinWait.nextSpinWillYield() && millisecondsTimeout <= (getTickCount() - num)) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] static std::int64_t getTickCount() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }
};

/**
 * Spin strategy that uses a SpinWait for IEventProcessors waiting on a barrier.
 * This strategy is a good compromise between performance and CPU resource.
 * Latency spikes can occur after quiet periods.
 */
struct SpinWaitWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequence) const {
        std::int64_t availableSequence;

        SpinWait     spinWait;
        while ((availableSequence = detail::getMinimumSequence(dependentSequence)) < sequence) {
            // optional: barrier check alert
            spinWait.spinOnce();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<SpinWaitWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<SpinWaitWaitStrategy>);

struct NO_SPIN_WAIT {};

template<typename SPIN_WAIT = NO_SPIN_WAIT>
class AtomicMutex {
    std::atomic_flag _lock = ATOMIC_FLAG_INIT;
    SPIN_WAIT        _spin_wait;

public:
    AtomicMutex()                    = default;
    AtomicMutex(const AtomicMutex &) = delete;
    AtomicMutex &operator=(const AtomicMutex &) = delete;

    //
    void lock() {
        while (_lock.test_and_set(std::memory_order_acquire)) {
            if constexpr (requires { _spin_wait.spin_once(); }) {
                _spin_wait.spin_once();
            }
        }
        if constexpr (requires { _spin_wait.spin_once(); }) {
            _spin_wait.reset();
        }
    }
    void unlock() { _lock.clear(std::memory_order::release); }
};


// clang-format on
} // namespace gr


#endif // GNURADIO_WAIT_STRATEGY_HPP

// #include "sequence.hpp"


namespace gr {

struct NoCapacityException : public std::runtime_error {
    NoCapacityException() : std::runtime_error("NoCapacityException"){};
};

// clang-format off

template<typename T>
concept ClaimStrategy = requires(T /*const*/ t, const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity,
        const std::int64_t cursorValue, const std::int64_t sequence, const std::int64_t availableSequence, const std::int32_t n_slots_to_claim) {
    { t.hasAvailableCapacity(dependents, requiredCapacity, cursorValue) } -> std::same_as<bool>;
    { t.next(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.tryNext(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.getRemainingCapacity(dependents) } -> std::same_as<std::int64_t>;
    { t.publish(sequence) } -> std::same_as<void>;
    { t.isAvailable(sequence) } -> std::same_as<bool>;
    { t.getHighestPublishedSequence(sequence, availableSequence) } -> std::same_as<std::int64_t>;
};

namespace claim_strategy::util {
constexpr unsigned    floorlog2(std::size_t x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }
constexpr unsigned    ceillog2(std::size_t x) { return x == 1 ? 0 : floorlog2(x - 1) + 1; }
}

template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class alignas(kCacheLine) SingleThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::int64_t _nextValue{ kInitialCursorValue }; // N.B. no need for atomics since this is called by a single publisher
    alignas(kCacheLine) mutable std::int64_t _cachedValue{ kInitialCursorValue };

public:
    SingleThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy){};
    SingleThreadedStrategy(const SingleThreadedStrategy &)  = delete;
    SingleThreadedStrategy(const SingleThreadedStrategy &&) = delete;
    void operator=(const SingleThreadedStrategy &) = delete;

    bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity, const std::int64_t /*cursorValue*/) const noexcept {
        if (const std::int64_t wrapPoint = (_nextValue + requiredCapacity) - static_cast<std::int64_t>(_size); wrapPoint > _cachedValue || _cachedValue > _nextValue) {
            auto minSequence = detail::getMinimumSequence(dependents, _nextValue);
            _cachedValue     = minSequence;
            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int32_t n_slots_to_claim = 1) noexcept {
        assert((n_slots_to_claim > 0 && n_slots_to_claim <= static_cast<std::int32_t>(_size)) && "n_slots_to_claim must be > 0 and <= bufferSize");

        auto nextSequence = _nextValue + n_slots_to_claim;
        auto wrapPoint    = nextSequence - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _cachedValue; wrapPoint > cachedGatingSequence || cachedGatingSequence > _nextValue) {
            _cursor.setValue(_nextValue);

            SpinWait     spinWait;
            std::int64_t minSequence;
            while (wrapPoint > (minSequence = detail::getMinimumSequence(dependents, _nextValue))) {
                if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                    _waitStrategy.signalAllWhenBlocking();
                }
                spinWait.spinOnce();
            }
            _cachedValue = minSequence;
        }
        _nextValue = nextSequence;

        return nextSequence;
    }

    std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t n_slots_to_claim) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        if (!hasAvailableCapacity(dependents, n_slots_to_claim, 0 /* unused cursor value */)) {
            throw NoCapacityException();
        }

        const auto nextSequence = _nextValue + n_slots_to_claim;
        _nextValue              = nextSequence;

        return nextSequence;
    }

    std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto consumed = detail::getMinimumSequence(dependents, _nextValue);
        const auto produced = _nextValue;

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        _cursor.setValue(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept { return sequence <= _cursor.value(); }
    [[nodiscard]] std::int64_t     getHighestPublishedSequence(std::int64_t /*nextSequence*/, std::int64_t availableSequence) const noexcept { return availableSequence; }
};
static_assert(ClaimStrategy<SingleThreadedStrategy<1024, NoWaitStrategy>>);

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class MultiThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::vector<std::int32_t> _availableBuffer; // tracks the state of each ringbuffer slot
    alignas(kCacheLine) std::shared_ptr<Sequence> _gatingSequenceCache = std::make_shared<Sequence>();
    const std::int32_t _indexMask;
    const std::int32_t _indexShift;

public:
    MultiThreadedStrategy() = delete;
    explicit MultiThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy), _availableBuffer(_size),
        _indexMask(_size - 1), _indexShift(claim_strategy::util::ceillog2(_size)) {
        for (std::size_t i = _size - 1; i != 0; i--) {
            setAvailableBufferValue(i, -1);
        }
        setAvailableBufferValue(0, -1);
    }
    MultiThreadedStrategy(const MultiThreadedStrategy &)  = delete;
    MultiThreadedStrategy(const MultiThreadedStrategy &&) = delete;
    void               operator=(const MultiThreadedStrategy &) = delete;

    [[nodiscard]] bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int64_t requiredCapacity, const std::int64_t cursorValue) const noexcept {
        const auto wrapPoint = (cursorValue + requiredCapacity) - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _gatingSequenceCache->value(); wrapPoint > cachedGatingSequence || cachedGatingSequence > cursorValue) {
            const auto minSequence = detail::getMinimumSequence(dependents, cursorValue);
            _gatingSequenceCache->setValue(minSequence);

            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        SpinWait     spinWait;
        do {
            current                           = _cursor.value();
            next                              = current + n_slots_to_claim;

            std::int64_t wrapPoint            = next - static_cast<std::int64_t>(_size);
            std::int64_t cachedGatingSequence = _gatingSequenceCache->value();

            if (wrapPoint > cachedGatingSequence || cachedGatingSequence > current) {
                std::int64_t gatingSequence = detail::getMinimumSequence(dependents, current);

                if (wrapPoint > gatingSequence) {
                    if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                        _waitStrategy.signalAllWhenBlocking();
                    }
                    spinWait.spinOnce();
                    continue;
                }

                _gatingSequenceCache->setValue(gatingSequence);
            } else if (_cursor.compareAndSet(current, next)) {
                break;
            }
        } while (true);

        return next;
    }

    [[nodiscard]] std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        do {
            current = _cursor.value();
            next    = current + n_slots_to_claim;

            if (!hasAvailableCapacity(dependents, n_slots_to_claim, current)) {
                throw NoCapacityException();
            }
        } while (!_cursor.compareAndSet(current, next));

        return next;
    }

    [[nodiscard]] std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto produced = _cursor.value();
        const auto consumed = detail::getMinimumSequence(dependents, produced);

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        setAvailable(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept {
        const auto index = calculateIndex(sequence);
        const auto flag  = calculateAvailabilityFlag(sequence);

        return _availableBuffer[static_cast<std::size_t>(index)] == flag;
    }

    [[nodiscard]] forceinline std::int64_t getHighestPublishedSequence(const std::int64_t lowerBound, const std::int64_t availableSequence) const noexcept {
        for (std::int64_t sequence = lowerBound; sequence <= availableSequence; sequence++) {
            if (!isAvailable(sequence)) {
                return sequence - 1;
            }
        }

        return availableSequence;
    }

private:
    void                      setAvailable(std::int64_t sequence) noexcept { setAvailableBufferValue(calculateIndex(sequence), calculateAvailabilityFlag(sequence)); }
    forceinline void          setAvailableBufferValue(std::size_t index, std::int32_t flag) noexcept { _availableBuffer[index] = flag; }
    [[nodiscard]] forceinline std::int32_t calculateAvailabilityFlag(const std::int64_t sequence) const noexcept { return static_cast<std::int32_t>(static_cast<std::uint64_t>(sequence) >> _indexShift); }
    [[nodiscard]] forceinline std::size_t calculateIndex(const std::int64_t sequence) const noexcept { return static_cast<std::size_t>(static_cast<std::int32_t>(sequence) & _indexMask); }
};
static_assert(ClaimStrategy<MultiThreadedStrategy<1024, NoWaitStrategy>>);
// clang-format on

enum class ProducerType {
    /**
     * creates a buffer assuming a single producer-thread and multiple consumer
     */
    Single,

    /**
     * creates a buffer assuming multiple producer-threads and multiple consumer
     */
    Multi
};

namespace detail {
template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
struct producer_type;

template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Single, WAIT_STRATEGY> {
    using value_type = SingleThreadedStrategy<size, WAIT_STRATEGY>;
};
template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Multi, WAIT_STRATEGY> {
    using value_type = MultiThreadedStrategy<size, WAIT_STRATEGY>;
};

template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
using producer_type_v = typename producer_type<size, producerType, WAIT_STRATEGY>::value_type;

} // namespace detail

} // namespace gr


#endif // GNURADIO_CLAIM_STRATEGY_HPP

// #include "wait_strategy.hpp"

// #include "sequence.hpp"

// #include "buffer.hpp"
#ifndef GNURADIO_BUFFER2_H
#define GNURADIO_BUFFER2_H

#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <span>

namespace gr {
namespace util {
template <typename T, typename...>
struct first_template_arg_helper;

template <template <typename...> class TemplateType,
          typename ValueType,
          typename... OtherTypes>
struct first_template_arg_helper<TemplateType<ValueType, OtherTypes...>> {
    using value_type = ValueType;
};

template <typename T>
constexpr auto* value_type_helper()
{
    if constexpr (requires { typename T::value_type; }) {
        return static_cast<typename T::value_type*>(nullptr);
    }
    else {
        return static_cast<typename first_template_arg_helper<T>::value_type*>(nullptr);
    }
}

template <typename T>
using value_type_t = std::remove_pointer_t<decltype(value_type_helper<T>())>;

template <typename... A>
struct test_fallback {
};

template <typename, typename ValueType>
struct test_value_type {
    using value_type = ValueType;
};

static_assert(std::is_same_v<int, value_type_t<test_fallback<int, float, double>>>);
static_assert(std::is_same_v<float, value_type_t<test_value_type<int, float>>>);
static_assert(std::is_same_v<int, value_type_t<std::span<int>>>);
static_assert(std::is_same_v<double, value_type_t<std::array<double, 42>>>);

} // namespace util

// clang-format off
// disable formatting until clang-format (v16) supporting concepts
template<class T>
concept BufferReader = requires(T /*const*/ t, const std::size_t n_items) {
    { t.get(n_items) }     -> std::same_as<std::span<const util::value_type_t<T>>>;
    { t.consume(n_items) } -> std::same_as<bool>;
    { t.position() }       -> std::same_as<std::int64_t>;
    { t.available() }      -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class Fn, typename T, typename ...Args>
concept WriterCallback = std::is_invocable_v<Fn, std::span<T>&, std::int64_t, Args...> || std::is_invocable_v<Fn, std::span<T>&, Args...>;

template<class T, typename ...Args>
concept BufferWriter = requires(T t, const std::size_t n_items, std::pair<std::size_t, std::int64_t> token, Args ...args) {
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                                 -> std::same_as<void>;
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }   -> std::same_as<void>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                             -> std::same_as<bool>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }-> std::same_as<bool>;
    { t.get(n_items) } -> std::same_as<std::pair<std::span<util::value_type_t<T>>, std::pair<std::size_t, std::int64_t>>>;
    { t.publish(token, n_items) } -> std::same_as<void>;
    { t.available() }         -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class T, typename ...Args>
concept Buffer = requires(T t, const std::size_t min_size, Args ...args) {
    { T(min_size, args...) };
    { t.size() }       -> std::same_as<std::size_t>;
    { t.new_reader() } -> BufferReader;
    { t.new_writer() } -> BufferWriter;
};

// compile-time unit-tests
namespace test {
template <typename T>
struct non_compliant_class {
};
template <typename T, typename... Args>
using WithSequenceParameter = decltype([](std::span<T>&, std::int64_t, Args...) { /* */ });
template <typename T, typename... Args>
using NoSequenceParameter = decltype([](std::span<T>&, Args...) { /* */ });
} // namespace test

static_assert(!Buffer<test::non_compliant_class<int>>);
static_assert(!BufferReader<test::non_compliant_class<int>>);
static_assert(!BufferWriter<test::non_compliant_class<int>>);

static_assert(WriterCallback<test::WithSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::WithSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::WithSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::NoSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
// clang-format on
} // namespace gr

#endif // GNURADIO_BUFFER2_H


namespace gr {

namespace util {
constexpr std::size_t
round_up(std::size_t num_to_round, std::size_t multiple) noexcept {
    if (multiple == 0) {
        return num_to_round;
    }
    const auto remainder = num_to_round % multiple;
    if (remainder == 0) {
        return num_to_round;
    }
    return num_to_round + multiple - remainder;
}
} // namespace util

// clang-format off
class double_mapped_memory_resource : public std::pmr::memory_resource {
#ifdef HAS_POSIX_MAP_INTERFACE
    [[nodiscard]] void* do_allocate(const std::size_t required_size, std::size_t alignment) override {

        const std::size_t size = 2 * required_size;
        if (size % 2LU != 0LU || size % static_cast<std::size_t>(getpagesize()) != 0LU) {
            throw std::runtime_error(fmt::format("incompatible buffer-byte-size: {} -> {} alignment: {} vs. page size: {}", required_size, size, alignment, getpagesize()));
        }
        const std::size_t size_half = size/2;

        static std::size_t _counter;
        const auto buffer_name = fmt::format("/double_mapped_memory_resource-{}-{}-{}", getpid(), size, _counter++);
        const auto memfd_create = [name = buffer_name.c_str()](unsigned int flags) -> long {
            return syscall(__NR_memfd_create, name, flags);
        };
        int shm_fd = static_cast<int>(memfd_create(0));
        if (shm_fd < 0) {
            throw std::runtime_error(fmt::format("{} - memfd_create error {}: {}",  buffer_name, errno, strerror(errno)));
        }

        if (ftruncate(shm_fd, static_cast<off_t>(size)) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - ftruncate {}: {}",  buffer_name, errno, strerror(errno)));
        }

        void* first_copy = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for first half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for second half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // map the first half into the now available hole where the
        if (const void* second_copy = mmap(static_cast<char*> (first_copy) + size_half, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t> (0)); second_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed mmap for second copy {}: {}",  buffer_name, errno, strerror(errno)));
        }

        close(shm_fd); // file-descriptor is no longer needed. The mapping is retained.
        return first_copy;
}
#else
    [[nodiscard]] void* do_allocate(const std::size_t, std::size_t) override {
        throw std::runtime_error("OS does not provide POSIX interface for mmap(...) and munmao(...)");
        // static_assert(false, "OS does not provide POSIX interface for mmap(...) and munmao(...)");
    }
#endif


#ifdef HAS_POSIX_MAP_INTERFACE
    void  do_deallocate(void* p, std::size_t size, size_t alignment) override {

        if (munmap(p, size) == -1) {
            throw std::runtime_error(fmt::format("double_mapped_memory_resource::do_deallocate(void*, {}, {}) - munmap(..) failed", size, alignment));
        }
    }
#else
    void  do_deallocate(void*, std::size_t, size_t) override { }
#endif

    bool  do_is_equal(const memory_resource& other) const noexcept override { return this == &other; }

public:
    static inline double_mapped_memory_resource* defaultAllocator() {
        static auto instance = double_mapped_memory_resource();
        return &instance;
    }

    template<typename T>
    requires (std::has_single_bit(sizeof(T)))
    static inline std::pmr::polymorphic_allocator<T> allocator()
    {
        return std::pmr::polymorphic_allocator<T>(gr::double_mapped_memory_resource::defaultAllocator());
    }
};



/**
 * @brief circular buffer implementation using double-mapped memory allocations
 * where the first SIZE-ed buffer is mirrored directly its end to mimic wrap-around
 * free bulk memory access. The buffer keeps a list of indices (using 'Sequence')
 * to keep track of which parts can be tread-safely read and/or written
 *
 *                          wrap-around point
 *                                 |
 *                                 v
 *  | buffer segment #1 (original) | buffer segment #2 (copy of #1) |
 *  0                            SIZE                            2*SIZE
 *                      writeIndex
 *                          v
 *  wrap-free write access  |<-  N_1 < SIZE   ->|
 *
 *  readIndex < writeIndex-N_2
 *      v
 *      |<- N_2 < SIZE ->|
 *
 * N.B N_AVAILABLE := (SIZE + writeIndex - readIndex ) % SIZE
 *
 * citation: <find appropriate first and educational reference>
 *
 * This implementation provides single- as well as multi-producer/consumer buffer
 * combinations for thread-safe CPU-to-CPU data transfer (optionally) using either
 * a) the POSIX mmaped(..)/munmapped(..) MMU interface, if available, and/or
 * b) the guaranteed portable standard C/C++ (de-)allocators as a fall-back.
 *
 * for more details see
 */
template <typename T, std::size_t SIZE = std::dynamic_extent, ProducerType producer_type = ProducerType::Single, WaitStrategy WAIT_STRATEGY = SleepingWaitStrategy>
requires (std::has_single_bit(sizeof(T)))
class circular_buffer
{
    using Allocator         = std::pmr::polymorphic_allocator<T>;
    using BufferType        = circular_buffer<T, SIZE, producer_type, WAIT_STRATEGY>;
    using ClaimType         = detail::producer_type_v<SIZE, producer_type, WAIT_STRATEGY>;
    using DependendsType    = std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>;

    struct buffer_impl {
        alignas(kCacheLine) Allocator                   _allocator{};
        alignas(kCacheLine) const bool                  _is_mmap_allocated;
        alignas(kCacheLine) const std::size_t           _size;
        alignas(kCacheLine) std::vector<T, Allocator>   _data;
        alignas(kCacheLine) Sequence                    _cursor;
        alignas(kCacheLine) WAIT_STRATEGY               _wait_strategy = WAIT_STRATEGY();
        alignas(kCacheLine) ClaimType                   _claim_strategy;
        // list of dependent reader indices
        alignas(kCacheLine) DependendsType              _read_indices{ std::make_shared<std::vector<std::shared_ptr<Sequence>>>() };

        buffer_impl() = delete;
        buffer_impl(const std::size_t min_size, Allocator allocator) : _allocator(allocator), _is_mmap_allocated(dynamic_cast<double_mapped_memory_resource *>(_allocator.resource())),
            _size(align_with_page_size(min_size, _is_mmap_allocated)), _data(buffer_size(_size, _is_mmap_allocated), _allocator), _claim_strategy(ClaimType(_cursor, _wait_strategy, _size)) {
        }

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _is_mmap_allocated) {
            return _is_mmap_allocated ? util::round_up(min_size * sizeof(T), static_cast<std::size_t>(getpagesize())) / sizeof(T) : min_size;
        }
#else
        static std::size_t align_with_page_size(const std::size_t min_size, bool) {
            return min_size; // mmap() & getpagesize() not supported for non-POSIX OS
        }
#endif

        static std::size_t buffer_size(const std::size_t size, bool _is_mmap_allocated) {
            // double-mmaped behaviour requires the different size/alloc strategy
            // i.e. the second buffer half may not default-constructed as it's identical to the first one
            // and would result in a double dealloc during the default destruction
            return _is_mmap_allocated ? size : 2 * size;
        }
    };

    template <typename U = T>
    class buffer_writer {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) bool                        _is_mmap_allocated;
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;
        alignas(kCacheLine) ClaimType*                  _claim_strategy;

    public:
        buffer_writer() = delete;
        buffer_writer(std::shared_ptr<buffer_impl> buffer) :
            _buffer(std::move(buffer)), _is_mmap_allocated(_buffer->_is_mmap_allocated),
            _size(_buffer->_size), _data(std::addressof(_buffer->_data)), _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer(buffer_writer&& other)
            : _buffer(std::move(other._buffer))
            , _is_mmap_allocated(_buffer->_is_mmap_allocated)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data))
            , _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer& operator=(buffer_writer tmp) {
            std::swap(_buffer, tmp._buffer);
            _is_mmap_allocated = _buffer->_is_mmap_allocated;
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            _claim_strategy = std::addressof(_buffer->_claim_strategy);

            return *this;
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        [[nodiscard]] constexpr auto get(std::size_t n_slots_to_claim) noexcept -> std::pair<std::span<U>, std::pair<std::size_t, std::int64_t>> {
            try {
                const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim); // alt: try_next
                const std::size_t index = (sequence + _size - n_slots_to_claim) % _size;
                return {{ &(*_data)[index], n_slots_to_claim }, {index, sequence - n_slots_to_claim } };
            } catch (const NoCapacityException &) {
                return { { /* empty span */ }, { 0, 0 }};
            }
        }

        constexpr void publish(std::pair<std::size_t, std::int64_t> token, std::size_t n_produced) {
            if (!_is_mmap_allocated) {
                // mirror samples below/above the buffer's wrap-around point
                const std::size_t index = token.first;
                const size_t nFirstHalf = std::min(_size - index, n_produced);
                const size_t nSecondHalf = n_produced  - nFirstHalf;

                auto& data = *_data;
                std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
            }
            _claim_strategy->publish(token.second + n_produced); // points at first non-writable index
        }

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return;
            }
            const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim);
            translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
        } // blocks until elements are available

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr bool try_publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return true;
            }
            try {
                const auto sequence = _claim_strategy->tryNext(*_buffer->_read_indices, n_slots_to_claim);
                translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
                return true;
            } catch (const NoCapacityException &) {
                return false;
            }
        }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _claim_strategy->getRemainingCapacity(*_buffer->_read_indices);
        }

        private:
        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void translate_and_publish(Translator&& translator, const std::size_t n_slots_to_claim, const std::int64_t publishSequence, const Args&... args) {
            try {
                auto& data = *_data;
                const std::size_t index = (publishSequence + _size - n_slots_to_claim) % _size;
                std::span<U> writable_data(&data[index], n_slots_to_claim);
                if constexpr (std::is_invocable<Translator, std::span<T>&, std::int64_t, Args...>::value) {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), publishSequence - n_slots_to_claim, args...);
                } else {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), args...);
                }

                if (!_is_mmap_allocated) {
                    // mirror samples below/above the buffer's wrap-around point
                    const size_t nFirstHalf = std::min(_size - index, n_slots_to_claim);
                    const size_t nSecondHalf = n_slots_to_claim  - nFirstHalf;

                    std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                    std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
                }
                _claim_strategy->publish(publishSequence); // points at first non-writable index
            } catch (const std::exception& e) {
                throw e;
            } catch (...) {
                throw std::runtime_error("circular_buffer::translate_and_publish() - unknown user exception thrown");
            }
        }
    };

    template<typename U = T>
    class buffer_reader
    {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) std::shared_ptr<Sequence>   _read_index = std::make_shared<Sequence>();
        alignas(kCacheLine) std::int64_t                _read_index_cached;
        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;

    public:
        buffer_reader() = delete;
        buffer_reader(std::shared_ptr<buffer_impl> buffer) :
            _buffer(buffer), _size(buffer->_size), _data(std::addressof(buffer->_data)) {
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_read_index});
            _read_index_cached = _read_index->value();
        }
        buffer_reader(buffer_reader&& other)
            : _read_index(std::move(other._read_index))
            , _read_index_cached(std::exchange(other._read_index_cached, _read_index->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data)) {
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_read_index, tmp._read_index);
            std::swap(_read_index_cached, tmp._read_index_cached);
            std::swap(_buffer, tmp._buffer);
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _read_index); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        template <bool strict_check = true>
        [[nodiscard]] constexpr std::span<const U> get(const std::size_t n_requested = 0) const noexcept {
            const auto& data = *_data;
            if constexpr (strict_check) {
                const std::size_t n = n_requested > 0 ? std::min(n_requested, available()) : available();
                return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
            }
            const std::size_t n = n_requested > 0 ? n_requested : available();
            return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
        }

        template <bool strict_check = true>
        [[nodiscard]] constexpr bool consume(const std::size_t n_elements = 1) noexcept {
            if constexpr (strict_check) {
                if (n_elements <= 0) {
                    return true;
                }
                if (n_elements > available()) {
                    return false;
                }
            }
            _read_index_cached = _read_index->addAndGet(static_cast<int64_t>(n_elements));
            return true;
        }

        [[nodiscard]] constexpr std::int64_t position() const noexcept { return _read_index_cached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _buffer->_cursor.value() - _read_index_cached;
        }
    };

    [[nodiscard]] constexpr static Allocator DefaultAllocator() {
        if constexpr (has_posix_mmap_interface) {
            return double_mapped_memory_resource::allocator<T>();
        } else {
            return Allocator();
        }
    }

    std::shared_ptr<buffer_impl> _shared_buffer_ptr;
    circular_buffer(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    circular_buffer() = delete;
    circular_buffer(std::size_t min_size, Allocator allocator = DefaultAllocator())
        : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size, allocator)) { }
    ~circular_buffer() = default;

    [[nodiscard]] std::size_t       size() const noexcept { return _shared_buffer_ptr->_size; }
    [[nodiscard]] BufferWriter auto new_writer() { return buffer_writer<T>(_shared_buffer_ptr); }
    [[nodiscard]] BufferReader auto new_reader() { return buffer_reader<T>(_shared_buffer_ptr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] auto n_readers()       { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] auto claim_strategy()  { return _shared_buffer_ptr->_claim_strategy; }
    [[nodiscard]] auto wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] auto cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<circular_buffer<int32_t>>);
// clang-format on

} // namespace gr

#endif // GNURADIO_CIRCULAR_BUFFER_HPP

// #include "buffer.hpp"

// #include "typelist.hpp"
#ifndef GNURADIO_TYPELIST_HPP
#define GNURADIO_TYPELIST_HPP

#include <bit>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <string_view>
#include <string>

namespace fair::meta {

template<typename... Ts>
struct typelist;

// concat ///////////////
namespace detail {
template<typename...>
struct concat_impl;

template<>
struct concat_impl<> {
    using type = typelist<>;
};

template<typename A>
struct concat_impl<A> {
    using type = typelist<A>;
};

template<typename... As>
struct concat_impl<typelist<As...>> {
    using type = typelist<As...>;
};

template<typename A, typename B>
struct concat_impl<A, B> {
    using type = typelist<A, B>;
};

template<typename... As, typename B>
struct concat_impl<typelist<As...>, B> {
    using type = typelist<As..., B>;
};

template<typename A, typename... Bs>
struct concat_impl<A, typelist<Bs...>> {
    using type = typelist<A, Bs...>;
};

template<typename... As, typename... Bs>
struct concat_impl<typelist<As...>, typelist<Bs...>> {
    using type = typelist<As..., Bs...>;
};

template<typename A, typename B, typename C>
struct concat_impl<A, B, C> {
    using type = typename concat_impl<typename concat_impl<A, B>::type, C>::type;
};

template<typename A, typename B, typename C, typename D, typename... More>
struct concat_impl<A, B, C, D, More...> {
    using type =
            typename concat_impl<typename concat_impl<A, B>::type, typename concat_impl<C, D>::type,
                                 typename concat_impl<More...>::type>::type;
};
} // namespace detail

template<typename... Ts>
using concat = typename detail::concat_impl<Ts...>::type;

// split_at, left_of, right_of ////////////////
namespace detail {
template<unsigned N>
struct splitter;

template<>
struct splitter<0> {
    template<typename...>
    using first = typelist<>;
    template<typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<1> {
    template<typename T0, typename...>
    using first = typelist<T0>;
    template<typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<2> {
    template<typename T0, typename T1, typename...>
    using first = typelist<T0, T1>;
    template<typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<4> {
    template<typename T0, typename T1, typename T2, typename T3, typename...>
    using first = typelist<T0, T1, T2, T3>;
    template<typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<8> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7>;

    template<typename, typename, typename, typename, typename, typename, typename, typename,
             typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<16> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>;

    template<typename, typename, typename, typename, typename, typename, typename, typename, typename,
             typename, typename, typename, typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<unsigned N>
struct splitter {
    static constexpr unsigned FirstSplit = std::has_single_bit(N) ? N / 2 : std::bit_floor(N);
    using A                              = splitter<FirstSplit>;
    using B                              = splitter<N - FirstSplit>;

    template<typename... Ts>
    using first = concat<typename A::template first<Ts...>,
                         typename B::template first<typename A::template second<Ts...>>>;

    template<typename... Ts>
    using second = typename B::template second<typename A::template second<Ts...>>;
};
} // namespace detail

template<unsigned N, typename List>
struct split_at;

template<unsigned N, typename... Ts>
struct split_at<N, typelist<Ts...>> {
    using first  = typename detail::splitter<N>::template first<Ts...>;
    using second = typename detail::splitter<N>::template second<Ts...>;
};

template<std::size_t N, typename List>
using left_of = typename split_at<N, List>::first;

template<std::size_t N, typename List>
using right_of = typename split_at<N + 1, List>::second;

// remove_at /////////////
template<std::size_t Idx, typename List>
using remove_at = concat<left_of<Idx, List>, right_of<Idx, List>>;

// first_type ////////////
namespace detail {
template<typename List>
struct first_type_impl {};

template<typename T0, typename... Ts>
struct first_type_impl<typelist<T0, Ts...>> {
    using type = T0;
};
} // namespace detail

template<typename List>
using first_type = typename detail::first_type_impl<List>::type;

// transform_types ////////////
namespace detail {
template<template<typename> class Template, typename List>
struct transform_types_impl;

template<template<typename> class Template, typename... Ts>
struct transform_types_impl<Template, typelist<Ts...>> {
    using type = typelist<Template<Ts>...>;
};
} // namespace detail

template<template<typename> class Template, typename List>
using transform_types = typename detail::transform_types_impl<Template, List>::type;

// transform_value_type
template<typename T>
using transform_value_type = typename T::value_type;

// reduce ////////////////
namespace detail {
template<template<typename, typename> class Method, typename List>
struct reduce_impl;

template<template<typename, typename> class Method, typename T0>
struct reduce_impl<Method, typelist<T0>> {
    using type = T0;
};

template<template<typename, typename> class Method, typename T0, typename T1, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, Ts...>>
    : public reduce_impl<Method, typelist<typename Method<T0, T1>::type, Ts...>> {};

template<template<typename, typename> class Method, typename T0, typename T1, typename T2,
         typename T3, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, T2, T3, Ts...>>
    : public reduce_impl<
              Method, typelist<typename Method<T0, T1>::type, typename Method<T2, T3>::type, Ts...>> {
};
} // namespace detail

template<template<typename, typename> class Method, typename List>
using reduce = typename detail::reduce_impl<Method, List>::type;

// typelist /////////////////
template<typename T>
concept is_typelist_v = requires { typename T::typelist_tag; };

template<typename... Ts>
struct typelist {
    using this_t = typelist<Ts...>;
    using typelist_tag = std::true_type;

    static inline constexpr std::integral_constant<std::size_t, sizeof...(Ts)> size = {};

    template<template<typename...> class Other>
    using apply = Other<Ts...>;

    template<std::size_t I>
    using at = first_type<typename detail::splitter<I>::template second<Ts...>>;

    template <typename Head>
    using prepend = typelist<Head, Ts...>;

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Ts, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Ts> && ...);

    template<typename F, typename Tup>
        requires(sizeof...(Ts) == std::tuple_size_v<std::remove_cvref_t<Tup>>)
    static constexpr auto
    construct(Tup &&args_tuple) {
        return std::apply(
                []<typename... Args>(Args &&...args) {
                    return std::make_tuple(F::template apply<Ts>(std::forward<Args>(args))...);
                },
                std::forward<Tup>(args_tuple));
    }

    template<template<typename> typename Trafo>
    using transform = meta::transform_types<Trafo, this_t>;

    template<template<typename...> typename Pred>
    constexpr static bool all_of = (Pred<Ts>::value && ...);

    template<template<typename...> typename Pred>
    constexpr static bool none_of = (!Pred<Ts>::value && ...);

    using safe_head = std::remove_pointer_t<decltype([] {
        if constexpr (sizeof...(Ts) > 0) {
            return static_cast<this_t::at<0>*>(nullptr);
        } else {
            return static_cast<void*>(nullptr);
        }
    }())>;

    template<typename Matcher = typename this_t::safe_head>
    constexpr static bool all_same =
        ((std::is_same_v<Matcher, Ts> && ...));

    template<template<typename...> typename Predicate>
    using filter = concat<std::conditional_t<Predicate<Ts>::value, typelist<Ts>, typelist<>>...>;

    using tuple_type    = std::tuple<Ts...>;
    using tuple_or_type = std::remove_pointer_t<decltype(
            [] {
                if constexpr (sizeof...(Ts) == 0) {
                    return static_cast<void*>(nullptr);
                } else if constexpr (sizeof...(Ts) == 1) {
                    return static_cast<at<0>*>(nullptr);
                } else {
                    return static_cast<tuple_type*>(nullptr);
                }
            }())>;

};


namespace detail {
    template <template <typename...> typename OtherTypelist, typename... Args>
    meta::typelist<Args...> to_typelist_helper(OtherTypelist<Args...>*);
} // namespace detail

template <typename OtherTypelist>
using to_typelist = decltype(detail::to_typelist_helper(static_cast<OtherTypelist*>(nullptr)));

} // namespace fair::meta

#endif // include guard

// #include "port.hpp"
#ifndef GNURADIO_PORT_HPP
#define GNURADIO_PORT_HPP

#include <variant>
#include <complex>
#include <span>

// #include "utils.hpp"
#ifndef GNURADIO_GRAPH_UTILS_HPP
#define GNURADIO_GRAPH_UTILS_HPP

#include <functional>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>

// #include "typelist.hpp"

// #include "vir/simd.h"
/*
    Copyright © 2022 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
                     Matthias Kretz <m.kretz@gsi.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VIR_SIMD_H_
#define VIR_SIMD_H_

#if __cplusplus < 201703L
#error "simd requires C++17 or later"
#endif

#if __has_include (<experimental/simd>) && !defined VIR_DISABLE_STDX_SIMD && !defined __clang__
#include <experimental/simd>
#endif

#if defined __cpp_lib_experimental_parallel_simd && __cpp_lib_experimental_parallel_simd >= 201803

namespace vir::stdx
{
  using namespace std::experimental::parallelism_v2;
}

#else

#include <cmath>
#include <cstring>
#ifdef _GLIBCXX_DEBUG_UB
#include <cstdio>
#endif
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef VIR_SIMD_TS_DROPIN
namespace std::experimental
{
  inline namespace parallelism_v2
#else
namespace vir::stdx
#endif
{
  using std::size_t;

  namespace detail
  {
    template <typename T>
      struct type_identity
      { using type = T; };

    template <typename T>
      using type_identity_t = typename type_identity<T>::type;

    constexpr size_t
    bit_ceil(size_t x)
    {
      size_t r = 1;
      while (r < x)
        r <<= 1;
      return r;
    }

    constexpr size_t
    bit_floor(size_t x)
    {
      size_t r = x;
      do {
        r = x;
        x &= x - 1;
      } while (x);
      return r;
    }

    template <typename T>
      typename T::value_type
      value_type_or_identity_impl(int);

    template <typename T>
      T
      value_type_or_identity_impl(float);

    template <typename T>
      using value_type_or_identity_t
        = decltype(value_type_or_identity_impl<T>(int()));

    class ExactBool
    {
      const bool data;

    public:
      constexpr ExactBool(bool b) : data(b) {}

      ExactBool(int) = delete;

      constexpr operator bool() const { return data; }
    };

    template <typename... Args>
      [[noreturn]] [[gnu::always_inline]] inline void
      invoke_ub([[maybe_unused]] const char* msg,
                [[maybe_unused]] const Args&... args)
      {
#ifdef _GLIBCXX_DEBUG_UB
        std::fprintf(stderr, msg, args...);
        __builtin_trap();
#else
        __builtin_unreachable();
#endif
      }

    template <typename T>
      using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <typename T>
      using L = std::numeric_limits<T>;

    template <bool B>
      using BoolConstant = std::integral_constant<bool, B>;

    template <size_t X>
      using SizeConstant = std::integral_constant<size_t, X>;

    template <size_t I, typename T, typename... Ts>
      constexpr auto
      pack_simd_subscript(const T& x0, const Ts&... xs)
      {
        if constexpr (I >= T::size())
          return pack_simd_subscript<I - T::size()>(xs...);
        else
          return x0[I];
      }

    template <class T>
      struct is_vectorizable : std::is_arithmetic<T>
      {};

    template <>
      struct is_vectorizable<bool> : std::false_type
      {};

    template <class T>
      inline constexpr bool is_vectorizable_v = is_vectorizable<T>::value;

    template <class T, typename = void>
      struct only_vectorizable
      {
        only_vectorizable() = delete;
        only_vectorizable(const only_vectorizable&) = delete;
        only_vectorizable(only_vectorizable&&) = delete;
        ~only_vectorizable() = delete;
      };

    template <class T>
      struct only_vectorizable<T, std::enable_if_t<is_vectorizable_v<T>>>
      {
      };

    // Deduces to a vectorizable type
    template <typename T, typename = std::enable_if_t<is_vectorizable_v<T>>>
      using Vectorizable = T;

    // Deduces to a floating-point type
    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
      using FloatingPoint = T;

    // Deduces to a signed integer type
    template <typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>,
                                                                         std::is_signed<T>>>>
      using SignedIntegral = T;

    // is_higher_integer_rank<T, U> (T has higher or equal integer rank than U)
    template <typename T, typename U, bool = (sizeof(T) > sizeof(U)),
              bool = (sizeof(T) == sizeof(U))>
      struct is_higher_integer_rank;

    template <typename T>
      struct is_higher_integer_rank<T, T, false, true>
      : public std::true_type
      {};

    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, true, false>
      : public std::true_type
      {};

    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, false, false>
      : public std::false_type
      {};

    // this may fail for char -> short if sizeof(char) == sizeof(short)
    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, false, true>
      : public std::is_same<decltype(std::declval<T>() + std::declval<U>()), T>
      {};

    // is_value_preserving<From, To>
    template <typename From, typename To, bool = std::is_arithmetic_v<From>,
              bool = std::is_arithmetic_v<To>>
      struct is_value_preserving;

    // ignore "signed/unsigned mismatch" in the following trait.
    // The implicit conversions will do the right thing here.
    template <typename From, typename To>
      struct is_value_preserving<From, To, true, true>
      : public BoolConstant<L<From>::digits <= L<To>::digits
                              && L<From>::max() <= L<To>::max()
                              && L<From>::lowest() >= L<To>::lowest()
                              && !(std::is_signed_v<From> && std::is_unsigned_v<To>)> {};

    template <typename T>
      struct is_value_preserving<T, bool, true, true>
      : public std::false_type {};

    template <>
      struct is_value_preserving<bool, bool, true, true>
      : public std::true_type {};

    template <typename T>
      struct is_value_preserving<T, T, true, true>
      : public std::true_type {};

    template <typename From, typename To>
      struct is_value_preserving<From, To, false, true>
      : public std::is_convertible<From, To> {};

    template <typename From, typename To,
              typename = std::enable_if_t<is_value_preserving<remove_cvref_t<From>, To>::value>>
      using ValuePreserving = From;

    template <typename From, typename To,
              typename DecayedFrom = remove_cvref_t<From>,
              typename = std::enable_if_t<std::conjunction<
                                            std::is_convertible<From, To>,
                                            std::disjunction<
                                              std::is_same<DecayedFrom, To>,
                                              std::is_same<DecayedFrom, int>,
                                              std::conjunction<std::is_same<DecayedFrom, unsigned>,
                                                               std::is_unsigned<To>>,
                                              is_value_preserving<DecayedFrom, To>>>::value>>
      using ValuePreservingOrInt = From;

    // LoadStorePtr / is_possible_loadstore_conversion
    template <typename Ptr, typename ValueType>
      struct is_possible_loadstore_conversion
      : std::conjunction<is_vectorizable<Ptr>, is_vectorizable<ValueType>>
      {};

    template <>
      struct is_possible_loadstore_conversion<bool, bool> : std::true_type {};

    // Deduces to a type allowed for load/store with the given value type.
    template <typename Ptr, typename ValueType,
              typename = std::enable_if_t<
                           is_possible_loadstore_conversion<Ptr, ValueType>::value>>
      using LoadStorePtr = Ptr;
  }

  namespace simd_abi
  {
    struct scalar
    {};

    template <typename>
      inline constexpr int max_fixed_size = 32;

    template <int N>
      struct fixed_size
      {};

    template <class T>
      using native =
        std::conditional_t<(sizeof(T) > 8),
                           scalar,
                           fixed_size<
#ifdef __AVX512F__
                             64
#elif defined __AVX2__
                             32
#elif defined __AVX__
                             std::is_floating_point_v<T> ? 32 : 16
#else
                             16
#endif
                               / sizeof(T)
                           >
                          >;

    template <class T>
      using compatible = std::conditional_t<(sizeof(T) > 8),
                                            scalar,
                                            fixed_size<16 / sizeof(T)>>;

    template <typename T, size_t N, typename...>
      struct deduce
      { using type = std::conditional_t<N == 1, scalar, fixed_size<int(N)>>; };

    template <typename T, size_t N, typename... Abis>
      using deduce_t = typename deduce<T, N, Abis...>::type;
  }

  // flags //
  struct element_aligned_tag
  {};

  struct vector_aligned_tag
  {};

  template <size_t>
    struct overaligned_tag
    {};

  inline constexpr element_aligned_tag element_aligned{};

  inline constexpr vector_aligned_tag vector_aligned{};

  template <size_t N>
    inline constexpr overaligned_tag<N> overaligned{};

  // fwd decls //
  template <class T, class A = simd_abi::compatible<T>>
    class simd
    {
      simd() = delete;
      simd(const simd&) = delete;
      ~simd() = delete;
    };

  template <class T, class A = simd_abi::compatible<T>>
    class simd_mask
    {
      simd_mask() = delete;
      simd_mask(const simd_mask&) = delete;
      ~simd_mask() = delete;
    };

  // aliases //
  template <class T>
    using native_simd = simd<T, simd_abi::native<T>>;

  template <class T>
    using native_simd_mask = simd_mask<T, simd_abi::native<T>>;

  template <class T, int N>
    using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;

  template <class T, int N>
    using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

  // Traits //
  template <class T>
    struct is_abi_tag : std::false_type
    {};

  template <class T>
    inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

  template <>
    struct is_abi_tag<simd_abi::scalar> : std::true_type
    {};

  template <int N>
    struct is_abi_tag<simd_abi::fixed_size<N>> : std::true_type
    {};

  template <class T>
    struct is_simd : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_v = is_simd<T>::value;

  template <class T, class A>
    struct is_simd<simd<T, A>>
    : std::conjunction<detail::is_vectorizable<T>, is_abi_tag<A>>
    {};

  template <class T>
    struct is_simd_mask : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

  template <class T, class A>
    struct is_simd_mask<simd_mask<T, A>>
    : std::conjunction<detail::is_vectorizable<T>, is_abi_tag<A>>
    {};

  template <class T>
    struct is_simd_flag_type : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

  template <class T, class A = simd_abi::compatible<T>>
    struct simd_size;

  template <class T, class A = simd_abi::compatible<T>>
    inline constexpr size_t simd_size_v = simd_size<T, A>::value;

  template <class T>
    struct simd_size<detail::Vectorizable<T>, simd_abi::scalar>
    : std::integral_constant<size_t, 1>
    {};

  template <class T, int N>
    struct simd_size<detail::Vectorizable<T>, simd_abi::fixed_size<N>>
    : std::integral_constant<size_t, N>
    {};

  template <class T, class U = typename T::value_type>
    struct memory_alignment;

  template <class T, class U = typename T::value_type>
    inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

  template <class T, class A, class U>
    struct memory_alignment<simd<T, A>, detail::Vectorizable<U>>
    : std::integral_constant<size_t, alignof(U)>
    {};

  template <class T, class A>
    struct memory_alignment<simd_mask<T, A>, bool>
    : std::integral_constant<size_t, alignof(bool)>
    {};

  template <class T, class V,
            class = typename std::conjunction<detail::is_vectorizable<T>,
                                              std::disjunction<is_simd<V>, is_simd_mask<V>>>::type>
    struct rebind_simd;

  template <class T, class V>
    using rebind_simd_t = typename rebind_simd<T, V>::type;

  template <class T, class U, class A>
    struct rebind_simd<T, simd<U, A>, std::true_type>
    { using type = simd<T, A>; };

  template <class T, class U, class A>
    struct rebind_simd<T, simd_mask<U, A>, std::true_type>
    { using type = simd_mask<T, A>; };

  template <int N, class V,
            class = typename std::conjunction<
                               detail::BoolConstant<(N > 0)>,
                               std::disjunction<is_simd<V>, is_simd_mask<V>>
                             >::type>
    struct resize_simd;

  template <int N, class V>
    using resize_simd_t = typename resize_simd<N, V>::type;

  template <int N, class T, class A>
    struct resize_simd<N, simd<T, A>, std::true_type>
    {
      using type = simd<T, std::conditional_t<N == 1, simd_abi::scalar, simd_abi::fixed_size<N>>>;
    };

  template <int N, class T, class A>
    struct resize_simd<N, simd_mask<T, A>, std::true_type>
    {
      using type = simd_mask<T, std::conditional_t<
                                  N == 1, simd_abi::scalar, simd_abi::fixed_size<N>>>;
    };

  // simd_mask (scalar)
  template <class T>
    class simd_mask<detail::Vectorizable<T>, simd_abi::scalar>
    : public detail::only_vectorizable<T>
    {
      bool data;

    public:
      using value_type = bool;
      using reference = bool&;
      using abi_type = simd_abi::scalar;
      using simd_type = simd<T, abi_type>;

      static constexpr size_t size() noexcept
      { return 1; }

      constexpr simd_mask() = default;
      constexpr simd_mask(const simd_mask&) = default;
      constexpr simd_mask(simd_mask&&) noexcept = default;
      constexpr simd_mask& operator=(const simd_mask&) = default;
      constexpr simd_mask& operator=(simd_mask&&) noexcept = default;

      // explicit broadcast constructor
      explicit constexpr
      simd_mask(bool x)
      : data(x) {}

      template <typename F>
        explicit constexpr
        simd_mask(F&& gen, std::enable_if_t<
                             std::is_same_v<decltype(std::declval<F>()(detail::SizeConstant<0>())),
                                            value_type>>* = nullptr)
        : data(gen(detail::SizeConstant<0>()))
        {}

      // load constructor
      template <typename Flags>
        simd_mask(const value_type* mem, Flags)
        : data(mem[0])
        {}

      template <typename Flags>
        simd_mask(const value_type* mem, simd_mask k, Flags)
        : data(k ? mem[0] : false)
        {}

      // loads [simd_mask.load]
      template <typename Flags>
        void
        copy_from(const value_type* mem, Flags)
        { data = mem[0]; }

      // stores [simd_mask.store]
      template <typename Flags>
        void
        copy_to(value_type* mem, Flags) const
        { mem[0] = data; }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      // negation
      constexpr simd_mask
      operator!() const
      { return simd_mask(not data); }

      // simd_mask binary operators [simd_mask.binary]
      friend constexpr simd_mask
      operator&&(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data && y.data); }

      friend constexpr simd_mask
      operator||(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data || y.data); }

      friend constexpr simd_mask
      operator&(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data & y.data); }

      friend constexpr simd_mask
      operator|(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data | y.data); }

      friend constexpr simd_mask
      operator^(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data ^ y.data); }

      friend constexpr simd_mask&
      operator&=(simd_mask& x, const simd_mask& y)
      {
        x.data &= y.data;
        return x;
      }

      friend constexpr simd_mask&
      operator|=(simd_mask& x, const simd_mask& y)
      {
        x.data |= y.data;
        return x;
      }

      friend constexpr simd_mask&
      operator^=(simd_mask& x, const simd_mask& y)
      {
        x.data ^= y.data;
        return x;
      }

      // simd_mask compares [simd_mask.comparison]
      friend constexpr simd_mask
      operator==(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data == y.data); }

      friend constexpr simd_mask
      operator!=(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data != y.data); }
    };

  // simd_mask (fixed_size)
  template <class T, int N>
    class simd_mask<detail::Vectorizable<T>, simd_abi::fixed_size<N>>
    : public detail::only_vectorizable<T>
    {
    private:
      template <typename V, int M, size_t Parts>
        friend std::enable_if_t<M == Parts * V::size() && is_simd_mask_v<V>, std::array<V, Parts>>
        split(const simd_mask<typename V::simd_type::value_type, simd_abi::fixed_size<M>>&);

      bool data[N];

      template <typename F, size_t... Is>
        constexpr
        simd_mask(std::index_sequence<Is...>, F&& init)
        : data {init(detail::SizeConstant<Is>())...}
        {}

    public:
      using value_type = bool;
      using reference = bool&;
      using abi_type = simd_abi::fixed_size<N>;
      using simd_type = simd<T, abi_type>;

      static constexpr size_t size() noexcept
      { return N; }

      constexpr simd_mask() = default;
      constexpr simd_mask(const simd_mask&) = default;
      constexpr simd_mask(simd_mask&&) noexcept = default;
      constexpr simd_mask& operator=(const simd_mask&) = default;
      constexpr simd_mask& operator=(simd_mask&&) noexcept = default;

      // explicit broadcast constructor
      explicit constexpr
      simd_mask(bool x)
      : simd_mask([x](size_t) { return x; })
      {}

      template <typename F>
        explicit constexpr
        simd_mask(F&& gen, std::enable_if_t<
                             std::is_same_v<decltype(std::declval<F>()(detail::SizeConstant<0>())),
                                            value_type>>* = nullptr)
        : simd_mask(std::make_index_sequence<N>(), std::forward<F>(gen))
        {}

      // implicit conversions
      template <typename U>
        constexpr
        simd_mask(const simd_mask<U, abi_type>& x)
        : simd_mask([&x](auto i) { return x[i]; })
        {}

      // load constructor
      template <typename Flags>
        simd_mask(const value_type* mem, Flags)
        : simd_mask([mem](size_t i) { return mem[i]; })
        {}

      template <typename Flags>
        simd_mask(const value_type* mem, const simd_mask& k, Flags)
        : simd_mask([mem, &k](size_t i) { return k[i] ? mem[i] : false; })
        {}

      // loads [simd_mask.load]
      template <typename Flags>
        void
        copy_from(const value_type* mem, Flags)
        { std::memcpy(data, mem, N * sizeof(bool)); }

      // stores [simd_mask.store]
      template <typename Flags>
        void
        copy_to(value_type* mem, Flags) const
        { std::memcpy(mem, data, N * sizeof(bool)); }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      // negation
      constexpr simd_mask
      operator!() const
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = !data[i];
        return r;
      }

      // simd_mask binary operators [simd_mask.binary]
      friend constexpr simd_mask
      operator&&(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] & y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator||(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] | y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator&(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] & y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator|(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] | y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator^(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] ^ y.data[i];
        return r;
      }

      friend constexpr simd_mask&
      operator&=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] &= y.data[i];
        return x;
      }

      friend constexpr simd_mask&
      operator|=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] |= y.data[i];
        return x;
      }

      friend constexpr simd_mask&
      operator^=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] ^= y.data[i];
        return x;
      }

      // simd_mask compares [simd_mask.comparison]
      friend constexpr simd_mask
      operator==(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] == y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator!=(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] != y.data[i];
        return r;
      }
    };

  // simd_mask reductions [simd_mask.reductions]
  template <typename T>
    constexpr bool
    all_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return k[0]; }

  template <typename T>
    constexpr bool
    any_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return k[0]; }

  template <typename T>
    constexpr bool
    none_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return not k[0]; }

  template <typename T>
    constexpr bool
    some_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return false; }

  template <typename T>
    constexpr int
    popcount(simd_mask<T, simd_abi::scalar> k) noexcept
    { return static_cast<int>(k[0]); }

  template <typename T>
    constexpr int
    find_first_set(simd_mask<T, simd_abi::scalar> k) noexcept
    {
      if (not k[0])
        detail::invoke_ub("find_first_set(empty mask) is UB");
      return 0;
    }

  template <typename T>
    constexpr int
    find_last_set(simd_mask<T, simd_abi::scalar> k) noexcept
    {
      if (not k[0])
        detail::invoke_ub("find_last_set(empty mask) is UB");
      return 0;
    }

  template <typename T, int N>
    constexpr bool
    all_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (not k[i])
            return false;
        }
      return true;
    }

  template <typename T, int N>
    constexpr bool
    any_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return true;
        }
      return false;
    }

  template <typename T, int N>
    constexpr bool
    none_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return false;
        }
      return true;
    }

  template <typename T, int N>
    constexpr bool
    some_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      bool last = k[0];
      for (int i = 1; i < N; ++i)
        {
          if (last != k[i])
            return true;
        }
      return false;
    }

  template <typename T, int N>
    constexpr int
    popcount(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      int cnt = k[0];
      for (int i = 1; i < N; ++i)
        cnt += k[i];
      return cnt;
    }

  template <typename T, int N>
    constexpr int
    find_first_set(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return i;
        }
      detail::invoke_ub("find_first_set(empty mask) is UB");
    }

  template <typename T, int N>
    constexpr int
    find_last_set(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = N - 1; i >= 0; --i)
        {
          if (k[i])
            return i;
        }
      detail::invoke_ub("find_last_set(empty mask) is UB");
    }

  constexpr bool
  all_of(detail::ExactBool x) noexcept
  { return x; }

  constexpr bool
  any_of(detail::ExactBool x) noexcept
  { return x; }

  constexpr bool
  none_of(detail::ExactBool x) noexcept
  { return !x; }

  constexpr bool
  some_of(detail::ExactBool) noexcept
  { return false; }

  constexpr int
  popcount(detail::ExactBool x) noexcept
  { return x; }

  constexpr int
  find_first_set(detail::ExactBool)
  { return 0; }

  constexpr int
  find_last_set(detail::ExactBool)
  { return 0; }

  // scalar_simd_int_base
  template <class T, bool = std::is_integral_v<T>>
    class scalar_simd_int_base
    {};

  template <class T>
    class scalar_simd_int_base<T, true>
    {
      using Derived = simd<T, simd_abi::scalar>;

      constexpr T&
      d() noexcept
      { return static_cast<Derived*>(this)->data; }

      constexpr const T&
      d() const noexcept
      { return static_cast<const Derived*>(this)->data; }

    public:
      friend constexpr Derived&
      operator%=(Derived& lhs, Derived x)
      {
        lhs.d() %= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator&=(Derived& lhs, Derived x)
      {
        lhs.d() &= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator|=(Derived& lhs, Derived x)
      {
        lhs.d() |= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator^=(Derived& lhs, Derived x)
      {
        lhs.d() ^= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator<<=(Derived& lhs, Derived x)
      {
        lhs.d() <<= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator>>=(Derived& lhs, Derived x)
      {
        lhs.d() >>= x.d();
        return lhs;
      }

      friend constexpr Derived
      operator%(Derived x, Derived y)
      {
        x.d() %= y.d();
        return x;
      }

      friend constexpr Derived
      operator&(Derived x, Derived y)
      {
        x.d() &= y.d();
        return x;
      }

      friend constexpr Derived
      operator|(Derived x, Derived y)
      {
        x.d() |= y.d();
        return x;
      }

      friend constexpr Derived
      operator^(Derived x, Derived y)
      {
        x.d() ^= y.d();
        return x;
      }

      friend constexpr Derived
      operator<<(Derived x, Derived y)
      {
        x.d() <<= y.d();
        return x;
      }

      friend constexpr Derived
      operator>>(Derived x, Derived y)
      {
        x.d() >>= y.d();
        return x;
      }

      friend constexpr Derived
      operator<<(Derived x, int y)
      {
        x.d() <<= y;
        return x;
      }

      friend constexpr Derived
      operator>>(Derived x, int y)
      {
        x.d() >>= y;
        return x;
      }

      constexpr Derived
      operator~() const
      { return Derived(static_cast<T>(~d())); }
    };

  // simd (scalar)
  template <class T>
    class simd<T, simd_abi::scalar>
    : public scalar_simd_int_base<T>, public detail::only_vectorizable<T>
    {
      friend class scalar_simd_int_base<T>;

      T data;

    public:
      using value_type = T;
      using reference = T&;
      using abi_type = simd_abi::scalar;
      using mask_type = simd_mask<T, abi_type>;

      static constexpr size_t size() noexcept
      { return 1; }

      constexpr simd() = default;
      constexpr simd(const simd&) = default;
      constexpr simd(simd&&) noexcept = default;
      constexpr simd& operator=(const simd&) = default;
      constexpr simd& operator=(simd&&) noexcept = default;

      // simd constructors
      template <typename U>
        constexpr
        simd(detail::ValuePreservingOrInt<U, value_type>&& value) noexcept
        : data(value)
        {}

      // generator constructor
      template <typename F>
        explicit constexpr
        simd(F&& gen, detail::ValuePreservingOrInt<
                        decltype(std::declval<F>()(std::declval<detail::SizeConstant<0>&>())),
                        value_type>* = nullptr)
        : data(gen(detail::SizeConstant<0>()))
        {}

      // load constructor
      template <typename U, typename Flags>
        simd(const U* mem, Flags)
        : data(mem[0])
        {}

      // loads [simd.load]
      template <typename U, typename Flags>
        void
        copy_from(const detail::Vectorizable<U>* mem, Flags)
        { data = mem[0]; }

      // stores [simd.store]
      template <typename U, typename Flags>
        void
        copy_to(detail::Vectorizable<U>* mem, Flags) const
        { mem[0] = data; }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      // increment and decrement:
      constexpr simd&
      operator++()
      {
        ++data;
        return *this;
      }

      constexpr simd
      operator++(int)
      {
        simd r = *this;
        ++data;
        return r;
      }

      constexpr simd&
      operator--()
      {
        --data;
        return *this;
      }

      constexpr simd
      operator--(int)
      {
        simd r = *this;
        --data;
        return r;
      }

      // unary operators
      constexpr mask_type
      operator!() const
      { return mask_type(not data); }

      constexpr simd
      operator+() const
      { return *this; }

      constexpr simd
      operator-() const
      { return -data; }

      // compound assignment [simd.cassign]
      constexpr friend simd&
      operator+=(simd& lhs, const simd& x)
      { return lhs = lhs + x; }

      constexpr friend simd&
      operator-=(simd& lhs, const simd& x)
      { return lhs = lhs - x; }

      constexpr friend simd&
      operator*=(simd& lhs, const simd& x)
      { return lhs = lhs * x; }

      constexpr friend simd&
        operator/=(simd& lhs, const simd& x)
      { return lhs = lhs / x; }

      // binary operators [simd.binary]
      constexpr friend simd
      operator+(const simd& x, const simd& y)
      { simd r = x; r.data += y.data; return r; }

      constexpr friend simd
      operator-(const simd& x, const simd& y)
      { simd r = x; r.data -= y.data; return r; }

      constexpr friend simd
      operator*(const simd& x, const simd& y)
      { simd r = x; r.data *= y.data; return r; }

      constexpr friend simd
      operator/(const simd& x, const simd& y)
      { simd r = x; r.data /= y.data; return r; }

      // compares [simd.comparison]
      constexpr friend mask_type
      operator==(const simd& x, const simd& y)
      { return mask_type(x.data == y.data); }

      constexpr friend mask_type
      operator!=(const simd& x, const simd& y)
      { return mask_type(x.data != y.data); }

      constexpr friend mask_type
      operator<(const simd& x, const simd& y)
      { return mask_type(x.data < y.data); }

      constexpr friend mask_type
      operator<=(const simd& x, const simd& y)
      { return mask_type(x.data <= y.data); }

      constexpr friend mask_type
      operator>(const simd& x, const simd& y)
      { return mask_type(x.data > y.data); }

      constexpr friend mask_type
      operator>=(const simd& x, const simd& y)
      { return mask_type(x.data >= y.data); }
    };

  // fixed_simd_int_base
  template <class T, int N, bool = std::is_integral_v<T>>
    class fixed_simd_int_base
    {};

  template <class T, int N>
    class fixed_simd_int_base<T, N, true>
    {
      using Derived = simd<T, simd_abi::fixed_size<N>>;

      constexpr T&
      d(int i) noexcept
      { return static_cast<Derived*>(this)->data[i]; }

      constexpr const T&
      d(int i) const noexcept
      { return static_cast<const Derived*>(this)->data[i]; }

    public:
      friend constexpr Derived&
      operator%=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) %= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator&=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) &= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator|=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) |= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator^=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) ^= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator<<=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) <<= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator>>=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) >>= x.d(i);
        return lhs;
      }

      friend constexpr Derived
      operator%(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] % y[i]; }); }

      friend constexpr Derived
      operator&(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] & y[i]; }); }

      friend constexpr Derived
      operator|(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] | y[i]; }); }

      friend constexpr Derived
      operator^(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] ^ y[i]; }); }

      friend constexpr Derived
      operator<<(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] << y[i]; }); }

      friend constexpr Derived
      operator>>(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] >> y[i]; }); }

      friend constexpr Derived
      operator<<(const Derived& x, int y)
      { return Derived([&](auto i) -> T { return x[i] << y; }); }

      friend constexpr Derived
      operator>>(const Derived& x, int y)
      { return Derived([&](auto i) -> T { return x[i] >> y; }); }

      constexpr Derived
      operator~() const
      { return Derived([&](auto i) -> T { return ~d(i); }); }
    };

  // simd (fixed_size)
  template <class T, int N>
    class simd<T, simd_abi::fixed_size<N>>
    : public fixed_simd_int_base<T, N>, public detail::only_vectorizable<T>
    {
    private:
      friend class fixed_simd_int_base<T, N>;

      template <typename V, int M, size_t Parts>
        friend std::enable_if_t<M == Parts * V::size() && is_simd_v<V>, std::array<V, Parts>>
        split(const simd<typename V::value_type, simd_abi::fixed_size<M>>&);

      template <size_t... Sizes, typename U>
        friend std::tuple<simd<U, simd_abi::deduce_t<U, int(Sizes)>>...>
        split(const simd<U, simd_abi::fixed_size<int((Sizes + ...))>>&);

      T data[N];

      template <typename F, size_t... Is>
        constexpr
        simd(std::index_sequence<Is...>, F&& init)
        : data {static_cast<value_type>(init(detail::SizeConstant<Is>()))...}
        {}

    public:
      using value_type = T;
      using reference = T&;
      using abi_type = simd_abi::fixed_size<N>;
      using mask_type = simd_mask<T, abi_type>;

      static constexpr size_t size() noexcept
      { return N; }

      constexpr simd() = default;
      constexpr simd(const simd&) = default;
      constexpr simd(simd&&) noexcept = default;
      constexpr simd& operator=(const simd&) = default;
      constexpr simd& operator=(simd&&) noexcept = default;

      // simd constructors
      template <typename U>
        constexpr
        simd(detail::ValuePreservingOrInt<U, value_type>&& value) noexcept
        : simd([v = static_cast<value_type>(value)](size_t) { return v; })
        {}

      // conversion constructors
      template <typename U,
                typename = std::enable_if_t<
                             std::conjunction_v<detail::is_value_preserving<U, value_type>,
                                                detail::is_higher_integer_rank<value_type, U>>>>
        constexpr
        simd(const simd<U, abi_type>& x)
        : simd([&x](auto i) { return static_cast<value_type>(x[i]); })
        {}

      // generator constructor
      template <typename F>
        explicit constexpr
        simd(F&& gen, detail::ValuePreservingOrInt<
                        decltype(std::declval<F>()(std::declval<detail::SizeConstant<0>&>())),
                        value_type>* = nullptr)
        : simd(std::make_index_sequence<N>(), std::forward<F>(gen))
        {}

      // load constructor
      template <typename U, typename Flags>
        simd(const U* mem, Flags)
        : simd([mem](auto i) -> value_type { return mem[i]; })
        {}

      // loads [simd.load]
      template <typename U, typename Flags>
        void
        copy_from(const detail::Vectorizable<U>* mem, Flags)
        {
          for (int i = 0; i < N; ++i)
            data[i] = mem[i];
        }

      // stores [simd.store]
      template <typename U, typename Flags>
        void
        copy_to(detail::Vectorizable<U>* mem, Flags) const
        {
          for (int i = 0; i < N; ++i)
            mem[i] = data[i];
        }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      // increment and decrement:
      constexpr simd&
      operator++()
      {
        for (int i = 0; i < N; ++i)
          ++data[i];
        return *this;
      }

      constexpr simd
      operator++(int)
      {
        simd r = *this;
        for (int i = 0; i < N; ++i)
          ++data[i];
        return r;
      }

      constexpr simd&
      operator--()
      {
        for (int i = 0; i < N; ++i)
          --data[i];
        return *this;
      }

      constexpr simd
      operator--(int)
      {
        simd r = *this;
        for (int i = 0; i < N; ++i)
          --data[i];
        return r;
      }

      // unary operators
      constexpr mask_type
      operator!() const
      { return mask_type([&](auto i) { return !data[i]; }); }

      constexpr simd
      operator+() const
      { return *this; }

      constexpr simd
      operator-() const
      { return simd([&](auto i) -> value_type { return -data[i]; }); }

      // compound assignment [simd.cassign]
      constexpr friend simd&
      operator+=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] += x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator-=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] -= x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator*=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] *= x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator/=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] /= x.data[i];
        return lhs;
      }

      // binary operators [simd.binary]
      constexpr friend simd
      operator+(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] + y.data[i]; }); }

      constexpr friend simd
      operator-(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] - y.data[i]; }); }

      constexpr friend simd
      operator*(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] * y.data[i]; }); }

      constexpr friend simd
      operator/(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] / y.data[i]; }); }

      // compares [simd.comparison]
      constexpr friend mask_type
      operator==(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] == y.data[i]; }); }

      constexpr friend mask_type
      operator!=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] != y.data[i]; }); }

      constexpr friend mask_type
      operator<(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] < y.data[i]; }); }

      constexpr friend mask_type
      operator<=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] <= y.data[i]; }); }

      constexpr friend mask_type
      operator>(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] > y.data[i]; }); }

      constexpr friend mask_type
      operator>=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] >= y.data[i]; }); }
    };

  // casts [simd.casts]
  // static_simd_cast
  template <typename T, typename U, typename A,
            typename = std::enable_if_t<detail::is_vectorizable_v<T>>>
    constexpr simd<T, A>
    static_simd_cast(const simd<U, A>& x)
    { return simd<T, A>([&x](auto i) { return static_cast<T>(x[i]); }); }

  template <typename V, typename U, typename A,
            typename = std::enable_if_t<is_simd_v<V>>>
    constexpr V
    static_simd_cast(const simd<U, A>& x)
    { return V([&x](auto i) { return static_cast<typename V::value_type>(x[i]); }); }

  template <typename T, typename U, typename A,
            typename = std::enable_if_t<detail::is_vectorizable_v<T>>>
    constexpr simd_mask<T, A>
    static_simd_cast(const simd_mask<U, A>& x)
    { return simd_mask<T, A>([&x](auto i) { return x[i]; }); }

  template <typename M, typename U, typename A,
            typename = std::enable_if_t<M::size() == simd_size_v<U, A>>>
    constexpr M
    static_simd_cast(const simd_mask<U, A>& x)
    { return M([&x](auto i) { return x[i]; }); }

  // simd_cast
  template <typename T, typename U, typename A,
            typename To = detail::value_type_or_identity_t<T>>
    constexpr auto
    simd_cast(const simd<detail::ValuePreserving<U, To>, A>& x)
    -> decltype(static_simd_cast<T>(x))
    { return static_simd_cast<T>(x); }

  // to_fixed_size
  template <typename T, int N>
    constexpr fixed_size_simd<T, N>
    to_fixed_size(const fixed_size_simd<T, N>& x)
    { return x; }

  template <typename T, int N>
    constexpr fixed_size_simd_mask<T, N>
    to_fixed_size(const fixed_size_simd_mask<T, N>& x)
    { return x; }

  template <typename T>
    constexpr fixed_size_simd<T, 1>
    to_fixed_size(const simd<T> x)
    { return x[0]; }

  template <typename T>
    constexpr fixed_size_simd_mask<T, 1>
    to_fixed_size(const simd_mask<T> x)
    { return fixed_size_simd_mask<T, 1>(x[0]); }

  // to_native
  template <typename T>
    constexpr simd<T>
    to_native(const fixed_size_simd<T, 1> x)
    { return x[0]; }

  template <typename T>
    constexpr simd_mask<T>
    to_native(const fixed_size_simd_mask<T, 1> x)
    { return simd_mask<T>(x[0]); }

  // to_compatible
  template <typename T>
    constexpr simd<T>
    to_compatible(const fixed_size_simd<T, 1> x)
    { return x[0]; }

  template <typename T>
    constexpr simd_mask<T>
    to_compatible(const fixed_size_simd_mask<T, 1> x)
    { return simd_mask<T>(x[0]); }

  // split(simd)
  template <typename V, int N, size_t Parts = N / V::size()>
    std::enable_if_t<N == Parts * V::size() && is_simd_v<V>, std::array<V, Parts>>
    split(const simd<typename V::value_type, simd_abi::fixed_size<N>>& x)
    {
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>)
               -> std::array<V, Parts> {
                 return {V(data + Is * V::size(), element_aligned)...};
               }(std::make_index_sequence<Parts>());
    }

  // split(simd_mask)
  template <typename V, int N, size_t Parts = N / V::size()>
    std::enable_if_t<N == Parts * V::size() && is_simd_mask_v<V>, std::array<V, Parts>>
    split(const simd_mask<typename V::simd_type::value_type, simd_abi::fixed_size<N>>& x)
    {
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>)
               -> std::array<V, Parts> {
                 return {V(data + Is * V::size(), element_aligned)...};
               }(std::make_index_sequence<Parts>());
    }

  // split<Sizes...>
  template <size_t... Sizes, typename T>
    std::tuple<simd<T, simd_abi::deduce_t<T, int(Sizes)>>...>
    split(const simd<T, simd_abi::fixed_size<int((Sizes + ...))>>& x)
    {
      using R = std::tuple<simd<T, simd_abi::deduce_t<T, int(Sizes)>>...>;
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>) -> R {
        constexpr size_t offsets[sizeof...(Sizes)] = {
          []<size_t... Js>(std::index_sequence<Js...>) {
            constexpr size_t sizes[sizeof...(Sizes)] = {Sizes...};
            return (sizes[Js] + ... + 0);
          }(std::make_index_sequence<Is>())...
        };
        return {simd<T, simd_abi::deduce_t<T, int(Sizes)>>(data + offsets[Is],
                                                           element_aligned)...};
      }(std::make_index_sequence<sizeof...(Sizes)>());
    }

  // split<V>(V)
  template <typename V>
    std::enable_if_t<std::disjunction_v<is_simd<V>, is_simd_mask<V>>, std::array<V, 1>>
    split(const V& x)
    { return {x}; }

  // concat(simd...)
  template <typename T, typename... As>
    inline constexpr
    simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>
    concat(const simd<T, As>&... xs)
    {
      using R = simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>;
      return R([&](auto i) {
               return detail::pack_simd_subscript<i>(xs...);
             });
    }

  // concat(simd_mask...)
  template <typename T, typename... As>
    inline constexpr
    simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>
    concat(const simd_mask<T, As>&... xs)
    {
      using R = simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>;
      return R([&](auto i) -> bool {
               return detail::pack_simd_subscript<i>(xs...);
             });
    }

  // concat(array<simd>)
  template <typename T, typename A, size_t N>
    inline constexpr
    simd<T, simd_abi::deduce_t<T, N * simd_size_v<T, A>>>
    concat(const std::array<simd<T, A>, N>& x)
    {
      constexpr int K = simd_size_v<T, A>;
      using R = simd<T, simd_abi::deduce_t<T, N * K>>;
      return R([&](auto i) {
               return x[i / K][i % K];
             });
    }

  // concat(array<simd_mask>)
  template <typename T, typename A, size_t N>
    inline constexpr
    simd_mask<T, simd_abi::deduce_t<T, N * simd_size_v<T, A>>>
    concat(const std::array<simd_mask<T, A>, N>& x)
    {
      constexpr int K = simd_size_v<T, A>;
      using R = simd_mask<T, simd_abi::deduce_t<T, N * K>>;
      return R([&](auto i) -> bool {
               return x[i / K][i % K];
             });
    }

  // const_where_expression<M, T>
  template <typename M, typename V>
    class const_where_expression
    {
      static_assert(std::is_same_v<V, detail::remove_cvref_t<V>>);

      struct Wrapper { using value_type = V; };

    protected:
      using value_type =
        typename std::conditional_t<std::is_arithmetic_v<V>, Wrapper, V>::value_type;

      friend const M&
      get_mask(const const_where_expression& x)
      { return x.m_k; }

      friend const V&
      get_lvalue(const const_where_expression& x)
      { return x.m_value; }

      const M& m_k;
      V& m_value;

    public:
      const_where_expression(const const_where_expression&) = delete;
      const_where_expression& operator=(const const_where_expression&) = delete;

      constexpr const_where_expression(const M& kk, const V& dd)
      : m_k(kk), m_value(const_cast<V&>(dd)) {}

      constexpr V
      operator-() const &&
      {
        return V([&](auto i) {
                 return m_k[i] ? static_cast<value_type>(-m_value[i]) : m_value[i];
               });
      }

      template <typename Up, typename Flags>
        [[nodiscard]] constexpr V
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          return V([&](auto i) {
                   return m_k[i] ? static_cast<value_type>(mem[i]) : m_value[i];
                 });
        }

      template <typename Up, typename Flags>
        constexpr void
        copy_to(detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                mem[i] = static_cast<Up>(m_value[i]);
            }
        }
    };

  // const_where_expression<bool, T>
  template <typename V>
    class const_where_expression<bool, V>
    {
      using M = bool;

      static_assert(std::is_same_v<V, detail::remove_cvref_t<V>>);

      struct Wrapper { using value_type = V; };

    protected:
      using value_type =
        typename std::conditional_t<std::is_arithmetic_v<V>, Wrapper, V>::value_type;

      friend const M&
      get_mask(const const_where_expression& x)
      { return x.m_k; }

      friend const V&
      get_lvalue(const const_where_expression& x)
      { return x.m_value; }

      const bool m_k;
      V& m_value;

    public:
      const_where_expression(const const_where_expression&) = delete;
      const_where_expression& operator=(const const_where_expression&) = delete;

      constexpr const_where_expression(const bool kk, const V& dd)
      : m_k(kk), m_value(const_cast<V&>(dd)) {}

      constexpr V
      operator-() const &&
      { return m_k ? -m_value : m_value; }

      template <typename Up, typename Flags>
        [[nodiscard]] constexpr V
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        { return m_k ? static_cast<V>(mem[0]) : m_value; }

      template <typename Up, typename Flags>
        constexpr void
        copy_to(detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          if (m_k)
            mem[0] = m_value;
        }
    };

  // where_expression<M, T>
  template <typename M, typename V>
    class where_expression : public const_where_expression<M, V>
    {
      static_assert(not std::is_const_v<V>,
                    "where_expression may only be instantiated with a non-const V parameter");

      using typename const_where_expression<M, V>::value_type;
      using const_where_expression<M, V>::m_k;
      using const_where_expression<M, V>::m_value;

      static_assert(std::is_same_v<typename M::abi_type, typename V::abi_type>);
      static_assert(M::size() == V::size());

      friend V&
      get_lvalue(where_expression& x)
      { return x.m_value; }

      template <typename Up>
        constexpr auto
        as_simd(Up&& x)
        {
          using UU = detail::remove_cvref_t<Up>;
          if constexpr (std::is_same_v<V, UU>)
            return x;
          else if constexpr (std::is_convertible_v<Up&&, value_type>)
            return V(static_cast<value_type>(static_cast<Up&&>(x)));
          else if constexpr (std::is_convertible_v<Up&&, V>)
            return static_cast<V>(static_cast<Up&&>(x));
          else
            return static_simd_cast<V>(static_cast<Up&&>(x));
        }

    public:
      where_expression(const where_expression&) = delete;
      where_expression& operator=(const where_expression&) = delete;

      constexpr where_expression(const M& kk, V& dd)
      : const_where_expression<M, V>(kk, dd)
      {}

      template <typename Up>
        constexpr void
        operator=(Up&& x) &&
        {
          const V& rhs = as_simd(x);
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                m_value[i] = rhs[i];
            }
        }

#define SIMD_OP_(op)                              \
      template <typename Up>                      \
        constexpr void                            \
        operator op##=(Up&& x) &&                 \
        {                                         \
          const V& rhs = as_simd(x);              \
          for (size_t i = 0; i < V::size(); ++i)  \
            {                                     \
              if (m_k[i])                         \
                m_value[i] op##= rhs[i];          \
            }                                     \
        }                                         \
      static_assert(true)
      SIMD_OP_(+);
      SIMD_OP_(-);
      SIMD_OP_(*);
      SIMD_OP_(/);
      SIMD_OP_(%);
      SIMD_OP_(&);
      SIMD_OP_(|);
      SIMD_OP_(^);
      SIMD_OP_(<<);
      SIMD_OP_(>>);
#undef SIMD_OP_

      constexpr void operator++() &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              ++m_value[i];
          }
      }

      constexpr void operator++(int) &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              ++m_value[i];
          }
      }

      constexpr void operator--() &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              --m_value[i];
          }
      }

      constexpr void operator--(int) &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              --m_value[i];
          }
      }

      // intentionally hides const_where_expression::copy_from
      template <typename Up, typename Flags>
        constexpr void
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) &&
        {
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                m_value[i] = mem[i];
            }
        }
    };

  // where_expression<bool, T>
  template <typename V>
    class where_expression<bool, V> : public const_where_expression<bool, V>
    {
      using M = bool;
      using typename const_where_expression<M, V>::value_type;
      using const_where_expression<M, V>::m_k;
      using const_where_expression<M, V>::m_value;

    public:
      where_expression(const where_expression&) = delete;
      where_expression& operator=(const where_expression&) = delete;

      constexpr where_expression(const M& kk, V& dd)
      : const_where_expression<M, V>(kk, dd) {}

#define SIMD_OP_(op)                                \
      template <typename Up>                        \
        constexpr void operator op(Up&& x) &&       \
        { if (m_k) m_value op static_cast<Up&&>(x); }

      SIMD_OP_(=)
      SIMD_OP_(+=)
      SIMD_OP_(-=)
      SIMD_OP_(*=)
      SIMD_OP_(/=)
      SIMD_OP_(%=)
      SIMD_OP_(&=)
      SIMD_OP_(|=)
      SIMD_OP_(^=)
      SIMD_OP_(<<=)
      SIMD_OP_(>>=)
#undef SIMD_OP_

      constexpr void operator++() &&
      { if (m_k) ++m_value; }

      constexpr void operator++(int) &&
      { if (m_k) ++m_value; }

      constexpr void operator--() &&
      { if (m_k) --m_value; }

      constexpr void operator--(int) &&
      { if (m_k) --m_value; }

      // intentionally hides const_where_expression::copy_from
      template <typename Up, typename Flags>
        constexpr void
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) &&
        { if (m_k) m_value = mem[0]; }
    };

  // where
  template <typename Tp, typename Ap>
    constexpr where_expression<simd_mask<Tp, Ap>, simd<Tp, Ap>>
    where(const typename simd<Tp, Ap>::mask_type& k, simd<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr const_where_expression<simd_mask<Tp, Ap>, simd<Tp, Ap>>
    where(const typename simd<Tp, Ap>::mask_type& k,
          const simd<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr where_expression<simd_mask<Tp, Ap>, simd_mask<Tp, Ap>>
    where(const std::remove_const_t<simd_mask<Tp, Ap>>& k,
          simd_mask<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr const_where_expression<simd_mask<Tp, Ap>, simd_mask<Tp, Ap>>
    where(const std::remove_const_t<simd_mask<Tp, Ap>>& k,
          const simd_mask<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp>
    constexpr where_expression<bool, Tp>
    where(detail::ExactBool k, Tp& value)
    { return {k, value}; }

  template <typename Tp>
    constexpr const_where_expression<bool, Tp>
    where(detail::ExactBool k, const Tp& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr void
    where(bool k, simd<Tp, Ap>& value) = delete;

  template <typename Tp, typename Ap>
    constexpr void
    where(bool k, const simd<Tp, Ap>& value) = delete;

  // reductions [simd.reductions]
  template <typename T, typename A, typename BinaryOperation = std::plus<>>
    constexpr T
    reduce(const simd<T, A>& v,
           BinaryOperation binary_op = BinaryOperation())
    {
      constexpr int N = simd_size_v<T, A>;
      if constexpr (N > 3)
        {
          constexpr int N2 = detail::bit_floor(N / 2);
          constexpr int NRem = N - 2 * N2;
          if constexpr (NRem > 0)
            {
              const auto [l, r, rem] = split<N2, N2, N - 2 * N2>(v);
              return binary_op(reduce(binary_op(l, r), binary_op), reduce(rem, binary_op));
            }
          else
            {
              const auto [l, r] = split<N2, N2>(v);
              return reduce(binary_op(l, r), binary_op);
            }
        }
      else
        {
          T r = v[0];
          for (size_t i = 1; i < simd_size_v<T, A>; ++i)
            r = binary_op(r, v[i]);
          return r;
        }
    }

  template <typename M, typename V, typename BinaryOperation = std::plus<>>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x,
        typename V::value_type identity_element,
        BinaryOperation binary_op)
    {
      const M& k = get_mask(x);
      const V& v = get_lvalue(x);
      auto r = identity_element;
      if (any_of(k)) [[likely]]
        {
          for (size_t i = 0; i < V::size(); ++i)
            if (k[i])
              r = binary_op(r, v[i]);
        }
      return r;
    }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::plus<> binary_op = {})
    { return reduce(x, 0, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::multiplies<> binary_op)
    { return reduce(x, 1, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_and<> binary_op)
    { return reduce(x, ~typename V::value_type(), binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_or<> binary_op)
    { return reduce(x, 0, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_xor<> binary_op)
    { return reduce(x, 0, binary_op); }

  template <typename T, typename A>
    constexpr T
    hmin(const simd<T, A>& v) noexcept
    {
      return reduce(v, [](const auto& l, const auto& r) {
               using std::min;
               return min(l, r);
             });
    }

  template <typename T, typename A>
    constexpr T
    hmax(const simd<T, A>& v) noexcept
    {
      return reduce(v, [](const auto& l, const auto& r) {
               using std::max;
               return max(l, r);
             });
    }

  template <typename M, typename V>
    constexpr typename V::value_type
    hmin(const const_where_expression<M, V>& x) noexcept
    {
      using T = typename V::value_type;
      constexpr T id_elem =
#ifdef __FINITE_MATH_ONLY__
        std::numeric_limits<T>::max();
#else
        std::numeric_limits<T>::infinity();
#endif
      return reduce(x, id_elem, [](const auto& l, const auto& r) {
               using std::min;
               return min(l, r);
             });
    }

  template <typename M, typename V>
    constexpr
    typename V::value_type
    hmax(const const_where_expression<M, V>& x) noexcept
    {
      using T = typename V::value_type;
      constexpr T id_elem =
#ifdef __FINITE_MATH_ONLY__
        std::numeric_limits<T>::lowest();
#else
        -std::numeric_limits<T>::infinity();
#endif
      return reduce(x, id_elem, [](const auto& l, const auto& r) {
               using std::max;
               return max(l, r);
             });
    }

  // algorithms [simd.alg]
  template <typename T, typename A>
    constexpr simd<T, A>
    min(const simd<T, A>& a, const simd<T, A>& b)
    { return simd<T, A>([&](auto i) { return std::min(a[i], b[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    max(const simd<T, A>& a, const simd<T, A>& b)
    { return simd<T, A>([&](auto i) { return std::max(a[i], b[i]); }); }

  template <typename T, typename A>
    constexpr
    std::pair<simd<T, A>, simd<T, A>>
    minmax(const simd<T, A>& a, const simd<T, A>& b)
    { return {min(a, b), max(a, b)}; }

  template <typename T, typename A>
    constexpr simd<T, A>
    clamp(const simd<T, A>& v, const simd<T, A>& lo,
        const simd<T, A>& hi)
    { return simd<T, A>([&](auto i) { return std::clamp(v[i], lo[i], hi[i]); }); }

  // math
#define SIMD_MATH_1ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x) noexcept                                      \
    { return return_temp<T, A>([&x](auto i) { return std::name(x[i]); }); }

#define SIMD_MATH_1ARG_FIXED(name, R)                                                              \
  template <typename T, typename A>                                                                \
    constexpr fixed_size_simd<R, simd_size_v<T, A>>                                                \
    name(const simd<detail::FloatingPoint<T>, A>& x) noexcept                                      \
    { return fixed_size_simd<R, simd_size_v<T, A>>([&x](auto i) { return std::name(x[i]); }); }

#define SIMD_MATH_2ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x, const simd<T, A>& y) noexcept                 \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }                   \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const detail::type_identity_t<simd<T, A>>& y) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }                   \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const simd<detail::FloatingPoint<T>, A>& y) noexcept                                      \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }

#define SIMD_MATH_3ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const simd<T, A>& y, const simd<T, A> &z) noexcept                                        \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const detail::type_identity_t<simd<T, A>>& y,                                             \
         const detail::type_identity_t<simd<T, A>> &z) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const simd<detail::FloatingPoint<T>, A>& y,                                               \
         const detail::type_identity_t<simd<T, A>> &z) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const detail::type_identity_t<simd<T, A>>& y,                                             \
         const simd<detail::FloatingPoint<T>, A> &z) noexcept                                      \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }

  template <typename T, typename A, typename U = detail::SignedIntegral<T>>
    constexpr simd<T, A>
    abs(const simd<T, A>& x) noexcept
    { return simd<T, A>([&x](auto i) { return std::abs(x[i]); }); }

  SIMD_MATH_1ARG(abs, simd)
  SIMD_MATH_1ARG(isnan, simd_mask)
  SIMD_MATH_1ARG(isfinite, simd_mask)
  SIMD_MATH_1ARG(isinf, simd_mask)
  SIMD_MATH_1ARG(isnormal, simd_mask)
  SIMD_MATH_1ARG(signbit, simd_mask)
  SIMD_MATH_1ARG_FIXED(fpclassify, int)

  SIMD_MATH_2ARG(hypot, simd)
  SIMD_MATH_3ARG(hypot, simd)

  template <typename T, typename A>
    constexpr simd<T, A>
    remquo(const simd<T, A>& x, const simd<T, A>& y,
           fixed_size_simd<int, simd_size_v<T, A>>* quo) noexcept
    { return simd<T, A>([&x, &y, quo](auto i) { return std::remquo(x[i], y[i], &(*quo)[i]); }); }

  SIMD_MATH_1ARG(erf, simd)
  SIMD_MATH_1ARG(erfc, simd)
  SIMD_MATH_1ARG(tgamma, simd)
  SIMD_MATH_1ARG(lgamma, simd)

  SIMD_MATH_2ARG(pow, simd)
  SIMD_MATH_2ARG(fmod, simd)
  SIMD_MATH_2ARG(remainder, simd)
  SIMD_MATH_2ARG(nextafter, simd)
  SIMD_MATH_2ARG(copysign, simd)
  SIMD_MATH_2ARG(fdim, simd)
  SIMD_MATH_2ARG(fmax, simd)
  SIMD_MATH_2ARG(fmin, simd)
  SIMD_MATH_2ARG(isgreater, simd_mask)
  SIMD_MATH_2ARG(isgreaterequal, simd_mask)
  SIMD_MATH_2ARG(isless, simd_mask)
  SIMD_MATH_2ARG(islessequal, simd_mask)
  SIMD_MATH_2ARG(islessgreater, simd_mask)
  SIMD_MATH_2ARG(isunordered, simd_mask)

  template <typename T, typename A>
    constexpr simd<T, A>
    modf(const simd<detail::FloatingPoint<T>, A>& x, simd<T, A>* iptr) noexcept
    { return simd<T, A>([&x, iptr](auto i) { return std::modf(x[i], &(*iptr)[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    frexp(const simd<detail::FloatingPoint<T>, A>& x,
          fixed_size_simd<int, simd_size_v<T, A>>* exp) noexcept
    { return simd<T, A>([&x, exp](auto i) { return std::frexp(x[i], &(*exp)[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    scalbln(const simd<detail::FloatingPoint<T>, A>& x,
            const fixed_size_simd<long int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::scalbln(x[i], exp[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    scalbn(const simd<detail::FloatingPoint<T>, A>& x,
           const fixed_size_simd<int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::scalbn(x[i], exp[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    ldexp(const simd<detail::FloatingPoint<T>, A>& x,
          const fixed_size_simd<int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::ldexp(x[i], exp[i]); }); }

  SIMD_MATH_1ARG(sqrt, simd)

  SIMD_MATH_3ARG(fma, simd)

  SIMD_MATH_1ARG(trunc, simd)
  SIMD_MATH_1ARG(ceil, simd)
  SIMD_MATH_1ARG(floor, simd)
  SIMD_MATH_1ARG(round, simd)
  SIMD_MATH_1ARG_FIXED(lround, long)
  SIMD_MATH_1ARG_FIXED(llround, long long)
  SIMD_MATH_1ARG(nearbyint, simd)
  SIMD_MATH_1ARG(rint, simd)
  SIMD_MATH_1ARG_FIXED(lrint, long)
  SIMD_MATH_1ARG_FIXED(llrint, long long)
  SIMD_MATH_1ARG_FIXED(ilogb, int)

  // trig functions
  SIMD_MATH_1ARG(sin, simd)
  SIMD_MATH_1ARG(cos, simd)
  SIMD_MATH_1ARG(tan, simd)
  SIMD_MATH_1ARG(asin, simd)
  SIMD_MATH_1ARG(acos, simd)
  SIMD_MATH_1ARG(atan, simd)
  SIMD_MATH_2ARG(atan2, simd)
  SIMD_MATH_1ARG(sinh, simd)
  SIMD_MATH_1ARG(cosh, simd)
  SIMD_MATH_1ARG(tanh, simd)
  SIMD_MATH_1ARG(asinh, simd)
  SIMD_MATH_1ARG(acosh, simd)
  SIMD_MATH_1ARG(atanh, simd)

  // logarithms
  SIMD_MATH_1ARG(log, simd)
  SIMD_MATH_1ARG(log10, simd)
  SIMD_MATH_1ARG(log1p, simd)
  SIMD_MATH_1ARG(log2, simd)
  SIMD_MATH_1ARG(logb, simd)

#undef SIMD_MATH_1ARG
#undef SIMD_MATH_1ARG_FIXED
#undef SIMD_MATH_2ARG
#undef SIMD_MATH_3ARG
}
#ifdef VIR_SIMD_TS_DROPIN
}

namespace vir::stdx
{
  using namespace std::experimental::parallelism_v2;
}
#endif

#endif
#endif  // VIR_SIMD_H_


#ifndef __EMSCRIPTEN__
#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
#endif

#ifndef DISABLE_SIMD
#define DISABLE_SIMD 0
#endif

namespace fair::literals {
    // C++23 has literal suffixes for std::size_t, but we are not
    // in C++23 just yet
    constexpr std::size_t operator"" _UZ(unsigned long long n) {
        return static_cast<std::size_t>(n);
    }
}

namespace fair::meta {

using namespace fair::literals;

template<typename... Ts>
struct print_types;

template<typename CharT, std::size_t SIZE>
struct fixed_string {
    constexpr static std::size_t N            = SIZE;
    CharT                        _data[N + 1] = {};

    constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
        if constexpr (N != 0)
            for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
    }

    [[nodiscard]] constexpr std::size_t
    size() const noexcept {
        return N;
    }

    [[nodiscard]] constexpr bool
    empty() const noexcept {
        return N == 0;
    }

    [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return { _data, N }; }

    [[nodiscard]] explicit           operator std::string() const noexcept { return { _data, N }; }

    [[nodiscard]]                    operator const char *() const noexcept { return _data; }

    [[nodiscard]] constexpr bool
    operator==(const fixed_string &other) const noexcept {
        return std::string_view{ _data, N } == std::string_view(other);
    }

    template<std::size_t N2>
    [[nodiscard]] friend constexpr bool
    operator==(const fixed_string &, const fixed_string<CharT, N2> &) {
        return false;
    }
};

template<typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

template<typename T>
[[nodiscard]] std::string
type_name() noexcept {
#ifndef __EMSCRIPTEN__
    std::string type_name = typeid(T).name();
    int         status;
    char       *demangled_name = abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status);
    if (status == 0) {
        return demangled_name;
    } else {
        return typeid(T).name();
    }
#else
    return typeid(T).name(); // TODO: to be replaced by refl-cpp
#endif
}

template<fixed_string val>
struct message_type {};

template<class... T>
constexpr bool always_false = false;

constexpr std::size_t invalid_index = -1_UZ;

#if HAVE_SOURCE_LOCATION
[[gnu::always_inline]] inline void
precondition(bool cond, const std::source_location loc = std::source_location::current()) {
    struct handle {
        [[noreturn]] static void
        failure(std::source_location const &loc) {
            std::clog << "failed precondition in " << loc.file_name() << ':' << loc.line() << ':' << loc.column() << ": `" << loc.function_name() << "`\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure(loc);
}
#else
[[gnu::always_inline]] inline void
precondition(bool cond) {
    struct handle {
        [[noreturn]] static void
        failure() {
            std::clog << "failed precondition\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure();
}
#endif

/**
 * T is tuple-like if it implements std::tuple_size, std::tuple_element, and std::get.
 * Tuples with size 0 are excluded.
 */
template<typename T>
concept tuple_like = (std::tuple_size<T>::value > 0) && requires(T tup) {
    { std::get<0>(tup) } -> std::same_as<typename std::tuple_element_t<0, T> &>;
};

static_assert(!tuple_like<int>);
static_assert(!tuple_like<std::tuple<>>);
static_assert(tuple_like<std::tuple<int>>);
static_assert(tuple_like<std::tuple<int&>>);
static_assert(tuple_like<std::tuple<const int&>>);
static_assert(tuple_like<std::tuple<const int>>);
static_assert(!tuple_like<std::array<int, 0>>);
static_assert(tuple_like<std::array<int, 2>>);
static_assert(tuple_like<std::pair<int, short>>);

namespace stdx = vir::stdx;

template<typename V, typename T = void>
concept any_simd = stdx::is_simd_v<V> && (std::same_as<T, void> || std::same_as<T, typename V::value_type>);

template<typename V, typename T>
concept t_or_simd = std::same_as<V, T> || any_simd<V, T>;

template<typename T>
concept vectorizable_v = std::constructible_from<stdx::simd<T>>;

template<typename T>
using vectorizable = std::integral_constant<bool, vectorizable_v<T>>;

/**
 * Determines the SIMD width of the given structure. This can either be a stdx::simd object or a
 * tuple-like of stdx::simd (recursively). The latter requires that the SIMD width is homogeneous.
 * If T is not a simd (or tuple thereof), value is 0.
 */
template<typename T>
struct simdize_size : std::integral_constant<std::size_t, 0> {};

template<typename T, typename A>
struct simdize_size<stdx::simd<T, A>> : stdx::simd_size<T, A> {};

template<tuple_like Tup>
struct simdize_size<Tup> : simdize_size<std::tuple_element_t<0, Tup>> {
    static_assert([]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((simdize_size<std::tuple_element_t<0, Tup>>::value == simdize_size<std::tuple_element_t<Is, Tup>>::value)
                && ...);
    }(std::make_index_sequence<std::tuple_size_v<Tup>>()));
};

template<typename T>
inline constexpr std::size_t simdize_size_v = simdize_size<T>::value;

namespace detail {
/**
 * Shortcut to determine the stdx::simd specialization with the most efficient ABI tag for the
 * requested element type T and width N.
 */
template<typename T, std::size_t N>
using deduced_simd = stdx::simd<T, stdx::simd_abi::deduce_t<T, N>>;

template<typename T, std::size_t N>
struct simdize_impl;

template<vectorizable_v T, std::size_t N>
requires requires { typename stdx::native_simd<T>; }
struct simdize_impl<T, N> {
    using type = deduced_simd<T, N == 0 ? stdx::native_simd<T>::size() : N>;
};

template<std::size_t N>
struct simdize_impl<std::tuple<>, N> {
    using type = std::tuple<>;
};

template<tuple_like Tup, std::size_t N>
    requires requires { typename simdize_impl<std::tuple_element_t<0, Tup>, N>::type; }
struct simdize_impl<Tup, N> {
    static inline constexpr std::size_t size = N > 0 ? N : []<std::size_t... Is>(std::index_sequence<Is...>) constexpr {
        return std::max({ simdize_size_v<typename simdize_impl<std::tuple_element_t<Is, Tup>, 0>::type>... });
    }(std::make_index_sequence<std::tuple_size_v<Tup>>());

    using type = decltype([]<std::size_t... Is>(std::index_sequence<Is...>)
                                  -> std::tuple<typename simdize_impl<std::tuple_element_t<Is, Tup>, size>::type...> {
        return {};
    }(std::make_index_sequence<std::tuple_size_v<Tup>>()));
};
} // namespace detail

/**
 * Meta-function that turns a vectorizable type or a tuple-like (recursively) of vectorizable types
 * into a stdx::simd or std::tuple (recursively) of stdx::simd. If N is non-zero, N determines the
 * resulting SIMD width. Otherwise, of all vectorizable types U the maximum
 * stdx::native_simd<U>::size() determines the resulting SIMD width.
 */
template<typename T, std::size_t N = 0>
using simdize = typename detail::simdize_impl<T, N>::type;

static_assert(std::same_as<simdize<std::tuple<std::tuple<int, double>, short, std::tuple<float>>>,
                           std::tuple<std::tuple<detail::deduced_simd<int, stdx::native_simd<short>::size()>,
                                                 detail::deduced_simd<double, stdx::native_simd<short>::size()>>,
                                      stdx::native_simd<short>,
                                      std::tuple<detail::deduced_simd<float, stdx::native_simd<short>::size()>>>>);

template<fixed_string Name, typename PortList>
consteval int
indexForName() {
    auto helper = []<std::size_t... Ids>(std::index_sequence<Ids...>) {
        constexpr int n_matches = ((PortList::template at<Ids>::static_name() == Name) + ...);
        static_assert(n_matches <= 1, "Multiple ports with that name were found. The name must be unique. You can "
                                      "still use a port index instead.");
        static_assert(n_matches == 1, "No port with the given name exists.");
        int result = -1;
        ((PortList::template at<Ids>::static_name() == Name ? (result = Ids) : 0), ...);
        return result;
    };
    return helper(std::make_index_sequence<PortList::size>());
}

template<typename... Lambdas>
struct overloaded : Lambdas... {
    using Lambdas::operator()...;
};

template<typename... Lambdas>
overloaded(Lambdas...) -> overloaded<Lambdas...>;

namespace detail {
template<template<typename...> typename Mapper, template<typename...> typename Wrapper, typename... Args>
Wrapper<Mapper<Args>...> *
type_transform_impl(Wrapper<Args...> *);

template<template<typename...> typename Mapper, typename T>
Mapper<T> *
type_transform_impl(T *);

template<template<typename...> typename Mapper>
void *
type_transform_impl(void *);
} // namespace detail

template<template<typename...> typename Mapper, typename T>
using type_transform = std::remove_pointer_t<decltype(detail::type_transform_impl<Mapper>(static_cast<T *>(nullptr)))>;

template<typename Arg, typename... Args>
auto safe_min(Arg&& arg, Args&&... args)
{
    if constexpr (sizeof...(Args) == 0) {
        return arg;
    } else {
        return std::min(std::forward<Arg>(arg), std::forward<Args>(args)...);
    }
}

template<typename Function, typename Tuple, typename... Tuples>
auto tuple_for_each(Function&& function, Tuple&& tuple, Tuples&&... tuples)
{
    static_assert(((std::tuple_size_v<std::remove_cvref_t<Tuple>> == std::tuple_size_v<std::remove_cvref_t<Tuples>>) && ...));
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        (([&function, &tuple, &tuples...](auto I) {
            function(std::get<I>(tuple), std::get<I>(tuples)...);
        }(std::integral_constant<std::size_t, Idx>{}), ...));
    }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple>>>());
}

template<typename Function, typename Tuple, typename... Tuples>
auto tuple_transform(Function&& function, Tuple&& tuple, Tuples&&... tuples)
{
    static_assert(((std::tuple_size_v<std::remove_cvref_t<Tuple>> == std::tuple_size_v<std::remove_cvref_t<Tuples>>) && ...));
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::make_tuple([&function, &tuple, &tuples...](auto I) {
                   return function(std::get<I>(tuple), std::get<I>(tuples)...);
               }(std::integral_constant<std::size_t, Idx>{})...);
    }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple>>>());
}


static_assert(std::is_same_v<std::vector<int>, type_transform<std::vector, int>>);
static_assert(std::is_same_v<std::tuple<std::vector<int>, std::vector<float>>, type_transform<std::vector, std::tuple<int, float>>>);
static_assert(std::is_same_v<void, type_transform<std::vector, void>>);

} // namespace fair::meta

#endif // include guard

// #include "circular_buffer.hpp"


namespace fair::graph {

using fair::meta::fixed_string;
using namespace fair::literals;

// #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> /*, ...*/>;

enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations
enum class connection_result_t { SUCCESS, FAILED };
enum class port_type_t { STREAM, MESSAGE }; // TODO: Think of a better name
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) { // dynamic definitions
                   typename T::value_type;
                   { t.pmt_type() } -> std::same_as<supported_type>;
                   { t.type() } -> std::same_as<port_type_t>;
                   { t.direction() } -> std::same_as<port_direction_t>;
                   { t.name() } -> std::same_as<std::string_view>;
                   { t.resize_buffer(n_items) } -> std::same_as<connection_result_t>;
                   { t.disconnect() } -> std::same_as<connection_result_t>;
               };


template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection, // TODO: sort default arguments
         std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent,
         gr::Buffer BufferType = gr::circular_buffer<T>>
class port {
public:
    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");

    using value_type                = T;

    static constexpr bool IS_INPUT  = PortDirection == port_direction_t::INPUT;
    static constexpr bool IS_OUTPUT = PortDirection == port_direction_t::OUTPUT;

    using port_tag                  = std::true_type;

    template <fixed_string NewName>
    using with_name = port<T, NewName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES, BufferType>;

private:
    using ReaderType          = decltype(std::declval<BufferType>().new_reader());
    using WriterType          = decltype(std::declval<BufferType>().new_writer());
    using IoType              = std::conditional_t<IS_INPUT, ReaderType, WriterType>;

    std::string  _name        = static_cast<std::string>(PortName);
    std::int16_t _priority    = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t  _min_samples = (MIN_SAMPLES == std::dynamic_extent ? 1 : MIN_SAMPLES);
    std::size_t  _max_samples = MAX_SAMPLES;
    bool         _connected   = false;

    IoType       _ioHandler   = new_io_handler();

public:
    [[nodiscard]] constexpr auto
    new_io_handler() const noexcept {
        if constexpr (IS_INPUT) {
            return BufferType(65536).new_reader();
        } else {
            return BufferType(65536).new_writer();
        }
    }

    [[nodiscard]] void *
    writer_handler_internal() noexcept {
        static_assert(IS_OUTPUT, "only to be used with output ports");
        return static_cast<void *>(std::addressof(_ioHandler));
    }

    [[nodiscard]] bool
    update_reader_internal(void *buffer_writer_handler_other) noexcept {
        static_assert(IS_INPUT, "only to be used with input ports");

        if (buffer_writer_handler_other == nullptr) {
            return false;
        }

        // TODO: If we want to allow ports with different buffer types to be mixed
        //       this will fail. We need to add a check that two ports that
        //       connect to each other use the same buffer type
        //       (std::any could be a viable approach)
        auto typed_buffer_writer = static_cast<WriterType *>(buffer_writer_handler_other);
        setBuffer(typed_buffer_writer->buffer());
        return true;
    }

public:
    port()             = default;

    port(const port &) = delete;

    auto
    operator=(const port &)
            = delete;

    port(std::string port_name, std::int16_t priority = 0, std::size_t min_samples = 0_UZ, std::size_t max_samples = SIZE_MAX) noexcept
        : _name(std::move(port_name))
        , _priority{ priority }
        , _min_samples(min_samples)
        , _max_samples(max_samples) {
        static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
    }

    constexpr port(port &&other) noexcept
        : _name(std::move(other._name))
        , _priority{ other._priority }
        , _min_samples(other._min_samples)
        , _max_samples(other._max_samples) {}

    constexpr port &
    operator=(port &&other) {
        port tmp(std::move(other));
        std::swap(_name, tmp._name);
        std::swap(_priority, tmp._priority);
        std::swap(_min_samples, tmp._min_samples);
        std::swap(_max_samples, tmp._max_samples);
        std::swap(_connected, tmp._connected);
        std::swap(_ioHandler, tmp._ioHandler);
        return *this;
    }

    [[nodiscard]] constexpr static port_type_t
    type() noexcept {
        return PortType;
    }

    [[nodiscard]] constexpr static port_direction_t
    direction() noexcept {
        return PortDirection;
    }

    [[nodiscard]] constexpr static decltype(PortName)
    static_name() noexcept
        requires(!PortName.empty())
    {
        return PortName;
    }

    [[nodiscard]] constexpr supported_type
    pmt_type() const noexcept {
        return T();
    }

    [[nodiscard]] constexpr std::string_view
    name() const noexcept {
        if constexpr (!PortName.empty()) {
            return static_cast<std::string_view>(PortName);
        } else {
            return _name;
        }
    }

    [[nodiscard]] constexpr std::int16_t
    priority() const noexcept {
        return _priority;
    }

    [[nodiscard]] constexpr static std::size_t
    available() noexcept {
        return 0;
    } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        if constexpr (MIN_SAMPLES == std::dynamic_extent) {
            return _min_samples;
        } else {
            return MIN_SAMPLES;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_buffer_size() const noexcept {
        if constexpr (MAX_SAMPLES == std::dynamic_extent) {
            return _max_samples;
        } else {
            return MAX_SAMPLES;
        }
    }

    [[nodiscard]] constexpr connection_result_t
    resize_buffer(std::size_t min_size) noexcept {
        if constexpr (IS_INPUT) {
            return connection_result_t::SUCCESS;
        } else {
            try {
                _ioHandler = BufferType(min_size).new_writer();
            } catch (...) {
                return connection_result_t::FAILED;
            }
        }
        return connection_result_t::SUCCESS;
    }

    [[nodiscard]] BufferType
    buffer() {
        return _ioHandler.buffer();
    }

    void
    setBuffer(gr::Buffer auto buffer) noexcept {
        if constexpr (IS_INPUT) {
            _ioHandler = std::move(buffer.new_reader());
            _connected = true;
        } else {
            _ioHandler = std::move(buffer.new_writer());
        }
    }

    [[nodiscard]] constexpr const ReaderType &
    reader() const noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr ReaderType &
    reader() noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr const WriterType &
    writer() const noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        if (_connected == false) {
            return connection_result_t::FAILED;
        }
        _ioHandler = new_io_handler();
        _connected = false;
        return connection_result_t::SUCCESS;
    }

    template<typename Other>
    [[nodiscard]] connection_result_t
    connect(Other &&other) {
        static_assert(IS_OUTPUT && std::remove_cvref_t<Other>::IS_INPUT);
        auto src_buffer = writer_handler_internal();
        return std::forward<Other>(other).update_reader_internal(src_buffer) ? connection_result_t::SUCCESS
                                                                             : connection_result_t::FAILED;
    }

    friend class dynamic_port;
};


namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, std::size_t... Is>
consteval fair::meta::typelist<just_t<T, Is>...>
repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

// TODO: Add port index to BaseName
template<std::size_t Count, typename T, fixed_string BaseName, port_type_t PortType, port_direction_t PortDirection, std::size_t MIN_SAMPLES = std::dynamic_extent,
         std::size_t MAX_SAMPLES = std::dynamic_extent>
using repeated_ports = decltype(detail::repeated_ports_impl<port<T, BaseName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES>>(std::make_index_sequence<Count>()));

template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using IN = port<T, PortName, port_type_t::STREAM, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using OUT = port<T, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using IN_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using OUT_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;

static_assert(Port<IN<float>>);
static_assert(Port<decltype(IN<float>())>);
static_assert(Port<OUT<float>>);
static_assert(Port<IN_MSG<float>>);
static_assert(Port<OUT_MSG<float>>);

static_assert(IN<float, 0, 0, "in">::static_name() == fixed_string("in"));
static_assert(requires { IN<float>("in").name(); });

static_assert(OUT_MSG<float, 0, 0, "out_msg">::static_name() == fixed_string("out_msg"));
static_assert(!(OUT_MSG<float, 0, 0, "out_msg">::with_name<"out_message">::static_name() == fixed_string("out_msg")));
static_assert(OUT_MSG<float, 0, 0, "out_msg">::with_name<"out_message">::static_name() == fixed_string("out_message"));

}

#endif // include guard

// #include "node.hpp"
#ifndef GNURADIO_NODE_HPP
#define GNURADIO_NODE_HPP

#include <map>

// #include <typelist.hpp>
 // localinclude
// #include <port.hpp>
 // localinclude
// #include <utils.hpp>
 // localinclude
// #include <node_traits.hpp>
#ifndef GNURADIO_NODE_NODE_TRAITS_HPP
#define GNURADIO_NODE_NODE_TRAITS_HPP

// #include <refl.hpp>
// The MIT License (MIT)
//
// Copyright (c) 2020 Veselin Karaganev (@veselink1) and Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef REFL_INCLUDE_HPP
#define REFL_INCLUDE_HPP

#include <stddef.h> // size_t
#include <cstring>
#include <array>
#include <utility> // std::move, std::forward
#include <optional>
#include <tuple>
#include <type_traits>
#include <ostream>
#include <sstream>
#include <iomanip> // std::quoted
#include <memory>
#include <complex>

#ifdef _MSC_VER
// Disable VS warning for "Not enough arguments for macro"
// (emitted when a REFL_ macro is not provided any attributes)
#pragma warning( disable : 4003 )
#endif

#if defined(__clang__)
  #if __has_feature(cxx_rtti)
    #define REFL_RTTI_ENABLED
  #endif
#elif defined(__GNUG__)
  #if defined(__GXX_RTTI)
    #define REFL_RTTI_ENABLED
  #endif
#elif defined(_MSC_VER)
  #if defined(_CPPRTTI)
    #define REFL_RTTI_ENABLED
  #endif
#endif

/**
 * @brief The top-level refl-cpp namespace
 * It contains a few core refl-cpp namespaces and directly exposes core classes and functions.
 * <ul>
 * <li>util - utility functions (for_each, map_to_tuple, etc.)</li>
 * <li>trait - type-traits and other operations on types (is_function_v, map_t, etc.)</li>
 * <li>runtime - utility functions and classes that always have a runtime overhead (proxy<T>, debug_str, etc.)</li>
 * <li>member - contains the empty classes member and function (used for tagging)</li>
 * <li>descriptor - contains the non-specialized member types (type|field_descriptor<T, N>, and operations on them (get_property, get_display_name, etc.))</li>
 * </ul>
 *
 * using util::type_list; <br>
 * using descriptor::type_descriptor; <br>
 * using descriptor::field_descriptor; <br>
 * using descriptor::function_descriptor; <br>
 * using util::const_string; <br>
 * using util::make_const_string; <br>
 */
namespace refl
{
    /**
     * @brief Contains utility types and functions for working with those types.
     */
    namespace util
    {
        /**
         * Converts a compile-time available const char* value to a const_string<N>.
         * The argument must be a *core constant expression* and be null-terminated.
         *
         * @see refl::util::const_string
         */
#define REFL_MAKE_CONST_STRING(CString) \
    (::refl::util::detail::copy_from_unsized<::refl::util::detail::strlen(CString)>(CString))

        /**
         * Represents a compile-time string. Used in refl-cpp
         * for representing names of reflected types and members.
         * Supports constexpr concatenation and substring,
         * and is explicitly-convertible to const char* and std::string.
         * REFL_MAKE_CONST_STRING can be used to create an instance from a literal string.
         *
         * @typeparam <N> The length of the string excluding the terminating '\0' character.
         * @see refl::descriptor::base_member_descriptor::name
         */
        template <size_t N>
        struct const_string
        {
            /** The largest positive value size_t can hold. */
            static constexpr size_t npos = static_cast<size_t>(-1);

            /** The length of the string excluding the terminating '\0' character. */
            static constexpr size_t size = N;

            /**
             * The statically-sized character buffer used for storing the string.
             */
            char data[N + 1];

            /**
             * Creates an empty const_string.
             */
            constexpr const_string() noexcept
                : data{}
            {
            }

            /**
             * Creates a copy of a const_string.
             */
            constexpr const_string(const const_string<N>& other) noexcept
                : const_string(other, std::make_index_sequence<N>())
            {
            }

            /**
             * Creates a const_string by copying the contents of data.
             */
            constexpr const_string(const char(&data)[N + 1]) noexcept
                : const_string(data, std::make_index_sequence<N>())
            {
            }

            /**
             * Explicitly converts to const char*.
             */
            explicit constexpr operator const char*() const noexcept
            {
                return data;
            }

            /**
             * Explicitly converts to std::string.
             */
            explicit operator std::string() const noexcept
            {
                return data;
            }

            /**
             * Returns a pointer to the contained zero-terminated string.
             */
            constexpr const char* c_str() const noexcept
            {
                return data;
            }

            /**
             * Returns the contained string as an std::string.
             */
            std::string str() const noexcept
            {
                return data;
            }

            /**
             * A constexpr version of std::string::substr.
             *
             * \code{.cpp}
             * make_const_string("Hello, World!").template substr<0, 4>() -> (const_string<4>) "Hell"
             * make_const_string("Hello, World!").template substr<1, 4>() -> (const_string<3>) "ell"
             * \endcode
             */
            template <size_t Pos, size_t Count = npos>
            constexpr auto substr() const noexcept
            {
                static_assert(Pos <= N);
                constexpr size_t NewSize = (std::min)(Count, N - Pos);

                char buf[NewSize + 1]{};
                for (size_t i = 0; i < NewSize; i++) {
                    buf[i] = data[Pos + i];
                }

                return const_string<NewSize>(buf);
            }

            /**
             * Searches the string for the first occurrence of the character and returns its position.
             *
             * \code{.cpp}
             * make_const_string("Hello, World!").find('e') -> 1
             * make_const_string("Hello, World!").find('z') -> static_cast<size_t>(-1)
             * \endcode
             */
            constexpr auto find(char ch, size_t pos = 0) const noexcept
            {
                for (size_t i = pos; i < N; i++) {
                    if (data[i] == ch) {
                        return i;
                    }
                }
                return npos;
            }

            /**
             * Searches the string for the last occurrence of the character and returns its position.
             *
             * \code{.cpp}
             * make_const_string("Hello, World!").rfind('o') -> 8
             * make_const_string("Hello, World!").rfind('z') -> static_cast<size_t>(-1)
             * \endcode
             */
            constexpr auto rfind(char ch, size_t pos = npos) const noexcept
            {
                for (size_t i = (pos == npos ? N - 1 : pos); i + 1 > 0; i--) {
                    if (data[i] == ch) {
                        return i;
                    }
                }
                return npos;
            }

        private:

            /**
             * Creates a copy of a const_string.
             */
            template <size_t... Idx>
            constexpr const_string(const const_string<N>& other, std::index_sequence<Idx...>) noexcept
                : data{ other.data[Idx]... }
            {
            }

            /**
             * Creates a const_string by copying the contents of data.
             */
            template <size_t... Idx>
            constexpr const_string(const char(&data)[sizeof...(Idx) + 1], std::index_sequence<Idx...>) noexcept
                : data{ data[Idx]... }
            {
            }

        };

        /**
         * Creates an empty instance of const_string<N>
         *
         * @see refl::util::const_string
         */
        constexpr const_string<0> make_const_string() noexcept
        {
            return {};
        }

        /**
         * Creates an instance of const_string<N>
         *
         * @see refl::util::const_string
         */
        template <size_t N>
        constexpr const_string<N - 1> make_const_string(const char(&str)[N]) noexcept
        {
            return str;
        }

        /**
         * Creates an instance of const_string<N>
         *
         * @see refl::util::const_string
         */
        constexpr const_string<1> make_const_string(char ch) noexcept
        {
            const char str[2]{ ch, '\0' };
            return make_const_string(str);
        }

        /**
         * Concatenates two const_strings together.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr const_string<N + M> operator+(const const_string<N>& a, const const_string<M>& b) noexcept
        {
            char data[N + M + 1] { };
            for (size_t i = 0; i < N; i++)
                data[i] = a.data[i];
            for (size_t i = 0; i < M; i++)
                data[N + i] = b.data[i];
            return data;
        }

        /**
         * Concatenates a const_string with a C-style string.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr const_string<N + M - 1> operator+(const const_string<N>& a, const char(&b)[M]) noexcept
        {
            return a + make_const_string(b);
        }

        /**
         * Concatenates a C-style string with a const_string.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr const_string<N + M - 1> operator+(const char(&a)[N], const const_string<M>& b) noexcept
        {
            return make_const_string(a) + b;
        }

        /**
         * Compares two const_strings for equality.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr bool operator==(const const_string<N>& a, const const_string<M>& b) noexcept
        {
            if constexpr (N != M) {
                return false;
            }
            else {
                for (size_t i = 0; i < M; i++) {
                    if (a.data[i] != b.data[i]) {
                        return false;
                    }
                }
                return true;
            }
        }

        /**
         * Compares two const_strings for equality.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr bool operator!=(const const_string<N>& a, const const_string<M>& b) noexcept
        {
            return !(a == b);
        }

        /**
         * Compares a const_string with a C-style string for equality.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr bool operator==(const const_string<N>& a, const char(&b)[M]) noexcept
        {
            return a == make_const_string(b);
        }

        /**
         * Compares a const_string with a C-style string for equality.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr bool operator!=(const const_string<N>& a, const char(&b)[M]) noexcept
        {
            return a != make_const_string(b);
        }

        /**
         * Compares a C-style string with a const_string for equality.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr bool operator==(const char(&a)[N], const const_string<M>& b) noexcept
        {
            return make_const_string(a) == b;
        }

        /**
         * Compares a C-style string with a const_string for equality.
         *
         * @see refl::util::const_string
         */
        template <size_t N, size_t M>
        constexpr bool operator!=(const char(&a)[N], const const_string<M>& b) noexcept
        {
            return make_const_string(a) != b;
        }

        template <size_t N>
        constexpr std::ostream& operator<<(std::ostream& os, const const_string<N>& str) noexcept
        {
            return os << str.c_str();
        }

        namespace detail
        {
            constexpr size_t strlen(const char* const str)
            {
                return *str ? 1 + strlen(str + 1) : 0;
            }

            template <size_t N>
            constexpr const_string<N> copy_from_unsized(const char* const str)
            {
                const_string<N> cstr;
                for (size_t i = 0; i < N; i++) {
                    cstr.data[i] = str[i];
                }
                return cstr;
            }
        } // namespace detail

        /**
         * Represents a compile-time list of types provided as variadic template parameters.
         * type_list is an empty TrivialType. Instances of it can freely be created to communicate
         * the list of represented types. type_lists support many standard operations that are
         * implicitly available with ADL-lookup. type_list is used by refl-cpp mostly to represent
         * the list of refl::field_descriptor, refl::function_descriptor specializations that
         * allow the compile-time reflection of a type's members.
         *
         * @see refl::util::for_each
         * @see refl::util::map_to_array
         * @see refl::util::map_to_tuple
         * @see refl::member_list
         *
         * # Examples
         * ```
         * for_each(type_list<int, float>(), [](auto) { ... });
         * ```
         */
        template <typename... Ts>
        struct type_list
        {
            /** The number of types in this type_list */
            static constexpr intptr_t size = sizeof...(Ts);
        };

        template <typename T>
        struct type_list<T>
        {
            typedef T type;
            static constexpr intptr_t size = 1;
        };

        template <typename T>
        using type_tag = type_list<T>;

    } // namespace util

    using util::const_string;
    using util::make_const_string;
    using util::type_list;
    using util::type_tag;

    /**
     * The contents of the refl::detail::macro_exports namespace
     * is implicitly available in the context of REFL_TYPE/FIELD/FUNC macros.
     * It is used to export the refl::attr:: standard attributes.
     */
    namespace detail
    {
        namespace macro_exports
        {
        }
    }

} // namespace refl

/**
 * refl_impl is an internal namespace that should not be used by the users of refl-cpp.
 */
namespace refl_impl
{
    /**
     * Contains the generated metadata types.
     * (i.e. type_info__)
     */
    namespace metadata
    {
        // Import everyting from macro_exports here to make it visible in REFL_ macro context.
        using namespace refl::detail::macro_exports;

        /**
         * The core reflection metadata type.
         * type_info__ holds data for a type T.
         *
         * The non-specialized type_info__ type has a member typedef invalid_marker
         * that can be used to detect it.
         *
         * Specializations of this type should provide all members of this
         * generic definition, except invalid_marker.
         *
         * @typeparam <T> The reflected type.
         */
        template <typename T>
        struct type_info__
        {
            /** Used for detecting this non-specialized type_info__ instance. */
            struct invalid_marker{};

            /**
             * This is a placeholder definition of which no type instances should be created.
             */
            template <size_t, typename>
            struct member;

            /** The number of reflected members of the target type T. */
            static constexpr size_t member_count{ 0 };

            /** This is a placeholder definition which shold not be referenced by well-formed programs. */
            static constexpr refl::const_string<0> name{ "" };

            /** This is a placeholder definition which shold not be referenced by well-formed programs. */
            static constexpr std::tuple<> attributes{ };
        };

        /**
         * Specializes type_info__ so that a type's const-qualification is effectively discarded.
         */
        template <typename T>
        struct type_info__<const T> : public type_info__<T> {};

        /**
         * Specializes type_info__ so that a type's volatile-qualification is effectively discarded.
         */
        template <typename T>
        struct type_info__<volatile T> : public type_info__<T> {};

        /**
         * Specializes type_info__ so that a type's const-volatile-qualification is effectively discarded.
         */
        template <typename T>
        struct type_info__<const volatile T> : public type_info__<T> {};

    } // namespace metadata

} // namespace refl_impl

namespace refl
{
    namespace detail
    {
        template <typename T>
        using type_info = refl_impl::metadata::type_info__<T>;

        template <typename T, size_t N>
        using member_info = typename type_info<T>::template member<N>;
    } // namespace detail

    /**
     * @brief Contains tag types denoting the different types of reflectable members.
     *
     * This namespace contains a number of empty types that correspond to
     * the different member types that refl-cpp supports reflection over.
     */
    namespace member
    {
        /**
         * An empty type which is equivalent to refl::member_descriptor_base::member_type
         * when the reflected member is a field.
         *
         * @see refl::descriptor::field_descriptor
         */
        struct field {};

        /**
         * An empty type which is equivalent to refl::member_descriptor_base::member_type
         * when the reflected member is a function.
         *
         * @see refl::descriptor::function_descriptor
         */
        struct function {};
    }

    namespace descriptor
    {
        template <typename>
        class type_descriptor;

        template <typename, size_t>
        class field_descriptor;

        template <typename, size_t>
        class function_descriptor;
    } // namespace descriptor

    /**
     * @brief Provides type-level operations for refl-cpp related use-cases.
     *
     * The refl::trait namespace provides type-level operations useful
     * for compile-time metaprogramming.
     */
    namespace trait
    {/**
         * Removes all reference and cv-qualifiers from T.
         * Equivalent to std::remove_cvref which is not currently
         * available on all C++17 compilers.
         */
        template <typename T>
        struct remove_qualifiers
        {
            typedef std::remove_cv_t<std::remove_reference_t<T>> type;
        };

        /**
         * Removes all reference and cv-qualifiers from T.
         * Equivalent to std::remove_cvref_t which is not currently
         * available on all C++17 compilers.
         */
        template <typename T>
        using remove_qualifiers_t = typename remove_qualifiers<T>::type;

        namespace detail
        {
            /** SFIANE support for detecting whether there is a type_info__ specialization for T. */
            template <typename T>
            decltype(typename refl::detail::type_info<T>::invalid_marker{}, std::false_type{}) is_reflectable_test(int);

            /** SFIANE support for detecting whether there is a type_info__ specialization for T. */
            template <typename T>
            std::true_type is_reflectable_test(...);
        } // namespace detail

        /**
         * Checks whether there is reflection metadata for the type T.
         * Inherits from std::bool_constant<>
         *
         * @see REFL_TYPE
         * @see REFL_AUTO
         * @see refl::is_reflectable
         */
        template <typename T>
        struct is_reflectable : decltype(detail::is_reflectable_test<T>(0))
        {
        };

        /**
         * Checks whether there is reflection metadata for the type T.
         * Inherits from std::bool_constant<>
         *
         * @see refl::trait::is_reflectable
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_reflectable_v{ is_reflectable<T>::value };

        namespace detail
        {
            /** SFIANE support for detecting whether the type T supports member .begin() and .end() operations. */
            template <typename U>
            [[maybe_unused]] static auto is_container_test(int) -> decltype(std::declval<U>().begin(), std::declval<U>().end(), std::true_type{});

            /** SFIANE support for detecting whether the type T supports member .begin() and .end() operations. */
            template <typename U>
            [[maybe_unused]] static std::false_type is_container_test(...);
        }

        /**
         * Checks whether objects of the type T support member .begin() and .end() operations.
         */
        template <typename T>
        struct is_container : decltype(detail::is_container_test<T>(0))
        {
        };

        /**
         * Checks whether objects of the type T support member .begin() and .end() operations.
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_container_v{ is_container<T>::value };

        namespace detail
        {

            template <size_t D, size_t N, typename... Ts>
            struct get;

            template <size_t D, size_t N>
            struct get<D, N>
            {
                static_assert(N > 0, "Missing arguments list for get<N, Ts...>!");
            };

            template <size_t N, typename T, typename... Ts>
            struct get<1, N, T, Ts...> : public get<
                                             (N > 16 ? (N > 64 ? 64 : 16) : 1),
                                             N - 1, Ts...>
            {
            };

            template <typename T, typename... Ts>
            struct get<1, 0, T, Ts...>
            {
                typedef T type;
            };

            template <typename T, typename... Ts>
            struct get<16, 0, T, Ts...>
            {
                typedef T type;
            };

            template <typename T, typename... Ts>
            struct get<64, 0, T, Ts...>
            {
                typedef T type;
            };

            template <
                size_t N, typename T0, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6, typename T7, typename T8,
                typename T9, typename T10, typename T11, typename T12,
                typename T13, typename T14, typename T15, typename... Ts>
            struct get<
                16, N, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                T13, T14, T15, Ts...> : get<1, N - 16, Ts...>
            {
            };

            template <
                size_t N, typename T0, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6, typename T7, typename T8,
                typename T9, typename T10, typename T11, typename T12,
                typename T13, typename T14, typename T15, typename T16,
                typename T17, typename T18, typename T19, typename T20,
                typename T21, typename T22, typename T23, typename T24,
                typename T25, typename T26, typename T27, typename T28,
                typename T29, typename T30, typename T31, typename T32,
                typename T33, typename T34, typename T35, typename T36,
                typename T37, typename T38, typename T39, typename T40,
                typename T41, typename T42, typename T43, typename T44,
                typename T45, typename T46, typename T47, typename T48,
                typename T49, typename T50, typename T51, typename T52,
                typename T53, typename T54, typename T55, typename T56,
                typename T57, typename T58, typename T59, typename T60,
                typename T61, typename T62, typename T63, typename... Ts>
            struct get<
                64, N, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
                T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
                T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51,
                T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63,
                Ts...> : get<1, N - 64, Ts...>
            {
            };

            template <size_t N, typename...>
            struct skip;

            template <size_t N, typename T, typename... Ts>
            struct skip<N, T, Ts...> : skip<N - 1, Ts...>
            {
            };

            template <typename T, typename... Ts>
            struct skip<0, T, Ts...>
            {
                typedef type_list<T, Ts...> type;
            };

            template <>
            struct skip<0>
            {
                typedef type_list<> type;
            };
        }

        /// \private
        template <size_t, typename>
        struct get;

        /**
         * Provides a member typedef type which is the
         * N-th type in the provided type_list.
         *
         * \code{.cpp}
         * typename get<0, type_list<int, float>>::type -> int
         * typename get<1, type_list<int, float>>::type -> float
         * \endcode
         */
        template <size_t N, typename... Ts>
        struct get<N, type_list<Ts...>> : detail::get<1, N, Ts...>
        {
        };

        /**
         * The N-th type in the provided type_list.
         * @see get
         */
        template <size_t N, typename TypeList>
        using get_t = typename get<N, TypeList>::type;

        /// \private
        template <size_t, typename>
        struct skip;

        /**
         * Skips the first N types in the provided type_list.
         * Provides a member typedef equivalent to the resuling type_list.
         *
         * \code{.cpp}
         * typename skip<1, type_list<int, float, double>>::type -> type_list<float, double>
         * typename skip<0, type_list<int, float, double>>::type -> type_list<int, float, double>
         * \endcode
         */
        template <size_t N, typename... Ts>
        struct skip<N, type_list<Ts...>> : detail::skip<N, Ts...>
        {
        };

        /**
         * Skips the first N types in the provided type_list.
         * @see skip
         */
        template <size_t N, typename TypeList>
        using skip_t = typename skip<N, TypeList>::type;

        /// \private
        template <typename>
        struct as_type_list;

        /**
         * Provides a member typedef type which is a type_list with
         * template type parameters equivalent to the type parameters of the provided
         * type. The provided type must be a template instance.
         *
         * \code{.cpp}
         * typename as_type_list<std::tuple<int, float>>::type -> type_list<int, float>
         * \endcode
         */
        template <template <typename...> typename T, typename... Ts>
        struct as_type_list<T<Ts...>>
        {
            typedef type_list<Ts...> type;
        };

        /// \private
        template <typename T>
        struct as_type_list : as_type_list<remove_qualifiers_t<T>>
        {
        };

        /**
         * A typedef for a type_list with
         * template type parameters equivalent to the type parameters of the provided
         * type. The provided type must be a template instance.
         * @see as_type_list
         */
        template <typename T>
        using as_type_list_t = typename as_type_list<T>::type;

        /// \private
        template <typename>
        struct as_tuple;

        /**
         * Provides a member typedef which is a std::tuple specialization with
         * template type parameters equivalent to the type parameters of the provided
         * type. The provided type must be a template specialization.
         *
         * \code{.cpp}
         * typename as_tuple<type_list<int, float>>::type -> std::tuple<int, float>
         * \endcode
         */
        template <template <typename...> typename T, typename... Ts>
        struct as_tuple<T<Ts...>>
        {
            typedef std::tuple<Ts...> type;
        };

        /// \private
        template <typename T>
        struct as_tuple : as_tuple<remove_qualifiers_t<T>>
        {
        };

        /**
         * A typedef for a std::tuple specialization with
         * template type parameters equivalent to the type parameters of the provided
         * type. The provided type must be a template specialization.
         * @see as_tuple
         */
        template <typename T>
        using as_tuple_t = typename as_tuple<T>::type;

        /**
         * Accesses first type in the list.
         */
        template <typename TypeList>
        using first = get<0, TypeList>;

        /**
         * Accesses last type in the list.
         * @see last
         */
        template <typename TypeList>
        using first_t = typename first<TypeList>::type;

        /**
         * Accesses last type in the list.
         */
        template <typename TypeList>
        using last = get<TypeList::size - 1, TypeList>;

        /**
         * Accesses last type in the list.
         * @see last
         */
        template <typename TypeList>
        using last_t = typename last<TypeList>::type;

        /**
         * Returns all but the first element of the list.
         */
        template <typename TypeList>
        using tail = skip<1, TypeList>;

        /**
         * Returns all but the first element of the list.
         * @see tail
         */
        template <typename TypeList>
        using tail_t = typename tail<TypeList>::type;

        namespace detail
        {
            template <typename, size_t, typename>
            struct take;

            template <typename... Us>
            struct take<type_list<Us...>, 0, type_list<>>
            {
                using type = type_list<Us...>;
            };

            template <typename... Us, typename T, typename... Ts>
            struct take<type_list<Us...>, 0, type_list<T, Ts...>>
            {
                using type = type_list<Us...>;
            };

            template <size_t N, typename... Us, typename T, typename... Ts>
            struct take<type_list<Us...>, N, type_list<T, Ts...>>
            {
                using type = typename take<type_list<Us..., T>, N - 1, type_list<Ts...>>::type;
            };
        }

        /**
         * Returns the first N elements of the list.
         */
        template <size_t N, typename TypeList>
        using take = detail::take<type_list<>, N, TypeList>;

        /**
         * Returns the first N elements of the list.
         */
        template <size_t N, typename TypeList>
        using take_t = typename take<N, TypeList>::type;

        /**
         * Returns all but the last element of the list.
         */
        template <typename TypeList>
        using init = take<TypeList::size - 1, TypeList>;

        /**
         * Returns all but the last element of the list.
         * @see tail
         */
        template <typename TypeList>
        using init_t = typename init<TypeList>::type;

        namespace detail
        {
            template <typename, typename>
            struct reverse_impl;

            template <typename... Us>
            struct reverse_impl<type_list<Us...>, type_list<>>
            {
                using type = type_list<Us...>;
            };

            template <typename... Us, typename T, typename... Ts>
            struct reverse_impl<type_list<Us...>, type_list<T, Ts...>>
            {
                using type = typename reverse_impl<type_list<T, Us...>, type_list<Ts...>>::type;
            };
        } // namespace detail

        /**
         * Reverses a list of types.
         *
         * \code{.cpp}
         * typename reverse<type_list<int, float>>::type -> type_list<float, int>
         * \endcode
         */
        template <typename TypeList>
        struct reverse : detail::reverse_impl<type_list<>, TypeList>
        {
        };

        /**
         * Reverses a list of types.
         * @see reverse
         */
        template <typename TypeList>
        using reverse_t = typename reverse<TypeList>::type;

        /**
         * Concatenates N lists together.
         *
         * \code{.cpp}
         * typename concat<type_list<int, float>, type_list<double>, type_list<long>>::type -> type_list<int, float, double, long>
         * \endcode
         */
        template <typename...>
        struct concat;

        /// \private
        template <>
        struct concat<>
        {
            using type = type_list<>;
        };

        /// \private
        template <typename... Ts>
        struct concat<type_list<Ts...>>
        {
            using type = type_list<Ts...>;
        };

        /**
         * Concatenates two lists together.
         */
        /// \private
        template <typename... Ts, typename... Us>
        struct concat<type_list<Ts...>, type_list<Us...>>
        {
            using type = type_list<Ts..., Us...>;
        };

        /**
         * Concatenates N lists together.
         */
        /// \private
        template <typename TypeList1, typename TypeList2, typename... TypeLists>
        struct concat<TypeList1, TypeList2, TypeLists...> : concat<typename concat<TypeList1, TypeList2>::type, TypeLists...>
        {
        };

        /**
         * Concatenates two lists together.
         * @see concat
         */
        template <typename... Ts>
        using concat_t = typename concat<Ts...>::type;

        /**
         * Appends a type to the list.
         */
        template <typename T, typename TypeList>
        struct append : concat<TypeList, type_list<T>>
        {
        };

        /**
         * Appends a type to the list.
         * @see prepend
         */
        template <typename T, typename TypeList>
        using append_t = typename append<T, TypeList>::type;

        template <typename, typename>
        struct prepend;

        /**
         * Prepends a type to the list.
         */
        template <typename T, typename TypeList>
        struct prepend : concat<type_list<T>, TypeList>
        {
        };

        /**
         * Prepends a type to the list.
         * @see prepend
         */
        template <typename T, typename TypeList>
        using prepend_t = typename prepend<T, TypeList>::type;

        namespace detail
        {
            template <template<typename> typename, typename...>
            struct filter_impl;

            template <template<typename> typename Predicate>
            struct filter_impl<Predicate>
            {
                using type = type_list<>;
            };

            template <template<typename> typename Predicate, typename Head, typename... Tail>
            struct filter_impl<Predicate, Head, Tail...>
            {
                using type = std::conditional_t<Predicate<Head>::value,
                    prepend_t<Head, typename filter_impl<Predicate, Tail...>::type>,
                    typename filter_impl<Predicate, Tail...>::type
                >;
            };

            template <template<typename> typename, typename...>
            struct map_impl;

            template <template<typename> typename Mapper>
            struct map_impl<Mapper>
            {
                using type = type_list<>;
            };

            template <template<typename> typename Mapper, typename Head, typename ...Tail>
            struct map_impl<Mapper, Head, Tail...>
            {
                using type = typename prepend<typename Mapper<Head>::type,
                    typename map_impl<Mapper, Tail...>::type>::type;
            };
        }

        /// \private
        template <template<typename> typename, typename>
        struct filter;

        /**
         * Filters a type_list according to a predicate template.
         *
         * \code{.cpp}
         * typename filter<std::is_reference, type_list<int, float&, double>>::type -> type_list<float&>
         * \endcode
         */
        template <template<typename> typename Predicate, typename... Ts>
        struct filter<Predicate, type_list<Ts...>>
        {
            using type = typename detail::filter_impl<Predicate, Ts...>::type;
        };

        /**
         * Filters a type_list according to a predicate template
         * with a static boolean member named "value" (e.g. std::is_trivial)
         * @see filter
         */
        template <template<typename> typename Predicate, typename TypeList>
        using filter_t = typename filter<Predicate, TypeList>::type;

        /// \private
        template <template<typename> typename, typename>
        struct map;

        /**
         * Transforms a type_list according to a predicate template.
         *
         * \code{.cpp}
         * typename map<std::add_reference, type_list<int, float&, double>>::type -> type_list<int&, float&, double&>
         * \endcode
         */
        template <template<typename> typename Mapper, typename... Ts>
        struct map<Mapper, type_list<Ts...>>
        {
            using type = typename detail::map_impl<Mapper, Ts...>::type;
        };

        /**
         * Transforms a type_list according to a predicate template
         * with a typedef named "type" (e.g. std::remove_reference)
         * @see map
         */
        template <template<typename> typename Mapper, typename... Ts>
        using map_t = typename map<Mapper, Ts...>::type;

        namespace detail
        {
            template <typename T>
            struct is_instance : public std::false_type {};

            template <template<typename...> typename T, typename... Args>
            struct is_instance<T<Args...>> : public std::true_type {};
        } // namespace detail

        /**
         * Detects whether T is a template specialization.
         * Inherits from std::bool_constant<>.
         *
         * \code{.cpp}
         * is_instance<type_list<>>::value -> true
         * is_instance<int>::value -> false
         * \endcode
         */
        template <typename T>
        struct is_instance : detail::is_instance<T>
        {
        };

        /**
         * Detects whether T is a template specialization.
         * @see is_instance
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_instance_v{ is_instance<T>::value };

        namespace detail
        {
            /**
             * Checks if T == U<Args...>.
             * If U<Args...> != T or is invalid the result is false.
             */
            template <typename T, template<typename...> typename U, typename... Args>
            struct is_same_template
            {
                template <template<typename...> typename V, typename = V<Args...>>
                static auto test(int) -> std::is_same<V<Args...>, T>;

                template <template<typename...> typename V>
                static std::false_type test(...);

                static constexpr bool value{decltype(test<U>(0))::value};
            };

            template <template<typename...> typename T, typename U>
            struct is_instance_of : public std::false_type {};

            template <template<typename...> typename T, template<typename...> typename U, typename... Args>
            struct is_instance_of<T, U<Args...>> : public is_same_template<U<Args...>, T, Args...>
            {
            };
        }

        /**
         * Detects whther the type U is a template specialization of T.
         * (e.g. is_instance_of<std::vector<>, std::vector<int>>)
         * Inherits from std::bool_constant<>.
         *
         * \code{.cpp}
         * is_instance_of<type_list, type_list<int>>::value -> true
         * is_instance_of<type_list, std::tuple<int>>::value -> false
         * \endcode
         */
        template <template<typename...>typename T, typename U>
        struct is_instance_of : detail::is_instance_of<T, std::remove_cv_t<U>>
        {
        };

        /**
         * Detects whther the type U is a template specialization of T.
         * @see is_instance_of_v
         */
        template <template<typename...>typename T, typename U>
        [[maybe_unused]] static constexpr bool is_instance_of_v{ is_instance_of<T, U>::value };

        /// \private
        template <typename, typename>
        struct contains;

        /**
         * Checks whether T is contained in the list of types.
         * Inherits from std::bool_constant<>.
         *
         * \code{.cpp}
         * contains<int, type_list<int, float>>::value -> true
         * contains<double, type_list<int, float>>::value -> false
         * \endcode
         */
        template <typename T, typename... Ts>
        struct contains<T, type_list<Ts...>> : std::disjunction<std::is_same<std::remove_cv_t<T>, std::remove_cv_t<Ts>>...>
        {
        };

        /**
         * Checks whether T is contained in the list of types.
         * @see contains
         */
        template <typename T, typename TypeList>
        [[maybe_unused]] static constexpr bool contains_v = contains<T, TypeList>::value;

        /// \private
        template <template<typename...> typename, typename>
        struct contains_instance;

        /**
         * Checks whether an instance of the template T is contained in the list of types.
         * Inherits from std::bool_constant<>.
         *
         * \code{.cpp}
         * contains_instance<std::tuple, type_list<int, float, std::tuple<short, double>>>::value -> true
         * contains_instance<std::vector, type_list<int, float, std::tuple<short, double>>>::value -> false
         * \endcode
         */
        template <template<typename...> typename T, typename... Ts>
        struct contains_instance<T, type_list<Ts...>> : std::disjunction<trait::is_instance_of<T, std::remove_cv_t<Ts>>...>
        {
        };

        /**
         * Checks whether an instance of the template T is contained in the list of types.
         * @see contains_instance
         */
        template <template<typename...> typename T, typename TypeList>
        [[maybe_unused]] static constexpr bool contains_instance_v = contains_instance<T, TypeList>::value;

        /// \private
        template <typename, typename>
        struct contains_base;

        /**
         * Checks whether a type deriving from T is contained in the list of types.
         * Inherits from std::bool_constant<>.
         *
         * \code{.cpp}
         * struct Base {};
         * struct Derived : Base {};
         * contains_base<Base, type_list<int, float, Derived>>::value -> true
         * contains_base<Base, type_list<int, float, Base>>::value -> true
         * contains_base<int, type_list<int, float, Derived>>::value -> false
         * \endcode
         */
        template <typename T, typename... Ts>
        struct contains_base<T, type_list<Ts...>> : std::disjunction<std::is_base_of<std::remove_cv_t<T>, std::remove_cv_t<Ts>>...>
        {
        };

        /**
         * Checks whether a type deriving from T is contained in the list of types.
         * @see contains_base
         */
        template <typename T, typename TypeList>
        [[maybe_unused]] static constexpr bool contains_base_v = contains_base<T, TypeList>::value;

        namespace detail
        {
            template <typename T, ptrdiff_t N, typename... Ts>
            constexpr ptrdiff_t index_of() noexcept
            {
                if constexpr (sizeof...(Ts) <= N) return -1;
                else if constexpr (std::is_same_v<T, trait::get_t<N, type_list<Ts...>>>) return N;
                else return index_of<T, N + 1, Ts...>();
            }

            template <typename T, ptrdiff_t N, typename... Ts>
            constexpr ptrdiff_t index_of_base() noexcept
            {
                if constexpr (sizeof...(Ts) <= N) return -1;
                else if constexpr (std::is_base_of_v<T, trait::get_t<N, type_list<Ts...>>>) return N;
                else return index_of_base<T, N + 1, Ts...>();
            }

            template <template<typename...> typename T, ptrdiff_t N, typename... Ts>
            constexpr ptrdiff_t index_of_instance() noexcept
            {
                if constexpr (sizeof...(Ts) <= N) return -1;
                else if constexpr (is_instance_of_v<T, trait::get_t<N, type_list<Ts...>>>) return N;
                else return index_of_instance<T, N + 1, Ts...>();
            }

            // This variable template was introduced to fix the build on VS2017, which
            // chokes when invoking index_of_instance() directly from struct index_of_instance.
            template <template<typename...> typename T, ptrdiff_t N, typename... Ts>
            static constexpr ptrdiff_t index_of_instance_v = index_of_instance<T, N, Ts...>();
        } // namespace detail

        /// \private
        template <typename, typename>
        struct index_of;

        /**
         * The index of the type in the type list, -1 if it doesn't exist.
         * @see contains
         */
        template <typename T, typename... Ts>
        struct index_of<T, type_list<Ts...>> : std::integral_constant<ptrdiff_t, detail::index_of<T, 0, Ts...>()>
        {
        };

        /**
         * The index of the type in the type list, -1 if it doesn't exist.
         * @see index_of
         */
        template <typename T, typename TypeList>
        static constexpr ptrdiff_t index_of_v = index_of<T, TypeList>::value;

        /// \private
        template <typename, typename>
        struct index_of_base;

        /**
         * The index of the type in the type list that is derived from T, -1 if it doesn't exist.
         * @see contains_base
         */
        template <typename T, typename... Ts>
        struct index_of_base<T, type_list<Ts...>> : std::integral_constant<ptrdiff_t, detail::index_of_base<T, 0, Ts...>()>
        {
        };

        /**
         * The index of the type in the type list that is derived from T, -1 if it doesn't exist.
         * @see index_of_base
         */
        template <typename T, typename TypeList>
        static constexpr ptrdiff_t index_of_base_v = index_of_base<T, TypeList>::value;

        /// \private
        template <template<typename...> typename, typename>
        struct index_of_instance;

        /**
         * The index of the type in the type list that is a template instance of T, -1 if it doesn't exist.
         * @see contains_instance
         */
        template <template<typename...> typename T, typename... Ts>
        struct index_of_instance<T, type_list<Ts...>> : std::integral_constant<ptrdiff_t, detail::index_of_instance_v<T, 0, Ts...>>
        {
        };

        /**
         * The index of the type in the type list that is a template instance of T, -1 if it doesn't exist.
         * @see index_of_instance
         */
        template <template<typename...> typename T, typename TypeList>
        static constexpr ptrdiff_t index_of_instance_v = index_of_instance<T, TypeList>::value;

        namespace detail
        {
            template <typename, typename>
            struct unique_impl;

            template <typename UniqueList>
            struct unique_impl<UniqueList, type_list<>>
            {
                using type = UniqueList;
            };

            template <typename UniqueList, typename T, typename... Ts>
            struct unique_impl<UniqueList, type_list<T, Ts...>> :
                std::conditional_t<contains_v<T, UniqueList>,
                    unique_impl<UniqueList, type_list<Ts...>>,
                    unique_impl<append_t<T, UniqueList>, type_list<Ts...>>>
            {
            };
        } // namespace detail

        /**
         * Creates a new list containing the repeating elements in the source list only once.
         *
         * \code{.cpp}
         * typename unique<type_list<int, float, int>>::type -> type_list<int, float>
         * \endcode
         */
        template <typename T>
        struct unique : detail::unique_impl<type_list<>, T>
        {
        };

        /**
         * Creates a new list containing the repeating elements in the source list only once.
         */
        template <typename T>
        using unique_t = typename unique<T>::type;

    } // namespace trait

    namespace util
    {
        /**
         * Ignores all parameters. Can take an optional template parameter
         * specifying the return type of ignore. The return object is iniailized by {}.
         */
        template <typename T = int, typename... Ts>
        constexpr int ignore(Ts&&...) noexcept
        {
            return {};
        }

        /**
         * Returns the input paratemeter as-is. Useful for expanding variadic
         * template lists when only one arguments is known to be present.
         */
        template <typename T>
        constexpr decltype(auto) identity(T&& t) noexcept
        {
            return std::forward<T>(t);
        }

        /**
         * Adds const to the input reference.
         */
        template <typename T>
        constexpr const T& make_const(const T& value) noexcept
        {
            return value;
        }

        /**
         * Adds const to the input reference.
         */
        template <typename T>
        constexpr const T& make_const(T& value) noexcept
        {
            return value;
        }

        /**
        * Creates an array of type 'T' from the provided tuple.
        * The common type T needs to be specified, in order to prevent any
        * errors when using the overload taking an empty std::tuple (as there is no common type then).
        */
        template <typename T, typename... Ts>
        constexpr std::array<T, sizeof...(Ts)> to_array(const std::tuple<Ts...>& tuple) noexcept
        {
            return std::apply([](auto&& ... args) -> std::array<T, sizeof...(Ts)> { return { std::forward<decltype(args)>(args)... }; }, tuple);
        }

        /**
         * Creates an empty array of type 'T.
         */
        /// \private
        template <typename T>
        constexpr std::array<T, 0> to_array(const std::tuple<>&) noexcept
        {
            return {};
        }

        namespace detail
        {
            template <typename T, size_t... Idx>
            constexpr auto to_tuple([[maybe_unused]] const std::array<T, sizeof...(Idx)>& array, std::index_sequence<Idx...>) noexcept
            {
                if constexpr (sizeof...(Idx) == 0) return std::tuple<>{};
                else return std::make_tuple(std::get<Idx>(array)...);
            }
        }

        /**
         * Creates a tuple from the provided array.
         */
        template <typename T, size_t N>
        constexpr auto to_tuple(const std::array<T, N>& array) noexcept
        {
            return detail::to_tuple<T>(array, std::make_index_sequence<N>{});
        }

        /**
         * Creates a matching std::tuple from a type_list.
         * Types in the type_list must be Trivial.
         */
        template <typename... Ts>
        constexpr std::tuple<Ts...> as_tuple(type_list<Ts...>) noexcept
        {
            static_assert((... && std::is_trivial_v<Ts>), "Non-trivial types in type_list as not allowed!");
            return {};
        }

        /**
         * Creates a matching type_list from a std::tuple.
         */
        template <typename... Ts>
        constexpr type_list<Ts...> as_type_list(const std::tuple<Ts...>&) noexcept
        {
            return {};
        }

        namespace detail
        {
            template <typename F, typename T>
            constexpr auto invoke_optional_index(F&& f, T&& t, size_t idx, int) -> decltype(f(std::forward<T>(t), idx))
            {
                return f(std::forward<T>(t), idx);
            }

            template <typename F, typename T>
            constexpr auto invoke_optional_index(F&& f, T&& t, size_t, ...) -> decltype(f(std::forward<T>(t)))
            {
                return f(std::forward<T>(t));
            }

            template <typename F, typename... Carry>
            constexpr auto eval_in_order_to_tuple(type_list<>, std::index_sequence<>, F&&, Carry&&... carry)
            {
                if constexpr (sizeof...(Carry) == 0) return std::tuple<>{};
                else return std::make_tuple(std::forward<Carry>(carry)...);
            }

            // This workaround is needed since C++ does not specify
            // the order in which function arguments are evaluated and this leads
            // to incorrect order of evaluation (noticeable when using indexes).
            // Otherwise we could simply do std::make_tuple(f(Ts{}, Idx)...).
            template <typename F, typename T, typename... Ts, size_t I, size_t... Idx, typename... Carry>
            constexpr auto eval_in_order_to_tuple(type_list<T, Ts...>, std::index_sequence<I, Idx...>, F&& f, Carry&&... carry)
            {
                static_assert(std::is_trivial_v<T>, "Argument is a non-trivial type!");

                auto&& result = invoke_optional_index(f, T{}, I, 0);
                return eval_in_order_to_tuple(
                    type_list<Ts...>{},
                    std::index_sequence<Idx...>{},
                    std::forward<F>(f),
                    std::forward<Carry>(carry)..., // carry the previous results over
                    std::forward<decltype(result)>(result) // pass the current result after them
                );
            }

            template <typename F>
            constexpr void eval_in_order(type_list<>, std::index_sequence<>, [[maybe_unused]]F&& f)
            {
            }

            // This workaround is needed since C++ does not specify
            // the order in which function arguments are evaluated and this leads
            // to incorrect order of evaluation (noticeable when using indexes).
            template <typename F, typename T, typename... Ts, size_t I, size_t... Idx>
            constexpr void eval_in_order(type_list<T, Ts...>, std::index_sequence<I, Idx...>, F&& f)
            {
                static_assert(std::is_trivial_v<T>, "Argument is a non-trivial type!");

                invoke_optional_index(f, T{}, I, 0);
                return eval_in_order(
                    type_list<Ts...>{},
                    std::index_sequence<Idx...>{},
                    std::forward<F>(f)
                );
            }
        }

        /**
         * Applies function F to each type in the type_list, aggregating
         * the results in a tuple. F can optionally take an index of type size_t.
         *
         * \code{.cpp}
         * map_to_tuple(reflect_types(type_list<int, float, double>{}), [](auto td) {
         *   return get_name(td);
         * })
         *   -> std::tuple{const_string{"int"}, const_string{"float"}, const_string{"double"}}
         * \endcode
         */
        template <typename F, typename... Ts>
        constexpr auto map_to_tuple(type_list<Ts...> list, F&& f)
        {
            return detail::eval_in_order_to_tuple(list, std::make_index_sequence<sizeof...(Ts)>{}, std::forward<F>(f));
        }

        /**
         * Applies function F to each type in the type_list, aggregating
         * the results in an array. F can optionally take an index of type size_t.
         *
         * \code{.cpp}
         * map_to_array<std::string>(reflect_types(type_list<int, float, double>{}), [](auto td) {
         *   return get_name(td).str();
         * })
         *   -> std::array{std::string{"int"}, std::string{"float"}, std::string{"double"}}
         * \endcode
         */
        template <typename T, typename F, typename... Ts>
        constexpr auto map_to_array(type_list<Ts...> list, F&& f)
        {
            return to_array<T>(map_to_tuple(list, std::forward<F>(f)));
        }

        /**
         * Applies function F to each type in the type_list.
         * F can optionally take an index of type size_t.
         *
         * \code{.cpp}
         * for_each(reflect_types(type_list<int, float, double>{}), [](auto td) {
         *   std::cout << get_name(td) << '\n';
         * });
         * \endcode
         */
        template <typename F, typename... Ts>
        constexpr void for_each(type_list<Ts...> list, F&& f)
        {
            detail::eval_in_order(list, std::make_index_sequence<sizeof...(Ts)>{}, std::forward<F>(f));
        }

        /*
         * Returns the initial_value unchanged.
         */
        /// \private
        template <typename R, typename F, typename... Ts>
        constexpr R accumulate(type_list<>, F&&, R&& initial_value)
        {
            return std::forward<R>(initial_value);
        }

        /*
        * Applies an accumulation function F to each type in the type_list.
        * Note: Breaking changes introduced in v0.7.0:
        *   Behaviour changed to imitate std::accumulate.
        *   F can now no longer take a second index argument.
        */
        template <typename R, typename F, typename T, typename... Ts>
        constexpr auto accumulate(type_list<T, Ts...>, F&& f, R&& initial_value)
        {
            static_assert(std::is_trivial_v<T>, "Argument is a non-trivial type!");

            return accumulate(type_list<Ts...> {},
                std::forward<F>(f),
                std::forward<std::invoke_result_t<F&&, R&&, T&&>>(
                    f(std::forward<R>(initial_value), T {})));
        }

        /**
         * Counts the number of times the predicate F returns true.
        * Note: Breaking changes introduced in v0.7.0:
        *   F can now no longer take a second index argument.
         */
        template <typename F, typename... Ts>
        constexpr size_t count_if(type_list<Ts...> list, F&& f)
        {
            return accumulate<size_t>(list,
                [&](size_t acc, const auto& t) -> size_t { return acc + (f(t) ? 1 : 0); },
                0);
        }

        namespace detail
        {
            template <typename, bool...>
            struct apply_mask;

            template <>
            struct apply_mask<type_list<>>
            {
                using type = type_list<>;
            };

            template <typename T, typename... Ts, bool... Bs>
            struct apply_mask<type_list<T, Ts...>, true, Bs...>
            {
                static_assert(std::is_trivial_v<T>, "Argument is a non-trivial type!");
                using type = trait::prepend_t<T, typename apply_mask<type_list<Ts...>, Bs...>::type>;
            };

            template <typename T, typename... Ts, bool... Bs>
            struct apply_mask<type_list<T, Ts...>, false, Bs...> : apply_mask<type_list<Ts...>, Bs...>
            {
                static_assert(std::is_trivial_v<T>, "Argument is a non-trivial type!");
            };

            template <typename F, typename... Ts>
            constexpr auto filter([[maybe_unused]] F f, type_list<Ts...>)
            {
                return typename apply_mask<type_list<Ts...>, f(Ts{})...>::type{};
            }
        }

        /**
         * Filters the list according to a *constexpr* predicate.
         * Calling f(Ts{})... should be valid in a constexpr context.
         *
         * \code{.cpp}
         * filter(reflect_types(type_list<int, long, float>{}), [](auto td) {
         *   return std::is_integral_v<typename decltype(td)::type>;
         * })
         *   -> type_list<type_descriptor<int>, type_descriptor<long>>
         * \endcode
         */
        template <typename F, typename... Ts>
        constexpr auto filter(type_list<Ts...> list, F&& f)
        {
            return decltype(detail::filter(std::forward<F>(f), list))();
        }

        /**
         * Returns the first instance that matches the *constexpr* predicate.
         * Calling f(Ts{})... should be valid in a constexpr context.
         */
        template <typename F, typename... Ts>
        constexpr auto find_first(type_list<Ts...> list, F&& f)
        {
            using result_list = decltype(detail::filter(std::forward<F>(f), list));
            static_assert(result_list::size != 0, "find_first did not match anything!");
            return trait::get_t<0, result_list>{};
        }

        /**
         * Returns the only instance that matches the *constexpr* predicate.
         * If there is no match or multiple matches, fails with static_assert.
         * Calling f(Ts{})... should be valid in a constexpr context.
         */
        template <typename F, typename... Ts>
        constexpr auto find_one(type_list<Ts...> list, F&& f)
        {
            using result_list = decltype(detail::filter(std::forward<F>(f), list));
            static_assert(result_list::size != 0, "find_one did not match anything!");
            static_assert(result_list::size == 1, "Cannot resolve multiple matches in find_one!");
            return trait::get_t<0, result_list>{};
        }

        /**
         * Returns true if any item in the list matches the predicate.
         * Calling f(Ts{})... should be valid in a constexpr context.
         */
        template <typename F, typename... Ts>
        constexpr bool contains(type_list<Ts...> list, F&& f)
        {
            using result_list = decltype(detail::filter(std::forward<F>(f), list));
            return result_list::size > 0;
        }

        /**
         * Returns true if the type_list contains the specified type.
         * @see refl::trait::contains
         */
        template <typename T, typename... Ts>
        constexpr bool contains(type_list<Ts...>)
        {
            return trait::contains_v<T, type_list<Ts...>>;
        }

        /**
         * Returns true if the tuple contains the specified type or a supertype.
         * @see refl::trait::contains_base
         */
        template <typename T, typename... Ts>
        constexpr bool contains_base(const std::tuple<Ts...>&)
        {
            return trait::contains_base_v<T, type_list<Ts...>>;
        }

        /**
         * Returns true if the tuple contains an instance of the specified type.
         * @see refl::trait::contains_instance
         */
        template <template <typename...> typename T, typename... Ts>
        constexpr bool contains_instance(const std::tuple<Ts...>&)
        {
            return trait::contains_instance_v<T, type_list<Ts...>>;
        }

        /**
         * Applies a function to the elements of the type_list.
         *
         * \code{.cpp}
         * apply(reflect_types(type_list<int, long, float>{}), [](auto td_int, auto td_long, auto td_float) {
         *   return get_name(td_int) + " " +get_name(td_long) + " " + get_name(td_float);
         * })
         *   -> "int long float"
         * \endcode
         */
        template <typename... Ts, typename F>
        constexpr auto apply(type_list<Ts...>, F&& f)
        {
            return f(Ts{}...);
        }

        /** A synonym for std::get<N>(tuple). */
        template <size_t N, typename... Ts>
        constexpr auto& get(std::tuple<Ts...>& ts) noexcept
        {
            return std::get<N>(ts);
        }

        /** A synonym for std::get<N>(tuple). */
        template <size_t N, typename... Ts>
        constexpr const auto& get(const std::tuple<Ts...>& ts) noexcept
        {
            return std::get<N>(ts);
        }

        /** A synonym for std::get<T>(tuple). */
        template <typename T, typename... Ts>
        constexpr T& get(std::tuple<Ts...>& ts) noexcept
        {
            return std::get<T>(ts);
        }

        /** A synonym for std::get<T>(tuple). */
        template <typename T, typename... Ts>
        constexpr const T& get(const std::tuple<Ts...>& ts) noexcept
        {
            return std::get<T>(ts);
        }

        /** Returns the value of type U, where U is a template instance of T. */
        template <template<typename...> typename T, typename... Ts>
        constexpr auto& get_instance(std::tuple<Ts...>& ts) noexcept
        {
            static_assert((... || trait::is_instance_of_v<T, Ts>), "The tuple does not contain a type that is a template instance of T!");
            constexpr size_t idx = static_cast<size_t>(trait::index_of_instance_v<T, type_list<Ts...>>);
            return std::get<idx>(ts);
        }

        /** Returns the value of type U, where U is a template instance of T. */
        template <template<typename...> typename T, typename... Ts>
        constexpr const auto& get_instance(const std::tuple<Ts...>& ts) noexcept
        {
            static_assert((... || trait::is_instance_of_v<T, Ts>), "The tuple does not contain a type that is a template instance of T!");
            constexpr size_t idx = static_cast<size_t>(trait::index_of_instance_v<T, type_list<Ts...>>);
            return std::get<idx>(ts);
        }

        /**
         * Converts a type_list of types to a type_list of the type_descriptors for these types.
         *
         * \code{.cpp}
         * reflect_types(type_list<int, float>{}) -> type_list<type_descriptor<int>, type_descriptor<float>>{}
         * \endcode
         */
        template <typename... Ts>
        constexpr type_list<descriptor::type_descriptor<Ts>...> reflect_types(type_list<Ts...>) noexcept
        {
            return {};
        }

        /**
         * Converts a type_list of type_descriptors to a type_list of the target types.
         *
         * \code{.cpp}
         * unreflect_types(type_list<type_descriptor<int>, type_descriptor<float>>{}) -> type_list<int, float>{}
         * \endcode
         */
        template <typename... Ts>
        constexpr type_list<Ts...> unreflect_types(type_list<descriptor::type_descriptor<Ts>...>) noexcept
        {
            return {};
        }
    } // namespace util

    /**
     * @brief Contains the definitions of the built-in attributes
     *
     * Contains the definitions of the built-in attributes which
     * are implicitly available in macro context as well as the
     * attr::usage namespace which contains constraints
     * for user-provieded attributes.
     *
     * # Examples
     * ```
     * REFL_TYPE(Point, debug(custom_printer))
     *     REFL_FIELD(x)
     *     REFL_FIELD(y)
     * REFL_END
     * ```
     */
    namespace attr
    {
        /**
         * @brief Contains a number of constraints applicable to refl-cpp attributes.
         *
         * Contains base types which create compile-time constraints
         * that are verified by refl-cpp. These base-types must be inherited
         * by custom attribute types.
         */
        namespace usage
        {
            /**
             * Specifies that an attribute type inheriting from this type can
             * only be used with REFL_TYPE()
             */
            struct type {};

            /**
             * Specifies that an attribute type inheriting from this type can
             * only be used with REFL_FUNC()
             */
            struct function {};

            /**
             * Specifies that an attribute type inheriting from this type can
             * only be used with REFL_FIELD()
             */
            struct field {};

            /**
             * Specifies that an attribute type inheriting from this type can
             * only be used with REFL_FUNC or REFL_FIELD.
             */
            struct member : public function, public field{};

            /**
             * Specifies that an attribute type inheriting from this type can
             * only be used with any one of REFL_TYPE, REFL_FIELD, REFL_FUNC.
             */
            struct any : public member, public type {};
        }

        /**
         * Used to decorate a function that serves as a property.
         * Takes an optional friendly name.
         */
        struct property : public usage::function
        {
            const std::optional<const char*> friendly_name;

            constexpr property() noexcept
                : friendly_name{}
            {
            }

            constexpr property(const char* friendly_name) noexcept
                : friendly_name(friendly_name)
            {
            }
        };

        /**
         * Used to specify how a type should be displayed in debugging contexts.
         */
        template <typename F>
        struct debug : public usage::any
        {
            const F write;

            constexpr debug(F write)
                : write(write)
            {
            }
        };

        /**
         * Used to specify the base types of the target type.
         */
        template <typename... Ts>
        struct base_types : usage::type
        {
            /** An alias for a type_list of the base types. */
            typedef type_list<Ts...> list_type;

            /** An instance of a type_list of the base types. */
            static constexpr list_type list{ };
        };

        /**
         * Used to specify the base types of the target type.
         */
        template <typename... Ts>
        [[maybe_unused]] static constexpr base_types<Ts...> bases{ };
    } // namespace attr


    namespace detail
    {
        namespace macro_exports
        {
            using attr::property;
            using attr::debug;
            using attr::bases;
        }
    }

    namespace trait
    {
        namespace detail
        {
            template <typename T>
            auto member_type_test(int) -> decltype(typename T::member_type{}, std::true_type{});

            template <typename T>
            std::false_type member_type_test(...);
        }

        /**
         * A trait for detecting whether the type 'T' is a member descriptor.
         */
        template <typename T>
        struct is_member : decltype(detail::member_type_test<T>(0))
        {
        };

        /**
         * A trait for detecting whether the type 'T' is a member descriptor.
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_member_v{ is_member<T>::value };

        namespace detail
        {
            template <typename T>
            struct is_field_2 : std::is_base_of<typename T::member_type, member::field>
            {
            };
        }

        /**
         * A trait for detecting whether the type 'T' is a field descriptor.
         */
        template <typename T>
        struct is_field : std::conjunction<is_member<T>, detail::is_field_2<T>>
        {
        };

        /**
         * A trait for detecting whether the type 'T' is a field descriptor.
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_field_v{ is_field<T>::value };

        namespace detail
        {
            template <typename T>
            struct is_function_2 : std::is_base_of<typename T::member_type, member::function>
            {
            };
        }

        /**
         * A trait for detecting whether the type 'T' is a function descriptor.
         */
        template <typename T>
        struct is_function : std::conjunction<is_member<T>, detail::is_function_2<T>>
        {
        };

        /**
         * A trait for detecting whether the type 'T' is a function descriptor.
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_function_v{ is_function<T>::value };

        /**
         * Detects whether the type T is a type_descriptor.
         * Inherits from std::bool_constant<>.
         */
        template <typename T>
        struct is_type : is_instance_of<descriptor::type_descriptor, T>
        {
        };

        /**
         * Detects whether the type T is a type_descriptor.
         * @see is_type
         */
        template <typename T>
        [[maybe_unused]] constexpr bool is_type_v{ is_type<T>::value };

        /**
         * A trait for detecting whether the type 'T' is a refl-cpp descriptor.
         */
        template <typename T>
        struct is_descriptor : std::disjunction<is_type<T>, is_member<T>>
        {
        };

        /**
         * A trait for detecting whether the type 'T' is a refl-cpp descriptor.
         */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_descriptor_v{ is_descriptor<T>::value };


        /** Checks whether T is marked as a property. */
        template <typename T>
        struct is_property : std::bool_constant<
            trait::is_function_v<T> && trait::contains_v<attr::property, typename T::attribute_types>>
        {
        };

        /** Checks whether T is marked as a property. */
        template <typename T>
        [[maybe_unused]] static constexpr bool is_property_v{ is_property<T>::value };
    } // namespace trait

    /**
     * @brief Contains the basic reflection primitives
     * as well as functions operating on those primitives
     */
    namespace descriptor
    {
        namespace detail
        {
            template <typename Member>
            struct static_field_invoker
            {
                static constexpr auto invoke() -> decltype(*Member::pointer)
                {
                    return *Member::pointer;
                }

                template <typename U, typename M = Member, std::enable_if_t<M::is_writable, int> = 0>
                static constexpr auto invoke(U&& value) -> decltype(*Member::pointer = std::forward<U>(value))
                {
                    return *Member::pointer = std::forward<U>(value);
                }
            };

            template <typename Member>
            struct instance_field_invoker
            {
                template <typename T>
                static constexpr auto invoke(T&& target) -> decltype(target.*(Member::pointer))
                {
                    return target.*(Member::pointer);
                }

                template <typename T, typename U, typename M = Member, std::enable_if_t<M::is_writable, int> = 0>
                static constexpr auto invoke(T&& target, U&& value) -> decltype(target.*(Member::pointer) = std::forward<U>(value))
                {
                    return target.*(Member::pointer) = std::forward<U>(value);
                }
            };

            template <typename Member>
            static_field_invoker<Member> field_type_switch(std::true_type);

            template <typename Member>
            instance_field_invoker<Member> field_type_switch(std::false_type);

            template <typename Member>
            constexpr decltype(nullptr) get_function_pointer(...)
            {
                return nullptr;
            }

            template <typename Member>
            constexpr auto get_function_pointer(int) -> decltype(Member::pointer())
            {
                return Member::pointer();
            }

            template <typename Member, typename Pointer>
            constexpr decltype(nullptr) resolve_function_pointer(...)
            {
                return nullptr;
            }

            template <typename Member, typename Pointer>
            constexpr auto resolve_function_pointer(int) -> decltype(Member::template resolve<Pointer>())
            {
                return Member::template resolve<Pointer>();
            }

            template <typename T, size_t N>
            using make_descriptor = std::conditional_t<refl::trait::is_field_v<refl::detail::member_info<T, N>>,
                field_descriptor<T, N>,
                std::conditional_t<refl::trait::is_function_v<refl::detail::member_info<T, N>>,
                    function_descriptor<T, N>,
                    void
                >>;

            template <typename T>
            type_list<> enumerate_members(std::index_sequence<>);

            template <typename T, size_t... Idx>
            type_list<make_descriptor<T, Idx>...> enumerate_members(std::index_sequence<Idx...>);

            template <typename T>
            struct declared_member_list
            {
                static_assert(refl::trait::is_reflectable_v<T>, "This type does not support reflection!");
                using type = decltype(enumerate_members<T>(std::make_index_sequence<refl::detail::type_info<T>::member_count>{}));
            };

            template <typename T>
            using attribute_types = trait::as_type_list_t<std::remove_cv_t<decltype(refl::detail::type_info<T>::attributes)>>;

            template <typename>
            struct flatten;

            template <typename... TypeLists>
            struct flatten<type_list<TypeLists...>> : trait::concat<TypeLists...>
            {
            };

            template <typename T, typename Base>
            static constexpr void validate_base()
            {
                static_assert(std::is_base_of_v<Base, T>, "Base is not a base type of T!");
            }

            template <typename T, typename... Bases>
            static constexpr void validate_bases(type_list<Bases...>)
            {
                util::ignore((validate_base<T, Bases>(), 0)...);
            }

            template <typename T>
            static constexpr auto get_declared_base_type_list()
            {
                if constexpr (trait::contains_instance_v<attr::base_types, attribute_types<T>>) {
                    using base_types_type = trait::remove_qualifiers_t<decltype(util::get_instance<attr::base_types>(refl::detail::type_info<T>::attributes))>;
                    validate_bases<T>(base_types_type::list);
                    return typename base_types_type::list_type{};
                }
                else {
                    return type_list<>{};
                }
            }

            template <typename T>
            struct declared_base_type_list
            {
                using type = decltype(get_declared_base_type_list<T>());
            };

            template <typename T>
            struct base_type_list;

            template <typename T>
            static constexpr auto get_base_type_list()
            {
                if constexpr (trait::contains_instance_v<attr::base_types, attribute_types<T>>) {
                    using declared_bases = typename declared_base_type_list<T>::type;
                    using rec_bases = typename flatten<trait::map_t<base_type_list, declared_bases>>::type;
                    return trait::unique_t<trait::concat_t<declared_bases, rec_bases>>{};
                }
                else {
                    return type_list<>{};
                }
            }

            template <typename T>
            struct base_type_list
            {
                using type = decltype(get_base_type_list<T>());
            };

            template <typename T>
            struct member_list : flatten<trait::map_t<declared_member_list, trait::prepend_t<T, typename base_type_list<T>::type>>>
            {
            };

        } // namespace detail

        /** A type_list of the declared member descriptors of the target type T. */
        template <typename T>
        using declared_member_list = typename detail::declared_member_list<T>::type;

        /** A type_list of the declared and inherited member descriptors of the target type T. */
        template <typename T>
        using member_list = typename detail::member_list<T>::type;

        /**
         * @brief The base type for member descriptors.
         */
        template <typename T, size_t N>
        class member_descriptor_base
        {
        protected:

            typedef refl::detail::member_info<T, N> member;

        public:

            /**
             * An alias for the declaring type of the reflected member.
             *
             * \code{.cpp}
             * struct Foo { const int* x; };
             * REFL_AUTO(type(Foo), field(x))
             *
             * get_t<0, member_list<Foo>>::declaring_type -> Foo
             * \endcode
             */
            typedef T declaring_type;

            /** An alias specifying the member type of member. */
            typedef typename member::member_type member_type;

            /**
             * An alias specifying the types of the attributes of the member. (Removes CV-qualifiers.)
             * \copydetails refl::descriptor::get_attribute_types
             */
            typedef trait::as_type_list_t<std::remove_cv_t<decltype(member::attributes)>> attribute_types;

            /**
             * The type_descriptor of the declaring type.
             * \copydetails refl::descriptor::get_declarator
             */
            static constexpr type_descriptor<T> declarator{ };

            /**
             * The name of the reflected member.
             * \copydetails refl::descriptor::get_name
             */
            static constexpr auto name{ member::name };

            /**
             * The attributes of the reflected member.
             * \copydetails refl::descriptor::get_attributes
             */
            static constexpr auto attributes{ member::attributes };

        };

        /**
         * @brief Represents a reflected field.
         */
        template <typename T, size_t N>
        class field_descriptor : public member_descriptor_base<T, N>
        {
            using typename member_descriptor_base<T, N>::member;
            static_assert(trait::is_field_v<member>);

        public:

            /**
             * Type value type of the member.
             *
             * \code{.cpp}
             * struct Foo { const int* x; };
             * REFL_AUTO(type(Foo), field(x))
             *
             * get_t<0, member_list<Foo>>::value_type -> const int*
             * \endcode
             */
            typedef typename member::value_type value_type;

            /**
             * Whether the field is static or not.
             * \copydetails refl::descriptor::is_static
             */
            static constexpr bool is_static{ !std::is_member_object_pointer_v<decltype(member::pointer)> };

            /**
             * Whether the field is const or not.
             * @see refl::descriptor::is_const
             */
            static constexpr bool is_writable{ !std::is_const_v<value_type> };

            /**
             * A member pointer to the reflected field of the appropriate type.
             * \copydetails refl::descriptor::get_pointer
             */
            static constexpr auto pointer{ member::pointer };

        private:

            using invoker = decltype(detail::field_type_switch<field_descriptor>(std::bool_constant<is_static>{}));

        public:

            /**
             * Returns the value of the field. (for static fields).
             * \copydetails refl::descriptor::invoke
             */
            template <decltype(nullptr) = nullptr>
            static constexpr decltype(auto) get() noexcept
            {
                return *member::pointer;
            }

            /**
             * Returns the value of the field. (for instance fields).
             * \copydetails refl::descriptor::invoke
             */
            template <typename U>
            static constexpr decltype(auto) get(U&& target) noexcept
            {
                return target.*(member::pointer);
            }

            /**
             * A synonym for get().
             * \copydetails refl::descriptor::invoke
             */
            template <typename... Args>
            constexpr auto operator()(Args&&... args) const noexcept -> decltype(invoker::invoke(std::forward<Args>(args)...))
            {
                return invoker::invoke(std::forward<Args>(args)...);
            }

        };

        /**
         * @brief Represents a reflected function.
         */
        template <typename T, size_t N>
        class function_descriptor : public member_descriptor_base<T, N>
        {
            using typename member_descriptor_base<T, N>::member;
            static_assert(trait::is_function_v<member>);

        public:

            /**
             * Invokes the function with the given arguments.
             * If the function is an instance function, a reference
             * to the instance is provided as first argument.
             * \copydetails refl::descriptor::invoke
             */
            template <typename... Args>
            static constexpr auto invoke(Args&&... args) -> decltype(member::invoke(std::declval<Args>()...))
            {
                return member::invoke(std::forward<Args>(args)...);
            }

            /**
             * The return type of an invocation of this member with Args... (as if by invoke(...)).
             * \copydetails refl::descriptor::return_type
             */
            template <typename... Args>
            using return_type = decltype(member::invoke(std::declval<Args>()...));

            /**
             * A synonym for invoke(args...).
             * \copydetails refl::descriptor::invoke
             */
            template <typename... Args>
            constexpr auto operator()(Args&&... args) const -> decltype(invoke(std::declval<Args>()...))
            {
                return invoke(std::forward<Args>(args)...);
            }

            /**
             * Returns a pointer to a non-overloaded function.
             * \copydetails refl::descriptor::get_pointer
             */
            static constexpr auto pointer{ detail::get_function_pointer<member>(0) };

            /**
             * Whether the pointer member was correctly resolved to a concrete implementation.
             * If this field is false, resolve() would need to be called instead.
             * \copydetails refl::descriptor::is_resolved
             */
            static constexpr bool is_resolved{ !std::is_same_v<decltype(pointer), const decltype(nullptr)> };

            /**
             * Whether the pointer can be resolved as with the specified type.
             * \copydetails refl::descriptor::can_resolve
             */
            template <typename Pointer>
            static constexpr bool can_resolve()
            {
                return !std::is_same_v<decltype(resolve<Pointer>()), decltype(nullptr)>;
            }

            /**
             * Resolves the function pointer as being of type Pointer.
             * Required when taking a pointer to an overloaded function.
             *
             * \copydetails refl::descriptor::resolve
             */
            template <typename Pointer>
            static constexpr auto resolve()
            {
                return detail::resolve_function_pointer<member, Pointer>(0);
            }

        };

        /** Represents a reflected type. */
        template <typename T>
        class type_descriptor
        {
        private:

            static_assert(refl::trait::is_reflectable_v<T>, "This type does not support reflection!");

            typedef refl::detail::type_info<T> type_info;

        public:

            /**
             * The reflected type T.
             *
             * \code{.cpp}
             * struct Foo {};
             * REFL_AUTO(type(Foo))
             *
             * type_descriptor<Foo>::type -> Foo
             * \endcode
             */
            typedef T type;

            /**
             * The declared base types (via base_types<Ts...> attribute) of T.
             * \copydetails refl::descriptor::get_declared_base_types
             */
            typedef typename detail::declared_base_type_list<T>::type declared_base_types;

            /**
             * The declared and inherited base types of T.
             * \copydetails refl::descriptor::get_base_types
             */
            typedef typename detail::base_type_list<T>::type base_types;

            /**
             * A synonym for declared_member_list<T>.
             * \copydetails refl::descriptor::declared_member_list
             */
            typedef declared_member_list<T> declared_member_types;

            /**
             * A synonym for member_list<T>.
             * \copydetails refl::descriptor::member_list
             */
            typedef member_list<T> member_types;

            /**
             * An alias specifying the types of the attributes of the member. (Removes CV-qualifiers.)
             * \copydetails refl::descriptor::get_attribute_types
             */
            typedef detail::attribute_types<T> attribute_types;

            /**
             * The declared base types (via base_types<Ts...> attribute) of T.
             * \copydetails refl::descriptor::get_declared_base_types
             */
            static constexpr declared_base_types declared_bases{};

            /**
             * The declared  and inherited base types of T.
             * \copydetails refl::descriptor::get_base_types
             */
            static constexpr base_types bases{};

            /**
             * The list of declared member descriptors.
             * \copydetails refl::descriptor::get_declared_members
             */
            static constexpr declared_member_types declared_members{  };

            /**
             * The list of declared and inherited member descriptors.
             * \copydetails refl::descriptor::get_members
             */
            static constexpr member_types members{  };

            /**
             * The name of the reflected type.
             * \copydetails refl::descriptor::get_name
             */
            static constexpr const auto name{ type_info::name };

            /**
             * The attributes of the reflected type.
             * \copydetails refl::descriptor::get_attributes
              */
            static constexpr const auto attributes{ type_info::attributes };

        };

        /**
         * Returns the full name of the descriptor
         *
         * \code{.cpp}
         * namespace ns {
         *   struct Foo {
         *     int x;
         *   };
         * }
         * REFL_AUTO(type(ns::Foo), field(x))
         *
         * get_name(reflect<Foo>()) -> "ns::Foo"
         * get_name(get_t<0, member_list<Foo>>()) -> "x"
         * \endcode
         */
        template <typename Descriptor>
        constexpr auto get_name(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return d.name;
        }

        /**
         * Returns a const reference to the descriptor's attribute tuple.
         *
         * \code{.cpp}
         * struct Foo {};
         * REFL_AUTO(type(Foo, bases<>, ns::serializable()))
         *
         * get_attributes(reflect<Foo>()) -> const std::tuple<attr::base_types<>, ns::serializable>&
         * \endcode
         */
        template <typename Descriptor>
        constexpr const auto& get_attributes(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return d.attributes;
        }

        /**
         * Returns a type_list of the descriptor's attribute types.
         *
         * \code{.cpp}
         * struct Foo {};
         * REFL_AUTO(type(Foo, bases<>, ns::serializable()))
         *
         * get_attribute_types(reflect<Foo>()) -> type_list<attr::base_types<>, ns::serializable>
         * \endcode
         */
        template <typename Descriptor>
        constexpr auto get_attribute_types(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return trait::as_type_list_t<std::remove_cv_t<decltype(d.attributes)>>{};
        }

        /**
         * Returns a type_list of the declared base types of the type.
         * Combine with reflect_types to obtain type_descriptors for those types.
         * @see reflect_types
         *
         * \code{.cpp}
         * struct Animal {};
         * REFL_AUTO(type(Animal))
         * struct Mammal : Animal {};
         * REFL_AUTO(type(Mammal, bases<Animal>))
         * struct Dog : Mammal {}:
         * REFL_AUTO(type(Dog, bases<Mammal>))
         *
         * get_base_types(reflect<Dog>()) -> type_list<Mammal>
         * \endcode
         */
        template <typename TypeDescriptor>
        constexpr auto get_declared_base_types(TypeDescriptor t) noexcept
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            return t.declared_bases;
        }

        /**
         * Returns a type_list of the declared and inherited base types of the type.
         * Combine with reflect_types to obtain type_descriptors for those types.
         * @see reflect_types
         *
         * \code{.cpp}
         * struct Animal {};
         * REFL_AUTO(type(Animal))
         * struct Mammal : Animal {};
         * REFL_AUTO(type(Mammal, bases<Animal>))
         * struct Dog : Mammal {}:
         * REFL_AUTO(type(Dog, bases<Mammal>))
         *
         * get_base_types(reflect<Dog>()) -> type_list<Mammal, Animal>
         * \endcode
         */
        template <typename TypeDescriptor>
        constexpr auto get_base_types(TypeDescriptor t) noexcept
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            return t.bases;
        }

        /**
         * Returns a type_list of the declared members of the type.
         *
         * \code{.cpp}
         * struct Base {
         *  int val;
         * };
         * struct Foo : Base {
         *   int bar, baz;
         * };
         * REFL_AUTO(type(Foo, bases<Base>), field(bar), field(baz))
         * get_declared_members(reflect<Foo>()) -> type_list<field_descriptor<Foo, 0> /bar/, field_descriptor<Foo, 1> /baz/>
         * \endcode
         */
        template <typename TypeDescriptor>
        constexpr auto get_declared_members(TypeDescriptor t) noexcept
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            return t.declared_members;
        }

        /**
         * Returns a type_list of the declared and inherited members of the type.
         *
         * \code{.cpp}
         * struct Base {
         *  int val;
         * };
         * struct Foo : Base {
         *   int bar, baz;
         * };
         * REFL_AUTO(type(Foo, bases<Base>), field(bar), field(baz))
         * get_members(reflect<Foo>()) -> type_list<field_descriptor<Foo, 0> /bar/, field_descriptor<Foo, 1> /baz/, field_descriptor<Base, 0> /val/>
         * \endcode
         */
        template <typename TypeDescriptor>
        constexpr auto get_members(TypeDescriptor t) noexcept
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            return t.members;
        }

        /**
         * Returns the type_descriptor of declaring type of the member.
         *
         * \code{.cpp}
         * struct Foo {
         *   int bar;
         * };
         * REFL_AUTO(type(Foo), field(bar)
         * get_declarator(get_t<0, member_list<Foo>>()) -> type_descriptor<Foo>{}
         * \endcode
         */
        template <typename MemberDescriptor>
        constexpr auto get_declarator(MemberDescriptor d) noexcept
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            return d.declarator;
        }

        /**
         * Returns a pointer to the reflected field/function.
         * When the member is a function, the return value might be nullptr
         * if the type of the function pointer cannot be resolved.
         * @see is_resolved
         * @see can_resolve
         * @see resolve
         *
         * \code{.cpp}
         * struct Foo {
         *   int bar;
         *   static int baz;
         * };
         * REFL_AUTO(type(Foo), field(bar), field(baz))
         * get_pointer(get_t<0, member_list<Foo>>()) -> (int Foo::*) &Foo::bar
         * get_pointer(get_t<1, member_list<Foo>>()) -> (int*) &Foo::baz
         * \endcode
         */
        template <typename MemberDescriptor>
        constexpr auto get_pointer(MemberDescriptor d) noexcept
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            return d.pointer;
        }

        /**
         * Invokes the member with the specified arguments.
         *
         * \code{.cpp}
         * struct Foo {
         *   int bar = 1;
         *   static int baz = 5;
         *   void foobar(int x) { return x * 2; }
         *   static void foobaz(int x) { return x * 3; }
         * };
         * REFL_AUTO(type(Foo), field(bar), field(baz), func(foobar), func(foobaz))
         * invoke(get_t<0, member_list<Foo>(), Foo()) -> 1 (Foo().bar)
         * invoke(get_t<1, member_list<Foo>>()) -> 5 (Foo::baz)
         * invoke(get_t<2, member_list<Foo>(), Foo(), 10) -> 20 (Foo().foobar())
         * invoke(get_t<3, member_list<Foo>>()) -> 30 (Foo::foobaz())
         * \endcode
         */
        template <typename MemberDescriptor, typename... Args>
        constexpr auto invoke(MemberDescriptor d, Args&&... args) noexcept -> decltype(d(std::forward<Args>(args)...))
        {
            return d(std::forward<Args>(args)...);
        }

        /**
         * Checks whether the field is declared as static.
         *
         * \code{.cpp}
         * struct Foo {
         *   int bar;
         *   static int baz;
         * };
         * REFL_AUTO(type(Foo), field(bar), field(baz))
         * is_static(get_t<0, member_list<Foo>>()) -> false
         * is_static(get_t<1, member_list<Foo>>()) -> true
         * \endcode
         */
        template <typename FieldDescriptor>
        constexpr auto is_static(FieldDescriptor d) noexcept
        {
            static_assert(trait::is_field_v<FieldDescriptor>);
            return d.is_static;
        }

        /**
         * Checks whether the value type of the field is const-qualified.
         *
         * \code{.cpp}
         * struct Foo {
         *   int bar;
         *   const int baz;
         * };
         * REFL_AUTO(type(Foo), field(bar), field(baz))
         * is_const(get_t<0, member_list<Foo>>()) -> false
         * is_const(get_t<1, member_list<Foo>>()) -> true
         * \endcode
         */
        template <typename FieldDescriptor>
        constexpr auto is_const(FieldDescriptor d) noexcept
        {
            static_assert(trait::is_field_v<FieldDescriptor>);
            return d.is_const;
        }

        /**
         * The return type when invoking the specified descriptor using the provided argument types.
         * Argument coversion will be applied as per C++ rules.
         */
        template <typename FunctionDescriptor, typename... Args>
        using result_type = typename FunctionDescriptor::template result_type<Args...>;

        /**
         * Checks whether the function pointer was automatically resolved.
         *
         * \code{.cpp}
         * struct Foo {
         *   void bar();
         *   void bar(int);
         *   void baz();
         * };
         * REFL_AUTO(type(Foo), func(bar), func(baz))
         * is_resolved(get_t<0, member_list<Foo>>()) -> false
         * is_resolved(get_t<1, member_list<Foo>>()) -> true
         * \endcode
         */
        template <typename FunctionDescriptor>
        constexpr auto is_resolved(FunctionDescriptor d) noexcept
        {
            static_assert(trait::is_function_v<FunctionDescriptor>);
            return d.is_resolved;
        }

        /**
         * Checks whether the function pointer can be resolved as
         * a pointer of the specified type.
         *
         * \code{.cpp}
         * struct Foo {
         *   void bar();
         *   void bar(int);
         * };
         * REFL_AUTO(type(Foo), func(bar))
         * can_resolve<void(Foo::*)()>(get_t<0, member_list<Foo>>()) -> true
         * can_resolve<void(Foo::*)(int)>(get_t<0, member_list<Foo>>()) -> true
         * can_resolve<void(Foo::*)(std::string)>(get_t<0, member_list<Foo>>()) -> false
         * \endcode
         */
        template <typename Pointer, typename FunctionDescriptor>
        constexpr auto can_resolve(FunctionDescriptor d) noexcept
        {
            static_assert(trait::is_function_v<FunctionDescriptor>);
            return d.template can_resolve<Pointer>();
        }

        /**
         * Resolves the function pointer as a pointer of the specified type.
         *
         * \code{.cpp}
         * struct Foo {
         *   void bar();
         *   void bar(int);
         * };
         * REFL_AUTO(type(Foo), func(bar))
         * resolve<void(Foo::*)()>(get_t<0, member_list<Foo>>()) -> <&Foo::bar()>
         * resolve<void(Foo::*)(int)>(get_t<0, member_list<Foo>>()) -> <&Foo::bar(int)>
         * resolve<void(Foo::*)(std::string)>(get_t<0, member_list<Foo>>()) -> nullptr
         * \endcode
         */
        template <typename Pointer, typename FunctionDescriptor>
        constexpr auto resolve(FunctionDescriptor d) noexcept
        {
            static_assert(trait::is_function_v<FunctionDescriptor>);
            return d.template resolve<Pointer>();
        }

        /**
         * Checks whether T is a field descriptor.
         *
         * @see refl::descriptor::field_descriptor
         *
         * \code{.cpp}
         * REFL_AUTO(type(Foo), func(bar), field(baz))
         * is_function(get_t<0, member_list<Foo>>()) -> false
         * is_function(get_t<1, member_list<Foo>>()) -> true
         * \endcode
         */
        template <typename Descriptor>
        constexpr bool is_field(Descriptor) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return trait::is_field_v<Descriptor>;
        }

        /**
         * Checks whether T is a function descriptor.
         *
         * @see refl::descriptor::function_descriptor
         *
         * \code{.cpp}
         * REFL_AUTO(type(Foo), func(bar), field(baz))
         * is_function(get_t<0, member_list<Foo>>()) -> true
         * is_function(get_t<1, member_list<Foo>>()) -> false
         * \endcode
         */
        template <typename Descriptor>
        constexpr bool is_function(Descriptor) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return trait::is_function_v<Descriptor>;
        }

        /**
         * Checks whether T is a type descriptor.
         *
         * @see refl::descriptor::type_descriptor
         *
         * \code{.cpp}
         * REFL_AUTO(type(Foo))
         * is_type(reflect<Foo>>()) -> true
         * \endcode
         */
        template <typename Descriptor>
        constexpr bool is_type(Descriptor) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return trait::is_type_v<Descriptor>;
        }

        /**
         * Checks whether T has an attribute of type A.
         *
         * \code{.cpp}
         * REFL_AUTO(type(User), func(get_name, property()), func(set_name, property()))
         * has_attribute<attr::property>(get_t<0, member_list<User>>{}) -> true
         * \endcode
         */
        template <typename A, typename Descriptor>
        constexpr bool has_attribute(Descriptor) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return trait::contains_base_v<A, typename Descriptor::attribute_types>;
        }

        /**
         * Checks whether T has an attribute of that is a template instance of A.
         *
         * \code{.cpp}
         * REFL_AUTO(type(Random, debug{ [](auto os, auto){ os << "[Random]"; } }))
         * has_attribute<attr::debug>(reflect<Random>()) -> true
         * \endcode
         */
        template <template<typename...> typename A, typename Descriptor>
        constexpr bool has_attribute(Descriptor) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return trait::contains_instance_v<A, typename Descriptor::attribute_types>;
        }

        /**
         * Returns the value of the attribute A on T.
         *
         * \code{.cpp}
         * REFL_AUTO(type(User), func(get_name, property()), func(set_name, property()))
         * get_attribute<attr::property>(get_t<0, member_list<User>>{}) -> property{ friendly_name = nullopt }
         * \endcode
         */
        template <typename A, typename Descriptor>
        constexpr const A& get_attribute(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return util::get<A>(d.attributes);
        }

        /**
         * Returns the value of the attribute A on T.
         *
         * \code{.cpp}
         * REFL_AUTO(type(Random, debug{ [](auto os, auto){ os << "[Random]"; } }))
         * get_attribute<attr::debug>(reflect<Random>()) -> instance of debug<LambdaType>
         * \endcode
         */
        template <template<typename...> typename A, typename Descriptor>
        constexpr const auto& get_attribute(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return util::get_instance<A>(d.attributes);
        }

        /**
         * Checks whether T is a member descriptor marked with the property attribute.
         *
         * @see refl::attr::property
         * @see refl::descriptor::get_property
         *
         * \code{.cpp}
         * REFL_AUTO(type(User), func(get_name, property("user_name")), func(set_name, property()))
         * is_property(get_t<0, member_list<User>>{}) -> true
         * \endcode
         */
        template <typename MemberDescriptor>
        constexpr bool is_property(MemberDescriptor d) noexcept
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            return has_attribute<attr::property>(d);
        }

        /**
         * Gets the property attribute.
         *
         * @see refl::attr::property
         * @see refl::descriptor::is_property
         *
         * \code{.cpp}
         * REFL_AUTO(type(User), func(get_name, property("user_name")), func(set_name, property()))
         * *get_property(get_t<0, member_list<User>>{}).friendly_name -> "user_name"
         * \endcode
         */
        template <typename FunctionDescriptor>
        constexpr attr::property get_property(FunctionDescriptor d) noexcept
        {
            static_assert(trait::is_function_v<FunctionDescriptor>);
            return get_attribute<attr::property>(d);
        }

        namespace detail
        {
            struct placeholder
            {
                template <typename T>
                operator T() const;
            };
        } // namespace detail

        /**
         * Checks if T is a 0-arg const-qualified member function with a property attribute or a field.
         *
         * \code{.cpp}
         * REFL_AUTO(type(User), func(get_name, property()), func(set_name, property()))
         * is_readable(get_t<0, member_list<User>>{}) -> true
         * is_readable(get_t<1, member_list<User>>{}) -> false
         * \endcode
         */
        template <typename MemberDescriptor>
        constexpr bool is_readable(MemberDescriptor) noexcept
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            if constexpr (trait::is_property_v<MemberDescriptor>) {
                if constexpr (std::is_invocable_v<MemberDescriptor, const typename MemberDescriptor::declaring_type&>) {
                    using return_type = typename MemberDescriptor::template return_type<const typename MemberDescriptor::declaring_type&>;
                    return !std::is_void_v<return_type>;
                }
                else {
                    return false;
                }
            }
            else {
                return trait::is_field_v<MemberDescriptor>;
            }
        }

        /**
         * Checks if T is a 1-arg non-const-qualified member function with a property attribute or a non-const field.
         *
         * \code{.cpp}
         * struct User { std::string get_name() const; }
         * REFL_AUTO(type(User), func(get_name, property()), func(set_name, property()))
         * is_writable(get_t<0, member_list<User>>{}) -> false
         * is_writable(get_t<1, member_list<User>>{}) -> true
         * \endcode
         */
        template <typename MemberDescriptor>
        constexpr bool is_writable(MemberDescriptor) noexcept
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            if constexpr (trait::is_property_v<MemberDescriptor>) {
                return std::is_invocable_v<MemberDescriptor, typename MemberDescriptor::declaring_type&, detail::placeholder>;
            }
            else if constexpr (trait::is_field_v<MemberDescriptor>) {
                return !std::is_const_v<typename trait::remove_qualifiers_t<MemberDescriptor>::value_type>;
            }
            else {
                return false;
            }
        }

        namespace detail
        {
            template <typename T>
            struct get_type_descriptor
            {
                typedef type_descriptor<T> type;
            };
        } // namespace detail

        /**
         * Checks if a type has a bases attribute.
         *
         * @deprecated Use has_base_types in combination with reflect_types instead.
         * @see refl::attr::bases
         * @see refl::descriptor::get_bases
         *
         * \code{.cpp}
         * REFL_AUTO(type(Dog, bases<Animal>))
         * has_bases(reflect<Dog>()) -> true
         * \endcode
         */
        template <typename TypeDescriptor>
        [[deprecated]] constexpr auto has_bases(TypeDescriptor t) noexcept
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            return has_attribute<attr::base_types>(t);
        }

        /**
         * Returns a list of the type_descriptor<T>s of the base types of the target,
         * as specified by the bases<A, B, ...> attribute.
         *
         * @deprecated Use get_base_types in combination with reflect_types instead.
         * @see refl::attr::bases
         * @see refl::descriptor::has_bases
         *
         * \code{.cpp}
         * REFL_AUTO(type(Dog, bases<Animal>))
         * get_bases(reflect<Dog>()) -> type_list<type_descriptor<Animal>>
         * \endcode
         */
        template <typename TypeDescriptor>
        [[deprecated]] constexpr auto get_bases(TypeDescriptor t) noexcept
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            static_assert(has_bases(t), "Target type does not have a bases<A, B, ...> attribute.");

            constexpr auto bases = get_attribute<attr::base_types>(t);
            using base_types = typename decltype(bases)::list_type;
            return trait::map_t<detail::get_type_descriptor, base_types>{};
        }

        /**
         * Returns the unqualified name of the type, discarding the namespace and typenames (if a template type).
         *
         * \code{.cpp}
         * get_simple_name(reflect<std::vector<float>>()) -> "vector"
         * \endcode
         */
        template <typename TypeDescriptor>
        constexpr auto get_simple_name(TypeDescriptor t)
        {
            static_assert(trait::is_type_v<TypeDescriptor>);
            constexpr size_t template_start = t.name.find('<');
            constexpr size_t scope_last = t.name.rfind(':', template_start);
            if constexpr (scope_last == const_string<0>::npos) {
                return t.name;
            }
            else {
                return t.name.template substr<scope_last + 1, template_start - scope_last - 1>();
            }
        }

        /**
         * Returns the debug name of T (In the form of 'declaring_type::member_name') as a const_string.
         *
         * \code{.cpp}
         * REFL_AUTO(type(Point), field(x), field(y))
         * get_debug_name_const(get_t<0, member_list<Point>>{}) -> "Point::x"
         * \endcode
         */
        template <typename MemberDescriptor>
        constexpr auto get_debug_name_const(MemberDescriptor d)
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            return d.declarator.name + "::" + d.name;
        }

        /**
         * Returns the debug name of T. (In the form of 'declaring_type::member_name').
         * \code{.cpp}
         * REFL_AUTO(type(Point), field(x), field(y))
         * get_debug_name(get_t<0, member_list<Point>>{}) -> "Point::x"
         * \endcode
         */
        template <typename MemberDescriptor>
        const char* get_debug_name(MemberDescriptor d)
        {
            static_assert(trait::is_member_v<MemberDescriptor>);
            static const std::string name(get_debug_name_const(d).str());
            return name.c_str();
        }

        namespace detail
        {
            constexpr bool is_upper(char ch)
            {
                return ch >= 'A' && ch <= 'Z';
            }

            constexpr char to_upper(char ch)
            {
                return ch >= 'a' && ch <= 'z'
                    ? char(ch + ('A' - 'a'))
                    : ch;
            }

            constexpr char to_lower(char ch)
            {
                return ch >= 'A' && ch <= 'Z'
                    ? char(ch + ('a' - 'A'))
                    : ch;
            }

            template <typename T, bool PreferUpper>
            constexpr auto normalize_bare_accessor_name()
            {
                constexpr auto str = T::name.template substr<3>();
                if constexpr (str.data[0] == '_') {
                    return str.template substr<1>();
                }
                else if constexpr (!PreferUpper && str.data[0] >= 'A' && str.data[0] <= 'Z') {
                    return make_const_string(to_lower(str.data[0])) + str.template substr<1>();
                }
                else if constexpr (PreferUpper) {
                    return make_const_string(to_upper(str.data[0])) + str.template substr<1>();
                }
                else {
                    return str;
                }
            }

            template <typename T>
            constexpr auto normalize_accessor_name(const T)
            {
                constexpr T t{};
                if constexpr (t.name.size > 3) {
                    constexpr auto prefix = t.name.template substr<0, 3>();
                    constexpr bool cont_snake_or_camel = (t.name.size > 4 && t.name.data[3] == '_' && !is_upper(t.name.data[4])) || is_upper(t.name.data[3]);
                    constexpr bool cont_pascal = is_upper(t.name.data[3]);

                    if constexpr ((is_readable(T{}) && ((prefix == "Get" && cont_pascal) || (prefix == "get" && cont_snake_or_camel)))
                        || (is_writable(T{}) && ((prefix == "Set" && cont_pascal) || (prefix == "set" && cont_snake_or_camel)))) {
                        constexpr bool prefer_upper = is_upper(prefix.data[0]);
                        return normalize_bare_accessor_name<T, prefer_upper>();
                    }
                    else {
                        return t.name;
                    }
                }
                else {
                    return t.name;
                }
            }

            template <typename T>
            constexpr auto get_display_name(const T t) noexcept
            {
                if constexpr (trait::is_property_v<T>) {
                    if constexpr (util::get<attr::property>(t.attributes).friendly_name) {
                        return REFL_MAKE_CONST_STRING(*util::get<attr::property>(t.attributes).friendly_name);
                    }
                    else {
                        return detail::normalize_accessor_name(t);
                    }
                }
                else {
                    return t.name;
                }
            }

            template <template <typename, size_t> typename MemberDescriptor, typename T, size_t N>
            constexpr size_t get_member_index(MemberDescriptor<T, N>) noexcept
            {
                return N;
            }

            // Compilers only instantiate templates once per set of template parameters.
            // Since each lambda is it's distinct type, and since we end up filtering
            // by these predicates in several places in the codebase, it is better to have
            // these lamdas defined here, to increase the likelihood that a template
            // instantiation of `util::filter` can be reused.
            static constexpr auto is_readable_p = [](auto m) { return is_readable(m); };
            static constexpr auto is_writable_p = [](auto m) { return is_writable(m); };

            template <typename Member>
            static constexpr auto display_name_equals_p = [](auto m) {
                return get_display_name_const(m) == get_display_name_const(Member{});
            };

            template <typename WritableMember>
            static constexpr bool has_reader_search(WritableMember)
            {
#ifdef REFL_DISALLOW_SEARCH_FOR_RW
                static_assert(WritableMember::name.data[0] == 0,
                    "REFL_DISALLOW_SEARCH_FOR_RW is defined. Make sure your property getters and setter are defined one after the other!");
#endif
                using member_types = typename type_descriptor<typename WritableMember::declaring_type>::declared_member_types;
                // Fallback to a slow linear search.
                using property_types = typename trait::filter_t<trait::is_property, member_types>;
                constexpr auto readable_properties = filter(property_types{}, detail::is_readable_p);
                return contains(readable_properties, display_name_equals_p<WritableMember>);
            }

            template <typename ReadableMember>
            static constexpr bool has_writer_search(ReadableMember)
            {
#ifdef REFL_DISALLOW_SEARCH_FOR_RW
                static_assert(ReadableMember::name.data[0] == 0,
                    "REFL_DISALLOW_SEARCH_FOR_RW is defined. Make sure your property getters and setter are defined one after the other!");
#endif
                using member_types = typename type_descriptor<typename ReadableMember::declaring_type>::declared_member_types;
                // Fallback to a slow linear search.
                using property_types = typename trait::filter_t<trait::is_property, member_types>;
                constexpr auto writable_properties = filter(property_types{}, detail::is_writable_p);
                return contains(writable_properties, display_name_equals_p<ReadableMember>);
            }

            template <typename WritableMember>
            static constexpr auto get_reader_search(WritableMember)
            {
#ifdef REFL_DISALLOW_SEARCH_FOR_RW
                static_assert(WritableMember::name.data[0] == 0,
                    "REFL_DISALLOW_SEARCH_FOR_RW is defined. Make sure your property getters and setter are defined one after the other!");
#endif
                using member_types = typename type_descriptor<typename WritableMember::declaring_type>::declared_member_types;
                // Fallback to a slow linear search.
                using property_types = typename trait::filter_t<trait::is_property, member_types>;
                constexpr auto readable_properties = filter(property_types{}, detail::is_readable_p);
                return find_one(readable_properties, display_name_equals_p<WritableMember>);
            }

            template <typename ReadableMember>
            static constexpr auto get_writer_search(ReadableMember)
            {
#ifdef REFL_DISALLOW_SEARCH_FOR_RW
                static_assert(ReadableMember::name.data[0] == 0,
                    "REFL_DISALLOW_SEARCH_FOR_RW is defined. Make sure your property getters and setter are defined one after the other!");
#endif
                using member_types = typename type_descriptor<typename ReadableMember::declaring_type>::declared_member_types;
                // Fallback to a slow linear search.
                using property_types = typename trait::filter_t<trait::is_property, member_types>;
                constexpr auto writable_properties = filter(property_types{}, detail::is_writable_p);
                return find_one(writable_properties, display_name_equals_p<ReadableMember>);
            }
        } // namespace detail

        /**
         * Returns the display name of T.
         * Uses the friendly_name of the property attribute, or the normalized name if no friendly_name was provided.
         *
         * \code{.cpp}
         * struct Foo {
         *   int get_foo() const;
         *   int GetFoo() const;
         *   int get_non_const() /missing const/;
         *   int get_custom() const;
         * };
         * REFL_AUTO(
         *   type(Foo),
         *   func(get_foo, property()),
         *   func(GetFoo, property()),
         *   func(get_non_const, property()),
         *   func(get_custom, property("value")),
         * )
         *
         * get_display_name(get_t<0, member_list<Foo>>{}) -> "foo"
         * get_display_name(get_t<1, member_list<Foo>>{}) -> "Foo"
         * get_display_name(get_t<2, member_list<Foo>>{}) -> "get_non_const"
         * get_display_name(get_t<3, member_list<Foo>>{}) -> "value"
         * \endcode
         */
        template <typename Descriptor>
        const char* get_display_name(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            static const std::string name(detail::get_display_name(d));
            return name.c_str();
        }

        /**
         * Returns the display name of T as a const_string<N>.
         * Uses the friendly_name of the property attribute, or the normalized name if no friendly_name was provided.
         * @see get_display_name
         */
        template <typename Descriptor>
        constexpr auto get_display_name_const(Descriptor d) noexcept
        {
            static_assert(trait::is_descriptor_v<Descriptor>);
            return detail::get_display_name(d);
        }

        /**
         * Checks if there exists a member that has the same display name as the one provied and is writable.
         * For getter methods with a property attribute, the return value will be true if there exists a
         * reflected setter method with a property with the same display name (property name normalization applies automatically).
         * For fields, returns true only if the field is writable.
         */
        template <typename ReadableMember>
        constexpr bool has_writer(ReadableMember member)
        {
            static_assert(is_writable(member) || is_property(member));
            if constexpr (is_writable(member)) {
                return true;
            }
            else {
                [[maybe_unused]] constexpr auto match = [](auto m) {
                    return is_property(m) && is_writable(m) && get_display_name_const(m) == get_display_name_const(ReadableMember{});
                };

                using member_types = typename type_descriptor<typename ReadableMember::declaring_type>::declared_member_types;
                constexpr auto member_index = detail::get_member_index(member);

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != 0) {
                    using likely_match = trait::get_t<member_index - 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return true;
                    }
                }

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != member_types::size - 1) {
                    using likely_match = trait::get_t<member_index + 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return true;
                    }
                    else {
                        return detail::has_writer_search(member);
                    }
                }
                else {
                    return detail::has_writer_search(member);
                }
            }
        }

        /**
         * Checks if there exists a member that has the same display name as the one provied and is readable.
         * For setter methods with a property attribute, the return value will be true if there exists a
         * reflected getter method with a property with the same display name (property name normalization applies automatically).
         * For fields, returns true.
         */
        template <typename WritableMember>
        constexpr bool has_reader(WritableMember member)
        {
            static_assert(is_readable(member) || is_property(member));
            if constexpr (is_readable(member)) {
                return true;
            }
            else {
                [[maybe_unused]] constexpr auto match = [](auto m) {
                    return is_property(m) && is_readable(m) && get_display_name_const(m) == get_display_name_const(WritableMember{});
                };

                using member_types = typename type_descriptor<typename WritableMember::declaring_type>::declared_member_types;
                constexpr auto member_index = detail::get_member_index(member);

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != member_types::size - 1) {
                    using likely_match = trait::get_t<member_index + 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return true;
                    }
                }

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != 0) {
                    using likely_match = trait::get_t<member_index - 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return true;
                    }
                    else {
                        return detail::has_reader_search(member);
                    }
                }
                else {
                    return detail::has_reader_search(member);
                }
            }
        }

        /**
         * Returns a member that has the same display name as the one provied and is writable.
         * For getter methods with a property attribute, the return value will the
         * reflected setter method with a property with the same display name (property name normalization applies automatically).
         * For fields, returns the same descriptor if writable.
         */
        template <typename ReadableMember>
        constexpr auto get_writer(ReadableMember member)
        {
            static_assert(is_writable(member) || is_property(member));
            if constexpr (is_writable(member)) {
                return member;
            }
            else if constexpr (has_writer(member)) {
                constexpr auto match = [](auto m) {
                    return is_property(m) && is_writable(m) && get_display_name_const(m) == get_display_name_const(ReadableMember{});
                };

                using member_types = typename type_descriptor<typename ReadableMember::declaring_type>::declared_member_types;
                constexpr auto member_index = detail::get_member_index(member);

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != 0) {
                    using likely_match = trait::get_t<member_index - 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return likely_match{};
                    }
                }

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != member_types::size - 1) {
                    using likely_match = trait::get_t<member_index + 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return likely_match{};
                    }
                    else {
                        return detail::get_writer_search(member);
                    }
                }
                else {
                    return detail::get_writer_search(member);
                }
            }
            else {
                static_assert(has_writer(member), "The property is not writable (could not find a setter method)!");
            }
        }

        /**
         * Returns a member that has the same display name as the one provied and is readable.
         * For setter methods with a property attribute, the return value will be a
         * reflected getter method with a property with the same display name (property name normalization applies automatically).
         * For fields, returns the same descriptor.
         */
        template <typename WritableMember>
        constexpr auto get_reader(WritableMember member)
        {
            static_assert(is_readable(member) || is_property(member));
            if constexpr (is_readable(member)) {
                return member;
            }
            else if constexpr (has_reader(member)) {
                constexpr auto match = [](auto m) {
                    return is_property(m) && is_readable(m) && get_display_name_const(m) == get_display_name_const(WritableMember{});
                };

                using member_types = typename type_descriptor<typename WritableMember::declaring_type>::declared_member_types;
                constexpr auto member_index = detail::get_member_index(member);

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != member_types::size - 1) {
                    using likely_match = trait::get_t<member_index + 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return likely_match{};
                    }
                }

                // Optimisation for the getter defined after setter pattern.
                if constexpr (member_index != 0) {
                    using likely_match = trait::get_t<member_index - 1, member_types>;
                    if constexpr (match(likely_match{})) {
                        return likely_match{};
                    }
                    else {
                        return detail::get_reader_search(member);
                    }
                }
                else {
                    return detail::get_reader_search(member);
                }
            }
            else {
                static_assert(has_reader(member), "The property is not readable (could not find a getter method)!");
            }
        }

    } // namespace descriptor

    using descriptor::member_list;
    using descriptor::declared_member_list;
    using descriptor::field_descriptor;
    using descriptor::function_descriptor;
    using descriptor::type_descriptor;

    /** Returns true if the type T is reflectable. */
    template <typename T>
    constexpr bool is_reflectable() noexcept
    {
        return trait::is_reflectable_v<T>;
    }

    /** Returns true if the non-qualified type T is reflectable.*/
    template <typename T>
    constexpr bool is_reflectable(const T&) noexcept
    {
        return trait::is_reflectable_v<T>;
    }

    /** Returns the type descriptor for the type T. */
    template<typename T>
    constexpr type_descriptor<T> reflect() noexcept
    {
        return {};
    }

    /** Returns the type descriptor for the non-qualified type T. */
    template<typename T>
    constexpr type_descriptor<T> reflect(const T&) noexcept
    {
        return {};
    }

#ifndef REFL_DETAIL_FORCE_EBO
#ifdef _MSC_VER
#define REFL_DETAIL_FORCE_EBO __declspec(empty_bases)
#else
#define REFL_DETAIL_FORCE_EBO
#endif
#endif

    /**
     * @brief Contains utilities that can have runtime-overhead (like proxy, debug, invoke)
     */
    namespace runtime
    {
        template <typename Derived, typename Target>
        struct REFL_DETAIL_FORCE_EBO proxy;

        namespace detail
        {
            template <typename T>
            struct get_member_info;

            template <typename T, size_t N>
            struct get_member_info<refl::function_descriptor<T, N>>
            {
                using type = refl::detail::member_info<T, N>;
            };

            template <typename T, size_t N>
            struct get_member_info<refl::field_descriptor<T, N>>
            {
                using type = refl::detail::member_info<T, N>;
            };

            template <typename T, typename U>
            constexpr T& static_ref_cast(U& value) noexcept
            {
                return static_cast<T&>(value);
            }

            template <typename T, typename U>
            constexpr const T& static_ref_cast(const U& value) noexcept
            {
                return static_cast<const T&>(value);
            }

            template <typename... Results>
            constexpr type_list<Results...> get_members_skip_shadowed(type_list<>, type_list<Results...>)
            {
                return {};
            }

            template <typename Member, typename... Members, typename... Results>
            constexpr auto get_members_skip_shadowed(type_list<Member, Members...>, type_list<Results...>)
            {
                if constexpr ((... || (Results::name == Member::name))) {
                    return get_members_skip_shadowed(type_list<Members...>{}, type_list<Results...>{});
                }
                else {
                    return get_members_skip_shadowed(type_list<Members...>{}, type_list<Results..., Member>{});
                }
            }

            template <typename T>
            using members_skip_shadowed = decltype(get_members_skip_shadowed(member_list<T>{}, type_list<>{}));

            /** Implements a proxy for a reflected function. */
            template <typename Derived, typename Func>
            struct REFL_DETAIL_FORCE_EBO function_proxy : public get_member_info<Func>::type::template remap<function_proxy<Derived, Func>>
            {
                function_proxy()
                {
                }

                template <typename Self, typename... Args>
                static constexpr decltype(auto) invoke_impl(Self&& self, Args&& ... args)
                {
                    return Derived::template invoke_impl<Func>(static_ref_cast<Derived>(self), std::forward<Args>(args)...);
                }
            };

            template <typename, typename>
            struct REFL_DETAIL_FORCE_EBO function_proxies;

            /** Implements a proxy for all reflected functions. */
            template <typename Derived, typename... Members>
            struct REFL_DETAIL_FORCE_EBO function_proxies<Derived, type_list<Members...>> : public function_proxy<Derived, Members>...
            {
            };

            /** Implements a proxy for a reflected field. */
            template <typename Derived, typename Field>
            struct REFL_DETAIL_FORCE_EBO field_proxy : public get_member_info<Field>::type::template remap<field_proxy<Derived, Field>>
            {
                field_proxy()
                {
                }

                template <typename Self, typename... Args>
                static constexpr decltype(auto) invoke_impl(Self&& self, Args&& ... args)
                {
                    return Derived::template invoke_impl<Field>(static_ref_cast<Derived>(self), std::forward<Args>(args)...);
                }
            };


            template <typename, typename>
            struct REFL_DETAIL_FORCE_EBO field_proxies;

            /** Implements a proxy for all reflected fields. */
            template <typename Derived, typename... Members>
            struct REFL_DETAIL_FORCE_EBO field_proxies<Derived, type_list<Members...>> : public field_proxy<Derived, Members>...
            {
            };

            template <typename T>
            using functions = trait::filter_t<trait::is_function, members_skip_shadowed<T>>;

            template <typename T>
            using fields = trait::filter_t<trait::is_field, members_skip_shadowed<T>>;

        } // namespace detail

        /**
         * @brief A proxy object that has a static interface identical to the reflected functions and fields of the target.
         *
         * A proxy object that has a static interface identical to the reflected functions and fields of the target.
         * Users should inherit from this class and specify an invoke_impl(Member member, Args&&... args) function.
         *
         * # Examples:
         * \code{.cpp}
         * template <typename T>
         * struct dummy_proxy : refl::runtime::proxy<dummy_proxy<T>, T> {
         *     template <typename Member, typename Self, typename... Args>
         *     static int invoke_impl(Self&& self, Args&&... args) {
         *          std::cout << get_debug_name(Member()) << " called with " << sizeof...(Args) << " parameters!\n";
         *          return 0;
         *     }
         * };
         * \endcode
         */
        template <typename Derived, typename Target>
        struct REFL_DETAIL_FORCE_EBO proxy
            : public detail::function_proxies<proxy<Derived, Target>, detail::functions<Target>>
            , public detail::field_proxies<proxy<Derived, Target>, detail::fields<Target>>
        {
            static_assert(
                sizeof(detail::function_proxies<proxy<Derived, Target>, detail::functions<Target>>) == 1 &&
                sizeof(detail::field_proxies<proxy<Derived, Target>, detail::fields<Target>>) == 1,
                "Multiple inheritance EBO did not kick in! "
                "You could try defining the REFL_DETAIL_FORCE_EBO macro appropriately to enable it on the required types. "
                "Default for MSC is `__declspec(empty_bases)`.");

            static_assert(
                trait::is_reflectable_v<Target>,
                "Target type must be reflectable!");

            typedef Target target_type;

            constexpr proxy() noexcept {}

        private:

            template <typename P, typename F>
            friend struct detail::function_proxy;

            template <typename P, typename F>
            friend struct detail::field_proxy;

            // Called by one of the function_proxy/field_proxy bases.
            template <typename Member, typename Self, typename... Args>
            static constexpr decltype(auto) invoke_impl(Self&& self, Args&& ... args)
            {
                return Derived::template invoke_impl<Member>(detail::static_ref_cast<Derived>(self), std::forward<Args>(args)...);
            }

        };

    } // namespace runtime

    namespace trait
    {
        template <typename>
        struct is_proxy;

        template <typename T>
        struct is_proxy
        {
        private:
            template <typename Derived, typename Target>
            static std::true_type test(runtime::proxy<Derived, Target>*);
            static std::false_type test(...);
        public:
            static constexpr bool value{ !std::is_reference_v<T> && decltype(test(std::declval<remove_qualifiers_t<T>*>()))::value };
        };

        template <typename T>
        [[maybe_unused]] static constexpr bool is_proxy_v{ is_proxy<T>::value };
    }

    namespace runtime
    {
        template <typename CharT, typename T>
        void debug(std::basic_ostream<CharT>& os, const T& value, bool compact = false);

        namespace detail
        {
            template <typename CharT, typename T, typename = decltype(std::declval<std::basic_ostream<CharT>&>() << std::declval<T>())>
            std::true_type is_ostream_printable_test(int);

            template <typename CharT, typename T>
            std::false_type is_ostream_printable_test(...);

            template <typename CharT, typename T>
            constexpr bool is_ostream_printable_v{ decltype(is_ostream_printable_test<CharT, T>(0))::value };

            namespace
            {
                [[maybe_unused]] int next_depth(int depth)
                {
                    return depth == -1 || depth > 8
                        ? -1
                        : depth + 1;
                }
            }

            template <typename CharT>
            void indent(std::basic_ostream<CharT>& os, int depth)
            {
                for (int i = 0; i < depth; i++) {
                    os << "    ";
                }
            }

            template <typename CharT, typename T>
            void debug_impl(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] int depth);

            template <typename CharT, typename T>
            void debug_detailed(std::basic_ostream<CharT>& os, const T& value, int depth)
            {
                using type_descriptor = type_descriptor<T>;
                bool compact = depth == -1;
                // print type with members enclosed in braces
                os << type_descriptor::name << " { ";
                if (!compact) os << '\n';

                constexpr auto readable_members = filter(type_descriptor::members, [](auto member) { return is_readable(member); });
                for_each(readable_members, [&](auto member, [[maybe_unused]] auto index) {
                    int new_depth = next_depth(depth);

                    indent(os, new_depth);
                    os << get_display_name(member) << " = ";

                    if constexpr (util::contains_instance<attr::debug>(member.attributes)) {
                        // use the debug attribute to print
                        auto debug_attr = util::get_instance<attr::debug>(member.attributes);
                        debug_attr.write(os, value);
                    }
                    else {
                        debug_impl(os, member(value), new_depth);
                    }

                    if (!compact || index + 1 != readable_members.size) {
                        os << ", ";
                    }
                    if (!compact) {
                        indent(os, depth);
                        os << '\n';
                    }
                });

                if (compact) os << ' ';
                indent(os, depth);
                os << '}';
            }

            template <typename CharT, typename T>
            void debug_reflectable(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] int depth)
            {
                using type_descriptor = type_descriptor<T>;
                if constexpr (trait::contains_instance_v<attr::debug, typename type_descriptor::attribute_types>) {
                    // use the debug attribute to print
                    auto debug_attr = util::get_instance<attr::debug>(type_descriptor::attributes);
                    debug_attr.write(os, value);
                }
                else if constexpr (detail::is_ostream_printable_v<CharT, T>) {
                    // type supports printing natively, just use that
                    os << value;
                }
                else {
                    debug_detailed(os, value, depth);
                }
            }

            template <typename CharT, typename T>
            void debug_container(std::basic_ostream<CharT>& os, const T& value, int depth)
            {
                bool compact = depth == -1;
                os << "[";

                auto end = value.end();
                for (auto it = value.begin(); it != end; ++it)
                {
                    if (!compact) os << '\n';
                    int new_depth = next_depth(depth);
                    indent(os, new_depth);

                    debug_impl(os, *it, new_depth);
                    if (std::next(it, 1) != end) {
                        os << ", ";
                    }
                    else if (!compact) {
                        os << '\n';
                    }
                }

                indent(os, depth);
                os << "]";
            }

            template <typename CharT, typename T>
            void debug_impl(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] int depth)
            {
                using no_pointer_t = std::remove_pointer_t<T>;

                if constexpr (std::is_same_v<bool, T>) {
                    os << (value ? "true" : "false");
                }
                else if constexpr (std::is_pointer_v<T> && !std::is_void_v<no_pointer_t> && trait::is_reflectable_v<no_pointer_t>) {
                    if (value == nullptr) {
                        os << "nullptr";
                    }
                    else {
                        os << '&';
                        debug_impl(os, *value, -1);
                    }
                }
                else if constexpr (trait::is_reflectable_v<T>) {
                    debug_reflectable(os, value, depth);
                }
                else if constexpr (detail::is_ostream_printable_v<CharT, T>) {
                    os << value;
                }
                else if constexpr (trait::is_container_v<T>) {
                    debug_container(os, value, depth);
                }
                else {
                    os << "(not printable)";
                }
            }
        }

        /**
         * Writes the debug representation of value to the given std::ostream.
         * Calls the function specified by the debug<F> attribute whenever possible,
         * before falling back to recursively interating the members and printing them.
         * Takes an optional arguments specifying whether to print a compact representation.
         * The compact representation contains no newlines.
         */
        template <typename CharT, typename T>
        void debug(std::basic_ostream<CharT>& os, const T& value, [[maybe_unused]] bool compact)
        {
            static_assert(trait::is_reflectable_v<T> || trait::is_container_v<T> || detail::is_ostream_printable_v<CharT, T>,
                "Type is not reflectable, not a container of reflectable types and does not support operator<<(std::ostream&, T)!");

            detail::debug_impl(os, value, compact ? -1 : 0);
        }

        /**
         * Writes the compact debug representation of the provided values to the given std::ostream.
         */
        template <typename CharT, typename... Ts>
        void debug_all(std::basic_ostream<CharT>& os, const Ts&... values)
        {
            refl::runtime::debug(os, std::forward_as_tuple(static_cast<const Ts&>(values)...), true);
        }

        /**
         * Writes the debug representation of the provided value to an std::string and returns it.
         * Takes an optional arguments specifying whether to print a compact representation.
         * The compact representation contains no newlines.
         */
        template <typename CharT = char, typename T>
        std::basic_string<CharT> debug_str(const T& value, bool compact = false)
        {
            std::basic_stringstream<CharT> ss;
            debug(ss, value, compact);
            return ss.str();
        }

        /**
         * Writes the compact debug representation of the provided values to an std::string and returns it.
         */
        template <typename CharT = char, typename... Ts>
        std::basic_string<CharT> debug_all_str(const Ts&... values)
        {
            return refl::runtime::debug_str(std::forward_as_tuple(static_cast<const Ts&>(values)...), true);
        }

        /**
         * Invokes the specified member with the provided arguments.
         * When used with a member that is a field, the function gets or sets the value of the field.
         * The list of members is initially filtered by the type of the arguments provided.
         * THe filtered list is then searched at runtime by member name for the specified member
         * and that member is then invoked by operator(). If no match is found,
         * an std::runtime_error is thrown.
         */
        template <typename U, typename T, typename... Args>
        U invoke(T&& target, const char* name, Args&&... args)
        {
            using type = std::remove_reference_t<T>;
            static_assert(refl::trait::is_reflectable_v<type>, "Unsupported type!");
            typedef type_descriptor<type> type_descriptor;

            std::conditional_t<std::is_void_v<U>, bool, std::optional<U>> result{};

            for_each(type_descriptor::members, [&](auto member) {
                using member_t = decltype(member);
                if (result) return;

                if constexpr (std::is_invocable_r_v<U, decltype(member), T, Args...>) {
                    if constexpr (trait::is_member_v<member_t>) {
                        if (std::strcmp(member.name.c_str(), name) == 0) {
                            if constexpr (std::is_void_v<U>) {
                                member(target, std::forward<Args>(args)...);
                                result = true;
                            }
                            else {
                                result.emplace(member(target, std::forward<Args>(args)...));
                            }
                        }
                    }
                }
            });

            if (!result) {
                throw std::runtime_error(std::string("The member ")
                    + type_descriptor::name.str() + "::" + name
                    + " is not compatible with the provided parameters or return type, is not reflected or does not exist!");
            }
            if constexpr (!std::is_void_v<U>) {
                return std::move(*result);
            }
        }

    } // namespace runtime

} // namespace refl

namespace refl::detail
{
    constexpr bool validate_attr_unique(type_list<>) noexcept
    {
        return true;
    }

    /** Statically asserts that all types in Ts... are unique. */
    template <typename T, typename... Ts>
    constexpr bool validate_attr_unique(type_list<T, Ts...>) noexcept
    {
        constexpr bool cond = (... && (!std::is_same_v<T, Ts> && validate_attr_unique(type_list<Ts>{})));
        static_assert(cond, "Some of the attributes provided have duplicate types!");
        return cond;
    }

    template <typename Req, typename Attr>
    constexpr bool validate_attr_usage() noexcept
    {
        return std::is_base_of_v<Req, Attr>;
    }

    /**
     * Statically asserts that all arguments inherit
     * from the appropriate bases to be used with Req.
     * Req must be one of the types defined in attr::usage.
     */
    template <typename Req, typename... Args>
    constexpr auto make_attributes(Args&&... args) noexcept
    {
        constexpr bool check_unique = validate_attr_unique(type_list<Args...>{});
        static_assert(check_unique, "Some of the supplied attributes cannot be used on this declaration!");

        constexpr bool check_usage = (... && validate_attr_usage<Req, trait::remove_qualifiers_t<Args>>());
        static_assert(check_usage, "Some of the supplied attributes cannot be used on this declaration!");

        return std::make_tuple(std::forward<Args>(args)...);
    }

    template <typename T, typename...>
    struct head
    {
        typedef T type;
    };

    /**
     * Accesses the first type T of Ts...
     * Used to allow for SFIANE to kick in in the implementation of REFL_FUNC.
     */
    template <typename T, typename... Ts>
    using head_t = typename head<T, Ts...>::type;

    template <typename T, typename U>
    struct transfer_const
    {
        using type = std::conditional_t<std::is_const_v<T>, std::add_const_t<U>, U>;
    };

    template <typename T, typename U>
    struct transfer_volatile
    {
        using type = std::conditional_t<std::is_volatile_v<T>, std::add_volatile_t<U>, U>;
    };

    template <typename T, typename U>
    struct transfer_cv : transfer_const<T, typename transfer_volatile<T, U>::type>
    {
    };

    template <typename T, typename U>
    struct transfer_lvalue_ref
    {
        using type = std::conditional_t<std::is_lvalue_reference_v<T>, std::add_lvalue_reference_t<U>, U>;
    };

    template <typename T, typename U>
    struct transfer_rvalue_ref
    {
        using type = std::conditional_t<std::is_rvalue_reference_v<T>, std::add_rvalue_reference_t<U>, U>;
    };

    template <typename T, typename U>
    struct transfer_ref : transfer_rvalue_ref<T, typename transfer_lvalue_ref<T, U>::type>
    {
    };

    template <typename T, typename U>
    struct transfer_cvref : transfer_ref<T, typename transfer_cv<std::remove_reference_t<T>, U>::type>
    {
    };

    template <typename T, typename U>
    constexpr auto forward_cast(std::remove_reference_t<T>& t) -> decltype(static_cast<typename transfer_cvref<T, U>::type&&>(t))
    {
        return static_cast<typename transfer_cvref<T, U>::type&&>(t);
    }

    template <typename T, typename U>
    constexpr auto forward_cast(std::remove_reference_t<T>&& t) -> decltype(static_cast<typename transfer_cvref<T, U>::type&&>(t))
    {
        static_assert(!std::is_lvalue_reference_v<T>, "template argument substituting T is an lvalue reference type");
        return static_cast<typename transfer_cvref<T, U>::type&&>(t);
    }

    template <typename T>
    constexpr auto get_type_name()
    {
        if constexpr (trait::is_reflectable_v<T>) {
            return type_descriptor<T>::name;
        }
        else {
            return make_const_string("(unknown)");
        }
    }

} // namespace refl::detail

/********************************/
/*  Metadata-generation macros  */
/********************************/

#define REFL_DETAIL_STR_IMPL(...) #__VA_ARGS__
/** Used to stringify input separated by commas (e.g. template specializations with multiple types). */
#define REFL_DETAIL_STR(...) REFL_DETAIL_STR_IMPL(__VA_ARGS__)
/** Used to group input containing commas (e.g. template specializations with multiple types). */
#define REFL_DETAIL_GROUP(...) __VA_ARGS__

/**
 * Expands to the appropriate attributes static member variable.
 * DeclType must be the name of one of the constraints defined in attr::usage.
 * __VA_ARGS__ is the list of attributes.
 */
#define REFL_DETAIL_ATTRIBUTES(DeclType, ...) \
        static constexpr auto attributes{ ::refl::detail::make_attributes<::refl::attr::usage:: DeclType>(__VA_ARGS__) }; \

/**
 * Expands to the body of a type_info__ specialization.
 */
#define REFL_DETAIL_TYPE_BODY(TypeName, ...) \
        typedef REFL_DETAIL_GROUP TypeName type; \
        REFL_DETAIL_ATTRIBUTES(type, __VA_ARGS__) \
        static constexpr auto name{ ::refl::util::make_const_string(REFL_DETAIL_STR(REFL_DETAIL_GROUP TypeName)) }; \
        static constexpr size_t member_index_offset = __COUNTER__ + 1; \
        template <size_t, typename = void> \
        struct member {};

/**
 * Creates reflection information for a specified type. Takes an optional attribute list.
 * This macro must only be expanded in the global namespace.
 *
 * # Examples:
 * ```
 * REFL_TYPE(Point)
 * ...
 * REFL_END
 * ```
 */
#define REFL_TYPE(TypeName, ...) \
    namespace refl_impl::metadata { template<> struct type_info__<TypeName> { \
        REFL_DETAIL_TYPE_BODY((TypeName), __VA_ARGS__)

/**
 * Creates reflection information for a specified type template. Takes an optional attribute list.
 * TemplateDeclaration must be a panenthesis-enclosed list declaring the template parameters. (e.g. (typename A, typename B)).
 * TypeName must be the fully-specialized type name and should also be enclosed in panenthesis. (e.g. (MyType<A, B>))
 * This macro must only be expanded in the global namespace.
 *
 * # Examples:
 * ```
 * REFL_TEMPLATE((typename T), (std::vector<T>))
 * ...
 * REFL_END
 * ```
 */
#define REFL_TEMPLATE(TemplateDeclaration, TypeName, ...) \
    namespace refl_impl::metadata { template <REFL_DETAIL_GROUP TemplateDeclaration> struct type_info__<REFL_DETAIL_GROUP TypeName> { \
        REFL_DETAIL_TYPE_BODY(TypeName, __VA_ARGS__)

/**
 * Terminated the declaration of reflection metadata for a particular type.
 *
 * # Examples:
 * ```
 * REFL_TYPE(Point)
 * ...
 * REFL_END
 */
#define REFL_END \
        static constexpr size_t member_count{ __COUNTER__ - member_index_offset }; \
    }; }

#define REFL_DETAIL_MEMBER_HEADER template<typename Unused__> struct member<__COUNTER__ - member_index_offset, Unused__>

#define REFL_DETAIL_MEMBER_COMMON(MemberType_, MemberName_, ...) \
        typedef ::refl::member::MemberType_ member_type; \
        static constexpr auto name{ ::refl::util::make_const_string(REFL_DETAIL_STR(MemberName_)) }; \
        REFL_DETAIL_ATTRIBUTES(MemberType_, __VA_ARGS__)

/** Creates the support infrastructure needed for the refl::runtime::proxy type. */
/*
    There can be a total of 12 differently qualified member functions with the same name.
    Providing remaps for non-const and const-only strikes a balance between compilation time and usability.
    And even though there are many other remap implementation possibilities (like virtual, field variants),
    adding them was considered to not be efficient from a compilation-time point of view.
*/
#define REFL_DETAIL_MEMBER_PROXY(MemberName_) \
        template <typename Proxy> struct remap { \
            template <typename... Args> decltype(auto) MemberName_(Args&&... args) { \
                return Proxy::invoke_impl(static_cast<Proxy&>(*this), ::std::forward<Args>(args)...); \
            } \
            template <typename... Args> decltype(auto) MemberName_(Args&&... args) const { \
                return Proxy::invoke_impl(static_cast<const Proxy&>(*this), ::std::forward<Args>(args)...); \
            } \
        }

/**
 * Creates reflection information for a public field. Takes an optional attribute list.
 */
#define REFL_FIELD(FieldName_, ...) \
    REFL_DETAIL_MEMBER_HEADER { \
        REFL_DETAIL_MEMBER_COMMON(field, FieldName_, __VA_ARGS__) \
    public: \
        typedef decltype(type::FieldName_) value_type; \
        static constexpr auto pointer{ &type::FieldName_ }; \
        REFL_DETAIL_MEMBER_PROXY(FieldName_); \
    };

/**
 * Creates reflection information for a public functions. Takes an optional attribute list.
 */
#define REFL_FUNC(FunctionName_, ...) \
    REFL_DETAIL_MEMBER_HEADER { \
        REFL_DETAIL_MEMBER_COMMON(function, FunctionName_, __VA_ARGS__) \
        public: \
        template<typename Self, typename... Args> static constexpr auto invoke(Self&& self, Args&&... args) -> decltype(::refl::detail::forward_cast<Self, type>(self).FunctionName_(::std::forward<Args>(args)...)) {\
            return ::refl::detail::forward_cast<Self, type>(self).FunctionName_(::std::forward<Args>(args)...); \
        } \
        template<typename... Args> static constexpr auto invoke(Args&&... args) -> decltype(::refl::detail::head_t<type, Args...>::FunctionName_(::std::declval<Args>()...)) { \
            return ::refl::detail::head_t<type, Args...>::FunctionName_(::std::forward<Args>(args)...); \
        } \
        template <typename Dummy = void> \
        static constexpr auto pointer() -> decltype(&::refl::detail::head_t<type, Dummy>::FunctionName_) { return &::refl::detail::head_t<type, Dummy>::FunctionName_; } \
        template <typename Pointer> \
        static constexpr auto resolve() -> ::std::decay_t<decltype(Pointer(&type::FunctionName_))> { return Pointer(&type::FunctionName_); } \
        REFL_DETAIL_MEMBER_PROXY(FunctionName_); \
    };

/********************************/
/*  Default Reflection Metadata */
/********************************/

#define REFL_DETAIL_PRIMITIVE(TypeName) \
    REFL_TYPE(TypeName) \
    REFL_END

    // Char types.
    REFL_DETAIL_PRIMITIVE(char)
    REFL_DETAIL_PRIMITIVE(wchar_t)
    REFL_DETAIL_PRIMITIVE(char16_t)
    REFL_DETAIL_PRIMITIVE(char32_t)
#ifdef __cpp_lib_char8_t
    REFL_DETAIL_PRIMITIVE(char8_t)
#endif

    // Integral types.
    REFL_DETAIL_PRIMITIVE(bool)
    REFL_DETAIL_PRIMITIVE(signed char)
    REFL_DETAIL_PRIMITIVE(unsigned char)
    REFL_DETAIL_PRIMITIVE(signed short)
    REFL_DETAIL_PRIMITIVE(unsigned short)
    REFL_DETAIL_PRIMITIVE(signed int)
    REFL_DETAIL_PRIMITIVE(unsigned int)
    REFL_DETAIL_PRIMITIVE(signed long)
    REFL_DETAIL_PRIMITIVE(unsigned long)
    REFL_DETAIL_PRIMITIVE(signed long long)
    REFL_DETAIL_PRIMITIVE(unsigned long long)

    // Floating point types.
    REFL_DETAIL_PRIMITIVE(float)
    REFL_DETAIL_PRIMITIVE(double)
    REFL_DETAIL_PRIMITIVE(long double)

    // Other types.
    REFL_DETAIL_PRIMITIVE(decltype(nullptr))

    // Void type.
    REFL_TYPE(void)
    REFL_END

#undef REFL_DETAIL_PRIMITIVE

#define REFL_DETAIL_POINTER(Ptr) \
        template<typename T> \
        struct type_info__<T Ptr> { \
            typedef T Ptr type; \
            template <size_t N> \
            struct member {}; \
            static constexpr auto name{ ::refl::detail::get_type_name<T>() + ::refl::util::make_const_string(#Ptr) }; \
            static constexpr ::std::tuple<> attributes{ }; \
            static constexpr size_t member_count{ 0 }; \
        }

    namespace refl_impl
    {
        namespace metadata
        {
            REFL_DETAIL_POINTER(*);
            REFL_DETAIL_POINTER(&);
            REFL_DETAIL_POINTER(&&);
        }
    }

#undef REFL_DETAIL_POINTER

namespace refl::detail
{
    template <typename CharT>
    std::basic_string<CharT> convert(const std::string& str)
    {
        return std::basic_string<CharT>(str.begin(), str.end());
    }

#ifdef __cpp_lib_string_view
    struct write_basic_string_view
    {
        template <typename CharT, typename Traits>
        void operator()(std::basic_ostream<CharT>& os, std::basic_string_view<CharT, Traits> str) const
        {
            // some vers of clang dont have std::quoted(string_view) overload
            if (!str.back()) { // no copy needed when null-terminated
                os << std::quoted(str.data());
            }
            else {
                os << std::quoted(std::basic_string<CharT, Traits>(str.begin(), str.end()));
            }
        }
    };
#endif

    struct write_basic_string
    {
        template <typename CharT, typename Traits, typename Allocator>
        void operator()(std::basic_ostream<CharT>& os, const std::basic_string<CharT, Traits, Allocator>& str) const
        {
            os << std::quoted(str);
        }
    };

    struct write_exception
    {
        template <typename CharT>
        void operator()(std::basic_ostream<CharT>& os, const std::exception& e) const
        {
            os << convert<CharT>("Exception");
    #ifdef REFL_RTTI_ENABLED
            os << convert<CharT>(" (") << convert<CharT>(typeid(e).name()) << convert<CharT>(")");
    #endif
            os << convert<CharT>(": `") << e.what() << convert<CharT>("`");
        }
    };

    struct write_tuple
    {
        template <typename CharT, typename Tuple, size_t... Idx>
        void write(std::basic_ostream<CharT>& os, Tuple&& t, std::index_sequence<Idx...>) const
        {
            os << CharT('(');
            for_each(type_list<std::integral_constant<size_t, Idx>...>{}, [&](auto idx_c) {
                using idx_t = decltype(idx_c);
                runtime::debug(os, std::get<idx_t::value>(t));
                if constexpr (sizeof...(Idx) - 1 != idx_t::value) {
                    os << convert<CharT>(", ");
                }
            });
            os << CharT(')');
        }

        template <typename CharT, typename... Ts>
        void operator()(std::basic_ostream<CharT>& os, const std::tuple<Ts...>& t) const
        {
            write(os, t, std::make_index_sequence<sizeof...(Ts)>{});
        }
    };

    struct write_pair
    {
        template <typename CharT, typename K, typename V>
        void operator()(std::basic_ostream<CharT>& os, const std::pair<K, V>& t) const
        {
            os << CharT('(');
            runtime::debug(os, t.first);
            os << convert<CharT>(", ");
            runtime::debug(os, t.second);
            os << CharT(')');
        }
    };

    struct write_unique_ptr
    {
        template <typename CharT, typename T, typename D>
        void operator()(std::basic_ostream<CharT>& os, const std::unique_ptr<T, D>& t) const
        {
            runtime::debug(os, t.get(), true);
        }
    };

    struct write_shared_ptr
    {
        template <typename CharT, typename T>
        void operator()(std::basic_ostream<CharT>& os, const std::shared_ptr<T>& t) const
        {
            runtime::debug(os, t.get(), true);
        }
    };

    struct write_weak_ptr
    {
        template <typename CharT, typename T>
        void operator()(std::basic_ostream<CharT>& os, const std::weak_ptr<T>& t) const
        {
            runtime::debug(os, t.lock().get(), true);
        }
    };

    struct write_complex
    {
        template <typename CharT, typename T>
        void operator()(std::basic_ostream<CharT>& os, const std::complex<T>& t) const
        {
            runtime::debug(os, t.real());
            os << CharT('+');
            runtime::debug(os, t.imag());
            os << CharT('i');
        }
    };
} // namespace refl::detail

// Custom reflection information for
// some common built-in types (std::basic_string, std::tuple, std::pair).

#ifndef REFL_NO_STD_SUPPORT

REFL_TYPE(std::exception, debug{ refl::detail::write_exception() })
    REFL_FUNC(what, property{ })
REFL_END

REFL_TEMPLATE(
    (typename Elem, typename Traits, typename Alloc),
    (std::basic_string<Elem, Traits, Alloc>),
    debug{ refl::detail::write_basic_string() })
    REFL_FUNC(size, property{ })
    REFL_FUNC(data, property{ })
REFL_END

#ifdef __cpp_lib_string_view

REFL_TEMPLATE(
    (typename Elem, typename Traits),
    (std::basic_string_view<Elem, Traits>),
    debug{ refl::detail::write_basic_string_view() })
    REFL_FUNC(size, property{ })
    REFL_FUNC(data, property{ })
REFL_END

#endif

REFL_TEMPLATE(
    (typename... Ts),
    (std::tuple<Ts...>),
    debug{ refl::detail::write_tuple() })
REFL_END

REFL_TEMPLATE(
    (typename T, typename D),
    (std::unique_ptr<T, D>),
    debug{ refl::detail::write_unique_ptr() })
REFL_END

REFL_TEMPLATE(
    (typename T),
    (std::shared_ptr<T>),
    debug{ refl::detail::write_shared_ptr() })
REFL_END

REFL_TEMPLATE(
    (typename K, typename V),
    (std::pair<K, V>),
    debug{ refl::detail::write_pair() })
REFL_END

#ifndef REFL_NO_STD_COMPLEX

REFL_TEMPLATE(
    (typename T),
    (std::complex<T>),
    debug{ refl::detail::write_complex() })
REFL_END

#endif // !REFL_NO_STD_COMPLEX

#endif // !REFL_NO_STD_SUPPORT

#ifndef REFL_NO_AUTO_MACRO

#define REFL_DETAIL_EXPAND(x) x
#define REFL_DETAIL_FOR_EACH_0(...)
#define REFL_DETAIL_FOR_EACH_1(what, x, ...) what(x)
#define REFL_DETAIL_FOR_EACH_2(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_1(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_3(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_2(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_4(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_3(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_5(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_4(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_6(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_5(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_7(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_6(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_8(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_7(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_9(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_8(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_10(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_9(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_11(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_10(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_12(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_11(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_13(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_12(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_14(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_13(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_15(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_14(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_16(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_15(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_17(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_16(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_18(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_17(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_19(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_18(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_20(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_19(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_21(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_20(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_22(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_21(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_23(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_22(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_24(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_23(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_25(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_24(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_26(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_25(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_27(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_26(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_28(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_27(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_29(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_28(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_30(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_29(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_31(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_30(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_32(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_31(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_33(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_32(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_34(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_33(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_35(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_34(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_36(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_35(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_37(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_36(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_38(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_37(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_39(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_38(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_40(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_39(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_41(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_40(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_42(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_41(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_43(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_42(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_44(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_43(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_45(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_44(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_46(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_45(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_47(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_46(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_48(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_47(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_49(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_48(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_50(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_49(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_51(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_50(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_52(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_51(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_53(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_52(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_54(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_53(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_55(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_54(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_56(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_55(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_57(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_56(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_58(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_57(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_59(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_58(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_60(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_59(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_61(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_60(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_62(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_61(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_63(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_62(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_64(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_63(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_65(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_64(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_66(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_65(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_67(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_66(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_68(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_67(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_69(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_68(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_70(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_69(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_71(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_70(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_72(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_71(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_73(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_72(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_74(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_73(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_75(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_74(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_76(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_75(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_77(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_76(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_78(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_77(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_79(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_78(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_80(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_79(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_81(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_80(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_82(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_81(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_83(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_82(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_84(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_83(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_85(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_84(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_86(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_85(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_87(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_86(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_88(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_87(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_89(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_88(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_90(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_89(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_91(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_90(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_92(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_91(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_93(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_92(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_94(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_93(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_95(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_94(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_96(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_95(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_97(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_96(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_98(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_97(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_99(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_98(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_100(what, x, ...) what(x) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_99(what, __VA_ARGS__))

#define REFL_DETAIL_FOR_EACH_NARG(...) REFL_DETAIL_FOR_EACH_NARG_(__VA_ARGS__, REFL_DETAIL_FOR_EACH_RSEQ_N())
#define REFL_DETAIL_FOR_EACH_NARG_(...) REFL_DETAIL_EXPAND(REFL_DETAIL_FOR_EACH_ARG_N(__VA_ARGS__))
#define REFL_DETAIL_FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93, _94, _95, _96, _97, _98, _99, _100, N, ...) N
#define REFL_DETAIL_FOR_EACH_RSEQ_N() 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define REFL_DETAIL_CONCATENATE(x, y) x##y
#define REFL_DETAIL_FOR_EACH_(N, what, ...) REFL_DETAIL_EXPAND(REFL_DETAIL_CONCATENATE(REFL_DETAIL_FOR_EACH_, N)(what, __VA_ARGS__))
#define REFL_DETAIL_FOR_EACH(what, ...) REFL_DETAIL_FOR_EACH_(REFL_DETAIL_FOR_EACH_NARG(__VA_ARGS__), what, __VA_ARGS__)

// Intellisense does not work nicely with passing variadic parameters (for the attributes)
// through all of the macro expansions and causes differently named member declarations to be
// used during code inspection.
#ifdef __INTELLISENSE__

#define REFL_DETAIL_EX_1_type(X, ...) REFL_TYPE(X)
#define REFL_DETAIL_EX_1_template(X, Y, ...) REFL_TEMPLATE(X, Y)
#define REFL_DETAIL_EX_1_field(X, ...) REFL_FIELD(X)
#define REFL_DETAIL_EX_1_func(X, ...) REFL_FUNC(X)

#else // !defined(__INTELLISENSE__)

#define REFL_DETAIL_EX_1_type(...) REFL_DETAIL_EX_EXPAND(REFL_DETAIL_EX_DEFER(REFL_TYPE)(__VA_ARGS__))
#define REFL_DETAIL_EX_1_template(...) REFL_DETAIL_EX_EXPAND(REFL_DETAIL_EX_DEFER(REFL_TEMPLATE)(__VA_ARGS__))
#define REFL_DETAIL_EX_1_field(...) REFL_DETAIL_EX_EXPAND(REFL_DETAIL_EX_DEFER(REFL_FIELD)(__VA_ARGS__))
#define REFL_DETAIL_EX_1_func(...) REFL_DETAIL_EX_EXPAND(REFL_DETAIL_EX_DEFER(REFL_FUNC)(__VA_ARGS__))

#endif // __INTELLISENSE__

#define REFL_DETAIL_EX_(Specifier, ...) REFL_DETAIL_EX_1_##Specifier __VA_ARGS__

#define REFL_DETAIL_EX_EMPTY()
#define REFL_DETAIL_EX_DEFER(Id) Id REFL_DETAIL_EX_EMPTY()
#define REFL_DETAIL_EX_EXPAND(...)  __VA_ARGS__

#define REFL_DETAIL_EX_END() REFL_END

#define REFL_AUTO(...) REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_, __VA_ARGS__) REFL_DETAIL_EX_EXPAND(REFL_DETAIL_EX_DEFER(REFL_DETAIL_EX_END)())

#endif // !defined(REFL_NO_AUTO_MACRO)

#endif // REFL_INCLUDE_HPP


// #include <port.hpp>
 // localinclude
// #include <port_traits.hpp>
#ifndef GNURADIO_NODE_PORT_TRAITS_HPP
#define GNURADIO_NODE_PORT_TRAITS_HPP

// #include "port.hpp"

// #include <refl.hpp>

// #include <utils.hpp>
 // localinclude

namespace fair::graph::traits::port {

template<typename T>
concept has_fixed_info_v = requires {
                                    typename T::value_type;
                                    { T::static_name() };
                                    { T::direction() } -> std::same_as<port_direction_t>;
                                    { T::type() } -> std::same_as<port_type_t>;
                                };

template<typename T>
using has_fixed_info = std::integral_constant<bool, has_fixed_info_v<T>>;

template<typename T>
struct has_fixed_info_or_is_typelist : std::false_type {};

template<typename T>
    requires has_fixed_info_v<T>
struct has_fixed_info_or_is_typelist<T> : std::true_type {};

template<typename T>
    requires(meta::is_typelist_v<T> and T::template all_of<has_fixed_info>)
struct has_fixed_info_or_is_typelist<T> : std::true_type {};

template<typename Port>
using type = typename Port::value_type;

template<typename Port>
using is_input = std::integral_constant<bool, Port::direction() == port_direction_t::INPUT>;

template<typename Port>
concept is_input_v = is_input<Port>::value;

template<typename Port>
using is_output = std::integral_constant<bool, Port::direction() == port_direction_t::OUTPUT>;

template<typename Port>
concept is_output_v = is_output<Port>::value;

template <typename Type>
concept is_port_v = is_output_v<Type> || is_input_v<Type>;

template<typename... Ports>
struct min_samples : std::integral_constant<std::size_t, std::max({ min_samples<Ports>::value... })> {};

template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection,
         std::size_t MIN_SAMPLES, std::size_t MAX_SAMPLES, gr::Buffer BufferType>
struct min_samples<fair::graph::port<T, PortName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES, BufferType>>
    : std::integral_constant<std::size_t, MIN_SAMPLES> {};

template<typename... Ports>
struct max_samples : std::integral_constant<std::size_t, std::min({ max_samples<Ports>::value... })> {};

template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection,
         std::size_t MIN_SAMPLES, std::size_t MAX_SAMPLES, gr::Buffer BufferType>
struct max_samples<fair::graph::port<T, PortName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES, BufferType>>
    : std::integral_constant<std::size_t, MAX_SAMPLES> {};

} // namespace port

#endif // include guard
 // localinclude
// #include <utils.hpp>
 // localinclude

// #include <vir/simd.h>


namespace fair::graph::traits::node {

namespace detail {
    template <typename FieldDescriptor>
    using member_type = typename FieldDescriptor::value_type;

    template <typename Type>
    using is_port = std::integral_constant<bool, port::is_port_v<Type>>;

    template <typename Port>
    constexpr bool is_port_descriptor_v = port::is_port_v<member_type<Port>>;

    template <typename Port>
    using is_port_descriptor = std::integral_constant<bool, is_port_descriptor_v<Port>>;

    template <typename PortDescriptor>
    using member_to_named_port = typename PortDescriptor::value_type::template with_name<fixed_string(refl::descriptor::get_name(PortDescriptor()).data)>;

    template<typename Node>
    struct member_ports_detector {
        static constexpr bool value = false;
    };

    template<class T, typename ValueType = std::remove_cvref_t<T>>
    concept Reflectable = refl::is_reflectable<ValueType>();

    template<Reflectable Node>
    struct member_ports_detector<Node> {
        using member_ports =
                    typename meta::to_typelist<refl::descriptor::member_list<Node>>
                        ::template filter<is_port_descriptor>
                        ::template transform<member_to_named_port>;

        static constexpr bool value = member_ports::size != 0;
    };

    template<typename Node>
    using port_name = typename Node::static_name();

    template<typename RequestedType>
    struct member_descriptor_has_type {
        template <typename Descriptor>
        using matches = std::is_same<RequestedType, member_to_named_port<Descriptor>>;
    };



} // namespace detail

template<typename...>
struct fixed_node_ports_data_helper;

// This specialization defines node attributes when the node is created
// with two type lists - one list for input and one for output ports
template<typename Node, meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template all_of<port::has_fixed_info> &&OutputPorts::template all_of<port::has_fixed_info>
struct fixed_node_ports_data_helper<Node, InputPorts, OutputPorts> {
    using member_ports_detector = std::false_type;

    // using member_ports_detector = detail::member_ports_detector<Node>;

    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port::type>;
    using output_port_types = typename output_ports ::template transform<port::type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

// This specialization defines node attributes when the node is created
// with a list of ports as template arguments
template<typename Node, port::has_fixed_info_v... Ports>
struct fixed_node_ports_data_helper<Node, Ports...> {
    using member_ports_detector = detail::member_ports_detector<Node>;

    using all_ports = std::remove_pointer_t<
        decltype([] {
            if constexpr (member_ports_detector::value) {
                return static_cast<typename member_ports_detector::member_ports*>(nullptr);
            } else {
                return static_cast<typename meta::concat<std::conditional_t<fair::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...>*>(nullptr);
            }
        }())>;

    using input_ports       = typename all_ports ::template filter<port::is_input>;
    using output_ports      = typename all_ports ::template filter<port::is_output>;

    using input_port_types  = typename input_ports ::template transform<port::type>;
    using output_port_types = typename output_ports ::template transform<port::type>;
};

// clang-format off
template<typename Node,
         typename Derived = typename Node::derived_t,
         typename ArgumentList = typename Node::node_template_parameters>
using fixed_node_ports_data =
    typename ArgumentList::template filter<port::has_fixed_info_or_is_typelist>
                         ::template prepend<Node>
                         ::template apply<fixed_node_ports_data_helper>;
// clang-format on

template<typename Node>
using all_ports = typename fixed_node_ports_data<Node>::all_ports;

template<typename Node>
using input_ports = typename fixed_node_ports_data<Node>::input_ports;

template<typename Node>
using output_ports = typename fixed_node_ports_data<Node>::output_ports;

template<typename Node>
using input_port_types = typename fixed_node_ports_data<Node>::input_port_types;

template<typename Node>
using output_port_types = typename fixed_node_ports_data<Node>::output_port_types;

template<typename Node>
using return_type = typename output_port_types<Node>::tuple_or_type;

template<typename Node>
using input_port_names = typename input_ports<Node>::template transform<detail::port_name>;

template<typename Node>
using output_port_names = typename output_ports<Node>::template transform<detail::port_name>;

template<typename Node>
constexpr bool node_defines_ports_as_member_variables = fixed_node_ports_data<Node>::member_ports_detector::value;

template<typename Node, typename PortType>
using get_port_member_descriptor =
    typename meta::to_typelist<refl::descriptor::member_list<Node>>
        ::template filter<detail::member_descriptor_has_type<PortType>::template matches>::template at<0>;

namespace detail {
template<std::size_t... Is>
auto
can_process_simd_invoke_test(auto &node, const auto &input, std::index_sequence<Is...>)
        -> decltype(node.process_one(std::get<Is>(input)...));
}

/* A node "can process simd" if its `process_one` function takes at least one argument and all
 * arguments can be simdized types of the actual port data types.
 *
 * The node can be a sink (no output ports).
 * The requirement of at least one function argument disallows sources.
 *
 * There is another (unnamed) concept for source nodes: Source nodes can implement
 * `process_one_simd(integral_constant)`, which returns SIMD object(s) of width N.
 */
template<typename Node>
concept can_process_simd
        =
#if DISABLE_SIMD
        false;
#else
        traits::node::input_port_types<Node>::size() > 0
       && requires(Node &node,
                   const meta::simdize<typename traits::node::input_port_types<Node>::template apply<std::tuple>>
                           &input_simds) {
              {
                  detail::can_process_simd_invoke_test(
                          node, input_simds, std::make_index_sequence<traits::node::input_ports<Node>::size()>())
              };
          };
#endif

} // namespace node

#endif // include guard
 // localinclude

#include <fmt/format.h>
// #include <refl.hpp>


namespace fair::graph {

using namespace fair::literals;

namespace stdx = vir::stdx;
using fair::meta::fixed_string;

template<typename F>
constexpr void
simd_epilogue(auto width, F &&fun) {
    static_assert(std::has_single_bit(+width));
    auto w2 = std::integral_constant<std::size_t, width / 2>{};
    if constexpr (w2 > 0) {
        fun(w2);
        simd_epilogue(w2, std::forward<F>(fun));
    }
}

template<std::ranges::contiguous_range... Ts, typename Flag = stdx::element_aligned_tag>
constexpr auto
simdize_tuple_load_and_apply(auto width, const std::tuple<Ts...> &rngs, auto offset, auto &&fun, Flag f = {}) {
    using Tup = meta::simdize<std::tuple<std::ranges::range_value_t<Ts>...>, width>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return fun(std::tuple_element_t<Is, Tup>(std::ranges::data(std::get<Is>(rngs)) + offset, f)...);
    }(std::make_index_sequence<sizeof...(Ts)>());
}

enum class work_return_t {
    ERROR = -100, /// error occurred in the work function
    INSUFFICIENT_OUTPUT_ITEMS =
        -3, /// work requires a larger output buffer to produce output
    INSUFFICIENT_INPUT_ITEMS =
        -2, /// work requires a larger input buffer to produce output
    DONE =
        -1, /// this block has completed its processing and the flowgraph should be done
    OK = 0, /// work call was successful and return values in i/o structs are valid
    CALLBACK_INITIATED =
        1, /// rather than blocking in the work function, the block will call back to the
           /// parent interface when it is ready to be called again
};

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    using requested_port_type = typename traits::node::input_ports<Self>::template at<Index>;
    if constexpr (traits::node::node_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::node::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    using requested_port_type = typename traits::node::output_ports<Self>::template at<Index>;
    if constexpr (traits::node::node_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::node::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::node::input_ports<Self>>();
    return input_port<Index, Self>(self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::node::output_ports<Self>>();
    return output_port<Index, Self>(self);
}

template<typename Self>
[[nodiscard]] constexpr auto
input_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(input_port<Idx>(self)...);
    }
    (std::make_index_sequence<traits::node::input_ports<Self>::size>());
}

template<typename Self>
[[nodiscard]] constexpr auto
output_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return std::tie(output_port<Idx>(self)...);
    }
    (std::make_index_sequence<traits::node::output_ports<Self>::size>());
}

/**
 * @brief The 'node<Derived>' is a base class for blocks that perform specific signal processing operations. It stores
 * references to its input and output 'ports' that can be zero, one, or many, depending on the use case.
 * As the base class for all user-defined nodes, it implements common convenience functions and a default public API
 * through the Curiously-Recurring-Template-Pattern (CRTP). For example:
 * @code
 * struct user_defined_block : node<user_defined_block> {
 *   IN<float> in;
 *   OUT<float> out;
 *   // implement one of the possible work or abstracted functions
 * };
 * ENABLE_REFLECTION(user_defined_block, in, out);
 * @endcode
 * The macro `ENABLE_REFLECTION` since it relies on a template specialisation needs to be declared on the global scope.
 *
 * As an alternative definition that does not require the 'ENABLE_REFLECTION' macro and that also supports arbitrary
 * types for input 'T' and for the return 'R':
 * @code
 * template<typename T, typename R>
 * struct user_defined_block : node<user_defined_block, IN<T, 0, N_MAX, "in">, OUT<R, 0, N_MAX, "out">> {
 *   // implement one of the possible work or abstracted functions
 * };
 * @endcode
 * This implementation provides efficient compile-time static polymorphism (i.e. access to the ports, settings, etc. does
 * not require virtual functions or inheritance, which can have performance penalties in high-performance computing contexts).
 * Note: The template parameter '<Derived>' can be dropped once C++23's 'deducing this' is widely supported by compilers.
 *
 * The 'node<Derived>' implementation provides simple defaults for users who want to focus on generic signal-processing
 * algorithms and don't need full flexibility (and complexity) of using the generic `work_return_t work() {...}`.
 * The following defaults are defined for one of the two 'user_defined_block' block definitions (WIP):
 * <ul>
 * <li> <b>case 1a</b> - non-decimating N-in->N-out mechanic and automatic handling of streaming tags and settings changes:
 * @code
 *  fg::IN<T> in;
 *  fg::OUT<R> out;
 *  T _factor = T{1.0};
 *
 *  [[nodiscard]] constexpr auto process_one(T a) const noexcept {
 *      return static_cast<R>(a * _factor);
 *  }
 * @endcode
 * The number, type, and ordering of input and arguments of `process_one(..)` are defined by the port definitions.
 * <li> <b>case 1b</b> - non-decimating N-in->N-out mechanic providing bulk access to the input/output data and automatic
 * handling of streaming tags and settings changes:
 * @code
 *  [[nodiscard]] constexpr auto process_bulk(std::span<const T> input, std::span<R> output) const noexcept {
 *      std::ranges::copy(input, output | std::views::transform([a = this->_factor](T x) { return static_cast<R>(x * a); }));
 *  }
 * @endcode
 * <li> <b>case 2a</b>: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N) (to-be-done)
 * <li> <b>case 2b</b>: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling (to-be-done)
 * <li> <b>case 3</b> -- generic `work()` providing full access/logic capable of handling any N-in->M-out tag-handling case:
 * @code
 * [[nodiscard]] constexpr work_return_t work() const noexcept {
 *     auto &out_port = output_port<"out">(this);
 *     auto &in_port = input_port<"in">(this);
 *
 *     auto &reader = in_port.reader();
 *     auto &writer = out_port.writer();
 *     const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
 *     const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
 *     if (n_readable == 0) {
 *         return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
 *     } else if (n_writable == 0) {
 *         return fair::graph::work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
 *     }
 *     const std::size_t n_to_publish = std::min(n_readable, n_writable); // N.B. here enforcing N_input == N_output
 *
 *     writer.publish([&reader, n_to_publish, this](std::span<T> output) {
 *         const auto input = reader.get(n_to_publish);
 *         for (; i < n_to_publish; i++) {
 *             output[i] = input[i] * value;
 *         }
 *     }, n_to_publish);
 *
 *     if (!reader.consume(n_to_publish)) {
 *         return fair::graph::work_return_t::ERROR;
 *     }
 *     return fair::graph::work_return_t::OK;
 * }
 * @endcode
 * <li> <b>case 4</b>:  Python -> map to cases 1-3 and/or dedicated callback (to-be-implemented)
 * <li> <b>special cases<b>: (to-be-implemented)
 *     * case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
 *     * case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)
 * <ul>
 * @tparam Derived the user-defined block CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * @tparam Arguments NTTP list containing the compile-time defined port instances, setting structs, or other constraints.
 */
template<typename Derived, typename... Arguments>
class node : protected std::tuple<Arguments...> {
public:
    using derived_t = Derived;
    using node_template_parameters = meta::typelist<Arguments...>;

private:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string _name{std::string(fair::meta::type_name<Derived>())};

    [[nodiscard]] constexpr auto &
    self() noexcept {
        return *static_cast<Derived *>(this);
    }

    [[nodiscard]] constexpr const auto &
    self() const noexcept {
        return *static_cast<const Derived *>(this);
    }

protected:
    constexpr bool
    enough_samples_for_output_ports(std::size_t n) {
        return std::apply([n](const auto &...port) noexcept {
                   return ((n >= port.min_buffer_size()) && ... && true);
               }, output_ports(&self()));
    }

    constexpr bool
    space_available_on_output_ports(std::size_t n) {
        return std::apply([n](const auto &...port) noexcept {
                   return ((n <= port.writer().available()) && ... && true);
               }, output_ports(&self()));
    }

public:
    [[nodiscard]] std::string_view
    name() const noexcept {
        return _name;
    }

    void
    set_name(std::string name) noexcept {
        _name = std::move(name);
    }

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    [[nodiscard]] constexpr auto static
    inputs_status(Self &self) noexcept {
        static_assert(traits::node::input_ports<Derived>::size > 0,
                      "A source node has no inputs, therefore no inputs status.");
        bool at_least_one_input_has_data = false;
        const auto availableForPort = [&at_least_one_input_has_data]<typename Port>(Port &port) noexcept {
            const std::size_t available = port.reader().available();
            if (available > 0_UZ) at_least_one_input_has_data = true;
            if (available < port.min_buffer_size()) {
                return 0_UZ;
            } else {
                return std::min(available, port.max_buffer_size());
            }
        };

        const std::size_t available_values_count
                = std::apply([&availableForPort](
                                     auto &...input_port) { return meta::safe_min(availableForPort(input_port)...); },
                             input_ports(&self));

        struct result {
            bool at_least_one_input_has_data;
           std::size_t available_values_count;
        };

        return result {
            .at_least_one_input_has_data = at_least_one_input_has_data,
            .available_values_count = available_values_count
        };
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    auto
    write_to_outputs(Self &self, std::size_t available_values_count, auto &writers_tuple) noexcept {
        if constexpr (traits::node::output_ports<Derived>::size > 0) {
            meta::tuple_for_each([available_values_count](auto &output_port, auto &writer) {
                                     output_port.writer().publish(writer.second, available_values_count);
                                 },
                                 output_ports(&self), writers_tuple);
        }
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    bool
    consume_readers(Self& self, std::size_t available_values_count) {
        bool success = true;
        if constexpr (traits::node::input_ports<Derived>::size > 0) {
            std::apply([available_values_count, &success] (auto&... input_port) {
                    ((success = success && input_port.reader().consume(available_values_count)), ...);
                }, input_ports(&self));
        }
        return success;
    }

    template <typename... Ts>
    constexpr auto
    invoke_process_one(Ts&&... inputs)
    {
        if constexpr (traits::node::output_ports<Derived>::size == 0) {
            self().process_one(std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
            return std::tuple{self().process_one(std::forward<Ts>(inputs)...)};
        } else {
            return self().process_one(std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto
    invoke_process_one_simd(auto width, Ts &&...input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::node::output_ports<Derived>::size == 0) {
                self().process_one_simd(width);
                return std::tuple{};
            } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
                return std::tuple{ self().process_one_simd(width) };
            } else {
                return self().process_one_simd(width);
            }
        } else {
            return invoke_process_one(std::forward<Ts>(input_simds)...);
        }
    }

    work_return_t
    work() noexcept {
        using input_types = traits::node::input_port_types<Derived>;
        using output_types = traits::node::output_port_types<Derived>;

        constexpr bool is_source_node = input_types::size == 0;
        constexpr bool is_sink_node = output_types::size == 0;

        std::size_t samples_to_process = 0;
        if constexpr (is_source_node) {
            if constexpr (requires(const Derived &d) {
                              { available_samples(d) } -> std::same_as<std::size_t>;
                          }) {
                // the (source) node wants to determine the number of samples to process
                samples_to_process = available_samples(self());
                if (not enough_samples_for_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_INPUT_ITEMS;
                }
                if (not space_available_on_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
                }
            } else if constexpr (is_sink_node) {
                // no input or output buffers, derive from internal "buffer sizes" (i.e. what the
                // buffer size would be if the node were not merged)
                constexpr std::size_t chunk_size = Derived::merged_work_chunk_size();
                static_assert(
                        chunk_size != std::dynamic_extent && chunk_size > 0,
                        "At least one internal port must define a maximum number of samples or the non-member/hidden "
                        "friend function `available_samples(const NodeType&)` must be defined.");
                samples_to_process = chunk_size;
            } else {
                // derive value from output buffer size
                samples_to_process = std::apply([&](const auto &...ports) {
                                         return std::min({ ports.writer().available()..., ports.max_buffer_size()... });
                                     }, output_ports(&self()));
                if (not enough_samples_for_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
                }
                // space_available_on_output_ports is true by construction of samples_to_process
            }
        } else {
            // Capturing structured bindings does not work in Clang...
            const auto [at_least_one_input_has_data, available_values_count] = self().inputs_status(self());
            if (available_values_count == 0) {
                return at_least_one_input_has_data ? work_return_t::INSUFFICIENT_INPUT_ITEMS : work_return_t::DONE;
            }
            samples_to_process = available_values_count;
            if (not enough_samples_for_output_ports(samples_to_process)) {
                return work_return_t::INSUFFICIENT_INPUT_ITEMS;
            }
            if (not space_available_on_output_ports(samples_to_process)) {
                return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
            }
        }

        const auto input_spans = meta::tuple_transform([samples_to_process](auto &input_port) noexcept {
            return input_port.reader().get(samples_to_process);
        }, input_ports(&self()));

        const auto writers_tuple = meta::tuple_transform([samples_to_process](auto &output_port) noexcept {
            return output_port.writer().get(samples_to_process);
        }, output_ports(&self()));

        // TODO: check here whether a process_one(...) or a bulk access process has been defined, cases:
        // case 1a: N-in->N-out -> process_one(...) -> auto-handling of streaming tags
        // case 1b: N-in->N-out -> process_bulk(<ins...>, <outs...>) -> auto-handling of streaming tags
        // case 2a: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
        // case 2b: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling
        // case 3:  N-in->M-out -> work() N,M arbitrary -> used need to handle the full logic (e.g. PLL algo)
        // case 4:  Python -> map to cases 1-3 and/or dedicated callback
        // special cases:
        // case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
        // case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)

        if constexpr (requires { &Derived::process_bulk;  }) {
            const work_return_t ret = std::apply([this](auto... args) { return static_cast<Derived *>(this)->process_bulk(args...); },
                                        std::tuple_cat(input_spans, meta::tuple_transform([](const auto &span) { return span.first; }, writers_tuple)));

            write_to_outputs(self(), samples_to_process, writers_tuple);
            const bool success = consume_readers(self(), samples_to_process);
            return success ? ret : work_return_t::ERROR;
        }

        using input_simd_types = meta::simdize<typename input_types::template apply<std::tuple>>;
        using output_simd_types = meta::simdize<typename output_types::template apply<std::tuple>>;

        std::integral_constant<std::size_t, (meta::simdize_size_v<input_simd_types> == 0
                                                 ? std::size_t(stdx::simd_abi::max_fixed_size<double>)
                                                 : std::min(std::size_t(stdx::simd_abi::max_fixed_size<double>),
                                                            meta::simdize_size_v<input_simd_types> * 4))> width{};

        if constexpr ((is_sink_node or meta::simdize_size_v<output_simd_types> != 0)
                      and ((is_source_node and requires(Derived &d) {
                               { d.process_one_simd(width) };
                           }) or (meta::simdize_size_v<input_simd_types> != 0 and traits::node::can_process_simd<Derived>))) {
            // SIMD loop
            std::size_t i = 0;
            for (; i + width <= samples_to_process; i += width) {
                const auto &results = simdize_tuple_load_and_apply(width, input_spans, i, [&](const auto &...input_simds) {
                    return invoke_process_one_simd(width, input_simds...);
                });
                meta::tuple_for_each(
                        [i](auto &writer, const auto &result) {
                            result.copy_to(writer.first /*data*/.data() + i, stdx::element_aligned);
                        },
                        writers_tuple, results);
            }
            simd_epilogue(width, [&](auto w) {
                if (i + w <= samples_to_process) {
                    const auto results = simdize_tuple_load_and_apply(w, input_spans, i, [&](auto &&...input_simds) {
                        return invoke_process_one_simd(w, input_simds...);
                    });
                    meta::tuple_for_each(
                            [i](auto &writer, auto &result) {
                                result.copy_to(writer.first /*data*/.data() + i, stdx::element_aligned);
                            },
                            writers_tuple, results);
                    i += w;
                }
            });
        } else {
            // Non-SIMD loop
            for (std::size_t i = 0; i < samples_to_process; ++i) {
                const auto results = std::apply([this, i](auto &...inputs) { return invoke_process_one(inputs[i]...); },
                                                input_spans);
                meta::tuple_for_each([i](auto &writer, auto &result) { writer.first /*data*/[i] = std::move(result); },
                                     writers_tuple, results);
            }
        }

        write_to_outputs(self(), samples_to_process, writers_tuple);

        const bool success = consume_readers(self(), samples_to_process);

#ifdef _DEBUG
        if (!success) {
            fmt::print("Node {} failed to consume {} values from inputs\n", self().name(), samples_to_process);
        }
#endif

        return success ? work_return_t::OK : work_return_t::ERROR;
    } // end: work_return_t work() noexcept { ..}
};

template<typename Node>
concept source_node = requires(Node &node, typename traits::node::input_port_types<Node>::tuple_type const &inputs) {
                          {
                              [](Node &n, auto &inputs) {
                                  constexpr std::size_t port_count = traits::node::input_port_types<Node>::size;
                                  if constexpr (port_count > 0) {
                                      return []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>)->decltype(n_inside.process_one(std::get<Is>(tup)...)) { return {}; }
                                      (n, inputs, std::make_index_sequence<port_count>());
                                  } else {
                                      return n.process_one();
                                  }
                              }(node, inputs)
                              } -> std::same_as<typename traits::node::return_type<Node>>;
                      };

template<typename Node>
concept sink_node = requires(Node &node, typename traits::node::input_port_types<Node>::tuple_type const &inputs) {
                        {
                            [](Node &n, auto &inputs) {
                                constexpr std::size_t port_count = traits::node::output_port_types<Node>::size;
                                []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>) {
                                    if constexpr (port_count > 0) {
                                        auto a [[maybe_unused]] = n_inside.process_one(std::get<Is>(tup)...);
                                    } else {
                                        n_inside.process_one(std::get<Is>(tup)...);
                                    }
                                }
                                (n, inputs, std::make_index_sequence<traits::node::input_port_types<Node>::size>());
                            }(node, inputs)
                        };
                    };

template<source_node Left, sink_node Right, std::size_t OutId, std::size_t InId>
class merged_node : public node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                                meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>> {
private:
    // copy-paste from above, keep in sync
    using base = node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                      meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>>;

    Left  left;
    Right right;

    // merged_work_chunk_size, that's what friends are for
    friend base;
    template<source_node, sink_node, std::size_t, std::size_t>
    friend class merged_node;

    // returns the minimum of all internal max_samples port template parameters
    static constexpr std::size_t
    merged_work_chunk_size() noexcept {
        constexpr std::size_t left_size = []() {
            if constexpr (requires {
                              { Left::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                          }) {
                return Left::merged_work_chunk_size();
            } else {
                return std::dynamic_extent;
            }
        }();
        constexpr std::size_t right_size = []() {
            if constexpr (requires {
                              { Right::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                          }) {
                return Right::merged_work_chunk_size();
            } else {
                return std::dynamic_extent;
            }
        }();
        return std::min({ traits::node::input_ports<Right>::template apply<traits::port::max_samples>::value,
                          traits::node::output_ports<Left>::template apply<traits::port::max_samples>::value, left_size,
                          right_size });
    }

    template<std::size_t I>
    constexpr auto
    apply_left(auto &&input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return left.process_one(std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...);
        }
                (std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset = traits::node::input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::node::input_port_types<Left>::size + sizeof...(Is);
            static_assert(
                    second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))...,
                                     std::forward<decltype(tmp)>(tmp), std::get<second_offset + Js>(input_tuple)...);
        }
                (std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename traits::node::input_port_types<base>;
    using output_port_types = typename traits::node::output_port_types<base>;
    using return_type       = typename traits::node::return_type<base>;

    constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    // if the left node (source) implements available_samples (a customization point), then pass the call through
    friend constexpr std::size_t
    available_samples(const merged_node &self) noexcept
        requires requires(const Left &l) {
            { available_samples(l) } -> std::same_as<std::size_t>;
        }
    {
        return available_samples(self.left);
    }

    template<meta::any_simd... Ts>
        requires traits::node::can_process_simd<Left> && traits::node::can_process_simd<Right>
    constexpr meta::simdize<return_type, meta::simdize_size_v<std::tuple<Ts...>>>
    process_one(const Ts &...inputs) {
        static_assert(traits::node::output_port_types<Left>::size == 1,
                      "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(
                std::tie(inputs...), apply_left<traits::node::input_port_types<Left>::size()>(std::tie(inputs...)));
    }

    constexpr auto
    process_one_simd(auto N)
        requires traits::node::can_process_simd<Right>
    {
        if constexpr (requires(Left &l) {
                          { l.process_one_simd(N) };
                      }) {
            return right.process_one(left.process_one_simd(N));
        } else {
            using LeftResult = typename traits::node::return_type<Left>;
            using V = meta::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0; i < V::size(); ++i) {
                tmp[i] = left.process_one();
            }
            return right.process_one(V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking
    // can_process_simd.
        requires(input_port_types::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr return_type
    process_one(Ts &&...inputs) {
        // if (sizeof...(Ts) == 0) we could call `return process_one_simd(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::node::output_port_types<Left>::size == 1) {
            // only the result from the right node needs to be returned
            return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                 apply_left<traits::node::input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::node::input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (traits::node::output_port_types<Left>::size == 2 && traits::node::output_port_types<Right>::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (traits::node::output_port_types<Left>::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (traits::node::output_port_types<Right>::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out));
                }
                        (std::make_index_sequence<OutId>(),
                         std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>,
                                                                                    std::index_sequence<Js...>,
                                                                                    std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))...,
                                           std::move(std::get<OutId + 1 + Js>(left_out))...,
                                           std::move(std::get<Ks>(right_out)...));
                }
                        (std::make_index_sequence<OutId>(),
                         std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>(),
                         std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    } // end:: process_one

    work_return_t
    work() noexcept {
        return base::work();
    }
};

/**
 * This methods can merge simple blocks that are defined via a single `auto process_one(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-ID API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
 * auto merged = merge_by_index<0, 0>(scale<int, -1>(), merge_by_index<0, 0>(scale<int, 2>(), adder<int>()));
 *
 * // execute graph
 * std::array<int, 4> a = { 1, 2, 3, 4 };
 * std::array<int, 4> b = { 10, 10, 10, 10 };
 *
 * int                r = 0;
 * for (std::size_t i = 0; i < 4; ++i) {
 *     r += merged.process_one(a[i], b[i]);
 * }
 * @endcode
 */
template<std::size_t OutId, std::size_t InId, source_node A, sink_node B>
constexpr auto
merge_by_index(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
        if constexpr (!std::is_same_v<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>) {
            fair::meta::print_types<fair::meta::message_type<"OUTPUT_PORTS_ARE:">, typename traits::node::output_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, OutId>,
                    typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,

                    fair::meta::message_type<"INPUT_PORTS_ARE:">, typename traits::node::input_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, InId>,
                    typename traits::node::input_port_types<std::remove_cvref_t<A>>::template at<InId>>{};
        }
        return { std::forward<A>(a), std::forward<B>(b) };
}

/**
 * This methods can merge simple blocks that are defined via a single `auto process_one(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-name API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> output
 * auto merged = merge<"scaled", "addend1">(scale<int, 2>(), adder<int>());
 *
 * // execute graph
 * std::array<int, 4> a = { 1, 2, 3, 4 };
 * std::array<int, 4> b = { 10, 10, 10, 10 };
 *
 * int                r = 0;
 * for (std::size_t i = 0; i < 4; ++i) {
 *     r += merged.process_one(a[i], b[i]);
 * }
 * @endcode
 */
template<fixed_string OutName, fixed_string InName, source_node A, sink_node B>
constexpr auto
merge(A &&a, B &&b) {
        constexpr std::size_t OutId = meta::indexForName<OutName, typename traits::node::output_ports<A>>();
        constexpr std::size_t InId  = meta::indexForName<InName, typename traits::node::input_ports<B>>();
        static_assert(OutId != -1);
        static_assert(InId != -1);
        static_assert(std::same_as<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>,
                      "Port types do not match");
        return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}

#if !DISABLE_SIMD
namespace test
{
struct copy : public node<copy, IN<float, 0, -1_UZ, "in">, OUT<float, 0, -1_UZ, "out">> {
    public:
        template<meta::t_or_simd<float> V>
        constexpr V
        process_one(const V &a) const noexcept {
            return a;
        }
};

static_assert(traits::node::input_port_types<copy>::size() == 1);
static_assert(std::same_as<traits::node::return_type<copy>, float>);
static_assert(traits::node::can_process_simd<copy>);
static_assert(traits::node::can_process_simd<decltype(merge_by_index<0, 0>(copy(), copy()))>);
}
#endif


/**
 *
 *                                |￣￣￣￣￣￣￣￣￣￣￣￣￣￣|
 *                                |     !!!Horray!!!      |
 *                                |      you made it!     |
 *                                |                       |
 *                                |        Warning!       |
 *                                |  Beneath are dragons! |
 *                                |＿＿＿＿＿＿＿＿＿＿＿＿＿＿|
 *                                    (\__/)  ||
 *                                    (•ㅅ•)  ||
 *                                    /  　  づ
 * ******************************************************************************************************************
 *
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠊⠶⣂⣠⢠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡑⡑⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣼⡾⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠂⠑⡳⠀⣀⠠⠤⠠⡀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣷⣤⡀⡄⠚⢸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠁⠁⠀⠀⠀⠙⡕⠀⠀⠀⠀⢻⣿⣿⣿⣿⢿⢧⡭⣶⣿⣿⣿⣷⣶⣶⣤⡀⣀⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠪⠤⡠⣙⣿⡿⡫⠉⣿⣭⡞⣿⣿⣿⣿⣿⣿⣿⣷⠂⠨⣌⠚⡴⠂⠤⣀⢀⡂⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠁⠀⠀⠈⢩⣿⢿⣿⣿⡋⣿⣿⣧⣀⠛⠛⠟⢿⢹⣿⣷⣶⣶⣤⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣦⣾⠿⣷⣯⣿⣿⣿⢤⣿⣿⣿⣶⣶⣤⣭⣒⣕⡁⠛⠛⠂⠝⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣞⣿⣾⣿⣿⠿⠿⠿⣅⡐⣶⣿⣿⣿⣿⣿⣿⣿⡏⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⣀⠅⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣷⣿⣶⣿⣿⣿⣿⣿⡿⣿⣿⣷⢿⣿⡄⠀⠛⠂⠀⠀⠀⠀⠀⠀⠀⠀⢀⢤⠀⠀⠀⢀⠠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠜⢠⢁⡁⠒⡈⠐⠒⠒⠛⠊
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣕⢂⣬⣶⠒⠠⠀⣿⣿⠉⣿⡇⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣷⠙⠛⡄⠀⠀⠀⠀⠀⠄⣀⢤⣴⣶⣶⡶⣿⣾⣿⣶⣶⣿⣶⣶⡂⡔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⣸⢃⣼⠫⢓⠬⠤⠄⠶⠒⠀
 *    ⠀⠀⠀⠀⠀⠀⠴⠂⣼⣿⡄⣿⠁⣴⣷⠾⠁⠉⠛⠏⣿⠛⠿⣷⡍⣿⣿⢿⣿⣿⣿⣈⠛⡄⠀⠀⠀⠀⠀⣀⡼⣶⣿⣷⣿⣿⣯⣿⢷⣟⣿⡻⣿⣯⣿⣿⣿⣿⣾⣖⠒⠤⠀⠀⠀⠀⠀⠀⣤⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⠀⡀⣿⠁⠁⣒⢒⡠⡀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⠛⣀⡤⢱⠀⠀⠀⠉⠃⠀⠈⣳⣿⡿⣿⠿⣿⠟⠀⠀⣀⣤⠀⠀⠁⣬⣿⣿⣿⣿⣿⣟⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⡿⠮⡂⠀⣀⠀⢀⣼⠏⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠢⠂⣞⡿⠰⠠⢀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⡿⠛⠛⠛⠃⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⡫⡱⠀⣾⠁⣀⠀⢴⣾⣿⣿⣷⣿⣿⣿⣿⣿⠟⠛⠉⠉⠁⢀⣉⡙⠛⣿⣿⣿⣿⣿⣯⣿⣿⣶⣧⣶⣟⣁⣾⠁⠀⠀⠀⠀⠀⣰⠃⣶⣷⣶⣶⣶⣶⣈⡄⢀⠒⣤⣶⣿⠁⡉⢒⠁⠑⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⠋⠙⣿⣿⣶⣄⠀⠀⠀⠀⠀⣴⣿⣿⢿⣿⣿⣿⠟⠀⢀⣾⣁⠀⠄⣴⣿⣿⣷⣿⣿⣿⣿⠟⠁⠀⠀⠀⢀⠀⣿⠉⢿⣿⢙⣐⡈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣭⢩⠂⠀⠀⠀⣀⠀⡠⣵⣯⣷⣗⣿⣿⣽⣽⣿⣯⣿⣿⠯⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠃⣒⢿⡏⠀⠀⠈⠛⣽⡽⠿⣿⣿⣿⣿⣿⣿⣿⣯⣿⣿⣤⣷⠿⣯⠟⢠⣢⣿⢿⣿⣿⣿⣽⣿⠟⠀⠀⠀⠀⠀⠀⠋⣦⣍⣻⣮⣿⣿⠋⠀⠀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣶⣥⣬⣿⣿⣿⣿⣿⠿⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣏⣶⣤⣿⢿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⢺⠋⠋⠉⠛⣿⣿⣯⢿⢿⣿⣯⣿⡿⢻⣿⣿⣽⢷⢿⣿⠿⣯⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣍⠙⣿⣿⣿⣿⣿⣿⢿⣿⣽⣿⣿⢿⣿⣿⣿⣷⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣆⡀⠁⠙⣿⣿⣧⡀⠛⣿⣿⣿⢿⣿⣿⣿⣿⣿⡟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣄⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣌⠝⠋⠀⠀⠉⠉⠉⠛⠁⠀⠀⠙⢿⣟⣟⣿⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢿⣿⣿⣿⣿⣿⣿⠿⣟⣿⣭⠤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⣤⡘⣿⣯⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⣤⠴⣒⣀⣄⠀⠀⣿⡿⣿⡻⣿⣧⣿⣿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣶⣥⣈⢿⣦⣿⣿⠃⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣟⠀⠛⣿⣦⣸⣿⠁⢀⣤⣿⣿⢿⣿⢿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢧⣁⣉⣭⣿⣿⠿⠛⠛⠛⠛⣹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⡻⣿⣷⣶⣿⣿⣿⣾⣿⣿⣽⡿⠟⠋⠁⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡈⢋⣴⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⢶⢷⣿⡿⠛⠛⠛⢿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠈⠊⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢄⠀⠀⠀⠀⠀⠀⠉⠛⡿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
 *
 * The following macros are helpers to wrap around the existing refl-cpp macros: https://github.com/veselink1/refl-cpp
 * The are basically needed to do a struct member-field introspections, to support
 *   a) compile-time serialiser generation between std::map<std::string, pmt::pmtv> <-> user-defined settings structs
 *   b) allow for block ports being defined a member fields rather than as NTTPs of the node<T, ...> template

 * Their use is limited to the namespace scope where the block is defined (i.e. not across .so boundaries) and will be
 * supplanted once the compile-time reflection language feature is merged with the C++ standard, e.g.
 * Matúš Chochlík, Axel Naumann, David Sankel: Static reflection, P0194R3, ISO/IEC JTC1 SC22 WG21
 *    https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0194r3.html
 *
 *  These macros need to be defined in a global scope due to relying on template specialisation that cannot be done in
 *  any other namespace than the one they were declared -- for illustration see, for example:
 *  https://github.com/veselink1/refl-cpp/issues/59
 *  https://compiler-explorer.com/z/MG7fxzK4j
 *
 *  For practical purposes, the macro can be defined either at the end of the struct declaring namespace or the specific
 *  namespace exited/re-enteres such as
 *  @code
 *  namespace private::library {
 *     struct my_struct {
 *         int field_a;
 *         std::string field_b;
 *     };
 *  }
 *  ENABLE_REFLECTION(private::library:my_struct, field_a, field_b)
 *  namespace private::library {
 *   // ...
 *  @endcode
 *
 *  And please, if you want to accelerator the compile-time reflection process, please give your support and shout-out
 *  to the above authors, and contact your C++ STD Committee representative that this feature should not be delayed.
 */


/**
 * This macro can be used for simple non-templated structs and classes, e.g.
 * @code
 * struct my_struct {
 *     int field_a;
 *      std::string field_b;
 * };
 * ENABLE_REFLECTION(private::library:my_struct, field_a, field_b)
 */
#define ENABLE_REFLECTION(TypeName, ...) \
    REFL_TYPE(TypeName __VA_OPT__(, )) \
    REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_1_field __VA_OPT__(, ) __VA_ARGS__) \
    REFL_END

/**
 * This macro can be used for arbitrary templated structs and classes, that depend
 * on mixed typename and NTTP parameters
 * @code
 * template<typename T, std::size_t size>
 * struct custom_struct {
 *     T field_a;
 *     T field_b;
 *
 *     [[nodiscard]] constexpr std::size_t size() const noexcept { return size; }
 * };
 * ENABLE_REFLECTION_FOR_TEMPLATE_FULL(typename T, std::size_t size), (custom_struct<T, size>), field_a, field_a);
 */
#define ENABLE_REFLECTION_FOR_TEMPLATE_FULL(TemplateDef, TypeName, ...) \
    REFL_TEMPLATE(TemplateDef, TypeName __VA_OPT__(, )) \
    REFL_DETAIL_FOR_EACH(REFL_DETAIL_EX_1_field __VA_OPT__(, ) __VA_ARGS__) \
    REFL_END

/**
 * This macro can be used for simple templated structs and classes, that depend
 * only on pure typename-template lists
 * @code
 * template<typename T>
 * struct my_templated_struct {
 *     T field_a;
 *     T field_b;
 * };
 * ENABLE_REFLECTION_FOR_TEMPLATE(my_templated_struct, field_a, field_b);
 */
#define ENABLE_REFLECTION_FOR_TEMPLATE(Type, ...) \
    ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename ...Ts), (Type<Ts...>), __VA_ARGS__)

} // namespace fair::graph

#endif // include guard


#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <ranges>
#include <tuple>
#include <variant>

#if !__has_include(<source_location>)
#define HAVE_SOURCE_LOCATION 0
#else

#include <source_location>

#if defined __cpp_lib_source_location && __cpp_lib_source_location >= 201907L
#define HAVE_SOURCE_LOCATION 1
#else
#define HAVE_SOURCE_LOCATION 0
#endif
#endif

namespace fair::graph {

using namespace fair::literals;

/**
 *  Runtime capable wrapper to be used within a block. It's primary purpose is to allow the runtime
 *  initialisation/connections between blocks that are not in the same compilation unit.
 *  Ownership is defined by if the strongly-typed port P is either passed
 *  a) as an lvalue (i.e. P& -> keep reference), or
 *  b) as an rvalue (P&& -> being moved into dyn_port).
 *
 *  N.B. the intended use is within the node/block interface where there is -- once initialised --
 *  always a strong-reference between the strongly-typed port and it's dyn_port wrapper. I.e no ports
 *  are added or removed after the initialisation and the port life-time is coupled to that of it's
 *  parent block/node.
 */
class dynamic_port {
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual supported_type
        pmt_type() const noexcept
                = 0;

        [[nodiscard]] virtual port_type_t
        type() const noexcept
                = 0;

        [[nodiscard]] virtual port_direction_t
        direction() const noexcept
                = 0;

        [[nodiscard]] virtual std::string_view
        name() const noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        resize_buffer(std::size_t min_size) noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        disconnect() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        connect(dynamic_port &dst_port) = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool
        update_reader_internal(void *buffer_other) noexcept
                = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType &> _value;

        [[nodiscard]] void *
        writer_handler_internal() noexcept {
            return _value.writer_handler_internal();
        };

        [[nodiscard]] bool
        update_reader_internal(void *buffer_other) noexcept override {
            if constexpr (T::IS_INPUT) {
                return _value.update_reader_internal(buffer_other);
            } else {
                assert(!"This works only on input ports");
                return false;
            }
        }

    public:
        wrapper()                = delete;

        wrapper(const wrapper &) = delete;

        auto &
        operator=(const wrapper &)
                = delete;

        auto &
        operator=(wrapper &&)
                = delete;

        explicit constexpr wrapper(T &arg) noexcept : _value{ arg } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg) } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        ~wrapper() override = default;

        [[nodiscard]] constexpr supported_type
        pmt_type() const noexcept override {
            return _value.pmt_type();
        }

        [[nodiscard]] constexpr port_type_t
        type() const noexcept override {
            return _value.type();
        }

        [[nodiscard]] constexpr port_direction_t
        direction() const noexcept override {
            return _value.direction();
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept override {
            return _value.name();
        }

        [[nodiscard]] connection_result_t
        resize_buffer(std::size_t min_size) noexcept override {
            return _value.resize_buffer(min_size);
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept override {
            return _value.disconnect();
        }

        [[nodiscard]] connection_result_t
        connect(dynamic_port &dst_port) override {
            if constexpr (T::IS_OUTPUT) {
                auto src_buffer = _value.writer_handler_internal();
                return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS
                                                                   : connection_result_t::FAILED;
            } else {
                assert(!"This works only on input ports");
                return connection_result_t::FAILED;
            }
        }
    };

    bool
    update_reader_internal(void *buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type         = void; // a sterile port

    constexpr dynamic_port() = delete;

    template<Port T>
    constexpr dynamic_port(const T &arg) = delete;

    template<Port T>
    explicit constexpr dynamic_port(T &arg) noexcept : _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<Port T>
    explicit constexpr dynamic_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type
    pmt_type() const noexcept {
        return _accessor->pmt_type();
    }

    [[nodiscard]] port_type_t
    type() const noexcept {
        return _accessor->type();
    }

    [[nodiscard]] port_direction_t
    direction() const noexcept {
        return _accessor->direction();
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _accessor->name();
    }

    [[nodiscard]] connection_result_t
    resize_buffer(std::size_t min_size) {
        if (direction() == port_direction_t::OUTPUT) {
            return _accessor->resize_buffer(min_size);
        }
        return connection_result_t::FAILED;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        return _accessor->disconnect();
    }

    [[nodiscard]] connection_result_t
    connect(dynamic_port &dst_port) {
        return _accessor->connect(dst_port);
    }
};

static_assert(Port<dynamic_port>);

#define ENABLE_PYTHON_INTEGRATION
#ifdef ENABLE_PYTHON_INTEGRATION

// TODO: Not yet implemented
class dynamic_node {
private:
    // TODO: replace the following with array<2, vector<dynamic_port>>
    using dynamic_ports = std::vector<dynamic_port>;
    dynamic_ports                                         _dynamic_input_ports;
    dynamic_ports                                         _dynamic_output_ports;

    std::function<void(dynamic_ports &, dynamic_ports &)> _process;

public:
    void
    work() {
        _process(_dynamic_input_ports, _dynamic_output_ports);
    }

    template<typename T>
    void
    add_port(T &&port) {
        switch (port.direction()) {
        case port_direction_t::INPUT:
            if (auto portID = port_index<port_direction_t::INPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined input port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_input_ports.emplace_back(std::forward<T>(port));
            break;

        case port_direction_t::OUTPUT:
            if (auto portID = port_index<port_direction_t::OUTPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined output port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_output_ports.emplace_back(std::forward<T>(port));
            break;

        default: assert(false && "cannot add port with ANY designation");
        }
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::size_t index) {
        return index < _dynamic_input_ports.size() ? std::optional{ &_dynamic_input_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_input_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_input_ports.cbegin(), _dynamic_input_ports.cend(), portNameMatches);
        return it != _dynamic_input_ports.cend() ? std::optional{ std::distance(_dynamic_input_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::string_view name) {
        if (const auto index = dynamic_input_port_index(name); index.has_value()) {
            return &_dynamic_input_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::size_t index) {
        return index < _dynamic_output_ports.size() ? std::optional{ &_dynamic_output_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_output_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_output_ports.cbegin(), _dynamic_output_ports.cend(), portNameMatches);
        return it != _dynamic_output_ports.cend() ? std::optional{ std::distance(_dynamic_output_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::string_view name) {
        if (const auto index = dynamic_output_port_index(name); index.has_value()) {
            return &_dynamic_output_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_input_ports() const noexcept {
        return _dynamic_input_ports;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_output_ports() const noexcept {
        return _dynamic_output_ports;
    }
};

#endif


class graph {
private:
    class node_model {
    public:
        virtual ~node_model() = default;

        virtual std::string_view
        name() const
                = 0;

        virtual work_return_t
        work() = 0;

        virtual void *
        raw()
                = 0;
    };

    template<typename T>
    class node_wrapper final : public node_model {
    private:
        static_assert(std::is_same_v<T, std::remove_reference_t<T>>);
        T _node;

    public:
        node_wrapper(const node_wrapper &other) = delete;

        node_wrapper &
        operator=(const node_wrapper &other)
                = delete;

        node_wrapper(node_wrapper &&other) : _node(std::exchange(other._node, nullptr)) {}

        node_wrapper &
        operator=(node_wrapper &&other) {
            auto tmp = std::move(other);
            std::swap(_node, tmp._node);
            return *this;
        }

        ~node_wrapper() override = default;

        node_wrapper() {}

        template<typename Arg>
            requires (!std::is_same_v<std::remove_cvref_t<Arg>, T>)
        node_wrapper(Arg&& arg) : _node(std::forward<Arg>(arg)) {}

        template<typename ...Args>
            requires (sizeof...(Args) > 1)
        node_wrapper(Args&&... args) : _node{std::forward<Args>(args)...} {}

        constexpr work_return_t
        work() override {
            return _node.work();
        }

        std::string_view
        name() const override {
            return _node.name();
        }

        void *
        raw() override {
            return std::addressof(_node);
        }
    };

    class edge {
    public:
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        node_model* _src_node;
        node_model* _dst_node;
        std::size_t                 _src_port_index;
        std::size_t                 _dst_port_index;
        int32_t                     _weight;
        std::string                 _name; // custom edge name
        bool                        _connected;

    public:
        edge()             = delete;

        edge(const edge &) = delete;

        edge &
        operator=(const edge &)
                = delete;

        edge(edge &&) noexcept = default;

        edge &
        operator=(edge &&) noexcept
                = default;

        edge(node_model* src_node, std::size_t src_port_index, node_model* dst_node, std::size_t dst_port_index, int32_t weight, std::string_view name)
            : _src_node(src_node)
            , _dst_node(dst_node)
            , _src_port_index(src_port_index)
            , _dst_port_index(dst_port_index)
            , _weight(weight)
            , _name(name) {
        }

        [[nodiscard]] constexpr int32_t
        weight() const noexcept {
            return _weight;
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept {
            return _name;
        }

        [[nodiscard]] constexpr bool
        connected() const noexcept {
            return _connected;
        }

        [[nodiscard]] connection_result_t
        connect() noexcept {
            return connection_result_t::FAILED;
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept { /* return _dst_node->port<INPUT>(_dst_port_index).value()->disconnect(); */
            return connection_result_t::FAILED;
        }
    };

    std::vector<edge>                        _edges;
    std::vector<std::unique_ptr<node_model>> _nodes;

    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source, typename SourcePort, typename Destination, typename DestinationPort>
    [[nodiscard]] connection_result_t
    connect_impl(Source &src_node_raw, SourcePort& source_port, Destination &dst_node_raw, DestinationPort& destination_port,
            int32_t weight = 0, std::string_view name = "unnamed edge") {
        static_assert(
                std::is_same_v<typename SourcePort::value_type, typename DestinationPort::value_type>,
                "The source port type needs to match the sink port type");

        if (!std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) {
            return registered_node->raw() == std::addressof(src_node_raw);
        })
            || !std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) {
            return registered_node->raw() == std::addressof(dst_node_raw);
        })) {
            throw std::runtime_error(fmt::format("Can not connect nodes that are not registered first:\n {}:{} -> {}:{}\n", src_node_raw.name(), src_port_index, dst_node_raw.name(), dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == connection_result_t::SUCCESS) {
            auto find_wrapper = [this] (auto* node) {
                auto it = std::find_if(_nodes.begin(), _nodes.end(), [node] (auto& wrapper) {
                        return wrapper->raw() == node;
                    });
                if (it == _nodes.end()) {
                    throw fmt::format("This node {} does not belong to this graph\n", node->name());
                }
                return it->get();
            };
            auto* src_node = find_wrapper(&src_node_raw);
            auto* dst_node = find_wrapper(&dst_node_raw);
            _edges.emplace_back(src_node, src_port_index, dst_node, src_port_index, weight, name);
        }

        return result;
    }

    std::vector<std::function<connection_result_t()>> _connection_definitions;

    // Just a dummy class that stores the graph and the source node and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template <typename Source, typename Port, std::size_t src_port_index = 1_UZ>
    struct source_connector {
        graph& self;
        Source& source;
        Port& port;

        source_connector(graph& _self, Source& _source, Port& _port) : self(_self), source(_source), port(_port) {}

    private:
        template <typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto to(Destination& destination, DestinationPort& destination_port) {
            // Not overly efficient as the node doesn't know the graph it belongs to,
            // but this is not a frequent operation and the check is important.
            auto is_node_known = [this] (const auto& query_node) {
                return std::any_of(self._nodes.cbegin(), self._nodes.cend(), [&query_node] (const auto& known_node) {
                    return known_node->raw() == std::addressof(query_node);
                        });

            };
            if (!is_node_known(source) || !is_node_known(destination)) {
                throw fmt::format("Source {} and/or destination {} do not belong to this graph\n", source.name(), destination.name());
            }
            self._connection_definitions.push_back([self = &self, source = &source, source_port = &port, destination = &destination, destination_port = &destination_port] () {
                return self->connect_impl<src_port_index, dst_port_index>(*source, *source_port, *destination, *destination_port);
            });
            return connection_result_t::SUCCESS;
        }

    public:
        template <typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto to(Destination& destination, DestinationPort Destination::* member_ptr) {
            return to<Destination, DestinationPort, dst_port_index>(destination, std::invoke(member_ptr, destination));
        }

        template <std::size_t dst_port_index, typename Destination>
        [[nodiscard]] constexpr auto to(Destination& destination) {
            auto &destination_port = input_port<dst_port_index>(&destination);
            return to<Destination, std::remove_cvref_t<decltype(destination_port)>, dst_port_index>(destination, destination_port);
        }

        template <fixed_string dst_port_name, typename Destination>
        [[nodiscard]] constexpr auto to(Destination& destination) {
            using destination_input_ports = typename traits::node::input_ports<Destination>;
            constexpr std::size_t dst_port_index = meta::indexForName<dst_port_name, destination_input_ports>();
            if constexpr (dst_port_index == meta::invalid_index) {
                meta::print_types<
                    meta::message_type<"There is no input port with the specified name in this destination node">,
                    Destination,
                    meta::message_type<dst_port_name>,
                    meta::message_type<"These are the known names:">,
                    traits::node::input_port_names<Destination>,
                    meta::message_type<"Full ports info:">,
                    destination_input_ports
                        > port_not_found_error{};
            }
            return to<dst_port_index, Destination>(destination);
        }

        source_connector(const source_connector&) = delete;
        source_connector(source_connector&&) = delete;
        source_connector& operator=(const source_connector&) = delete;
        source_connector& operator=(source_connector&&) = delete;
    };

    struct init_proof {
        init_proof(bool _success) : success(_success) {}
        bool success = true;

        operator bool() const { return success; }
    };

    template<std::size_t src_port_index, typename Source>
    friend
    auto connect(Source& source);

    template<fixed_string src_port_name, typename Source>
    friend
    auto connect(Source& source);

    template<typename Source, typename Port>
    friend
    auto connect(Source& source, Port Source::* member_ptr);

public:
    auto
    edges_count() const {
        return _edges.size();
    }

    template<typename Node, typename... Args>
    auto&
    make_node(Args&&... args) {
        static_assert(std::is_same_v<Node, std::remove_reference_t<Node>>);
        auto& new_node_ref = _nodes.emplace_back(std::make_unique<node_wrapper<Node>>(std::forward<Args>(args)...));
        return *static_cast<Node*>(new_node_ref->raw());
    }

    template<std::size_t src_port_index, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        auto &port = output_port<src_port_index>(&source);
        return graph::source_connector<Source, std::remove_cvref_t<decltype(port)>, src_port_index>(*this, source, port);
    }

    template<fixed_string src_port_name, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        using source_output_ports = typename traits::node::output_ports<Source>;
        constexpr std::size_t src_port_index = meta::indexForName<src_port_name, source_output_ports>();
        if constexpr (src_port_index == meta::invalid_index) {
            meta::print_types<
                meta::message_type<"There is no output port with the specified name in this source node">,
                Source,
                meta::message_type<src_port_name>,
                meta::message_type<"These are the known names:">,
                traits::node::output_port_names<Source>,
                meta::message_type<"Full ports info:">,
                source_output_ports
                    > port_not_found_error{};
        }
        return connect<src_port_index, Source>(source);
    }

    template<typename Source, typename Port>
    [[nodiscard]] auto connect(Source& source, Port Source::* member_ptr) {
        return graph::source_connector<Source, Port>(*this, source, std::invoke(member_ptr, source));
    }

    init_proof init() {
        auto result = init_proof(
            std::all_of(_connection_definitions.begin(), _connection_definitions.end(), [] (auto& connection_definition) {
                return connection_definition() == connection_result_t::SUCCESS;
            }));
        _connection_definitions.clear();
        return result;
    }

    work_return_t
    work(init_proof& init) {
        if (!init) {
            return work_return_t::ERROR;
        }
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _nodes) {
                auto result = node->work();
                if (result == work_return_t::ERROR) {
                    return work_return_t::ERROR;
                } else if (result == work_return_t::INSUFFICIENT_INPUT_ITEMS) {
                    // nothing
                } else if (result == work_return_t::DONE) {
                    // nothing
                } else if (result == work_return_t::OK) {
                    something_happened = true;
                } else if (result == work_return_t::INSUFFICIENT_OUTPUT_ITEMS) {
                    something_happened = true;
                }
            }

            run = something_happened;
        }

        return work_return_t::DONE;
    }
};

// TODO: add nicer enum formatter
inline std::ostream &
operator<<(std::ostream &os, const connection_result_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_type_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_direction_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_domain_t &value) {
    return os << static_cast<int>(value);
}

#ifndef __EMSCRIPTEN__
auto
this_source_location(std::source_location l = std::source_location::current()) {
    return fmt::format("{}:{},{}", l.file_name(), l.line(), l.column());
}
#else
auto
this_source_location() {
    return "not yet implemented";
}
#endif // __EMSCRIPTEN__

} // namespace fair::graph

#endif // include guard
