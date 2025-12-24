#ifndef GNURADIO_BLOCKINGSYNC_HPP
#define GNURADIO_BLOCKINGSYNC_HPP

#include <atomic>
#include <chrono>
#include <concepts>
#include <format>
#include <type_traits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

namespace gr {

namespace blocking_sync::detail {

template<typename T>
concept IsClockPortType = requires {
    requires std::same_as<typename std::remove_cvref_t<T>::value_type, std::uint8_t>;
    requires std::remove_cvref_t<T>::kIsInput;
};

// clang-format off
template<typename T>
concept HasClockPort = requires(T t) { { t.clk_in } -> IsClockPortType; } || requires(T t) { { t.clk } -> IsClockPortType; };

template<typename T>
concept HasSampleRate = requires(T t) { { t.sample_rate } -> std::convertible_to<float>; };

template<typename T>
concept HasChunkSize = requires(T t) { { t.chunk_size } -> std::convertible_to<gr::Size_t>; };

template<typename T>
concept HasUseInternalThread = requires(T t) { { t.use_internal_thread } -> std::convertible_to<bool>; };

template<typename T>
constexpr auto& getClockPort(T& block) {
    if constexpr (requires { { block.clk_in } -> IsClockPortType; }) {
        return block.clk_in;
    } else if constexpr (requires { { block.clk } -> IsClockPortType; }) {
        return block.clk;
    }
}

template<typename T>
constexpr const auto& getClockPort(const T& block) {
    if constexpr (requires { { block.clk_in } -> IsClockPortType; }) {
        return block.clk_in;
    } else if constexpr (requires { { block.clk } -> IsClockPortType; }) {
        return block.clk;
    }
}
// clang-format on

} // namespace blocking_sync::detail

/**
 * @brief CRTP mixin for wall-clock synchronised timing in source/generator blocks.
 *
 * Provides three operating modes:
 *
 * 1. **Clock-driven** (clock port connected):
 *    Synchronous operation driven by input samples. One output per input.
 *
 * 2. **Free-running with internal timer** (`use_internal_thread = true`, default):
 *    Timer thread wakes scheduler at chunk intervals for consistent output rate.
 *
 * 3. **Free-running, scheduler-driven** (`use_internal_thread = false`):
 *    No timer thread. Samples computed from elapsed time when scheduler calls processBulk.
 *    Suitable for HW drivers with external timing.
 *
 * Requirements for derived class:
 * - `sample_rate` member (float, required)
 * - `chunk_size` member (gr::Size_t, optional - defaults to sample_rate/10)
 * - `use_internal_thread` member (bool, optional - defaults to true)
 * - clock port named `clk_in` or `clk` of type PortIn<std::uint8_t[, Optional]> (optional)
 *
 * Usage:
 * @code
 * template<typename T>
 * struct MyGenerator : Block<MyGenerator<T>>, BlockingSync<MyGenerator<T>> {
 *     PortIn<std::uint8_t, Optional> clk_in;
 *     PortOut<T> out;
 *
 *     Annotated<float, "sample_rate"> sample_rate = 1000.f;
 *     Annotated<gr::Size_t, "chunk_size"> chunk_size = 100;       // optional
 *     Annotated<bool, "use_internal_thread"> use_internal_thread = true; // optional
 *
 *     void start() { this->blockingSyncStart(); }
 *     void stop()  { this->blockingSyncStop(); }
 *
 *     work::Status processBulk(InputSpanLike auto& clkIn, OutputSpanLike auto& out) {
 *         const auto nSamples = this->syncSamples(clkIn, out);
 *         if (nSamples == 0) {
 *             return work::Status::INSUFFICIENT_INPUT_ITEMS;
 *         }
 *         for (std::size_t i = 0; i < nSamples; ++i) {
 *             out[i] = generateSample();
 *         }
 *         std::ignore = clkIn.consume(nSamples);
 *         out.publish(nSamples);
 *         return work::Status::OK;
 *     }
 * };
 * @endcode
 *
 * For pure source blocks without clock input:
 * @code
 * work::Status processBulk(OutputSpanLike auto& out) {
 *     const auto nSamples = this->syncSamples(out);
 *     // ...
 * }
 * @endcode
 */
template<typename Derived, typename ClockSourceType = std::chrono::system_clock>
struct BlockingSync {
    using TimePoint = std::chrono::time_point<ClockSourceType>;

private:
    TimePoint         _blockingSync_startTime{};
    TimePoint         _blockingSync_lastUpdateTime{};
    std::atomic<bool> _blockingSync_timerRunning{false};
    std::atomic<bool> _blockingSync_timerDone{true};

    [[nodiscard]] constexpr Derived&       self() noexcept { return *static_cast<Derived*>(this); }
    [[nodiscard]] constexpr const Derived& self() const noexcept { return *static_cast<const Derived*>(this); }

    [[nodiscard]] float getSampleRate() const {
        static_assert(blocking_sync::detail::HasSampleRate<Derived>, "Derived class must have 'sample_rate' member");
        return static_cast<float>(self().sample_rate);
    }

    [[nodiscard]] gr::Size_t getChunkSize() const {
        if constexpr (blocking_sync::detail::HasChunkSize<Derived>) {
            return static_cast<gr::Size_t>(self().chunk_size);
        } else {
            return static_cast<gr::Size_t>(std::max(1.f, getSampleRate() / 10.f));
        }
    }

    [[nodiscard]] bool useInternalThread() const {
        if constexpr (blocking_sync::detail::HasUseInternalThread<Derived>) {
            return static_cast<bool>(self().use_internal_thread);
        }
        return true;
    }

    [[nodiscard]] bool isClockPortConnected() const {
        if constexpr (blocking_sync::detail::HasClockPort<Derived>) {
            return blocking_sync::detail::getClockPort(self()).isConnected();
        }
        return false;
    }

    [[nodiscard]] std::chrono::microseconds getChunkPeriod() const {
        const float chunkSize = static_cast<float>(getChunkSize());
        const auto  periodUs  = static_cast<long>(1e6f * chunkSize / getSampleRate());
        return std::chrono::microseconds(periodUs);
    }

    [[nodiscard]] std::size_t computeWallClockSamples(std::size_t maxSamples, bool dropIfBehind) {
        const TimePoint now            = ClockSourceType::now();
        const double    elapsedSeconds = std::chrono::duration<double>(now - _blockingSync_lastUpdateTime).count();
        const auto      sampleRate     = static_cast<double>(getSampleRate());
        const auto      samplesNeeded  = static_cast<std::size_t>(elapsedSeconds * sampleRate);
        const auto      chunkSize      = static_cast<std::size_t>(getChunkSize());

        const auto nSamples = std::min({samplesNeeded, chunkSize, maxSamples});

        if (dropIfBehind) {
            _blockingSync_lastUpdateTime = now;
        } else if (nSamples > 0) {
            const auto samplesTime = std::chrono::duration<double>(static_cast<double>(nSamples) / sampleRate);
            _blockingSync_lastUpdateTime += std::chrono::duration_cast<typename ClockSourceType::duration>(samplesTime);
        }

        return nSamples;
    }

public:
    /**
     * @brief Returns true if running in free-running (wall-clock driven) mode.
     */
    [[nodiscard]] bool isFreeRunning() const noexcept { return !isClockPortConnected(); }

    /**
     * @brief Returns true if using internal timer thread.
     */
    [[nodiscard]] bool isUsingInternalThread() const noexcept { return useInternalThread(); }

    [[nodiscard]] TimePoint blockingSyncStartTime() const noexcept { return _blockingSync_startTime; }
    [[nodiscard]] TimePoint blockingSyncLastUpdateTime() const noexcept { return _blockingSync_lastUpdateTime; }

    /**
     * @brief Compute number of samples to process for blocks with optional clock input.
     *
     * In clock-driven mode: returns min(input.size(), output.size()).
     * In free-running mode: returns samples based on elapsed wall-clock time.
     *
     * @param input clock input span
     * @param output output span
     * @param dropIfBehind if true, skip ahead when behind schedule (prevents catch-up)
     * @return number of samples to produce
     */
    template<ReaderSpanLike TInput, WriterSpanLike TOutput>
    [[nodiscard]] std::size_t syncSamples(TInput& input, TOutput& output, bool dropIfBehind = false) {
        if (!isFreeRunning()) {
            return std::min(input.size(), output.size());
        }
        return computeWallClockSamples(output.size(), dropIfBehind);
    }

    /**
     * @brief Compute number of samples to process for pure source blocks (no clock input).
     *
     * Always uses wall-clock timing.
     *
     * @param output output span
     * @param dropIfBehind if true, skip ahead when behind schedule (prevents catch-up)
     * @return number of samples to produce
     */
    template<WriterSpanLike TOutput>
    [[nodiscard]] std::size_t syncSamples(TOutput& output, bool dropIfBehind = false) {
        return computeWallClockSamples(output.size(), dropIfBehind);
    }

    /**
     * @brief Compute number of samples based on wall-clock timing (direct size overload).
     *
     * Useful for testing or when output span size is known separately.
     *
     * @param maxSamples maximum samples to produce
     * @param dropIfBehind if true, skip ahead when behind schedule (prevents catch-up)
     * @return number of samples to produce
     */
    [[nodiscard]] std::size_t syncSamples(std::size_t maxSamples, bool dropIfBehind = false) { return computeWallClockSamples(maxSamples, dropIfBehind); }

    /**
     * @brief Manually reset timing baseline to now.
     *
     * Useful when using external timing (use_internal_thread = false) after receiving HW data.
     */
    void blockingSyncResetTiming() { _blockingSync_lastUpdateTime = ClockSourceType::now(); }

    /**
     * @brief Start the timing system. Call from block's start() method.
     */
    void blockingSyncStart() {
        _blockingSync_startTime      = ClockSourceType::now();
        _blockingSync_lastUpdateTime = _blockingSync_startTime;

        if (isFreeRunning() && useInternalThread()) {
            startTimer();
        }
    }

    /**
     * @brief Stop the timing system. Call from block's stop() method.
     */
    void blockingSyncStop() { stopTimer(); }

private:
    void startTimer() {
        if (_blockingSync_timerRunning.exchange(true)) {
            return;
        }

        _blockingSync_timerDone = false;
        thread_pool::Manager::defaultIoPool()->execute([this]() {
            thread_pool::thread::setThreadName(std::format("sync:{}", self().name.value));

            TimePoint nextWakeUp = ClockSourceType::now();

            while (getSampleRate() > 0.f && lifecycle::isActive(self().state())) {
                const auto period = getChunkPeriod();
                nextWakeUp += period;
                std::this_thread::sleep_until(nextWakeUp);

                if (self().state() == lifecycle::State::PAUSED) {
                    _blockingSync_lastUpdateTime = ClockSourceType::now();
                    continue;
                }

                self().progress->incrementAndGet();
                self().progress->notify_all();
            }

            _blockingSync_timerRunning = false;
            _blockingSync_timerDone    = true;
            _blockingSync_timerDone.notify_all();
        });
    }

    void stopTimer() {
        if (!_blockingSync_timerRunning.exchange(false)) {
            return;
        }
        _blockingSync_timerDone.wait(false);
    }
};

} // namespace gr

#endif // GNURADIO_BLOCKINGSYNC_HPP
