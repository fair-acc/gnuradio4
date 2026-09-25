#ifndef GNURADIO_TRIGGER_MONOTONICCLOCK_HPP
#define GNURADIO_TRIGGER_MONOTONICCLOCK_HPP

#include <chrono>
#include <cstdint>

namespace gr::blocks::trigger {

inline constexpr std::uint64_t kUnknownTime = 0U;

/**
 * A block's own measure of elapsed time, from a steady source rather than the wall clock.
 *
 * It times a wait that no sample stream can measure -- a debounce, a throttle, a sequence timeout -- and none of that
 * needs a calendar, only monotonicity, so a clock adjustment cannot make a duration negative here.
 *
 * This clock is **private to one block**. Two blocks must not compare readings from it, and it never travels on an
 * event: the `trigger_time` family is what relates one block's events to another's.
 *
 * A device body has no portable clock: SYCL exposes none, and a cycle counter is neither in nanoseconds nor
 * comparable across queues. A kernel therefore reads `kUnknownTime`.
 */
[[nodiscard]] inline std::uint64_t monotonicNowNs() noexcept {
#if defined(__SYCL_DEVICE_ONLY__) || defined(__HIPSYCL_DEVICE_ONLY__) || defined(SYCL_DEVICE_ONLY)
    return kUnknownTime;
#else
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_MONOTONICCLOCK_HPP
