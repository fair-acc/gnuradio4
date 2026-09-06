#ifndef GNURADIO_DEVICE_TEST_HELPERS_HPP
#define GNURADIO_DEVICE_TEST_HELPERS_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <print>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

namespace gr::test {

/// the filter lengths every throughput table sweeps: below 16 a FIR comparison measures memory layout rather
/// than arithmetic, and above 65536 it measures cache misses
inline constexpr std::array<std::size_t, 8UZ> kFilterLengths{16UZ, 32UZ, 64UZ, 512UZ, 1024UZ, 8192UZ, 32768UZ, 65536UZ};

/// half of a transform this size is useful output, and never fewer than 4096 samples of it
[[nodiscard]] inline std::size_t windowForFilterLength(std::size_t nTaps) { return std::max(4096UZ, std::bit_ceil(nTaps)); }

/// a direct filter costs one multiply-add per tap, so a sample count held fixed across the sweep would spend
/// all of its time in the longest filter; a transform-based arm has no such problem and wants kStreamSamples
[[nodiscard]] inline std::size_t samplesForDirectFilter(std::size_t nTaps) {
    constexpr std::size_t kMultiplyAddBudget = 1UZ << 26;
    return std::clamp(kMultiplyAddBudget / nTaps, 4UZ * windowForFilterLength(nTaps), 1UZ << 20);
}

/// repeating a cheap run is free and a long filter is not
[[nodiscard]] inline int timingAttemptsForFilterLength(std::size_t nTaps) { return nTaps <= 1024UZ ? 3 : 1; }

/// long enough that the fixed cost of building and running a graph does not read as the block's throughput
inline constexpr std::size_t kStreamSamples = 1UZ << 20;

[[nodiscard]] inline std::vector<std::string_view> servedDomains() {
    std::vector<std::string_view> domains{"host"};
    for (std::string_view candidate : {"host:sycl", "gpu:sycl"}) {
        if (gr::device::DeviceContextRegistry::instance().tryResolve(candidate) != nullptr) {
            domains.push_back(candidate);
        }
    }
    return domains;
}

[[nodiscard]] inline std::optional<std::string_view> firstServedDomain(std::initializer_list<std::string_view> preference) {
    const auto isServed = [](std::string_view domain) { return gr::device::DeviceContextRegistry::instance().tryResolve(domain) != nullptr; };
    const auto match    = std::ranges::find_if(preference, isServed);
    return match == preference.end() ? std::nullopt : std::optional<std::string_view>(*match);
}

// a SYCL CPU device stands in for a GPU wherever the test checks API behaviour rather than device
// performance, so a machine without a GPU still exercises the path instead of skipping it
[[nodiscard]] inline std::optional<std::string_view> firstServedSyclDomain() {
    std::ignore                                    = gr::device::registerSyclRuntime();
    const std::optional<std::string_view> selected = firstServedDomain({"gpu:sycl", "host:sycl"});

    static bool announced = false;
    if (!announced) {
        announced = true;
        std::println("SYCL device tests run on '{}'", selected.value_or("<none registered — device assertions skipped>"));
    }
    return selected;
}

template<typename TScheduler>
inline void runAbsorbingRefusal(TScheduler& scheduler) {
#if __cpp_exceptions
    try {
        std::ignore = scheduler.runAndWait();
    } catch (...) { // NOLINT(bugprone-empty-catch) — the scheduler's final state is the assertion
    }
#else
    std::ignore = scheduler.runAndWait();
#endif
}

/**
 * @brief How many times the dispatcher refused to run a block on its declared device while `run` executed.
 *
 * Matching output between a host run and a device run does NOT prove the device ran anything — a refused block
 * never produces output at all now, but a block on a domain that names no device computes the same answer on the
 * host. This reads the dispatcher's own report instead.
 *
 * Zero means "the dispatcher did not refuse a kernel", not "the block was dispatched at all": a block on a domain
 * that names no device never reaches dispatch and scores zero trivially. Pair it with a served device domain --
 * `firstServedSyclDomain()` -- and the two together are the proof.
 *
 * The matched prefix is a contract with `ExecutionStrategy::refuseDeviceDispatch`. Changing either text alone
 * makes this silently count zero, which reads exactly like a passing test.
 */
template<typename TRun>
[[nodiscard]] inline std::size_t deviceRefusalsDuring(TRun&& run) {
    gr::log::HistoryLoggerBackend recorded;
    gr::log::Backend*             previous = gr::log::setBackend(&recorded);
    run();
    gr::log::setBackend(previous);

    std::size_t refusals = 0UZ;
    std::ignore          = recorded.snapshot(
        [](const gr::log::LogRecord& record, void* user) noexcept {
            if (std::string_view(record.text, record.textLength).contains("device dispatch refused")) {
                ++*static_cast<std::size_t*>(user);
            }
        },
        &refusals);
    return refusals;
}

// device kernel code lives in device_test_helpers.cpp, separate from Boost.UT
// suite registration, to avoid AdaptiveCpp SSCP interference with global constructors.

void deviceParallelMultiply(const float* in, float* out, std::size_t N, float factor);
void deviceParallelComplexRotate(const gr::complex<float>* in, gr::complex<float>* out, std::size_t N, gr::complex<float> factor);

} // namespace gr::test

#endif // GNURADIO_DEVICE_TEST_HELPERS_HPP
