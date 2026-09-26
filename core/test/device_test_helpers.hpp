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
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

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

/// whether a kernel body is compiled a second time, for a device. Only then can it tell the two apart: the
/// library-only OpenMP backend defines `__acpp_if_target_device` as `if constexpr (false)`, so a body that did run
/// as a kernel still reports the host, and an assertion on that string measures the compilation model, not where
/// the code ran.
#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__) || defined(__ACPP_ENABLE_CUDA_TARGET__) || defined(__ACPP_ENABLE_HIP_TARGET__)
inline constexpr bool kKernelHasDeviceCompilationPass = true;
#else
inline constexpr bool kKernelHasDeviceCompilationPass = false;
#endif

// the sweep helpers live in the installed header now; these names stay so the tests that use them keep working
using gr::testing::firstServedDomain;
using gr::testing::firstServedSyclDomain;
using gr::testing::servedDomains;

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
 * How many times the dispatcher refused to run a block on its declared device while `run` executed.
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
// this header is also compiled into `device_test_helpers.cpp`, a device-kernel TU that links no test framework, so
// the one helper that asserts is offered only where one is present
#if __has_include(<boost/ut.hpp>)
/// Call once per device test binary. A build with a SYCL backend always enumerates a CPU device, so `host:sycl`
/// being unserved means the runtime is broken -- and a file that skips on that silently asserts nothing at all,
/// which reads exactly like a pass. Returns whether SYCL is available, so the caller can skip its device cases.
[[nodiscard]] inline bool requireHostSycl() {
    const bool available = gr::device::registerSyclRuntime();
    boost::ut::expect(!available || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";
    return available;
}
#endif

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
