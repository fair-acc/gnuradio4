#ifndef GNURADIO_DEVICE_TEST_HELPERS_HPP
#define GNURADIO_DEVICE_TEST_HELPERS_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <print>
#include <string_view>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

namespace gr::test {

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

/**
 * @brief How many times the dispatcher announced a CPU fallback while `run` executed.
 *
 * Matching output between a host run and a device run does NOT prove the device ran anything — a block that
 * quietly fell back computes the same answer. This reads the dispatcher's own report instead.
 *
 * Zero means "the dispatcher did not refuse a kernel", not "the block was dispatched at all": a block on a domain
 * that names no device never reaches dispatch and scores zero trivially. Pair it with a served device domain --
 * `firstServedSyclDomain()` -- and the two together are the proof.
 */
template<typename TRun>
[[nodiscard]] inline std::size_t cpuFallbacksDuring(TRun&& run) {
    gr::log::HistoryLoggerBackend recorded;
    gr::log::Backend*             previous = gr::log::setBackend(&recorded);
    run();
    gr::log::setBackend(previous);

    std::size_t fallbacks = 0UZ;
    std::ignore           = recorded.snapshot(
        [](const gr::log::LogRecord& record, void* user) noexcept {
            if (std::string_view(record.text, record.textLength).contains("on the CPU")) {
                ++*static_cast<std::size_t*>(user);
            }
        },
        &fallbacks);
    return fallbacks;
}

// device kernel code lives in device_test_helpers.cpp, separate from Boost.UT
// suite registration, to avoid AdaptiveCpp SSCP interference with global constructors.

void deviceParallelMultiply(const float* in, float* out, std::size_t N, float factor);
void deviceParallelComplexRotate(const gr::complex<float>* in, gr::complex<float>* out, std::size_t N, gr::complex<float> factor);

} // namespace gr::test

#endif // GNURADIO_DEVICE_TEST_HELPERS_HPP
