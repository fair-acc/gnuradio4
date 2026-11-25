#ifndef GNURADIO_PERFORMANCEMONITOR_HPP
#define GNURADIO_PERFORMANCEMONITOR_HPP

#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <format>
#include <limits>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/thread/MemoryMonitor.hpp>

namespace gr::testing {

namespace details {
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
std::string to_si_prefix(T value_base, std::string_view unit = "s", std::size_t significant_digits = 0) {
    static constexpr std::array  si_prefixes{'q', 'r', 'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y', 'R', 'Q'};
    static constexpr long double base  = 1000.0l;
    long double                  value = static_cast<long double>(value_base);

    std::size_t exponent = 10u;
    if (value == 0.0l) {
        return std::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ", si_prefixes[exponent], unit);
    }
    while (value >= base && exponent < si_prefixes.size()) {
        value /= base;
        ++exponent;
    }
    while (value < 1.0l && exponent > 0u) {
        value *= base;
        --exponent;
    }
    if (significant_digits == 0 && exponent > 10) {
        if (value < 10.0l) {
            significant_digits = 2u;
        } else if (value < 100.0l) {
            significant_digits = 1u;
        }
    } else if (significant_digits == 1 && value >= 100.0l) {
        --significant_digits;
    } else if (significant_digits >= 2u) {
        if (value >= 100.0l) {
            significant_digits -= 2u;
        } else if (value >= 10.0l) {
            --significant_digits;
        }
    }

    return std::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ", si_prefixes[exponent], unit);
}

[[nodiscard]] constexpr double perSecond(std::uint64_t count, std::int64_t microseconds) noexcept { return microseconds <= 0 ? 0. : static_cast<double>(count) * 1.e6 / static_cast<double>(microseconds); }
} // namespace details

struct DeviceCounts {
    std::uint64_t syncs             = 0U;
    std::uint64_t dispatches        = 0U;
    std::uint64_t deviceToHostBytes = 0U;
};

GR_REGISTER_BLOCK(gr::testing::PerformanceMonitor, [T], [ float, double, std::complex<float>, gr::DataSet<float> ])

template<typename T>
struct PerformanceMonitor : Block<PerformanceMonitor<T>> {
    using Description = Doc<R""(The `PerformanceMonitor` block is used to track and report on performance metrics.
Specifically, it monitors memory usage, including resident and virtual memory, as well as the sample rate.
The results of this monitoring can be printed to the console or saved to a CSV file for further analysis.
Its optional asynchronous ports stream the resident and virtual memory, the rate in items per second (a `DataSet` input counts data sets, never reading them),
and, for a `device_domain`, that device context's synchronisations, kernel dispatches and device-to-host bytes per second.
It evaluates every `evaluate_perf_rate` items or every `publish_rate` seconds, whichever comes first.
)"">;

    using ClockSourceType = std::chrono::steady_clock;
    PortIn<T>                        in;
    PortOut<double, Optional, Async> outRes;                // bytes the process has in real memory (Resident Set Size); 0 under Emscripten
    PortOut<double, Optional, Async> outVirt;               // virtual memory in bytes, as `gr::memory::Stat::virtualSize`
    PortOut<double, Optional, Async> outRate;               // items per second
    PortOut<double, Optional, Async> outDeviceSyncRate;     // per second
    PortOut<double, Optional, Async> outDeviceDispatchRate; // per second; 0 on a backend that does not count them
    PortOut<double, Optional, Async> outDeviceToHostRate;   // bytes per second; 0 on a backend that does not count them

    gr::Annotated<gr::Size_t, "in samples", Doc<"evaluate performance every `N` samples">>                               evaluate_perf_rate{1'000'000};
    gr::Annotated<float, "in sec", Doc<"evaluate and write output at least every `N` seconds">>                          publish_rate{1.f};
    gr::Annotated<std::string, "file path", Doc<"path to output csv file, `` -> print to console">>                      output_csv_file_path = "";
    gr::Annotated<std::string, "device domain", Doc<"compute domain whose context's counters are reported, `` -> none">> device_domain        = "";

    GR_MAKE_REFLECTABLE(PerformanceMonitor, in, outRes, outVirt, outRate, outDeviceSyncRate, outDeviceDispatchRate, outDeviceToHostRate, publish_rate, evaluate_perf_rate, output_csv_file_path, device_domain);

    // statistics of updates
    gr::Size_t n_writes{0U};
    gr::Size_t n_updates_res{0U};
    gr::Size_t n_updates_rate{0U};

    gr::Size_t                               _nSamplesCounter{0};
    std::ofstream                            _file;
    std::chrono::time_point<ClockSourceType> _lastTimePoint = ClockSourceType::now();
    float                                    _timeFromLastUpdate{0.f}; // in sec
    bool                                     _addCsvHeader        = true;
    const gr::device::DeviceContext*         _deviceDomainContext = nullptr;
    DeviceCounts                             _lastDeviceCounts{}; // at the last evaluation

    void start() {
        _lastTimePoint       = ClockSourceType::now();
        _nSamplesCounter     = 0;
        n_writes             = 0U;
        n_updates_res        = 0U;
        n_updates_rate       = 0U;
        _deviceDomainContext = device_domain.value.empty() ? nullptr : gr::device::DeviceContextRegistry::instance().tryResolve(device_domain.value);
        _lastDeviceCounts    = deviceCounts();
        if (!device_domain.value.empty() && _deviceDomainContext == nullptr) {
            gr::log::warning("{}: device_domain '{}' is not served, so its rates stay 0", this->name, device_domain.value);
        }
        openFile();
    }

    void stop() { closeFile(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outResSpan, gr::OutputSpanLike auto& outVirtSpan, gr::OutputSpanLike auto& outRateSpan, gr::OutputSpanLike auto& outSyncSpan, gr::OutputSpanLike auto& outDispatchSpan, gr::OutputSpanLike auto& outToHostSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), static_cast<std::size_t>(evaluate_perf_rate - _nSamplesCounter));
        std::ignore                = inSpan.consume(nSamples);
        _nSamplesCounter += static_cast<gr::Size_t>(nSamples);

        const bool intervalElapsed = ClockSourceType::now() - _lastTimePoint >= std::chrono::duration<float>(publish_rate.value);
        if (_nSamplesCounter >= evaluate_perf_rate || intervalElapsed) {
            addNewMetrics(outResSpan, outVirtSpan, outRateSpan, outSyncSpan, outDispatchSpan, outToHostSpan);
        } else {
            for (auto* span : {&outResSpan, &outVirtSpan, &outRateSpan, &outSyncSpan, &outDispatchSpan, &outToHostSpan}) {
                if (span->size() > 0UZ) {
                    span->publish(0);
                }
            }
        }
        return gr::work::Status::OK;
    }

private:
    void closeFile() {
        if (_file.is_open()) {
            _file.close();
        }
    }

    void openFile() {
        if (output_csv_file_path != "") {
            _file.open(output_csv_file_path, std::ios::out);
            _addCsvHeader = true;
            if (!_file) {
                throw gr::exception(std::format("failed to open file '{}'.", output_csv_file_path));
            }
        }
    }

    [[nodiscard]] DeviceCounts deviceCounts() const noexcept {
        if (_deviceDomainContext == nullptr) {
            return {};
        }
        const gr::device::DeviceContext& context = *_deviceDomainContext;
        return {.syncs = context.syncCount(), .dispatches = context.gpuDispatches.load(std::memory_order_relaxed), .deviceToHostBytes = context.deviceToHostBytes.load(std::memory_order_relaxed)};
    }

    void addNewMetrics(gr::OutputSpanLike auto& outResSpan, gr::OutputSpanLike auto& outVirtSpan, gr::OutputSpanLike auto& outRateSpan, gr::OutputSpanLike auto& outSyncSpan, gr::OutputSpanLike auto& outDispatchSpan, gr::OutputSpanLike auto& outToHostSpan) {
        const std::chrono::time_point<ClockSourceType> timeNow   = ClockSourceType::now();
        const auto                                     dTime     = std::chrono::duration_cast<std::chrono::microseconds>(timeNow - _lastTimePoint).count();
        const auto                                     perSecond = [dTime](std::uint64_t count) { return details::perSecond(count, dTime); };
        const double                                   rate      = perSecond(_nSamplesCounter);

        _timeFromLastUpdate += static_cast<float>(dTime) / 1.e6f; // microseconds to seconds

        const auto memoryStat   = gr::memory::getUsage();
        const auto residentSize = static_cast<double>(memoryStat.residentSize);
        const auto virtualSize  = static_cast<double>(memoryStat.virtualSize);

        const auto publishOne = [](auto& span, double value) {
            if (span.size() == 0UZ) {
                return false;
            }
            span[0] = value;
            span.publish(1);
            return true;
        };
        n_updates_res += publishOne(outResSpan, residentSize) ? 1U : 0U;
        std::ignore = publishOne(outVirtSpan, virtualSize);
        n_updates_rate += publishOne(outRateSpan, rate) ? 1U : 0U;
        const DeviceCounts counts = deviceCounts();
        std::ignore               = publishOne(outSyncSpan, perSecond(counts.syncs - _lastDeviceCounts.syncs));
        std::ignore               = publishOne(outDispatchSpan, perSecond(counts.dispatches - _lastDeviceCounts.dispatches));
        std::ignore               = publishOne(outToHostSpan, perSecond(counts.deviceToHostBytes - _lastDeviceCounts.deviceToHostBytes));
        _lastDeviceCounts         = counts;

        if (_timeFromLastUpdate >= publish_rate) {
            if (output_csv_file_path == "") {
                const std::string rateText     = details::to_si_prefix(rate, "S/s");
                const std::string residentText = details::to_si_prefix(residentSize, "b");
                const std::string virtualText  = details::to_si_prefix(virtualSize, "b");
                std::println("Performance at {}, #{} dT:{} s, rate:{}, memory_resident:{}, memory_virtual:{}", gr::time::getIsoTime(), n_writes, _timeFromLastUpdate, rateText, residentText, virtualText);
            } else {
                if (_file.is_open()) {
                    if (_addCsvHeader) {
                        _file << "Id,Time,Rate [Hz],Memory.Resident[bytes],Memory.Virtual[bytes]\n";
                        _addCsvHeader = false;
                    }
                    _file << n_writes << "," << gr::time::getIsoTime() << "," << rate << "," << residentSize << "," << virtualSize << '\n';
                }
            }
            _timeFromLastUpdate = 0.f;
            n_writes++;
        }

        _nSamplesCounter = 0;
        _lastTimePoint   = timeNow;
    }
};

} // namespace gr::testing

#endif // GNURADIO_PERFORMANCEMONITOR_HPP
