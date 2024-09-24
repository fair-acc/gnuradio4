#ifndef GNURADIO_PERFORMANCEMONITOR_HPP
#define GNURADIO_PERFORMANCEMONITOR_HPP

#include <limits>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/reflection.hpp>
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
        return fmt::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ", si_prefixes[exponent], unit);
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

    return fmt::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ", si_prefixes[exponent], unit);
}
} // namespace details

template<typename T>
struct PerformanceMonitor : public Block<PerformanceMonitor<T>> {
    using Description = Doc<R""(The `PerformanceMonitor` block is used to track and report on performance metrics.
Specifically, it monitors memory usage, including resident and virtual memory, as well as the sample rate.
The results of this monitoring can be printed to the console or saved to a CSV file for further analysis.
Additionally, the block provides three optional output ports for real-time streaming of the current resident memory, virtual memory, and sample rate.
)"">;

    using ClockSourceType = std::chrono::system_clock;
    PortIn<T>                 in;
    PortOut<double, Optional> outRes;  // Resident Set Size: number of bytes the process has in real memory, estimated and smoothed
    PortOut<double, Optional> outRate; // Effective sample rate in Hz

    gr::Annotated<gr::Size_t, "in samples", Doc<"evaluate performance every `N` samples">, Visible> evaluate_perf_rate{1'000'000};
    // Note: `publish_rate` is approximate and depends on `evaluate_perf_rate`.
    // If it takes more time to collect `evaluate_perf_rate` samples than the actual update rate can be much higher than `publish_rate`.
    gr::Annotated<float, "in sec", Doc<"write output approx. every `N` seconds">, Visible>                   publish_rate{1.f};
    gr::Annotated<std::string, "file path", Doc<"path to output csv file, `` -> print to console">, Visible> output_csv_file_path = "";

    GR_MAKE_REFLECTABLE(PerformanceMonitor, in, outRes, outRate, publish_rate, evaluate_perf_rate, output_csv_file_path);

    // statistics of updates
    gr::Size_t n_writes{0U};
    gr::Size_t n_updates_res{0U};
    gr::Size_t n_updates_rate{0U};

    gr::Size_t                               _nSamplesCounter{0};
    std::ofstream                            _file;
    std::chrono::time_point<ClockSourceType> _lastTimePoint = ClockSourceType::now();
    float                                    _timeFromLastUpdate{0.f}; // in sec
    bool                                     _addCsvHeader = true;

    void start() {
        _lastTimePoint   = ClockSourceType::now();
        _nSamplesCounter = 0;
        n_writes         = 0U;
        n_updates_res    = 0U;
        n_updates_rate   = 0U;
        openFile();
    }

    void stop() { closeFile(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outResSpan, gr::OutputSpanLike auto& outRateSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), static_cast<std::size_t>(evaluate_perf_rate - _nSamplesCounter));
        std::ignore                = inSpan.consume(nSamples);
        _nSamplesCounter += static_cast<gr::Size_t>(nSamples);

        if (_nSamplesCounter >= evaluate_perf_rate) {
            addNewMetrics(outResSpan, outRateSpan);
        } else {
            if (outResSpan.size() > 0) {
                outResSpan.publish(0);
            }
            if (outRateSpan.size() > 0) {
                outRateSpan.publish(0);
            }
        };

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
                throw gr::exception(fmt::format("failed to open file '{}'.", output_csv_file_path));
            }
        }
    }

    void addNewMetrics(gr::OutputSpanLike auto& outResSpan, gr::OutputSpanLike auto& outRateSpan) {
        const std::chrono::time_point<ClockSourceType> timeNow = ClockSourceType::now();
        const auto                                     dTime   = std::chrono::duration_cast<std::chrono::microseconds>(timeNow - _lastTimePoint).count();
        const double                                   rate    = (dTime == 0) ? 0. : static_cast<double>(_nSamplesCounter) * 1.e6 / static_cast<double>(dTime);

        _timeFromLastUpdate += static_cast<float>(dTime) / 1.e6f; // microseconds to seconds

        const auto memoryStat   = gr::memory::getUsage();
        const auto residentSize = static_cast<double>(memoryStat.residentSize);

        // write to the output ports
        if (outResSpan.size() >= 1) {
            outResSpan[0] = residentSize;
            outResSpan.publish(1);
            n_updates_res++;
        }
        if (outRateSpan.size() >= 1) {
            outRateSpan[0] = rate;
            outRateSpan.publish(1);
            n_updates_rate++;
        }

        if (_timeFromLastUpdate >= publish_rate) {
            if (output_csv_file_path == "") {
                fmt::println("Performance at {}, #{} dT:{} s, rate:{}, memory_resident:{}", //
                    gr::time::getIsoTime(), n_writes, _timeFromLastUpdate, details::to_si_prefix(rate, "S/s"), details::to_si_prefix(residentSize, "b"));
            } else {
                if (_file.is_open()) {
                    if (_addCsvHeader) {
                        _file << "Id,Time,Rate [Hz],Memory.Resident[bytes],Memory.Virtual[bytes]" << std::endl;
                        _addCsvHeader = false;
                    }
                    _file << n_writes << "," << gr::time::getIsoTime() << "," << rate << "," << residentSize << std::endl;
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

auto registerPerformanceMonitor = gr::registerBlock<gr::testing::PerformanceMonitor, float, double>(gr::globalBlockRegistry());

#endif // GNURADIO_PERFORMANCEMONITOR_HPP
