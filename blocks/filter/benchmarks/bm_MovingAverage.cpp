#include <benchmark.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/filter/MovingAverage.hpp>

namespace {
using gr::filter::Statistic;

constexpr std::size_t kSamples = 1UZ << 22UZ;
constexpr std::size_t kRepeats = 5UZ;

template<typename T>
[[nodiscard]] std::vector<T> noise() {
    constexpr std::size_t             kLanes = gr::meta::complex_like<T> ? 2UZ : 1UZ;
    gr::signal::NoiseGenerator<float> generator;
    generator.configure(gr::signal::NoiseType::Gaussian, 1.f, 1.f, 7U);
    std::vector<float> drawn(kSamples * kLanes);
    generator.fill(drawn);
    if constexpr (gr::meta::complex_like<T>) {
        return std::views::iota(0UZ, kSamples) | std::views::transform([&drawn](std::size_t i) { return T{drawn[2UZ * i], drawn[2UZ * i + 1UZ]}; }) | std::ranges::to<std::vector<T>>();
    } else {
        return drawn;
    }
}

template<typename T, Statistic statistic, bool exponential>
void benchmarkAverage(std::string_view name, gr::Size_t frameSize, gr::Size_t nFrames, gr::Size_t decimation) {
    using namespace benchmark;
    using Block = gr::filter::Average<T, statistic, exponential>;
    Block block({{"frame_size", frameSize}, {"n_frames", nFrames}, {"decimation", decimation}});
    block.init(block.progress);
    const std::vector<T>              samples = noise<T>();
    std::vector<typename Block::TOut> figures(kSamples / decimation);

    ::benchmark::benchmark<1LU>{name}.repeat<kRepeats>(kSamples) = [&] {
        std::ignore = block.processBulk(std::span<const T>(samples), std::span(figures));
        force_to_memory(figures);
    };
}
} // namespace

inline const boost::ut::suite<"MovingAverage"> _averages = [] {
    using namespace benchmark;
    benchmarkAverage<float, Statistic::mean, false>("moving mean<float>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::variance, false>("moving variance<float>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::maximum, false>("moving maximum<float>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::meanSquare, false>("moving meanSquare<float>, 4096 x 16 (control for logPower)", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::logPower, false>("moving logPower<float>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<std::complex<float>, Statistic::mean, false>("moving mean<complex<float>>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<std::complex<float>, Statistic::meanSquare, false>("moving meanSquare<complex<float>>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::mean, true>("exponential mean<float>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::variance, true>("exponential variance<float>, 4096 x 16", 4096U, 16U, 1U);
    benchmarkAverage<float, Statistic::rms, false>("moving rms<float>, 1 x 64, integrate and dump", 1U, 64U, 64U);

    static const std::vector<float> stream = noise<float>();
    static std::vector<float>       rms(kSamples / 64UZ);
    "control: rms of 64-sample windows, plain double loop"_benchmark.repeat<kRepeats>(kSamples) = [] {
        for (std::size_t window = 0UZ; window < rms.size(); ++window) {
            double squares = 0.;
            for (std::size_t i = window * 64UZ; i < (window + 1UZ) * 64UZ; ++i) {
                squares += static_cast<double>(stream[i]) * static_cast<double>(stream[i]);
            }
            rms[window] = static_cast<float>(std::sqrt(squares / 64.));
        }
        force_to_memory(rms);
    };
};

int main() {}
