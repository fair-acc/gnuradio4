#include <benchmark.hpp>

#include <complex>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/filter/Hilbert.hpp>

namespace {
constexpr std::size_t kSamples = 1UZ << 22UZ;
constexpr std::size_t kRepeats = 5UZ;

[[nodiscard]] const std::vector<float>& noise() {
    static const std::vector<float> samples = [] {
        gr::signal::NoiseGenerator<float> generator;
        generator.configure(gr::signal::NoiseType::Gaussian, 1.f, 0.f, 7U);
        std::vector<float> drawn(kSamples);
        generator.fill(drawn);
        return drawn;
    }();
    return samples;
}

template<typename TBlock, typename... TOut>
void benchmarkHilbert(std::string_view name) {
    TBlock block;
    block.init(block.progress);
    block.start();
    std::tuple<std::vector<TOut>...> outputs{std::vector<TOut>(kSamples)...};
    const std::span<const float>     samples(noise());
    ::benchmark::benchmark<1LU>{name}.repeat<kRepeats>(kSamples) = [&] {
        std::apply([&](auto&... out) { std::ignore = block.processBulk(samples, std::span(out)...); }, outputs);
        benchmark::force_to_memory(std::get<0>(outputs));
    };
}
} // namespace

inline const boost::ut::suite<"Hilbert"> _hilbert = [] {
    benchmarkHilbert<gr::filter::HilbertTransform<float>, std::complex<float>>("HilbertTransform<float>, 65 taps");
    benchmarkHilbert<gr::filter::HilbertTransformRealImag<float>, float, float>("HilbertTransformRealImag<float>, 65 taps");
    benchmarkHilbert<gr::filter::HilbertEnvelope<float>, float>("HilbertEnvelope<float>, 65 taps");
};

int main() {}
