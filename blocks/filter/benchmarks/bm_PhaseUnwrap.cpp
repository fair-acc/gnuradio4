#include <benchmark.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <gnuradio-4.0/filter/PhaseUnwrap.hpp>

namespace {
constexpr std::size_t kSamples = 1UZ << 22UZ;
constexpr std::size_t kRepeats = 5UZ;

[[nodiscard]] std::vector<float> wrappedRamp() {
    std::vector<float> samples(kSamples);
    for (std::size_t i = 0UZ; i < kSamples; ++i) {
        const float phase = 0.37f * static_cast<float>(i % 4096UZ);
        samples[i]        = std::atan2(std::sin(phase), std::cos(phase));
    }
    return samples;
}

void benchmarkUnwrap(std::string_view name, gr::Size_t frameSize, std::string axis) {
    using namespace benchmark;
    static const std::vector<float> samples = wrappedRamp();
    gr::filter::PhaseUnwrap<float>  block({{"frame_size", frameSize}, {"axis", std::move(axis)}});
    block.init(block.progress);
    std::vector<float> unwrapped(kSamples);
    ::benchmark::benchmark<1LU>{name}.repeat<kRepeats>(kSamples) = [&] {
        std::ignore = block.processBulk(std::span<const float>(samples), std::span(unwrapped));
        force_to_memory(unwrapped);
    };
}
} // namespace

inline const boost::ut::suite<"PhaseUnwrap"> _unwrap = [] {
    using namespace benchmark;
    benchmarkUnwrap("PhaseUnwrap<float>, frames, 1", 1U, "frames");
    benchmarkUnwrap("PhaseUnwrap<float>, frames, 4096", 4096U, "frames");
    benchmarkUnwrap("PhaseUnwrap<float>, positions, 4096", 4096U, "positions");

    static const std::vector<float> samples = wrappedRamp();
    static std::vector<float>       unwrapped(kSamples);
    "control: copy only"_benchmark.repeat<kRepeats>(kSamples) = [] {
        std::ranges::copy(samples, unwrapped.begin());
        force_to_memory(unwrapped);
    };
    "control: fft::unwrapPhase over the whole span"_benchmark.repeat<kRepeats>(kSamples) = [] {
        std::ranges::copy(samples, unwrapped.begin());
        gr::algorithm::fft::unwrapPhase(std::span<float>(unwrapped));
        force_to_memory(unwrapped);
    };
};

int main() {}
