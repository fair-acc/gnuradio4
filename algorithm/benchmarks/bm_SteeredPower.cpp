#include "benchmark.hpp"

#include <array>
#include <cmath>

#include <gnuradio-4.0/algorithm/SteeredPower.hpp>

const boost::ut::suite<"SteeredPower"> powerBenchmarks = [] {
    using namespace benchmark;
    constexpr std::size_t               kPairs  = 120UZ;
    constexpr std::size_t               kLength = 55UZ;
    std::array<float, kPairs * kLength> curves{};
    std::array<float, kPairs>           lags{};
    for (std::size_t i = 0UZ; i < curves.size(); ++i) {
        curves[i] = std::sin(static_cast<float>(i));
    }
    for (std::size_t p = 0UZ; p < lags.size(); ++p) {
        lags[p] = static_cast<float>(p % 43UZ) - 21.25f;
    }
    "120 pair fractional-lag sum"_benchmark.repeat<10'000>() = [&] {
        force_to_memory(curves);
        force_to_memory(lags);
        const float power = gr::algorithm::computeSteeredPower(curves.data(), lags.data(), kPairs, kLength);
        fake_read(power);
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
