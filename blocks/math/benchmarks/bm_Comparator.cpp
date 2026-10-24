#include <CopyGraph.hpp>
#include <benchmark.hpp>

#include <cstdint>

#include <gnuradio-4.0/math/Comparator.hpp>

namespace {
constexpr gr::Size_t kSamples = 1U << 22U;
} // namespace

inline const boost::ut::suite<"Comparator"> _comparator = [] {
    using namespace benchmark;
    using gr::blocks::math::Comparator;
    "control: Copy<float>, the same graph"_benchmark.repeat<5>(kSamples) = [] { gr::testing::runCopyGraph<gr::testing::Copy<float>, float>(kSamples); };
    "Comparator<float>"_benchmark.repeat<5>(kSamples)                    = [] { gr::testing::runCopyGraph<Comparator<float>, float>(kSamples, {{"low", 1000.f}, {"high", 3000.f}}); };
    "Comparator<float, uint8_t>"_benchmark.repeat<5>(kSamples)           = [] { gr::testing::runCopyGraph<Comparator<float, std::uint8_t>, float, std::uint8_t>(kSamples, {{"low", 1000.f}, {"high", 3000.f}}); };
};

int main() { /* the suite runs at static initialisation */ }
