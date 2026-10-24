#include <CopyGraph.hpp>
#include <benchmark.hpp>

#include <gnuradio-4.0/filter/Delay.hpp>

namespace {
constexpr gr::Size_t kSamples = 1U << 22U;
} // namespace

inline const boost::ut::suite<"Delay"> _delay = [] {
    using namespace benchmark;
    "control: Copy<float>, the same graph"_benchmark.repeat<5>(kSamples) = [] { gr::testing::runCopyGraph<gr::testing::Copy<float>, float>(kSamples); };
    "Delay<float>, delay 0"_benchmark.repeat<5>(kSamples)                = [] { gr::testing::runCopyGraph<gr::filter::Delay<float>, float>(kSamples, {{"delay", gr::Size_t(0)}}); };
    "Delay<float>, delay 7"_benchmark.repeat<5>(kSamples)                = [] { gr::testing::runCopyGraph<gr::filter::Delay<float>, float>(kSamples, {{"delay", gr::Size_t(7)}}); };
    "Delay<float>, delay 4096"_benchmark.repeat<5>(kSamples)             = [] { gr::testing::runCopyGraph<gr::filter::Delay<float>, float>(kSamples, {{"delay", gr::Size_t(4096)}}); };
};

int main() {}
