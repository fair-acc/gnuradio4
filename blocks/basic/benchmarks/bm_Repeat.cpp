#include <CopyGraph.hpp>
#include <benchmark.hpp>

#include <gnuradio-4.0/basic/Repeat.hpp>

namespace {
constexpr gr::Size_t kSamples = 1U << 20U;
} // namespace

inline const boost::ut::suite<"Repeat"> _repeat = [] {
    using namespace benchmark;
    "control: Copy<float>, the same graph"_benchmark.repeat<5>(kSamples) = [] { gr::testing::runCopyGraph<gr::testing::Copy<float>, float>(kSamples); };
    "Repeat<float>, n 1"_benchmark.repeat<5>(kSamples)                   = [] { gr::testing::runCopyGraph<gr::basic::Repeat<float>, float>(kSamples, {{"n", gr::Size_t(1)}}); };
    "Repeat<float>, n 4"_benchmark.repeat<5>(kSamples)                   = [] { gr::testing::runCopyGraph<gr::basic::Repeat<float>, float>(kSamples, {{"n", gr::Size_t(4)}}); };
    "Repeat<float>, n 64"_benchmark.repeat<5>(kSamples)                  = [] { gr::testing::runCopyGraph<gr::basic::Repeat<float>, float>(kSamples, {{"n", gr::Size_t(64)}}); };
};

int main() {}
