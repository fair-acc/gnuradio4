#include <CopyGraph.hpp>
#include <benchmark.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>

#include <gnuradio-4.0/math/Math.hpp>

namespace {
constexpr gr::Size_t kSamples = 1U << 22U;

template<typename T>
struct ScalarClamp : gr::Block<ScalarClamp<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    T              low  = T{100};
    T              high = T{3000};
    GR_MAKE_REFLECTABLE(ScalarClamp, in, out, low, high);

    [[nodiscard]] constexpr T processOne(T sample) const noexcept { return std::clamp(sample, low, high); }
};
} // namespace

inline const boost::ut::suite<"Clamp"> _clamp = [] {
    using namespace benchmark;
    using gr::blocks::math::Clamp;
    "control: Copy<float>, the same graph"_benchmark.repeat<5>(kSamples)               = [] { gr::testing::runCopyGraph<gr::testing::Copy<float>, float>(kSamples); };
    "Clamp<float>"_benchmark.repeat<5>(kSamples)                                       = [] { gr::testing::runCopyGraph<Clamp<float>, float>(kSamples, {{"low", 100.f}, {"high", 3000.f}}); };
    "control: Copy<int16_t>, the same graph"_benchmark.repeat<5>(kSamples)             = [] { gr::testing::runCopyGraph<gr::testing::Copy<std::int16_t>, std::int16_t>(kSamples); };
    "Clamp<int16_t>"_benchmark.repeat<5>(kSamples)                                     = [] { gr::testing::runCopyGraph<Clamp<std::int16_t>, std::int16_t>(kSamples, {{"low", std::int16_t{100}}, {"high", std::int16_t{3000}}}); };
    "control: scalar std::clamp processOne<float>"_benchmark.repeat<5>(kSamples)       = [] { gr::testing::runCopyGraph<ScalarClamp<float>, float>(kSamples); };
    "control: scalar std::clamp processOne<int16_t>"_benchmark.repeat<5>(kSamples)     = [] { gr::testing::runCopyGraph<ScalarClamp<std::int16_t>, std::int16_t>(kSamples); };
    "control: Copy<std::complex<float>>, the same graph"_benchmark.repeat<5>(kSamples) = [] { gr::testing::runCopyGraph<gr::testing::Copy<std::complex<float>>, std::complex<float>>(kSamples); };
    "Clamp<std::complex<float>>"_benchmark.repeat<5>(kSamples)                         = [] { gr::testing::runCopyGraph<Clamp<std::complex<float>>, std::complex<float>>(kSamples, {{"low", 100.f}, {"high", 3000.f}}); };
};

int main() { /* the suite runs at static initialisation */ }
