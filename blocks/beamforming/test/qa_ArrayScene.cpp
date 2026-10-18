#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ranges>

#include <gnuradio-4.0/testing/ArrayScene.hpp>

namespace {
using namespace boost::ut;

const auto scene = [] {
    "sensor noise is reproducible, distinct per microphone and 20 dB below clean mixture power"_test = [] {
        constexpr std::array kClean{-3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f};
        auto                 noisy = kClean;
        gr::testing::addMicrophoneNoise(noisy, 0UZ);
        double addedPower = 0.0;
        for (std::size_t n = 0UZ; n < kClean.size(); ++n) {
            const double added = static_cast<double>(noisy[n]) - static_cast<double>(kClean[n]);
            addedPower += added * added;
        }
        expect(le(std::abs(44.0 / addedPower - 100.0), 1e-4));
        auto repeated = kClean;
        gr::testing::addMicrophoneNoise(repeated, 0UZ);
        expect(std::ranges::equal(noisy, repeated));
        auto otherMicrophone = kClean;
        gr::testing::addMicrophoneNoise(otherMicrophone, 1UZ);
        expect(!std::ranges::equal(noisy, otherMicrophone));
        std::array<float, 8> silent{};
        gr::testing::addMicrophoneNoise(silent, 0UZ);
        expect(std::ranges::all_of(silent, [](float value) { return value == 0.f; }));
        gr::testing::addMicrophoneNoise({}, 0UZ);
        std::array single{2.f};
        gr::testing::addMicrophoneNoise(single, 0UZ);
        expect(le(std::abs(std::abs(single[0] - 2.f) - 0.2f), 1e-6f));
        for (const float amplitude : {1e20f, 1e-20f}) {
            std::array extreme{amplitude};
            gr::testing::addMicrophoneNoise(extreme, 0UZ);
            const double relativeNoise = (static_cast<double>(extreme[0]) - static_cast<double>(amplitude)) / static_cast<double>(amplitude);
            expect(std::isfinite(extreme[0]));
            expect(le(std::abs(std::abs(relativeNoise) - 0.1), 1e-6));
        }
    };

    "arrival sign, fractional interpolation, gain and mixing follow explicit sample fixtures"_test = [] {
        constexpr std::size_t   kCount = 16UZ;
        constexpr std::uint32_t kSeed  = 21U;
        const auto              noise  = gr::testing::whiteNoise(kCount + 2UZ * gr::testing::kMaxArrivalDelaySamples, kSeed);
        constexpr std::array    kDelays{0.0, 1.0, -1.0, 0.25, -0.25};
        for (std::size_t fixture = 0UZ; fixture < kDelays.size(); ++fixture) {
            const double     delay = kDelays[fixture];
            const std::array arrivals{gr::testing::Arrival{delay, -2.0, kSeed}};
            const auto       actual = gr::testing::delayedNoises(arrivals, kCount);
            for (std::size_t n = 0UZ; n < kCount; ++n) {
                const std::size_t centre = n + gr::testing::kMaxArrivalDelaySamples;
                const std::array  expected{noise[centre], noise[centre - 1UZ], noise[centre + 1UZ], 0.75 * noise[centre] + 0.25 * noise[centre - 1UZ], 0.75 * noise[centre] + 0.25 * noise[centre + 1UZ]};
                expect(le(std::abs(static_cast<double>(actual[n]) + 2.0 * expected[fixture]), 1e-7));
            }
            const auto longer = gr::testing::delayedNoises(arrivals, 2UZ * kCount);
            expect(std::ranges::equal(actual, longer | std::views::take(kCount)));
        }
        const std::array cancellation{gr::testing::Arrival{0.25, 1.0, kSeed}, gr::testing::Arrival{0.25, -1.0, kSeed}};
        expect(std::ranges::all_of(gr::testing::delayedNoises(cancellation, kCount), [](float value) { return value == 0.f; }));
        expect(std::ranges::all_of(gr::testing::delayedNoises({}, kCount), [](float value) { return value == 0.f; }));
        expect(gr::testing::delayedNoises(cancellation, 0UZ).empty());
    };
};
} // namespace

int main() {
    "the array scene"_test = scene;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
