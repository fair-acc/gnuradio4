#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/AllpassHalfBand.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using HalfBand = gr::algorithm::filter::AllpassHalfBand<double>;

struct BandPower {
    double low  = 0.0;
    double high = 0.0;
};

/// drive a long real tone at `f` cycles/sample through the splitter and measure both band powers
[[nodiscard]] BandPower powerAt(double f, std::size_t nSections) {
    const auto          alphas = HalfBand::designAllpass(nSections);
    std::vector<double> even, odd;
    HalfBand::splitCoefficients(alphas, even, odd);

    constexpr std::size_t kSamples = 20000UZ;
    std::vector<double>   x(kSamples);
    for (std::size_t n = 0UZ; n < kSamples; ++n) {
        x[n] = std::cos(2.0 * std::numbers::pi * f * static_cast<double>(n));
    }
    std::vector<double> low(kSamples / 2UZ), high(kSamples / 2UZ);
    std::vector<double> stateEven(HalfBand::stateSize(even.size()), 0.0), stateOdd(HalfBand::stateSize(odd.size()), 0.0);
    HalfBand::split<double>(x, even, odd, std::span<double>{stateEven}, std::span<double>{stateOdd}, std::span<double>{low}, std::span<double>{high});

    BandPower         power;
    const std::size_t from = low.size() / 2UZ; // past the recursion's settling
    for (std::size_t m = from; m < low.size(); ++m) {
        power.low += low[m] * low[m];
        power.high += high[m] * high[m];
    }
    const double n = static_cast<double>(low.size() - from);
    power.low      = power.low / n * 2.0; // a real tone carries half its power in each rotation
    power.high     = power.high / n * 2.0;
    return power;
}
} // namespace

const boost::ut::suite<"AllpassHalfBand"> halfBandTests = [] {
    using namespace boost::ut;
    using gr::test::eq;

    "the design reproduces the Butterworth half-band coefficients"_test = [] {
        // one section is the order-3 Butterworth half-band, whose pole pair sits at +/- j/sqrt(3)
        const auto one = HalfBand::designAllpass(1UZ);
        expect(eq(one.size(), 1UZ)) << "the order-3 half-band has exactly one allpass section";
        if (!one.empty()) { // boost::ut does not short-circuit, and indexing an empty design is UB
            expect(approx(one[0], 1.0 / 3.0, 1e-9)) << std::format("expected 1/3, got {:.9f}", one[0]);
        }

        for (std::size_t k = 1UZ; k <= 4UZ; ++k) {
            const auto alphas = HalfBand::designAllpass(k);
            expect(eq(alphas.size(), k)) << "one coefficient per section";
            for (const double a : alphas) {
                expect(gt(a, 0.0)) << "an allpass coefficient must be inside the unit disc";
                expect(lt(a, 1.0)) << "an allpass coefficient must be inside the unit disc";
            }
        }
    };

    "the two bands are power complementary"_test = [] {
        // |H_lp|^2 + |H_hp|^2 == 1 holds for ANY allpass pair: it tests the structure, not the design
        for (const double f : {0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45}) {
            const auto power = powerAt(f, 3UZ);
            expect(approx(power.low + power.high, 1.0, 5e-3)) << std::format("at f={:.2f} the bands sum to {:.5f}", f, power.low + power.high);
        }
    };

    "the crossover sits at a quarter of the sample rate"_test = [] {
        const auto power = powerAt(0.25, 3UZ);
        expect(approx(power.low, 0.5, 5e-3)) << std::format("low band holds {:.5f} at f=0.25", power.low);
        expect(approx(power.high, 0.5, 5e-3)) << std::format("high band holds {:.5f} at f=0.25", power.high);
    };

    "the response is a monotone lowpass, not an oscillation"_test = [] {
        // the branch pairing is easy to get backwards, and the symptom is a response that passes f=0.2 into
        // the HIGH band; pinned here so it cannot regress silently
        double previous = 2.0;
        for (const double f : {0.05, 0.1, 0.15, 0.2}) {
            const auto power = powerAt(f, 3UZ);
            expect(gt(power.low, 0.9)) << std::format("f={:.2f} should be well inside the low band, holds {:.5f}", f, power.low);
            expect(le(power.low, previous + 1e-3)) << std::format("the low band must not rise with frequency (f={:.2f})", f);
            previous = power.low;
        }
        for (const double f : {0.3, 0.35, 0.4, 0.45}) {
            const auto power = powerAt(f, 3UZ);
            expect(gt(power.high, 0.9)) << std::format("f={:.2f} should be well inside the high band, holds {:.5f}", f, power.high);
        }
    };

    "more sections sharpen the transition"_test = [] {
        const auto few  = powerAt(0.2, 1UZ);
        const auto many = powerAt(0.2, 4UZ);
        expect(gt(many.low, few.low)) << std::format("4 sections pass {:.5f} at f=0.2 against 1 section's {:.5f}", many.low, few.low);
    };
};

int main() { /* not needed for UT */ }
