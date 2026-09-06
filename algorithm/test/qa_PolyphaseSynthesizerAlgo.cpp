#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseSynthesizer.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using Synthesizer = gr::algorithm::filter::PolyphaseSynthesizer<float>;
using Complex     = std::complex<float>;

/// run both passes over `nSets` sets of `nChannels` channel samples, returning the wideband stream
[[nodiscard]] std::vector<Complex> synthesise(std::span<const Complex> channels, std::span<const float> phases, std::size_t phaseLen, std::size_t nChannels, std::size_t nSets) {
    const std::size_t nArmSets = Synthesizer::windowLength(nSets, phaseLen);

    const auto           table = Synthesizer::designTwiddles(nChannels);
    std::vector<Complex> twiddles(nChannels);
    for (std::size_t j = 0UZ; j < nChannels; ++j) {
        twiddles[j] = Complex{table[j].real(), table[j].imag()};
    }
    std::vector<Complex> armInputs(nArmSets * nChannels);
    for (std::size_t m = 0UZ; m < nArmSets; ++m) {
        for (std::size_t p = 0UZ; p < nChannels; ++p) {
            armInputs[m * nChannels + p] = Synthesizer::armInputAt(channels.data() + m * nChannels, twiddles.data(), nChannels, p);
        }
    }

    std::vector<Complex>           out(nSets * nChannels);
    const std::span<const Complex> armSpan{armInputs};
    for (std::size_t m = 0UZ; m < nSets; ++m) {
        for (std::size_t p = 0UZ; p < nChannels; ++p) {
            out[Synthesizer::outputIndex(m, nChannels, p)] = Synthesizer::outputAt(armSpan, phases, phaseLen, nChannels, m, p);
        }
    }
    return out;
}
} // namespace

const boost::ut::suite<"PolyphaseSynthesizer"> synthesizerTests = [] {
    using namespace boost::ut;
    using gr::test::eq;

    "a constant on channel 0 synthesises a constant"_test = [] {
        constexpr std::size_t kChannels = 4UZ;
        constexpr std::size_t kSets     = 24UZ;
        const auto            prototype = Synthesizer::designPrototype(64UZ, kChannels);
        const auto            phases    = Synthesizer::decompose(prototype, kChannels);
        const std::size_t     phaseLen  = Synthesizer::phaseLength(prototype.size(), kChannels);

        std::vector<Complex> channels(Synthesizer::windowLength(kSets, phaseLen) * kChannels, Complex{});
        for (std::size_t m = 0UZ; m < Synthesizer::windowLength(kSets, phaseLen); ++m) {
            channels[m * kChannels + 0UZ] = Complex{1.f, 0.f}; // DC channel only
        }

        const auto out = synthesise(channels, phases, phaseLen, kChannels, kSets);

        // the last set is past the filter's settling, so every sample there should carry the same DC level
        const std::size_t last  = (kSets - 1UZ) * kChannels;
        const float       level = std::abs(out[last]);
        expect(gt(level, 0.f)) << "a DC channel produced nothing";
        for (std::size_t i = 0UZ; i < kChannels; ++i) {
            expect(approx(std::abs(out[last + i]), level, 1e-3f)) << std::format("sample {} of the set is {:.5f} against {:.5f}: DC must not ripple", i, std::abs(out[last + i]), level);
        }
    };

    "a constant on one channel synthesises a constant-magnitude tone"_test = [] {
        constexpr std::size_t kChannels = 4UZ;
        constexpr std::size_t kSets     = 24UZ;
        constexpr std::size_t kTone     = 1UZ;
        const auto            prototype = Synthesizer::designPrototype(64UZ, kChannels);
        const auto            phases    = Synthesizer::decompose(prototype, kChannels);
        const std::size_t     phaseLen  = Synthesizer::phaseLength(prototype.size(), kChannels);

        std::vector<Complex> channels(Synthesizer::windowLength(kSets, phaseLen) * kChannels, Complex{});
        for (std::size_t m = 0UZ; m < Synthesizer::windowLength(kSets, phaseLen); ++m) {
            channels[m * kChannels + kTone] = Complex{1.f, 0.f};
        }

        const auto        out   = synthesise(channels, phases, phaseLen, kChannels, kSets);
        const std::size_t last  = (kSets - 1UZ) * kChannels;
        const float       level = std::abs(out[last]);
        expect(gt(level, 0.f)) << "an occupied channel produced nothing";
        for (std::size_t i = 0UZ; i < kChannels; ++i) { // a pure tone has constant magnitude, whatever its phase
            expect(approx(std::abs(out[last + i]), level, 1e-3f)) << std::format("sample {} has magnitude {:.5f} against {:.5f}", i, std::abs(out[last + i]), level);
        }
    };

    "an empty bank synthesises silence"_test = [] {
        constexpr std::size_t kChannels = 8UZ;
        constexpr std::size_t kSets     = 8UZ;
        const auto            prototype = Synthesizer::designPrototype(64UZ, kChannels);
        const auto            phases    = Synthesizer::decompose(prototype, kChannels);
        const std::size_t     phaseLen  = Synthesizer::phaseLength(prototype.size(), kChannels);

        const std::vector<Complex> channels(Synthesizer::windowLength(kSets, phaseLen) * kChannels, Complex{});
        const auto                 out = synthesise(channels, phases, phaseLen, kChannels, kSets);
        for (const auto& sample : out) {
            expect(lt(std::abs(sample), 1e-6f)) << "silence in, silence out";
        }
    };
};

int main() { /* not needed for UT */ }
