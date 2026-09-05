#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using Resampler = gr::algorithm::filter::PolyphaseResampler<float>;

/// upsample by L, convolve with the prototype, keep every Mth sample -- the definition the polyphase form optimises
/// the prototype the polyphase bank actually realises, read back out of it
[[nodiscard]] std::vector<float> reconstructPrototype(std::span<const float> phases, std::size_t phaseLen, std::size_t interpolation) {
    std::vector<float> prototype(interpolation * phaseLen, 0.f);
    for (std::size_t p = 0UZ; p < interpolation; ++p) {
        for (std::size_t j = 0UZ; j < phaseLen; ++j) {
            prototype[j * interpolation + p] = phases[p * phaseLen + j];
        }
    }
    return prototype;
}

[[nodiscard]] std::vector<float> resampleByDefinition(std::span<const float> signal, std::span<const float> prototype, std::size_t interpolation, std::size_t decimation, std::size_t nOutputs) {
    std::vector<float> upsampled(signal.size() * interpolation, 0.f);
    for (std::size_t i = 0UZ; i < signal.size(); ++i) {
        upsampled[i * interpolation] = signal[i];
    }
    std::vector<float> outputs(nOutputs, 0.f);
    for (std::size_t n = 0UZ; n < nOutputs; ++n) {
        const std::size_t centre = n * decimation;
        float             acc    = 0.f;
        for (std::size_t k = 0UZ; k < prototype.size(); ++k) {
            if (centre >= k) {
                acc += prototype[k] * upsampled[centre - k];
            }
        }
        outputs[n] = acc;
    }
    return outputs;
}
} // namespace

const boost::ut::suite<"PolyphaseResampler"> _polyphase = [] {
    using namespace boost::ut;

    "the prototype is a symmetric low-pass of unit DC gain"_test = [] {
        const std::vector<float> prototype = Resampler::designPrototype(33UZ, 0.25f);
        expect(eq(prototype.size(), 33UZ));
        expect(std::abs(std::accumulate(prototype.begin(), prototype.end(), 0.f) - 1.f) < 1e-5f) << "a resampler must not change the level of a DC signal";
        bool symmetric = true;
        for (std::size_t k = 0UZ; k < prototype.size() / 2UZ; ++k) {
            symmetric = symmetric && std::abs(prototype[k] - prototype[prototype.size() - 1UZ - k]) < 1e-6f;
        }
        expect(symmetric) << "an asymmetric prototype would make the delay frequency-dependent";
    };

    "decomposition keeps every tap exactly once"_test = [] {
        constexpr std::size_t    kInterpolation = 4UZ;
        const std::vector<float> prototype      = Resampler::designPrototype(18UZ, 0.2f);
        const std::vector<float> phases         = Resampler::decompose(prototype, kInterpolation);
        const std::size_t        phaseLen       = Resampler::phaseLength(prototype.size(), kInterpolation);
        expect(eq(phases.size(), kInterpolation * phaseLen));
        for (std::size_t p = 0UZ; p < kInterpolation; ++p) {
            const float sum = std::accumulate(phases.begin() + static_cast<std::ptrdiff_t>(p * phaseLen), phases.begin() + static_cast<std::ptrdiff_t>((p + 1UZ) * phaseLen), 0.f);
            expect(std::abs(sum - 1.f) < 1e-5f) << std::format("phase {} sums to {}; unequal phases turn a constant into a ripple at the output rate", p, sum);
        }
    };

    "L/M conversion matches upsample-filter-decimate"_test = [] {
        for (const auto [interpolation, decimation] : std::vector<std::pair<std::size_t, std::size_t>>{{1UZ, 1UZ}, {3UZ, 2UZ}, {2UZ, 3UZ}, {5UZ, 4UZ}}) {
            const std::vector<float> prototype = Resampler::designPrototype(24UZ, 0.5f / static_cast<float>(std::max(interpolation, decimation)));
            const std::vector<float> phases    = Resampler::decompose(prototype, interpolation);
            const std::size_t        phaseLen  = Resampler::phaseLength(prototype.size(), interpolation);

            constexpr std::size_t kOutputs = 40UZ;
            std::vector<float>    signal(Resampler::windowLength(kOutputs, interpolation, decimation, phaseLen) + 4UZ);
            for (std::size_t i = 0UZ; i < signal.size(); ++i) { // something with content at several frequencies
                signal[i] = std::sin(0.11f * static_cast<float>(i)) + 0.5f * std::cos(0.37f * static_cast<float>(i));
            }

            // the window the polyphase form indexes starts phaseLen-1 samples before the first sample the definition uses
            std::vector<float> window(signal.size() + phaseLen - 1UZ, 0.f);
            std::ranges::copy(signal, window.begin() + static_cast<std::ptrdiff_t>(phaseLen - 1UZ));

            const std::vector<float> realised = reconstructPrototype(phases, phaseLen, interpolation);
            const std::vector<float> expected = resampleByDefinition(signal, realised, interpolation, decimation, kOutputs);
            bool                     matches  = true;
            for (std::size_t n = 0UZ; n < kOutputs; ++n) {
                const float got = Resampler::sampleAt(window, phases, phaseLen, interpolation, decimation, n);
                matches         = matches && std::abs(got - expected[n]) < 1e-4f;
            }
            expect(matches) << std::format("L/M = {}/{} disagrees with the definition it is supposed to compute", interpolation, decimation);
        }
    };

    "a unit prototype at 1:1 is a pure delay"_test = [] {
        const std::vector<float> prototype{1.f};
        const std::vector<float> phases   = Resampler::decompose(prototype, 1UZ);
        const std::size_t        phaseLen = Resampler::phaseLength(1UZ, 1UZ);
        std::vector<float>       window(16UZ);
        std::iota(window.begin(), window.end(), 1.f);
        bool identity = true;
        for (std::size_t n = 0UZ; n < window.size(); ++n) {
            identity = identity && std::abs(Resampler::sampleAt(window, phases, phaseLen, 1UZ, 1UZ, n) - window[n]) < 1e-6f;
        }
        expect(identity) << "one unit tap at 1:1 has nothing to do";
    };
};

int main() { /* tests run from the suite */ }
