#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseArbitraryResampler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using Resampler = gr::algorithm::filter::PolyphaseArbitraryResampler<float>;

[[nodiscard]] std::vector<float> resample(std::span<const float> input, double ratio, std::size_t nPhases, std::size_t nOutputs) {
    const auto        prototype = Resampler::designPrototype(32UZ * nPhases, nPhases);
    const auto        phases    = Resampler::decompose(prototype, nPhases);
    const std::size_t phaseLen  = Resampler::phaseLength(prototype.size(), nPhases);

    std::vector<float> out(nOutputs);
    double             phase    = 0.0;
    const auto         progress = Resampler::resample<float>(input, out, phases, phaseLen, nPhases, ratio, phase);
    out.resize(progress.produced);
    return out;
}
} // namespace

const boost::ut::suite<"PolyphaseArbitraryResampler"> arbTests = [] {
    using namespace boost::ut;

    "a constant resamples to the same constant"_test = [] {
        constexpr std::size_t    kPhases = 16UZ;
        const std::vector<float> input(512UZ, 2.5f);
        const auto               out = resample(input, 1.37, kPhases, 128UZ);

        for (std::size_t n = 64UZ; n < out.size(); ++n) { // past the filter's settling
            expect(approx(out[n], 2.5f, 0.05f)) << std::format("output {} is {:.5f}, not the constant 2.5", n, out[n]);
        }
    };

    "the phase carries, so a split run matches a whole one"_test = [] {
        constexpr std::size_t kPhases   = 16UZ;
        constexpr double      kRatio    = 1.37;
        const auto            prototype = Resampler::designPrototype(32UZ * kPhases, kPhases);
        const auto            phases    = Resampler::decompose(prototype, kPhases);
        const std::size_t     phaseLen  = Resampler::phaseLength(prototype.size(), kPhases);

        std::vector<float> input(1024UZ);
        for (std::size_t n = 0UZ; n < input.size(); ++n) {
            input[n] = std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(n) / 37.f);
        }

        std::vector<float> whole(200UZ);
        double             phaseWhole = 0.0;
        const auto         gotWhole   = Resampler::resample<float>(input, whole, phases, phaseLen, kPhases, kRatio, phaseWhole);

        // the same stream handed over in two halves must land on the same samples: that is what the carried
        // phase buys, and precisely what an integer stride breaks
        std::vector<float> split(200UZ);
        double             phaseSplit = 0.0;
        const auto         first      = Resampler::resample<float>(std::span<const float>{input}.first(400UZ), std::span<float>{split}.first(100UZ), phases, phaseLen, kPhases, kRatio, phaseSplit);
        const auto         second     = Resampler::resample<float>(std::span<const float>{input}.subspan(first.consumed), std::span<float>{split}.subspan(first.produced, 100UZ), phases, phaseLen, kPhases, kRatio, phaseSplit);

        const std::size_t common = std::min(gotWhole.produced, first.produced + second.produced);
        expect(gt(common, 50UZ)) << "too few samples to compare";
        for (std::size_t n = 0UZ; n < common; ++n) {
            expect(approx(split[n], whole[n], 1e-4f)) << std::format("sample {} differs: split {:.6f} vs whole {:.6f}", n, split[n], whole[n]);
        }
    };

    "a slow sine survives a fractional ratio"_test = [] {
        constexpr std::size_t kPhases = 32UZ;
        constexpr double      kRatio  = 0.75; // decimating
        constexpr std::size_t kPeriod = 64UZ;

        std::vector<float> input(1024UZ);
        for (std::size_t n = 0UZ; n < input.size(); ++n) {
            input[n] = std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(n) / static_cast<float>(kPeriod));
        }
        const auto out = resample(input, kRatio, kPhases, 256UZ);

        // a sine in, a sine out: the magnitude must stay bounded and the signal must not decay to nothing
        float peak = 0.f;
        for (std::size_t n = 128UZ; n < out.size(); ++n) {
            peak = std::max(peak, std::abs(out[n]));
        }
        expect(gt(peak, 0.5f)) << std::format("the resampled sine peaks at only {:.4f}", peak);
        expect(lt(peak, 1.5f)) << std::format("the resampled sine peaks at {:.4f}: the bank has gain", peak);
    };

    "a chunked stream resamples exactly as one long span does"_test = [] {
        // the same streaming property the Hermite kernel is held to. The case that matters here is a step WIDER
        // than the window, where the read position stops past the span: what cannot be released has to ride on in
        // `phase`, and the arm interpolation has to stay continuous across a whole sample so that reaching the
        // same position as (base, 0.999...) or as (base + 1, 0.0) gives the same answer.
        constexpr std::size_t kPhases   = 16UZ;
        const auto            prototype = Resampler::designPrototype(32UZ * kPhases, kPhases);
        const auto            phases    = Resampler::decompose(prototype, kPhases);
        const std::size_t     phaseLen  = Resampler::phaseLength(prototype.size(), kPhases);

        for (const double step : {0.5, 1.0, 2.0, 3.5, 12.3}) {
            std::vector<float> signal(4096);
            for (std::size_t n = 0UZ; n < signal.size(); ++n) {
                signal[n] = std::sin(0.01f * static_cast<float>(n));
            }

            std::vector<float> whole(300);
            double             wholePhase = 0.0;
            const auto         reference  = Resampler::resample<float>(signal, whole, phases, phaseLen, kPhases, 1.0 / step, wholePhase);

            std::vector<float> chunked;
            double             phase = 0.0;
            std::size_t        base  = 0UZ;
            while (base < signal.size() && chunked.size() < reference.produced) {
                const std::size_t  take = std::min<std::size_t>(97UZ, signal.size() - base);
                std::vector<float> piece(32);
                const auto         progress = Resampler::resample<float>(std::span<const float>{signal.data() + base, take}, piece, phases, phaseLen, kPhases, 1.0 / step, phase);
                expect(le(progress.consumed, take)) << std::format("step {}: released {} of a {}-sample span", step, progress.consumed, take);
                if (progress.consumed == 0UZ && progress.produced == 0UZ) {
                    break;
                }
                for (std::size_t n = 0UZ; n < progress.produced && chunked.size() < reference.produced; ++n) {
                    chunked.push_back(piece[n]);
                }
                base += progress.consumed;
            }

            expect(eq(chunked.size(), static_cast<std::size_t>(reference.produced))) << std::format("step {}: the chunked run stopped short", step);
            const bool   representable = step * 4.0 == std::floor(step * 4.0);
            const double tolerance     = representable ? 0.0 : 1e-6; // see the note in qa_HermiteResampler
            double       worst         = 0.0;
            for (std::size_t n = 0UZ; n < chunked.size(); ++n) {
                worst = std::max(worst, std::abs(static_cast<double>(chunked[n]) - static_cast<double>(whole[n])));
            }
            expect(le(worst, tolerance)) << std::format("step {}: chunking moved the result by {:.3e}, so the phase bookkeeping loses position", step, worst);
        }
    };

    "the arm interpolation carries on into the next sample"_test = [] {
        // the last arm's neighbour is arm 0 of the NEXT sample. Holding the last arm instead would flatten the top
        // 1/nPhases of every sample interval, which is invisible in a sweep across the boundary itself and shows up
        // as a floor that follows the arm count rather than the filter.
        constexpr std::size_t kPhases   = 16UZ;
        const auto            prototype = Resampler::designPrototype(32UZ * kPhases, kPhases);
        const auto            phases    = Resampler::decompose(prototype, kPhases);
        const std::size_t     phaseLen  = Resampler::phaseLength(prototype.size(), kPhases);

        std::vector<float> signal(256);
        for (std::size_t n = 0UZ; n < signal.size(); ++n) {
            signal[n] = std::sin(0.11f * static_cast<float>(n));
        }

        // approaching an integer position from below must converge on the value AT it, not stop an arm short
        const float at        = Resampler::sampleAt<float>(signal, phases, phaseLen, kPhases, 64.0);
        const float justBelow = Resampler::sampleAt<float>(signal, phases, phaseLen, kPhases, 64.0 - 1e-9);
        expect(lt(std::abs(at - justBelow), 1e-5f)) << std::format("the top arm does not reach the next sample: {:.6f} against {:.6f}", justBelow, at);

        // and the approach is monotone in the tone's own direction rather than flattening out
        const float quarter = Resampler::sampleAt<float>(signal, phases, phaseLen, kPhases, 64.0 - 0.25 / static_cast<double>(kPhases));
        const float half    = Resampler::sampleAt<float>(signal, phases, phaseLen, kPhases, 64.0 - 0.50 / static_cast<double>(kPhases));
        expect(lt(std::abs(at - quarter), std::abs(at - half))) << "the last arm's span is flat, so it is not interpolating onward";
    };
};

int main() { /* not needed for UT */ }
