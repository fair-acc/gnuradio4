#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FastConvolution.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using Convolution = gr::algorithm::filter::FastConvolution<float>;

/// the definition the frequency-domain form is supposed to compute
[[nodiscard]] std::vector<float> convolveDirectly(std::span<const float> signal, std::span<const float> taps, std::size_t firstOutput, std::size_t nOutputs) {
    std::vector<float> outputs(nOutputs, 0.f);
    for (std::size_t n = 0UZ; n < nOutputs; ++n) {
        const std::size_t centre = firstOutput + n;
        float             acc    = 0.f;
        for (std::size_t k = 0UZ; k < taps.size(); ++k) {
            if (centre >= k) {
                acc += taps[k] * signal[centre - k];
            }
        }
        outputs[n] = acc;
    }
    return outputs;
}
} // namespace

const boost::ut::suite<"FastConvolution"> _fastConvolution = [] {
    using namespace boost::ut;

    "the frame is the smallest power of two that still yields output"_test = [] {
        expect(eq(Convolution::frameSizeFor(8UZ, 1UZ), 8UZ));
        expect(eq(Convolution::frameSizeFor(33UZ, 32UZ), 64UZ));
        expect(eq(Convolution::outputsPerFrame(64UZ, 33UZ), 32UZ)) << "a frame yields frameSize + 1 - nTaps useful samples";
    };

    "one frame equals the direct convolution over the samples it keeps"_test = [] {
        for (const std::size_t nTaps : {4UZ, 17UZ, 33UZ}) {
            const std::size_t frameSize = Convolution::frameSizeFor(nTaps, 64UZ);
            const std::size_t nOutputs  = Convolution::outputsPerFrame(frameSize, nTaps);

            std::vector<float> taps(nTaps);
            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                taps[k] = std::sin(0.3f * static_cast<float>(k)) / static_cast<float>(nTaps);
            }
            std::vector<float> frame(frameSize);
            for (std::size_t i = 0UZ; i < frameSize; ++i) {
                frame[i] = std::cos(0.11f * static_cast<float>(i)) + 0.4f * std::sin(0.53f * static_cast<float>(i));
            }

            const std::vector<gr::algorithm::filter::FastConvolution<float>::Complex> tapSpectrum = Convolution::transformTaps(taps, frameSize);
            std::vector<float>                                                        got(nOutputs);
            Convolution::convolveFrame(frame, tapSpectrum, got);

            // the frame's first nTaps-1 outputs are wrap-around and are discarded, so output n is y[n + nTaps - 1]
            const std::vector<float> expected = convolveDirectly(frame, taps, nTaps - 1UZ, nOutputs);
            float                    worst    = 0.f;
            for (std::size_t n = 0UZ; n < nOutputs; ++n) {
                worst = std::max(worst, std::abs(got[n] - expected[n]));
            }
            expect(lt(worst, 1e-4f)) << std::format("{} taps: worst departure from the direct convolution is {}", nTaps, worst);
        }
    };

    "a single unit tap passes the frame through, delayed by nothing"_test = [] {
        constexpr std::size_t    frameSize = 32UZ;
        const std::vector<float> taps{1.f};
        std::vector<float>       frame(frameSize);
        for (std::size_t i = 0UZ; i < frameSize; ++i) {
            frame[i] = static_cast<float>(i + 1UZ);
        }
        const auto         tapSpectrum = Convolution::transformTaps(taps, frameSize);
        std::vector<float> got(Convolution::outputsPerFrame(frameSize, 1UZ));
        Convolution::convolveFrame(frame, tapSpectrum, got);
        bool identity = true;
        for (std::size_t n = 0UZ; n < got.size(); ++n) {
            identity = identity && std::abs(got[n] - frame[n]) < 1e-3f;
        }
        expect(identity) << "one unit tap has nothing to do";
    };
};

int main() { /* tests run from the suite */ }
