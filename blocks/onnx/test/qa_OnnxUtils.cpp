#include <boost/ut.hpp>

#include <gnuradio-4.0/onnx/OnnxUtils.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <print>
#include <vector>

using namespace boost::ut;
using namespace gr::blocks::onnx;

namespace {

std::vector<float> makeGaussianSpectrum(std::size_t n, std::span<const std::pair<float, float>> peaks, float noiseFloor = 0.1f) {
    std::vector<float> spectrum(n, noiseFloor);
    for (const auto& [center, amplitude] : peaks) {
        float sigma = 5.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - center;
            spectrum[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
    }
    return spectrum;
}

} // namespace

const boost::ut::suite<"OnnxUtils normalise"> normaliseTests = [] {
    "normalise produces finite output"_test = [] {
        std::vector<float> input(64);
        std::iota(input.begin(), input.end(), 1.0f);

        std::vector<float> output(64);
        normalise(input, output);

        for (std::size_t i = 0; i < output.size(); ++i) {
            expect(std::isfinite(output[i])) << "index" << i;
            expect(ge(output[i], -5.0f));
            expect(le(output[i], 10.0f));
        }
    };

    "normalise handles constant input"_test = [] {
        std::vector<float> input(128, 42.0f);
        std::vector<float> output(128);
        normalise(input, output);

        for (const auto v : output) {
            expect(std::isfinite(v));
            expect(ge(v, -5.0f));
            expect(le(v, 10.0f));
        }
    };

    "normalise handles negative values"_test = [] {
        std::vector<float> input = {-10.f, -5.f, 0.f, 5.f, 10.f, 20.f, 50.f, 100.f};
        std::vector<float> output(input.size());
        normalise(input, output);

        for (std::size_t i = 0; i < output.size(); ++i) {
            expect(std::isfinite(output[i])) << "index" << i;
        }
    };

    "normalise preserves monotonicity for sorted input"_test = [] {
        std::vector<float> input(256);
        std::iota(input.begin(), input.end(), 1.0f);

        std::vector<float> output(256);
        normalise(input, output);

        // log-MAD normalisation of a monotonically increasing sequence should stay monotonic
        for (std::size_t i = 1; i < output.size(); ++i) {
            expect(ge(output[i], output[i - 1])) << "monotonicity broken at index" << i;
        }
    };

    "normalise with synthetic spectrum produces expected structure"_test = [] {
        std::pair<float, float> peaks[]  = {{200.f, 50.f}, {500.f, 80.f}, {800.f, 40.f}};
        auto                    spectrum = makeGaussianSpectrum(1024, peaks);

        std::vector<float> output(1024);
        normalise(spectrum, output);

        // peak positions should have higher normalised values than noise floor
        float noiseMean = 0.f;
        for (std::size_t i = 0; i < 100; ++i) {
            noiseMean += output[i]; // bins 0..99 are noise-only
        }
        noiseMean /= 100.f;

        expect(gt(output[200], noiseMean + 1.f)) << "peak at 200 should stand above noise";
        expect(gt(output[500], noiseMean + 1.f)) << "peak at 500 should stand above noise";
        expect(gt(output[800], noiseMean + 1.f)) << "peak at 800 should stand above noise";

        // all values should be clipped to [-5, 10]
        for (std::size_t i = 0; i < output.size(); ++i) {
            expect(ge(output[i], -5.0f)) << "below clip at index" << i;
            expect(le(output[i], 10.0f)) << "above clip at index" << i;
        }
    };
};

const boost::ut::suite<"OnnxUtils resample"> resampleTests = [] {
    "identity resample"_test = [] {
        std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f};
        std::vector<float> output(5);
        resample(input, output);
        for (std::size_t i = 0; i < input.size(); ++i) {
            expect(lt(std::abs(output[i] - input[i]), 1e-6f));
        }
    };

    "upsample 2x"_test = [] {
        std::vector<float> input = {0.f, 10.f};
        std::vector<float> output(5);
        resample(input, output);
        // expected: 0, 2.5, 5, 7.5, 10
        expect(lt(std::abs(output[0] - 0.f), 1e-6f));
        expect(lt(std::abs(output[2] - 5.f), 1e-6f));
        expect(lt(std::abs(output[4] - 10.f), 1e-6f));
    };

    "downsample"_test = [] {
        std::vector<float> input = {0.f, 1.f, 2.f, 3.f, 4.f};
        std::vector<float> output(3);
        resample(input, output);
        // endpoints preserved
        expect(lt(std::abs(output[0] - 0.f), 1e-6f));
        expect(lt(std::abs(output[2] - 4.f), 1e-6f));
        // middle should be 2.0
        expect(lt(std::abs(output[1] - 2.f), 1e-6f));
    };
};

const boost::ut::suite<"OnnxUtils extractPeaks"> extractPeaksTests = [] {
    "finds known peaks in hand-crafted heatmap"_test = [] {
        constexpr std::size_t n = 100;
        constexpr std::size_t r = 8;

        std::vector<float> heatmap(n, 0.f);
        std::vector<float> regression(n * r, 0.f);

        // place 3 peaks at positions 20, 50, 80
        auto placePeak = [&](std::size_t idx, float conf, float offset, float sigma) {
            heatmap[idx]            = conf;
            regression[idx * r + 0] = offset / static_cast<float>(n);       // normalised offset
            regression[idx * r + 1] = 1.0f;                                 // amplitude
            regression[idx * r + 2] = sigma / static_cast<float>(n);        // normalised sigma
            regression[idx * r + 3] = 2.0f * sigma / static_cast<float>(n); // w68
            regression[idx * r + 4] = 4.0f * sigma / static_cast<float>(n); // w96
            regression[idx * r + 5] = 6.0f * sigma / static_cast<float>(n); // w99
            regression[idx * r + 6] = 3.0f / 10.0f;                         // kurtosis / 10
            regression[idx * r + 7] = 0.5f;                                 // (excess_kurtosis + 5) / 10
        };

        placePeak(20, 0.9f, 0.3f, 5.0f);
        placePeak(50, 0.7f, -0.1f, 3.0f);
        placePeak(80, 0.5f, 0.0f, 8.0f);

        auto peaks = extractPeaks(heatmap, regression, r, 0.3f, 5, 10);
        expect(eq(peaks.size(), 3UZ));

        // sorted by confidence descending
        expect(gt(peaks[0].confidence, peaks[1].confidence));
        expect(gt(peaks[1].confidence, peaks[2].confidence));

        // check positions are approximately correct
        expect(lt(std::abs(peaks[0].position - 20.3f), 1.0f));
        expect(lt(std::abs(peaks[1].position - 49.9f), 1.0f));
        expect(lt(std::abs(peaks[2].position - 80.0f), 1.0f));
    };

    "respects maxPeaks limit"_test = [] {
        constexpr std::size_t n = 50;
        constexpr std::size_t r = 8;

        std::vector<float> heatmap(n, 0.f);
        std::vector<float> regression(n * r, 0.f);

        heatmap[10] = 0.9f;
        heatmap[20] = 0.8f;
        heatmap[30] = 0.7f;
        heatmap[40] = 0.6f;

        auto peaks = extractPeaks(heatmap, regression, r, 0.3f, 3, 2);
        expect(eq(peaks.size(), 2UZ));
        expect(gt(peaks[0].confidence, peaks[1].confidence));
    };

    "NMS suppresses nearby peaks"_test = [] {
        constexpr std::size_t n = 50;
        constexpr std::size_t r = 8;

        std::vector<float> heatmap(n, 0.f);
        std::vector<float> regression(n * r, 0.f);

        heatmap[20] = 0.9f;
        heatmap[22] = 0.5f; // within minDistance=5 of peak at 20

        auto peaks = extractPeaks(heatmap, regression, r, 0.3f, 5, 10);
        expect(eq(peaks.size(), 1UZ));
        expect(lt(std::abs(peaks[0].confidence - 0.9f), 1e-6f));
    };

    "returns empty for flat heatmap"_test = [] {
        std::vector<float> heatmap(100, 0.1f);
        std::vector<float> regression(100 * 8, 0.f);
        auto               peaks = extractPeaks(heatmap, regression, 8, 0.3f, 5, 10);
        expect(eq(peaks.size(), 0UZ));
    };
};

int main() { /* boost::ut */ }
