#include <boost/ut.hpp>

#ifndef GR_ONNX_MINIMAL_BUILD
#define GR_ONNX_MINIMAL_BUILD 0
#endif

#include <gnuradio-4.0/onnx/OnnxSession.hpp>
#include <gnuradio-4.0/onnx/OnnxUtils.hpp>

#include <cmath>
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

const boost::ut::suite<"OnnxSession"> sessionTests = [] {
    "load bundled model"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N1024_PATH);
        if (!result) {
            std::println("skip: model not loadable: {}", result.error());
            return;
        }
        expect(session.isLoaded());
        expect(gt(session.modelN(), 0UZ));
        std::println("model N={}, regression channels={}, arch='{}'", session.modelN(), session.regressionChannels(), session.metadata().architecture);
    };

    "run inference with flat input"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N1024_PATH);
        if (!result) {
            std::println("skip: {}", result.error());
            return;
        }

        const std::size_t n = session.modelN();
        expect(eq(n, 1024UZ));

        std::vector<float> flatInput(n, 0.f);

        auto output = session.run(flatInput);
        expect(output.has_value()) << "inference failed: " << (output.has_value() ? "" : output.error());
        if (!output) {
            return;
        }

        std::size_t expectedOut = n + n * session.regressionChannels();
        std::println("output size: {} (expected {})", output->size(), expectedOut);
        expect(eq(output->size(), expectedOut));

        bool allFinite = true;
        for (float v : *output) {
            if (!std::isfinite(v)) {
                allFinite = false;
                break;
            }
        }
        expect(allFinite) << "output contains NaN or inf";
    };

    "run inference with synthetic spectrum"_test = [] {
        OnnxSession session;
        auto        loadResult = session.load(MODEL_N1024_PATH);
        if (!loadResult) {
            std::println("skip: {}", loadResult.error());
            return;
        }

        std::pair<float, float> peaks[]  = {{200.f, 50.f}, {500.f, 80.f}, {800.f, 40.f}};
        auto                    spectrum = makeGaussianSpectrum(1024, peaks);

        std::vector<float> normalised(spectrum.size());
        normalise(spectrum, normalised);

        auto output = session.run(normalised);
        expect(output.has_value());
        if (!output) {
            return;
        }

        std::size_t n = session.modelN();
        std::size_t r = session.regressionChannels();
        expect(eq(output->size(), n + n * r));

        std::span<const float> heatmap(output->data(), n);
        float                  maxHeat = *std::max_element(heatmap.begin(), heatmap.end());
        std::println("heatmap max: {:.4f}", maxHeat);
        expect(gt(maxHeat, 0.01f)) << "heatmap should show some response for a peaked spectrum";

        auto detectedPeaks = extractPeaks(heatmap, std::span<const float>(output->data() + n, n * r), r, 0.3f, 8, 8);
        std::println("detected {} peaks", detectedPeaks.size());
        for (const auto& p : detectedPeaks) {
            std::println("  pos={:.1f} conf={:.3f} sigma={:.1f}", p.position, p.confidence, p.sigma);
        }
    };

    "load with bad path returns error"_test = [] {
        OnnxSession session;
        auto        result = session.load("/nonexistent/model.onnx");
        expect(!result.has_value());
        std::println("expected error: {}", result.error());
    };

    "run without loading returns error"_test = [] {
        OnnxSession        session;
        std::vector<float> dummy(10, 0.f);
        auto               result = session.run(dummy);
        expect(!result.has_value());
    };

    "wrong input size returns error"_test = [] {
        OnnxSession session;
        auto        loadResult = session.load(MODEL_N1024_PATH);
        if (!loadResult) {
            std::println("skip: {}", loadResult.error());
            return;
        }

        std::vector<float> wrongSize(42, 0.f);
        auto               result = session.run(wrongSize);
        expect(!result.has_value());
        std::println("expected error: {}", result.error());
    };

    "reset releases session"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N1024_PATH);
        if (!result) {
            std::println("skip: {}", result.error());
            return;
        }
        expect(session.isLoaded());
        session.reset();
        expect(!session.isLoaded());
    };
};

int main() { /* boost::ut */ }
