#include <boost/ut.hpp>

#ifndef GR_ONNX_MINIMAL_BUILD
#define GR_ONNX_MINIMAL_BUILD 0
#endif

#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <print>
#include <string>
#include <vector>

using namespace boost::ut;
using namespace gr::blocks::onnx;

namespace {

// the bundled peak_detector_* fixtures emit the flat [N + N*R] heatmap+regression layout
constexpr std::size_t kLegacyRegressionChannels = 8UZ;

std::vector<float> makeGaussianSpectrum(std::size_t n, std::span<const std::pair<float, float>> peaks, float noiseFloor = 0.1f) {
    std::vector<float> spectrum(n, noiseFloor);
    for (const auto& [centre, amplitude] : peaks) {
        float sigma = 5.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - centre;
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
            expect(false) << std::format("tracked model failed to load: {}", result.error());
            return;
        }
        expect(session.isLoaded());
        expect(gt(session.modelN(), 0UZ));
        std::println("model N={}, arch='{}'", session.modelN(), session.metadata().architecture);
    };

    "run inference with flat input"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N1024_PATH);
        if (!result) {
            expect(false) << std::format("tracked model failed to load: {}", result.error());
            return;
        }

        const std::size_t n = session.modelN();
        expect(eq(n, 1024UZ));

        std::vector<float> flatInput(n, 0.f);

        auto output = session.run(flatInput);
        expect(output.has_value()) << "inference failed: " << (output.has_value() ? "" : output.error().message);
        if (!output) {
            return;
        }

        std::size_t expectedOut = n + n * kLegacyRegressionChannels;
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
            expect(false) << std::format("tracked model failed to load: {}", loadResult.error());
            return;
        }

        std::pair<float, float> peaks[]  = {{200.f, 50.f}, {500.f, 80.f}, {800.f, 40.f}};
        auto                    spectrum = makeGaussianSpectrum(1024, peaks);

        std::vector<float> normalised(spectrum.size());
        OnnxPreprocess<float>::normaliseLogMAD(spectrum, normalised);

        auto output = session.run(normalised);
        expect(output.has_value());
        if (!output) {
            return;
        }

        std::size_t n = session.modelN();
        expect(eq(output->size(), n + n * kLegacyRegressionChannels));

        std::span<const float> heatmap(output->data(), n);
        float                  maxHeat = *std::max_element(heatmap.begin(), heatmap.end());
        std::println("heatmap max: {:.4f}", maxHeat);
        expect(gt(maxHeat, 0.01f)) << "heatmap should show some response for a peaked spectrum";
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
            expect(false) << std::format("tracked model failed to load: {}", loadResult.error());
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
            expect(false) << std::format("tracked model failed to load: {}", result.error());
            return;
        }
        expect(session.isLoaded());
        session.reset();
        expect(!session.isLoaded());
    };
};

const boost::ut::suite<"OnnxSession execution provider"> executionProviderTests = [] {
    "available providers always include the CPU provider"_test = [] {
        const std::vector<std::string> providers = OnnxSession::availableProviders();
        expect(std::ranges::contains(providers, "CPUExecutionProvider")) << std::format("providers: {}", providers);
        std::println("linked ONNX Runtime provides: {}", providers);
    };

    "explicit cpu provider loads and infers"_test = [] {
        OnnxSession session;
        session.setExecutionProvider("cpu");
        auto result = session.load(MODEL_N1024_PATH);
        if (!result) {
            expect(false) << std::format("tracked model failed to load: {}", result.error());
            return;
        }
        std::vector<float> flatInput(session.modelN(), 0.f);
        expect(session.run(flatInput).has_value()) << "cpu provider must infer exactly as before";
    };

    "requesting an unavailable provider fails naming it and listing the alternatives"_test = [] {
        if (std::ranges::contains(OnnxSession::availableProviders(), "CUDAExecutionProvider")) {
            std::println("skipped: the linked ONNX Runtime offers CUDA, so the unavailable-provider path cannot be exercised here");
            return;
        }
        OnnxSession session;
        session.setExecutionProvider("cuda");
        auto result = session.load(MODEL_N1024_PATH);
        expect(!result.has_value()) << "an unavailable provider must fail the load, never fall back to cpu";
        expect(!session.isLoaded()) << "a failed provider request must not leave a usable session behind";
        if (!result) {
            expect(result.error().message.contains("cuda")) << result.error().message;
            expect(result.error().message.contains("CPUExecutionProvider")) << result.error().message;
        }
    };

    "unknown provider name is rejected before the model bytes are parsed"_test = [] {
        OnnxSession session;
        session.setExecutionProvider("quantum");
        auto result = session.loadFromMemory({0xDE, 0xAD, 0xBE, 0xEF}); // not a model — the provider must be rejected first
        expect(!result.has_value());
        if (!result) {
            expect(result.error().message.contains("quantum")) << result.error().message;
            expect(result.error().message.contains("cpu")) << result.error().message;
        }
    };

    "switching back to cpu after a failed provider request recovers"_test = [] {
        OnnxSession session;
        session.setExecutionProvider("quantum");
        expect(!session.load(MODEL_N1024_PATH).has_value());
        session.setExecutionProvider("cpu");
        auto result = session.load(MODEL_N1024_PATH);
        expect(result.has_value()) << "the provider request must not poison later loads";
        expect(session.isLoaded());
    };
};

#ifdef MODEL_N64_M4_PATH

const boost::ut::suite<"OnnxSession MxN"> mxnSessionTests = [] {
    "load MxN identity model detects history depth"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N64_M4_PATH);
        if (!result) {
            expect(false) << std::format("tracked MxN model failed to load: {}", result.error().message);
            return;
        }
        expect(session.isLoaded());
        expect(eq(session.modelN(), 64UZ));
        expect(eq(session.historyDepth(), 4UZ));
        std::println("MxN model: N={}, M={}", session.modelN(), session.historyDepth());
    };

    "run MxN inference with full batch"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N64_M4_PATH);
        if (!result) {
            expect(false) << std::format("tracked model failed to load: {}", result.error().message);
            return;
        }

        // identity model [1, 4, 64]: input = 4*64 = 256 floats
        std::vector<float> input(4 * 64);
        std::iota(input.begin(), input.end(), 0.f);

        auto output = session.run(input);
        expect(output.has_value()) << "MxN inference should succeed";
        if (!output) {
            return;
        }
        expect(eq(output->size(), 256UZ)) << "identity output should be 4*64";
        for (std::size_t i = 0; i < 256; ++i) {
            expect(lt(std::abs((*output)[i] - input[i]), 1e-5f)) << "sample " << i;
        }
    };

    "run MxN with wrong input size returns error"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_N64_M4_PATH);
        if (!result) {
            expect(false) << std::format("tracked model failed to load: {}", result.error().message);
            return;
        }

        // pass 64 samples instead of 256
        std::vector<float> wrongSize(64, 0.f);
        auto               output = session.run(wrongSize);
        expect(!output.has_value()) << "should reject wrong input size for MxN model";
    };
};

#endif // MODEL_N64_M4_PATH

#ifdef MODEL_HISTORY_N1024_M16_PATH

const boost::ut::suite<"OnnxSession history peak detector"> historyPeakDetectorTests = [] {
    "load history model detects M=16 N=1024"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_HISTORY_N1024_M16_PATH);
        if (!result) {
            expect(false) << std::format("tracked history model failed to load: {}", result.error().message);
            return;
        }
        expect(session.isLoaded());
        expect(eq(session.modelN(), 1024UZ));
        expect(eq(session.historyDepth(), 16UZ));
        std::println("history model: N={}, M={}, arch='{}'", session.modelN(), session.historyDepth(), session.metadata().architecture);
    };

    "history model inference with flat input"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_HISTORY_N1024_M16_PATH);
        if (!result) {
            expect(false) << std::format("tracked model failed to load: {}", result.error().message);
            return;
        }

        // M*N = 16*1024 = 16384 flat input
        std::vector<float> input(16 * 1024, 0.f);
        auto               output = session.run(input);
        expect(output.has_value()) << "inference should succeed";
        if (!output) {
            return;
        }

        std::size_t expectedOut = 1024 + 1024 * kLegacyRegressionChannels;
        expect(eq(output->size(), expectedOut));

        bool allFinite = std::ranges::all_of(*output, [](float v) { return std::isfinite(v); });
        expect(allFinite) << "output contains NaN or inf";
    };

    "history model detects peaks across temporal slices"_test = [] {
        OnnxSession session;
        auto        result = session.load(MODEL_HISTORY_N1024_M16_PATH);
        if (!result) {
            expect(false) << std::format("tracked model failed to load: {}", result.error().message);
            return;
        }

        // build 16 slices with consistent peaks (simulating slowly evolving spectrum)
        constexpr std::size_t M = 16;
        constexpr std::size_t N = 1024;
        std::vector<float>    input(M * N);

        std::pair<float, float> peaks[] = {{200.f, 60.f}, {500.f, 100.f}, {800.f, 45.f}};
        for (std::size_t slice = 0; slice < M; ++slice) {
            auto spectrum = makeGaussianSpectrum(N, peaks);
            // normalise each slice (LogMAD-style)
            std::vector<float> normalised(N);
            OnnxPreprocess<float>::normaliseLogMAD(spectrum, normalised);
            std::copy(normalised.begin(), normalised.end(), input.begin() + static_cast<std::ptrdiff_t>(slice * N));
        }

        auto output = session.run(input);
        expect(output.has_value()) << "inference should succeed";
        if (!output) {
            return;
        }

        std::span<const float> heatmap(output->data(), N);
        float                  maxHeat = *std::max_element(heatmap.begin(), heatmap.end());
        std::println("history model heatmap max: {:.4f}", maxHeat);
        expect(gt(maxHeat, 0.01f)) << "heatmap should show response for peaked spectra";
    };

    "single-slice peak detector vs history model comparison"_test = [] {
        OnnxSession singleSlice;
        auto        r1 = singleSlice.load(MODEL_N1024_PATH);

        OnnxSession history;
        auto        r2 = history.load(MODEL_HISTORY_N1024_M16_PATH);

        if (!r1 || !r2) {
            expect(false) << "tracked comparison models failed to load";
            return;
        }

        // same spectrum with clear peaks
        std::pair<float, float> peaks[]  = {{200.f, 60.f}, {500.f, 100.f}, {800.f, 45.f}};
        auto                    spectrum = makeGaussianSpectrum(1024, peaks);

        std::vector<float> normalised(1024);
        OnnxPreprocess<float>::normaliseLogMAD(spectrum, normalised);

        // single-slice: run directly
        auto out1 = singleSlice.run(normalised);
        expect(out1.has_value());

        // history: replicate normalised slice 16 times
        std::vector<float> historyInput(16 * 1024);
        for (std::size_t s = 0; s < 16; ++s) {
            std::copy(normalised.begin(), normalised.end(), historyInput.begin() + static_cast<std::ptrdiff_t>(s * 1024));
        }
        auto out2 = history.run(historyInput);
        expect(out2.has_value());

        if (out1 && out2) {
            std::span<const float> hm1(out1->data(), 1024);
            std::span<const float> hm2(out2->data(), 1024);
            float                  maxSingle  = *std::max_element(hm1.begin(), hm1.end());
            float                  maxHistory = *std::max_element(hm2.begin(), hm2.end());

            std::println("single-slice: heatmap max={:.4f}", maxSingle);
            std::println("history (M=16): heatmap max={:.4f}", maxHistory);

            // both models should detect something
            expect(gt(maxSingle, 0.01f)) << "single-slice should respond to peaks";
            expect(gt(maxHistory, 0.001f)) << "history model should produce non-zero heatmap";
        }
    };
};

#endif // MODEL_HISTORY_N1024_M16_PATH

int main() { /* boost::ut */ }
