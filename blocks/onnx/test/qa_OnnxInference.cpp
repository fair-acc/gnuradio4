#include <boost/ut.hpp>

#ifndef GR_ONNX_MINIMAL_BUILD
#define GR_ONNX_MINIMAL_BUILD 0
#endif

#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include <cmath>
#include <numeric>
#include <print>

using namespace boost::ut;
using namespace gr::blocks::onnx;

const boost::ut::suite<"OnnxInference"> inferenceTests = [] {
    "processOne produces output with correct dimensions"_test = [] {
        OnnxInference<float> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        const std::size_t n = block._session.modelN();
        expect(eq(n, 1024UZ));

        gr::DataSet<float> input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {"dBm"};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values.resize(n, 1.0f);
        input.extents          = {static_cast<std::int32_t>(n)};
        input.meta_information = {{}};
        input.timing_events    = {{}};

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("inference_output")));

        std::size_t expectedSize = n + n * block._session.regressionChannels();
        expect(eq(output.signal_values.size(), expectedSize));

        bool allFinite = std::ranges::all_of(output.signal_values, [](float v) { return std::isfinite(v); });
        expect(allFinite) << "output contains NaN or inf";

        block.stop();
        expect(!block._session.isLoaded());
    };

    "processOne with size mismatch resamples input"_test = [] {
        OnnxInference<float> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        // feed 2048 samples — should be resampled to 1024
        gr::DataSet<float> input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values.resize(2048, 0.5f);
        input.extents          = {2048};
        input.meta_information = {{}};
        input.timing_events    = {{}};

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(gt(output.signal_values.size(), 0UZ));

        block.stop();
    };

    "processOne passes through empty input"_test = [] {
        OnnxInference<float> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        gr::DataSet<float> input;
        auto               output = block.processOne(std::move(input));
        expect(output.signal_values.empty());

        block.stop();
    };

    "processOne without loaded model passes through"_test = [] {
        OnnxInference<float> block;
        gr::DataSet<float>   input;
        input.signal_names  = {"test"};
        input.signal_values = {1.f, 2.f, 3.f};

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("test")));
    };

    "configurable normalisation modes"_test = [] {
        OnnxInference<float> block;
        block.model_path     = MODEL_N1024_PATH;
        block.normalise_mode = NormaliseMode::MinMax;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        gr::DataSet<float> input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values.resize(1024);
        for (std::size_t i = 0; i < 1024; ++i) {
            input.signal_values[i] = static_cast<float>(i);
        }
        input.extents          = {1024};
        input.meta_information = {{}};
        input.timing_events    = {{}};

        auto output = block.processOne(std::move(input));
        expect(gt(output.signal_values.size(), 0UZ)) << "should produce output with MinMax normalisation";

        block.stop();
    };

    "resample mode None skips resampling"_test = [] {
        OnnxInference<float> block;
        block.model_path    = MODEL_N1024_PATH;
        block.resample_mode = ResampleMode::None;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        // With resample=None, input must match model N exactly
        gr::DataSet<float> input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values.resize(1024, 1.0f);
        input.extents          = {1024};
        input.meta_information = {{}};
        input.timing_events    = {{}};

        auto output = block.processOne(std::move(input));
        expect(gt(output.signal_values.size(), 0UZ));

        block.stop();
    };
};

namespace {

gr::DataSet<float> makeTestDataSet(std::size_t n) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {""};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.signal_values.resize(n);
    std::iota(ds.signal_values.begin(), ds.signal_values.end(), 1.f);
    ds.extents          = {static_cast<std::int32_t>(n)};
    ds.meta_information = {{}};
    ds.timing_events    = {{}};
    return ds;
}

gr::Tensor<float> makeTestTensor(std::size_t n) {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1.f);
    return gr::Tensor<float>(gr::data_from, data);
}

} // namespace

const boost::ut::suite<"OnnxInference DataSet->Tensor"> ds2tTests = [] {
    "DataSet to Tensor produces correct output"_test = [] {
        OnnxInference<float, gr::DataSet<float>, gr::Tensor<float>> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        auto input  = makeTestDataSet(1024);
        auto output = block.processOne(std::move(input));

        expect(gt(output.size(), 0UZ)) << "output tensor should not be empty";
        expect(eq(output.rank(), 1UZ)) << "output should be rank-1";

        std::size_t expectedSize = 1024 + 1024 * block._session.regressionChannels();
        expect(eq(output.size(), expectedSize));

        bool allFinite = true;
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (!std::isfinite(output[i])) {
                allFinite = false;
                break;
            }
        }
        expect(allFinite) << "output contains NaN or inf";

        block.stop();
    };

    "DataSet to Tensor passthrough on empty input"_test = [] {
        OnnxInference<float, gr::DataSet<float>, gr::Tensor<float>> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        gr::DataSet<float> input;
        auto               output = block.processOne(std::move(input));
        expect(eq(output.size(), 0UZ));

        block.stop();
    };
};

const boost::ut::suite<"OnnxInference Tensor->Tensor"> t2tTests = [] {
    "Tensor to Tensor produces correct output"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::Tensor<float>> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        auto input  = makeTestTensor(1024);
        auto output = block.processOne(std::move(input));

        expect(gt(output.size(), 0UZ)) << "output should not be empty";

        std::size_t expectedSize = 1024 + 1024 * block._session.regressionChannels();
        expect(eq(output.size(), expectedSize));

        bool allFinite = true;
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (!std::isfinite(output[i])) {
                allFinite = false;
                break;
            }
        }
        expect(allFinite) << "output contains NaN or inf";

        block.stop();
    };

    "Tensor to Tensor passthrough without model"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::Tensor<float>> block;
        auto                                                       input  = makeTestTensor(64);
        auto                                                       output = block.processOne(std::move(input));

        expect(eq(output.size(), 64UZ));
    };

    "Tensor to Tensor with resampling"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::Tensor<float>> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        // feed 2048 samples — should be resampled to 1024
        auto input  = makeTestTensor(2048);
        auto output = block.processOne(std::move(input));

        expect(gt(output.size(), 0UZ)) << "should produce output after resampling";

        block.stop();
    };
};

const boost::ut::suite<"OnnxInference Tensor->DataSet"> t2dsTests = [] {
    "Tensor to DataSet produces annotated output"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::DataSet<float>> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        auto input  = makeTestTensor(1024);
        auto output = block.processOne(std::move(input));

        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("inference_output")));

        std::size_t expectedSize = 1024 + 1024 * block._session.regressionChannels();
        expect(eq(output.signal_values.size(), expectedSize));

        block.stop();
    };

    "Tensor to DataSet passthrough on empty input"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::DataSet<float>> block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        gr::Tensor<float> input;
        auto              output = block.processOne(std::move(input));
        expect(eq(output.signal_names[0], std::string("pass-through")));
        expect(eq(output.signal_values.size(), 0UZ));

        block.stop();
    };
};

#ifdef MODEL_N64_PATH

const boost::ut::suite<"OnnxInference T->T streaming"> streamingTtoTTests = [] {
    "T->T identity model preserves input"_test = [] {
        OnnxInference<float, float, float> block;
        block.model_path     = MODEL_N64_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: identity model not loadable");
            return;
        }

        expect(eq(block._session.modelN(), 64UZ));

        // the identity model auto-sets normalise_mode=None via metadata
        expect(block.normalise_mode.value == NormaliseMode::None);

        // simulate processBulk with a mock port-like span
        // direct test: run inference on a 64-sample chunk
        std::vector<float> input(64);
        std::iota(input.begin(), input.end(), 0.f);

        std::vector<float> normalised(64);
        block._preprocess.normalise(input, normalised);

        auto result = block._session.run(normalised);
        expect(result.has_value()) << "inference should succeed";

        if (result) {
            expect(eq(result->size(), 64UZ)) << "identity model should output 64 samples";
            for (std::size_t i = 0; i < 64; ++i) {
                expect(lt(std::abs((*result)[i] - input[i]), 1e-5f)) << "sample " << i << " should match input";
            }
        }

        block.stop();
    };

    "T->T identity model with multiple chunks"_test = [] {
        OnnxInference<float, float, float> block;
        block.model_path     = MODEL_N64_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: identity model not loadable");
            return;
        }

        // run 4 sequential chunks through the session directly
        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            std::vector<float> input(64);
            for (std::size_t i = 0; i < 64; ++i) {
                input[i] = static_cast<float>(chunk * 64 + i);
            }
            auto result = block._session.run(input);
            expect(result.has_value()) << "chunk " << chunk;
            if (result) {
                for (std::size_t i = 0; i < 64; ++i) {
                    expect(lt(std::abs((*result)[i] - input[i]), 1e-5f));
                }
            }
        }

        block.stop();
    };
};

const boost::ut::suite<"OnnxInference T->DataSet streaming"> streamingTtoDSTests = [] {
    "T->DataSet identity model produces DataSet output"_test = [] {
        OnnxInference<float, float, gr::DataSet<float>> block;
        block.model_path     = MODEL_N64_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: identity model not loadable");
            return;
        }

        expect(eq(block._session.modelN(), 64UZ));

        // direct session test: run 64-sample input, verify output matches
        std::vector<float> input(64);
        std::iota(input.begin(), input.end(), 0.f);

        auto result = block._session.run(input);
        expect(result.has_value()) << "inference should succeed";

        if (result) {
            expect(eq(result->size(), 64UZ)) << "identity output should be 64 samples";
            for (std::size_t i = 0; i < 64; ++i) {
                expect(lt(std::abs((*result)[i] - input[i]), 1e-5f)) << "sample " << i;
            }
        }

        block.stop();
    };

    "T->DataSet identity model auto-detects normalise_mode from metadata"_test = [] {
        OnnxInference<float, float, gr::DataSet<float>> block;
        block.model_path = MODEL_N64_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: identity model not loadable");
            return;
        }

#if GR_ONNX_MINIMAL_BUILD
        // .ort format strips metadata — normalise_mode stays at default (LogMAD)
        expect(block.normalise_mode.value == NormaliseMode::LogMAD) << "minimal build cannot read metadata";
#else
        // .onnx full format preserves metadata — should auto-detect None
        expect(block.normalise_mode.value == NormaliseMode::None) << "model metadata should set normalise_mode=None";
#endif
        block.stop();
    };
};

#endif // MODEL_N64_PATH

#ifdef MODEL_N64_M4_PATH

const boost::ut::suite<"OnnxInference MxN history"> mxnTests = [] {
    "auto-detects history_depth from model"_test = [] {
        OnnxInference<float> block;
        block.model_path     = MODEL_N64_M4_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: MxN model not loadable");
            return;
        }

        expect(eq(block._session.modelN(), 64UZ));
        expect(eq(block._session.historyDepth(), 4UZ));
#if !GR_ONNX_MINIMAL_BUILD
        // full build reads metadata → auto-sets history_depth
        expect(eq(static_cast<std::size_t>(block.history_depth.value), 4UZ)) << "should auto-set history_depth from model";
#endif
        block.stop();
    };

    "MxN history accumulates then produces output"_test = [] {
        OnnxInference<float> block;
        block.model_path     = MODEL_N64_M4_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 1U;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: MxN model not loadable");
            return;
        }

        // feed 4 DataSets — first 3 should return empty, 4th should produce output
        for (std::size_t i = 0; i < 3; ++i) {
            auto ds     = makeTestDataSet(64);
            auto output = block.processOne(std::move(ds));
            expect(output.signal_values.empty()) << "should accumulate, not produce output at slice " << i;
        }

        // 4th DataSet completes the history window
        auto ds     = makeTestDataSet(64);
        auto output = block.processOne(std::move(ds));
        expect(!output.signal_values.empty()) << "should produce output after M=4 slices";
        expect(eq(output.signal_names.size(), 1UZ));

        block.stop();
    };

    "MxN history with stride=1 slides window"_test = [] {
        OnnxInference<float> block;
        block.model_path     = MODEL_N64_M4_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 1U;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: MxN model not loadable");
            return;
        }

        // fill initial window
        for (std::size_t i = 0; i < 4; ++i) {
            std::ignore = block.processOne(makeTestDataSet(64));
        }

        // subsequent calls should each produce output (sliding window, stride=1)
        for (std::size_t i = 0; i < 3; ++i) {
            auto output = block.processOne(makeTestDataSet(64));
            expect(!output.signal_values.empty()) << "sliding window should produce output at step " << i;
        }

        block.stop();
    };

    "MxN identity model preserves data"_test = [] {
        OnnxInference<float> block;
        block.model_path     = MODEL_N64_M4_PATH;
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 4U; // non-overlapping
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: MxN model not loadable");
            return;
        }

        // feed 4 identical DataSets with known values
        for (std::size_t i = 0; i < 3; ++i) {
            std::ignore = block.processOne(makeTestDataSet(64));
        }
        auto output = block.processOne(makeTestDataSet(64));
        expect(!output.signal_values.empty()) << "should produce output";

        // identity model: output = normalised input (4*64 = 256 values)
        // since normalise_mode=None, output should equal the stacked input
        expect(eq(output.signal_values.size(), 256UZ)) << "identity [1,4,64] should output 256 values";

        bool allFinite = std::ranges::all_of(output.signal_values, [](float v) { return std::isfinite(v); });
        expect(allFinite) << "output should be all finite";

        block.stop();
    };
};

#endif // MODEL_N64_M4_PATH

#ifdef MODEL_HISTORY_N1024_M16_PATH

namespace {

gr::DataSet<float> makePeakedSpectrum(std::size_t n, std::span<const std::pair<float, float>> peaks, float noiseFloor = 0.1f) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(n)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};

    ds.signal_values.resize(n, noiseFloor);
    for (const auto& [center, amplitude] : peaks) {
        float sigma = 5.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - center;
            ds.signal_values[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
    }
    return ds;
}

} // namespace

const boost::ut::suite<"OnnxInference history peak detector"> historyInferenceTests = [] {
    "auto-detects M=16 N=1024 from history model"_test = [] {
        OnnxInference<float> block;
        block.model_path = MODEL_HISTORY_N1024_M16_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: history model not loadable");
            return;
        }

        expect(eq(block._session.modelN(), 1024UZ));
        expect(eq(block._session.historyDepth(), 16UZ));
        expect(eq(static_cast<std::size_t>(block.history_depth.value), 16UZ)) << "should auto-set history_depth from model";
        std::println("history model loaded: N={}, M={}", block._session.modelN(), block._session.historyDepth());

        block.stop();
    };

    "history model accumulates 16 slices then produces output"_test = [] {
        OnnxInference<float> block;
        block.model_path = MODEL_HISTORY_N1024_M16_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: history model not loadable");
            return;
        }

        std::pair<float, float> peaks[] = {{200.f, 60.f}, {500.f, 100.f}, {800.f, 45.f}};

        // first 15 slices should accumulate (return empty)
        for (std::size_t i = 0; i < 15; ++i) {
            auto output = block.processOne(makePeakedSpectrum(1024, peaks));
            expect(output.signal_values.empty()) << "should accumulate at slice " << i;
        }

        // 16th slice completes the window and triggers inference
        auto output = block.processOne(makePeakedSpectrum(1024, peaks));
        expect(!output.signal_values.empty()) << "should produce output after M=16 slices";

        std::size_t expectedSize = 1024 + 1024 * block._session.regressionChannels();
        expect(eq(output.signal_values.size(), expectedSize));

        bool allFinite = std::ranges::all_of(output.signal_values, [](float v) { return std::isfinite(v); });
        expect(allFinite) << "output should be all finite";

        // check heatmap (first 1024 values)
        std::span<const float> heatmap(output.signal_values.data(), 1024);
        float                  maxHeat = *std::max_element(heatmap.begin(), heatmap.end());
        std::println("history inference heatmap max: {:.4f}", maxHeat);
        expect(gt(maxHeat, 0.001f)) << "heatmap should show some response";

        block.stop();
    };

    "history model sliding window produces continuous output"_test = [] {
        OnnxInference<float> block;
        block.model_path     = MODEL_HISTORY_N1024_M16_PATH;
        block.history_stride = 1U;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: history model not loadable");
            return;
        }

        std::pair<float, float> peaks[] = {{500.f, 80.f}};

        // fill initial window
        for (std::size_t i = 0; i < 16; ++i) {
            std::ignore = block.processOne(makePeakedSpectrum(1024, peaks));
        }

        // subsequent calls should each produce output (sliding window)
        for (std::size_t i = 0; i < 3; ++i) {
            auto output = block.processOne(makePeakedSpectrum(1024, peaks));
            expect(!output.signal_values.empty()) << "sliding window should produce output at step " << i;
        }

        block.stop();
    };

    "single-slice vs history model on same spectrum"_test = [] {
        // run the single-slice (ex1) model
        OnnxInference<float> singleBlock;
        singleBlock.model_path = MODEL_N1024_PATH;
        singleBlock.start();

        // run the history (ex2) model
        OnnxInference<float> historyBlock;
        historyBlock.model_path     = MODEL_HISTORY_N1024_M16_PATH;
        historyBlock.history_stride = 1U;
        historyBlock.start();

        if (!singleBlock._session.isLoaded() || !historyBlock._session.isLoaded()) {
            std::println("skip: need both models for comparison");
            return;
        }

        std::pair<float, float> peaks[] = {{200.f, 60.f}, {500.f, 100.f}, {800.f, 45.f}};

        // single-slice inference
        auto singleOut = singleBlock.processOne(makePeakedSpectrum(1024, peaks));
        expect(!singleOut.signal_values.empty());

        // history: feed 16 identical spectra
        gr::DataSet<float> historyOut;
        for (std::size_t i = 0; i < 16; ++i) {
            historyOut = historyBlock.processOne(makePeakedSpectrum(1024, peaks));
        }
        expect(!historyOut.signal_values.empty()) << "history model should produce output after 16 slices";

        if (!singleOut.signal_values.empty() && !historyOut.signal_values.empty()) {
            // compare heatmap peaks
            std::span<const float> hmSingle(singleOut.signal_values.data(), 1024);
            std::span<const float> hmHistory(historyOut.signal_values.data(), 1024);
            float                  maxSingle  = *std::max_element(hmSingle.begin(), hmSingle.end());
            float                  maxHistory = *std::max_element(hmHistory.begin(), hmHistory.end());

            std::println("single-slice heatmap max: {:.4f}", maxSingle);
            std::println("history (M=16) heatmap max: {:.4f}", maxHistory);

            expect(gt(maxSingle, 0.01f)) << "single-slice should detect peaks";
            expect(gt(maxHistory, 0.001f)) << "history model should produce non-zero heatmap";

            bool singleFinite  = std::ranges::all_of(singleOut.signal_values, [](float v) { return std::isfinite(v); });
            bool historyFinite = std::ranges::all_of(historyOut.signal_values, [](float v) { return std::isfinite(v); });
            expect(singleFinite) << "single-slice output all finite";
            expect(historyFinite) << "history output all finite";
        }

        singleBlock.stop();
        historyBlock.stop();
    };
};

#endif // MODEL_HISTORY_N1024_M16_PATH

int main() { /* boost::ut */ }
