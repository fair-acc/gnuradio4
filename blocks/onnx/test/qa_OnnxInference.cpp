#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include "../ModelPath.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <print>
#include <string>

using namespace boost::ut;
using namespace gr::blocks::onnx;
using gr::blocks::onnx::test::modelPath;

// the bundled peak_detector_* fixtures emit the flat [N + N*R] heatmap+regression layout

namespace mock {

// minimal stand-ins for the scheduler-provided spans, for direct processBulk invocation
struct MetaSpan {
    std::vector<gr::Tensor<float>> slots;
    std::size_t                    published = 0UZ;

    [[nodiscard]] std::size_t size() const { return slots.size(); }
    gr::Tensor<float>&        operator[](std::size_t i) { return slots[i]; }
    void                      publish(std::size_t n) { published += n; }
};

template<typename TVal>
struct IoSpan {
    std::span<TVal> data;
    std::size_t     consumed  = 0UZ;
    std::size_t     published = 0UZ;

    [[nodiscard]] std::size_t size() const { return data.size(); }
    TVal&                     operator[](std::size_t i) { return data[i]; }
    bool                      consume(std::size_t n) {
        consumed += n;
        return true;
    }
    void publish(std::size_t n) { published += n; }
};

using CfgSpan = IoSpan<float>;

} // namespace mock

bool isMarkedPassthrough(const gr::DataSet<float>& ds) {
    if (ds.meta_information.empty()) {
        return false;
    }
    const auto it = ds.meta_information[0].find(std::pmr::string(kPassthroughKey));
    if (it == ds.meta_information[0].end()) {
        return false;
    }
    const bool* flag = it->second.get_if<bool>();
    return flag != nullptr && *flag;
}

// single-sample processBulk invocation matching the previous processOne semantics
template<typename TOutVal, typename TBlock, typename TInVal>
TOutVal processSingle(TBlock& block, TInVal input) {
    std::array<TInVal, 1>     inArr{std::move(input)};
    std::array<TOutVal, 1>    outArr{};
    mock::IoSpan<TInVal>      inSpan{.data = inArr};
    mock::IoSpan<TOutVal>     outSpan{.data = outArr};
    std::span<mock::CfgSpan>  cfgSpan{};
    std::span<mock::MetaSpan> metaSpan{};
    std::ignore = block.processBulk(inSpan, cfgSpan, outSpan, metaSpan);
    return std::move(outArr[0]);
}

const boost::ut::suite<"OnnxInference"> inferenceTests = [] {
    "a double-valued block runs the same float32 graph"_test = [] {
        // ORT tensors are float32; the block narrows on the way in and widens on the way out, so a
        // double graph is a convenience for the caller's buffers, not extra precision
        OnnxInference<double> block;
        block.model_path = modelPath("identity_N64_M4.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        const std::size_t   n = block._session.modelN();
        gr::DataSet<double> input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<double>{0., 0.}};
        input.signal_values.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            input.signal_values[i] = 0.25 * static_cast<double>(i);
        }
        input.extents = {static_cast<std::int32_t>(n)};

        const auto output = processSingle<gr::DataSet<double>>(block, input);
        expect(eq(output.signal_values.size(), n)) << "identity model must return one value per input";
        double worst = 0.;
        for (std::size_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(output.signal_values[i] - input.signal_values[i]));
        }
        expect(lt(worst, 1e-5)) << "identity through a float32 graph must round-trip within float precision";
        block.stop();
    };

    "processOne produces output with correct dimensions"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
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

        auto output = processSingle<gr::DataSet<float>>(block, std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("inference_output")));

        std::size_t expectedSize = n;
        expect(eq(output.signal_values.size(), expectedSize));

        bool allFinite = std::ranges::all_of(output.signal_values, [](float v) { return std::isfinite(v); });
        expect(allFinite) << "output contains NaN or inf";
        expect(!isMarkedPassthrough(output)) << "real inference output must not carry the onnx_passthrough marker";

        block.stop();
        expect(!block.isModelLoaded());
    };

    "processOne with size mismatch resamples input"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
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

        auto output = processSingle<gr::DataSet<float>>(block, std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(gt(output.signal_values.size(), 0UZ));

        block.stop();
    };

    "processOne passes through empty input"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        gr::DataSet<float> input;
        auto               output = processSingle<gr::DataSet<float>>(block, std::move(input));
        expect(output.signal_values.empty());

        block.stop();
    };

    "processOne without loaded model passes through"_test = [] {
        OnnxInference<float> block;
        gr::DataSet<float>   input;
        input.signal_names  = {"test"};
        input.signal_values = {1.f, 2.f, 3.f};

        auto output = processSingle<gr::DataSet<float>>(block, std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("test")));
        expect(isMarkedPassthrough(output)) << "forwarded frames must carry the onnx_passthrough marker";
    };

    "configurable normalisation modes"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("peaks_fixture_N1024.ort.gz");
        block.normalise_mode = NormaliseMode::MinMax;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
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

        auto output = processSingle<gr::DataSet<float>>(block, std::move(input));
        expect(gt(output.signal_values.size(), 0UZ)) << "should produce output with MinMax normalisation";

        block.stop();
    };

    "resample mode None skips resampling"_test = [] {
        OnnxInference<float> block;
        block.model_path    = modelPath("peaks_fixture_N1024.ort.gz");
        block.resample_mode = ResampleMode::None;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
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

        auto output = processSingle<gr::DataSet<float>>(block, std::move(input));
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

struct FixtureSpectrumSource : gr::Block<FixtureSpectrumSource> {
    using Description = gr::Doc<"Test source emitting a preset DataSet n_samples times.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_samples = 1U;

    GR_MAKE_REFLECTABLE(FixtureSpectrumSource, out, n_samples);

    gr::DataSet<float> dataset;
    gr::Size_t         _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_samples - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = dataset;
        }
        _emitted += static_cast<gr::Size_t>(n);
        outSpan.publish(n);
        return _emitted >= n_samples ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

template<typename T>
struct CollectSink : gr::Block<CollectSink<T>> {
    using Description = gr::Doc<"Test sink collecting every received sample.">;

    gr::PortIn<T> in;

    GR_MAKE_REFLECTABLE(CollectSink, in);

    std::vector<T> received;

    void processOne(T value) { received.push_back(std::move(value)); }
};

struct ScalarConfigSource : gr::Block<ScalarConfigSource> {
    using Description = gr::Doc<"Test source emitting a constant float n_samples times.">;

    gr::PortOut<float> out;

    float      value     = 1.f;
    gr::Size_t n_samples = 1U;

    GR_MAKE_REFLECTABLE(ScalarConfigSource, out, value, n_samples);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_samples - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = value;
        }
        _emitted += static_cast<gr::Size_t>(n);
        outSpan.publish(n);
        return _emitted >= n_samples ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

} // namespace

const boost::ut::suite<"OnnxInference DataSet->Tensor"> ds2tTests = [] {
    "DataSet to Tensor produces correct output"_test = [] {
        OnnxInference<float, gr::DataSet<float>, gr::Tensor<float>> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        auto input  = makeTestDataSet(1024);
        auto output = processSingle<gr::Tensor<float>>(block, std::move(input));

        expect(gt(output.size(), 0UZ)) << "output tensor should not be empty";
        expect(eq(output.rank(), 1UZ)) << "output should be rank-1";

        std::size_t expectedSize = 1024;
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
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        gr::DataSet<float> input;
        auto               output = processSingle<gr::Tensor<float>>(block, std::move(input));
        expect(eq(output.size(), 0UZ));

        block.stop();
    };
};

const boost::ut::suite<"OnnxInference Tensor->Tensor"> t2tTests = [] {
    "Tensor to Tensor produces correct output"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::Tensor<float>> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        auto input  = makeTestTensor(1024);
        auto output = processSingle<gr::Tensor<float>>(block, std::move(input));

        expect(gt(output.size(), 0UZ)) << "output should not be empty";

        std::size_t expectedSize = 1024;
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
        auto                                                       output = processSingle<gr::Tensor<float>>(block, std::move(input));

        expect(eq(output.size(), 64UZ));
    };

    "Tensor to Tensor with resampling"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::Tensor<float>> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        // feed 2048 samples — should be resampled to 1024
        auto input  = makeTestTensor(2048);
        auto output = processSingle<gr::Tensor<float>>(block, std::move(input));

        expect(gt(output.size(), 0UZ)) << "should produce output after resampling";

        block.stop();
    };
};

const boost::ut::suite<"OnnxInference Tensor->DataSet"> t2dsTests = [] {
    "Tensor to DataSet produces annotated output"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::DataSet<float>> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        auto input  = makeTestTensor(1024);
        auto output = processSingle<gr::DataSet<float>>(block, std::move(input));

        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("inference_output")));

        std::size_t expectedSize = 1024;
        expect(eq(output.signal_values.size(), expectedSize));

        block.stop();
    };

    "Tensor to DataSet passthrough on empty input"_test = [] {
        OnnxInference<float, gr::Tensor<float>, gr::DataSet<float>> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        gr::Tensor<float> input;
        auto              output = processSingle<gr::DataSet<float>>(block, std::move(input));
        expect(eq(output.signal_names[0], std::string("pass-through")));
        expect(eq(output.signal_values.size(), 0UZ));

        block.stop();
    };
};

const boost::ut::suite<"OnnxInference MxN history"> mxnTests = [] {
    "auto-detects history_depth from model"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("identity_N64_M4.ort.gz");
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        expect(eq(block._session.modelN(), 64UZ));
        expect(eq(block._session.historyDepth(), 4UZ));
        expect(eq(static_cast<std::size_t>(block.history_depth.value), 4UZ)) << "should auto-set history_depth from model";
        block.stop();
    };

    "MxN history accumulates then produces output"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("identity_N64_M4.ort.gz");
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 1U;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        // feed 4 DataSets — first 3 pass through while the window fills, 4th produces inference output
        for (std::size_t i = 0; i < 3; ++i) {
            auto ds     = makeTestDataSet(64);
            auto output = processSingle<gr::DataSet<float>>(block, std::move(ds));
            expect(eq(output.signal_values.size(), 64UZ)) << "warm-up input passes through at slice " << i;
            expect(eq(output.signal_names[0], std::string("Spectrum"))) << "warm-up output is the unmodified input";
            expect(isMarkedPassthrough(output)) << "warm-up frame " << i << " must carry the onnx_passthrough marker";
        }

        // 4th DataSet completes the history window
        auto ds     = makeTestDataSet(64);
        auto output = processSingle<gr::DataSet<float>>(block, std::move(ds));
        expect(!output.signal_values.empty()) << "should produce output after M=4 slices";
        expect(eq(output.signal_names.size(), 1UZ));
        expect(!isMarkedPassthrough(output)) << "real inference output must not carry the onnx_passthrough marker";

        block.stop();
    };

    "MxN history with stride=1 slides window"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("identity_N64_M4.ort.gz");
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 1U;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        // fill initial window
        for (std::size_t i = 0; i < 4; ++i) {
            std::ignore = processSingle<gr::DataSet<float>>(block, makeTestDataSet(64));
        }

        // subsequent calls should each produce output (sliding window, stride=1)
        for (std::size_t i = 0; i < 3; ++i) {
            auto output = processSingle<gr::DataSet<float>>(block, makeTestDataSet(64));
            expect(!output.signal_values.empty()) << "sliding window should produce output at step " << i;
        }

        block.stop();
    };

    "MxN decimating stride=4 consumes 8 inputs and produces 2 outputs"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("identity_N64_M4.ort.gz");
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 4U; // non-overlapping windows: genuine 4:1 decimation
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        std::vector<gr::DataSet<float>>  inputs(8, makeTestDataSet(64));
        std::vector<gr::DataSet<float>>  outputs(8);
        mock::IoSpan<gr::DataSet<float>> inSpan{.data = inputs};
        mock::IoSpan<gr::DataSet<float>> outSpan{.data = outputs};
        std::span<mock::CfgSpan>         cfgSpan{};
        std::span<mock::MetaSpan>        metaSpan{};

        expect(block.processBulk(inSpan, cfgSpan, outSpan, metaSpan) == gr::work::Status::OK);
        expect(eq(inSpan.consumed, 8UZ)) << "all offered chunks must be consumed";
        expect(eq(outSpan.published, 2UZ)) << "genuine 4:1 decimation: 8 inputs -> 2 outputs";

        // identity model, normalise=None: output = the stacked [4x64] window verbatim
        expect(eq(outputs[0].signal_values.size(), 256UZ)) << "identity [1,4,64] should output 256 values";
        for (std::size_t row = 0; row < 4; ++row) {
            expect(eq(outputs[0].signal_values[row * 64], 1.f)) << "each stacked row starts with the iota ramp";
        }

        block.stop();
    };

    "MxN decimating overlapping windows (stride < depth)"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("identity_N64_M4.ort.gz");
        block.normalise_mode = NormaliseMode::None;
        block.resample_mode  = ResampleMode::None;
        block.history_depth  = 4U;
        block.history_stride = 2U; // 2:1 decimation, windows overlap by 2 frames
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        std::vector<gr::DataSet<float>>  inputs(8, makeTestDataSet(64));
        std::vector<gr::DataSet<float>>  outputs(8);
        mock::IoSpan<gr::DataSet<float>> inSpan{.data = inputs};
        mock::IoSpan<gr::DataSet<float>> outSpan{.data = outputs};
        std::span<mock::CfgSpan>         cfgSpan{};
        std::span<mock::MetaSpan>        metaSpan{};

        expect(block.processBulk(inSpan, cfgSpan, outSpan, metaSpan) == gr::work::Status::OK);
        expect(eq(inSpan.consumed, 8UZ));
        expect(eq(outSpan.published, 3UZ)) << "windows complete at inputs 4, 6, 8 -> 3 outputs";

        block.stop();
    };

    "decimating history mode reduces the output count through the scheduler"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<FixtureSpectrumSource>({{"n_samples", 8U}});
        source.dataset   = makeTestDataSet(64);
        auto& inference  = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("identity_N64_M4.ort.gz")}, {"history_depth", 4U}, {"history_stride", 4U}, {"normalise_mode", "None"}, {"resample_mode", "None"}});
        auto& sink       = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();

        if (!inference.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("identity_N64_M4.ort.gz");
            return;
        }

        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.received.size(), 2UZ)) << "genuine output-count reduction: 8 inputs -> 2 outputs";
        for (const auto& output : sink.received) {
            expect(eq(output.signal_values.size(), 256UZ)) << "identity [1,4,64] emits the stacked window";
        }
    };
};

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
    for (const auto& [centre, amplitude] : peaks) {
        float sigma = 5.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - centre;
            ds.signal_values[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
    }
    return ds;
}

} // namespace

const boost::ut::suite<"OnnxInference history peak detector"> historyInferenceTests = [] {
    "auto-detects M=16 N=1024 from history model"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024_M16.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024_M16.ort.gz");
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
        block.model_path = modelPath("peaks_fixture_N1024_M16.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024_M16.ort.gz");
            return;
        }

        std::pair<float, float> peaks[] = {{200.f, 60.f}, {500.f, 100.f}, {800.f, 45.f}};

        // first 15 slices pass through while the window fills (module-wide warm-up contract)
        for (std::size_t i = 0; i < 15; ++i) {
            auto output = processSingle<gr::DataSet<float>>(block, makePeakedSpectrum(1024, peaks));
            expect(eq(output.signal_values.size(), 1024UZ)) << "warm-up input passes through at slice " << i;
            expect(eq(output.signal_names[0], std::string("Spectrum"))) << "warm-up output is the unmodified input";
        }

        // 16th slice completes the window and triggers inference
        auto output = processSingle<gr::DataSet<float>>(block, makePeakedSpectrum(1024, peaks));
        expect(!output.signal_values.empty()) << "should produce output after M=16 slices";

        std::size_t expectedSize = 1024;
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
        block.model_path     = modelPath("peaks_fixture_N1024_M16.ort.gz");
        block.history_stride = 1U;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked model failed to load: " << modelPath("peaks_fixture_N1024_M16.ort.gz");
            return;
        }

        std::pair<float, float> peaks[] = {{500.f, 80.f}};

        // fill initial window
        for (std::size_t i = 0; i < 16; ++i) {
            std::ignore = processSingle<gr::DataSet<float>>(block, makePeakedSpectrum(1024, peaks));
        }

        // subsequent calls should each produce output (sliding window)
        for (std::size_t i = 0; i < 3; ++i) {
            auto output = processSingle<gr::DataSet<float>>(block, makePeakedSpectrum(1024, peaks));
            expect(!output.signal_values.empty()) << "sliding window should produce output at step " << i;
        }

        block.stop();
    };

    "single-slice vs history model on same spectrum"_test = [] {
        // run the single-slice (ex1) model
        OnnxInference<float> singleBlock;
        singleBlock.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        singleBlock.start();

        // run the history (ex2) model
        OnnxInference<float> historyBlock;
        historyBlock.model_path     = modelPath("peaks_fixture_N1024_M16.ort.gz");
        historyBlock.history_stride = 1U;
        historyBlock.start();

        if (!singleBlock.isModelLoaded() || !historyBlock.isModelLoaded()) {
            expect(false) << "tracked comparison models failed to load";
            return;
        }

        std::pair<float, float> peaks[] = {{200.f, 60.f}, {500.f, 100.f}, {800.f, 45.f}};

        // single-slice inference
        auto singleOut = processSingle<gr::DataSet<float>>(singleBlock, makePeakedSpectrum(1024, peaks));
        expect(!singleOut.signal_values.empty());

        // history: feed 16 identical spectra
        gr::DataSet<float> historyOut;
        for (std::size_t i = 0; i < 16; ++i) {
            historyOut = processSingle<gr::DataSet<float>>(historyBlock, makePeakedSpectrum(1024, peaks));
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

namespace {

// peak bin of the fixture heatmap for a ramp input (the only strict local maximum)
[[nodiscard]] float heatmapPeakValue(const gr::DataSet<float>& output) { return output.signal_values.empty() ? 0.f : output.signal_values.back(); }

} // namespace

const boost::ut::suite<"OnnxInference multi-output + overrides"> multiOutputTests = [] {
    "meta ports are sized from the model"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        expect(eq(block.meta_out.size(), 4UZ)) << "outputs 1..n-1 of the 5-output fixture";
        expect(eq(block.meta_output_names.value.size(), 4UZ));
        expect(eq(block.meta_output_names.value[0], std::string("peaks")));
        expect(eq(block.meta_output_names.value[1], std::string("peak_rescore")));

        block.stop();
    };

    "model_overrides replaces a model-baked initializer"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        const float baseline = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));
        expect(gt(baseline, 0.5f)) << "fixture heatmap peaks at ~1 with the baked score_scale=1";

        block.model_overrides.value[std::pmr::string("score_scale")] = gr::pmt::Value(2.0f);
        const float scaled                                           = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));

        expect(lt(std::abs(scaled - 2.f * baseline), 1e-4f)) << "override must scale the fixture heatmap output";

        block.stop();
    };

    "meta_out carries model outputs 1..n-1 in a graph"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<FixtureSpectrumSource>({{"n_samples", 2U}});
        source.dataset   = makeTestDataSet(1024);
        auto& inference  = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("peaks_fixture_N1024.ort.gz")}});
        auto& outSink    = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();
        auto& peaksSink  = graph.emplaceBlock<CollectSink<gr::Tensor<float>>>();

        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, outSink, "in"s).has_value());
        expect(graph.connect(inference, "meta_out#0"s, peaksSink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(outSink.received.size(), 2UZ)) << "one primary output per input";
        expect(eq(peaksSink.received.size(), 2UZ)) << "one peaks tensor per inference";
        if (!peaksSink.received.empty()) {
            expect(eq(peaksSink.received[0].size(), 8UZ * 13UZ)) << "peaks tensor is [1, K=8, P=13]";
        }
    };
};

const boost::ut::suite<"OnnxInference model_overrides validation"> overrideValidationTests = [] {
    "Tensor<double> override scales the fixture output"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        const float baseline = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));

        block.model_overrides.value[std::pmr::string("score_scale")] = gr::pmt::Value(gr::Tensor<double>(gr::data_from, std::vector<double>{2.0}));
        const float scaled                                           = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));

        expect(lt(std::abs(scaled - 2.f * baseline), 1e-4f)) << "Tensor<double> override must scale the fixture output";
        block.stop();
    };

    "std::vector<float> override scales the fixture output"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        const float baseline = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));

        block.model_overrides.value[std::pmr::string("score_scale")] = gr::pmt::Value(std::vector<float>{3.f});
        const float scaled                                           = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));

        expect(lt(std::abs(scaled - 3.f * baseline), 1e-4f)) << "std::vector<float> override must scale the fixture output";
        block.stop();
    };

    "size-mismatched override triggers ErrorPolicy::Stop"_test = [] {
        OnnxInference<float> block;
        block.model_path   = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy = ErrorPolicy::Stop;
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        block.model_overrides.value[std::pmr::string("score_scale")] = gr::pmt::Value(std::vector<float>{1.f, 2.f, 3.f}); // initializer expects 1 element

        std::ignore = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "size mismatch must be reported, not silently ignored";
    };

    "unsupported override type latches pass-through under ErrorPolicy::Passthrough"_test = [] {
        OnnxInference<float> block;
        block.model_path   = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy = ErrorPolicy::Passthrough;
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        block.model_overrides.value[std::pmr::string("score_scale")] = gr::pmt::Value(std::string("2.0"));

        auto output = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(block.state() != gr::lifecycle::State::REQUESTED_STOP) << "Passthrough must not stop the block";
        expect(block._passthrough) << "unusable override must latch pass-through";
        expect(eq(output.signal_values.size(), 1024UZ)) << "input passes through unchanged";
        block.stop();
    };

    "override naming a non-initializer triggers ErrorPolicy::Stop"_test = [] {
        OnnxInference<float> block;
        block.model_path   = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy = ErrorPolicy::Stop;
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        block.model_overrides.value[std::pmr::string("no_such_initializer")] = gr::pmt::Value(1.0f);

        std::ignore = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "unknown initializer name must be reported";
    };
};

const boost::ut::suite<"OnnxInference config_in ports"> configInTests = [] {
    "config ports are sized from the model's overridable initializers"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        expect(eq(block.config_in.size(), 1UZ)) << "one config port per overridable initializer";
        expect(eq(block.config_input_names.value.size(), 1UZ));
        expect(eq(block.config_input_names.value[0], std::string("score_scale")));

        block.stop();
    };

    "config_in latches the last received value across process calls"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        const float baseline = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));
        expect(gt(baseline, 0.5f)) << "fixture heatmap peaks at ~1 with the baked score_scale=1";

        std::array<gr::DataSet<float>, 1> inArr{makeTestDataSet(1024)};
        std::array<gr::DataSet<float>, 1> outArr{};
        std::array<float, 1>              cfgValues{2.f};
        mock::IoSpan<gr::DataSet<float>>  inSpan{.data = inArr};
        mock::IoSpan<gr::DataSet<float>>  outSpan{.data = outArr};
        std::array<mock::CfgSpan, 1>      cfgArr{mock::CfgSpan{.data = cfgValues}};
        std::span<mock::CfgSpan>          cfgSpan(cfgArr);
        std::span<mock::MetaSpan>         metaSpan{};

        expect(block.processBulk(inSpan, cfgSpan, outSpan, metaSpan) == gr::work::Status::OK);
        expect(eq(cfgArr[0].consumed, 1UZ)) << "the config sample is latched and fully consumed (the Optional port exempts the feed's EOS from stopping the block)";
        expect(lt(std::abs(heatmapPeakValue(outArr[0]) - 2.f * baseline), 1e-4f)) << "fed config value must scale the fixture output";

        const float latched = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));
        expect(lt(std::abs(latched - 2.f * baseline), 1e-4f)) << "latched config value must persist without new port samples";

        block.stop();
    };

    "fed config_in port wins over model_overrides in a graph"_test = [] {
        using namespace std::string_literals;

        float baseline = 0.f;
        {
            OnnxInference<float> reference;
            reference.model_path = modelPath("peaks_fixture_N1024.ort.gz");
            reference.start();
            if (!reference.isModelLoaded()) {
                expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
                return;
            }
            baseline = heatmapPeakValue(processSingle<gr::DataSet<float>>(reference, makeTestDataSet(1024)));
            reference.stop();
        }
        expect(gt(baseline, 0.5f)) << "fixture heatmap peaks at ~1 with the baked score_scale=1";

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<FixtureSpectrumSource>({{"n_samples", 2U}});
        source.dataset   = makeTestDataSet(1024);
        auto& cfgSource  = graph.emplaceBlock<ScalarConfigSource>({{"value", 2.0f}, {"n_samples", 1U}});
        auto& inference  = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("peaks_fixture_N1024.ort.gz")}});
        auto& sink       = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();

        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        inference.model_overrides.value[std::pmr::string("score_scale")] = gr::pmt::Value(3.0f); // must lose to the fed port

        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(cfgSource, "out"s, inference, "config_in#0"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.received.size(), 2UZ)) << "one output per input spectrum";
        if (!sink.received.empty()) {
            const float scaled = heatmapPeakValue(sink.received.back());
            expect(lt(std::abs(scaled - 2.f * baseline), 1e-4f)) << "connected+fed config_in port must take precedence over the model_overrides entry";
        }
    };

    "config feed that EOSes having sent nothing leaves the block running"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<FixtureSpectrumSource>({{"n_samples", 3U}});
        source.dataset   = makeTestDataSet(1024);
        auto& cfgSource  = graph.emplaceBlock<ScalarConfigSource>({{"n_samples", 0U}}); // publishes nothing, then EOS
        auto& inference  = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("peaks_fixture_N1024.ort.gz")}});
        auto& sink       = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();

        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(cfgSource, "out"s, inference, "config_in#0"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.received.size(), 3UZ)) << "an empty config feed's EOS must not end the inference block";
    };

    // regression: handleError() must not emitErrorMessage under Passthrough. scheduler::Simple turns an
    // unconsumed error message into a thrown gr::exception, so the policy that exists to keep a graph
    // running aborted it instead. Driving the block via processOne cannot see this -- only a scheduler can.
    "ErrorPolicy::Passthrough survives a bad model inside a running graph"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<FixtureSpectrumSource>({{"n_samples", 3U}});
        source.dataset   = makeTestDataSet(1024);
        auto& inference  = graph.emplaceBlock<OnnxInference<float>>({{"model_path", "/nonexistent/model.onnx"s}, {"error_policy", "Passthrough"s}});
        auto& sink       = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();

        expect(!inference.isModelLoaded()) << "the point of this test is a model that cannot load";

        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value()) << "Passthrough must not abort the graph it exists to keep running";

        expect(eq(sink.received.size(), 3UZ)) << "every input frame must be forwarded unchanged";
        for (const auto& ds : sink.received) {
            expect(isMarkedPassthrough(ds)) << "forwarded frames must carry the onnx_passthrough marker";
        }
    };
};

const boost::ut::suite<"OnnxInference settings & error policy"> settingsAndErrorTests = [] {
    "model_path swap through the settings system reloads on a started block"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        expect(eq(block._session.modelN(), 1024UZ));

        expect(block.settings().set({{"model_path", modelPath("peaks_fixture_N512.ort.gz")}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(block._session.modelN(), 512UZ)) << "settingsChanged must reload the model";
        auto output = processSingle<gr::DataSet<float>>(block, makeTestDataSet(512));
        expect(gt(output.signal_values.size(), 0UZ)) << "swapped model must be usable";
        block.stop();
    };

    "normalise_mode Expression through the settings system reaches the inference input"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        const float baseline = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));
        expect(gt(baseline, 0.5f)) << "fixture heatmap peaks at ~1 for the ramp input";

        expect(block.settings().set({{"normalise_mode", "Expression"}, {"normalise_expr", std::string("vecOut := vecIn * 0")}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        std::ignore = block.settings().applyStagedParameters();
        expect(block.normalise_mode == NormaliseMode::Expression);

        const float zeroed = heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024)));
        expect(lt(zeroed, 0.1f)) << "the compiled expression must reach the inference input";
        block.stop();
    };

    "ErrorPolicy::Stop requests stop when inference fails"_test = [] {
        OnnxInference<float> block;
        block.model_path    = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy  = ErrorPolicy::Stop;
        block.history_depth = 2U; // fixture is M=1, so the [2xN] history batch is rejected at run time
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        std::ignore = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(block.state() != gr::lifecycle::State::REQUESTED_STOP) << "first sample only accumulates history";
        std::ignore = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "failed inference must trigger ErrorPolicy::Stop";
    };

    "ErrorPolicy::Passthrough falls back to pass-through when inference fails"_test = [] {
        OnnxInference<float> block;
        block.model_path    = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy  = ErrorPolicy::Passthrough;
        block.history_depth = 2U;
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        std::ignore = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        auto output = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(block.state() != gr::lifecycle::State::REQUESTED_STOP) << "Passthrough must not stop the block";
        expect(block._passthrough) << "block must latch pass-through after the failure";
        expect(eq(output.signal_values.size(), 1024UZ)) << "failed inference passes the latest input through";
        expect(isMarkedPassthrough(output)) << "error pass-through frames must carry the onnx_passthrough marker";
        block.stop();
    };
};

const boost::ut::suite<"OnnxInference execution provider"> executionProviderBlockTests = [] {
    "default cpu execution provider infers and reports the runtime's provider list"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        expect(eq(block.execution_provider.value, std::string("cpu")));
        expect(std::ranges::contains(block.available_providers.value, "CPUExecutionProvider")) << "available_providers must mirror Ort::GetAvailableProviders()";
        expect(gt(heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024))), 0.5f)) << "the cpu default must keep the fixture inference working";
        block.stop();
    };

    "unavailable execution provider triggers ErrorPolicy::Stop at load"_test = [] {
        if (std::ranges::contains(OnnxSession::availableProviders(), "CUDAExecutionProvider")) {
            std::println("skipped: the linked ONNX Runtime offers CUDA, so the unavailable-provider path cannot be exercised here");
            return;
        }
        OnnxInference<float> block;
        block.model_path         = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy       = ErrorPolicy::Stop;
        block.execution_provider = "cuda";
        block.start();
        expect(!block.isModelLoaded()) << "an unavailable provider must not fall back to a silently-created cpu session";
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "the provider failure must be reported through error_policy";
    };

    "unavailable execution provider latches pass-through under ErrorPolicy::Passthrough"_test = [] {
        if (std::ranges::contains(OnnxSession::availableProviders(), "CUDAExecutionProvider")) {
            std::println("skipped: the linked ONNX Runtime offers CUDA, so the unavailable-provider path cannot be exercised here");
            return;
        }
        OnnxInference<float> block;
        block.model_path         = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy       = ErrorPolicy::Passthrough;
        block.execution_provider = "cuda";
        block.start();
        expect(!block.isModelLoaded());
        expect(block._passthrough) << "the failed provider request must latch pass-through";
        expect(std::ranges::contains(block.available_providers.value, "CPUExecutionProvider")) << "the provider list must stay inspectable after the failure";
        auto output = processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024));
        expect(eq(output.signal_values.size(), 1024UZ)) << "input passes through unchanged";
        block.stop();
    };

    "execution_provider swap through the settings system re-creates the session"_test = [] {
        if (std::ranges::contains(OnnxSession::availableProviders(), "CUDAExecutionProvider")) {
            std::println("skipped: the linked ONNX Runtime offers CUDA, so the unavailable-provider path cannot be exercised here");
            return;
        }
        OnnxInference<float> block;
        block.model_path   = modelPath("peaks_fixture_N1024.ort.gz");
        block.error_policy = ErrorPolicy::Passthrough;
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        expect(block.settings().set({{"execution_provider", std::string("cuda")}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        std::ignore = block.settings().applyStagedParameters();
        expect(!block.isModelLoaded()) << "switching to an unavailable provider must tear the session down, not keep the old cpu session";
        expect(block._passthrough);

        expect(block.settings().set({{"execution_provider", std::string("cpu")}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        std::ignore = block.settings().applyStagedParameters();
        expect(block.isModelLoaded()) << "switching back to cpu must reload the model";
        expect(gt(heatmapPeakValue(processSingle<gr::DataSet<float>>(block, makeTestDataSet(1024))), 0.5f)) << "the re-created cpu session must infer again";
        block.stop();
    };
};

int main() { /* boost::ut */ }
