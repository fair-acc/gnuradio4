// Use-case (a): 1:1 elementwise inference — one DataSet in, one DataSet out, same length.
//
// Wires SineFrameSource -> OnnxInference -> CollectSink through gr::Graph and
// gr::scheduler::Simple, runs the analytic fixture y = 2x + 1 (affine_N64.onnx)
// and compares the measured output against the closed-form expectation.
//
// Starting point for your own single-input single-output model: swap model_path —
// input_size and normalise_mode are read from the ONNX custom metadata, so the
// block configures itself from the model file.
//
//   ex01_elementwise [model.onnx|model.ort]

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <print>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kFrameSize = 64UZ;
constexpr float       kScale     = 2.f; // baked into affine_N64.onnx
constexpr float       kOffset    = 1.f;

gr::DataSet<float> makeSineFrame() {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(kFrameSize)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};
    ds.signal_values.resize(kFrameSize);
    for (std::size_t i = 0; i < kFrameSize; ++i) {
        ds.signal_values[i] = std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(kFrameSize));
    }
    return ds;
}

struct SineFrameSource : gr::Block<SineFrameSource> {
    using Description = gr::Doc<"Emits one sine period as a DataSet, n_frames times.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(SineFrameSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeSineFrame();
        }
        _emitted += static_cast<gr::Size_t>(n);
        outSpan.publish(n);
        return _emitted >= n_frames ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

struct CollectSink : gr::Block<CollectSink> {
    using Description = gr::Doc<"Collects every received DataSet.">;

    gr::PortIn<gr::DataSet<float>> in;

    GR_MAKE_REFLECTABLE(CollectSink, in);

    std::vector<gr::DataSet<float>> received;

    void processOne(gr::DataSet<float> value) { received.push_back(std::move(value)); }
};

void drawChart(std::span<const float> input, std::span<const float> output) {
    std::vector<float> xAxis(kFrameSize);
    std::iota(xAxis.begin(), xAxis.end(), 0.f);

    auto chart        = gr::graphs::ImChart<80UZ, 30UZ>({{0.0, static_cast<double>(kFrameSize)}, {-1.5, 3.5}});
    chart.axis_name_x = "bin";
    chart.axis_name_y = "amplitude [a.u.]";
    chart.draw(xAxis, input, "input x");
    chart.draw(xAxis, output, "output y = 2x + 1");
    chart.draw();
}

void printComparisonTable(std::span<const float> input, std::span<const float> output) {
    std::println("  {:>4} {:>9} {:>9} {:>9} {:>10}", "bin", "input", "expected", "measured", "error");
    float maxError = 0.f;
    for (std::size_t i = 0; i < kFrameSize; ++i) {
        const float expected = kScale * input[i] + kOffset;
        maxError             = std::max(maxError, std::abs(output[i] - expected));
        if (i % 8UZ == 0UZ) {
            std::println("  {:>4} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.2e}", i, input[i], expected, output[i], output[i] - expected);
        }
    }
    std::println("  max |error| = {:.2e} over all {} bins\n", maxError, kFrameSize);
}

} // namespace

int main(int argc, char* argv[]) {
    std::string modelPath;
    if (argc > 1) {
        modelPath = argv[1];
    } else {
#ifdef MODEL_AFFINE_N64_PATH
        modelPath = MODEL_AFFINE_N64_PATH;
#else
        std::println("no model path provided (pass an .onnx/.ort model as first argument)");
        return 1;
#endif
    }

    std::println("=== OnnxInference use-case (a): 1:1 elementwise (y = 2x + 1) ===");
    std::println("model: {}\n", modelPath);

    gr::Graph graph;
    auto&     source    = graph.emplaceBlock<SineFrameSource>({{"n_frames", 4U}});
    auto&     inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({{"model_path", modelPath}});
    auto&     sink      = graph.emplaceBlock<CollectSink>();

    if (!inference.isModelLoaded()) {
        std::println("failed to load model '{}'", modelPath);
        return 1;
    }

    if (!graph.connect(source, std::string("out"), inference, std::string("in")).has_value() || !graph.connect(inference, std::string("out"), sink, std::string("in")).has_value()) {
        std::println("failed to connect the graph");
        return 1;
    }

    gr::scheduler::Simple scheduler;
    if (auto exchanged = scheduler.exchange(std::move(graph)); !exchanged.has_value()) {
        std::println("scheduler exchange failed: {}", exchanged.error().message);
        return 1;
    }
    if (auto result = scheduler.runAndWait(); !result.has_value()) {
        std::println("scheduler run failed: {}", result.error().message);
        return 1;
    }

    std::println("rate: {} inputs -> {} outputs (1:1)\n", 4, sink.received.size());
    if (sink.received.empty()) {
        std::println("no output received");
        return 1;
    }

    const auto             input = makeSineFrame();
    std::span<const float> outputValues(sink.received.front().signal_values);
    drawChart(input.signal_values, outputValues);
    printComparisonTable(input.signal_values, outputValues);
    return 0;
}
