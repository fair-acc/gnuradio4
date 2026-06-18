// Identity use-case — the integration litmus test for the OnnxInference block.
//
// Wires TestFrameSource -> OnnxInference -> CollectSink through gr::Graph and
// gr::scheduler::Simple, runs a model that computes y = x exactly (identity_N64.onnx)
// and checks that every output sample equals its input sample bit for bit. The model
// has no arithmetic to get subtly wrong, so any nonzero error here points at the GR4
// block wiring or the ONNX Runtime session, never at the model.
//
// This example doubles as a "is my model wired up correctly?" diagnostic: it also
// prints the model's I/O shape and the execution providers ONNX Runtime offers, both
// as read back from the OnnxInference block after loading.
//
//   ex00_identity [model.onnx|model.ort]

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
constexpr gr::Size_t  kNFrames   = 4U;

gr::DataSet<float> makeTestFrame() {
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
        const float t       = static_cast<float>(i) / static_cast<float>(kFrameSize);
        ds.signal_values[i] = std::sin(2.f * std::numbers::pi_v<float> * 3.f * t) + 0.4f * std::cos(2.f * std::numbers::pi_v<float> * 7.f * t);
    }
    return ds;
}

struct TestFrameSource : gr::Block<TestFrameSource> {
    using Description = gr::Doc<"Emits the same two-tone test frame n_frames times.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(TestFrameSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeTestFrame();
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

    auto chart        = gr::graphs::ImChart<80UZ, 20UZ>({{0.0, static_cast<double>(kFrameSize)}, {-1.5, 1.5}});
    chart.axis_name_x = "bin";
    chart.axis_name_y = "amplitude [a.u.]";
    chart.draw(xAxis, input, "input x");
    chart.draw(xAxis, output, "output y (identity)");
    chart.draw();
}

void printComparisonTable(std::span<const float> input, std::span<const float> output) {
    std::println("  {:>4} {:>9} {:>9} {:>10}", "bin", "input", "output", "|error|");
    float maxError = 0.f;
    for (std::size_t i = 0; i < kFrameSize; ++i) {
        maxError = std::max(maxError, std::abs(output[i] - input[i]));
        if (i % 8UZ == 0UZ) {
            std::println("  {:>4} {:>9.4f} {:>9.4f} {:>10.2e}", i, input[i], output[i], std::abs(output[i] - input[i]));
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
#ifdef MODEL_IDENTITY_N64_PATH
        modelPath = MODEL_IDENTITY_N64_PATH;
#else
        std::println("no model path provided (pass an .onnx/.ort model as first argument)");
        return 1;
#endif
    }

    std::println("=== OnnxInference identity litmus test (y = x) ===");
    std::println("model: {}\n", modelPath);

    gr::Graph graph;
    auto&     source    = graph.emplaceBlock<TestFrameSource>({{"n_frames", kNFrames}});
    auto&     inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({{"model_path", modelPath}});
    auto&     sink      = graph.emplaceBlock<CollectSink>();

    if (!inference.isModelLoaded()) {
        std::println("failed to load model '{}'", modelPath);
        return 1;
    }

    std::println("model I/O shape:      {}", inference.model_io_shape.value);
    std::println("available providers:  {}\n", inference.available_providers.value);

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

    std::println("rate: {} inputs -> {} outputs (1:1)\n", kNFrames, sink.received.size());
    if (sink.received.empty()) {
        std::println("no output received");
        return 1;
    }

    // real inference tags its output "inference_output"; pass-through (no model, warm-up,
    // ErrorPolicy::Passthrough) forwards the source's own signal_names unchanged, so a
    // pass-through frame would otherwise read as a silent, tautological PASS below
    const bool anyPassthrough = std::ranges::any_of(sink.received, [](const gr::DataSet<float>& frame) { return frame.signal_names.empty() || frame.signal_names.front() != "inference_output"; });
    if (anyPassthrough) {
        std::println("FAIL: block degraded to pass-through, inference never ran — {}", inference.passthrough_reason.value);
        return 1;
    }

    const auto             input = makeTestFrame();
    std::span<const float> firstOutput(sink.received.front().signal_values);
    std::println("chart: input and output must overlap exactly — a single visible curve is the pass signature\n");
    drawChart(input.signal_values, firstOutput);
    printComparisonTable(input.signal_values, firstOutput);

    float maxError = 0.f;
    for (const gr::DataSet<float>& frame : sink.received) {
        for (std::size_t i = 0; i < std::min(kFrameSize, frame.signal_values.size()); ++i) {
            maxError = std::max(maxError, std::abs(frame.signal_values[i] - input.signal_values[i]));
        }
    }

    if (maxError > 0.f) {
        std::println("FAIL: identity mismatch — max |error| = {:.2e} across {} frame(s)", maxError, sink.received.size());
        return 1;
    }
    std::println("PASS: output == input exactly — max |error| = {:.2e} across {} frame(s)", maxError, sink.received.size());
    return 0;
}
