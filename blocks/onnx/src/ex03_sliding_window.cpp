// Use-case (c): sliding-window history inference — history_depth = M, stride 1.
//
// Wires ModulatedFrameSource -> OnnxInference -> CollectSink through gr::Graph and
// gr::scheduler::Simple. The analytic fixture frame_delta_N64_M8.onnx emits newest
// minus oldest frame of an M = 8 window. With stride 1 the block stays 1:1: the
// first M-1 inputs pass through unchanged while the window fills (warm-up), then
// every input produces one inference output over the newest M frames.
//
// The source modulates the frame level with a slow sine, so the emitted delta traces
// the 7-frame-lag discrete derivative of that modulation — chart and table compare
// it against the closed-form expectation.
//
//   ex03_sliding_window [model.onnx|model.ort]

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include "../ModelPath.hpp"

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <print>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kFrameSize    = 64UZ;
constexpr gr::Size_t  kNFrames      = 24U;
constexpr std::size_t kHistoryDepth = 8UZ; // window depth M baked into frame_delta_N64_M8.onnx

float frameLevel(std::size_t frameIndex) { return std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(frameIndex) / 16.f); }

gr::DataSet<float> makeConstantFrame(float fillValue) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.signal_values.assign(kFrameSize, fillValue);
    ds.extents          = {static_cast<std::int32_t>(kFrameSize)};
    ds.meta_information = {{}};
    ds.timing_events    = {{}};
    return ds;
}

struct ModulatedFrameSource : gr::Block<ModulatedFrameSource> {
    using Description = gr::Doc<"Emits n_frames DataSets; frame k is filled with sin(2*pi*k/16).">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(ModulatedFrameSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeConstantFrame(frameLevel(_emitted + i));
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

float expectedDelta(std::size_t frameIndex) { return frameLevel(frameIndex) - frameLevel(frameIndex - (kHistoryDepth - 1UZ)); }

void drawChart(std::span<const float> measured) {
    std::vector<float> xAxis(measured.size());
    std::iota(xAxis.begin(), xAxis.end(), 0.f);

    std::vector<float> expected(measured.size(), std::numeric_limits<float>::quiet_NaN());
    for (std::size_t k = kHistoryDepth - 1UZ; k < expected.size(); ++k) {
        expected[k] = expectedDelta(k);
    }

    auto chart        = gr::graphs::ImChart<80UZ, 30UZ>({{0.0, static_cast<double>(measured.size())}, {-2.2, 2.2}});
    chart.axis_name_x = "frame index";
    chart.axis_name_y = "delta [a.u.]";
    chart.draw(xAxis, measured, "measured");
    chart.draw<gr::graphs::Style::Marker>(xAxis, expected, "expected");
    chart.draw();
}

} // namespace

int main(int argc, char* argv[]) {
    const std::string kDefaultModel = gr::blocks::onnx::test::modelPath("frame_delta_N64_M8.ort.gz");
    const std::string modelPath     = (argc > 1) ? std::string(argv[1]) : kDefaultModel;

    std::println("=== OnnxInference use-case (c): sliding-window history (newest - oldest, M = 8) ===");
    std::println("model: {}\n", modelPath);

    gr::Graph graph;
    auto&     source    = graph.emplaceBlock<ModulatedFrameSource>({{"n_frames", kNFrames}});
    auto&     inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({{"model_path", modelPath}});
    auto&     sink      = graph.emplaceBlock<CollectSink>();

    if (!inference.isModelLoaded()) {
        std::println("failed to load model '{}'", modelPath);
        return 1;
    }
    std::println("history_depth = {} (from model metadata), history_stride = {}", inference.history_depth.value, inference.history_stride.value);

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

    std::println("rate: {} inputs -> {} outputs (1:1, first {} warm-up frames pass through)\n", kNFrames, sink.received.size(), kHistoryDepth - 1UZ);

    std::vector<float> measured;
    for (const gr::DataSet<float>& output : sink.received) {
        measured.push_back(output.signal_values.empty() ? 0.f : output.signal_values.front());
    }
    drawChart(measured);

    std::println("  {:>5} {:>9} {:>9} {:>9} {:>10}  {}", "frame", "input", "expected", "measured", "error", "path");
    float maxError = 0.f;
    for (std::size_t k = 0; k < measured.size(); ++k) {
        const bool  warm     = k >= kHistoryDepth - 1UZ;
        const float expected = warm ? expectedDelta(k) : frameLevel(k); // warm-up passes the input through
        maxError             = std::max(maxError, std::abs(measured[k] - expected));
        std::println("  {:>5} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.2e}  {}", k, frameLevel(k), expected, measured[k], measured[k] - expected, warm ? "inference" : "warm-up pass-through");
    }
    std::println("  max |error| = {:.2e}\n", maxError);
    return 0;
}
