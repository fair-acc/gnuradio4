// Use-case (b): N:1 decimating inference — the block consumes N DataSets per published
// output, a genuine rate change through the framework's gr::Resampling machinery.
//
// Wires SequenceFrameSource -> OnnxInference -> CollectSink through gr::Graph and
// gr::scheduler::Simple. The analytic fixture frame_mean_N64_M4.onnx averages a window
// of M = 4 frames; history_depth comes from the model metadata, history_stride = 4
// makes the windows non-overlapping: 16 inputs become 4 outputs.
//
// Starting point for your own batching model: swap model_path and set history_stride
// to the number of inputs to consume per inference (stride < depth overlaps windows).
//
//   ex02_decimating [model.onnx|model.ort]

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include "../ModelPath.hpp"

#include <cmath>
#include <print>
#include <vector>

namespace {

constexpr std::size_t kFrameSize = 64UZ;
constexpr gr::Size_t  kNFrames   = 16U;
constexpr gr::Size_t  kStride    = 4U; // frames consumed per output; equals the fixture's window depth M

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

struct SequenceFrameSource : gr::Block<SequenceFrameSource> {
    using Description = gr::Doc<"Emits n_frames DataSets; frame k is filled with the value k.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(SequenceFrameSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeConstantFrame(static_cast<float>(_emitted + i));
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

} // namespace

int main(int argc, char* argv[]) {
    const std::string kDefaultModel = gr::blocks::onnx::test::modelPath("frame_mean_N64_M4.ort.gz");
    const std::string modelPath     = (argc > 1) ? std::string(argv[1]) : kDefaultModel;

    std::println("=== OnnxInference use-case (b): N:1 decimating (window mean, M = 4) ===");
    std::println("model: {}\n", modelPath);

    gr::Graph graph;
    auto&     source    = graph.emplaceBlock<SequenceFrameSource>({{"n_frames", kNFrames}});
    auto&     inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({{"model_path", modelPath}, {"history_stride", kStride}});
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

    std::println("rate: {} inputs -> {} outputs ({}:1 decimation)\n", kNFrames, sink.received.size(), kStride);

    std::println("  {:>6} {:>9} {:>13} {:>13} {:>10}", "window", "frames", "expected mean", "measured mean", "error");
    float maxError = 0.f;
    for (std::size_t w = 0; w < sink.received.size(); ++w) {
        const float first    = static_cast<float>(w * kStride);
        const float expected = first + 0.5f * static_cast<float>(kStride - 1U); // mean of first..first+stride-1
        const float measured = sink.received[w].signal_values.empty() ? 0.f : sink.received[w].signal_values.front();
        maxError             = std::max(maxError, std::abs(measured - expected));
        std::println("  {:>6} {:>4.0f}..{:<4.0f} {:>13.4f} {:>13.4f} {:>10.2e}", w, first, first + static_cast<float>(kStride - 1U), expected, measured, measured - expected);
    }
    std::println("  max |error| = {:.2e}\n", maxError);
    return 0;
}
