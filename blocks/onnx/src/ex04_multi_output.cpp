// Use-case (d): multi-output inference — a model emitting several named outputs.
//
// Wires OffsetSineSource -> OnnxInference -> {CollectSink, TensorSink} through
// gr::Graph and gr::scheduler::Simple. The analytic fixture mean_rms_N64.onnx
// emits two named outputs per input vector: model output 0 ("mean") leaves on the
// block's `out` port as a DataSet, output 1 ("rms") leaves on the dynamic
// `meta_out#0` port as a Tensor. The full named output set is only requested from
// ONNX Runtime while at least one meta_out port is connected.
//
// The source emits frames o_k + a_k * sin over one full period, so mean = o_k and
// rms = sqrt(o_k^2 + a_k^2 / 2) hold exactly.
//
//   ex04_multi_output [model.onnx|model.ort]

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>
#include <vector>

namespace {

constexpr std::size_t kFrameSize = 64UZ;
constexpr gr::Size_t  kNFrames   = 4U;

float frameOffset(std::size_t frameIndex) { return static_cast<float>(frameIndex); }
float frameAmplitude(std::size_t frameIndex) { return 1.f + static_cast<float>(frameIndex); }

gr::DataSet<float> makeOffsetSineFrame(std::size_t frameIndex) {
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
        ds.signal_values[i] = frameOffset(frameIndex) + frameAmplitude(frameIndex) * std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(kFrameSize));
    }
    return ds;
}

struct OffsetSineSource : gr::Block<OffsetSineSource> {
    using Description = gr::Doc<"Emits n_frames DataSets; frame k is k + (1 + k) * sin over one period.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(OffsetSineSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeOffsetSineFrame(_emitted + i);
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

struct TensorSink : gr::Block<TensorSink> {
    using Description = gr::Doc<"Collects every received Tensor.">;

    gr::PortIn<gr::Tensor<float>> in;

    GR_MAKE_REFLECTABLE(TensorSink, in);

    std::vector<gr::Tensor<float>> received;

    void processOne(gr::Tensor<float> value) { received.push_back(std::move(value)); }
};

} // namespace

int main(int argc, char* argv[]) {
    std::string modelPath;
    if (argc > 1) {
        modelPath = argv[1];
    } else {
#ifdef MODEL_MEAN_RMS_N64_PATH
        modelPath = MODEL_MEAN_RMS_N64_PATH;
#else
        std::println("no model path provided (pass an .onnx/.ort model as first argument)");
        return 1;
#endif
    }

    std::println("=== OnnxInference use-case (d): multi-output (out = mean, meta_out#0 = rms) ===");
    std::println("model: {}\n", modelPath);

    gr::Graph graph;
    auto&     source    = graph.emplaceBlock<OffsetSineSource>({{"n_frames", kNFrames}});
    auto&     inference = graph.emplaceBlock<gr::blocks::onnx::OnnxInference<float>>({{"model_path", modelPath}});
    auto&     meanSink  = graph.emplaceBlock<CollectSink>();
    auto&     rmsSink   = graph.emplaceBlock<TensorSink>();

    if (!inference.isModelLoaded()) {
        std::println("failed to load model '{}'", modelPath);
        return 1;
    }
    std::println("meta outputs: {}", inference.meta_output_names.value);

    const bool connected = graph.connect(source, std::string("out"), inference, std::string("in")).has_value()      //
                           && graph.connect(inference, std::string("out"), meanSink, std::string("in")).has_value() //
                           && graph.connect(inference, std::string("meta_out#0"), rmsSink, std::string("in")).has_value();
    if (!connected) {
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

    std::println("rate: {} inputs -> {} mean DataSets on `out`, {} rms Tensors on `meta_out#0`\n", kNFrames, meanSink.received.size(), rmsSink.received.size());

    std::println("  {:>5} {:>8} {:>8} | {:>9} {:>9} | {:>9} {:>9} | {:>10}", "frame", "offset", "amp", "mean", "mean'", "rms", "rms'", "max error");
    float maxError = 0.f;
    for (std::size_t k = 0; k < std::min(meanSink.received.size(), rmsSink.received.size()); ++k) {
        const float expectedMean = frameOffset(k);
        const float expectedRms  = std::sqrt(expectedMean * expectedMean + 0.5f * frameAmplitude(k) * frameAmplitude(k));
        const float measuredMean = meanSink.received[k].signal_values.empty() ? 0.f : meanSink.received[k].signal_values.front();
        const float measuredRms  = rmsSink.received[k].empty() ? 0.f : rmsSink.received[k][0];
        const float error        = std::max(std::abs(measuredMean - expectedMean), std::abs(measuredRms - expectedRms));
        maxError                 = std::max(maxError, error);
        std::println("  {:>5} {:>8.1f} {:>8.1f} | {:>9.4f} {:>9.4f} | {:>9.4f} {:>9.4f} | {:>10.2e}", k, frameOffset(k), frameAmplitude(k), expectedMean, measuredMean, expectedRms, measuredRms, error);
    }
    std::println("  max |error| = {:.2e}\n", maxError);
    return 0;
}
