#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include "../ModelPath.hpp"

#include <cmath>
#include <numeric>
#include <string>

using namespace boost::ut;
using namespace gr::blocks::onnx;
using gr::blocks::onnx::test::modelPath;

namespace {

constexpr std::size_t kFrameSize = 64UZ;

gr::DataSet<float> makeFrame(float fillValue) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {""};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.signal_values.assign(kFrameSize, fillValue);
    ds.extents          = {static_cast<std::int32_t>(kFrameSize)};
    ds.meta_information = {{}};
    ds.timing_events    = {{}};
    return ds;
}

gr::DataSet<float> makeRampFrame() {
    gr::DataSet<float> ds = makeFrame(0.f);
    std::iota(ds.signal_values.begin(), ds.signal_values.end(), 1.f);
    return ds;
}

struct RampFrameSource : gr::Block<RampFrameSource> {
    using Description = gr::Doc<"Test source emitting the 1..64 ramp DataSet n_frames times.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(RampFrameSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeRampFrame();
        }
        _emitted += static_cast<gr::Size_t>(n);
        outSpan.publish(n);
        return _emitted >= n_frames ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

struct SequenceFrameSource : gr::Block<SequenceFrameSource> {
    using Description = gr::Doc<"Test source emitting n_frames DataSets; frame k is filled with the value k.">;

    gr::PortOut<gr::DataSet<float>> out;

    gr::Size_t n_frames = 1U;

    GR_MAKE_REFLECTABLE(SequenceFrameSource, out, n_frames);

    gr::Size_t _emitted = 0U;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(static_cast<std::size_t>(n_frames - _emitted), outSpan.size());
        for (std::size_t i = 0; i < n; ++i) {
            outSpan[i] = makeFrame(static_cast<float>(_emitted + i));
        }
        _emitted += static_cast<gr::Size_t>(n);
        outSpan.publish(n);
        return _emitted >= n_frames ? gr::work::Status::DONE : gr::work::Status::OK;
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

bool allValuesNear(const std::vector<float>& values, float expected, float tolerance = 1e-5f) {
    return std::ranges::all_of(values, [expected, tolerance](float v) { return std::abs(v - expected) < tolerance; });
}

} // namespace

const boost::ut::suite<"OnnxInference use-case (a): 1:1 elementwise"> useCaseATests = [] {
    "affine fixture loads and self-configures from metadata"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("affine_N64.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("affine_N64.ort.gz");
            return;
        }
        expect(block.model_input_shape.value == std::vector<gr::Size_t>{1U, 1U, 64U}) << "input shape";
        expect(block.model_output_shape.value == std::vector<gr::Size_t>{1U, 64U}) << "output shape";
        expect(block.normalise_mode == NormaliseMode::None) << "normalise_mode must come from the model metadata";
        block.stop();
    };

    "graph run computes y = 2x + 1, one output per input"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<RampFrameSource>({{"n_frames", 3U}});
        auto&     inference = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("affine_N64.ort.gz")}, {"normalise_mode", "None"}, {"resample_mode", "None"}});
        auto&     sink      = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();
        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("affine_N64.ort.gz");
            return;
        }
        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.received.size(), 3UZ)) << "identity rate: one output per input";
        for (const gr::DataSet<float>& output : sink.received) {
            expect(eq(output.signal_values.size(), kFrameSize));
            for (std::size_t j = 0; j < output.signal_values.size(); ++j) {
                expect(lt(std::abs(output.signal_values[j] - (2.f * static_cast<float>(j + 1) + 1.f)), 1e-5f)) << "y = 2x + 1 at bin " << j;
            }
        }
    };
};

const boost::ut::suite<"OnnxInference use-case (b): N:1 decimating"> useCaseBTests = [] {
    "frame-mean fixture loads and self-configures history_depth = 4"_test = [] {
        OnnxInference<float> block;
        block.model_path     = modelPath("frame_mean_N64_M4.ort.gz");
        block.history_stride = 4U;
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("frame_mean_N64_M4.ort.gz");
            return;
        }
        expect(eq(static_cast<std::size_t>(block.history_depth.value), 4UZ)) << "history_depth must come from the model";
        block.stop();
    };

    "graph run decimates 8 inputs to 2 window means"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<SequenceFrameSource>({{"n_frames", 8U}});
        auto&     inference = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("frame_mean_N64_M4.ort.gz")}, {"history_stride", 4U}, {"normalise_mode", "None"}, {"resample_mode", "None"}});
        auto&     sink      = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();
        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("frame_mean_N64_M4.ort.gz");
            return;
        }
        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.received.size(), 2UZ)) << "genuine 4:1 rate change: 8 inputs -> 2 outputs";
        if (sink.received.size() == 2UZ) {
            expect(allValuesNear(sink.received[0].signal_values, 1.5f)) << "mean of frames 0..3";
            expect(allValuesNear(sink.received[1].signal_values, 5.5f)) << "mean of frames 4..7";
        }
    };
};

const boost::ut::suite<"OnnxInference use-case (c): sliding-window history"> useCaseCTests = [] {
    "frame-delta fixture loads and self-configures history_depth = 8"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("frame_delta_N64_M8.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("frame_delta_N64_M8.ort.gz");
            return;
        }
        expect(eq(static_cast<std::size_t>(block.history_depth.value), 8UZ)) << "history_depth must come from the model";
        block.stop();
    };

    "graph run passes 7 warm-up frames through, then emits newest - oldest per input"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<SequenceFrameSource>({{"n_frames", 10U}});
        auto&     inference = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("frame_delta_N64_M8.ort.gz")}, {"normalise_mode", "None"}, {"resample_mode", "None"}});
        auto&     sink      = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();
        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("frame_delta_N64_M8.ort.gz");
            return;
        }
        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink.received.size(), 10UZ)) << "stride 1 keeps the block 1:1";
        for (std::size_t i = 0; i < std::min(sink.received.size(), 7UZ); ++i) {
            expect(eq(sink.received[i].signal_names[0], std::string("Spectrum"))) << "warm-up frame " << i << " passes through unchanged";
            expect(allValuesNear(sink.received[i].signal_values, static_cast<float>(i)));
        }
        for (std::size_t i = 7; i < sink.received.size(); ++i) {
            expect(eq(sink.received[i].signal_names[0], std::string("inference_output"))) << "warm frame " << i << " is an inference result";
            expect(allValuesNear(sink.received[i].signal_values, 7.f)) << "newest(k) - oldest(k-7) = 7 at frame " << i;
        }
    };
};

const boost::ut::suite<"OnnxInference use-case (d): multi-output"> useCaseDTests = [] {
    "mean/rms fixture loads and exposes one meta_out port"_test = [] {
        OnnxInference<float> block;
        block.model_path = modelPath("mean_rms_N64.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("mean_rms_N64.ort.gz");
            return;
        }
        expect(eq(block.meta_out.size(), 1UZ)) << "model outputs 1..n-1 feed meta_out";
        expect(block.meta_output_names.value == std::vector<std::string>{"rms"});
        block.stop();
    };

    "graph run routes mean to out and rms to meta_out#0"_test = [] {
        using namespace std::string_literals;

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<RampFrameSource>({{"n_frames", 3U}});
        auto&     inference = graph.emplaceBlock<OnnxInference<float>>({{"model_path", modelPath("mean_rms_N64.ort.gz")}, {"normalise_mode", "None"}, {"resample_mode", "None"}});
        auto&     meanSink  = graph.emplaceBlock<CollectSink<gr::DataSet<float>>>();
        auto&     rmsSink   = graph.emplaceBlock<CollectSink<gr::Tensor<float>>>();
        if (!inference.isModelLoaded()) {
            expect(false) << "tracked fixture failed to load: " << modelPath("mean_rms_N64.ort.gz");
            return;
        }
        expect(graph.connect(source, "out"s, inference, "in"s).has_value());
        expect(graph.connect(inference, "out"s, meanSink, "in"s).has_value());
        expect(graph.connect(inference, "meta_out#0"s, rmsSink, "in"s).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.runAndWait().has_value());

        const float expectedMean = 32.5f;              // mean of 1..64
        const float expectedRms  = std::sqrt(1397.5f); // sqrt(mean(x^2)) of 1..64
        expect(eq(meanSink.received.size(), 3UZ)) << "one mean DataSet per input";
        expect(eq(rmsSink.received.size(), 3UZ)) << "one rms tensor per inference";
        for (const gr::DataSet<float>& mean : meanSink.received) {
            expect(eq(mean.signal_values.size(), 1UZ));
            expect(lt(std::abs(mean.signal_values[0] - expectedMean), 1e-3f));
        }
        for (const gr::Tensor<float>& rms : rmsSink.received) {
            expect(eq(rms.size(), 1UZ));
            expect(lt(std::abs(rms[0] - expectedRms), 1e-3f));
        }
    };
};

int main() { /* boost::ut */ }
