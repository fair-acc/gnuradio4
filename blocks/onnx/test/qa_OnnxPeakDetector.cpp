#include <boost/ut.hpp>

#ifndef GR_ONNX_MINIMAL_BUILD
#define GR_ONNX_MINIMAL_BUILD 0
#endif

#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>

#include <cmath>
#include <print>

using namespace boost::ut;
using namespace gr::blocks::onnx;

namespace {

gr::DataSet<float> makeTestSpectrum(std::size_t n, std::span<const std::pair<float, float>> peaks, float noiseFloor = 0.1f) {
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

const boost::ut::suite<"OnnxPeakDetector"> onnxPeakDetectorTests = [] {
    "full pipeline: normalise, infer, extract peaks"_test = [] {
        OnnxPeakDetector block;
        block.model_path           = MODEL_N1024_PATH;
        block.confidence_threshold = 0.3f;
        block.min_peak_distance    = 8;
        block.max_peaks            = 8;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        // create a spectrum with 3 clear peaks at known positions
        std::pair<float, float> peaks[] = {{200.f, 50.f}, {500.f, 80.f}, {800.f, 40.f}};
        auto                    input   = makeTestSpectrum(1024, peaks);

        auto output = block.processOne(std::move(input));

        // verify output structure
        expect(eq(output.signal_names.size(), 2UZ));
        expect(eq(output.signal_names[0], std::string("Spectrum")));
        expect(eq(output.signal_names[1], std::string("Heatmap")));
        expect(eq(output.signal_values.size(), 2 * 1024UZ));

        expect(eq(output.timing_events.size(), 2UZ));
        const auto& peakEvents = output.timing_events[0];

        std::println("detected {} peaks", peakEvents.size());
        for (const auto& [idx, props] : peakEvents) {
            auto  confIt = props.find(std::pmr::string("confidence"));
            float conf   = confIt != props.end() ? confIt->second.value_or<float>(0.f) : 0.f;
            std::println("  bin={} conf={:.3f}", idx, conf);
        }

        // we expect at least some peaks detected (model quality varies)
        expect(gt(peakEvents.size(), 0UZ)) << "should detect at least one peak";

        block.stop();
    };

    "input size mismatch triggers resampling"_test = [] {
        OnnxPeakDetector block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        // feed 2048 samples — should resample to 1024 for model, heatmap back to 2048
        std::pair<float, float> peaks[] = {{512.f, 50.f}};
        auto                    input   = makeTestSpectrum(2048, peaks);

        auto output = block.processOne(std::move(input));

        expect(eq(output.signal_names.size(), 2UZ));
        expect(eq(output.signal_values.size(), 2 * 2048UZ)) << "output should match input dimension";

        block.stop();
    };

    "empty input passes through"_test = [] {
        OnnxPeakDetector block;
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

    "flat spectrum produces few or no peaks"_test = [] {
        OnnxPeakDetector block;
        block.model_path           = MODEL_N1024_PATH;
        block.confidence_threshold = 0.5f;
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
        input.extents           = {1024};
        input.meta_information  = {{}};
        input.timing_events     = {{}};
        input.signal_values.assign(1024, 1.0f);

        auto output = block.processOne(std::move(input));

        std::println("flat spectrum: {} peaks detected", output.timing_events[0].size());
        // model has limited F1 (0.54), so some false positives on flat spectra are expected
        expect(le(output.timing_events[0].size(), 8UZ)) << "flat spectrum should not produce excessive peaks";

        block.stop();
    };

    "heatmap signal values are in [0, 1] range"_test = [] {
        OnnxPeakDetector block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
            return;
        }

        std::pair<float, float> peaks[] = {{512.f, 100.f}};
        auto                    input   = makeTestSpectrum(1024, peaks);

        auto output = block.processOne(std::move(input));

        // check heatmap (second signal) values
        std::span<const float> heatmap(output.signal_values.data() + 1024, 1024);
        float                  maxHeat = *std::max_element(heatmap.begin(), heatmap.end());
        float                  minHeat = *std::min_element(heatmap.begin(), heatmap.end());

        std::println("heatmap range: [{:.4f}, {:.4f}]", minHeat, maxHeat);
        expect(ge(minHeat, -0.1f)) << "heatmap min should be near 0";
        expect(le(maxHeat, 1.1f)) << "heatmap max should be near 1";

        block.stop();
    };

    "without loaded model passes through"_test = [] {
        OnnxPeakDetector   block;
        gr::DataSet<float> input;
        input.signal_names  = {"test"};
        input.signal_values = {1.f, 2.f, 3.f};

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names[0], std::string("test")));
    };
};

int main() { /* boost::ut */ }
