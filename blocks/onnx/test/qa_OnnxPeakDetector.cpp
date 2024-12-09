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

gr::DataSet<float> makeFlatSpectrum(std::size_t n, float value = 1.0f) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {""};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(n)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};
    ds.signal_values.assign(n, value);
    return ds;
}

struct ModelTestCase {
    const char* modelPath;
    std::size_t modelN;
    const char* label;
};

void runPeakDetectionTest(const ModelTestCase& tc, std::size_t inputN) {
    OnnxPeakDetector block;
    block.model_path           = tc.modelPath;
    block.confidence_threshold = 0.3f;
    block.min_peak_distance    = 8;
    block.max_peaks            = 8;
    block.start();

    if (!block._session.isLoaded()) {
        std::println("skip: {} not loadable", tc.label);
        return;
    }

    expect(eq(block._session.modelN(), tc.modelN)) << tc.label << " model dimension";

    // scale peak positions to input size
    float                   scale    = static_cast<float>(inputN) / 1024.f;
    std::pair<float, float> peaks[]  = {{200.f * scale, 50.f}, {500.f * scale, 80.f}, {800.f * scale, 40.f}};
    auto                    input    = makeTestSpectrum(inputN, peaks);
    bool                    resamples = (inputN != tc.modelN);

    auto output = block.processOne(std::move(input));

    // output structure
    expect(eq(output.signal_names.size(), 2UZ)) << tc.label;
    expect(eq(output.signal_names[0], std::string("Spectrum"))) << tc.label;
    expect(eq(output.signal_names[1], std::string("Heatmap"))) << tc.label;
    expect(eq(output.signal_values.size(), 2 * inputN)) << tc.label << " output matches input dimension";

    // heatmap range
    std::span<const float> heatmap(output.signal_values.data() + static_cast<std::ptrdiff_t>(inputN), inputN);
    float                  maxHeat = *std::max_element(heatmap.begin(), heatmap.end());
    float                  minHeat = *std::min_element(heatmap.begin(), heatmap.end());
    expect(ge(minHeat, -0.1f)) << tc.label << " heatmap min";
    expect(le(maxHeat, 1.1f)) << tc.label << " heatmap max";

    // peak detection
    const auto& peakEvents = output.timing_events[0];
    std::println("[{}] inputN={} modelN={} resamples={}: {} peaks, heatmap=[{:.4f}, {:.4f}]",
                 tc.label, inputN, tc.modelN, resamples, peakEvents.size(), minHeat, maxHeat);
    for (const auto& [idx, props] : peakEvents) {
        auto  confIt = props.find(std::pmr::string("confidence"));
        float conf   = confIt != props.end() ? confIt->second.value_or<float>(0.f) : 0.f;
        std::println("  bin={} conf={:.3f}", idx, conf);
    }

    expect(gt(peakEvents.size(), 0UZ)) << tc.label << " should detect at least one peak";

    block.stop();
}

void runFlatSpectrumTest(const ModelTestCase& tc) {
    OnnxPeakDetector block;
    block.model_path           = tc.modelPath;
    block.confidence_threshold = 0.5f;
    block.start();

    if (!block._session.isLoaded()) {
        std::println("skip: {} not loadable", tc.label);
        return;
    }

    auto output = block.processOne(makeFlatSpectrum(tc.modelN));

    std::println("[{}] flat spectrum: {} peaks detected", tc.label, output.timing_events[0].size());
    expect(le(output.timing_events[0].size(), 8UZ)) << tc.label << " flat spectrum should not produce excessive peaks";

    block.stop();
}

} // namespace

const boost::ut::suite<"OnnxPeakDetector N1024"> n1024Tests = [] {
    constexpr ModelTestCase tc{MODEL_N1024_PATH, 1024, "N1024"};

    "N1024: detect peaks at native resolution"_test = [&tc] {
        runPeakDetectionTest(tc, 1024);
    };

    "N1024: detect peaks with decimation (2048 → 1024)"_test = [&tc] {
        runPeakDetectionTest(tc, 2048);
    };

    "N1024: detect peaks with interpolation (512 → 1024)"_test = [&tc] {
        runPeakDetectionTest(tc, 512);
    };

    "N1024: flat spectrum produces few peaks"_test = [&tc] {
        runFlatSpectrumTest(tc);
    };
};

#ifdef MODEL_N4096_PATH

const boost::ut::suite<"OnnxPeakDetector N4096"> n4096Tests = [] {
    constexpr ModelTestCase tc{MODEL_N4096_PATH, 4096, "N4096"};

    "N4096: detect peaks at native resolution"_test = [&tc] {
        runPeakDetectionTest(tc, 4096);
    };

    "N4096: detect peaks with decimation (8192 → 4096)"_test = [&tc] {
        runPeakDetectionTest(tc, 8192);
    };

    "N4096: detect peaks with interpolation (1024 → 4096)"_test = [&tc] {
        runPeakDetectionTest(tc, 1024);
    };

    "N4096: flat spectrum produces few peaks"_test = [&tc] {
        runFlatSpectrumTest(tc);
    };
};

#endif // MODEL_N4096_PATH

const boost::ut::suite<"OnnxPeakDetector common"> commonTests = [] {
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
