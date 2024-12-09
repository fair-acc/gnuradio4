#include <boost/ut.hpp>

#include <gnuradio-4.0/onnx/PeakExtractor.hpp>

#include <cmath>
#include <print>

using namespace boost::ut;
using namespace gr::blocks::onnx;

const boost::ut::suite<"PeakExtractor"> peakExtractorTests = [] {
    "extracts peaks from synthetic inference output"_test = [] {
        constexpr std::size_t N = 100;
        constexpr std::size_t R = 8;

        // build synthetic inference output: [N heatmap, N*R regression]
        std::vector<float> inferenceOut(N + N * R, 0.f);

        // place peaks in heatmap
        auto placePeak = [&](std::size_t idx, float conf, float offset, float sigma) {
            inferenceOut[idx]             = conf; // heatmap
            inferenceOut[N + idx * R + 0] = offset / static_cast<float>(N);
            inferenceOut[N + idx * R + 1] = 1.0f; // amplitude
            inferenceOut[N + idx * R + 2] = sigma / static_cast<float>(N);
            inferenceOut[N + idx * R + 3] = 2.0f * sigma / static_cast<float>(N); // w68
            inferenceOut[N + idx * R + 4] = 4.0f * sigma / static_cast<float>(N); // w96
            inferenceOut[N + idx * R + 5] = 6.0f * sigma / static_cast<float>(N); // w99
            inferenceOut[N + idx * R + 6] = 0.3f;                                 // kurtosis / 10
            inferenceOut[N + idx * R + 7] = 0.5f;                                 // (excess_kurtosis + 5) / 10
        };

        placePeak(20, 0.9f, 0.3f, 5.0f);
        placePeak(50, 0.7f, -0.1f, 3.0f);
        placePeak(80, 0.5f, 0.0f, 8.0f);

        gr::DataSet<float> input;
        input.signal_names      = {"inference_output"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values     = std::move(inferenceOut);
        input.extents           = {static_cast<std::int32_t>(N + N * R)};
        input.meta_information  = {{}};
        input.timing_events     = {{}};

        PeakExtractor block;
        block.regression_channels  = R;
        block.model_n              = N;
        block.confidence_threshold = 0.3f;
        block.min_peak_distance    = 5;
        block.max_peaks            = 10;

        auto output = block.processOne(std::move(input));

        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("Heatmap")));
        expect(eq(output.signal_values.size(), N));

        expect(ge(output.timing_events.size(), 1UZ));
        const auto& peakEvents = output.timing_events[0];

        std::println("PeakExtractor detected {} peaks", peakEvents.size());
        for (const auto& [idx, props] : peakEvents) {
            auto  confIt = props.find(std::pmr::string("confidence"));
            float conf   = confIt != props.end() ? confIt->second.value_or<float>(0.f) : 0.f;
            std::println("  bin={} conf={:.3f}", idx, conf);
        }

        expect(eq(peakEvents.size(), 3UZ)) << "should find 3 peaks";
    };

    "auto-detects model N from output size"_test = [] {
        constexpr std::size_t N = 64;
        constexpr std::size_t R = 8;

        std::vector<float> inferenceOut(N + N * R, 0.f);
        inferenceOut[30] = 0.8f; // one peak in heatmap

        gr::DataSet<float> input;
        input.signal_names      = {"inference_output"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values     = std::move(inferenceOut);
        input.extents           = {static_cast<std::int32_t>(N + N * R)};
        input.meta_information  = {{}};
        input.timing_events     = {{}};

        PeakExtractor block;
        block.regression_channels  = R;
        block.model_n              = 0; // auto-detect
        block.confidence_threshold = 0.5f;

        auto output = block.processOne(std::move(input));

        // auto-detected N = totalOut / (1+R) = 576 / 9 = 64
        expect(eq(output.signal_values.size(), N));
    };

    "empty input passes through"_test = [] {
        PeakExtractor      block;
        gr::DataSet<float> input;
        auto               output = block.processOne(std::move(input));
        expect(output.signal_values.empty());
    };

    "peak properties contain required fields"_test = [] {
        constexpr std::size_t N = 50;
        constexpr std::size_t R = 8;

        std::vector<float> inferenceOut(N + N * R, 0.f);
        inferenceOut[25] = 0.9f;
        // fill regression for peak at 25
        inferenceOut[N + 25 * R + 0] = 0.0f;
        inferenceOut[N + 25 * R + 1] = 1.0f;
        inferenceOut[N + 25 * R + 2] = 0.1f;
        inferenceOut[N + 25 * R + 3] = 0.2f;
        inferenceOut[N + 25 * R + 4] = 0.4f;
        inferenceOut[N + 25 * R + 5] = 0.6f;
        inferenceOut[N + 25 * R + 6] = 0.3f;
        inferenceOut[N + 25 * R + 7] = 0.5f;

        gr::DataSet<float> input;
        input.signal_names      = {"inference_output"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.signal_values     = std::move(inferenceOut);
        input.extents           = {static_cast<std::int32_t>(N + N * R)};
        input.meta_information  = {{}};
        input.timing_events     = {{}};

        PeakExtractor block;
        block.regression_channels  = R;
        block.model_n              = N;
        block.confidence_threshold = 0.5f;

        auto output = block.processOne(std::move(input));

        expect(ge(output.timing_events.size(), 1UZ));
        expect(eq(output.timing_events[0].size(), 1UZ));

        if (!output.timing_events[0].empty()) {
            const auto& props = output.timing_events[0][0].second;
            expect(props.contains(std::pmr::string("confidence")));
            expect(props.contains(std::pmr::string("sigma")));
            expect(props.contains(std::pmr::string("amplitude")));
            expect(props.contains(std::pmr::string("w68")));
            expect(props.contains(std::pmr::string("w96")));
            expect(props.contains(std::pmr::string("w99")));
            expect(props.contains(std::pmr::string("kurtosis")));
        }
    };
};

int main() { /* boost::ut */ }
