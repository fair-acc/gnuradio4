#include <boost/ut.hpp>

#ifndef GR_ONNX_MINIMAL_BUILD
#define GR_ONNX_MINIMAL_BUILD 0
#endif

#include <gnuradio-4.0/onnx/OnnxInference.hpp>

#include <cmath>
#include <print>

using namespace boost::ut;
using namespace gr::blocks::onnx;

const boost::ut::suite<"OnnxInference"> inferenceTests = [] {
    "processOne produces output with correct dimensions"_test = [] {
        OnnxInference block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
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

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("inference_output")));

        std::size_t expectedSize = n + n * block._session.regressionChannels();
        expect(eq(output.signal_values.size(), expectedSize));

        bool allFinite = std::ranges::all_of(output.signal_values, [](float v) { return std::isfinite(v); });
        expect(allFinite) << "output contains NaN or inf";

        block.stop();
        expect(!block._session.isLoaded());
    };

    "processOne with size mismatch resamples input"_test = [] {
        OnnxInference block;
        block.model_path = MODEL_N1024_PATH;
        block.start();

        if (!block._session.isLoaded()) {
            std::println("skip: model not loadable");
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

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(gt(output.signal_values.size(), 0UZ));

        block.stop();
    };

    "processOne passes through empty input"_test = [] {
        OnnxInference block;
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

    "processOne without loaded model passes through"_test = [] {
        OnnxInference      block;
        gr::DataSet<float> input;
        input.signal_names  = {"test"};
        input.signal_values = {1.f, 2.f, 3.f};

        auto output = block.processOne(std::move(input));
        expect(eq(output.signal_names.size(), 1UZ));
        expect(eq(output.signal_names[0], std::string("test")));
    };
};

int main() { /* boost::ut */ }
