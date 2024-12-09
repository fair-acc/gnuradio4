#ifndef GR_ONNX_INFERENCE_HPP
#define GR_ONNX_INFERENCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/onnx/OnnxSession.hpp>
#include <gnuradio-4.0/onnx/OnnxUtils.hpp>

#include <print>

namespace gr::blocks::onnx {

struct OnnxInference : gr::Block<OnnxInference> {
    using Description = Doc<"Generic ONNX Runtime inference on the first signal of a DataSet.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    Annotated<std::string, "model path"> model_path = "";

    GR_MAKE_REFLECTABLE(OnnxInference, in, out, model_path);

    OnnxSession _session;

    void start() {
        if (model_path.value.empty()) {
            using namespace gr::message;
            this->emitErrorMessage("start()", "model_path is empty");
            this->requestStop();
            return;
        }
        auto result = _session.load(model_path);
        if (!result) {
            using namespace gr::message;
            this->emitErrorMessage("start()", result.error());
            this->requestStop();
        }
    }

    void stop() { _session.reset(); }

    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) noexcept {
        if (!_session.isLoaded() || inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t modelN    = _session.modelN();
        const std::size_t inputSize = inData.signal_values.size() / std::max(1UZ, inData.signal_names.size());

        // resample if input dimension differs from model expectation
        std::vector<float> modelInput(modelN);
        if (inputSize == modelN) {
            std::copy_n(inData.signal_values.begin(), modelN, modelInput.begin());
        } else {
            std::span<const float> firstSignal(inData.signal_values.data(), inputSize);
            resample(firstSignal, modelInput);
        }

        // normalise
        std::vector<float> normalised(modelN);
        normalise(modelInput, normalised);

        auto result = _session.run(normalised);
        if (!result) {
            return inData;
        }

        gr::DataSet<float> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        const std::size_t outSize = result->size();
        output.extents            = {static_cast<std::int32_t>(outSize)};

        output.signal_names      = {"inference_output"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_values     = std::move(*result);
        output.signal_ranges     = {gr::Range<float>{0.f, 0.f}};

        output.meta_information = {{}};
        output.timing_events    = {{}};

        return output;
    }
};

} // namespace gr::blocks::onnx

inline const auto registerOnnxInference = gr::registerBlock<gr::blocks::onnx::OnnxInference>(gr::globalBlockRegistry());

#endif // GR_ONNX_INFERENCE_HPP
