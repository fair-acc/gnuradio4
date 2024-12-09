#ifndef GR_ONNX_PEAK_EXTRACTOR_HPP
#define GR_ONNX_PEAK_EXTRACTOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/onnx/PeakExtraction.hpp>

namespace gr::blocks::onnx {

/**
 * @brief Standalone peak extraction post-processing block.
 *
 * Takes raw ONNX inference output (heatmap + regression channels as a single
 * DataSet<float>) and produces a DataSet<float> with the original spectrum,
 * heatmap, and peak annotations in timing_events.
 *
 * This block is the standalone counterpart to the fused OnnxPeakDetector —
 * it can be composed with OnnxInference in a flow graph:
 *   spectrum → OnnxInference → PeakExtractor → output
 *
 * The input DataSet is expected to contain [N + N*R] signal values where:
 *   - first N values = heatmap (confidence per bin)
 *   - remaining N*R values = regression channels (row-major)
 *
 * Port API matches PeakDetector (drop-in replaceable):
 *   in: DataSet<float>, out: DataSet<float>
 *   Output has 2 signals: "Spectrum" (pass-through from metadata) and "Heatmap"
 *   Peaks in timing_events[0] with: confidence, sigma, amplitude, w68, w96, w99, kurtosis
 */
struct PeakExtractor : gr::Block<PeakExtractor> {
    using Description = Doc<"Extracts peaks from ONNX inference output (heatmap + regression).">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    Annotated<float, "confidence threshold">     confidence_threshold = 0.4f;
    Annotated<gr::Size_t, "min peak distance">   min_peak_distance    = 8U;
    Annotated<gr::Size_t, "max peaks">           max_peaks            = 8U;
    Annotated<gr::Size_t, "regression channels"> regression_channels  = 8U;
    Annotated<gr::Size_t, "model N">             model_n              = 0U; // 0 = auto-detect from input

    GR_MAKE_REFLECTABLE(PeakExtractor, in, out, confidence_threshold, min_peak_distance, max_peaks, regression_channels, model_n);

    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) noexcept {
        if (inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t R        = regression_channels;
        const std::size_t totalOut = inData.signal_values.size();

        // determine model N: either from setting or from output structure
        std::size_t N = model_n;
        if (N == 0 && R > 0) {
            // auto-detect: totalOut = N + N*R = N*(1+R)
            N = totalOut / (1 + R);
        }

        if (N == 0 || totalOut < N + N * R || R < 8) {
            return inData;
        }

        std::span<const float> heatmap(inData.signal_values.data(), N);
        std::span<const float> regression(inData.signal_values.data() + N, N * R);

        auto peaks = extractPeaks(heatmap, regression, R, confidence_threshold, min_peak_distance, max_peaks);

        // build output DataSet
        gr::DataSet<float> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        output.signal_names      = {"Heatmap"};
        output.signal_quantities = {""};
        output.signal_units      = {""};
        output.signal_ranges     = {gr::Range<float>{0.f, 1.f}};
        output.extents           = {static_cast<std::int32_t>(N)};

        output.signal_values.assign(heatmap.begin(), heatmap.end());

        output.meta_information = {{}};

        // build peak events
        std::vector<gr::DataSet<float>::idx_pmt_map> peakEvents;
        for (const auto& p : peaks) {
            gr::property_map props{
                {std::pmr::string("confidence"), gr::pmt::Value(p.confidence)},
                {std::pmr::string("sigma"), gr::pmt::Value(p.sigma)},
                {std::pmr::string("amplitude"), gr::pmt::Value(p.amplitude)},
                {std::pmr::string("w68"), gr::pmt::Value(p.w68)},
                {std::pmr::string("w96"), gr::pmt::Value(p.w96)},
                {std::pmr::string("w99"), gr::pmt::Value(p.w99)},
                {std::pmr::string("kurtosis"), gr::pmt::Value(p.kurtosis)},
            };
            peakEvents.emplace_back(static_cast<std::ptrdiff_t>(p.position), std::move(props));
        }
        output.timing_events = {std::move(peakEvents)};

        return output;
    }
};

} // namespace gr::blocks::onnx

inline const auto registerPeakExtractor = gr::registerBlock<gr::blocks::onnx::PeakExtractor>(gr::globalBlockRegistry());

#endif // GR_ONNX_PEAK_EXTRACTOR_HPP
