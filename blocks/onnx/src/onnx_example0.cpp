#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>
#include <gnuradio-4.0/onnx/PeakDetector.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <vector>

namespace {

struct GroundTruthPeak {
    float center, amplitude, sigma;
};

gr::DataSet<float> generateSpectrum(std::size_t n, std::span<const GroundTruthPeak> peaks) {
    gr::DataSet<float> ds;
    ds.axis_names = {"Frequency"};
    ds.axis_units = {"bin"};
    ds.axis_values.resize(1);
    ds.axis_values[0].resize(n);
    std::iota(ds.axis_values[0].begin(), ds.axis_values[0].end(), 0.f);

    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {"magnitude"};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(n)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};

    ds.signal_values.resize(n, 0.1f);

    for (const auto& [center, amplitude, sigma] : peaks) {
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - center;
            ds.signal_values[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
        ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(center), gr::property_map{
                                                                                  {std::pmr::string("confidence"), gr::pmt::Value(1.0f)},
                                                                                  {std::pmr::string("sigma"), gr::pmt::Value(sigma)},
                                                                                  {std::pmr::string("amplitude"), gr::pmt::Value(amplitude)},
                                                                                  {std::pmr::string("source"), gr::pmt::Value(std::pmr::string("ground_truth"))},
                                                                              });
    }

    const auto [minIt, maxIt] = std::ranges::minmax_element(ds.signal_values);
    ds.signal_ranges[0]       = {*minIt, *maxIt};

    return ds;
}

void printPeaks(std::string_view label, const std::vector<gr::DataSet<float>::idx_pmt_map>& peaks) {
    std::println("  {} detected {} peaks:", label, peaks.size());
    for (const auto& [idx, props] : peaks) {
        auto  confIt = props.find(std::pmr::string("confidence"));
        auto  sigIt  = props.find(std::pmr::string("sigma"));
        float conf   = confIt != props.end() ? confIt->second.value_or<float>(0.f) : 0.f;
        float sigma  = sigIt != props.end() ? sigIt->second.value_or<float>(0.f) : 0.f;
        std::println("    bin={:4d}  confidence={:8.2f}  sigma={:.1f}", idx, conf, sigma);
    }
}

float toDb(float v) { return 20.f * std::log10(std::max(v, 1e-10f)); }

void drawPeakComparison(std::string_view title, const gr::DataSet<float>& input, const gr::DataSet<float>& output, std::span<const GroundTruthPeak> gtPeaks) {
    const auto  n             = static_cast<std::size_t>(input.extents[0]);
    const auto& xAxis         = input.axis_values[0];
    auto        inputSpectrum = input.signalValues(0);

    // convert to dB
    std::vector<float> spectrumDb(n);
    std::ranges::transform(inputSpectrum, spectrumDb.begin(), toDb);

    float yMin   = *std::ranges::min_element(spectrumDb);
    float yMax   = *std::ranges::max_element(spectrumDb);
    float margin = (yMax - yMin) * 0.1f;

    auto chart        = gr::graphs::ImChart<std::dynamic_extent, std::dynamic_extent>({{0.0, static_cast<double>(n)}, {yMin - margin, yMax + margin}}, 130UZ, 28UZ);
    chart.axis_name_x = "Frequency [bin]";
    chart.axis_name_y = std::format("{} [dB]", title);

    chart.draw(xAxis, spectrumDb, "spectrum");

    // ground-truth markers
    if (!input.timing_events.empty()) {
        std::vector<float> gtMarkers(n, std::numeric_limits<float>::quiet_NaN());
        for (const auto& [idx, props] : input.timing_events[0]) {
            if (idx >= 0 && static_cast<std::size_t>(idx) < n) {
                gtMarkers[static_cast<std::size_t>(idx)] = spectrumDb[static_cast<std::size_t>(idx)];
            }
        }
        chart.draw<gr::graphs::Style::Marker>(xAxis, gtMarkers, "ground truth");
    }

    // detected peak markers
    if (!output.timing_events.empty() && !output.timing_events[0].empty()) {
        std::vector<float> detMarkers(n, std::numeric_limits<float>::quiet_NaN());
        for (const auto& [idx, props] : output.timing_events[0]) {
            if (idx >= 0 && static_cast<std::size_t>(idx) < n) {
                detMarkers[static_cast<std::size_t>(idx)] = spectrumDb[std::min(static_cast<std::size_t>(idx), n - 1)];
            }
        }
        chart.draw<gr::graphs::Style::Marker>(xAxis, detMarkers, "detected");
    }

    chart.draw();

    // numerical comparison: match detected peaks to ground truth
    if (!output.timing_events.empty() && !output.timing_events[0].empty()) {
        std::println("  Peak matching ({}):", title);
        std::println("  {:>8} {:>8} {:>10} {:>10} {:>8}", "GT bin", "Det bin", "Δ bin", "GT σ", "Det σ");
        for (const auto& gt : gtPeaks) {
            float bestDist   = std::numeric_limits<float>::max();
            float bestDetBin = -1.f;
            float bestDetSig = 0.f;
            for (const auto& [idx, props] : output.timing_events[0]) {
                float dist = std::abs(static_cast<float>(idx) - gt.center);
                if (dist < bestDist) {
                    bestDist   = dist;
                    bestDetBin = static_cast<float>(idx);
                    auto sigIt = props.find(std::pmr::string("sigma"));
                    bestDetSig = sigIt != props.end() ? sigIt->second.value_or<float>(0.f) : 0.f;
                }
            }
            bool matched = bestDist < 3.f * gt.sigma;
            std::println("  {:8.0f} {:8.0f} {:>+10.1f} {:10.1f} {:8.1f} {}", gt.center, bestDetBin, bestDetBin - gt.center, gt.sigma, bestDetSig, matched ? "" : "MISS");
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    constexpr std::size_t n = 1024;
    std::println("=== ONNX Peak Detection Example ===");
    std::println("spectrum size: {} bins, 3 known peaks at bins 200, 500, 800\n", n);

    std::array groundTruth = {GroundTruthPeak{200.f, 50.f, 5.f}, GroundTruthPeak{500.f, 80.f, 8.f}, GroundTruthPeak{800.f, 30.f, 3.f}};
    auto       spectrum    = generateSpectrum(n, groundTruth);

    // classical peak detector
    {
        gr::blocks::onnx::PeakDetector detector;
        detector.prominence_threshold = 3.0f;
        detector.min_peak_distance    = 5;
        detector.max_peaks            = 8;

        auto output = detector.processOne(gr::DataSet<float>(spectrum));
        printPeaks("Classical", output.timing_events[0]);
        drawPeakComparison("Classical", spectrum, output, groundTruth);
        std::println();
    }

    // ML peak detector
    {
        std::string modelPath;
        if (argc > 1) {
            modelPath = argv[1];
        } else {
#ifdef MODEL_N1024_PATH
            modelPath = MODEL_N1024_PATH;
#else
            std::println("  ML: no model path provided (pass as first argument)");
            return 0;
#endif
        }

        gr::blocks::onnx::OnnxPeakDetector detector;
        detector.model_path           = modelPath;
        detector.confidence_threshold = 0.3f;
        detector.min_peak_distance    = 8;
        detector.max_peaks            = 8;
        detector.start();

        if (!detector._session.isLoaded()) {
            std::println("  ML: failed to load model from '{}'", modelPath);
            return 1;
        }

        auto output = detector.processOne(gr::DataSet<float>(spectrum));
        printPeaks("ML (ONNX)", output.timing_events[0]);
        drawPeakComparison("ML (ONNX)", spectrum, output, groundTruth);
        detector.stop();
    }

    return 0;
}
