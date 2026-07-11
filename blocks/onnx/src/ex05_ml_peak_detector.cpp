// Entry-point example for wiring an ONNX model into GNU Radio 4.0.
//
// Demonstrates the merged OnnxPeakDetector block in two operating modes on a
// deterministic synthetic spectrum:
//   1. single-stage — one forward pass, peaks from timing_events[0]
//   2. cascaded     — detect-subtract-repeat over the model's own `residual`
//                     output, so peaks hidden below the relative gate of the
//                     first pass become detectable once the dominant ones are
//                     stripped
// Each mode ends with a braille chart (spectrum + reference/detected markers)
// and a comparison table of expected vs measured peak parameters.
//
//   ex05_ml_peak_detector [model.onnx|model.ort]   (defaults to the bundled analytic fixture)

#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <print>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kSpectrumSize  = 1024UZ;
constexpr std::size_t kMaxStages     = 4UZ;
constexpr float       kMinProminence = 8.f; // cascade stop: newly found peaks must exceed this many noise sigma

struct ReferencePeak {
    float centre;
    float amplitude;
    float sigma;
};

struct DetectedPeak {
    float centre     = 0.f;
    float amplitude  = 0.f;
    float sigmaLeft  = 0.f;
    float sigmaRight = 0.f;
    float confidence = 0.f;
    float prominence = 0.f; // in noise sigma units
    int   stage      = 0;
};

// two strong peaks plus two weak ones that fall below the relative gate of a single pass
constexpr std::array kReferencePeaks{ReferencePeak{150.3f, 5.0f, 4.0f}, ReferencePeak{420.7f, 3.0f, 12.0f}, ReferencePeak{700.0f, 1.5f, 25.0f}, ReferencePeak{860.5f, 1.0f, 6.0f}};

gr::DataSet<float> makeSpectrum(std::span<const ReferencePeak> peaks, std::uint32_t seed = 42U) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(kSpectrumSize)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};

    std::mt19937                    rng(seed);
    std::normal_distribution<float> noise(0.f, 0.01f);
    ds.signal_values.resize(kSpectrumSize);
    for (std::size_t i = 0; i < kSpectrumSize; ++i) {
        ds.signal_values[i] = 0.1f + noise(rng);
        for (const auto& [centre, amplitude, sigma] : peaks) {
            const float x = static_cast<float>(i) - centre;
            ds.signal_values[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
    }
    return ds;
}

float getProp(const gr::property_map& props, const char* key) {
    auto it = props.find(std::pmr::string(key));
    return it != props.end() ? it->second.value_or<float>(0.f) : 0.f;
}

std::vector<DetectedPeak> peaksOf(const gr::DataSet<float>& output, int stage) {
    std::vector<DetectedPeak> peaks;
    if (output.timing_events.empty()) {
        return peaks;
    }
    for (const auto& [idx, props] : output.timing_events[0]) {
        peaks.push_back(DetectedPeak{
            .centre     = getProp(props, "centre"),
            .amplitude  = getProp(props, "amplitude"),
            .sigmaLeft  = getProp(props, "sigma_left"),
            .sigmaRight = getProp(props, "sigma_right"),
            .confidence = getProp(props, "confidence"),
            .prominence = getProp(props, "prominence"),
            .stage      = stage,
        });
    }
    return peaks;
}

// signal `index` of a multi-signal DataSet (0 Spectrum, 1 Heatmap, 2 Reconstruction, 3 Residual)
std::span<const float> signalOf(const gr::DataSet<float>& ds, std::size_t index) {
    const std::size_t n = static_cast<std::size_t>(ds.extents.empty() ? 0 : ds.extents[0]);
    return {ds.signal_values.data() + index * n, n};
}

// detect-subtract-repeat: feed the model's own residual back until nothing prominent is left
std::vector<DetectedPeak> runCascade(gr::blocks::onnx::OnnxPeakDetector& detector, const gr::DataSet<float>& input) {
    std::vector<DetectedPeak> committed;
    gr::DataSet<float>        stageInput = input;

    for (std::size_t stage = 0; stage < kMaxStages; ++stage) {
        const auto output     = detector.processOne(stageInput);
        bool       anyNewPeak = false;

        for (const DetectedPeak& peak : peaksOf(output, static_cast<int>(stage) + 1)) {
            if (peak.prominence < kMinProminence) {
                continue; // noise bump rescaled to a high score by the residual renormalisation
            }
            const bool duplicate = std::ranges::any_of(committed, [&peak](const DetectedPeak& c) { return std::abs(c.centre - peak.centre) < std::max(3.f, c.sigmaLeft); });
            if (duplicate) {
                continue; // subtraction leftover of an already-committed peak
            }
            committed.push_back(peak);
            anyNewPeak = true;
        }
        if (!anyNewPeak) {
            break;
        }

        // next stage operates on the residual (signal 3) with all kept peaks stripped in-graph
        const auto residual = signalOf(output, 3UZ);
        stageInput.signal_values.assign(residual.begin(), residual.end());
    }

    std::ranges::sort(committed, {}, &DetectedPeak::centre);
    return committed;
}

void drawChart(std::string_view title, const gr::DataSet<float>& input, std::span<const DetectedPeak> detected) {
    const std::size_t  n = kSpectrumSize;
    std::vector<float> xAxis(n);
    std::iota(xAxis.begin(), xAxis.end(), 0.f);
    std::span<const float> spectrum(input.signal_values.data(), n);

    const auto [minIt, maxIt] = std::ranges::minmax_element(spectrum);
    const float margin        = (*maxIt - *minIt) * 0.1f;

    auto chart        = gr::graphs::ImChart<130UZ, 28UZ>({{0.0, static_cast<double>(n)}, {*minIt - margin, *maxIt + margin}});
    chart.axis_name_x = "frequency [bin]";
    chart.axis_name_y = std::format("{} [a.u.]", title);

    chart.draw(xAxis, spectrum, "spectrum");

    auto markerSeries = [n, spectrum](auto&& centres) {
        std::vector<float> markers(n, std::numeric_limits<float>::quiet_NaN());
        for (float centre : centres) {
            const auto bin = static_cast<std::size_t>(std::clamp(std::lround(centre), 0L, static_cast<long>(n - 1)));
            markers[bin]   = spectrum[bin];
        }
        return markers;
    };
    const auto referenceMarkers = markerSeries(kReferencePeaks | std::views::transform(&ReferencePeak::centre));
    const auto detectedMarkers  = markerSeries(detected | std::views::transform(&DetectedPeak::centre));

    chart.draw<gr::graphs::Style::Marker>(xAxis, referenceMarkers, "reference");
    chart.draw<gr::graphs::Style::Marker>(xAxis, detectedMarkers, "detected");
    chart.draw();
}

void printComparisonTable(std::string_view label, std::span<const DetectedPeak> detected) {
    std::vector<bool> used(detected.size(), false);

    std::println("  {} — expected vs measured:", label);
    std::println("  {:>8} {:>8} {:>7} | {:>7} {:>7} {:>7} | {:>6} {:>11} {:>7} | {:>5}", "pos", "pos'", "Δpos", "amp", "amp'", "Δamp%", "σ", "σL'/σR'", "Δσ%", "stage");

    std::size_t matched = 0;
    for (const ReferencePeak& ref : kReferencePeaks) {
        std::size_t best     = detected.size();
        float       bestDist = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < detected.size(); ++i) {
            const float dist = std::abs(detected[i].centre - ref.centre);
            if (!used[i] && dist < bestDist) {
                bestDist = dist;
                best     = i;
            }
        }
        if (best == detected.size() || bestDist > std::max(3.f, 3.f * ref.sigma)) {
            std::println("  {:>8.1f} {:>8} {:>7} | {:>7.2f} {:>7} {:>7} | {:>6.1f} {:>11} {:>7} | {:>5}  MISS", ref.centre, "-", "-", ref.amplitude, "-", "-", ref.sigma, "-", "-", "-");
            continue;
        }
        used[best] = true;
        ++matched;
        const DetectedPeak& det      = detected[best];
        const float         sigmaAvg = 0.5f * (det.sigmaLeft + det.sigmaRight);
        std::println("  {:>8.1f} {:>8.2f} {:>+7.2f} | {:>7.2f} {:>7.2f} {:>+6.1f}% | {:>6.1f} {:>5.1f}/{:<5.1f} {:>+6.1f}% | {:>5}", //
            ref.centre, det.centre, det.centre - ref.centre,                                                                         //
            ref.amplitude, det.amplitude, 100.f * (det.amplitude - ref.amplitude) / ref.amplitude,                                   //
            ref.sigma, det.sigmaLeft, det.sigmaRight, 100.f * (sigmaAvg - ref.sigma) / ref.sigma, det.stage);
    }

    const std::size_t spurious = static_cast<std::size_t>(std::ranges::count(used, false));
    std::println("  summary: {}/{} reference peaks detected, {} spurious\n", matched, kReferencePeaks.size(), spurious);
}

} // namespace

int main(int argc, char* argv[]) {
    std::string modelPath;
    if (argc > 1) {
        modelPath = argv[1];
    } else {
#ifdef MODEL_PEAKS_FIXTURE_PATH
        modelPath = MODEL_PEAKS_FIXTURE_PATH;
#else
        std::println("no model path provided (pass a peaks-tensor .onnx/.ort as first argument)");
        return 1;
#endif
    }

    std::println("=== OnnxPeakDetector example: single-stage vs cascaded operation ===");
    std::println("model: {}", modelPath);
    std::println("spectrum: {} bins, {} reference peaks, fixed seed\n", kSpectrumSize, kReferencePeaks.size());

    gr::blocks::onnx::OnnxPeakDetector detector;
    detector.model_path     = modelPath;
    detector.gate_threshold = 0.5f;
    detector.start();
    if (!detector.isModelLoaded()) {
        std::println("failed to load model from '{}' — pass a peaks-tensor model as first argument", modelPath);
        return 1;
    }

    const auto input = makeSpectrum(kReferencePeaks);

    // 1. single-stage: one forward pass; only peaks above the relative gate survive
    const auto singleOutput = detector.processOne(input);
    auto       singlePeaks  = peaksOf(singleOutput, 1);
    std::ranges::sort(singlePeaks, {}, &DetectedPeak::centre);

    drawChart("single-stage", input, singlePeaks);
    printComparisonTable("single-stage", singlePeaks);

    // 2. cascaded: strip detected peaks via the model's residual output and re-detect
    const auto cascadePeaks = runCascade(detector, input);

    drawChart("cascaded", input, cascadePeaks);
    printComparisonTable(std::format("cascaded ({} stages max)", kMaxStages), cascadePeaks);

    detector.stop();
    return 0;
}
