// Entry-point example for wiring an ONNX model into GNU Radio 4.0.
//
// Demonstrates OnnxPeakDetector on a deterministic synthetic spectrum in the two shipped
// topologies. Both are single forward passes; they differ in the model, not in host code:
//   single-stage — reports the peaks of one pass
//   cascade      — ten stages unrolled in-graph, each subtracting the one peak it promotes,
//                  so peaks hidden below the first pass's relative gate still surface
// Each ends with a braille chart (spectrum + reference/detected markers) and a comparison
// table of expected vs measured peak parameters.
//
//   ex05_ml_peak_detector [single | model.onnx | model.ort]   (default: the bundled cascade)

#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>

#include "../ModelPath.hpp"

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <print>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kSpectrumSize = 1024UZ;

struct ReferencePeak {
    float centre;
    float amplitude;
    float sigma;
};

// the reference peaks are rendered as Gaussians, so their designed sigma maps exactly onto the
// half-maximum width the detectors report
[[nodiscard]] constexpr float referenceFwhm(const ReferencePeak& peak) noexcept { return 2.f * peak.sigma / gr::blocks::fourier::kGaussianHalfMaxToSigma; }

struct DetectedPeak {
    float centre     = 0.f;
    float amplitude  = 0.f;
    float hwhmLeft   = 0.f;
    float hwhmRight  = 0.f;
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

std::vector<DetectedPeak> peaksOf(const gr::DataSet<float>& output) {
    std::vector<DetectedPeak> peaks;
    if (output.timing_events.empty()) {
        return peaks;
    }
    for (const auto& [idx, props] : output.timing_events[0]) {
        peaks.push_back(DetectedPeak{
            .centre     = getProp(props, "centre"),
            .amplitude  = getProp(props, "amplitude"),
            .hwhmLeft   = getProp(props, "hwhm_l"),
            .hwhmRight  = getProp(props, "hwhm_r"),
            .confidence = getProp(props, "confidence"),
            .prominence = getProp(props, "prominence"),
            // provenance comes from the model: 0 for a single pass, otherwise the promoting stage
            .stage = static_cast<int>(getProp(props, "type_tag")),
        });
    }
    return peaks;
}

void drawChart(std::string_view title, const gr::DataSet<float>& input, std::span<const DetectedPeak> detected) {
    const std::size_t  n = kSpectrumSize;
    std::vector<float> xAxis(n);
    std::iota(xAxis.begin(), xAxis.end(), 0.f);
    std::span<const float> spectrum(input.signal_values.data(), n);

    const auto [minIt, maxIt] = std::ranges::minmax_element(spectrum);
    const float margin        = (*maxIt - *minIt) * 0.1f;

    auto chart        = gr::graphs::ImChart<96UZ, 28UZ>({{0.0, static_cast<double>(n)}, {*minIt - margin, *maxIt + margin}});
    chart.axis_name_x = "frequency [bin]";
    chart.axis_name_y = std::format("{} [a.u.]", title);

    chart.draw(xAxis, spectrum, "spectrum");

    auto markerSeries = [spectrum](auto&& centres) {
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
    std::println("  {:>8} {:>8} {:>7} | {:>7} {:>7} {:>7} | {:>6} {:>11} {:>7} | {:>5}", "pos", "pos'", "Δpos", "amp", "amp'", "Δamp%", "fwhm", "hwhmL'/R'", "Δfwhm%", "stage");

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
            std::println("  {:>8.1f} {:>8} {:>7} | {:>7.2f} {:>7} {:>7} | {:>6.1f} {:>11} {:>7} | {:>5}  MISS", ref.centre, "-", "-", ref.amplitude, "-", "-", referenceFwhm(ref), "-", "-", "-");
            continue;
        }
        used[best] = true;
        ++matched;
        const DetectedPeak& det     = detected[best];
        const float         detFwhm = det.hwhmLeft + det.hwhmRight;
        const float         refFwhm = referenceFwhm(ref);
        std::println("  {:>8.1f} {:>8.2f} {:>+7.2f} | {:>7.2f} {:>7.2f} {:>+6.1f}% | {:>6.1f} {:>5.1f}/{:<5.1f} {:>+6.1f}% | {:>5}", //
            ref.centre, det.centre, det.centre - ref.centre,                                                                         //
            ref.amplitude, det.amplitude, 100.f * (det.amplitude - ref.amplitude) / ref.amplitude,                                   //
            refFwhm, det.hwhmLeft, det.hwhmRight, 100.f * (detFwhm - refFwhm) / refFwhm, det.stage);
    }

    const std::size_t spurious = static_cast<std::size_t>(std::ranges::count(used, false));
    std::println("  summary: {}/{} reference peaks detected, {} spurious\n", matched, kReferencePeaks.size(), spurious);
}

} // namespace

int main(int argc, char* argv[]) {
    // the cascade is a SECOND model with ten stages unrolled in-graph, never a host-side loop: each
    // stage subtracts the one peak it promotes, or passes its input through unchanged when nothing
    // is confidently detected. Both models are run so the two tables are genuinely different results.
    const bool modelFromArgv = argc > 1;
    const auto input         = makeSpectrum(kReferencePeaks);

    const std::array<std::pair<std::string_view, std::string>, 2> topologies{{
        {"single-stage", gr::blocks::onnx::test::deliverableModelPath("ex05_peak_detector_single_stage")},
        {"cascaded (10 stages, in-graph)", gr::blocks::onnx::test::deliverableModelPath("ex05_peak_detector_cascaded")},
    }};

    std::println("=== OnnxPeakDetector example: single-stage vs cascaded operation ===");
    std::println("spectrum: {} bins, {} reference peaks, fixed seed\n", kSpectrumSize, kReferencePeaks.size());

    for (const auto& [topologyLabel, defaultPath] : topologies) {
        const std::string                  path  = modelFromArgv ? std::string(argv[1]) : defaultPath;
        const std::string_view             label = modelFromArgv ? std::string_view("model from argv[1]") : topologyLabel;
        gr::blocks::onnx::OnnxPeakDetector detector;
        detector.model_path     = path;
        detector.gate_threshold = 0.5f;
        detector.start();
        if (!detector.isModelLoaded()) {
            std::println("failed to load model from '{}' -- pass a peaks-tensor model as first argument", path);
            return 1;
        }
        auto peaks = peaksOf(detector.processOne(input));
        std::ranges::sort(peaks, {}, &DetectedPeak::centre);
        drawChart(label, input, peaks);
        printComparisonTable(label, peaks);
        detector.stop();
        if (modelFromArgv) {
            break; // an explicit model replaces the pair, so run it once rather than twice
        }
    }
    return 0;
}
