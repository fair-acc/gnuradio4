#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>
#include <gnuradio-4.0/onnx/PeakDetector.hpp>
#include <gnuradio-4.0/testing/EvolvingPeakSpectrum.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <print>
#include <vector>

namespace {

constexpr float       kMatchRadiusSigma = 3.0f;
constexpr std::size_t kNumSpectra       = 200;

struct DetectionEvent {
    std::uint32_t peakId;
    std::size_t   onsetSpectrum;
    std::size_t   firstDetectedSpectrum = std::numeric_limits<std::size_t>::max();
    float         trueCenter            = 0.f;
    float         detectedCenter        = 0.f;
    float         trueSigma             = 0.f;
    float         detectedSigma         = 0.f;
    float         trueAmplitude         = 0.f;
    float         detectedConfidence    = 0.f;
    bool          everDetected          = false;
};

struct LatencyStats {
    std::string              name;
    std::vector<std::size_t> latencies;
    std::size_t              falsePositives  = 0;
    std::size_t              falseNegatives  = 0;
    std::size_t              totalGtPeaks    = 0;
    std::size_t              totalDetections = 0;
    float                    positionMae     = 0.f;
    float                    widthMae        = 0.f;
};

float getProp(const gr::property_map& props, const char* key, float fallback = 0.f) {
    auto it = props.find(std::pmr::string(key));
    return (it != props.end()) ? it->second.value_or<float>(std::move(fallback)) : fallback;
}

std::string getStringProp(const gr::property_map& props, const char* key) {
    auto it = props.find(std::pmr::string(key));
    return (it != props.end()) ? std::string(it->second.value_or<std::pmr::string>(std::pmr::string(""))) : "";
}

LatencyStats runDetector(auto& detector, gr::testing::EvolvingPeakSpectrum<float>& generator, std::string_view name) {
    generator.start();
    LatencyStats stats{.name = std::string(name)};

    std::map<std::uint32_t, DetectionEvent> peakTracker;
    std::size_t                             totalFp   = 0;
    float                                   posErrSum = 0.f, widthErrSum = 0.f;
    std::size_t                             matchCount = 0;

    for (std::size_t specIdx = 0; specIdx < kNumSpectra; ++specIdx) {
        std::uint8_t       tick = 1;
        gr::DataSet<float> genOut;
        std::ignore = generator.processBulk(std::span<const std::uint8_t>(&tick, 1), std::span<gr::DataSet<float>>(&genOut, 1));

        // collect ground-truth peaks from generator output
        struct GtPeak {
            std::uint32_t id;
            float         center, sigma, amplitude;
            std::string   event;
        };
        std::vector<GtPeak> gtPeaks;
        if (!genOut.timing_events.empty()) {
            for (const auto& [idx, props] : genOut.timing_events[0]) {
                auto   peakIdIt = props.find(std::pmr::string("peak_id"));
                auto   peakId   = peakIdIt != props.end() ? static_cast<std::uint32_t>(peakIdIt->second.value_or<std::int32_t>(std::int32_t(-1))) : 0U;
                GtPeak gp{
                    .id        = peakId,
                    .center    = getProp(props, "center"),
                    .sigma     = getProp(props, "sigma"),
                    .amplitude = getProp(props, "target_amplitude"),
                    .event     = getStringProp(props, "event"),
                };
                gtPeaks.push_back(gp);

                if (!peakTracker.contains(gp.id)) {
                    peakTracker[gp.id] = DetectionEvent{
                        .peakId        = gp.id,
                        .onsetSpectrum = specIdx,
                        .trueCenter    = gp.center,
                        .trueSigma     = gp.sigma,
                        .trueAmplitude = gp.amplitude,
                    };
                }
            }
        }

        auto output = detector.processOne(std::move(genOut));

        // match detected peaks to ground truth
        std::vector<bool> gtMatched(gtPeaks.size(), false);

        if (!output.timing_events.empty()) {
            for (const auto& [detIdx, detProps] : output.timing_events[0]) {
                float detCenter = static_cast<float>(detIdx);
                float detConf   = getProp(detProps, "confidence");
                float detSigma  = getProp(detProps, "sigma");

                bool matched = false;
                for (std::size_t gi = 0; gi < gtPeaks.size(); ++gi) {
                    if (gtMatched[gi]) {
                        continue;
                    }
                    const auto& gp          = gtPeaks[gi];
                    float       matchRadius = std::max(kMatchRadiusSigma * gp.sigma, 10.f);
                    if (std::abs(detCenter - gp.center) < matchRadius) {
                        gtMatched[gi] = true;
                        matched       = true;

                        if (peakTracker.contains(gp.id)) {
                            auto& de = peakTracker[gp.id];
                            if (!de.everDetected) {
                                de.firstDetectedSpectrum = specIdx;
                                de.detectedCenter        = detCenter;
                                de.detectedSigma         = detSigma;
                                de.detectedConfidence    = detConf;
                                de.everDetected          = true;
                            }
                            posErrSum += std::abs(detCenter - gp.center);
                            widthErrSum += std::abs(detSigma - gp.sigma);
                            ++matchCount;
                        }
                        break;
                    }
                }
                if (!matched) {
                    ++totalFp;
                }
            }
            stats.totalDetections += output.timing_events[0].size();
        }
    }

    // compute stats
    for (const auto& [id, de] : peakTracker) {
        ++stats.totalGtPeaks;
        if (de.everDetected) {
            stats.latencies.push_back(de.firstDetectedSpectrum - de.onsetSpectrum);
        } else {
            ++stats.falseNegatives;
        }
    }
    stats.falsePositives = totalFp;
    stats.positionMae    = matchCount > 0 ? posErrSum / static_cast<float>(matchCount) : 0.f;
    stats.widthMae       = matchCount > 0 ? widthErrSum / static_cast<float>(matchCount) : 0.f;

    generator.reset();
    return stats;
}

void printStats(const LatencyStats& stats) {
    std::println("  {}", stats.name);
    std::println("    Ground-truth peaks: {}", stats.totalGtPeaks);
    std::println("    Total detections:   {}", stats.totalDetections);

    if (!stats.latencies.empty()) {
        auto sorted = stats.latencies;
        std::ranges::sort(sorted);
        std::size_t median = sorted[sorted.size() / 2];
        std::size_t p90    = sorted[sorted.size() * 9 / 10];
        std::println("    Median latency:     {} spectra", median);
        std::println("    90th pct latency:   {} spectra", p90);
    }

    float fpRate = stats.totalDetections > 0 ? 100.f * static_cast<float>(stats.falsePositives) / static_cast<float>(stats.totalDetections) : 0.f;
    float fnRate = stats.totalGtPeaks > 0 ? 100.f * static_cast<float>(stats.falseNegatives) / static_cast<float>(stats.totalGtPeaks) : 0.f;
    std::println("    False positive rate: {:.1f}%", fpRate);
    std::println("    False negative rate: {:.1f}%", fnRate);
    std::println("    Position MAE:       {:.1f} bins", stats.positionMae);
    std::println("    Width MAE:          {:.1f} bins", stats.widthMae);
}

void drawLatencyHistogram(const LatencyStats& classical, const LatencyStats& ml) {
    constexpr std::size_t kMaxBin = 50;

    auto makeHisto = [](const std::vector<std::size_t>& latencies, std::size_t maxBin) {
        std::vector<float> bins(maxBin + 1, 0.f);
        for (auto l : latencies) {
            bins[std::min(l, maxBin)] += 1.f;
        }
        return bins;
    };

    std::vector<float> xVals(kMaxBin + 1);
    std::iota(xVals.begin(), xVals.end(), 0.f);

    auto hClassical = makeHisto(classical.latencies, kMaxBin);
    auto hMl        = makeHisto(ml.latencies, kMaxBin);

    float yMax = std::max(*std::ranges::max_element(hClassical), *std::ranges::max_element(hMl));

    auto chart        = gr::graphs::ImChart<std::dynamic_extent, std::dynamic_extent>({{0.0, static_cast<double>(kMaxBin)}, {0.0, static_cast<double>(yMax + 1)}}, 130UZ, 20UZ);
    chart.axis_name_x = "Detection latency [spectra]";
    chart.axis_name_y = "Count";
    chart.draw<gr::graphs::Style::Bars>(xVals, hClassical, "Classical");
    chart.draw<gr::graphs::Style::Bars>(xVals, hMl, "ML (ONNX)");
    chart.draw();
}

} // namespace

int main(int argc, char* argv[]) {
    std::println("=== Detection Latency Benchmark ===");
    std::println("{} spectra, peaks fade in/out with drift\n", kNumSpectra);

    gr::testing::EvolvingPeakSpectrum<float> generator;
    generator.spectrum_size          = 1024;
    generator.seed                   = 42;
    generator.max_concurrent_peaks   = 5;
    generator.peak_spawn_probability = 0.3f;
    generator.min_onset_spectra      = 1;
    generator.max_onset_spectra      = 10;
    generator.min_steady_spectra     = 10;
    generator.max_steady_spectra     = 30;
    generator.min_decay_spectra      = 1;
    generator.max_decay_spectra      = 10;
    generator.max_drift_rate         = 0.3f;
    generator.tag_mode               = gr::testing::TagMode::everySpectrum;

    // classical detector
    gr::blocks::onnx::PeakDetector classical;
    classical.prominence_threshold = 3.0f;
    classical.min_peak_distance    = 5;
    classical.max_peaks            = 8;
    auto classicalStats            = runDetector(classical, generator, "Classical (PeakDetector)");

    // ML detector
    std::string modelPath;
    if (argc > 1) {
        modelPath = argv[1];
    } else {
#ifdef MODEL_N1024_PATH
        modelPath = MODEL_N1024_PATH;
#else
        std::println("ML: no model path (pass as argument)");
        printStats(classicalStats);
        return 0;
#endif
    }

    gr::blocks::onnx::OnnxPeakDetector mlDetector;
    mlDetector.model_path           = modelPath;
    mlDetector.confidence_threshold = 0.3f;
    mlDetector.min_peak_distance    = 8;
    mlDetector.max_peaks            = 8;
    mlDetector.start();

    if (!mlDetector._session.isLoaded()) {
        std::println("ML: failed to load model from '{}'", modelPath);
        return 1;
    }

    auto mlStats = runDetector(mlDetector, generator, "ML (OnnxPeakDetector)");
    mlDetector.stop();

    std::println("Detection Latency Report ({} spectra, seed=42)", kNumSpectra);
    std::println("{:─<60}", "");
    printStats(classicalStats);
    std::println();
    printStats(mlStats);

    std::println("\nLatency distribution:");
    drawLatencyHistogram(classicalStats, mlStats);

    return 0;
}
