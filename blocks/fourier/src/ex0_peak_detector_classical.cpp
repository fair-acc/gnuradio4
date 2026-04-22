// Visual demonstrator for the classical PeakDetector.
//
// Wires TickSource -> SyntheticPeakSpectrum -> GroundTruthTap -> PeakDetector -> DetectionSink
// through gr::Graph and gr::scheduler::Simple, then renders the spectrum and both peak sets with
// ImChart: the spectrum as a braille trace, designed (ground-truth) peaks as one marker/colour,
// detected peaks as a different marker/colour, plus a printed designed-vs-detected error table.
//
//   ex0_peak_detector_classical

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/fourier/PeakDetector.hpp>
#include <gnuradio-4.0/testing/SyntheticPeakSpectrum.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <print>
#include <string>
#include <vector>

using namespace gr::blocks::fourier;
using namespace gr::graphs;

namespace {

// Seed 1893 designs 7 well-separated peaks at SyntheticPeakSpectrum's defaults
// (spectrum_size=1024): amplitudes 43.15/17.87/12.35/6.82/3.67/3.11/2.32, minimum separation ~4.6
// designed sigma. At PeakDetector's shipped defaults (see below) the classical detector recovers
// the 4 strongest (amplitude >= 6.82), each matched within 1 bin of its designed centre; the 3
// weakest (amplitude <= 3.67, below the ~5-sigma noise floor at noise_level=1) are missed -- a
// legible worked example of ordinary detection with a visible amplitude cutoff. The companion
// regression test (qa_PeakDetector.cpp) exercises this seed plus the zero-peak seeds (0, 1, 15,
// 30) and the contaminated, blended-broad seed 28066 in detail.
constexpr std::uint64_t kTrainingSeed = 1893ULL;
constexpr gr::Size_t    kSpectrumSize = 1024U;

// PeakDetector's defaults (PeakDetector.hpp): noise_rejection_threshold = min_prominence = 5 -- a
// 2-sigma threshold self-triggers on pure noise (order statistics put ~23 excursions above 2 sigma
// per 1024-bin spectrum; measured false positives on SyntheticPeakSpectrum's zero-peak seeds persist
// up to ~4.5 sigma because of edge-tapered bins, so 5 is the lowest round threshold that clears
// them). No min_amplitude floor is needed on top of that any more -- this example used to carry one
// as a workaround for the old, too-permissive defaults.

struct TickSource : gr::Block<TickSource> {
    using Description = gr::Doc<"Emits a single clock tick to drive one spectrum generation.">;

    gr::PortOut<std::uint8_t> out;

    GR_MAKE_REFLECTABLE(TickSource, out);

    bool _emitted = false;

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        if (_emitted || outSpan.empty()) {
            outSpan.publish(0UZ);
            return gr::work::Status::DONE;
        }
        outSpan[0] = std::uint8_t{0};
        outSpan.publish(1UZ);
        _emitted = true;
        return gr::work::Status::DONE;
    }
};

struct GroundTruthTap : gr::Block<GroundTruthTap> {
    using Description = gr::Doc<"Pass-through that records the designed (ground-truth) spectrum before detection.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    GR_MAKE_REFLECTABLE(GroundTruthTap, in, out);

    gr::DataSet<float> designed;

    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> value) {
        designed = value;
        return value;
    }
};

struct DetectionSink : gr::Block<DetectionSink> {
    using Description = gr::Doc<"Collects the classical detector's output DataSet.">;

    gr::PortIn<gr::DataSet<float>> in;

    GR_MAKE_REFLECTABLE(DetectionSink, in);

    std::vector<gr::DataSet<float>> received;

    void processOne(gr::DataSet<float> value) { received.push_back(std::move(value)); }
};

struct DesignedPeak {
    float       centre;
    float       amplitude;
    float       sigma;
    std::string shape;
};

struct DetectedPeak {
    float centre;
    float amplitudeMeasured;
    float sigma;
};

std::vector<DesignedPeak> extractDesigned(const gr::DataSet<float>& ds) {
    std::vector<DesignedPeak> peaks;
    for (const auto& [idx, props] : ds.timing_events[0]) {
        peaks.push_back({
            .centre    = props.value_or<float>(std::pmr::string("centre"), 0.f),
            .amplitude = props.value_or<float>(std::pmr::string("amplitude"), 0.f),
            .sigma     = props.value_or<float>(std::pmr::string("sigma"), 0.f),
            .shape     = std::string(props.value_or<std::string_view>(std::pmr::string("shape"), std::string_view{})),
        });
    }
    std::ranges::sort(peaks, {}, &DesignedPeak::centre);
    return peaks;
}

std::vector<DetectedPeak> extractDetected(const gr::DataSet<float>& ds) {
    std::vector<DetectedPeak> peaks;
    for (const auto& [idx, props] : ds.timing_events[0]) {
        peaks.push_back({
            .centre            = props.value_or<float>(std::pmr::string("centre"), 0.f),
            .amplitudeMeasured = props.value_or<float>(std::pmr::string("amplitude_measured"), 0.f),
            .sigma             = props.value_or<float>(std::pmr::string("sigma"), 0.f),
        });
    }
    std::ranges::sort(peaks, {}, &DetectedPeak::centre);
    return peaks;
}

// display heuristic for the printed table only (qa_PeakDetector.cpp's regression test pins a tight
// 2-bin bound on the one narrow match it can state unambiguously, and does not assert the broad,
// mutually overlapping ones at all -- see that file for why). Here, narrow high-SNR peaks get a
// tight 2-bin floor; broad/overlapping designed peaks get up to one designed sigma of slack so the
// table can flag a plausible match without claiming sub-bin precision it cannot have.
[[nodiscard]] float positionTolerance(float designedSigma) { return std::max(2.f, designedSigma); }

} // namespace

int main() {
    std::println("=== PeakDetector visual demonstrator (classical iterative prominence stripping) ===");
    std::println("training seed: {} (multi-peak showcase, spectrum_size=1024 defaults), detector at default settings\n", kTrainingSeed);

    gr::Graph graph;
    auto&     source    = graph.emplaceBlock<TickSource>();
    auto&     generator = graph.emplaceBlock<gr::testing::SyntheticPeakSpectrum<float>>({{"seed", kTrainingSeed}, {"spectrum_size", kSpectrumSize}});
    auto&     tap       = graph.emplaceBlock<GroundTruthTap>();
    auto&     detector  = graph.emplaceBlock<PeakDetector>();
    auto&     sink      = graph.emplaceBlock<DetectionSink>();

    const bool connected = graph.connect(source, std::string("out"), generator, std::string("in")).has_value() //
                           && graph.connect(generator, std::string("out"), tap, std::string("in")).has_value() //
                           && graph.connect(tap, std::string("out"), detector, std::string("in")).has_value()  //
                           && graph.connect(detector, std::string("out"), sink, std::string("in")).has_value();
    if (!connected) {
        std::println("failed to connect the graph");
        return 1;
    }

    gr::scheduler::Simple scheduler;
    if (auto exchanged = scheduler.exchange(std::move(graph)); !exchanged.has_value()) {
        std::println("scheduler exchange failed: {}", exchanged.error().message);
        return 1;
    }
    if (auto result = scheduler.runAndWait(); !result.has_value()) {
        std::println("scheduler run failed: {}", result.error().message);
        return 1;
    }
    if (sink.received.empty()) {
        std::println("no detection output produced");
        return 1;
    }

    const auto&            designedDs = tap.designed;
    const auto&            detectedDs = sink.received.front();
    const std::size_t      n          = static_cast<std::size_t>(kSpectrumSize);
    std::span<const float> spectrum(designedDs.signal_values.data(), n);

    const auto designedPeaks = extractDesigned(designedDs);
    const auto detectedPeaks = extractDetected(detectedDs);

    auto sampleAt = [&](float centre) -> double {
        const auto bin = static_cast<std::size_t>(std::clamp(std::lround(centre), 0L, static_cast<long>(n - 1)));
        return static_cast<double>(spectrum[bin]);
    };

    std::vector<double> xAxis(n), ySpectrum(n);
    for (std::size_t i = 0; i < n; ++i) {
        xAxis[i]     = static_cast<double>(i);
        ySpectrum[i] = static_cast<double>(spectrum[i]);
    }

    std::vector<double> xDesigned, yDesigned;
    for (const auto& p : designedPeaks) {
        xDesigned.push_back(p.centre);
        yDesigned.push_back(sampleAt(p.centre));
    }
    std::vector<double> xDetected, yDetected;
    for (const auto& p : detectedPeaks) {
        xDetected.push_back(p.centre);
        yDetected.push_back(sampleAt(p.centre));
    }

    ImChart<140, 36> chart;
    chart.axis_name_x = "frequency bin []";
    chart.axis_name_y = "magnitude [a.u.]";

    chart._lastColor = Color::Type::Blue;                     // ImChart.hpp:212 (public field), Color::Type at ImChart.hpp:41
    chart.draw<Style::Braille>(xAxis, ySpectrum, "spectrum"); // ImChart.hpp:239 draw<Style>, Style::Braille default at ImChart.hpp:105
    if (!xDesigned.empty()) {
        chart._lastColor = Color::Type::LightGreen;
        chart.draw<Style::Marker>(xDesigned, yDesigned, "designed (ground truth)"); // marker glyph kMarker[_n_datasets-1]="O", ImChart.hpp:202,350
    }
    if (!xDetected.empty()) {
        chart._lastColor = Color::Type::LightRed;
        chart.draw<Style::Marker>(xDetected, yDetected, "detected"); // marker glyph kMarker[_n_datasets-1]="★", ImChart.hpp:202,350
    }
    chart.draw(); // draws border/axes/legend and prints the screen, ImChart.hpp:362

    // mutual-nearest-neighbour match: a detection can only be claimed by the designed peak it is
    // itself nearest to, so one detection cannot appear as a "match" against several designed rows
    std::vector<std::size_t> detectionsNearestDesigned(detectedPeaks.size(), designedPeaks.size());
    for (std::size_t j = 0; j < detectedPeaks.size(); ++j) {
        float bestDelta = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < designedPeaks.size(); ++i) {
            const float delta = std::abs(detectedPeaks[j].centre - designedPeaks[i].centre);
            if (delta < bestDelta) {
                bestDelta                    = delta;
                detectionsNearestDesigned[j] = i;
            }
        }
    }

    std::println("\n{:>3}  {:>10} {:>9} {:>8} {:>10}   {:>10} {:>9} {:>8}   {:>8}  {}", "idx", "designed c", "amp", "sigma", "shape", "matched c", "amp_meas", "sigma", "pos err", "match");
    std::size_t matched = 0;
    for (std::size_t i = 0; i < designedPeaks.size(); ++i) {
        const auto& d = designedPeaks[i];

        std::size_t bestJ = detectedPeaks.size();
        for (std::size_t j = 0; j < detectedPeaks.size(); ++j) {
            if (detectionsNearestDesigned[j] != i) {
                continue; // this detection belongs to a different designed peak
            }
            const float delta = std::abs(detectedPeaks[j].centre - d.centre);
            if (delta < positionTolerance(d.sigma)) {
                bestJ = j;
                break;
            }
        }

        if (bestJ < detectedPeaks.size()) {
            const auto& m     = detectedPeaks[bestJ];
            const float delta = std::abs(m.centre - d.centre);
            ++matched;
            std::println("{:>3}  {:>10.2f} {:>9.2f} {:>8.2f} {:>10}   {:>10.2f} {:>9.2f} {:>8.2f}   {:>8.2f}  {}", i, d.centre, d.amplitude, d.sigma, d.shape, m.centre, m.amplitudeMeasured, m.sigma, delta, "yes");
        } else {
            std::println("{:>3}  {:>10.2f} {:>9.2f} {:>8.2f} {:>10}   {:>10} {:>9} {:>8}   {:>8}  {}", i, d.centre, d.amplitude, d.sigma, d.shape, "-", "-", "-", "-", "MISSED");
        }
    }

    std::println("\n{} designed, {} detected, {} designed peaks matched (display heuristic, see positionTolerance)\n", designedPeaks.size(), detectedPeaks.size(), matched);
    return 0;
}
