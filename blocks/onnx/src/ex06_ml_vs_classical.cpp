#include <gnuradio-4.0/fourier/PeakDetector.hpp>
#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>

#include "../ModelPath.hpp"
#include <gnuradio-4.0/testing/SyntheticPeakSpectrum.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr float       kMatchRadiusSigma     = 3.0f;
constexpr float       kMatchRadiusFloorBins = 2.0f; // keeps sub-bin estimates matchable at sigma ~1
constexpr float       kDuplicateRadiusSigma = 3.0f;
constexpr std::size_t kDefaultNumSpectra    = 2000; // spectra=10000 for a high-statistics run

struct MatchRecord {
    float gtFwhm;
    float posErr;
    float detFwhm;
    float gtAmplitude;
    float detAmplitude;
};

struct DetectorStats {
    std::string              name;
    std::vector<MatchRecord> matches;
    std::size_t              fabrications            = 0;
    std::size_t              duplicatesOnMatchedPeak = 0;
    std::size_t              falseNegatives          = 0;
    std::size_t              totalGtPeaks            = 0;
    std::size_t              totalDetections         = 0;
    std::size_t              passthroughFrames       = 0; // frames the detector forwarded unscored
    float                    positionMae             = 0.f;
    float                    widthMae                = 0.f;
};

float getProp(const gr::property_map& props, const char* key, float fallback = 0.f) {
    auto it = props.find(std::pmr::string(key));
    return (it != props.end()) ? it->second.value_or<float>(std::move(fallback)) : fallback;
}

DetectorStats runDetector(auto& detector, gr::testing::SyntheticPeakSpectrum<float>& generator, std::string_view name, std::size_t numSpectra) {
    generator.start();
    DetectorStats stats{.name = std::string(name), .matches = {}};

    std::size_t totalFp = 0, totalDuplicates = 0, matchCount = 0;
    float       posErrSum = 0.f, widthErrSum = 0.f;

    struct GtPeak {
        float centre, fwhm, amplitude;

        // the match and duplicate radii are calibrated in Gaussian-sigma units, so they stay on
        // that scale while the reported widths are the half-maximum ones both detectors publish
        [[nodiscard]] constexpr float gaussianEquivalentSigma() const noexcept { return 0.5f * fwhm * gr::blocks::fourier::kGaussianHalfMaxToSigma; }
    };

    for (std::size_t specIdx = 0; specIdx < numSpectra; ++specIdx) {
        std::uint8_t       tick = 1;
        gr::DataSet<float> genOut;
        std::ignore = generator.processBulk(std::span<const std::uint8_t>(&tick, 1), std::span<gr::DataSet<float>>(&genOut, 1));

        std::vector<GtPeak> gtPeaks;
        if (!genOut.timing_events.empty()) {
            for (const auto& [idx, props] : genOut.timing_events[0]) {
                gtPeaks.push_back({getProp(props, "centre"), getProp(props, "fwhm"), getProp(props, "amplitude")});
            }
        }
        stats.totalGtPeaks += gtPeaks.size();

        auto output = detector.processOne(std::move(genOut));
        if (gr::blocks::onnx::isMarkedPassthrough(output)) {
            stats.passthroughFrames++;
            stats.falseNegatives += gtPeaks.size();
            continue;
        }
        if (output.timing_events.empty()) {
            stats.falseNegatives += gtPeaks.size();
            continue;
        }
        stats.totalDetections += output.timing_events[0].size();

        std::vector<bool> gtMatched(gtPeaks.size(), false);
        for (const auto& [detIdx, detProps] : output.timing_events[0]) {
            // fractional centre keeps the sub-bin accuracy; the event index is an integer bin
            const float detCentre    = getProp(detProps, "centre", static_cast<float>(detIdx));
            const float detFwhm      = getProp(detProps, "fwhm");
            const float detAmplitude = getProp(detProps, "amplitude");

            std::size_t nearest  = gtPeaks.size();
            float       nearestD = std::numeric_limits<float>::max();
            for (std::size_t gi = 0; gi < gtPeaks.size(); ++gi) {
                if (gtMatched[gi]) {
                    continue;
                }
                const float radius = std::max(kMatchRadiusSigma * gtPeaks[gi].gaussianEquivalentSigma(), kMatchRadiusFloorBins);
                const float dist   = std::abs(detCentre - gtPeaks[gi].centre);
                if (dist < radius && dist < nearestD) {
                    nearest  = gi;
                    nearestD = dist;
                }
            }

            if (nearest < gtPeaks.size()) {
                gtMatched[nearest] = true;
                posErrSum += nearestD;
                widthErrSum += std::abs(detFwhm - gtPeaks[nearest].fwhm);
                stats.matches.push_back({.gtFwhm = gtPeaks[nearest].fwhm, .posErr = nearestD, .detFwhm = detFwhm, .gtAmplitude = gtPeaks[nearest].amplitude, .detAmplitude = detAmplitude});
                ++matchCount;
            } else {
                const bool onMatchedPeak = std::ranges::any_of(std::views::iota(0UZ, gtPeaks.size()), [&](std::size_t gi) { return gtMatched[gi] && std::abs(detCentre - gtPeaks[gi].centre) < std::max(kDuplicateRadiusSigma * gtPeaks[gi].gaussianEquivalentSigma(), kMatchRadiusFloorBins); });
                onMatchedPeak ? ++totalDuplicates : ++totalFp;
            }
        }
        stats.falseNegatives += static_cast<std::size_t>(std::ranges::count(gtMatched, false));
    }

    stats.fabrications            = totalFp;
    stats.duplicatesOnMatchedPeak = totalDuplicates;
    stats.positionMae             = matchCount > 0 ? posErrSum / static_cast<float>(matchCount) : 0.f;
    stats.widthMae                = matchCount > 0 ? widthErrSum / static_cast<float>(matchCount) : 0.f;

    generator.reset();
    return stats;
}

constexpr std::array<std::pair<float, float>, 5> kBands{{{0.f, 3.f}, {3.f, 10.f}, {10.f, 40.f}, {40.f, 120.f}, {120.f, 1e9f}}};

[[nodiscard]] std::string bandName(std::pair<float, float> band) { return band.second > 1e8f ? std::format(">{:.0f}", band.first) : std::format("{:.0f}-{:.0f}", band.first, band.second); }

[[nodiscard]] float relativeAmplitudeError(const MatchRecord& m) { return std::abs(m.detAmplitude - m.gtAmplitude) / std::max(m.gtAmplitude, 1e-6f); }

[[nodiscard]] float meanOver(std::span<const MatchRecord> matches, auto&& value) {
    if (matches.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float sum = 0.f;
    for (const MatchRecord& m : matches) {
        sum += value(m);
    }
    return sum / static_cast<float>(matches.size());
}

[[nodiscard]] std::vector<MatchRecord> inBand(const DetectorStats& stats, std::pair<float, float> band) {
    std::vector<MatchRecord> out;
    std::ranges::copy_if(stats.matches, std::back_inserter(out), [band](const MatchRecord& m) { return m.gtFwhm >= band.first && m.gtFwhm < band.second; });
    return out;
}

/// "lower is better" for every metric reported here except the width ratio, which targets 1.0
[[nodiscard]] std::string_view better(float classical, float ml, bool towardsOne = false) {
    const float a = towardsOne ? std::abs(classical - 1.f) : classical;
    const float b = towardsOne ? std::abs(ml - 1.f) : ml;
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) < 1e-6f) {
        return "-";
    }
    return b < a ? "ML" : "classical";
}

void printComparison(const DetectorStats& cl, const DetectorStats& ml, std::size_t numSpectra) {
    auto rate = [](std::size_t part, std::size_t whole) { return whole > 0 ? 100.f * static_cast<float>(part) / static_cast<float>(whole) : 0.f; };

    std::println("Peak detection: classical vs ML   ({} spectra, seed=42, single-frame scoring)\n", numSpectra);
    std::println("Detection quality");
    std::println("  {:<38}{:>12}{:>12}{:>12}", "metric", "classical", "ML", "better");

    auto row = [](std::string_view label, auto a, auto b, std::string_view winner, std::string_view unit = "") { std::println("  {:<38}{:>12}{:>12}{:>12}", label, std::format("{}{}", a, unit), std::format("{}{}", b, unit), winner); };

    // a silently forwarded frame carries the generator's ground truth, not the detector's findings;
    // scoring it would credit the detector with peaks it never produced, so it is counted apart
    if (cl.passthroughFrames > 0UZ || ml.passthroughFrames > 0UZ) {
        std::println("  WARNING: frames forwarded without inference (counted as missed): classical {}, ML {}", cl.passthroughFrames, ml.passthroughFrames);
    }
    row("ground-truth peaks", cl.totalGtPeaks, ml.totalGtPeaks, "-");
    row("total detections", cl.totalDetections, ml.totalDetections, "-");
    row("matched peaks", cl.matches.size(), ml.matches.size(), ml.matches.size() > cl.matches.size() ? "ML" : "classical");

    const float clFn = rate(cl.falseNegatives, cl.totalGtPeaks), mlFn = rate(ml.falseNegatives, ml.totalGtPeaks);
    const float clFp = rate(cl.fabrications, cl.totalDetections), mlFp = rate(ml.fabrications, ml.totalDetections);
    const float clDup = rate(cl.duplicatesOnMatchedPeak, cl.totalDetections), mlDup = rate(ml.duplicatesOnMatchedPeak, ml.totalDetections);
    row("false negatives (missed)", std::format("{:.1f}", clFn), std::format("{:.1f}", mlFn), better(clFn, mlFn), "%");
    row("fabrications (no true peak)", std::format("{:.1f}", clFp), std::format("{:.1f}", mlFp), better(clFp, mlFp), "%");
    row("duplicates (extra hit on matched)", std::format("{:.1f}", clDup), std::format("{:.1f}", mlDup), better(clDup, mlDup), "%");
    row("position MAE [bins]", std::format("{:.2f}", cl.positionMae), std::format("{:.2f}", ml.positionMae), better(cl.positionMae, ml.positionMae));
    row("width MAE [bins]", std::format("{:.2f}", cl.widthMae), std::format("{:.2f}", ml.widthMae), better(cl.widthMae, ml.widthMae));

    const float clAmp = meanOver(cl.matches, relativeAmplitudeError), mlAmp = meanOver(ml.matches, relativeAmplitudeError);
    row("amplitude error [relative]", std::format("{:.3f}", clAmp), std::format("{:.3f}", mlAmp), better(clAmp, mlAmp));

    std::println("\nParameter accuracy by peak width   (classical / ML)");
    std::println("  {:>10}{:>16}{:>18}{:>18}{:>18}{:>18}", "fwhm band", "n", "pos [bins]", "pos [fwhm]", "fwhm_est/gt", "amp [relative]");
    for (const auto& band : kBands) {
        const auto c = inBand(cl, band);
        const auto m = inBand(ml, band);
        if (c.empty() && m.empty()) {
            continue;
        }
        auto pair2 = [](float a, float b) { return std::format("{:.2f} / {:.2f}", a, b); };
        std::println("  {:>10}{:>16}{:>18}{:>18}{:>18}{:>18}", bandName(band), std::format("{} / {}", c.size(), m.size()), //
            pair2(meanOver(c, [](const MatchRecord& r) { return r.posErr; }), meanOver(m, [](const MatchRecord& r) { return r.posErr; })), pair2(meanOver(c, [](const MatchRecord& r) { return r.posErr / std::max(r.gtFwhm, 1e-6f); }), meanOver(m, [](const MatchRecord& r) { return r.posErr / std::max(r.gtFwhm, 1e-6f); })), pair2(meanOver(c, [](const MatchRecord& r) { return r.detFwhm / std::max(r.gtFwhm, 1e-6f); }), meanOver(m, [](const MatchRecord& r) { return r.detFwhm / std::max(r.gtFwhm, 1e-6f); })), pair2(meanOver(c, relativeAmplitudeError), meanOver(m, relativeAmplitudeError)));
    }
    std::println("\n  width ratio targets 1.00; rates are per detection except false negatives, which are per ground-truth peak");
}

} // namespace

// argv comes from a person, so a bad value must report rather than throw out of main
template<typename T>
[[nodiscard]] std::optional<T> parseNumber(std::string_view text) {
    T value{};
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    return ec == std::errc() && ptr == text.data() + text.size() ? std::optional{value} : std::nullopt;
}

int main(int argc, char* argv[]) {
    std::println("=== ML vs classical peak detection ===");

    // ex06 [single|cascade] [spectra=N] [model] [gate] [noise_reject_k] [max_peaks]
    // topology and spectrum count are keywords so either can be given alone; the remaining four stay
    // positional and are consumed in order
    constexpr std::string_view    kSpectraCount  = "spectra=";
    bool                          useSingleStage = false;
    std::size_t                   numSpectra     = kDefaultNumSpectra;
    std::vector<std::string_view> options;
    for (std::string_view arg : std::span<char*>(argv, static_cast<std::size_t>(argc)).subspan(1)) {
        if (arg == "single" || arg == "cascade") {
            useSingleStage = (arg == "single");
        } else if (arg.starts_with(kSpectraCount)) {
            numSpectra = parseNumber<std::size_t>(arg.substr(kSpectraCount.size())).value_or(numSpectra);
        } else {
            options.push_back(arg);
        }
    }
    const auto option = [&options](std::size_t i) -> std::optional<std::string_view> { return i < options.size() ? std::optional(options[i]) : std::nullopt; };
    std::println("{} independent spectra, scored per spectrum\n", numSpectra);

    gr::testing::SyntheticPeakSpectrum<float> generator;
    generator.spectrum_size = 1024;
    generator.seed          = 42;

    // one budget for both detectors -- an asymmetric cap would decide the comparison by itself
    constexpr gr::Size_t kDefaultMaxPeaks = 10U;
    const auto           maxPeaksOption   = option(3);
    const gr::Size_t     maxPeaks         = maxPeaksOption ? parseNumber<gr::Size_t>(*maxPeaksOption).value_or(kDefaultMaxPeaks) : kDefaultMaxPeaks;
    std::println("max peaks: {}{} (both detectors)", maxPeaks, maxPeaksOption ? " (from argv)" : " (compiled-in default)");

    // classical detector — left at its defaults: overriding the noise threshold downwards would
    // buy recall by accepting false positives, which is not the comparison this example is making
    gr::blocks::fourier::PeakDetector classical;
    classical.max_peaks = maxPeaks;
    auto classicalStats = runDetector(classical, generator, "Classical (PeakDetector)", numSpectra);

    // same canonical models and topology switch as ex05, so the two demos cannot drift apart:
    // "single" selects the one-shot detector, otherwise the 10-stage in-graph cascade
    const std::string kTrainedRegressorModel = gr::blocks::onnx::test::deliverableModelPath(useSingleStage ? "ex05_peak_detector_single_stage" : "ex05_peak_detector_cascaded");

    const auto        modelOption = option(0);
    const std::string modelPath   = modelOption ? std::string(*modelOption) : kTrainedRegressorModel;
    // on the peak_present scale of the deliverable; the model's score_output metadata key names
    // the scale in force, and a value tuned for one scale silently over- or under-gates the other
    constexpr float kDefaultGateThreshold = 0.45f;
    const auto      gateOption            = option(1);
    const float     gateThreshold         = gateOption ? parseNumber<float>(*gateOption).value_or(kDefaultGateThreshold) : kDefaultGateThreshold;
    std::println("ML model: {}{}", modelPath, modelOption ? " (from argv)" : " (compiled-in default)");
    std::println("ML gate:  {:.2f}{}", gateThreshold, gateOption ? " (from argv)" : " (compiled-in default)");

    gr::blocks::onnx::OnnxPeakDetector mlDetector;
    mlDetector.model_path     = modelPath;
    mlDetector.gate_threshold = gateThreshold;
    mlDetector.max_peaks      = maxPeaks;
    if (const auto rejectOption = option(2)) {
        if (const auto rejectK = parseNumber<float>(*rejectOption)) {
            mlDetector.model_overrides.value[std::pmr::string("noise_reject_k")] = gr::pmt::Value(*rejectK);
            std::println("ML noise_reject_k: {:.2f} (from argv, overriding the model's baked value)", *rejectK);
        } else {
            std::println("ignoring unparseable noise_reject_k '{}'", *rejectOption);
        }
    }
    mlDetector.start();

    if (!mlDetector.isModelLoaded()) {
        std::println("ML: failed to load model from '{}'", modelPath);
        std::println("    pass a different .onnx/.ort as argv[1] to override");
        return 1;
    }

    auto mlStats = runDetector(mlDetector, generator, "ML (OnnxPeakDetector)", numSpectra);
    mlDetector.stop();

    printComparison(classicalStats, mlStats, numSpectra);

    return 0;
}
