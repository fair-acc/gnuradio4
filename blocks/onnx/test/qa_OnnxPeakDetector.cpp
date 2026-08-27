#include <boost/ut.hpp>

#include <gnuradio-4.0/Compression.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>
#include <gnuradio-4.0/testing/SyntheticPeakSpectrum.hpp>

#include "../ModelPath.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <vector>

using namespace boost::ut;
using namespace gr::blocks::onnx;
using namespace gr::graphs;
using gr::blocks::onnx::test::modelPath;

namespace {

constexpr std::size_t kProps = 13UZ;

// one row of the peaks tensor in the model's declared column order:
// 0 peak_present, 1 centre, 2 amplitude, 3 sigma_left, 4 sigma_right,
// 5 eta, 6 sigma_avg, 7 score, 8 type_tag, 9 is_top1, 10-12 reserved
std::array<float, kProps> makePeakRow(float present, float centre, float amplitude, float sigmaL, float sigmaR, float eta = 0.f) {
    std::array<float, kProps> row{};
    row[0] = present;
    row[1] = centre;
    row[2] = amplitude;
    row[3] = sigmaL;
    row[4] = sigmaR;
    row[5] = eta;
    row[6] = 0.5f * (sigmaL + sigmaR);
    row[7] = present;
    return row;
}

std::vector<float> makePeakTensor(std::span<const std::array<float, kProps>> rows) {
    std::vector<float> flat;
    flat.reserve(rows.size() * kProps);
    for (const auto& row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

struct InjectedPeak {
    float centre;
    float amplitude;
    float sigma;
};

gr::DataSet<float> makeTestSpectrum(std::size_t n, std::span<const InjectedPeak> peaks, float noiseFloor = 0.1f) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(n)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};

    ds.signal_values.resize(n, noiseFloor);
    for (const auto& [centre, amplitude, sigma] : peaks) {
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - centre;
            ds.signal_values[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
    }
    return ds;
}

bool isMarkedPassthrough(const gr::DataSet<float>& ds) {
    if (ds.meta_information.empty()) {
        return false;
    }
    const auto it = ds.meta_information[0].find(std::pmr::string(kPassthroughKey));
    if (it == ds.meta_information[0].end()) {
        return false;
    }
    const bool* flag = it->second.get_if<bool>();
    return flag != nullptr && *flag;
}

float getProp(const gr::property_map& props, const char* key, float fallback = 0.f) {
    auto it = props.find(std::pmr::string(key));
    return it != props.end() ? it->second.value_or<float>(std::move(fallback)) : fallback;
}

[[nodiscard]] bool verboseCharts() { return std::getenv("GR_QA_VERBOSE") != nullptr; }

struct NearestDesignedMatch {
    float delta;
    float gaussianEquivalentSigma;
};

// mirrors blocks/fourier/test/qa_PeakDetector.cpp's nearestDesignedPeak; duplicated since this
// target has no cross-directory include path to share it directly.
[[nodiscard]] NearestDesignedMatch nearestDesigned(const std::vector<gr::DataSet<float>::idx_pmt_map>& designedEvents, float detectedCentre) {
    NearestDesignedMatch best{std::numeric_limits<float>::max(), 0.f};
    for (const auto& [idx, props] : designedEvents) {
        const float centre = getProp(props, "centre");
        const float fwhm   = getProp(props, "fwhm");
        const float delta  = std::abs(detectedCentre - centre);
        if (delta < best.delta) {
            best = {delta, 0.5f * fwhm * gr::blocks::fourier::kGaussianHalfMaxToSigma};
        }
    }
    return best;
}

void printDesignedVsDetectedChart(std::string_view title, std::span<const float> spectrum, std::span<const InjectedPeak> designed, const std::vector<gr::DataSet<float>::idx_pmt_map>& detected) {
    const std::size_t n = spectrum.size();
    if (n == 0) {
        return;
    }

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
    for (const auto& peak : designed) {
        xDesigned.push_back(static_cast<double>(peak.centre));
        yDesigned.push_back(sampleAt(peak.centre));
    }
    std::vector<double> xDetected, yDetected;
    for (const auto& [idx, props] : detected) {
        const float centre = getProp(props, "centre");
        xDetected.push_back(static_cast<double>(centre));
        yDetected.push_back(sampleAt(centre));
    }

    std::println("\n=== {}: designed vs detected ===", title);
    ImChart<140, 36> chart;
    chart.axis_name_x = "frequency bin []";
    chart.axis_name_y = "magnitude [a.u.]";

    chart._lastColor = Color::Type::Blue;
    chart.draw<Style::Braille>(xAxis, ySpectrum, "spectrum");
    if (!xDesigned.empty()) {
        chart._lastColor = Color::Type::LightGreen;
        chart.draw<Style::Marker>(xDesigned, yDesigned, "designed (ground truth)");
    }
    if (!xDetected.empty()) {
        chart._lastColor = Color::Type::LightRed;
        chart.draw<Style::Marker>(xDetected, yDetected, "detected");
    }
    chart.draw();
}

void printDesignedVsDetectedChart(std::string_view title, std::span<const float> spectrum, const std::vector<gr::DataSet<float>::idx_pmt_map>& designedEvents, const std::vector<gr::DataSet<float>::idx_pmt_map>& detected) {
    const std::size_t n = spectrum.size();
    if (n == 0) {
        return;
    }

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
    for (const auto& [idx, props] : designedEvents) {
        const float centre = getProp(props, "centre");
        xDesigned.push_back(static_cast<double>(centre));
        yDesigned.push_back(sampleAt(centre));
    }
    std::vector<double> xDetected, yDetected;
    for (const auto& [idx, props] : detected) {
        const float centre = getProp(props, "centre");
        xDetected.push_back(static_cast<double>(centre));
        yDetected.push_back(sampleAt(centre));
    }

    std::println("\n=== {}: designed vs detected ===", title);
    ImChart<140, 36> chart;
    chart.axis_name_x = "frequency bin []";
    chart.axis_name_y = "magnitude [a.u.]";

    chart._lastColor = Color::Type::Blue;
    chart.draw<Style::Braille>(xAxis, ySpectrum, "spectrum");
    if (!xDesigned.empty()) {
        chart._lastColor = Color::Type::LightGreen;
        chart.draw<Style::Marker>(xDesigned, yDesigned, "designed (ground truth)");
    }
    if (!xDetected.empty()) {
        chart._lastColor = Color::Type::LightRed;
        chart.draw<Style::Marker>(xDetected, yDetected, "detected");
    }
    chart.draw();
}

} // namespace

const boost::ut::suite<"extractPeaksRegressor"> extractionTests = [] {
    "maps the declared column layout onto PeakResult"_test = [] {
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 100.5f, 0.8f, 4.f, 6.f, 0.5f)};
        const auto                                   tensor = makePeakTensor(rows);

        const auto peaks = extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f);

        expect(eq(peaks.size(), 1UZ));
        expect(eq(peaks[0].confidence, 0.9f));
        expect(eq(peaks[0].centre, 100.5f));
        expect(eq(peaks[0].amplitude, 0.8f));
        // the model regresses Gaussian sigma; PeakResult carries the half-maximum widths
        expect(approx(peaks[0].hwhmLeft, 4.f / gr::blocks::fourier::kGaussianHalfMaxToSigma, 1e-4f));
        expect(approx(peaks[0].hwhmRight, 6.f / gr::blocks::fourier::kGaussianHalfMaxToSigma, 1e-4f));
        expect(approx(peaks[0].fwhm(), 10.f / gr::blocks::fourier::kGaussianHalfMaxToSigma, 1e-4f));
        expect(approx(peaks[0].gaussianEquivalentSigma(), 5.f, 1e-4f)) << "the conversion round-trips";
        // this fixture declares only the 13 columns of the older layout, so the widths are measured
        // host-side from the model's own flanks and eta rather than read from the tensor
        const auto expected = gr::blocks::fourier::containmentWidths(4.f, 6.f, 0.5f);
        expect(approx(peaks[0].w68, expected.w68, 1e-4f));
        expect(approx(peaks[0].w96, expected.w96, 1e-4f));
        expect(approx(peaks[0].w99, expected.w99, 1e-4f));
        // eta is surfaced through the model's own kurt = 5·eta mapping
        expect(eq(peaks[0].kurtosis, 2.5f));
    };

    "prefers the model's own containment widths over the Gaussian fallback"_test = [] {
        constexpr std::size_t         kWideProps = 16UZ;
        std::array<float, kWideProps> row{};
        row[0]  = 0.9f;   // peak_present
        row[1]  = 100.5f; // centre
        row[2]  = 0.8f;   // amplitude
        row[3]  = 4.f;    // sigma_left
        row[4]  = 6.f;    // sigma_right
        row[6]  = 5.f;    // sigma_avg
        row[13] = 3.f;    // w68 — deliberately unlike any multiple of sigma_avg
        row[14] = 11.f;   // w96
        row[15] = 47.f;   // w99

        const auto peaks = extractPeaksRegressor(std::span<const float>(row.data(), row.size()), 1UZ, kWideProps, 0.5f);
        expect(eq(peaks.size(), 1UZ));
        expect(eq(peaks[0].w68, 3.f)) << "a model that publishes containment widths must be believed";
        expect(eq(peaks[0].w96, 11.f));
        expect(eq(peaks[0].w99, 47.f));
    };

    "rejects rows below the gate threshold"_test = [] {
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 100.f, 0.8f, 5.f, 5.f), makePeakRow(0.2f, 300.f, 0.4f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);

        expect(eq(extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f).size(), 1UZ));
        expect(eq(extractPeaksRegressor(tensor, rows.size(), kProps, 0.1f).size(), 2UZ));
        expect(eq(extractPeaksRegressor(tensor, rows.size(), kProps, 0.95f).size(), 0UZ));
    };

    "the rescore output supersedes column 0, demoting a peak below the gate threshold"_test = [] {
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 100.f, 0.8f, 5.f, 5.f), makePeakRow(0.9f, 300.f, 0.4f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);
        const std::vector<float>                     rescore{0.8f, 0.1f};

        const auto peaks = extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f, rescore);

        expect(eq(peaks.size(), 1UZ));
        expect(eq(peaks[0].centre, 100.f));
        expect(eq(peaks[0].confidence, 0.8f)) << "confidence must come from the re-scorer, not column 0";
    };

    "results are sorted by centre ascending"_test = [] {
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 700.f, 0.5f, 5.f, 5.f), makePeakRow(0.9f, 100.f, 0.5f, 5.f, 5.f), makePeakRow(0.9f, 400.f, 0.5f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);

        const auto peaks = extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f);

        expect(eq(peaks.size(), 3UZ));
        expect(std::ranges::is_sorted(peaks, {}, &PeakResult::centre));
    };

    "applies no host-side suppression to adjacent peaks"_test = [] {
        // NMS/sigma-dedup already ran upstream in the graph; two peaks one sigma apart must both survive here.
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 500.f, 0.8f, 5.f, 5.f), makePeakRow(0.9f, 505.f, 0.7f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);

        expect(eq(extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f).size(), 2UZ));
    };

    "maxPeaks keeps the highest-scoring peaks"_test = [] {
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.6f, 100.f, 0.5f, 5.f, 5.f), makePeakRow(0.9f, 400.f, 0.5f, 5.f, 5.f), makePeakRow(0.7f, 700.f, 0.5f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);

        const auto peaks = extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f, {}, {}, {0.f, 1.f}, 2UZ);

        expect(eq(peaks.size(), 2UZ));
        expect(eq(peaks[0].centre, 400.f)) << "strongest peak survives the cap";
        expect(eq(peaks[1].centre, 700.f)) << "second-strongest peak survives, sorted by centre";
    };

    "measures amplitude, prominence and uncertainties against a supplied spectrum"_test = [] {
        std::vector<float> spectrum(1024, 0.1f);
        spectrum[200] = 0.9f;
        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 200.f, 0.8f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);

        const auto peaks = extractPeaksRegressor(tensor, rows.size(), kProps, 0.5f, {}, spectrum, NoiseEstimate{0.1f, 0.01f});

        expect(eq(peaks.size(), 1UZ));
        expect(lt(std::abs(peaks[0].amplitudeMeasured - 0.8f), 1e-5f));
        expect(gt(peaks[0].prominence, 0.f)) << "prominence in noise-sigma units";
        expect(gt(peaks[0].positionUncertainty, 0.f));
        expect(gt(peaks[0].widthUncertainty, 0.f));
        expect(gt(peaks[0].amplitudeUncertainty, 0.f));
    };

    "rejects a malformed tensor instead of reading out of bounds"_test = [] {
        const std::vector<float> tooSmall(kProps, 0.f);
        expect(eq(extractPeaksRegressor(tooSmall, 4UZ, kProps, 0.5f).size(), 0UZ));

        const std::vector<std::array<float, kProps>> rows{makePeakRow(0.9f, 100.f, 0.8f, 5.f, 5.f)};
        const auto                                   tensor = makePeakTensor(rows);
        expect(eq(extractPeaksRegressor(tensor, rows.size(), 4UZ, 0.5f).size(), 0UZ)) << "fewer than 10 properties is not the regressor layout";
    };

    "asymmetry is the scale-free log ratio of the flanks"_test = [] {
        expect(approx(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 5.f, .hwhmRight = 5.f}), 0.f, 1e-6f)) << "symmetric peak has zero asymmetry";
        expect(gt(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 4.f, .hwhmRight = 8.f}), 0.f)) << "a wider right flank is positive";
        expect(lt(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 8.f, .hwhmRight = 4.f}), 0.f)) << "a wider left flank is negative";
        expect(approx(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 4.f, .hwhmRight = 8.f}), gr::blocks::fourier::asymmetryOf({.hwhmLeft = 8.f, .hwhmRight = 16.f}), 1e-6f));
        expect(approx(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 4.f, .hwhmRight = 8.f}), -gr::blocks::fourier::asymmetryOf({.hwhmLeft = 8.f, .hwhmRight = 4.f}), 1e-6f));

        expect(eq(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 0.f, .hwhmRight = 5.f}), 0.f)) << "degenerate flank must not produce inf/NaN";
        expect(eq(gr::blocks::fourier::asymmetryOf({.hwhmLeft = 5.f, .hwhmRight = 0.f}), 0.f)) << "degenerate flank must not produce inf/NaN";
    };
};

const boost::ut::suite<"peaks layout validation"> layoutTests = [] {
    "accepts the compiled-in layout and declared supersets"_test = [] {
        expect(matchesPeaksLayout(kPeaksLayoutPrefix));
        expect(matchesPeaksLayout("peak_present,centre,amplitude,sigma_left,sigma_right,eta,sigma_avg,score,type_tag,is_top1,reserved0,reserved1,reserved2"));
        expect(matchesPeaksLayout("")) << "absent metadata (e.g. minimal build) assumes the compiled-in layout";
    };

    "rejects a reordered or renamed layout"_test = [] {
        expect(!matchesPeaksLayout("centre,peak_present,amplitude,sigma_left,sigma_right,eta,sigma_avg"));
        expect(!matchesPeaksLayout("peak_present,center,amplitude,sigma_left,sigma_right,eta,sigma_avg")) << "US spelling must not be accepted";
        expect(!matchesPeaksLayout("peak_present,centre,amplitude"));
    };
};

const boost::ut::suite<"OnnxPeakDetector block"> blockTests = [] {
    "passes input through when no model is loaded"_test = [] {
        OnnxPeakDetector block;

        auto input = makeTestSpectrum(16, {});

        const auto output = block.processOne(input);

        expect(eq(output.signal_values.size(), 16UZ));
        expect(eq(output.signal_names.size(), 1UZ)) << "pass-through must not synthesise the 4-signal output";
        expect(isMarkedPassthrough(output)) << "forwarded frames must carry the onnx_passthrough marker";
    };

    "empty input passes through"_test = [] {
        OnnxPeakDetector block;
        const auto       output = block.processOne(gr::DataSet<float>{});
        expect(eq(output.signal_values.size(), 0UZ));
    };

    "defaults to no input normalisation"_test = [] {
        // peaks-tensor models are trained on native linear amplitude; LogMAD would be garbage in
        OnnxPeakDetector block;
        expect(block.normalise_mode == NormaliseMode::None);
        expect(eq(block.gate_threshold, 0.5f));
        expect(eq(static_cast<std::size_t>(block.max_peaks.value), 8UZ));
    };
};

const boost::ut::suite<"OnnxPeakDetector execution provider"> executionProviderTests = [] {
    "default cpu provider loads the fixture and reports the runtime's provider list"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        expect(eq(block.execution_provider.value, std::string("cpu")));
        expect(std::ranges::contains(block.available_providers.value, "CPUExecutionProvider")) << "available_providers must mirror Ort::GetAvailableProviders()";
        block.stop();
    };

    "unavailable execution provider stops the block instead of silently using cpu"_test = [] {
        if (std::ranges::contains(OnnxSession::availableProviders(), "CUDAExecutionProvider")) {
            std::println("skipped: the linked ONNX Runtime offers CUDA, so the unavailable-provider path cannot be exercised here");
            return;
        }
        OnnxPeakDetector block;
        block.model_path         = modelPath("peaks_fixture_N1024.ort.gz");
        block.execution_provider = "cuda";
        block.start();
        expect(!block.isModelLoaded()) << "an unavailable provider must not fall back to a silently-created cpu session";
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "the failed provider request must stop the block";

        // no session left behind, so this is the same "no model loaded" pass-through as never having set model_path
        const auto output = block.processOne(makeTestSpectrum(1024, {}));
        expect(eq(output.signal_names.size(), 1UZ)) << "the spectrum passes through without peak annotation";
        expect(isMarkedPassthrough(output));
        block.stop();
    };
};

const boost::ut::suite<"OnnxPeakDetector fixture semantics"> fixtureTests = [] {
    "peak injected at bin 200 produces an event near bin 200"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        expect(eq(block._session.modelN(), 1024UZ));

        const InjectedPeak peaks[] = {{200.25f, 5.f, 5.f}};
        const auto         output  = block.processOne(makeTestSpectrum(1024, peaks));

        expect(eq(output.signal_names.size(), 4UZ));
        expect(eq(output.signal_names[0], std::string("Spectrum")));
        expect(eq(output.signal_names[1], std::string("Heatmap")));
        expect(eq(output.signal_names[2], std::string("Reconstruction")));
        expect(eq(output.signal_names[3], std::string("Residual")));
        expect(eq(output.signal_values.size(), 4UZ * 1024UZ));
        expect(!isMarkedPassthrough(output)) << "real inference output must not carry the onnx_passthrough marker";

        const auto& events     = output.timing_events[0];
        const bool  countOk    = events.size() == 1UZ;
        bool        positionOk = true;
        if (countOk) {
            const float c = getProp(events[0].second, "centre");
            positionOk    = std::abs(c - 200.25f) < 0.5f && std::lround(c) == events[0].first && std::abs(getProp(events[0].second, "gaussian_equivalent_sigma") - 5.f) < 1.5f && std::abs(getProp(events[0].second, "amplitude_measured") - 5.f) < 0.5f;
        }
        if (!countOk || !positionOk || verboseCharts()) {
            printDesignedVsDetectedChart("peak injected at bin 200", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 1UZ)) << "exactly one injected peak";
        if (events.empty()) {
            return;
        }
        const auto& [idx, props] = events[0];
        const float centre       = getProp(props, "centre");
        std::println("fixture: idx={} centre={:.2f} fwhm={:.2f} amp_meas={:.2f}", idx, centre, getProp(props, "fwhm"), getProp(props, "amplitude_measured"));

        expect(lt(std::abs(centre - 200.25f), 0.5f)) << "sub-bin centre";
        expect(eq(idx, std::lround(centre))) << "event index is the nearest bin to the fractional centre";
        expect(lt(std::abs(getProp(props, "gaussian_equivalent_sigma") - 5.f), 1.5f)) << "curvature width estimate";
        expect(lt(std::abs(getProp(props, "amplitude_measured") - 5.f), 0.5f)) << "measured amplitude above the noise floor";

        block.stop();
    };

    "fractional centre rounds to the nearest bin, not truncates"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        // peak centred at 300.6: truncation would report bin 300, rounding must give 301
        const InjectedPeak peaks[] = {{300.6f, 5.f, 5.f}};
        const auto         output  = block.processOne(makeTestSpectrum(1024, peaks));

        const auto& events  = output.timing_events[0];
        const bool  countOk = events.size() == 1UZ;
        const bool  roundOk = countOk && std::abs(getProp(events[0].second, "centre") - 300.6f) < 0.4f && events[0].first == std::ptrdiff_t(301);
        if (!countOk || !roundOk || verboseCharts()) {
            printDesignedVsDetectedChart("fractional centre 300.6 rounds to bin 301", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 1UZ));
        if (!events.empty()) {
            const float centre = getProp(events[0].second, "centre");
            expect(lt(std::abs(centre - 300.6f), 0.4f));
            expect(eq(events[0].first, std::ptrdiff_t(301))) << "std::lround(300.6) == 301";
        }

        block.stop();
    };

    "emits every timing-event key of the classical PeakDetector"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        const InjectedPeak peaks[] = {{512.f, 5.f, 6.f}};
        const auto         output  = block.processOne(makeTestSpectrum(1024, peaks));

        expect(ge(output.timing_events[0].size(), 1UZ));
        if (!output.timing_events[0].empty()) {
            const auto& props = output.timing_events[0][0].second;
            for (const char* key : {"confidence", "hwhm_l", "hwhm_r", "asymmetry", "amplitude", "amplitude_measured", "prominence", "isolation", "w68", "w96", "w99", "kurtosis", "noise_sigma", "noise_floor", "position_uncertainty", "width_uncertainty", "amplitude_uncertainty", "centre", "fwhm", "gaussian_equivalent_sigma"}) {
                expect(props.contains(std::pmr::string(key))) << "missing key: " << key;
            }
            // asymmetry must agree with the flanks it is derived from, not be an independent guess
            const float sL = props.value_or<float>(std::pmr::string("hwhm_l"), 0.f);
            const float sR = props.value_or<float>(std::pmr::string("hwhm_r"), 0.f);
            if (sL > 0.f && sR > 0.f) {
                expect(approx(props.value_or<float>(std::pmr::string("asymmetry"), -99.f), 0.5f * std::log(sR / sL), 1e-5f));
            }
        }

        block.stop();
    };

    "gate threshold rejects weak peaks"_test = [] {
        OnnxPeakDetector block;
        block.model_path     = modelPath("peaks_fixture_N1024.ort.gz");
        block.gate_threshold = 0.5f;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        // fixture score is the min-max normalised height: 1.5/5 = 0.3 < gate
        const InjectedPeak peaks[] = {{200.f, 5.f, 5.f}, {700.f, 1.5f, 5.f}};
        const auto         output  = block.processOne(makeTestSpectrum(1024, peaks));

        const auto& events  = output.timing_events[0];
        const bool  countOk = events.size() == 1UZ;
        const bool  gateOk  = countOk && std::abs(getProp(events[0].second, "centre") - 200.f) < 1.f;
        if (!countOk || !gateOk || verboseCharts()) {
            printDesignedVsDetectedChart("gate threshold rejects the weak (700-bin) injected peak", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 1UZ));
        if (!events.empty()) {
            expect(lt(std::abs(getProp(events[0].second, "centre") - 200.f), 1.f));
        }

        block.stop();
    };

    "max_peaks caps the events to the strongest peaks"_test = [] {
        OnnxPeakDetector block;
        block.model_path     = modelPath("peaks_fixture_N1024.ort.gz");
        block.gate_threshold = 0.5f;
        block.max_peaks      = 2U;
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        // scores 1.0 / 0.8 / 0.6 — the cap must drop the weakest, not the last
        const InjectedPeak peaks[] = {{200.f, 5.f, 5.f}, {500.f, 4.f, 5.f}, {800.f, 3.f, 5.f}};
        const auto         output  = block.processOne(makeTestSpectrum(1024, peaks));

        const auto& events  = output.timing_events[0];
        const bool  countOk = events.size() == 2UZ;
        const bool  capOk   = countOk && std::abs(getProp(events[0].second, "centre") - 200.f) < 1.f && std::abs(getProp(events[1].second, "centre") - 500.f) < 1.f;
        if (!countOk || !capOk || verboseCharts()) {
            printDesignedVsDetectedChart("max_peaks=2 caps to the two strongest of three injected peaks", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 2UZ));
        if (events.size() == 2UZ) {
            expect(lt(std::abs(getProp(events[0].second, "centre") - 200.f), 1.f));
            expect(lt(std::abs(getProp(events[1].second, "centre") - 500.f), 1.f));
        }

        block.stop();
    };

    "resampled input maps events back to the input grid"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        // 2048-bin input, peak at input bin 400 → model bin 200 → event back at ~400
        const InjectedPeak peaks[] = {{400.f, 5.f, 10.f}};
        const auto         output  = block.processOne(makeTestSpectrum(2048, peaks));

        expect(eq(output.signal_values.size(), 4UZ * 2048UZ)) << "output stays on the input grid";
        const auto& events  = output.timing_events[0];
        const bool  countOk = events.size() == 1UZ;
        const bool  mapOk   = countOk && std::abs(getProp(events[0].second, "centre") - 400.f) < 2.f;
        if (!countOk || !mapOk || verboseCharts()) {
            printDesignedVsDetectedChart("resampled 2048-bin input, event mapped back to bin ~400", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 1UZ));
        if (!events.empty()) {
            expect(lt(std::abs(getProp(events[0].second, "centre") - 400.f), 2.f));
        }

        block.stop();
    };

    "event mapping preserves the resampler's endpoints"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        // 4096 -> 1024 puts the model's last bin at input bin 4095, not 4092, so the inverse map must use
        // (inputSize-1)/(modelN-1). A naive inputSize/modelN drifts ~2.9 bins at this distance from bin 0.
        const InjectedPeak peaks[] = {{3800.f, 10.f, 10.f}};
        const auto         output  = block.processOne(makeTestSpectrum(4096, peaks));

        const auto& events = output.timing_events[0];
        if (events.size() != 1UZ || verboseCharts()) {
            printDesignedVsDetectedChart("resampled 4096-bin input, event mapped back to bin ~3800", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 1UZ));
        if (!events.empty()) {
            expect(lt(std::abs(getProp(events[0].second, "centre") - 3800.f), 1.5f)) << "endpoint-preserving inverse of OnnxPreprocess::resample";
        }

        block.stop();
    };

    "residual output has the detected peak stripped"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        const InjectedPeak peaks[] = {{300.f, 5.f, 5.f}};
        const auto         input   = makeTestSpectrum(1024, peaks);
        const auto         output  = block.processOne(input);

        std::span<const float> residual(output.signal_values.data() + 3 * 1024, 1024);
        expect(lt(residual[300], 1.f)) << "peak removed from residual (was ~5.1)";

        block.stop();
    };

    "model_path swap through the settings system reloads on a started block"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }
        expect(eq(block._session.modelN(), 1024UZ));

        expect(block.settings().set({{"model_path", modelPath("peaks_fixture_N512.ort.gz")}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(block._session.modelN(), 512UZ)) << "settingsChanged must reload the model";
        block.stop();
    };

    "a model declaring a different property_layout stops the block at load"_test = [] {
        // same-length in-place rename of the second layout column corrupts nothing else in
        // the flatbuffer, so the load succeeds and only the layout validation must trip
        std::ifstream     in(modelPath("peaks_fixture_N1024.ort.gz"), std::ios::binary);
        std::vector<char> packed((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto              inflated = gr::compression::decompress(std::as_bytes(std::span(packed)), gr::compression::Format::gzip);
        expect(inflated.has_value()) << "the tracked fixture must inflate before its metadata can be patched";
        std::string bytes(reinterpret_cast<const char*>(inflated->data()), inflated->size());
        const auto  pos = bytes.find("peak_present,centre");
        expect(pos != std::string::npos) << "fixture must declare the property_layout metadata";
        bytes.replace(pos, std::string_view("peak_present,centre").size(), "peak_present,middle");

        const auto patchedPath = std::filesystem::temp_directory_path() / "gr_onnx_qa_bad_layout.ort";
        {
            std::ofstream outFile(patchedPath, std::ios::binary);
            outFile.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }

        // assert loadability on a standalone session, since the block tears its own session down on stop
        OnnxSession bareSession;
        expect(bareSession.load(patchedPath.string()).has_value()) << "the patched model itself must load";

        OnnxPeakDetector block;
        block.model_path = patchedPath.string();
        block.start();

        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "mismatching property_layout must stop the block";
        expect(!block.isModelLoaded()) << "stopping must release the rejected model";

        block.stop();
        std::filesystem::remove(patchedPath);
    };
};

// One isolated peak per width regime, log-spaced across the generator's width range. max_peaks must
// stay at the block default (8U) -- overriding it to 1 draws nPeaks from {0,1} and these seeds all
// land on 0.
// This fixture (unlike the trained regressor) always proposes a fixed K=8 candidate rows; only
// gate_threshold decides how many clear it -- 1 of 8 on the two narrowest canaries, all 8 from
// sigma~15 up. The best-matching row is nonetheless a near-perfect sub-bin fit in every regime,
// which is what expectedCount/positionTolerance below pin.
const boost::ut::suite<"OnnxPeakDetector canaries"> canaryTests = [] {
    struct Canary {
        std::uint32_t seed;
        const char*   regime;
        float         centre;
        float         sigma;
        std::size_t   expectedCount;
        float         positionTolerance;
    };
    static constexpr std::array<Canary, 6> kCanaries{{
        {13545U, "sigma~2", 1003.89f, 2.00f, 1UZ, 0.5f},
        {10579U, "sigma~6", 686.32f, 5.99f, 1UZ, 0.5f},
        {28189U, "sigma~15", 258.16f, 15.01f, 8UZ, 2.f},
        {26978U, "sigma~40", 73.53f, 40.12f, 8UZ, 1.f},
        {49509U, "sigma~80", 64.29f, 79.93f, 8UZ, 1.5f},
        {11792U, "sigma~130", 534.51f, 129.88f, 8UZ, 1.f},
    }};

    for (const auto& canary : kCanaries) {
        boost::ut::test(std::format("a single {} peak at defaults", canary.regime)) = [canary] {
            gr::testing::SyntheticPeakSpectrum<float> gen;
            gen.spectrum_size = 1024U;
            gen.seed          = canary.seed;
            gen.start();

            std::vector<std::uint8_t>       tick(1UZ, 0U);
            std::vector<gr::DataSet<float>> genOut(1UZ);
            expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
            const auto& designed = genOut[0].timing_events[0];
            expect(eq(designed.size(), 1UZ)) << std::format("seed {} must design exactly one peak", canary.seed);

            OnnxPeakDetector block;
            block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
            block.start();
            if (!block.isModelLoaded()) {
                expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
                return;
            }

            const auto  output   = block.processOne(genOut[0]);
            const auto& detected = output.timing_events[0];

            printDesignedVsDetectedChart(std::format("seed {} (single {} peak, sigma {:.1f})", canary.seed, canary.regime, canary.sigma), output.signalValues(0), designed, detected);

            std::println("seed {}: fixture finds {} detection(s) on the {} canary", canary.seed, detected.size(), canary.regime);
            float bestDelta = std::numeric_limits<float>::max();
            for (const auto& [idx, props] : detected) {
                const float centre = getProp(props, "centre");
                bestDelta          = std::min(bestDelta, std::abs(centre - canary.centre));
                std::println("  bin={} centre={:.2f} conf={:.2f}", idx, centre, getProp(props, "confidence"));
            }

            expect(eq(detected.size(), canary.expectedCount)) << std::format("seed {}: measured detection count for the {} regime", canary.seed, canary.regime);
            if (!detected.empty()) {
                std::println("seed {}: nearest detection is {:.2f} bins from the designed centre {:.2f}", canary.seed, bestDelta, canary.centre);
                expect(lt(bestDelta, canary.positionTolerance)) << std::format("seed {}: best detection should land tightly on the single designed peak", canary.seed);
            }

            block.stop();
        };
    }
};

// Seeds {0, 1, 15, 30} design zero peaks at generator defaults (spectrum_size=1024).
// Unlike the classical detector, this untrained fixture has no notion of "nothing here": min-max
// normalisation plus unconditional TopK means all 8 of its fixed candidate rows still clear
// gate_threshold on pure noise (conf 0.87-1.00). That is measured behaviour of the fixture, not a
// claim about the trained regressor's false-positive rate, which is evaluated in Python.
const boost::ut::suite<"OnnxPeakDetector zero-peak scenes"> zeroPeakTests = [] {
    for (std::uint64_t seed : {0ULL, 1ULL, 15ULL, 30ULL}) {
        boost::ut::test(std::format("seed {} designs zero peaks, yet the fixture still reports all 8 candidate rows", seed)) = [seed] {
            gr::testing::SyntheticPeakSpectrum<float> gen;
            gen.spectrum_size = 1024U;
            gen.seed          = seed;
            gen.start();

            std::vector<std::uint8_t>       tick(1UZ, 0U);
            std::vector<gr::DataSet<float>> genOut(1UZ);
            expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
            const auto& designed = genOut[0].timing_events[0];
            expect(eq(designed.size(), 0UZ)) << std::format("seed {} should design zero peaks", seed);

            OnnxPeakDetector block;
            block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
            block.start();
            if (!block.isModelLoaded()) {
                expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
                return;
            }

            const auto  output   = block.processOne(genOut[0]);
            const auto& detected = output.timing_events[0];

            // charted unconditionally: this IS the surprising result, not a failure path
            printDesignedVsDetectedChart(std::format("seed {} (zero-peak scene)", seed), output.signalValues(0), designed, detected);

            std::println("seed {}: fixture finds {} detection(s) on a signal-free spectrum -- see suite comment", seed, detected.size());
            for (const auto& [idx, props] : detected) {
                std::println("  bin={} centre={:.2f} conf={:.2f}", idx, getProp(props, "centre"), getProp(props, "confidence"));
            }

            expect(eq(detected.size(), 8UZ)) << std::format("seed {}: measured -- all 8 of the fixture's fixed candidate rows clear gate even on a signal-free spectrum", seed);

            block.stop();
        };
    }
};

// A tall dominant peak compresses this fixture's min-max-normalised scores, so most of its 8 fixed
// candidate rows gate out here: seed 1893 (7 designed peaks) clears 5 of 8, clustered on the
// dominant peak's crest; seed 100 (3 designed peaks) clears exactly 3, one per well-separated peak.
const boost::ut::suite<"OnnxPeakDetector crowded scenes"> crowdedSceneTests = [] {
    "seven-peak crowded scene at seed 1893"_test = [] {
        gr::testing::SyntheticPeakSpectrum<float> gen;
        gen.spectrum_size = 1024U;
        gen.seed          = 1893ULL;
        gen.start();

        std::vector<std::uint8_t>       tick(1UZ, 0U);
        std::vector<gr::DataSet<float>> genOut(1UZ);
        expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
        const auto& designed = genOut[0].timing_events[0];
        expect(eq(designed.size(), 7UZ)) << "seed 1893 should design 7 peaks at generator defaults";

        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        const auto  output   = block.processOne(genOut[0]);
        const auto& detected = output.timing_events[0];

        printDesignedVsDetectedChart("seed 1893 (7-peak crowded scene)", output.signalValues(0), designed, detected);

        std::println("seed 1893: fixture finds {} detection(s) against 7 designed peaks", detected.size());
        float bestConfidence = -1.f;
        float bestCentre     = 0.f;
        for (const auto& [idx, props] : detected) {
            const float centre     = getProp(props, "centre");
            const float confidence = getProp(props, "confidence");
            std::println("  bin={} centre={:.2f} conf={:.2f}", idx, centre, confidence);
            if (confidence > bestConfidence) {
                bestConfidence = confidence;
                bestCentre     = centre;
            }
        }

        expect(eq(detected.size(), 4UZ)) << "seed 1893: measured -- 4 of 8 fixed candidate rows clear gate, see suite comment";
        if (!detected.empty()) {
            const auto match = nearestDesigned(designed, bestCentre);
            std::println("seed 1893: highest-confidence detection c={:.2f}, nearest designed delta={:.2f} bins (sigma={:.2f})", bestCentre, match.delta, match.gaussianEquivalentSigma);
            expect(lt(match.delta, 1.f)) << "seed 1893: the highest-confidence detection should tightly match the dominant designed peak at centre 493.67";
        }

        block.stop();
    };

    "three-peak well-separated scene at seed 100"_test = [] {
        gr::testing::SyntheticPeakSpectrum<float> gen;
        gen.spectrum_size = 1024U;
        gen.seed          = 100ULL;
        gen.start();

        std::vector<std::uint8_t>       tick(1UZ, 0U);
        std::vector<gr::DataSet<float>> genOut(1UZ);
        expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
        const auto& designed = genOut[0].timing_events[0];
        expect(eq(designed.size(), 3UZ)) << "seed 100 should design 3 peaks at generator defaults";

        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked fixture model failed to load: " << modelPath("peaks_fixture_N1024.ort.gz");
            return;
        }

        const auto  output   = block.processOne(genOut[0]);
        const auto& detected = output.timing_events[0];

        printDesignedVsDetectedChart("seed 100 (3-peak scene)", output.signalValues(0), designed, detected);

        std::println("seed 100: fixture finds {} detection(s) against 3 designed peaks", detected.size());
        float bestConfidence = -1.f;
        float bestCentre     = 0.f;
        for (const auto& [idx, props] : detected) {
            const float centre     = getProp(props, "centre");
            const float confidence = getProp(props, "confidence");
            std::println("  bin={} centre={:.2f} conf={:.2f}", idx, centre, confidence);
            if (confidence > bestConfidence) {
                bestConfidence = confidence;
                bestCentre     = centre;
            }
        }

        expect(eq(detected.size(), 3UZ)) << "seed 100: measured -- 3 of 8 fixed candidate rows clear gate, one per well-separated designed peak, see suite comment";
        if (!detected.empty()) {
            const auto match = nearestDesigned(designed, bestCentre);
            std::println("seed 100: highest-confidence detection c={:.2f}, nearest designed delta={:.2f} bins (sigma={:.2f})", bestCentre, match.delta, match.gaussianEquivalentSigma);
            expect(lt(match.delta, 1.f)) << "seed 100: the highest-confidence detection should tightly match the dominant designed peak at centre 705.65";
        }

        block.stop();
    };
};

const boost::ut::suite<"OnnxPeakDetector temporal path"> temporalTests = [] {
    // The M>1 branch had no coverage at all although the class doc advertises it. The warm-up
    // contract matters beyond the block: a forwarded frame still carries whatever timing_events its
    // producer attached, so a consumer that scores them credits the detector with peaks it never
    // found — which is exactly what ex06 had to be taught to reject.
    "an M=16 model forwards the first M-1 frames unannotated, then detects"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024_M16.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            expect(false) << "tracked M=16 fixture failed to load: " << modelPath("peaks_fixture_N1024_M16.ort.gz");
            return;
        }
        expect(eq(block._session.historyDepth(), 16UZ)) << "history depth must come from the model";

        const InjectedPeak peaks[] = {{200.25f, 5.f, 5.f}};

        for (std::size_t frame = 0UZ; frame < 15UZ; ++frame) {
            const auto input  = makeTestSpectrum(1024, peaks);
            const auto output = block.processOne(input);
            expect(isMarkedPassthrough(output)) << std::format("frame {} fills the window and must be forwarded, not scored", frame);
            expect(eq(output.signal_values.size(), input.signal_values.size())) << "a forwarded frame must pass through unaltered";
        }

        const auto detected = block.processOne(makeTestSpectrum(1024, peaks));
        expect(!isMarkedPassthrough(detected)) << "the 16th frame completes the window and must be inferred";
        expect(!detected.timing_events.empty());
        if (!detected.timing_events.empty()) {
            const auto& events = detected.timing_events[0];
            expect(eq(events.size(), 1UZ)) << "the designed peak must be found once the window is full";
            if (!events.empty()) {
                expect(lt(std::abs(getProp(events[0].second, "centre") - 200.25f), 1.f)) << "temporal detection must land on the designed centre";
            }
        }
        block.stop();
    };

    "a reset drops the part-filled window so the next frame starts warm-up again"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_N1024_M16.ort.gz");
        block.start();
        if (!block.isModelLoaded()) {
            return;
        }
        const InjectedPeak peaks[] = {{200.25f, 5.f, 5.f}};
        for (std::size_t frame = 0UZ; frame < 15UZ; ++frame) {
            std::ignore = block.processOne(makeTestSpectrum(1024, peaks));
        }
        block.stop();
        block.start();
        const auto afterRestart = block.processOne(makeTestSpectrum(1024, peaks));
        expect(isMarkedPassthrough(afterRestart)) << "a restarted block must refill its window rather than infer on stale frames";
        block.stop();
    };
};

const boost::ut::suite<"OnnxPeakDetector legacy model rejection"> legacyTests = [] {
    "a heatmap+regression model without a peaks output stops the block after passing the failing frame through"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("heatmap_only_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "tracked heatmap-only fixture failed to load: " << modelPath("heatmap_only_N1024.ort.gz");
            return;
        }

        const InjectedPeak peaks[] = {{200.f, 5.f, 5.f}};
        const auto         input   = makeTestSpectrum(1024, peaks);
        const auto         output  = block.processOne(input);

        expect(eq(output.signal_values.size(), 1024UZ)) << "must pass through untouched";
        expect(eq(output.signal_names.size(), 1UZ));
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "missing peaks output must stop the block, not silently keep forwarding";

        block.stop();
    };
};

const boost::ut::suite<"OnnxPeakDetector model-declared expression"> exprFixtureTests = [] {
    "a model declaring normalise_mode=Expression works end to end"_test = [] {
        OnnxPeakDetector block;
        block.model_path = modelPath("peaks_fixture_expr_N1024.ort.gz");
        block.start();

        if (!block.isModelLoaded()) {
            expect(false) << "expression fixture present but not loadable";
            return;
        }
        expect(block.normalise_mode == NormaliseMode::Expression) << "normalise_mode must be adopted from model metadata";
        expect(eq(block.normalise_expr.value, std::string("vecOut := vecIn"))) << "normalise_expr must be adopted from model metadata";
        expect(block.state() != gr::lifecycle::State::REQUESTED_STOP) << "the declared expression must compile at load";

        // identity expression: detection must match the plain fixture semantics
        const InjectedPeak peaks[] = {{200.25f, 5.f, 5.f}};
        const auto         output  = block.processOne(makeTestSpectrum(1024, peaks));

        const auto& events  = output.timing_events[0];
        const bool  countOk = events.size() == 1UZ;
        const bool  exprOk  = countOk && std::abs(getProp(events[0].second, "centre") - 200.25f) < 0.5f;
        if (!countOk || !exprOk || verboseCharts()) {
            printDesignedVsDetectedChart("model-declared Expression normalisation, identity-equivalent detection", output.signalValues(0), peaks, events);
        }

        expect(eq(events.size(), 1UZ)) << "expression-normalised inference must still detect the peak";
        if (!events.empty()) {
            expect(lt(std::abs(getProp(events[0].second, "centre") - 200.25f), 0.5f));
        }

        block.stop();
    };
};

int main() { /* boost::ut */ }
