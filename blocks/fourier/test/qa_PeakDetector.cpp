#include <boost/ut.hpp>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/fourier/PeakDetector.hpp>
#include <gnuradio-4.0/testing/SyntheticPeakSpectrum.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <numbers>
#include <print>
#include <span>

using namespace boost::ut;
using namespace gr;
using namespace gr::blocks::fourier;
using namespace gr::graphs;

namespace {

gr::DataSet<float> makeSpectrum(std::size_t n, auto peakFn, float noiseFloor = 0.1f) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(n)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};

    ds.signal_values.resize(n, noiseFloor);
    peakFn(ds.signal_values);
    return ds;
}

void addGaussian(std::vector<float>& v, float centre, float amplitude, float sigma = 3.f) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        float x = static_cast<float>(i) - centre;
        v[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
    }
}

void addLorentzian(std::vector<float>& v, float centre, float amplitude, float gamma = 3.f) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        float x = static_cast<float>(i) - centre;
        v[i] += amplitude * gamma * gamma / (gamma * gamma + x * x);
    }
}

struct NearestDesignedMatch {
    float delta;
    float sigma;
};

// nearest ground-truth peak (by position) to a detected centre, from SyntheticPeakSpectrum's
// timing_events ("centre"/"sigma" keys, see SyntheticPeakSpectrum.hpp:139-143)
[[nodiscard]] NearestDesignedMatch nearestDesigned(const std::vector<gr::DataSet<float>::idx_pmt_map>& designedEvents, float detectedCentre) {
    NearestDesignedMatch best{std::numeric_limits<float>::max(), 0.f};
    for (const auto& [idx, props] : designedEvents) {
        float centre = props.value_or<float>("centre"_spmr, 0.f);
        float sigma  = props.value_or<float>("sigma"_spmr, 0.f);
        float delta  = std::abs(detectedCentre - centre);
        if (delta < best.delta) {
            best = {delta, sigma};
        }
    }
    return best;
}

// narrow, high-SNR peaks get a tight 2-bin floor; broad/overlapping designed peaks (where the
// detected centroid is ambiguous by construction) get up to one designed sigma of slack
[[nodiscard]] float positionTolerance(float nearestSigma) { return std::max(2.f, nearestSigma); }

// designed-vs-detected spectrum chart, same ImChart approach as ex0_peak_detector_classical.cpp:
// spectrum as a braille trace, designed (ground-truth) peaks and detections as distinct markers
void printDesignedVsDetectedChart(std::string_view title, std::span<const float> spectrum, const std::vector<gr::DataSet<float>::idx_pmt_map>& designedEvents, const std::vector<gr::DataSet<float>::idx_pmt_map>& detectedEvents) {
    const std::size_t n = spectrum.size();

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
        const float centre = props.value_or<float>("centre"_spmr, 0.f);
        xDesigned.push_back(static_cast<double>(centre));
        yDesigned.push_back(sampleAt(centre));
    }
    std::vector<double> xDetected, yDetected;
    for (const auto& [idx, props] : detectedEvents) {
        const float centre = props.value_or<float>("centre"_spmr, 0.f);
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

// Canaries: one isolated peak, nothing else. The simplest scene the detector can be asked
// about, so a failure here localises to the detector rather than to peak interaction. Seeds
// were found by scanning SyntheticPeakSpectrum at max_peaks = 1 for single peaks, one per width
// regime; the measured truth is in the table below.
const boost::ut::suite<"PeakDetector canaries"> canaryTests = [] {
    struct Canary {
        std::uint32_t seed;
        const char*   regime;
        float         centre;
        float         sigma;
        bool          expectDetected;
    };
    // one per width regime, all comfortably above the noise floor. The broad case is included
    // deliberately: a sigma-85 peak inflates the block's own global noise estimate, so it is the
    // regime most likely to regress first even though it currently succeeds
    static constexpr std::array<Canary, 3> kCanaries{{
        {4U, "narrow", 679.87f, 1.01f, true},
        {215U, "medium", 488.84f, 40.92f, true},
        {117U, "broad", 790.26f, 84.82f, true},
    }};

    for (const auto& canary : kCanaries) {
        boost::ut::test(std::format("a single {} peak at defaults", canary.regime)) = [canary] {
            gr::testing::SyntheticPeakSpectrum<float> gen;
            gen.spectrum_size = 1024U;
            gen.max_peaks     = 1U;
            gen.seed          = canary.seed;
            gen.start();

            std::vector<std::uint8_t>       tick(1UZ, 0U);
            std::vector<gr::DataSet<float>> genOut(1UZ);
            expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
            const auto& designed = genOut[0].timing_events[0];
            expect(eq(designed.size(), 1UZ)) << std::format("seed {} must design exactly one peak", canary.seed);

            PeakDetector detector; // shipped defaults, no overrides
            auto         detectedDs = detector.processOne(genOut[0]);
            const auto&  detected   = detectedDs.timing_events[0];

            printDesignedVsDetectedChart(std::format("seed {} (single {} peak, sigma {:.1f})", canary.seed, canary.regime, canary.sigma), std::span<const float>(genOut[0].signal_values), designed, detected);

            if (!canary.expectDetected) {
                std::println("seed {}: {} single peak yields {} detection(s) -- broad peaks inflate the global noise estimate", canary.seed, canary.regime, detected.size());
                return;
            }
            expect(ge(detected.size(), 1UZ)) << std::format("seed {}: an isolated {} peak must be found at default settings", canary.seed, canary.regime);
            if (!detected.empty()) {
                float best = std::numeric_limits<float>::max();
                for (const auto& [idx, props] : detected) {
                    best = std::min(best, std::abs(props.value_or<float>("centre"_spmr, 0.f) - canary.centre));
                }
                std::println("seed {}: nearest detection is {:.2f} bins from the designed centre {:.2f}", canary.seed, best, canary.centre);
                expect(lt(best, std::max(2.f * canary.sigma, 3.f))) << std::format("seed {}: detected centre should land on the single designed peak", canary.seed);
            }
        };
    }
};

const boost::ut::suite<"PeakDetector"> peakDetectorTests = [] {
    "iterative stripping resolves overlapping peaks"_test = [] {
        constexpr std::size_t n = 1024;
        // broad Schottky peak with narrow betatron line on its shoulder
        auto input = makeSpectrum(n, [](auto& v) {
            addGaussian(v, 400.f, 8.f, 80.f); // broad Schottky
            addGaussian(v, 450.f, 15.f, 3.f); // narrow betatron on shoulder
            addGaussian(v, 800.f, 12.f, 3.f); // isolated narrow
        });

        PeakDetector detector;
        detector.noise_rejection_threshold = 2.0f;
        detector.min_prominence            = 2.0f;
        detector.max_iterations            = 10;
        detector.min_isolation             = 0.0f; // allow close peaks

        auto output = detector.processOne(std::move(input));

        const auto& events = output.timing_events[0];
        std::println("overlap test: {} peaks detected", events.size());
        for (const auto& [idx, props] : events) {
            float sigL = props.value_or<float>("sigma_left"_spmr, 0.f);
            float sigR = props.value_or<float>("sigma_right"_spmr, 0.f);
            float prom = props.value_or<float>("prominence"_spmr, 0.f);
            float amp  = props.value_or<float>("amplitude"_spmr, 0.f);
            std::println("  bin={} sigL={:.1f} sigR={:.1f} prom={:.1f} amp={:.1f}", idx, sigL, sigR, prom, amp);
        }

        expect(ge(events.size(), 2UZ)) << "should find at least the narrow + broad peak";
    };

    "detects known peaks at high SNR"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) {
            addGaussian(v, 100.f, 10.f);
            addGaussian(v, 250.f, 8.f);
            addGaussian(v, 400.f, 12.f);
        });

        PeakDetector detector;
        detector.noise_rejection_threshold = 2.0f;
        detector.max_peaks                 = 8;

        auto output = detector.processOne(std::move(input));

        const auto& events = output.timing_events[0];
        std::println("3-peak test: {} detected", events.size());
        expect(ge(events.size(), 3UZ)) << "should detect all 3 peaks";
    };

    "broad peak detected and correctly characterised"_test = [] {
        constexpr std::size_t n     = 1024;
        auto                  input = makeSpectrum(n, [](auto& v) {
            addGaussian(v, 500.f, 5.f, 100.f); // very broad
        });

        PeakDetector detector;
        detector.noise_rejection_threshold = 1.5f;

        auto        output = detector.processOne(std::move(input));
        const auto& events = output.timing_events[0];

        std::println("broad peak: {} detected", events.size());
        expect(ge(events.size(), 1UZ)) << "should detect the broad peak";

        if (!events.empty()) {
            float sigL = events[0].second.value_or<float>("sigma_left"_spmr, 0.f);
            float sigR = events[0].second.value_or<float>("sigma_right"_spmr, 0.f);
            std::println("  sigma L={:.1f} R={:.1f} (true ~100)", sigL, sigR);
            expect(gt(sigL, 30.f)) << "left width should be substantial";
            expect(gt(sigR, 30.f)) << "right width should be substantial";
        }
    };

    "adaptive shape selection detects a Lorentzian peak"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) {
            addLorentzian(v, 256.f, 20.f, 10.f); // Lorentzian peak
        });

        PeakDetector detector;
        detector.subtraction_shape = 0; // Auto

        auto        output = detector.processOne(std::move(input));
        const auto& events = output.timing_events[0];

        std::println("Lorentzian test: {} detected", events.size());
        expect(ge(events.size(), 1UZ));

        if (!events.empty()) {
            float kurt = events[0].second.value_or<float>("kurtosis"_spmr, 0.f);
            std::println("  kurtosis={:.2f} (Lorentzian has excess kurtosis > 0)", kurt);
        }
    };

    "provides uncertainty estimates that scale with SNR"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) {
            addGaussian(v, 128.f, 50.f, 5.f); // high SNR
            addGaussian(v, 384.f, 2.f, 20.f); // low SNR, broad
        });

        PeakDetector detector;
        detector.noise_rejection_threshold = 1.5f;
        auto output                        = detector.processOne(std::move(input));

        const auto& events = output.timing_events[0];
        std::println("uncertainty test: {} peaks", events.size());

        for (const auto& [idx, props] : events) {
            float posUnc = props.value_or<float>("position_uncertainty"_spmr, 0.f);
            float ampUnc = props.value_or<float>("amplitude_uncertainty"_spmr, 0.f);
            float amp    = props.value_or<float>("amplitude"_spmr, 0.f);
            std::println("  bin={} amp={:.1f} pos_unc={:.3f} amp_unc={:.3f}", idx, amp, posUnc, ampUnc);
        }
    };

    "measured amplitude from raw spectrum"_test = [] {
        constexpr std::size_t n     = 256;
        auto                  input = makeSpectrum(n, [](auto& v) { addGaussian(v, 128.f, 20.f, 5.f); });

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));

        expect(ge(output.timing_events[0].size(), 1UZ));
        if (!output.timing_events[0].empty()) {
            float ampMeas = output.timing_events[0][0].second.value_or<float>("amplitude_measured"_spmr, 0.f);
            std::println("  amplitude_measured={:.2f} (should be ~20)", ampMeas);
            expect(gt(ampMeas, 15.f)) << "measured amplitude should be close to 20";
        }
    };

    "respects max_peaks limit"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) {
            addGaussian(v, 50.f, 10.f);
            addGaussian(v, 150.f, 8.f);
            addGaussian(v, 250.f, 12.f);
            addGaussian(v, 350.f, 6.f);
            addGaussian(v, 450.f, 9.f);
        });

        PeakDetector detector;
        detector.noise_rejection_threshold = 1.5f;
        detector.max_peaks                 = 3;

        auto output = detector.processOne(std::move(input));
        expect(le(output.timing_events[0].size(), 3UZ));
    };

    "flat spectrum produces no peaks"_test = [] {
        constexpr std::size_t n     = 256;
        auto                  input = makeSpectrum(n, [](auto&) {}, 1.0f);

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));
        expect(output.timing_events[0].empty());
    };

    "output preserves metadata"_test = [] {
        constexpr std::size_t n     = 128;
        auto                  input = makeSpectrum(n, [](auto& v) { addGaussian(v, 64.f, 10.f); });
        input.timestamp             = 12345;
        input.axis_names            = {"frequency"};
        input.axis_units            = {"Hz"};

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));

        expect(eq(output.timestamp, std::int64_t(12345)));
        expect(eq(output.axis_names[0], std::string("frequency")));
    };

    "all required properties present"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) { addGaussian(v, 256.f, 15.f); });

        PeakDetector detector;
        detector.max_peaks = 1;
        auto output        = detector.processOne(std::move(input));

        expect(ge(output.timing_events[0].size(), 1UZ));
        if (!output.timing_events[0].empty()) {
            const auto& props = output.timing_events[0][0].second;
            expect(props.contains("confidence"_spmr));
            expect(props.contains("centre"_spmr));
            expect(props.contains("sigma"_spmr));
            expect(props.contains("sigma_left"_spmr));
            expect(props.contains("sigma_right"_spmr));
            expect(props.contains("amplitude"_spmr));
            expect(props.contains("amplitude_measured"_spmr));
            expect(props.contains("prominence"_spmr));
            expect(props.contains("isolation"_spmr));
            expect(props.contains("w68"_spmr));
            expect(props.contains("w96"_spmr));
            expect(props.contains("w99"_spmr));
            expect(props.contains("kurtosis"_spmr));
            expect(props.contains("noise_sigma"_spmr));
            expect(props.contains("noise_floor"_spmr));
            expect(props.contains("position_uncertainty"_spmr));
            expect(props.contains("width_uncertainty"_spmr));
            expect(props.contains("amplitude_uncertainty"_spmr));
        }
    };

    "emits the fractional centre and mean sigma matching the OnnxPeakDetector key set"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) { addGaussian(v, 256.4f, 15.f, 4.f); });

        PeakDetector detector;
        detector.max_peaks = 1;
        auto output        = detector.processOne(std::move(input));

        expect(ge(output.timing_events[0].size(), 1UZ));
        if (!output.timing_events[0].empty()) {
            const auto& [idx, props] = output.timing_events[0][0];
            const float centre       = props.value_or<float>("centre"_spmr, 0.f);
            const float sigma        = props.value_or<float>("sigma"_spmr, 0.f);
            expect(lt(std::abs(centre - 256.4f), 1.f)) << "sub-bin interpolated centre";
            expect(eq(idx, std::lround(centre))) << "event index is the nearest bin to the fractional centre";
            const float sigL = props.value_or<float>("sigma_left"_spmr, 0.f);
            const float sigR = props.value_or<float>("sigma_right"_spmr, 0.f);
            expect(lt(std::abs(sigma - 0.5f * (sigL + sigR)), 1e-5f)) << "sigma is the mean of the half-widths";
        }
    };

    "peaks sorted by position ascending"_test = [] {
        constexpr std::size_t n     = 512;
        auto                  input = makeSpectrum(n, [](auto& v) {
            addGaussian(v, 400.f, 8.f);
            addGaussian(v, 100.f, 12.f);
            addGaussian(v, 250.f, 10.f);
        });

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));
        const auto&  events = output.timing_events[0];

        for (std::size_t i = 1; i < events.size(); ++i) {
            expect(ge(events[i].first, events[i - 1].first)) << "peaks should be position-sorted";
        }
    };

    "default settings report zero detections on the zero-peak seeds"_test = [] {
        // regression guard for the false-positive floor that motivated raising
        // noise_rejection_threshold/min_prominence from 2 to 5 (PeakDetector.hpp): seeds 0, 1, 15
        // and 30 design zero peaks at SyntheticPeakSpectrum's defaults (spectrum_size=1024), so any
        // detection here is a false positive by construction. At the old defaults (2, 2) this scene
        // measures 10 false detections per spectrum; at the shipped defaults it must measure zero.
        for (std::uint64_t seed : {0ULL, 1ULL, 15ULL, 30ULL}) {
            gr::testing::SyntheticPeakSpectrum<float> gen;
            gen.spectrum_size = 1024U;
            gen.seed          = seed;
            gen.start();

            std::vector<std::uint8_t>       tick(1UZ, 0U);
            std::vector<gr::DataSet<float>> genOut(1UZ);
            expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
            const auto& designed = genOut[0].timing_events[0];
            expect(eq(designed.size(), 0UZ)) << std::format("seed {} should design zero peaks", seed);

            PeakDetector detector; // default settings, no overrides
            auto         detectedDs = detector.processOne(genOut[0]);
            const auto&  detected   = detectedDs.timing_events[0];

            // charted unconditionally: an empty marker set IS the confirmation that a
            // signal-free spectrum produces nothing at default settings
            printDesignedVsDetectedChart(std::format("seed {} (zero-peak scene)", seed), std::span<const float>(genOut[0].signal_values), designed, detected);
            expect(eq(detected.size(), 0UZ)) << std::format("seed {}: default settings must not detect anything on a signal-free spectrum", seed);
        }
    };

    "classical detector isolates the dominant peak in a three-peak blended-broad scene at seed 28066"_test = [] {
        // seed 28066 designs three heavily overlapping broad peaks (centres 790.65/804.36/818.06,
        // sigma 130.73/17.29/21.44, amplitude 2.31/7.22/22.30). The classical detector's residual
        // noise estimate is contaminated by peaks this broad and overlapping, so only the dominant
        // one (amplitude 22.30) clears the prominence bar -- measured identically at the shipped
        // defaults (noise_rejection_threshold=min_prominence=5) and at the pre-fix thresholds (2,
        // 2), so this is a structural limitation of a single global noise estimate on broad,
        // overlapping scenes, not a threshold-tuning issue.
        gr::testing::SyntheticPeakSpectrum<float> gen;
        gen.spectrum_size = 1024U;
        gen.seed          = 28066ULL;
        gen.start();

        std::vector<std::uint8_t>       tick(1UZ, 0U);
        std::vector<gr::DataSet<float>> genOut(1UZ);
        expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
        const auto& designed = genOut[0].timing_events[0];
        expect(eq(designed.size(), 3UZ)) << "seed 28066 should design 3 peaks";

        PeakDetector detector; // shipped defaults, no overrides
        auto         detectedDs = detector.processOne(genOut[0]);
        const auto&  detected   = detectedDs.timing_events[0];

        float matchDelta = 0.f;
        float matchSigma = 0.f;
        if (!detected.empty()) {
            const auto match = nearestDesigned(designed, detected[0].second.value_or<float>("centre"_spmr, 0.f));
            matchDelta       = match.delta;
            matchSigma       = match.sigma;
        }
        printDesignedVsDetectedChart("seed 28066 (3-peak blended-broad scene)", std::span<const float>(genOut[0].signal_values), designed, detected);

        expect(eq(detected.size(), 1UZ)) << "measured regression: only the dominant, best-isolated peak clears the noise floor";

        if (!detected.empty()) {
            const float centre = detected[0].second.value_or<float>("centre"_spmr, 0.f);
            std::println("seed 28066: detected c={:.2f}, nearest designed delta={:.2f} bins (sigma={:.2f})", centre, matchDelta, matchSigma);
            expect(lt(matchDelta, positionTolerance(matchSigma))) << std::format("detected centre {:.2f} should land within tolerance of the nearest designed peak", centre);
        }
    };

    "classical detector's four detections in a seven-peak well-separated cascade at seed 1893 include a tight match on the isolated narrow peak"_test = [] {
        // seed 1893 is the multi-peak showcase seed (spectrum_size=1024, defaults; see also
        // ex0_peak_detector_classical.cpp): 7 designed peaks, well separated (minimum separation
        // ~4.6 designed sigma), amplitude spanning 43.15 down to 2.32. At the shipped defaults
        // (noise_rejection_threshold=min_prominence=5, measured identically at the pre-fix (2, 2)
        // thresholds) the classical detector finds 4 of the 7 -- the amplitude 43.15, 17.87, 12.35
        // and 6.82 designed peaks, each matched within 1 bin of its designed centre. The three
        // weakest (amplitude 3.67, 3.11, 2.32 -- all below the ~5-sigma noise floor at
        // noise_level=1) are missed. The isolated, high-SNR designed peak at centre 777.06 (sigma
        // 1.35) gets the tightest match (measured delta 0.04 bins) and is pinned below as the
        // regression signal, together with the total count.
        gr::testing::SyntheticPeakSpectrum<float> gen;
        gen.spectrum_size = 1024U;
        gen.seed          = 1893ULL;
        gen.start();

        std::vector<std::uint8_t>       tick(1UZ, 0U);
        std::vector<gr::DataSet<float>> genOut(1UZ);
        expect(gen.processBulk(tick, genOut) == gr::work::Status::OK);
        const auto& designed = genOut[0].timing_events[0];
        expect(eq(designed.size(), 7UZ)) << "seed 1893 should design 7 peaks";

        PeakDetector detector; // shipped defaults, no overrides
        auto         detectedDs = detector.processOne(genOut[0]);
        const auto&  detected   = detectedDs.timing_events[0];

        constexpr float kNarrowDesignedCentre = 777.06f;
        constexpr float kNarrowMatchTolerance = 2.0f; // designed sigma is 1.35 bins; measured delta 0.04
        float           bestDelta             = std::numeric_limits<float>::max();
        for (const auto& [idx, props] : detected) {
            const float centre = props.value_or<float>("centre"_spmr, 0.f);
            bestDelta          = std::min(bestDelta, std::abs(centre - kNarrowDesignedCentre));
        }

        printDesignedVsDetectedChart("seed 1893 (7-peak well-separated cascade)", std::span<const float>(genOut[0].signal_values), designed, detected);

        expect(eq(detected.size(), 4UZ)) << "measured regression: four of the seven designed peaks clear the shipped defaults";
        std::println("seed 1893: nearest detection to the narrow designed peak (c={:.2f}) is delta={:.2f} bins away", kNarrowDesignedCentre, bestDelta);
        expect(lt(bestDelta, kNarrowMatchTolerance)) << "one of the four detections should tightly match the narrow, high-SNR designed peak";
    };
};

int main() { /* boost::ut */ }
