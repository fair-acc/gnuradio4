#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>

#include <gnuradio-4.0/testing/EvolvingPeakSpectrum.hpp>

#include "PeakSpectrumChart.hpp"

// Golden values below were derived independently of the block under test, by hand-transcribing
// EvolvingPeakSpectrum::addNoise() (a plain gr::rng::GaussianNoise<double> draw per bin) against a
// standalone gr::rng::Xoshiro256pp(42), per the documented algorithm in
// algorithm/include/gnuradio-4.0/algorithm/rng/GaussianNoise.hpp. maybeSpawnPeak() draws from a
// *separate* _spawnRng instance, so with peak_spawn_probability=0 (guaranteeing no spawn ever,
// since a uniform01() draw is never negative) the noise sequence below is untouched by any other
// RNG consumption and matches a fresh GaussianNoise<double> draw sequence exactly.
namespace {
constexpr std::array<double, 8> kGaussianNoiseSeed42 = {
    0.98139839007249863,  //
    -0.56572010467395595, //
    1.3403256427520227,   //
    0.40231287029926083,  //
    -0.96422050629413836, //
    0.27055086445825288,  //
    0.19622652967452661,  //
    1.1536067585699392,   //
};
}

const boost::ut::suite<"EvolvingPeakSpectrum"> evolvingPeakSpectrumTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    constexpr auto kFloatingTypes = std::tuple<float, double>();

    "processBulk runs for both registered types, producing an exact all-zero spectrum at amplitude underflow"_test = []<typename T>(const T&) {
        // snr fixed pins targetAmplitude = 10; a single-spectrum onset (min==max==1) starts a peak at
        // full amplitude on spectrum 0 already, so use a 2-spectrum onset to hit amplitude == 0 first.
        EvolvingPeakSpectrum<T> block;
        block.spectrum_size          = 8U;
        block.noise_level            = T(0);
        block.max_concurrent_peaks   = 1U;
        block.min_onset_spectra      = 2U;
        block.max_onset_spectra      = 2U;
        block.snr_min_db             = T(20);
        block.snr_max_db             = T(20);
        block.peak_spawn_probability = T(1);
        block.max_drift_rate         = T(0);
        block.seed                   = 42ULL;
        block.start();

        std::vector<std::uint8_t>   in(1UZ, 0U);
        std::vector<gr::DataSet<T>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        expect(gr::dataset::checkConsistency(out[0]).has_value());
        auto       bins    = out[0].signalValues(0);
        const bool allZero = std::ranges::all_of(bins, [](T v) { return v == T(0); });
        if (!allZero || gr::testing::qa::verboseCharts()) {
            const std::array<gr::testing::qa::PeakMarkerSet, 1> markerSets{gr::testing::qa::designedPeakMarkers(out[0].timing_events[0])};
            gr::testing::qa::printSpectrumChart<T>("EvolvingPeakSpectrum amplitude-underflow spectrum", bins, markerSets);
        }
        expect(allZero) << "amplitude underflow with no noise must yield an exact all-zero spectrum";
    } | kFloatingTypes;

    "reset() clears in-flight peaks and reseeds the RNG"_test = [] {
        EvolvingPeakSpectrum<double> block;
        block.spectrum_size          = static_cast<gr::Size_t>(kGaussianNoiseSeed42.size());
        block.max_concurrent_peaks   = 1U;
        block.peak_spawn_probability = 1.0;
        block.seed                   = 42ULL;
        block.start();

        std::vector<std::uint8_t>        churnIn(3UZ, 0U);
        std::vector<gr::DataSet<double>> churnOut(3UZ);
        expect(block.processBulk(churnIn, churnOut) == gr::work::Status::OK);
        expect(gt(churnOut[0].meta_information[0].value_or<std::int32_t>("active_peaks"_spmr, -1), std::int32_t{0})) << "sanity: a peak is alive before reset";

        block.reset();
        block.peak_spawn_probability = 0.0; // isolate the post-reset noise floor from a fresh spawn

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        expect(eq(out[0].meta_information[0].value_or<std::int32_t>("active_peaks"_spmr, -1), std::int32_t{0}));
        expect(out[0].timing_events[0].empty());

        auto bins   = out[0].signalValues(0);
        bool binsOk = true;
        for (std::size_t i = 0; i < kGaussianNoiseSeed42.size(); ++i) {
            binsOk = binsOk && std::abs(bins[i] - kGaussianNoiseSeed42[i]) < 1e-9;
        }
        if (!binsOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("EvolvingPeakSpectrum post-reset noise floor", bins);
        }
        for (std::size_t i = 0; i < kGaussianNoiseSeed42.size(); ++i) {
            expect(approx(bins[i], kGaussianNoiseSeed42[i], 1e-9)) << std::format("bin {}", i);
        }
    };

    "default construction reports the lifecycle-generator defaults"_test = [] {
        EvolvingPeakSpectrum<float> block;
        expect(eq(block.spectrum_size.value, gr::Size_t{1024}));
        expect(approx(block.noise_level.value, 1.f, 1e-6f));
        expect(eq(block.seed.value, std::uint64_t{42}));
        expect(eq(block.max_concurrent_peaks.value, gr::Size_t{5}));
        expect(eq(block.min_onset_spectra.value, gr::Size_t{1}));
        expect(eq(block.max_onset_spectra.value, gr::Size_t{30}));
        expect(eq(block.min_steady_spectra.value, gr::Size_t{20}));
        expect(eq(block.max_steady_spectra.value, gr::Size_t{80}));
        expect(eq(block.min_decay_spectra.value, gr::Size_t{1}));
        expect(eq(block.max_decay_spectra.value, gr::Size_t{30}));
        expect(approx(block.snr_min_db.value, 6.f, 1e-6f));
        expect(approx(block.snr_max_db.value, 40.f, 1e-6f));
        expect(approx(block.peak_spawn_probability.value, 0.1f, 1e-6f));
        expect(approx(block.max_drift_rate.value, 0.5f, 1e-6f));
        expect(block.tag_mode.value == TagMode::everySpectrum);
    };

    "processBulk with zero spawn probability is a deterministic Gaussian noise floor, no peaks ever"_test = [] {
        EvolvingPeakSpectrum<double> block;
        block.spectrum_size          = static_cast<gr::Size_t>(kGaussianNoiseSeed42.size());
        block.peak_spawn_probability = 0.0;
        block.noise_level            = 1.0;
        block.seed                   = 42ULL;
        block.start();

        std::vector<std::uint8_t>        in(3UZ, 0U);
        std::vector<gr::DataSet<double>> out(3UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        for (const auto& ds : out) {
            expect(gr::dataset::checkConsistency(ds).has_value());
            expect(eq(ds.meta_information[0].value_or<std::int32_t>("active_peaks"_spmr, -1), std::int32_t{0}));
            expect(ds.timing_events[0].empty());
        }

        auto bins   = out[0].signalValues(0);
        bool binsOk = true;
        for (std::size_t i = 0; i < kGaussianNoiseSeed42.size(); ++i) {
            binsOk = binsOk && std::abs(bins[i] - kGaussianNoiseSeed42[i]) < 1e-9;
        }
        if (!binsOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("EvolvingPeakSpectrum zero-spawn-probability noise floor", bins);
        }
        for (std::size_t i = 0; i < kGaussianNoiseSeed42.size(); ++i) {
            expect(approx(bins[i], kGaussianNoiseSeed42[i], 1e-9)) << std::format("bin {}", i);
        }
    };

    "a peak's onset/steady/decay amplitude ramp is exactly reproduced when durations and SNR are pinned"_test = [] {
        // snr_min_db == snr_max_db pins targetAmplitude = 10^(20/20) = 10 regardless of the RNG draw.
        // min == max for onset/steady/decay pins each duration to 4 spectra regardless of the RNG draw
        // (uniformInt(lo, lo+1) == lo for any u in [0,1)). max_concurrent_peaks=1 with spawn probability=1
        // guarantees exactly one peak, spawned on the very first spectrum, with no further spawns while
        // it is alive. tag_mode=everySpectrum emits one timing event per spectrum for as long as the peak
        // is active, so its currentAmplitude() progression is directly observable in the public output.
        EvolvingPeakSpectrum<double> block;
        block.spectrum_size          = 32U;
        block.noise_level            = 0.0;
        block.max_concurrent_peaks   = 1U;
        block.min_onset_spectra      = 4U;
        block.max_onset_spectra      = 4U;
        block.min_steady_spectra     = 4U;
        block.max_steady_spectra     = 4U;
        block.min_decay_spectra      = 4U;
        block.max_decay_spectra      = 4U;
        block.snr_min_db             = 20.0;
        block.snr_max_db             = 20.0;
        block.peak_spawn_probability = 1.0;
        block.max_drift_rate         = 0.0;
        block.tag_mode               = TagMode::everySpectrum;
        block.seed                   = 42ULL;
        block.start();

        constexpr std::array<double, 12> kExpectedAmplitude = {
            0.0, 10.0 / 3.0, 20.0 / 3.0, 10.0, // onset
            10.0, 10.0, 10.0, 10.0,            // steady
            10.0, 20.0 / 3.0, 10.0 / 3.0, 0.0, // decay
        };
        constexpr std::array<const char*, 12> kExpectedEvent = {
            "peak_start", "peak_active", "peak_active", "peak_active",       //
            "peak_active", "peak_active", "peak_active", "peak_active",      //
            "peak_decay_start", "peak_active", "peak_active", "peak_active", //
        };
        // meta_information["active_peaks"] is written *after* advancePeaks()/removeDeadPeaks(), whereas
        // timing_events are captured *before* that step -- so the peak's very last decay sample (11) still
        // carries a valid "peak_active" timing event at amplitude 0, yet already reports active_peaks=0
        // because the peak's decay duration elapsed and it was removed within that same processBulk call.
        constexpr std::array<std::int32_t, 12> kExpectedActivePeaksAfterAdvance = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

        std::vector<std::uint8_t>        in(kExpectedAmplitude.size(), 0U);
        std::vector<gr::DataSet<double>> out(kExpectedAmplitude.size());
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        for (std::size_t s = 0; s < out.size(); ++s) {
            const auto& ds = out[s];
            expect(gr::dataset::checkConsistency(ds).has_value());
            expect(eq(ds.meta_information[0].value_or<std::int32_t>("active_peaks"_spmr, -1), kExpectedActivePeaksAfterAdvance[s])) << std::format("sample {}", s);
            expect(eq(ds.timing_events[0].size(), 1UZ)) << std::format("sample {}", s);

            const auto& props             = ds.timing_events[0].front().second;
            const float amplitudeMeasured = props.value_or<float>("amplitude"_spmr, -1.f);
            const auto  eventName         = props.value_or<std::string_view>("event"_spmr, std::string_view{});
            const bool  sampleOk          = std::abs(amplitudeMeasured - static_cast<float>(kExpectedAmplitude[s])) < 1e-3f && eventName == std::string_view(kExpectedEvent[s]);
            if (!sampleOk || gr::testing::qa::verboseCharts()) {
                const std::array<gr::testing::qa::PeakMarkerSet, 1> markerSets{gr::testing::qa::designedPeakMarkers(ds.timing_events[0])};
                gr::testing::qa::printSpectrumChart<double>(std::format("EvolvingPeakSpectrum amplitude ramp, sample {} ({})", s, eventName), ds.signalValues(0), markerSets);
            }

            expect(approx(amplitudeMeasured, static_cast<float>(kExpectedAmplitude[s]), 1e-3f)) << std::format("sample {} amplitude", s);
            expect(eq(eventName, std::string_view(kExpectedEvent[s]))) << std::format("sample {} event", s);

            if (kExpectedAmplitude[s] == 0.0) {
                auto bins = ds.signalValues(0);
                expect(std::ranges::all_of(bins, [](double v) { return v == 0.0; })) << std::format("sample {} expected an all-zero spectrum (amplitude underflow, no noise)", s);
            }
        }
    };

    "tag_mode=transitions only emits tags at phase boundaries"_test = [] {
        EvolvingPeakSpectrum<double> block;
        block.spectrum_size          = 32U;
        block.noise_level            = 0.0;
        block.max_concurrent_peaks   = 1U;
        block.min_onset_spectra      = 4U;
        block.max_onset_spectra      = 4U;
        block.min_steady_spectra     = 4U;
        block.max_steady_spectra     = 4U;
        block.min_decay_spectra      = 4U;
        block.max_decay_spectra      = 4U;
        block.snr_min_db             = 20.0;
        block.snr_max_db             = 20.0;
        block.peak_spawn_probability = 1.0;
        block.max_drift_rate         = 0.0;
        block.tag_mode               = TagMode::transitions;
        block.seed                   = 42ULL;
        block.start();

        constexpr std::size_t kNumSamples = 12UZ; // full onset(4) + steady(4) + decay(4) lifecycle

        std::vector<std::uint8_t>        in(kNumSamples, 0U);
        std::vector<gr::DataSet<double>> out(kNumSamples);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        // transitions fire exactly at phaseCounter == 0: sample 0 (onset start), 4 (steady start), 8 (decay start)
        constexpr std::array<bool, kNumSamples> kExpectTag = {true, false, false, false, true, false, false, false, true, false, false, false};

        std::size_t totalTags = 0;
        for (std::size_t s = 0; s < kNumSamples; ++s) {
            expect(eq(out[s].timing_events[0].size(), kExpectTag[s] ? 1UZ : 0UZ)) << std::format("sample {}", s);
            totalTags += out[s].timing_events[0].size();
        }
        expect(eq(totalTags, 3UZ));
    };

    "settingsChanged on seed reseeds the RNG, matching a block constructed with that seed from the start"_test = [] {
        auto configure = [](EvolvingPeakSpectrum<double>& block) {
            block.spectrum_size          = static_cast<gr::Size_t>(kGaussianNoiseSeed42.size());
            block.peak_spawn_probability = 0.0;
        };

        EvolvingPeakSpectrum<double> blockA;
        configure(blockA);
        blockA.seed = 99ULL;
        blockA.start();

        EvolvingPeakSpectrum<double> blockB;
        configure(blockB);
        blockB.start(); // seeds _rng with the default seed = 42
        blockB.seed = 99ULL;
        blockB.settingsChanged({}, {{"seed", std::uint64_t{99}}});

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> outA(1UZ);
        std::vector<gr::DataSet<double>> outB(1UZ);
        expect(blockA.processBulk(in, outA) == gr::work::Status::OK);
        expect(blockB.processBulk(in, outB) == gr::work::Status::OK);

        auto binsA     = outA[0].signalValues(0);
        auto binsB     = outB[0].signalValues(0);
        bool binsMatch = true;
        for (std::size_t i = 0; i < binsA.size(); ++i) {
            binsMatch = binsMatch && std::abs(binsA[i] - binsB[i]) < 1e-12;
        }
        if (!binsMatch || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("EvolvingPeakSpectrum seed-reseeded spectrum (blockA, should equal blockB)", binsA);
        }
        for (std::size_t i = 0; i < binsA.size(); ++i) {
            expect(approx(binsA[i], binsB[i], 1e-12)) << std::format("bin {}", i);
        }
    };

    "processBulk handles an empty input span"_test = [] {
        EvolvingPeakSpectrum<double> block;
        block.start();

        std::vector<std::uint8_t>        in{};
        std::vector<gr::DataSet<double>> out{};
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        expect(out.empty());
    };

    "processBulk handles a single-bin spectrum boundary"_test = [] {
        EvolvingPeakSpectrum<double> block;
        block.spectrum_size          = 1U;
        block.peak_spawn_probability = 0.0;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        expect(gr::dataset::checkConsistency(out[0]).has_value());
        expect(eq(static_cast<std::size_t>(out[0].extents[0]), 1UZ));
    };
};

int main() { /* not needed for UT */ }
