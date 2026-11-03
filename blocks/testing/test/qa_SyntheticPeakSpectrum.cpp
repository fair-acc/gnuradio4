#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>

#include <gnuradio-4.0/testing/SyntheticPeakSpectrum.hpp>

#include "PeakSpectrumChart.hpp"

// derived independently of the block under test, by hand-transcribing SyntheticPeakSpectrum's
// generateBaseline()/addNoise() sequence (dc_feature_prob=0.5, noise_level=1,
// add_coloured_noise/add_edge_effects/add_spurious_spikes=false; seed 42's dc-feature draw comes
// out >= 0.5, so no DC feature is added) against a standalone gr::rng::Xoshiro256pp(42) +
// gr::rng::GaussianNoise<double>. Pins the block's bit-reproducibility contract for the peak-free
// (max_peaks=0) path.
namespace {
constexpr std::array<double, 8> kBaselinePlusNoiseSeed42 = {
    0.65944191794802076,  //
    0.96228198987926572,  //
    0.23420711648587411,  //
    0.99598923524134142,  //
    0.46702371396950032,  //
    1.2985143875705631,   //
    0.72454382639847636,  //
    0.099548273440037055, //
};
}

const boost::ut::suite<"SyntheticPeakSpectrum"> syntheticPeakSpectrumTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    constexpr auto kFloatingTypes = std::tuple<float, double>();

    "processBulk runs for both registered types, producing a consistent peak-free spectrum"_test = []<typename T>(const T&) {
        SyntheticPeakSpectrum<T> block;
        block.spectrum_size       = 8U;
        block.max_peaks           = 0U;
        block.add_coloured_noise  = false;
        block.add_edge_effects    = false;
        block.add_spurious_spikes = false;
        block.seed                = 42ULL;
        block.start();

        std::vector<std::uint8_t>   in(1UZ, 0U);
        std::vector<gr::DataSet<T>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        const auto& ds = out[0];
        expect(gr::dataset::checkConsistency(ds).has_value());

        auto axis = ds.axisValues(0);
        for (std::size_t i = 0; i < axis.size(); ++i) {
            expect(approx(axis[i], static_cast<T>(i), T(1e-6)));
        }
        expect(eq(ds.meta_information[0].template value_or<std::int32_t>("peak_count"_spmr, -1), std::int32_t{0}));
        expect(ds.timing_events[0].empty());
    } | kFloatingTypes;

    "reset() reseeds the RNG so post-reset output matches the very first spectrum again"_test = [] {
        SyntheticPeakSpectrum<double> block;
        block.spectrum_size       = static_cast<gr::Size_t>(kBaselinePlusNoiseSeed42.size());
        block.max_peaks           = 0U;
        block.add_coloured_noise  = false;
        block.add_edge_effects    = false;
        block.add_spurious_spikes = false;
        block.seed                = 42ULL;
        block.start();

        std::vector<std::uint8_t>        churnIn(4UZ, 0U);
        std::vector<gr::DataSet<double>> churnOut(4UZ);
        expect(block.processBulk(churnIn, churnOut) == gr::work::Status::OK); // advance the RNG well past the first spectrum

        block.reset();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        auto bins   = out[0].signalValues(0);
        bool binsOk = true;
        for (std::size_t i = 0; i < kBaselinePlusNoiseSeed42.size(); ++i) {
            binsOk = binsOk && std::abs(bins[i] - kBaselinePlusNoiseSeed42[i]) < 1e-9;
        }
        if (!binsOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("SyntheticPeakSpectrum post-reset baseline+noise", bins);
        }
        for (std::size_t i = 0; i < kBaselinePlusNoiseSeed42.size(); ++i) {
            expect(approx(bins[i], kBaselinePlusNoiseSeed42[i], 1e-9)) << std::format("bin {}", i);
        }
    };

    "default construction reports the ex1_training.py-matching defaults"_test = [] {
        SyntheticPeakSpectrum<float> block;
        expect(eq(block.spectrum_size.value, gr::Size_t{1024}));
        expect(eq(block.max_peaks.value, gr::Size_t{8}));
        expect(approx(block.snr_min_db.value, 6.f, 1e-6f));
        expect(approx(block.snr_max_db.value, 40.f, 1e-6f));
        expect(approx(block.noise_level.value, 1.f, 1e-6f));
        expect(approx(block.peak_width_min_bins.value, 1.f, 1e-6f));
        expect(approx(block.peak_width_max_bins.value, 250.f, 1e-6f));
        expect(approx(block.width_tail_exponent.value, 4.f, 1e-6f));
        expect(approx(block.dc_feature_prob.value, 0.5f, 1e-6f));
        expect(approx(block.dc_amp_max.value, 1.f, 1e-6f));
        expect(approx(block.dc_decay_min_bins.value, 2.f, 1e-6f));
        expect(approx(block.dc_decay_max_bins.value, 20.f, 1e-6f));
        expect(block.add_coloured_noise.value);
        expect(block.add_edge_effects.value);
        expect(block.add_spurious_spikes.value);
        expect(eq(block.seed.value, std::uint64_t{42}));
    };

    "processBulk with max_peaks=0 is a deterministic, peak-free baseline-plus-noise spectrum for a fixed seed"_test = [] {
        // uniformInt(0, max_peaks+1) with max_peaks=0 collapses to floor(u*1) == 0 for any u in [0,1),
        // so nPeaks is guaranteed 0 regardless of the RNG draw -- no peak-shape logic to replicate.
        SyntheticPeakSpectrum<double> block;
        block.spectrum_size       = static_cast<gr::Size_t>(kBaselinePlusNoiseSeed42.size());
        block.max_peaks           = 0U;
        block.add_coloured_noise  = false;
        block.add_edge_effects    = false;
        block.add_spurious_spikes = false;
        block.seed                = 42ULL;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        const auto& ds = out[0];
        expect(gr::dataset::checkConsistency(ds).has_value());

        auto axis = ds.axisValues(0);
        for (std::size_t i = 0; i < axis.size(); ++i) {
            expect(approx(axis[i], static_cast<double>(i), 1e-12));
        }

        auto bins   = ds.signalValues(0);
        bool binsOk = true;
        for (std::size_t i = 0; i < kBaselinePlusNoiseSeed42.size(); ++i) {
            binsOk = binsOk && std::abs(bins[i] - kBaselinePlusNoiseSeed42[i]) < 1e-9;
        }
        if (!binsOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("SyntheticPeakSpectrum peak-free baseline+noise", bins);
        }
        for (std::size_t i = 0; i < kBaselinePlusNoiseSeed42.size(); ++i) {
            expect(approx(bins[i], kBaselinePlusNoiseSeed42[i], 1e-9)) << std::format("bin {}", i);
        }

        expect(eq(ds.meta_information[0].value_or<std::int32_t>("peak_count"_spmr, -1), std::int32_t{0}));
        expect(ds.timing_events[0].empty());
    };

    "peak metadata is self-consistent with the documented SNR-to-amplitude conversion"_test = [] {
        SyntheticPeakSpectrum<double> block;
        block.spectrum_size = 128U;
        block.max_peaks     = 5U;
        block.snr_min_db    = 6.0;
        block.snr_max_db    = 40.0;
        block.seed          = 42ULL;
        block.start();

        static constexpr std::array<std::string_view, 7> kKnownShapes = {"gaussian", "asymmetric_gaussian", "lorentzian", "sinc2", "parabolic", "pseudo_voigt", "dual_gaussian"};

        std::vector<std::uint8_t>        in(20UZ, 0U);
        std::vector<gr::DataSet<double>> out(20UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        bool sawAtLeastOnePeak = false;
        for (const auto& ds : out) {
            expect(gr::dataset::checkConsistency(ds).has_value());

            const auto peakCount = ds.meta_information[0].value_or<std::int32_t>("peak_count"_spmr, -1);
            expect(ge(peakCount, std::int32_t{0}));
            expect(le(peakCount, std::int32_t{5}));
            expect(eq(static_cast<std::size_t>(peakCount), ds.timing_events[0].size()));

            bool dsOk = true;
            for (const auto& [idx, props] : ds.timing_events[0]) {
                sawAtLeastOnePeak     = true;
                const float centre    = props.value_or<float>("centre"_spmr, -1.f);
                const float amplitude = props.value_or<float>("amplitude"_spmr, -1.f);
                const float snrDb     = props.value_or<float>("snr_db"_spmr, 0.f);
                const auto  shape     = props.value_or<std::string_view>("shape"_spmr, std::string_view{});

                dsOk = dsOk && centre >= 0.f && centre < 128.f && snrDb >= 6.f && snrDb <= 40.f //
                       && std::abs(amplitude - std::pow(10.f, snrDb / 20.f)) < 1e-3f && std::ranges::contains(kKnownShapes, shape) && idx == static_cast<std::ptrdiff_t>(centre);

                expect(ge(centre, 0.f));
                expect(lt(centre, 128.f));
                expect(ge(snrDb, 6.f));
                expect(le(snrDb, 40.f));
                expect(approx(amplitude, std::pow(10.f, snrDb / 20.f), 1e-3f)) << std::format("amplitude<->SNR mismatch for shape {}", shape);
                expect(std::ranges::contains(kKnownShapes, shape)) << std::format("unknown shape '{}'", shape);
                expect(eq(idx, static_cast<std::ptrdiff_t>(centre)));
            }

            if (!ds.timing_events[0].empty() && (!dsOk || gr::testing::qa::verboseCharts())) {
                const std::array<gr::testing::qa::PeakMarkerSet, 1> markerSets{gr::testing::qa::designedPeakMarkers(ds.timing_events[0])};
                gr::testing::qa::printSpectrumChart<double>("SyntheticPeakSpectrum peak metadata scene", ds.signalValues(0), markerSets);
            }
        }
        expect(sawAtLeastOnePeak) << "expected at least one peak across 20 spectra at seed 42";
    };

    "processBulk output is bit-reproducible for two independently constructed blocks sharing a seed"_test = [] {
        auto configure = [](SyntheticPeakSpectrum<double>& block) {
            block.spectrum_size = 64U;
            block.max_peaks     = 4U;
            block.seed          = 7ULL;
            block.start();
        };

        SyntheticPeakSpectrum<double> blockA;
        SyntheticPeakSpectrum<double> blockB;
        configure(blockA);
        configure(blockB);

        std::vector<std::uint8_t>        in(5UZ, 0U);
        std::vector<gr::DataSet<double>> outA(5UZ);
        std::vector<gr::DataSet<double>> outB(5UZ);
        expect(blockA.processBulk(in, outA) == gr::work::Status::OK);
        expect(blockB.processBulk(in, outB) == gr::work::Status::OK);

        for (std::size_t s = 0; s < outA.size(); ++s) {
            auto binsA = outA[s].signalValues(0);
            auto binsB = outB[s].signalValues(0);

            bool spectrumMatch = binsA.size() == binsB.size() && outA[s].timing_events[0].size() == outB[s].timing_events[0].size();
            for (std::size_t i = 0; spectrumMatch && i < binsA.size(); ++i) {
                spectrumMatch = std::abs(binsA[i] - binsB[i]) < 1e-12;
            }
            if (!spectrumMatch || gr::testing::qa::verboseCharts()) {
                const std::array<gr::testing::qa::PeakMarkerSet, 1> markerSets{gr::testing::qa::designedPeakMarkers(outA[s].timing_events[0])};
                gr::testing::qa::printSpectrumChart<double>(std::format("SyntheticPeakSpectrum bit-reproducibility, spectrum {} (blockA, should equal blockB)", s), binsA, markerSets);
            }

            expect(eq(binsA.size(), binsB.size()));
            for (std::size_t i = 0; i < binsA.size(); ++i) {
                expect(approx(binsA[i], binsB[i], 1e-12)) << std::format("spectrum {} bin {}", s, i);
            }
            expect(eq(outA[s].timing_events[0].size(), outB[s].timing_events[0].size()));
        }
    };

    "settingsChanged on seed reseeds the RNG, matching a block constructed with that seed from the start"_test = [] {
        auto configure = [](SyntheticPeakSpectrum<double>& block) {
            block.spectrum_size       = static_cast<gr::Size_t>(kBaselinePlusNoiseSeed42.size());
            block.max_peaks           = 0U;
            block.add_coloured_noise  = false;
            block.add_edge_effects    = false;
            block.add_spurious_spikes = false;
        };

        SyntheticPeakSpectrum<double> blockA;
        configure(blockA);
        blockA.seed = 99ULL;
        blockA.start();

        SyntheticPeakSpectrum<double> blockB;
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
            gr::testing::qa::printSpectrumChart<double>("SyntheticPeakSpectrum seed-reseeded spectrum (blockA, should equal blockB)", binsA);
        }
        for (std::size_t i = 0; i < binsA.size(); ++i) {
            expect(approx(binsA[i], binsB[i], 1e-12)) << std::format("bin {}", i);
        }
    };

    "processBulk handles an empty input span"_test = [] {
        SyntheticPeakSpectrum<double> block;
        block.start();

        std::vector<std::uint8_t>        in{};
        std::vector<gr::DataSet<double>> out{};
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        expect(out.empty());
    };

    "processBulk handles a single-bin spectrum boundary"_test = [] {
        SyntheticPeakSpectrum<double> block;
        block.spectrum_size = 1U;
        block.max_peaks     = 0U;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        expect(gr::dataset::checkConsistency(out[0]).has_value());
        expect(eq(static_cast<std::size_t>(out[0].extents[0]), 1UZ));
    };
};

int main() { /* not needed for UT */ }
