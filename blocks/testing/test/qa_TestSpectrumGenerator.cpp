#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetHelper.hpp>

#include <gnuradio-4.0/testing/TestSpectrumGenerator.hpp>

#include "PeakSpectrumChart.hpp"

// Golden values below are derived independently of the block under test, from the documented
// gr::rng::Xoshiro256pp algorithm (SplitMix64 seeding + xoshiro256++ core, see
// algorithm/include/gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp), by seeding a standalone
// gr::rng::Xoshiro256pp(42) and reading off successive triangularM11<double>() draws. They pin
// TestSpectrumGenerator's bit-reproducibility contract rather than merely re-deriving it from the
// block's own code.
namespace {
constexpr std::array<double, 8> kTriangularM11Seed42 = {
    0.13312618518457109,   //
    0.68502976631224444,   //
    0.38160295615928863,   //
    -0.26952510928008533,  //
    0.14106428926815218,   //
    0.40956929896059657,   //
    -0.25046305834752536,  //
    -0.050922859983867363, //
};
}

const boost::ut::suite<"TestSpectrumGenerator"> testSpectrumGeneratorTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    constexpr auto kFloatingTypes = std::tuple<float, double>();

    "processBulk runs for both registered types, using values that are exact in float and double alike"_test = []<typename T>(const T&) {
        // Schottky peak centre (dist=0 -> signalDb = peakDb exactly, no noise/RNG dependence).
        TestSpectrumGenerator<T> peakBlock;
        peakBlock.spectrum_size           = 16U;
        peakBlock.show_schottky           = true;
        peakBlock.show_sweep_line         = false;
        peakBlock.show_interference_lines = false;
        peakBlock.noise_floor_db          = T(-80);
        peakBlock.initial_peak_db         = T(6);
        peakBlock.seed                    = 42ULL;
        peakBlock.start();

        std::vector<std::uint8_t>   in(1UZ, 0U);
        std::vector<gr::DataSet<T>> peakOut(1UZ);
        expect(peakBlock.processBulk(in, peakOut) == gr::work::Status::OK);
        const bool schottkyOk = std::abs(static_cast<double>(peakOut[0].signalValues(0)[8]) - (-74.0)) < 1e-4;
        if (!schottkyOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<T>("TestSpectrumGenerator Schottky peak centre bin", peakOut[0].signalValues(0));
        }
        expect(approx(peakOut[0].signalValues(0)[8], T(-74), T(1e-4)));

        // Narrow interference line centre (kLinePositions[1] == 0.25 -> exact bin 4, dist=0 -> amplitudeDb exactly).
        TestSpectrumGenerator<T> lineBlock;
        lineBlock.spectrum_size           = 16U;
        lineBlock.show_schottky           = false;
        lineBlock.show_sweep_line         = false;
        lineBlock.show_interference_lines = true;
        lineBlock.noise_floor_db          = T(-80);
        lineBlock.line_amplitude_db       = T(12);
        lineBlock.seed                    = 42ULL;
        lineBlock.start();

        std::vector<gr::DataSet<T>> lineOut(1UZ);
        expect(lineBlock.processBulk(in, lineOut) == gr::work::Status::OK);
        const bool lineOk = std::abs(static_cast<double>(lineOut[0].signalValues(0)[4]) - (-68.0)) < 1e-4;
        if (!lineOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<T>("TestSpectrumGenerator narrow interference line centre bin", lineOut[0].signalValues(0));
        }
        expect(approx(lineOut[0].signalValues(0)[4], T(-68), T(1e-4)));
    } | kFloatingTypes;

    "reset() reseeds the RNG so post-reset output matches the very first spectrum again"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size           = static_cast<gr::Size_t>(kTriangularM11Seed42.size());
        block.show_schottky           = false;
        block.show_sweep_line         = false;
        block.show_interference_lines = false;
        block.noise_floor_db          = -80.0;
        block.noise_spread_db         = 0.2;
        block.seed                    = 42ULL;
        block.start();

        std::vector<std::uint8_t>        churnIn(5UZ, 0U);
        std::vector<gr::DataSet<double>> churnOut(5UZ);
        expect(block.processBulk(churnIn, churnOut) == gr::work::Status::OK); // advance the RNG well past the first spectrum

        block.reset();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        auto bins   = out[0].signalValues(0);
        bool binsOk = true;
        for (std::size_t i = 0; i < kTriangularM11Seed42.size(); ++i) {
            binsOk = binsOk && std::abs(bins[i] - (-80.0 + kTriangularM11Seed42[i] * 0.2)) < 1e-9;
        }
        if (!binsOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator post-reset noise floor", bins);
        }
        for (std::size_t i = 0; i < kTriangularM11Seed42.size(); ++i) {
            const double expected = -80.0 + kTriangularM11Seed42[i] * 0.2;
            expect(approx(bins[i], expected, 1e-9)) << std::format("bin {}", i);
        }
    };

    "default construction reports the OpenDigitizer beam-spectrum defaults"_test = [] {
        TestSpectrumGenerator<float> block;
        expect(eq(block.spectrum_size.value, gr::Size_t{4096}));
        expect(approx(block.centre_freq.value, 100e6f, 1.f));
        expect(approx(block.signal_bandwidth.value, 1e6f, 1.f));
        expect(approx(block.clock_rate.value, 25.f, 1e-6f));
        expect(eq(block.seed.value, std::uint64_t{42}));
        expect(approx(block.active_duration.value, 10.f, 1e-6f));
        expect(approx(block.pause_duration.value, 1.f, 1e-6f));
        expect(approx(block.noise_floor_db.value, -80.f, 1e-6f));
        expect(approx(block.noise_spread_db.value, 0.2f, 1e-6f));
        expect(block.show_schottky.value);
        expect(block.show_sweep_line.value);
        expect(block.show_interference_lines.value);
        expect(eq(block.log_interval.value, gr::Size_t{0}));
    };

    "processBulk emits one DataSet<T> per input tick with linear frequency axis and correct metadata"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size           = 8U;
        block.centre_freq             = 1000.0;
        block.signal_bandwidth        = 800.0;
        block.clock_rate              = 25.0;
        block.show_schottky           = false;
        block.show_sweep_line         = false;
        block.show_interference_lines = false;
        block.start();

        std::vector<std::uint8_t>        in(3UZ, 0U);
        std::vector<gr::DataSet<double>> out(3UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        constexpr std::array<double, 8> kExpectedAxis = {600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0};

        for (const auto& ds : out) {
            expect(gr::dataset::checkConsistency(ds).has_value());
            expect(eq(ds.extents.size(), 1UZ));
            expect(eq(static_cast<std::size_t>(ds.extents[0]), 8UZ));
            expect(eq(ds.axis_names[0], std::string("Frequency")));
            expect(eq(ds.axis_units[0], std::string("Hz")));

            auto axis = ds.axisValues(0);
            for (std::size_t i = 0; i < kExpectedAxis.size(); ++i) {
                expect(approx(axis[i], kExpectedAxis[i], 1e-9));
            }

            expect(eq(ds.meta_information.size(), 1UZ));
            expect(approx(ds.meta_information[0].value_or<float>("sample_rate"_spmr, 0.f), 800.f, 1e-3f));
            expect(approx(ds.meta_information[0].value_or<float>("centre_frequency"_spmr, 0.f), 1000.f, 1e-3f));
            expect(ds.meta_information[0].value_or<bool>("output_in_db"_spmr, false));
            expect(approx(ds.meta_information[0].value_or<float>("clock_rate"_spmr, 0.f), 25.f, 1e-3f));

            expect(eq(ds.signal_ranges.size(), 1UZ));
            expect(le(ds.signal_ranges[0].min, ds.signal_ranges[0].max));
        }
    };

    "noise floor bins are bit-reproducible for a fixed seed via Xoshiro256++"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size           = static_cast<gr::Size_t>(kTriangularM11Seed42.size());
        block.show_schottky           = false;
        block.show_sweep_line         = false;
        block.show_interference_lines = false;
        block.noise_floor_db          = -80.0;
        block.noise_spread_db         = 0.2;
        block.seed                    = 42ULL;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        auto bins   = out[0].signalValues(0);
        bool binsOk = true;
        for (std::size_t i = 0; i < kTriangularM11Seed42.size(); ++i) {
            binsOk = binsOk && std::abs(bins[i] - (-80.0 + kTriangularM11Seed42[i] * 0.2)) < 1e-9;
        }
        if (!binsOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator noise floor", bins);
        }
        for (std::size_t i = 0; i < kTriangularM11Seed42.size(); ++i) {
            const double expected = -80.0 + kTriangularM11Seed42[i] * 0.2;
            expect(approx(bins[i], expected, 1e-9)) << std::format("noise bin {}", i);
        }
    };

    "the Schottky peak reaches its analytically-known amplitude and centre bin at t=0"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size           = 16U;
        block.show_schottky           = true;
        block.show_sweep_line         = false;
        block.show_interference_lines = false;
        block.noise_floor_db          = -80.0;
        block.noise_spread_db         = 0.2;
        block.initial_peak_db         = 6.0;
        block.initial_sigma           = 0.1;
        block.width_ratio             = 10.0;
        block.active_duration         = 10.0;
        block.seed                    = 42ULL;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        auto bins = out[0].signalValues(0);

        // peakDb = noise_floor_db + initial_peak_db + t = -80 + 6 + 0 = -74 at the centre bin (0.5 * N = 8),
        // decaying as signalDb = peakDb - dist^2 / (2*sigma^2) * (10 / ln 10), sigma = initial_sigma * N = 1.6.
        // These values comfortably exceed the +-0.2 dB noise floor at |dist| <= 2, so the max() in
        // addSchottkyPeak deterministically picks the peak regardless of the noise draw at that bin.
        const bool peakOk = std::abs(bins[6] - (-77.392925639869148)) < 1e-9 && std::abs(bins[7] - (-74.848231409967283)) < 1e-9 && std::abs(bins[8] - (-74.0)) < 1e-9 //
                            && std::abs(bins[9] - (-74.848231409967283)) < 1e-9 && std::abs(bins[10] - (-77.392925639869148)) < 1e-9                                   //
                            && std::ranges::distance(bins.begin(), std::ranges::max_element(bins)) == std::ptrdiff_t{8};
        if (!peakOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator Schottky peak at t=0", bins);
        }
        expect(approx(bins[6], -77.392925639869148, 1e-9));
        expect(approx(bins[7], -74.848231409967283, 1e-9));
        expect(approx(bins[8], -74.0, 1e-9));
        expect(approx(bins[9], -74.848231409967283, 1e-9));
        expect(approx(bins[10], -77.392925639869148, 1e-9));

        expect(eq(std::ranges::distance(bins.begin(), std::ranges::max_element(bins)), std::ptrdiff_t{8}));
    };

    "a narrow interference line reaches its configured amplitude exactly at its centre bin"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size           = 16U; // kLinePositions[1] == 0.25 -> exact bin 4
        block.show_schottky           = false;
        block.show_sweep_line         = false;
        block.show_interference_lines = true;
        block.noise_floor_db          = -80.0;
        block.line_amplitude_db       = 12.0;
        block.seed                    = 42ULL;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);

        auto       bins   = out[0].signalValues(0);
        const bool lineOk = std::abs(bins[4] - (-68.0)) < 1e-9;
        if (!lineOk || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator narrow interference line", bins);
        }
        expect(approx(bins[4], -68.0, 1e-9));
    };

    "settingsChanged on seed reseeds the RNG, matching a block constructed with that seed from the start"_test = [] {
        auto configure = [](TestSpectrumGenerator<double>& block) {
            block.spectrum_size           = static_cast<gr::Size_t>(kTriangularM11Seed42.size());
            block.show_schottky           = false;
            block.show_sweep_line         = false;
            block.show_interference_lines = false;
        };

        TestSpectrumGenerator<double> blockA;
        configure(blockA);
        blockA.seed = 99ULL;
        blockA.start();

        TestSpectrumGenerator<double> blockB;
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
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator seed-reseeded spectrum (blockA, should equal blockB)", binsA);
        }
        for (std::size_t i = 0; i < binsA.size(); ++i) {
            expect(approx(binsA[i], binsB[i], 1e-12)) << std::format("bin {}", i);
        }
    };

    "settingsChanged on morse_pattern rebuilds the key so an empty pattern keeps the keyed line always on"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size           = 20U; // kLinePositions[2] == 0.85 -> exact bin 17
        block.show_schottky           = false;
        block.show_sweep_line         = false;
        block.show_interference_lines = true;
        block.noise_floor_db          = -80.0;
        block.noise_spread_db         = 0.2;
        block.line_amplitude_db       = 12.0;
        block.clock_rate              = 25.0;
        block.morse_unit_duration     = 0.2;
        block.seed                    = 42ULL;
        block.start(); // builds the default "HELLO FOIL!" key, whose first unit is a keyed dot

        // sample index 5 -> elapsed = 5 / clock_rate = 0.2 s -> the second morse unit, which is the
        // (always-off) inter-symbol gap after the first dot: the keyed line must be silent there.
        std::vector<std::uint8_t>        in(6UZ, 0U);
        std::vector<gr::DataSet<double>> out(6UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        if (out[5].signalValues(0)[17] >= -75.0 || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator morse key, gap sample (line should be silent)", out[5].signalValues(0));
        }
        expect(lt(out[5].signalValues(0)[17], -75.0)) << "third line silent under the default morse pattern";

        block.morse_pattern = std::string{};
        block.settingsChanged({}, {{"morse_pattern", std::string{}}});

        std::vector<std::uint8_t>        in2(1UZ, 0U);
        std::vector<gr::DataSet<double>> out2(1UZ);
        expect(block.processBulk(in2, out2) == gr::work::Status::OK);
        if (std::abs(out2[0].signalValues(0)[17] - (-68.0)) >= 1e-9 || gr::testing::qa::verboseCharts()) {
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator morse key, empty pattern (line always on)", out2[0].signalValues(0));
        }
        expect(approx(out2[0].signalValues(0)[17], -68.0, 1e-9)) << "third line always on once the key is empty";
    };

    "processBulk handles an empty input span"_test = [] {
        TestSpectrumGenerator<double> block;
        block.start();

        std::vector<std::uint8_t>        in{};
        std::vector<gr::DataSet<double>> out{};
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        expect(out.empty());
    };

    "processBulk handles a single-bin spectrum boundary"_test = [] {
        TestSpectrumGenerator<double> block;
        block.spectrum_size = 1U;
        block.start();

        std::vector<std::uint8_t>        in(1UZ, 0U);
        std::vector<gr::DataSet<double>> out(1UZ);
        expect(block.processBulk(in, out) == gr::work::Status::OK);
        expect(gr::dataset::checkConsistency(out[0]).has_value());
        expect(eq(static_cast<std::size_t>(out[0].extents[0]), 1UZ));

        if (gr::testing::qa::verboseCharts()) { // exercises the single-bin, degenerate-axis chart path
            gr::testing::qa::printSpectrumChart<double>("TestSpectrumGenerator single-bin spectrum", out[0].signalValues(0));
        }
    };
};

int main() { /* not needed for UT */ }
