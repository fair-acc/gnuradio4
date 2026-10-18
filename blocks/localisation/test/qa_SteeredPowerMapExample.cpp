#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include "../src/SteeredPowerMapExample.hpp"

const boost::ut::suite<"FrontPowerMap"> kAngleTests = [] {
    using namespace boost::ut;

    "front, right, left, up and down have independently known display angles"_test = [] {
        constexpr std::array<std::array<double, 3>, 5> kDirections{{{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0}}};
        constexpr std::array<std::array<double, 2>, 5> kExpected{{{0.0, 0.0}, {90.0, 0.0}, {-90.0, 0.0}, {0.0, 90.0}, {0.0, -90.0}}};
        for (std::size_t i = 0UZ; i < kDirections.size(); ++i) {
            const auto angles = gr::examples::frontAnglesDegrees(kDirections[i]);
            expect(approx(angles[0UZ], kExpected[i][0UZ], 1e-12));
            expect(approx(angles[1UZ], kExpected[i][1UZ], 1e-12));
        }
    };

    "vertical poles from float scientific coordinates have a canonical zero bearing"_test = [] {
        for (const float azimuth : {-std::numbers::pi_v<float> / 2.f, std::numbers::pi_v<float> / 2.f}) {
            const auto direction = gr::algorithm::directionOf(static_cast<double>(azimuth), 0.0);
            const auto angles    = gr::examples::frontAnglesDegrees(direction);
            expect(eq(angles[0UZ], 0.0));
            expect(eq(angles[1UZ], std::copysign(90.0, static_cast<double>(azimuth))));
        }
    };

    "diagonals and a 3-4-5 direction obey conventional bearing and elevation"_test = [] {
        const double root = std::sqrt(0.5);
        for (const double signX : {-1.0, 1.0}) {
            for (const double signY : {-1.0, 1.0}) {
                const auto angles = gr::examples::frontAnglesDegrees({signX * 0.5, signY * root, 0.5});
                expect(approx(angles[0UZ], signX * 45.0, 1e-12));
                expect(approx(angles[1UZ], signY * 45.0, 1e-12));
            }
        }
        const auto angles = gr::examples::frontAnglesDegrees({0.6, 0.0, 0.8});
        expect(approx(angles[0UZ], 36.86989764584402, 1e-12));
        expect(approx(angles[1UZ], 0.0, 1e-12));
    };

    "spherical map samples move to front-centred cardinal and diagonal cells without inventing rear coverage"_test = [] {
        constexpr std::size_t kAzimuthSteps   = 72UZ;
        constexpr std::size_t kElevationSteps = 19UZ;
        constexpr std::size_t kSide           = 37UZ;
        std::vector<float>    power(kAzimuthSteps * kElevationSteps);
        for (std::size_t row = 0UZ; row < kElevationSteps; ++row) {
            for (std::size_t column = 0UZ; column < kAzimuthSteps; ++column) {
                power[row * kAzimuthSteps + column] = static_cast<float>(1000UZ * row + column);
            }
        }
        const auto map = gr::examples::projectFrontPower(power, kAzimuthSteps, kElevationSteps);
        expect(eq(map.anglesDegrees.size(), kSide));
        expect(eq(map.power.size(), kSide * kSide));
        expect(eq(map.anglesDegrees.front(), -90.0));
        expect(eq(map.anglesDegrees[18UZ], 0.0));
        expect(eq(map.anglesDegrees.back(), 90.0));
        expect(eq(map.power[18UZ * kSide + 18UZ], 18036.f));
        expect(eq(map.power[18UZ * kSide + 36UZ], 36.f));
        expect(eq(map.power[18UZ * kSide], 0.f));
        expect(eq(map.power[36UZ * kSide + 18UZ], 54.f));
        expect(eq(map.power[18UZ], 18.f));
        expect(eq(map.power[27UZ * kSide + 27UZ], 6047.f));
        expect(eq(map.power[9UZ * kSide + 9UZ], 6011.f));
        expect(std::ranges::all_of(map.power, [](float value) { return std::isfinite(value); }));
        for (std::size_t i = 1UZ; i < kSide; ++i) {
            expect(eq(map.anglesDegrees[i] - map.anglesDegrees[i - 1UZ], 5.0));
        }
    };

    "a front-only maximum survives regridding and terminal sampling before markers are drawn"_test = [] {
        std::vector<float> power(19UZ * 72UZ, 0.f);
        std::fill_n(power.begin() + 18UZ * 72UZ, 72UZ, 1.f);
        const auto                                                                             map = gr::examples::projectFrontPower(power, 72UZ, 19UZ);
        gr::graphs::ImChart<gr::examples::kMapChartColumns, gr::examples::kAngularHeatmapRows> chart({{-90.0, 90.0}, {-90.0, 90.0}});
        expect(chart.drawHeatmap(std::span<const float>(map.power), 37UZ, 37UZ, {0.0, 1.0}, {}, gr::graphs::HeatmapStyle::monochrome).has_value());
        expect(std::ranges::any_of(chart._screen[24UZ], [](const auto& cell) { return cell == "⣿"; }));
        std::string horizontalLabels;
        for (const auto& cell : chart._screen[49UZ]) {
            horizontalLabels += cell;
        }
        expect(horizontalLabels.contains("-90"));
        expect(horizontalLabels.contains("+0"));
        expect(horizontalLabels.contains("+90"));
        expect(horizontalLabels.contains("-45"));
        expect(horizontalLabels.contains("+45"));
        const std::vector<float> constant(37UZ * 37UZ, 1.f);
        expect(chart.drawHeatmap(std::span<const float>(constant), 37UZ, 37UZ, {0.0, 1.0}, {}, gr::graphs::HeatmapStyle::monochrome).has_value());
        for (const std::size_t row : {1UZ, 24UZ, 47UZ}) {
            for (const std::size_t column : {6UZ, 52UZ, 98UZ}) {
                expect(eq(chart._screen[row][column], std::string("⣿")));
            }
        }
    };

    "a scientific peak at elevation 20 and azimuth 35 reaches bearing 65 and elevation 35"_test = [] {
        std::vector<float> power(19UZ * 72UZ, -3.f);
        power[4UZ * 72UZ + 43UZ] = 9.f;
        const auto map           = gr::examples::projectFrontPower(power, 72UZ, 19UZ);
        expect(eq(map.power[25UZ * 37UZ + 31UZ], 9.f));
        expect(eq(*std::ranges::max_element(map.power), 9.f));
        expect(eq(*std::ranges::min_element(map.power), -3.f));
    };

    "scientific directions that share a display bin retain their strongest signed score"_test = [] {
        for (const bool swap : {false, true}) {
            std::vector<float> power(19UZ * 72UZ, -9.f);
            power[16UZ * 72UZ]       = swap ? -7.f : -1.f;
            power[16UZ * 72UZ + 1UZ] = swap ? -1.f : -7.f;
            const auto map           = gr::examples::projectFrontPower(power, 72UZ, 19UZ);
            expect(eq(map.power[18UZ * 37UZ + 16UZ], -1.f));
            expect(eq(*std::ranges::max_element(map.power), -1.f));
        }
    };

    "each distinct scientific direction can contribute an isolated maximum"_test = [] {
        std::vector<float> power(19UZ * 72UZ, 0.f);
        for (std::size_t cell = 0UZ; cell < 18UZ * 72UZ; ++cell) {
            power[cell]    = 1.f;
            const auto map = gr::examples::projectFrontPower(power, 72UZ, 19UZ);
            expect(eq(*std::ranges::max_element(map.power), 1.f)) << cell;
            power[cell] = 0.f;
        }
        power[18UZ * 72UZ + 36UZ] = 1.f;
        const auto zenith         = gr::examples::projectFrontPower(power, 72UZ, 19UZ);
        expect(eq(zenith.power[18UZ * 37UZ + 18UZ], 1.f));
    };

    "signed constant power remains signed over the measured hemisphere"_test = [] {
        const std::vector<float> power(19UZ * 72UZ, -2.5f);
        const auto               map = gr::examples::projectFrontPower(power, 72UZ, 19UZ);
        expect(std::ranges::all_of(map.power, [](float value) { return value == -2.5f; }));
    };
};

const boost::ut::suite<"stereo sweep and cross-check arithmetic"> kSweepTests = [] {
    using namespace boost::ut;
    constexpr double kDegrees = 180.0 / std::numbers::pi;

    "the sweep starts at minus eighty degrees, crosses zero mid-scene, ends just below plus eighty and repeats"_test = [] {
        constexpr std::size_t kScene = 1000UZ;
        expect(approx(gr::examples::sweepBearingRadians(0UZ, kScene) * kDegrees, -80.0, 1e-12));
        expect(approx(gr::examples::sweepBearingRadians(kScene / 2UZ, kScene) * kDegrees, 0.0, 1e-12));
        expect(approx(gr::examples::sweepBearingRadians(kScene - 1UZ, kScene) * kDegrees, 80.0 * (1.0 - 2.0 / 1000.0), 1e-12));
        expect(approx(gr::examples::sweepBearingRadians(250UZ, kScene) * kDegrees, -40.0, 1e-12));
        expect(eq(gr::examples::sweepBearingRadians(kScene + 250UZ, kScene), gr::examples::sweepBearingRadians(250UZ, kScene)));
    };

    "angular separation is zero for equal, ninety for orthogonal, one-eighty for opposite and 36.87 for a 3-4-5 pair"_test = [] {
        constexpr std::array<double, 3> kX{1.0, 0.0, 0.0};
        constexpr std::array<double, 3> kY{0.0, 1.0, 0.0};
        constexpr std::array<double, 3> kMinusX{-1.0, 0.0, 0.0};
        constexpr std::array<double, 3> kThreeFourFive{0.8, 0.6, 0.0};
        expect(approx(gr::examples::angularSeparationDegrees(kX, kX), 0.0, 1e-9));
        expect(approx(gr::examples::angularSeparationDegrees(kX, kY), 90.0, 1e-9));
        expect(approx(gr::examples::angularSeparationDegrees(kX, kMinusX), 180.0, 1e-9));
        expect(approx(gr::examples::angularSeparationDegrees(kX, kThreeFourFive), std::atan2(0.6, 0.8) * kDegrees, 1e-9));
        expect(eq(gr::examples::angularSeparationDegrees(kThreeFourFive, kX), gr::examples::angularSeparationDegrees(kX, kThreeFourFive)));
    };
};

const boost::ut::suite<"band steered power and bearing verdict"> kBandTests = [] {
    using namespace boost::ut;
    constexpr double      kSpeed          = 343.0;
    constexpr double      kSpacing        = 0.3;
    constexpr std::size_t kBearings       = 181UZ;
    constexpr std::size_t kBins           = 257UZ;
    constexpr double      kBinHz          = 48000.0 / 512.0;
    constexpr double      kNyquistHz      = 24000.0;
    constexpr std::size_t kRightIndex     = 113UZ; // +23 deg on the -90..+90 deg axis in 1 deg steps
    constexpr std::size_t kNyquistBands   = 12UZ;  // 0..24 kHz in 2 kHz bands
    constexpr std::size_t kFourHertzBands = 16UZ;  // 0..64 Hz
    const auto            bearings        = std::views::iota(0UZ, kBearings) | std::views::transform([](std::size_t b) { return -90.0 + static_cast<double>(b); }) | std::ranges::to<std::vector>();
    const auto            stereo          = gr::algorithm::geometry::line(2UZ, kSpacing);
    const auto            stereoDelays    = gr::examples::bearingPairDelays(std::span<const double>(stereo.data(), stereo.size()), bearings, kSpeed);
    // a source spectrum S_k of uneven magnitude and phase; element 0 hears it tau0 later: X_0 = S e^{-j 2 pi f tau0}, X_1 = S
    const auto delayedPair = [](std::size_t nBins, double binHz, double tau0, std::size_t nFrames) {
        std::vector<std::complex<float>> spectra;
        for (std::size_t frame = 0UZ; frame < nFrames; ++frame) {
            std::vector<std::complex<float>> source(nBins);
            for (std::size_t k = 0UZ; k < nBins; ++k) {
                source[k] = std::polar(static_cast<float>(1UZ + (k + frame) % 3UZ), 0.7f * static_cast<float>(k * (frame + 1UZ)));
            }
            for (std::size_t k = 0UZ; k < nBins; ++k) {
                spectra.push_back(source[k] * std::complex<float>(std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(k) * binHz * tau0)));
            }
            spectra.insert(spectra.end(), source.begin(), source.end());
        }
        return spectra;
    };
    const auto randomPhases = [](std::size_t count, unsigned seed) {
        std::mt19937                          generator(seed);
        std::uniform_real_distribution<float> phase(-std::numbers::pi_v<float>, std::numbers::pi_v<float>);
        return std::views::iota(0UZ, count) | std::views::transform([&](std::size_t) { return std::polar(1.f, phase(generator)); }) | std::ranges::to<std::vector>();
    };
    const auto peakIndex = [](std::span<const float> row) { return static_cast<std::size_t>(std::ranges::max_element(row) - row.begin()); };

    "on a stereo line input 1 sits at +x, and a positive bearing delays input 0 by d sin(bearing) / c"_test = [&] {
        expect(eq(stereo[0UZ, 0UZ], -kSpacing / 2.0));
        expect(eq(stereo[1UZ, 0UZ], kSpacing / 2.0));
        expect(eq(stereoDelays.size(), kBearings));
        for (const std::size_t b : {0UZ, 60UZ, 90UZ, kRightIndex, 180UZ}) {
            expect(approx(stereoDelays[b], kSpacing * std::sin(bearings[b] * std::numbers::pi / 180.0) / kSpeed, 1e-15)) << bearings[b];
        }
        expect(gt(stereoDelays[kRightIndex], 0.0));
    };

    "bands of at most 2 kHz: 24 cover 0..48 kHz, 12 cover 0..24 kHz, 3 cover 300..5000 Hz and one a 1 Hz band"_test = [&] {
        expect(eq(gr::examples::bandCountOf(0.0, 48000.0), 24UZ));
        expect(eq(gr::examples::bandCountOf(0.0, 24000.0), kNyquistBands));
        expect(eq(gr::examples::bandCountOf(300.0, 5000.0), 3UZ));
        expect(eq(gr::examples::bandCountOf(3000.0, 9000.0), 3UZ));
        expect(eq(gr::examples::bandCountOf(1000.0, 1001.0), 1UZ));
    };

    "a pure delay between two elements makes every band peak at the bearing of that pair delay with value one"_test = [&] {
        const double tau0  = stereoDelays[kRightIndex];
        const auto   power = gr::examples::bandSteeredPower(delayedPair(kBins, kBinHz, tau0, 3UZ), 2UZ, kBins, kBinHz, stereoDelays, gr::examples::bandCountOf(0.0, kNyquistHz), 0.0, kNyquistHz);
        expect(eq(power.size(), kNyquistBands * kBearings));
        for (std::size_t band = 0UZ; band < kNyquistBands; ++band) {
            const std::span<const float> row(power.data() + band * kBearings, kBearings);
            expect(eq(peakIndex(row), kRightIndex)) << "band" << band;
            expect(approx(row[kRightIndex], 1.f, 1e-5f)) << "band" << band;
        }
    };

    "independent random phases on three elements give band powers near zero"_test = [&] {
        constexpr std::size_t kFrames = 64UZ;
        const auto            line    = gr::algorithm::geometry::line(3UZ, kSpacing);
        const auto            delays  = gr::examples::bearingPairDelays(std::span<const double>(line.data(), line.size()), bearings, kSpeed);
        const auto            power   = gr::examples::bandSteeredPower(randomPhases(kFrames * 3UZ * kBins, 7U), 3UZ, kBins, kBinHz, delays, gr::examples::bandCountOf(0.0, kNyquistHz), 0.0, kNyquistHz);
        expect(eq(delays.size(), kBearings * 3UZ));
        expect(std::ranges::all_of(power, [](float value) { return std::abs(value) < 0.1f; }));
    };

    "bins outside f_min..f_max are ignored, even when they hold another coherent source"_test = [&] {
        constexpr double                 kMinHz        = 3000.0;
        constexpr double                 kMaxHz        = 9000.0;
        constexpr auto                   kOutsideIndex = 30UZ; // -60 deg
        const auto                       inside        = delayedPair(kBins, kBinHz, stereoDelays[kRightIndex], 1UZ);
        const auto                       outside       = delayedPair(kBins, kBinHz, stereoDelays[kOutsideIndex], 1UZ);
        std::vector<std::complex<float>> mixed(inside.size());
        for (std::size_t n = 0UZ; n < mixed.size(); ++n) {
            const double f = static_cast<double>(n % kBins) * kBinHz;
            mixed[n]       = f >= kMinHz && f <= kMaxHz ? inside[n] : outside[n];
        }
        const std::size_t nBands = gr::examples::bandCountOf(kMinHz, kMaxHz);
        const auto        power  = gr::examples::bandSteeredPower(mixed, 2UZ, kBins, kBinHz, stereoDelays, nBands, kMinHz, kMaxHz);
        expect(eq(power.size(), nBands * kBearings));
        for (std::size_t band = 0UZ; band < nBands; ++band) {
            const std::span<const float> row(power.data() + band * kBearings, kBearings);
            expect(eq(peakIndex(row), kRightIndex)) << "band" << band;
            expect(approx(row[kRightIndex], 1.f, 1e-5f)) << "band" << band;
        }
        const auto full = gr::examples::bandSteeredPower(mixed, 2UZ, kBins, kBinHz, stereoDelays, 1UZ, 0.0, kNyquistHz);
        expect(lt(full[kRightIndex], 0.5f)) << "the outside source dominates the full band";
    };

    "a bin whose cross-spectrum vanishes contributes zero: half the bins silent halve the coherent score"_test = [&] {
        constexpr std::size_t kFewBins = 65UZ;
        constexpr double      kTau0    = 0.004;
        auto                  spectra  = delayedPair(kFewBins, 1.0, kTau0, 1UZ);
        for (std::size_t k = 1UZ; k < kFewBins; k += 2UZ) {
            spectra[kFewBins + k] = {};
        }
        const std::vector<double> delays{kTau0};
        const auto                power = gr::examples::bandSteeredPower(spectra, 2UZ, kFewBins, 1.0, delays, kFourHertzBands, 0.0, 64.0);
        for (std::size_t band = 0UZ; band + 1UZ < kFourHertzBands; ++band) {
            expect(approx(power[band], 0.5f, 1e-5f)) << "band" << band << "holds 2 of 4 bins";
        }
        expect(approx(power.back(), 0.6f, 1e-5f)) << "the last band holds bins 60..64, 3 of 5 sounding";
    };

    "one band over f_min..f_max is the bin-count-weighted mean of the sixteen bands"_test = [&] {
        constexpr std::size_t     kFewBins = 65UZ;
        const std::vector<double> delays{-0.01, 0.0, 0.004, 0.02};
        const auto                spectra   = randomPhases(4UZ * 2UZ * kFewBins, 11U);
        const auto                bands     = gr::examples::bandSteeredPower(spectra, 2UZ, kFewBins, 1.0, delays, kFourHertzBands, 0.0, 64.0);
        const auto                broadband = gr::examples::bandSteeredPower(spectra, 2UZ, kFewBins, 1.0, delays, 1UZ, 0.0, 64.0);
        for (std::size_t b = 0UZ; b < delays.size(); ++b) {
            double weighted = 0.0;
            for (std::size_t band = 0UZ; band < kFourHertzBands; ++band) {
                const double binsInBand = band + 1UZ < kFourHertzBands ? 4.0 : 5.0; // width 4 Hz; 64 Hz joins the last band
                weighted += binsInBand * static_cast<double>(bands[band * delays.size() + b]);
            }
            expect(approx(static_cast<double>(broadband[b]), weighted / 65.0, 1e-5)) << b;
        }
    };

    "a source on input 1's side of a stereo line reads right, on input 0's side left"_test = [&] {
        const double tau0 = kSpacing * std::sin(23.0 * std::numbers::pi / 180.0) / kSpeed;
        for (const double side : {1.0, -1.0}) {
            const auto        broadband = gr::examples::bandSteeredPower(delayedPair(kBins, kBinHz, side * tau0, 2UZ), 2UZ, kBins, kBinHz, stereoDelays, 1UZ, 0.0, kNyquistHz);
            const std::string verdict   = gr::examples::bearingVerdict(broadband);
            expect(eq(peakIndex(broadband), side > 0.0 ? kRightIndex : 180UZ - kRightIndex));
            expect(verdict.starts_with(side > 0.0 ? "source +23° right, confidence " : "source -23° left, confidence ")) << verdict;
            expect(verdict.ends_with(", front or back")) << verdict;
        }
    };

    "the verdict names bearing, side and peak over second-highest local maximum, or no clear source below the minimum peak"_test = [] {
        const auto verdictOf = [](std::initializer_list<std::pair<std::size_t, float>> samples) {
            std::vector<float> power(kBearings, 0.f);
            for (const auto& [index, value] : samples) {
                power[index] = value;
            }
            return gr::examples::bearingVerdict(power);
        };
        expect(eq(verdictOf({{112UZ, 0.5f}, {113UZ, 1.f}, {114UZ, 0.5f}, {50UZ, 0.4f}}), std::string("source +23° right, confidence 2.5, front or back")));
        expect(eq(verdictOf({{94UZ, 0.5f}, {95UZ, 1.f}, {96UZ, 0.5f}, {20UZ, 0.25f}}), std::string("source +5° centre, confidence 4.0, front or back")));
        expect(eq(verdictOf({{95UZ, 0.5f}, {96UZ, 1.f}, {97UZ, 0.5f}, {20UZ, 0.25f}}), std::string("source +6° right, confidence 4.0, front or back")));
        expect(eq(verdictOf({{44UZ, 0.4f}, {45UZ, 0.8f}, {46UZ, 0.4f}, {150UZ, 0.2f}}), std::string("source -45° left, confidence 4.0, front or back")));
        expect(eq(verdictOf({{113UZ, 0.05f}}), std::string("no clear source (peak 0.05)")));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
