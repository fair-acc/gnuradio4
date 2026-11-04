#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/localisation/FrontProjection.hpp>

namespace {
constexpr std::size_t kMapChartColumns    = 100UZ;
constexpr std::size_t kAngularHeatmapRows = 51UZ;
} // namespace

const boost::ut::suite<"FrontPowerMap"> kAngleTests = [] {
    using namespace boost::ut;

    "front, right, left, up and down have independently known display angles"_test = [] {
        constexpr std::array<std::array<double, 3>, 5> kDirections{{{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0}}};
        constexpr std::array<std::array<double, 2>, 5> kExpected{{{0.0, 0.0}, {90.0, 0.0}, {-90.0, 0.0}, {0.0, 90.0}, {0.0, -90.0}}};
        for (std::size_t i = 0UZ; i < kDirections.size(); ++i) {
            const auto angles = gr::localisation::frontAnglesDegrees(kDirections[i]);
            expect(approx(angles[0UZ], kExpected[i][0UZ], 1e-12));
            expect(approx(angles[1UZ], kExpected[i][1UZ], 1e-12));
        }
    };

    "vertical poles from float scientific coordinates have a canonical zero bearing"_test = [] {
        for (const float azimuth : {-std::numbers::pi_v<float> / 2.f, std::numbers::pi_v<float> / 2.f}) {
            const auto direction = gr::algorithm::directionOf(static_cast<double>(azimuth), 0.0);
            const auto angles    = gr::localisation::frontAnglesDegrees(direction);
            expect(eq(angles[0UZ], 0.0));
            expect(eq(angles[1UZ], std::copysign(90.0, static_cast<double>(azimuth))));
        }
    };

    "diagonals and a 3-4-5 direction obey conventional bearing and elevation"_test = [] {
        const double root = std::sqrt(0.5);
        for (const double signX : {-1.0, 1.0}) {
            for (const double signY : {-1.0, 1.0}) {
                const auto angles = gr::localisation::frontAnglesDegrees({signX * 0.5, signY * root, 0.5});
                expect(approx(angles[0UZ], signX * 45.0, 1e-12));
                expect(approx(angles[1UZ], signY * 45.0, 1e-12));
            }
        }
        const auto angles = gr::localisation::frontAnglesDegrees({0.6, 0.0, 0.8});
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
        const auto map = gr::localisation::projectFrontPower(power, kAzimuthSteps, kElevationSteps);
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
        const auto                                                 map = gr::localisation::projectFrontPower(power, 72UZ, 19UZ);
        gr::graphs::ImChart<kMapChartColumns, kAngularHeatmapRows> chart({{-90.0, 90.0}, {-90.0, 90.0}});
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
        const auto map           = gr::localisation::projectFrontPower(power, 72UZ, 19UZ);
        expect(eq(map.power[25UZ * 37UZ + 31UZ], 9.f));
        expect(eq(*std::ranges::max_element(map.power), 9.f));
        expect(eq(*std::ranges::min_element(map.power), -3.f));
    };

    "scientific directions that share a display bin retain their strongest signed score"_test = [] {
        for (const bool swap : {false, true}) {
            std::vector<float> power(19UZ * 72UZ, -9.f);
            power[16UZ * 72UZ]       = swap ? -7.f : -1.f;
            power[16UZ * 72UZ + 1UZ] = swap ? -1.f : -7.f;
            const auto map           = gr::localisation::projectFrontPower(power, 72UZ, 19UZ);
            expect(eq(map.power[18UZ * 37UZ + 16UZ], -1.f));
            expect(eq(*std::ranges::max_element(map.power), -1.f));
        }
    };

    "each distinct scientific direction can contribute an isolated maximum"_test = [] {
        std::vector<float> power(19UZ * 72UZ, 0.f);
        for (std::size_t cell = 0UZ; cell < 18UZ * 72UZ; ++cell) {
            power[cell]    = 1.f;
            const auto map = gr::localisation::projectFrontPower(power, 72UZ, 19UZ);
            expect(eq(*std::ranges::max_element(map.power), 1.f)) << cell;
            power[cell] = 0.f;
        }
        power[18UZ * 72UZ + 36UZ] = 1.f;
        const auto zenith         = gr::localisation::projectFrontPower(power, 72UZ, 19UZ);
        expect(eq(zenith.power[18UZ * 37UZ + 18UZ], 1.f));
    };

    "signed constant power remains signed over the measured hemisphere"_test = [] {
        const std::vector<float> power(19UZ * 72UZ, -2.5f);
        const auto               map = gr::localisation::projectFrontPower(power, 72UZ, 19UZ);
        expect(std::ranges::all_of(map.power, [](float value) { return value == -2.5f; }));
    };
};

const boost::ut::suite<"angular separation"> kSeparationTests = [] {
    using namespace boost::ut;
    constexpr double kDegrees = 180.0 / std::numbers::pi;

    "angular separation is zero for equal, ninety for orthogonal, one-eighty for opposite and 36.87 for a 3-4-5 pair"_test = [] {
        constexpr std::array<double, 3> kX{1.0, 0.0, 0.0};
        constexpr std::array<double, 3> kY{0.0, 1.0, 0.0};
        constexpr std::array<double, 3> kMinusX{-1.0, 0.0, 0.0};
        constexpr std::array<double, 3> kThreeFourFive{0.8, 0.6, 0.0};
        expect(approx(gr::localisation::angularSeparationDegrees(kX, kX), 0.0, 1e-9));
        expect(approx(gr::localisation::angularSeparationDegrees(kX, kY), 90.0, 1e-9));
        expect(approx(gr::localisation::angularSeparationDegrees(kX, kMinusX), 180.0, 1e-9));
        expect(approx(gr::localisation::angularSeparationDegrees(kX, kThreeFourFive), std::atan2(0.6, 0.8) * kDegrees, 1e-9));
        expect(eq(gr::localisation::angularSeparationDegrees(kThreeFourFive, kX), gr::localisation::angularSeparationDegrees(kX, kThreeFourFive)));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
