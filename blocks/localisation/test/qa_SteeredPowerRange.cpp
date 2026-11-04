#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <vector>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>

#include <gnuradio-4.0/localisation/SteeredPowerRange.hpp>

namespace {
using namespace boost::ut;
using gr::localisation::SteeredPowerRange;

constexpr double      kPi      = std::numbers::pi;
constexpr std::size_t kColumns = SteeredPowerRange::kAzimuthSteps;
constexpr std::size_t kRows    = SteeredPowerRange::kElevationSteps;
constexpr double      kFar     = std::numeric_limits<double>::infinity();

const boost::ut::suite<"SteeredPowerRange"> rangeTests = [] {
    "the default distance samples span 0.05 to 5 metres with the two decades meeting at 0.5 metres"_test = [] {
        const SteeredPowerRange scan{.positionsMetres = {{-0.021, 0.0, 0.0}, {0.021, 0.0, 0.0}}};
        constexpr std::array    kSilence{0.f, 0.f, 0.f};
        const auto              result = gr::localisation::scanSteeredPowerRange(scan, kSilence, kSilence.size());
        expect(fatal(result.has_value()));
        expect(eq(result->distancesMetres.front(), 0.05));
        expect(eq(result->distancesMetres.back(), 5.0));
        expect(le(std::abs(result->distancesMetres[48UZ] - 0.5), 1e-12));
        expect(le(std::abs(result->distancesMetres[24UZ] - 0.05 * std::sqrt(10.0)), 1e-12));
        expect(le(std::abs(result->distancesMetres[72UZ] - 0.5 * std::sqrt(10.0)), 1e-12));
        expect(result->profiles.empty());
    };

    "a 5-12-13 source geometry aligns six independent integer-delay correlation peaks at five metres"_test = [] {
        SteeredPowerRange                         scan{.positionsMetres = {{0.0, 0.0, 0.0}, {12.0, 0.0, 0.0}, {0.0, 6.0, 0.0}, {3.0, 3.0, 0.0}}, .sampleRateHz = 1000.0, .propagationSpeedMetresPerSecond = 1000.0};
        constexpr std::size_t                     kLength = 25UZ;
        constexpr std::array                      kLags{-8, 0, 0, 8, 8, 0};
        std::array<float, kLags.size() * kLength> curves{};
        for (std::size_t p = 0UZ; p < kLags.size(); ++p) {
            curves[p * kLength + static_cast<std::size_t>(12 + kLags[p])] = 1.f;
        }
        expect(scan.validate().has_value());
        expect(eq(scan.scoreAt(curves, kLength, {0.0, 0.6, 0.8}, 5.0), 6.f));
        expect(lt(scan.scoreAt(curves, kLength, {0.0, 0.6, 0.8}, 2.0), 6.f));
        expect(lt(scan.scoreAt(curves, kLength, {0.0, 0.6, 0.8}, kFar), 6.f));
    };

    "a known source is recovered at the zenith and five metres while a zero-lag scene favours the far-field limit"_test = [] {
        const SteeredPowerRange scan{.positionsMetres = {{0.0, 0.0, 0.0}, {12.0, 0.0, 0.0}, {0.0, 12.0, 0.0}, {-12.0, 0.0, 0.0}}, .sampleRateHz = 1000.0, .propagationSpeedMetresPerSecond = 1000.0, .minDistanceMetres = 1.0, .maxDistanceMetres = 25.0};
        constexpr std::size_t   kLength = 25UZ;
        constexpr std::array    kSphericalLags{-8, -8, -8, 0, 0, 0};
        for (const bool spherical : {true, false}) {
            std::array<float, kLength * kSphericalLags.size()> curves{};
            for (std::size_t p = 0UZ; p < kSphericalLags.size(); ++p) {
                const int lag                                            = spherical ? kSphericalLags[p] : 0;
                curves[p * kLength + static_cast<std::size_t>(12 + lag)] = 1.f;
            }
            const auto result = gr::localisation::scanSteeredPowerRange(scan, curves, kLength);
            expect(fatal(result.has_value()));
            expect(fatal(!result->profiles.empty()));
            const auto& profile = result->profiles.front();
            expect(le(std::abs(profile.elevationRadians - kPi / 2.0), 1e-6));
            expect(le(std::abs(profile.azimuthRadians), 1e-6));
            const auto maximum = std::ranges::max_element(profile.power);
            if (spherical) {
                const auto index = static_cast<std::size_t>(maximum - profile.power.begin());
                expect(le(std::abs(result->distancesMetres[index] - 5.0), 1e-12));
                expect(le(std::abs(*maximum - 6.f), 1e-4f));
                expect(le(std::abs(profile.farPower - 3.f), 1e-4f));
            } else {
                expect(le(std::abs(profile.farPower - 6.f), 1e-4f));
                expect(gt(profile.farPower, *maximum));
            }
        }
    };

    "plane-wave lags have the analytic projection sign and distant spheres approach them"_test = [] {
        SteeredPowerRange    scan{.positionsMetres = {{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}}, .sampleRateHz = 1.0, .propagationSpeedMetresPerSecond = 1.0};
        constexpr std::array kCurve{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        expect(eq(scan.scoreAt(kCurve, kCurve.size(), {1.0, 0.0, 0.0}, kFar), 7.f));
        expect(eq(scan.scoreAt(kCurve, kCurve.size(), {-1.0, 0.0, 0.0}, kFar), 1.f));
        expect(le(std::abs(scan.scoreAt(kCurve, kCurve.size(), {0.0, 0.0, 1.0}, 1e6) - 4.f), 1e-5f));
        expect(eq(scan.scoreAt(kCurve, kCurve.size(), {0.0, 0.6, 0.8}, 5.0), scan.scoreAt(kCurve, kCurve.size(), {0.0, 0.6, -0.8}, 5.0)));
    };

    "range projection includes both endpoints and the plane-wave limit and profiles hold the detected direction fixed"_test = [] {
        SteeredPowerRange                          scan{.positionsMetres = {{-0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}}, .sampleRateHz = 2.0, .propagationSpeedMetresPerSecond = 1.0, .minDistanceMetres = 1.0, .maxDistanceMetres = 25.0};
        constexpr std::size_t                      kLength = 9UZ;
        constexpr std::array<float, kLength * 3UZ> kCurves{0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0};
        const auto                                 result = gr::localisation::scanSteeredPowerRange(scan, kCurves, kLength);
        expect(fatal(result.has_value()));
        expect(eq(result->distancesMetres.size(), SteeredPowerRange::kDistanceSteps));
        expect(eq(result->distancesMetres.front(), 1.0));
        expect(eq(result->distancesMetres.back(), 25.0));
        expect(eq(result->angularPower.size(), kRows * kColumns));
        expect(std::ranges::is_sorted(result->distancesMetres));
        for (const std::size_t cell : {0UZ, 35UZ, 600UZ, (kRows - 1UZ) * kColumns}) {
            const double     elevation = static_cast<double>(cell / kColumns) * kPi / 36.0;
            const double     azimuth   = -kPi + static_cast<double>(cell % kColumns) * kPi / 36.0;
            const std::array direction{std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation)};
            float            expected = scan.scoreAt(kCurves, kLength, direction, kFar);
            for (const double radius : result->distancesMetres) {
                expected = std::max(expected, scan.scoreAt(kCurves, kLength, direction, radius));
            }
            expect(le(std::abs(result->angularPower[cell] - expected), 1e-6f));
        }
        expect(fatal(!result->profiles.empty()));
        for (const auto& profile : result->profiles) {
            const std::array direction{std::cos(profile.elevationRadians) * std::cos(profile.azimuthRadians), std::cos(profile.elevationRadians) * std::sin(profile.azimuthRadians), std::sin(profile.elevationRadians)};
            expect(eq(profile.power.size(), result->distancesMetres.size()));
            for (std::size_t r = 0UZ; r < profile.power.size(); ++r) {
                expect(eq(profile.power[r], scan.scoreAt(kCurves, kLength, direction, result->distancesMetres[r])));
            }
            expect(eq(profile.farPower, scan.scoreAt(kCurves, kLength, direction, kFar)));
        }
    };

    "the horizon and azimuth seam are eligible peaks and a flat map has none"_test = [] {
        std::vector<float> map(kRows * kColumns, 0.f);
        expect(gr::localisation::findRangeDirections(map, 3UZ).empty());
        map[0UZ]              = 3.f;
        const auto directions = gr::localisation::findRangeDirections(map, 3UZ);
        expect(fatal(eq(directions.size(), 1UZ)));
        expect(le(std::abs(directions[0UZ][0UZ]), 1e-6f));
        expect(le(std::abs(directions[0UZ][1UZ] + std::numbers::pi_v<float>), 1e-6f));
        expect(eq(directions[0UZ][2UZ], 3.f));
    };

    "one canonical zenith is emitted and a stronger penultimate ring suppresses it without nonfinite interpolation"_test = [] {
        std::vector<float> map(kRows * kColumns, 0.f);
        std::fill_n(map.begin() + (kRows - 1UZ) * kColumns, kColumns, 2.f);
        const auto zenith = gr::localisation::findRangeDirections(map, 3UZ);
        expect(fatal(eq(zenith.size(), 1UZ)));
        expect(le(std::abs(zenith[0UZ][0UZ] - std::numbers::pi_v<float> / 2.f), 1e-6f));
        expect(le(std::abs(zenith[0UZ][1UZ]), 1e-6f));
        map[(kRows - 2UZ) * kColumns + 5UZ] = 3.f;
        const auto ring                     = gr::localisation::findRangeDirections(map, 3UZ);
        expect(fatal(eq(ring.size(), 1UZ)));
        expect(lt(ring[0UZ][0UZ], std::numbers::pi_v<float> / 2.f));
        expect(std::ranges::all_of(ring[0UZ], [](float v) { return std::isfinite(v); }));
    };

    "one to three separated peaks are returned strongest first with the requested count limit"_test = [] {
        std::vector<float> map(kRows * kColumns, 0.f);
        for (std::size_t n = 1UZ; n <= 3UZ; ++n) {
            map[5UZ * kColumns + n * 15UZ] = 4.f - static_cast<float>(n);
            for (std::size_t limit = 1UZ; limit <= 3UZ; ++limit) {
                const auto peaks = gr::localisation::findRangeDirections(map, limit);
                expect(eq(peaks.size(), std::min(n, limit)));
                for (std::size_t k = 1UZ; k < peaks.size(); ++k) {
                    expect(gt(peaks[k - 1UZ][2UZ], peaks[k][2UZ]));
                }
            }
        }
    };

    "invalid geometry, rates, distance bounds, peak counts and correlation frames are refused"_test = [] {
        SteeredPowerRange scan;
        expect(!scan.validate());
        scan.positionsMetres = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        expect(scan.validate().has_value());
        for (const double bad : {0.0, -1.0, kFar, std::numeric_limits<double>::quiet_NaN()}) {
            auto changed         = scan;
            changed.sampleRateHz = bad;
            expect(!changed.validate());
            changed                                 = scan;
            changed.propagationSpeedMetresPerSecond = bad;
            expect(!changed.validate());
            changed                   = scan;
            changed.minDistanceMetres = bad;
            expect(!changed.validate());
            changed                   = scan;
            changed.maxDistanceMetres = bad;
            expect(!changed.validate());
        }
        for (const std::size_t bad : {0UZ, 4UZ}) {
            auto changed     = scan;
            changed.maxPeaks = bad;
            expect(!changed.validate());
        }
        auto changed              = scan;
        changed.maxDistanceMetres = scan.minDistanceMetres;
        expect(!changed.validate());
        changed                           = scan;
        changed.positionsMetres[0UZ][0UZ] = kFar;
        expect(!changed.validate());
        changed = scan;
        changed.positionsMetres.resize(33UZ);
        expect(!changed.validate());
        constexpr std::array kCurve{0.f, 1.f, 0.f};
        expect(!gr::localisation::scanSteeredPowerRange(scan, {}, 0UZ));
        expect(!gr::localisation::scanSteeredPowerRange(scan, kCurve, 2UZ));
        expect(!gr::localisation::scanSteeredPowerRange(scan, kCurve, 5UZ));
        constexpr std::array kNonfinite{0.f, std::numeric_limits<float>::quiet_NaN(), 0.f};
        expect(!gr::localisation::scanSteeredPowerRange(scan, kNonfinite, 3UZ));
    };
};

} // namespace

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
