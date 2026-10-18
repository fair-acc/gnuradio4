#ifndef GNURADIO_LOCALISATION_STEERED_POWER_RANGE_HPP
#define GNURADIO_LOCALISATION_STEERED_POWER_RANGE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/SteeredPower.hpp>
#include <gnuradio-4.0/algorithm/peak/Peak.hpp>

namespace gr::localisation {

struct SteeredPowerRange {
    static constexpr std::size_t kMaxScanElements  = 32UZ;
    static constexpr std::size_t kMaxPeaks         = 3UZ;
    static constexpr std::size_t kDistanceSteps    = 97UZ;
    static constexpr std::size_t kAzimuthSteps     = 72UZ;
    static constexpr std::size_t kElevationSteps   = 19UZ;
    static constexpr double      kAngleStepRadians = std::numbers::pi / 36.0; // azimuth and elevation

    std::vector<std::array<double, 3>> positionsMetres;
    double                             sampleRateHz                    = 48000.0;
    double                             propagationSpeedMetresPerSecond = 343.0;
    double                             minDistanceMetres               = 0.05;
    double                             maxDistanceMetres               = 5.0;
    std::size_t                        maxPeaks                        = kMaxPeaks;

    [[nodiscard]] std::expected<void, std::string_view> validate() const;
    /** validated configuration; curves has M(M-1)/2 * length finite entries, length >= 3 odd; direction is unit length, distanceMetres > 0 or +infinity for a plane wave. */
    [[nodiscard]] float scoreAt(std::span<const float> curves, std::size_t length, const std::array<double, 3>& direction, double distanceMetres) const noexcept;
};

struct SteeredPowerRangeProfile {
    double             elevationRadians;
    double             azimuthRadians;
    float              score;
    std::vector<float> power;
    float              farPower;
};

struct SteeredPowerRangeScan {
    std::vector<float>                    angularPower; // [elevation, azimuth], max over finite distances and plane wave
    std::vector<double>                   distancesMetres;
    std::vector<SteeredPowerRangeProfile> profiles;
};

inline std::expected<void, std::string_view> SteeredPowerRange::validate() const {
    if (positionsMetres.size() < 2UZ || positionsMetres.size() > kMaxScanElements || !std::ranges::all_of(positionsMetres, [](const auto& position) { return std::ranges::all_of(position, [](double v) { return std::isfinite(v); }); })) {
        return std::unexpected("range scan requires 2..32 finite microphone positions");
    }
    if (!(sampleRateHz > 0.0) || !std::isfinite(sampleRateHz) || !(propagationSpeedMetresPerSecond > 0.0) || !std::isfinite(propagationSpeedMetresPerSecond)) {
        return std::unexpected("range scan requires finite positive sample rate and propagation speed");
    }
    constexpr double kMaxDistanceMetres = 100.0;
    constexpr double kMinDistanceMetres = 0.01;
    if (!(minDistanceMetres >= kMinDistanceMetres && maxDistanceMetres > minDistanceMetres && maxDistanceMetres <= kMaxDistanceMetres) || maxPeaks < 1UZ || maxPeaks > kMaxPeaks) {
        return std::unexpected("range scan requires 0.01 <= min distance < max distance <= 100 m and 1..3 peaks");
    }
    return {};
}

inline float SteeredPowerRange::scoreAt(std::span<const float> curves, std::size_t length, const std::array<double, 3>& direction, double distanceMetres) const noexcept {
    const bool                           far   = std::isinf(distanceMetres);
    const std::array<double, 3>          point = far ? direction : std::array{direction[0] * distanceMetres, direction[1] * distanceMetres, direction[2] * distanceMetres};
    const std::size_t                    M     = positionsMetres.size();
    const gr::algorithm::Field           field = far ? gr::algorithm::Field::far : gr::algorithm::Field::near;
    std::array<double, kMaxScanElements> delays{};
    for (std::size_t m = 0UZ; m < M; ++m) {
        delays[m] = gr::algorithm::delayOf(positionsMetres[m].data(), point, field, propagationSpeedMetresPerSecond) * sampleRateHz;
    }
    constexpr std::size_t        kMaxPairs = gr::algorithm::pairsOf(kMaxScanElements);
    std::array<float, kMaxPairs> lags{};
    const std::size_t            P = gr::algorithm::pairsOf(M);
    for (std::size_t p = 0UZ; p < P; ++p) {
        const auto ij = gr::algorithm::pairOf(p, M);
        lags[p]       = static_cast<float>(delays[ij.i] - delays[ij.j]);
    }
    return gr::algorithm::computeSteeredPower(curves.data(), lags.data(), P, length);
}

/** map has kElevationSteps * kAzimuthSteps finite entries; strict maxima, including the mirrored horizon and one canonical zenith; separation is four grid cells. */
[[nodiscard]] inline std::vector<std::array<float, 4>> findRangeDirections(std::span<const float> map, std::size_t maxPeaks) {
    constexpr std::size_t                  kColumns   = SteeredPowerRange::kAzimuthSteps;
    constexpr std::size_t                  kRows      = SteeredPowerRange::kElevationSteps;
    constexpr auto                         kAngleStep = static_cast<float>(SteeredPowerRange::kAngleStepRadians);
    constexpr std::size_t                  kCells     = (kRows + 2UZ) * kColumns;
    const gr::algorithm::peak::Grid<float> grid{.extents = {kRows + 2UZ, kColumns, 1UZ}, .rank = 2UZ, .origin = {-kAngleStep, -std::numbers::pi_v<float>, 0.f}, .step = {kAngleStep, kAngleStep, 1.f}, .periodic = {false, true, false}};
    std::vector<float>                     values(kCells);
    std::ranges::copy(map, values.begin() + kColumns);
    std::copy_n(map.begin() + kColumns, kColumns, values.begin());
    std::copy_n(map.begin() + (kRows - 2UZ) * kColumns, kColumns, values.begin() + (kRows + 1UZ) * kColumns);
    std::vector<float> candidates(kCells, -std::numeric_limits<float>::infinity());
    const auto         ring       = map.subspan((kRows - 2UZ) * kColumns, kColumns);
    const std::size_t  zenithCell = kRows * kColumns + kColumns / 2UZ;
    const bool         zenithPeak = values[zenithCell] > *std::ranges::max_element(ring);
    for (std::size_t cell = kColumns; cell < (kRows + 1UZ) * kColumns; ++cell) {
        const bool atZenith = cell >= kRows * kColumns;
        const bool peak     = atZenith ? cell == zenithCell && zenithPeak : gr::algorithm::peak::isPeak(values.data(), grid, cell, kCells);
        if (peak) {
            std::array<float, 4> row{};
            gr::algorithm::peak::describePeak(values.data(), grid, cell, kCells, 1.f, row.data());
            candidates[cell] = row[2UZ];
        }
    }
    constexpr std::size_t                                          kSeparationSquared = 16UZ;
    constexpr float                                                kRelativeThreshold = 0.3f;
    std::array<std::uint32_t, SteeredPowerRange::kMaxPeaks + 1UZ>  accepted{};
    std::array<std::array<float, 4>, SteeredPowerRange::kMaxPeaks> rows{};
    gr::algorithm::peak::selectPeaks(values.data(), candidates.data(), grid, kCells, std::min(maxPeaks, SteeredPowerRange::kMaxPeaks), kRelativeThreshold, kSeparationSquared, accepted.data(), rows[0UZ].data(), 4UZ);
    return {rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(accepted[0UZ])};
}

/** full azimuth [-pi,pi), elevation [0,pi/2]; the zenith has one canonical azimuth, zero. Input pairs use lexicographic i<j order, zero lag in the middle. */
[[nodiscard]] inline std::expected<SteeredPowerRangeScan, std::string_view> scanSteeredPowerRange(const SteeredPowerRange& scan, std::span<const float> curves, std::size_t length) {
    if (const auto valid = scan.validate(); !valid) {
        return std::unexpected(valid.error());
    }
    const std::size_t M     = scan.positionsMetres.size();
    const std::size_t pairs = gr::algorithm::pairsOf(M);
    if (length < 3UZ || length % 2UZ == 0UZ || length > curves.size() / pairs || curves.size() != pairs * length || !std::ranges::all_of(curves, [](float v) { return std::isfinite(v); })) {
        return std::unexpected("range scan expects finite correlation curves [M(M-1)/2, odd length >= 3]");
    }
    constexpr double      kPi          = std::numbers::pi;
    constexpr double      kAngleStep   = SteeredPowerRange::kAngleStepRadians;
    constexpr double      kFarDistance = std::numeric_limits<double>::infinity();
    SteeredPowerRangeScan result;
    for (std::size_t r = 0UZ; r < SteeredPowerRange::kDistanceSteps; ++r) {
        const double fraction = static_cast<double>(r) / static_cast<double>(SteeredPowerRange::kDistanceSteps - 1UZ);
        result.distancesMetres.push_back(scan.minDistanceMetres * std::pow(scan.maxDistanceMetres / scan.minDistanceMetres, fraction));
    }
    result.distancesMetres.front() = scan.minDistanceMetres;
    result.distancesMetres.back()  = scan.maxDistanceMetres;
    for (std::size_t e = 0UZ; e < SteeredPowerRange::kElevationSteps; ++e) {
        const bool zenith = e + 1UZ == SteeredPowerRange::kElevationSteps;
        for (std::size_t a = 0UZ; a < SteeredPowerRange::kAzimuthSteps; ++a) {
            const auto direction = zenith ? std::array{0.0, 0.0, 1.0} : gr::algorithm::directionOf(-kPi + static_cast<double>(a) * kAngleStep, static_cast<double>(e) * kAngleStep);
            float      power     = scan.scoreAt(curves, length, direction, kFarDistance);
            for (const double radius : result.distancesMetres) {
                power = std::max(power, scan.scoreAt(curves, length, direction, radius));
            }
            result.angularPower.push_back(power);
        }
    }
    for (const auto& peak : findRangeDirections(result.angularPower, scan.maxPeaks)) {
        const double             elevation = static_cast<double>(peak[0UZ]);
        const double             azimuth   = static_cast<double>(peak[1UZ]);
        const auto               direction = gr::algorithm::directionOf(azimuth, elevation);
        SteeredPowerRangeProfile profile{.elevationRadians = elevation, .azimuthRadians = azimuth, .score = peak[2UZ], .power = {}, .farPower = scan.scoreAt(curves, length, direction, kFarDistance)};
        for (const double radius : result.distancesMetres) {
            profile.power.push_back(scan.scoreAt(curves, length, direction, radius));
        }
        result.profiles.push_back(std::move(profile));
    }
    return result;
}

} // namespace gr::localisation

#endif // GNURADIO_LOCALISATION_STEERED_POWER_RANGE_HPP
