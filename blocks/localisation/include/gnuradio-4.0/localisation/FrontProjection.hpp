#ifndef GNURADIO_LOCALISATION_FRONT_PROJECTION_HPP
#define GNURADIO_LOCALISATION_FRONT_PROJECTION_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>

namespace gr::localisation {

inline constexpr double kFrontAngleLimitDegrees = 90.0;

struct FrontPowerMap {
    std::vector<double> anglesDegrees;
    std::vector<float>  power;
};

/** unit direction in array coordinates: +x right, +y up, +z front; returns horizontal bearing and vertical elevation in degrees, with bearing zero within float epsilon of either vertical pole. */
[[nodiscard]] inline std::array<double, 2> frontAnglesDegrees(const std::array<double, 3>& direction) noexcept {
    const auto [x, y, z]                      = direction;
    constexpr double kFloatDirectionTolerance = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double     horizontalMagnitude      = std::hypot(x, z);
    if (horizontalMagnitude <= kFloatDirectionTolerance) {
        return {0.0, std::copysign(90.0, y)};
    }
    return {gr::algorithm::degreesOf(std::atan2(x, z)), gr::algorithm::degreesOf(std::atan2(y, horizontalMagnitude))};
}

/** power is [elevationSteps, azimuthSteps], uniformly covering elevation [0,90] and periodic azimuth [-180,180); elevationSteps >= 2, azimuthSteps a positive multiple of four.
 * Display samples span [-90,90] at the scientific azimuth spacing; inverse nearest fill covers the field, forward maximum binning retains maxima; the zenith uses azimuth zero once.
 */
[[nodiscard]] inline FrontPowerMap projectFrontPower(std::span<const float> power, std::size_t azimuthSteps, std::size_t elevationSteps) {
    assert(azimuthSteps >= 4UZ && azimuthSteps % 4UZ == 0UZ && elevationSteps >= 2UZ && power.size() == azimuthSteps * elevationSteps);
    constexpr double  kPi  = std::numbers::pi;
    const std::size_t side = azimuthSteps / 2UZ + 1UZ;
    FrontPowerMap     result{.anglesDegrees = std::vector<double>(side), .power = std::vector<float>(side * side, std::numeric_limits<float>::quiet_NaN())};
    for (std::size_t i = 0UZ; i < side; ++i) {
        result.anglesDegrees[i] = -kFrontAngleLimitDegrees + 2.0 * kFrontAngleLimitDegrees * static_cast<double>(i) / static_cast<double>(side - 1UZ);
    }
    for (std::size_t row = 0UZ; row < side; ++row) {
        const double vertical = result.anglesDegrees[row];
        for (std::size_t column = 0UZ; column < side; ++column) {
            const double horizontal           = result.anglesDegrees[column];
            const double bearing              = gr::algorithm::radiansOf(horizontal);
            const double pitch                = gr::algorithm::radiansOf(vertical);
            const double x                    = std::cos(pitch) * std::sin(bearing);
            const double y                    = std::sin(pitch);
            const double z                    = std::cos(pitch) * std::cos(bearing);
            const double azimuth              = std::atan2(y, x);
            const double elevation            = std::atan2(z, std::hypot(x, y));
            const auto   sourceColumn         = static_cast<std::size_t>(std::llround((azimuth + kPi) * static_cast<double>(azimuthSteps) / (2.0 * kPi))) % azimuthSteps;
            const auto   sourceRow            = std::min(static_cast<std::size_t>(std::llround(elevation * static_cast<double>(elevationSteps - 1UZ) / (kPi / 2.0))), elevationSteps - 1UZ);
            result.power[row * side + column] = power[sourceRow * azimuthSteps + sourceColumn];
        }
    }
    for (std::size_t row = 0UZ; row < elevationSteps; ++row) {
        const double      elevation   = static_cast<double>(row) * (kPi / 2.0) / static_cast<double>(elevationSteps - 1UZ);
        const bool        atZenith    = row + 1UZ == elevationSteps;
        const std::size_t firstColumn = atZenith ? azimuthSteps / 2UZ : 0UZ;
        const std::size_t endColumn   = atZenith ? firstColumn + 1UZ : azimuthSteps;
        for (std::size_t column = firstColumn; column < endColumn; ++column) {
            const double      azimuth      = -kPi + static_cast<double>(column) * (2.0 * kPi) / static_cast<double>(azimuthSteps);
            const auto        direction    = gr::algorithm::directionOf(azimuth, elevation);
            const auto        angles       = frontAnglesDegrees(direction);
            const auto        targetColumn = static_cast<std::size_t>(std::llround((angles[0UZ] + kFrontAngleLimitDegrees) * static_cast<double>(side - 1UZ) / (2.0 * kFrontAngleLimitDegrees)));
            const auto        targetRow    = static_cast<std::size_t>(std::llround((angles[1UZ] + kFrontAngleLimitDegrees) * static_cast<double>(side - 1UZ) / (2.0 * kFrontAngleLimitDegrees)));
            const std::size_t cell         = targetRow * side + targetColumn;
            result.power[cell]             = std::max(result.power[cell], power[row * azimuthSteps + column]);
        }
    }
    return result;
}

/** angle between two unit directions, 0 to 180. */
[[nodiscard]] inline double angularSeparationDegrees(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept {
    const double cosine = a[0UZ] * b[0UZ] + a[1UZ] * b[1UZ] + a[2UZ] * b[2UZ];
    return std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

} // namespace gr::localisation

#endif // GNURADIO_LOCALISATION_FRONT_PROJECTION_HPP
