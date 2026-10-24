#ifndef GNURADIO_STEERED_POWER_RANGE_HPP
#define GNURADIO_STEERED_POWER_RANGE_HPP

#include <array>
#include <cstddef>
#include <expected>
#include <numbers>
#include <span>
#include <string_view>
#include <vector>

namespace gr::examples {

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

/** full azimuth [-pi,pi), elevation [0,pi/2]; the zenith has one canonical azimuth, zero. Input pairs use lexicographic i<j order, zero lag in the middle. */
[[nodiscard]] std::expected<SteeredPowerRangeScan, std::string_view> scanSteeredPowerRange(const SteeredPowerRange& scan, std::span<const float> curves, std::size_t length);

/** map has kElevationSteps * kAzimuthSteps finite entries; strict maxima, including the mirrored horizon and one canonical zenith; separation is four grid cells. */
[[nodiscard]] std::vector<std::array<float, 4>> findRangeDirections(std::span<const float> map, std::size_t maxPeaks);

} // namespace gr::examples

#endif // GNURADIO_STEERED_POWER_RANGE_HPP
