#ifndef GNURADIO_STEERED_POWER_MAP_EXAMPLE_HPP
#define GNURADIO_STEERED_POWER_MAP_EXAMPLE_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <deque>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gr::examples {

inline constexpr std::size_t kMapChartColumns    = 100UZ;
inline constexpr std::size_t kAngularHeatmapRows = 51UZ;

inline constexpr double kFrontAngleLimitDegrees   = 90.0;
inline constexpr double kSweepBearingLimitDegrees = 80.0;

inline constexpr double kBandWidthHz    = 2000.0; /**< the widest band of the band map */
inline constexpr double kCentreDegrees  = 5.0;
inline constexpr double kMinVerdictPeak = 0.1; /**< broadband SRP-PHAT, 1 for a fully coherent source */

enum class MapDisplay { projections, heatmap, waterfall, bands };

struct FrontPowerMap {
    std::vector<double> anglesDegrees;
    std::vector<float>  power;
};

/** unit direction in array coordinates: +x right, +y up, +z front; returns horizontal bearing and vertical elevation in degrees, with bearing zero within float epsilon of either vertical pole. */
[[nodiscard]] std::array<double, 2> frontAnglesDegrees(const std::array<double, 3>& direction) noexcept;

/** power is [elevationSteps, azimuthSteps], uniformly covering elevation [0,90] and periodic azimuth [-180,180); elevationSteps >= 2, azimuthSteps a positive multiple of four.
 * Display samples span [-90,90] at the scientific azimuth spacing; inverse nearest fill covers the field, forward maximum binning retains maxima; the zenith uses azimuth zero once.
 */
[[nodiscard]] FrontPowerMap projectFrontPower(std::span<const float> power, std::size_t azimuthSteps, std::size_t elevationSteps);

/** bearing of the synthetic stereo scene's source at a sample of the repeating scene: a linear sweep from -kSweepBearingLimitDegrees to +kSweepBearingLimitDegrees, right positive; sceneSamples > 0. */
[[nodiscard]] double sweepBearingRadians(std::size_t sampleIndex, std::size_t sceneSamples) noexcept;

/** angle between two unit directions, 0 to 180. */
[[nodiscard]] double angularSeparationDegrees(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept;

/** tau_i - tau_j in s, [bearings, P] for the pairs i < j of `positions` [M,3] (row-major), of a far source at each bearing of a line map: degrees from broadside, positive towards +x */
[[nodiscard]] std::vector<double> bearingPairDelays(std::span<const double> positions, std::span<const double> bearingsDegrees, double speed);

/**
 * per-band SRP-PHAT [nBands, bearings] of one-sided spectra [frames, M, F], bin k at k binHz, steered by `pairDelaysSeconds` [bearings, P] as from bearingPairDelays:
 * per pair i < j, G = X_i conj(X_j) / |G| (0 where |G| = 0) summed over the frames, Re(G e^{+j 2 pi f tau_ij}) summed over the band's bins,
 * divided by its bins, the pairs and the frames, so a coherent source at the bearing scores 1.
 * The bands split [fMinHz, fMaxHz] into equal widths (fMaxHz into the last); bins outside are ignored, a band holding none scores 0.
 */
/** fMinHz < fMaxHz; the fewest bands no wider than kBandWidthHz that cover [fMinHz, fMaxHz] */
[[nodiscard]] std::size_t bandCountOf(double fMinHz, double fMaxHz) noexcept;

[[nodiscard]] std::vector<float> bandSteeredPower(std::span<const std::complex<float>> spectra, std::size_t M, std::size_t F, double binHz, std::span<const double> pairDelaysSeconds, std::size_t nBands, double fMinHz, double fMaxHz);

/**
 * one line on a line array's broadband steered power over bearings -90..+90 in equal steps (an odd count, 0 in the middle): the highest peak's bearing,
 * its side (centre within kCentreDegrees) and PairDelay's confidence (that peak over the second-highest local maximum); "no clear source" below kMinVerdictPeak.
 * A line cannot tell front from back, so a source is always "front or back".
 */
[[nodiscard]] std::string bearingVerdict(std::span<const float> power);

/** bounded equal-width maps at one frame rate; a rate or shape change starts a new history. */
struct SteeredPowerHistory {
    static constexpr std::size_t kMaxFrames = 256UZ;

    std::deque<std::vector<float>> rows;
    float                          frameRate = 0.f;

    [[nodiscard]] std::expected<void, std::string_view> append(std::span<const float> values, float rate);
    /** row must index the retained history; the newest row has age zero. */
    [[nodiscard]] double ageSeconds(std::size_t row) const noexcept;
};

int runSteeredPowerMapExample(int argc, char** argv, MapDisplay display);

} // namespace gr::examples

#endif // GNURADIO_STEERED_POWER_MAP_EXAMPLE_HPP
