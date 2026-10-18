#ifndef GNURADIO_LOCALISATION_BAND_STEERED_POWER_HPP
#define GNURADIO_LOCALISATION_BAND_STEERED_POWER_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/localisation/FrontProjection.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>

namespace gr::localisation {

inline constexpr double kMinBandHz      = 150.0;
inline constexpr double kCentreDegrees  = 5.0;
inline constexpr double kMinVerdictPeak = 0.1; /**< broadband SRP-PHAT, 1 for a fully coherent source */

/** tau_i - tau_j in s, [bearings, P] for the pairs i < j of `positions` [M,3] (row-major), of a far source at each bearing of a line map: degrees from broadside, positive towards +x */
[[nodiscard]] inline std::vector<double> bearingPairDelays(std::span<const double> positions, std::span<const double> bearingsDegrees, double speed) {
    const std::size_t   M = positions.size() / 3UZ;
    const std::size_t   P = gr::algorithm::pairsOf(M);
    std::vector<double> delays;
    delays.reserve(bearingsDegrees.size() * P);
    for (const double bearing : bearingsDegrees) {
        const auto direction = gr::algorithm::directionOf(gr::algorithm::radiansOf(kFrontAngleLimitDegrees - bearing), 0.0);
        for (std::size_t p = 0UZ; p < P; ++p) {
            delays.push_back(gr::algorithm::pairDelayOf(positions.data(), gr::algorithm::pairOf(p, M), direction, gr::algorithm::Field::far, speed));
        }
    }
    return delays;
}

/** finite 0 < fMinHz < fMaxHz; half-octaves anchored at fMinHz with the final edge clipped to fMaxHz. */
[[nodiscard]] inline std::vector<double> halfOctaveEdges(double fMinHz, double fMaxHz) {
    assert(std::isfinite(fMinHz) && std::isfinite(fMaxHz) && 0.0 < fMinHz && fMinHz < fMaxHz);
    std::vector<double> edges{fMinHz};
    for (std::size_t step = 1UZ;; ++step) {
        const double next = fMinHz * std::exp2(static_cast<double>(step) / 2.0);
        if (next >= fMaxHz) {
            edges.push_back(fMaxHz);
            return edges;
        }
        edges.push_back(next);
    }
}

/** a power-of-two Hann frame of at least 100 ms whose four-bin main lobe fits the lowest band; refuses frames above 65536 samples. */
[[nodiscard]] inline std::expected<std::size_t, std::string_view> bandFrameSamples(double sampleRateHz, double fMinHz, double fMaxHz) noexcept {
    constexpr std::size_t kMinFrameSamples  = 1024UZ;
    constexpr std::size_t kMaxFrameSamples  = 65536UZ;
    constexpr double      kHannMainLobeBins = 4.0;
    constexpr double      kMinWindowSeconds = 0.1;
    const double          widthHz           = std::min(fMinHz * (std::numbers::sqrt2 - 1.0), fMaxHz - fMinHz);
    const double          requiredSamples   = std::ceil(std::max(kHannMainLobeBins * sampleRateHz / widthHz, kMinWindowSeconds * sampleRateHz));
    if (!std::isfinite(sampleRateHz) || sampleRateHz <= 0.0 || !std::isfinite(fMinHz) || fMinHz <= 0.0 || !std::isfinite(fMaxHz) || fMinHz >= fMaxHz || !std::isfinite(requiredSamples) || requiredSamples > static_cast<double>(kMaxFrameSamples)) {
        return std::unexpected("frequency bands need positive increasing bounds and a Hann frame of at most 65536 samples");
    }
    return std::max(kMinFrameSamples, std::bit_ceil(static_cast<std::size_t>(requiredSamples)));
}

/** per-band SRP-PHAT [edges.size()-1, bearings] of one-sided spectra [frames, M, F], bin k at k binHz, steered by `pairDelaysSeconds` [bearings, P] as from bearingPairDelays:
 * per pair i < j, G = X_i conj(X_j) / |G| (0 where |G| = 0) summed over the frames, Re(G e^{+j 2 pi f tau_ij}) summed over the band's bins,
 * divided by its bins, the pairs and the frames, so a coherent source at the bearing scores 1.
 * Finite edges increase strictly; intervals are half-open except the last upper endpoint. Bins outside are ignored; a band holding none is NaN.
 */
[[nodiscard]] inline std::vector<float> bandSteeredPower(std::span<const std::complex<float>> spectra, std::size_t M, std::size_t F, double binHz, std::span<const double> pairDelaysSeconds, std::span<const double> edges) {
    const std::size_t P = gr::algorithm::pairsOf(M);
    assert(M >= 2UZ && F > 0UZ && edges.size() >= 2UZ && std::ranges::all_of(edges, [](double value) { return std::isfinite(value); }) && std::ranges::adjacent_find(edges, std::greater_equal{}) == edges.end() && !spectra.empty() && spectra.size() % (M * F) == 0UZ && pairDelaysSeconds.size() % P == 0UZ);
    const std::size_t nBands    = edges.size() - 1UZ;
    const std::size_t nFrames   = spectra.size() / (M * F);
    const std::size_t nBearings = pairDelaysSeconds.size() / P;

    std::vector<std::complex<double>> phatCrossSpectra(P * F);
    for (std::size_t frame = 0UZ; frame < nFrames; ++frame) {
        const std::complex<float>* X = spectra.data() + frame * M * F;
        for (std::size_t p = 0UZ; p < P; ++p) {
            const auto [i, j] = gr::algorithm::pairOf(p, M);
            for (std::size_t k = 0UZ; k < F; ++k) {
                const std::complex<double> G         = std::complex<double>(X[i * F + k]) * std::conj(std::complex<double>(X[j * F + k]));
                const double               magnitude = std::abs(G);
                if (magnitude > 0.0) {
                    phatCrossSpectra[p * F + k] += G / magnitude;
                }
            }
        }
    }

    std::vector<double>      sums(nBands * nBearings, 0.0);
    std::vector<std::size_t> bins(nBands, 0UZ);
    for (std::size_t k = 0UZ; k < F; ++k) {
        const double f = static_cast<double>(k) * binHz;
        if (f < edges.front() || f > edges.back()) {
            continue;
        }
        const std::size_t band = std::min(static_cast<std::size_t>(std::ranges::upper_bound(edges, f) - edges.begin() - 1), nBands - 1UZ);
        ++bins[band];
        for (std::size_t b = 0UZ; b < nBearings; ++b) {
            for (std::size_t p = 0UZ; p < P; ++p) {
                const std::complex<double> steering = std::polar(1.0, 2.0 * std::numbers::pi * f * pairDelaysSeconds[b * P + p]);
                sums[band * nBearings + b] += (phatCrossSpectra[p * F + k] * steering).real();
            }
        }
    }

    std::vector<float> power(nBands * nBearings, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t band = 0UZ; band < nBands; ++band) {
        if (bins[band] == 0UZ) {
            continue;
        }
        const double contributions = static_cast<double>(bins[band] * P * nFrames);
        for (std::size_t b = 0UZ; b < nBearings; ++b) {
            power[band * nBearings + b] = static_cast<float>(sums[band * nBearings + b] / contributions);
        }
    }
    return power;
}

/**
 * one line on a line array's broadband steered power over bearings -90..+90 in equal steps (an odd count, 0 in the middle): the highest peak's bearing,
 * its side (centre within kCentreDegrees) and PairDelay's confidence (that peak over the second-highest local maximum); "no clear source" below kMinVerdictPeak.
 * A line cannot tell front from back, so a source is always "front or back".
 */
[[nodiscard]] inline std::string bearingVerdict(std::span<const float> power) {
    assert(power.size() >= 3UZ && power.size() % 2UZ == 1UZ);
    const auto                                                stepDegrees = static_cast<float>(2.0 * kFrontAngleLimitDegrees / static_cast<double>(power.size() - 1UZ));
    std::array<float, gr::localisation::detail::kPeakColumns> row{};
    gr::localisation::detail::findPairDelay(power.data(), power.size(), power.size(), stepDegrees, row.data());
    const auto [bearing, peak, confidence] = row;
    if (static_cast<double>(peak) < kMinVerdictPeak) {
        return std::format("no clear source (peak {:.2f})", peak);
    }
    const auto side = [bearing] {
        if (std::abs(static_cast<double>(bearing)) <= kCentreDegrees) {
            return "centre";
        }
        return bearing > 0.f ? "right" : "left";
    }();
    return std::format("source {:+.0f}° {}, confidence {:.1f}, front or back", bearing, side, confidence);
}

} // namespace gr::localisation

#endif // GNURADIO_LOCALISATION_BAND_STEERED_POWER_HPP
