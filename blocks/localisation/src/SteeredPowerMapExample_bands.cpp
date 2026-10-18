#include "SteeredPowerMapExample.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numbers>

#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>

namespace gr::examples {

std::vector<double> bearingPairDelays(std::span<const double> positions, std::span<const double> bearingsDegrees, double speed) {
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

std::size_t bandCountOf(double fMinHz, double fMaxHz) noexcept {
    assert(fMinHz < fMaxHz);
    return static_cast<std::size_t>(std::ceil((fMaxHz - fMinHz) / kBandWidthHz));
}

std::vector<float> bandSteeredPower(std::span<const std::complex<float>> spectra, std::size_t M, std::size_t F, double binHz, std::span<const double> pairDelaysSeconds, std::size_t nBands, double fMinHz, double fMaxHz) {
    const std::size_t P = gr::algorithm::pairsOf(M);
    assert(M >= 2UZ && F > 0UZ && nBands > 0UZ && fMinHz < fMaxHz && !spectra.empty() && spectra.size() % (M * F) == 0UZ && pairDelaysSeconds.size() % P == 0UZ);
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

    const double             bandHz = (fMaxHz - fMinHz) / static_cast<double>(nBands);
    std::vector<double>      sums(nBands * nBearings, 0.0);
    std::vector<std::size_t> bins(nBands, 0UZ);
    for (std::size_t k = 0UZ; k < F; ++k) {
        const double f = static_cast<double>(k) * binHz;
        if (f < fMinHz || f > fMaxHz) {
            continue;
        }
        const std::size_t band = std::min(static_cast<std::size_t>((f - fMinHz) / bandHz), nBands - 1UZ);
        ++bins[band];
        for (std::size_t b = 0UZ; b < nBearings; ++b) {
            for (std::size_t p = 0UZ; p < P; ++p) {
                const std::complex<double> steering = std::polar(1.0, 2.0 * std::numbers::pi * f * pairDelaysSeconds[b * P + p]);
                sums[band * nBearings + b] += (phatCrossSpectra[p * F + k] * steering).real();
            }
        }
    }

    std::vector<float> power(nBands * nBearings, 0.f);
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

std::string bearingVerdict(std::span<const float> power) {
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

} // namespace gr::examples
