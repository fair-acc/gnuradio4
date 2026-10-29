#ifndef GNURADIO_ALGORITHM_RAISED_COSINE_HPP
#define GNURADIO_ALGORITHM_RAISED_COSINE_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <vector>

namespace gr::algorithm::filter {

[[nodiscard]] inline double raisedCosine(double timeInSymbols, double excessBandwidth) noexcept {
    constexpr double kPi                        = std::numbers::pi;
    constexpr double kRemovableSingularityWidth = 1e-12;
    if (timeInSymbols == 0.) {
        return 1.;
    }
    const double sinc       = std::sin(kPi * timeInSymbols) / (kPi * timeInSymbols);
    const double scaledTime = 2. * excessBandwidth * timeInSymbols;
    if (std::abs(std::abs(scaledTime) - 1.) < kRemovableSingularityWidth) {
        return kPi / 4. * sinc;
    }
    return sinc * std::cos(kPi * excessBandwidth * timeInSymbols) / (1. - scaledTime * scaledTime);
}

[[nodiscard]] inline double rootRaisedCosine(double timeInSymbols, double excessBandwidth) noexcept {
    constexpr double kPi                        = std::numbers::pi;
    constexpr double kRemovableSingularityWidth = 1e-12;
    if (timeInSymbols == 0.) {
        return 1. + excessBandwidth * (4. / kPi - 1.);
    }
    const double scaledTime = 4. * excessBandwidth * timeInSymbols;
    if (std::abs(std::abs(scaledTime) - 1.) < kRemovableSingularityWidth) {
        const double quarterPhase = kPi / (4. * excessBandwidth);
        return excessBandwidth / std::numbers::sqrt2 * ((1. + 2. / kPi) * std::sin(quarterPhase) + (1. - 2. / kPi) * std::cos(quarterPhase));
    }
    return (std::sin(kPi * timeInSymbols * (1. - excessBandwidth)) + scaledTime * std::cos(kPi * timeInSymbols * (1. + excessBandwidth))) / (kPi * timeInSymbols * (1. - scaledTime * scaledTime));
}

template<std::floating_point T>
[[nodiscard]] std::vector<T> rootRaisedCosineTaps(std::size_t samplesPerSymbol, double excessBandwidth, std::size_t spanInSymbols) {
    const std::size_t   nTaps  = spanInSymbols * samplesPerSymbol + 1UZ;
    const double        centre = static_cast<double>(nTaps - 1UZ) / 2.;
    std::vector<double> pulse(nTaps);
    for (std::size_t i = 0UZ; i < nTaps; ++i) {
        pulse[i] = rootRaisedCosine((static_cast<double>(i) - centre) / static_cast<double>(samplesPerSymbol), excessBandwidth);
    }
    const double   unitEnergyScale = 1. / std::sqrt(std::inner_product(pulse.begin(), pulse.end(), pulse.begin(), 0.));
    std::vector<T> taps(nTaps);
    std::ranges::transform(pulse, taps.begin(), [unitEnergyScale](double value) { return static_cast<T>(value * unitEnergyScale); });
    return taps;
}

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_RAISED_COSINE_HPP
