#ifndef GNURADIO_ALGORITHM_FFT_COMMON_HPP
#define GNURADIO_ALGORITHM_FFT_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <fmt/format.h>
#include <ranges>

namespace gr::algorithm::fft {

struct ConfigMagnitude {
    bool computeHalfSpectrum = false;
    bool outputInDb          = false;
    bool shiftSpectrum       = false;
};

template<std::ranges::input_range TContainerIn, std::ranges::output_range<typename TContainerIn::value_type::value_type> TContainerOut = std::vector<typename TContainerIn::value_type::value_type>, typename T = TContainerIn::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
auto computeMagnitudeSpectrum(const TContainerIn& fftIn, TContainerOut&& magOut = {}, ConfigMagnitude config = {}) {
    const std::size_t N = fftIn.size();
    if (N == 0) {
        throw std::invalid_argument("fftIn cannot be empty.");
    }

    const std::size_t magSize = config.computeHalfSpectrum ? (N / 2UZ) : N;
    if constexpr (requires(std::size_t n) { magOut.resize(n); }) {
        if (magOut.size() != magSize) {
            magOut.resize(magSize);
        }
    } else {
        static_assert(std::tuple_size_v<TContainerIn> == std::tuple_size_v<TContainerOut>, "Size mismatch for fixed-size container.");
    }

    using PrecisionType = typename T::value_type;
    std::transform(fftIn.begin(), std::next(fftIn.begin(), static_cast<std::ptrdiff_t>(magSize)), magOut.begin(), [N, outputInDb = config.outputInDb](const auto& c) {
        const auto mag{std::hypot(c.real(), c.imag()) * PrecisionType(2.) / static_cast<PrecisionType>(N)};
        if (outputInDb && mag > PrecisionType(0)) { // avoids log of zero
            return PrecisionType(20.) * std::log10(mag);
        } else if (outputInDb) {
            return std::numeric_limits<PrecisionType>::lowest(); // represents -infinity in dB
        }
        return mag;
    });

    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        if (!config.computeHalfSpectrum && config.shiftSpectrum) {
            auto halfN = std::ssize(magOut) / 2;
            std::ranges::rotate(magOut, std::ranges::begin(magOut) + halfN); // rotate so that negative frequencies appear at the front
        }
    }

    return magOut;
}

template<std::ranges::input_range TContainerIn, typename T = TContainerIn::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
auto computeMagnitudeSpectrum(const TContainerIn& fftIn, ConfigMagnitude config) {
    return computeMagnitudeSpectrum(fftIn, {}, config);
}

struct ConfigPhase {
    bool computeHalfSpectrum = false;
    bool outputInDeg         = false;
    bool unwrapPhase         = false;
    bool shiftSpectrum       = false;
};

template<std::ranges::input_range TContainerInOut, typename T = TContainerInOut::value_type>
requires(std::floating_point<T>)
void unwrapPhase(TContainerInOut& phase) {
    const auto pi   = std::numbers::pi_v<T>;
    auto       prev = phase.front();
    std::transform(phase.begin() + 1, phase.end(), phase.begin() + 1, [&prev, pi](T& current) {
        T diff = current - prev;
        while (diff > pi) {
            current -= 2 * pi;
            diff = current - prev;
        }
        while (diff < -pi) {
            current += 2 * pi;
            diff = current - prev;
        }
        prev = current;
        return current;
    });
}

template<std::ranges::input_range TContainerIn, std::ranges::output_range<typename TContainerIn::value_type::value_type> TContainerOut = std::vector<typename TContainerIn::value_type::value_type>, typename T = TContainerIn::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
auto computePhaseSpectrum(const TContainerIn& fftIn, TContainerOut&& phaseOut = {}, ConfigPhase config = {}) {
    const std::size_t N = fftIn.size();
    if (N == 0) {
        throw std::invalid_argument("fftIn cannot be empty.");
    }

    std::size_t phaseSize = config.computeHalfSpectrum ? (N / 2) : N;
    if constexpr (requires(std::size_t n) { phaseOut.resize(n); }) {
        if (phaseOut.size() != phaseSize) {
            phaseOut.resize(phaseSize);
        }
    } else {
        static_assert(std::tuple_size_v<TContainerIn> == std::tuple_size_v<TContainerOut>, "Size mismatch for fixed-size container.");
    }
    std::transform(fftIn.begin(), std::next(fftIn.begin(), static_cast<std::ptrdiff_t>(phaseOut.size())), phaseOut.begin(), [](const auto& c) { return std::atan2(c.imag(), c.real()); });

    if (config.unwrapPhase) {
        unwrapPhase(phaseOut);
    }

    if (config.outputInDeg) {
        std::ranges::transform(phaseOut, phaseOut.begin(), [](const auto& phase) { return phase * static_cast<typename T::value_type>(180.) * std::numbers::inv_pi_v<typename T::value_type>; });
    }

    if (!config.computeHalfSpectrum && config.shiftSpectrum) {
        auto halfN = std::ssize(phaseOut) / 2;
        std::ranges::rotate(phaseOut, phaseOut.begin() + halfN); // rotate so that negative frequencies appear at the front
    }

    return phaseOut;
}

template<std::ranges::input_range TContainerIn, typename T = TContainerIn::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
auto computePhaseSpectrum(const TContainerIn& fftIn, ConfigPhase config) {
    return computePhaseSpectrum(fftIn, {}, config);
}

} // namespace gr::algorithm::fft
#endif // GNURADIO_ALGORITHM_FFT_COMMON_HPP
