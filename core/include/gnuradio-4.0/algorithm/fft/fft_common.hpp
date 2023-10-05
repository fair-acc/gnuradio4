#ifndef GRAPH_PROTOTYPE_ALGORITHM_FFT_COMMON_HPP
#define GRAPH_PROTOTYPE_ALGORITHM_FFT_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "fft_types.hpp"

namespace gr::algorithm {

template<ComplexType T, typename U>
void
computeMagnitudeSpectrum(const std::vector<T> &fftOut, std::vector<U> &magnitudeSpectrum, std::size_t fftSize, bool outputInDb) {
    if (fftOut.size() < magnitudeSpectrum.size()) {
        throw std::invalid_argument(fmt::format("FFT vector size ({}) must be more or equal than magnitude vector size ({}).", fftOut.size(), magnitudeSpectrum.size()));
    }
    using PrecisionType = typename T::value_type;
    std::transform(fftOut.begin(), std::next(fftOut.begin(), static_cast<std::ptrdiff_t>(magnitudeSpectrum.size())), magnitudeSpectrum.begin(), [fftSize, outputInDb](const auto &c) {
        const auto mag{ std::hypot(c.real(), c.imag()) * PrecisionType(2.0) / static_cast<PrecisionType>(fftSize) };
        return static_cast<U>(outputInDb ? PrecisionType(20.) * std::log10(std::abs(mag)) : mag);
    });
}

template<ComplexType T, typename U>
void
computePhaseSpectrum(const std::vector<T> &fftOut, std::vector<U> &phaseSpectrum) {
    if (fftOut.size() < phaseSpectrum.size()) {
        throw std::invalid_argument(fmt::format("FFT vector size ({}) must be more or equal than phase vector size ({}).", fftOut.size(), phaseSpectrum.size()));
    }
    std::transform(fftOut.begin(), std::next(fftOut.begin(), static_cast<std::ptrdiff_t>(phaseSpectrum.size())), phaseSpectrum.begin(),
                   [](const auto &c) { return static_cast<U>(std::atan2(c.imag(), c.real())); });
}

} // namespace gr::algorithm
#endif // GRAPH_PROTOTYPE_ALGORITHM_FFT_COMMON_HPP
