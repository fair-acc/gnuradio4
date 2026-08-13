#ifndef GNURADIO_ALGORITHM_FFT_COMMON_HPP
#define GNURADIO_ALGORITHM_FFT_COMMON_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

#include <format>
#include <ranges>

namespace gr::algorithm::fft {

// per-element core: multiplies a (real or complex) sample by a real window coefficient
template<typename T, std::floating_point W>
[[nodiscard]] constexpr T applyWindowOne(T sample, W coefficient) noexcept {
    return sample * coefficient;
}

// kernel-callable core: allocation-free, span in/out, plain index loop
template<typename T, std::floating_point W>
constexpr void applyWindow(std::span<T> samples, std::span<const W> window) noexcept {
    assert(samples.size() == window.size());
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        samples[i] = applyWindowOne(samples[i], window[i]);
    }
}

// maps a shifted-spectrum output index to its natural-order (unshifted) input index (fftshift);
// half-spectrum selection needs no remapping at all (output index == input index)
[[nodiscard]] constexpr std::size_t fftShiftIndex(std::size_t outputIndex, std::size_t fftSize) noexcept { return (outputIndex + fftSize / 2UZ) % fftSize; }

struct ConfigMagnitude {
    bool computeHalfSpectrum = false;
    bool includeNyquist      = false; // rfft convention: DC..Nyquist inclusive, N/2+1 bins instead of N/2
    bool outputInDb          = false;
    bool shiftSpectrum       = false;
};

// per-element core: |fftBin|, normalised by fftSize, with optional linear->dB conversion
template<typename T, typename PrecisionType = typename T::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
[[nodiscard]] constexpr PrecisionType computeMagnitudeOne(T fftBin, std::size_t fftSize, bool outputInDb) noexcept {
    const auto mag{std::hypot(fftBin.real(), fftBin.imag()) * PrecisionType(2.) / static_cast<PrecisionType>(fftSize)};
    if (outputInDb && mag > PrecisionType(0)) { // avoids log of zero
        return PrecisionType(20.) * std::log10(mag);
    } else if (outputInDb) {
        return std::numeric_limits<PrecisionType>::lowest(); // represents -infinity in dB
    }
    return mag;
}

template<std::ranges::input_range TContainerIn, std::ranges::output_range<typename TContainerIn::value_type::value_type> TContainerOut = std::vector<typename TContainerIn::value_type::value_type>, typename T = TContainerIn::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
auto computeMagnitudeSpectrum(const TContainerIn& fftIn, TContainerOut&& magOut = {}, ConfigMagnitude config = {}) {
    const std::size_t N = fftIn.size();
    if (N == 0) {
        throw std::invalid_argument("fftIn cannot be empty.");
    }

    const std::size_t magSize = config.computeHalfSpectrum ? (N / 2UZ + (config.includeNyquist ? 1UZ : 0UZ)) : N;
    if constexpr (requires(std::size_t n) { magOut.resize(n); }) {
        if (magOut.size() != magSize) {
            magOut.resize(magSize);
        }
    } else {
        static_assert(std::tuple_size_v<TContainerIn> == std::tuple_size_v<TContainerOut>, "Size mismatch for fixed-size container.");
    }

    using PrecisionType = typename T::value_type;
    const std::span<PrecisionType> magSpan(std::span<PrecisionType>(magOut).first(magSize));
    for (std::size_t i = 0UZ; i < magSpan.size(); ++i) {
        magSpan[i] = computeMagnitudeOne(fftIn[i], N, config.outputInDb);
    }

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
    bool includeNyquist      = false; // rfft convention: DC..Nyquist inclusive, N/2+1 bins instead of N/2
    bool outputInDeg         = false;
    bool unwrapPhase         = false;
    bool shiftSpectrum       = false;
};

// kernel-callable core: allocation-free, in-place, single serial pass over the span.
// derives an integer correction count from raw consecutive differences; its prefix sum is exact.
// precondition: phase in RADIANS -- the pi threshold is a radian quantity.
template<std::floating_point T>
constexpr void unwrapPhase(std::span<T> phase) noexcept {
    if (phase.size() < 2UZ) {
        return;
    }
    const T   pi          = std::numbers::pi_v<T>;
    T         previousRaw = phase[0];
    long long k           = 0;
    for (std::size_t i = 1UZ; i < phase.size(); ++i) {
        const T currentRaw = phase[i];
        const T rawDiff    = currentRaw - previousRaw;
        if (std::isfinite(rawDiff)) { // a NaN bin must not poison the running count for every later bin
            // comparisons rather than llrint, which is an unresolved extern under AdaptiveCpp SSCP. Looping because
            // the public container overload accepts phase that never came from atan2 and can jump by more than 2pi;
            // for atan2 output this runs at most one iteration. An exact +/-pi tie stays uncorrected.
            for (T d = rawDiff; d > pi; d -= static_cast<T>(2) * pi) {
                --k;
            }
            for (T d = rawDiff; d < -pi; d += static_cast<T>(2) * pi) {
                ++k;
            }
        }
        phase[i]    = currentRaw + static_cast<T>(2) * pi * static_cast<T>(k);
        previousRaw = currentRaw;
    }
}

template<std::ranges::input_range TContainerInOut, typename T = TContainerInOut::value_type>
requires(std::floating_point<T>)
void unwrapPhase(TContainerInOut& phase) {
    unwrapPhase<T>(std::span<T>{phase});
}

// per-element core: atan2(imag, real)
template<typename T, typename PrecisionType = typename T::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
[[nodiscard]] constexpr PrecisionType computePhaseOne(T fftBin) noexcept {
    return std::atan2(fftBin.imag(), fftBin.real());
}

// per-element core: radians -> degrees
template<std::floating_point PrecisionType>
[[nodiscard]] constexpr PrecisionType radToDeg(PrecisionType radians) noexcept {
    return radians * static_cast<PrecisionType>(180.) * std::numbers::inv_pi_v<PrecisionType>;
}

template<std::ranges::input_range TContainerIn, std::ranges::output_range<typename TContainerIn::value_type::value_type> TContainerOut = std::vector<typename TContainerIn::value_type::value_type>, typename T = TContainerIn::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
auto computePhaseSpectrum(const TContainerIn& fftIn, TContainerOut&& phaseOut = {}, ConfigPhase config = {}) {
    const std::size_t N = fftIn.size();
    if (N == 0) {
        throw std::invalid_argument("fftIn cannot be empty.");
    }

    std::size_t phaseSize = config.computeHalfSpectrum ? (N / 2 + (config.includeNyquist ? 1UZ : 0UZ)) : N;
    if constexpr (requires(std::size_t n) { phaseOut.resize(n); }) {
        if (phaseOut.size() != phaseSize) {
            phaseOut.resize(phaseSize);
        }
    } else {
        static_assert(std::tuple_size_v<TContainerIn> == std::tuple_size_v<TContainerOut>, "Size mismatch for fixed-size container.");
    }

    using PrecisionType = typename T::value_type;
    const std::span<PrecisionType> phaseSpan{phaseOut};
    for (std::size_t i = 0UZ; i < phaseSpan.size(); ++i) {
        phaseSpan[i] = computePhaseOne(fftIn[i]);
    }

    if (config.unwrapPhase) {
        unwrapPhase(phaseOut);
    }

    if (config.outputInDeg) {
        std::ranges::transform(phaseOut, phaseOut.begin(), [](const auto& phase) { return radToDeg(phase); });
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
