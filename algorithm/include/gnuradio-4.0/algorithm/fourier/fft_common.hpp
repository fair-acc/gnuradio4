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

#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::algorithm::fft {

template<typename T, std::floating_point W>
[[nodiscard]] GR_DEVICE_FN constexpr T applyWindowOne(T sample, W coefficient) noexcept {
    return sample * coefficient;
}

// kernel-callable: allocation-free, span in/out
template<typename T, std::floating_point W>
constexpr void applyWindow(std::span<T> samples, std::span<const W> window) noexcept {
    assert(samples.size() == window.size());
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        samples[i] = applyWindowOne(samples[i], window[i]);
    }
}

// half-spectrum selection needs no remapping: output index == input index
[[nodiscard]] constexpr std::size_t fftShiftIndex(std::size_t outputIndex, std::size_t fftSize) noexcept { return (outputIndex + fftSize / 2UZ) % fftSize; }

struct ConfigMagnitude {
    bool computeHalfSpectrum = false;
    bool includeNyquist      = false; // rfft convention: DC..Nyquist inclusive, N/2+1 bins instead of N/2
    bool outputInDb          = false;
    bool shiftSpectrum       = false;
};

// normalised by fftSize, optional linear->dB
template<typename T, typename PrecisionType = typename T::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
[[nodiscard]] GR_DEVICE_FN constexpr PrecisionType computeMagnitudeOne(T fftBin, std::size_t fftSize, bool outputInDb, bool hasMirrorTwin = true) noexcept {
    // a one-sided amplitude spectrum folds each bin onto its negative-frequency twin, so it doubles -- but DC and,
    // for even N, Nyquist have no twin and take 1/N
    const auto scale{(hasMirrorTwin ? PrecisionType(2.) : PrecisionType(1.)) / static_cast<PrecisionType>(fftSize)};
    const auto mag{std::hypot(fftBin.real(), fftBin.imag()) * scale};
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
    const std::size_t              nyquist = N / 2UZ;
    for (std::size_t i = 0UZ; i < magSpan.size(); ++i) {
        // only a folded spectrum has twins to sum: there, DC and -- for even N -- Nyquist are their own mirror
        // image and take 1/N, where every other bin takes 2/N. A two-sided spectrum keeps the legacy scaling.
        const bool hasMirrorTwin = !config.computeHalfSpectrum || (i != 0UZ && !(N % 2UZ == 0UZ && i == nyquist));
        magSpan[i]               = computeMagnitudeOne(fftIn[i], N, config.outputInDb, hasMirrorTwin);
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

// kernel-callable: allocation-free, in-place
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
            // closed form, not a subtract-in-a-loop: the loop does not terminate once the difference is large
            // enough that d - 2*pi == d. Each direction is counted separately to keep the tie convention -- an
            // exact +pi or -pi stays uncorrected. (floor, not llrint: llrint is unresolved under AdaptiveCpp SSCP.)
            const T twoPi = static_cast<T>(2) * pi;
            if (rawDiff > pi) {
                k -= static_cast<long long>(std::floor((rawDiff - pi) / twoPi)) + 1LL;
            } else if (rawDiff < -pi) {
                k += static_cast<long long>(std::floor((-rawDiff - pi) / twoPi)) + 1LL;
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

template<typename T, typename PrecisionType = typename T::value_type>
requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
[[nodiscard]] GR_DEVICE_FN constexpr PrecisionType computePhaseOne(T fftBin) noexcept {
    return std::atan2(fftBin.imag(), fftBin.real());
}

template<std::floating_point PrecisionType>
[[nodiscard]] GR_DEVICE_FN constexpr PrecisionType radToDeg(PrecisionType radians) noexcept {
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
