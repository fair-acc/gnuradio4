#ifndef GNURADIO_ALGORITHM_SIGNAL_GENERATOR_CORE_HPP
#define GNURADIO_ALGORITHM_SIGNAL_GENERATOR_CORE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <type_traits>

#include <gnuradio-4.0/algorithm/BurstTaper.hpp>
#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

namespace gr::signal {

enum class SignalType : int { Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos, UniformNoise, TriangularNoise, GaussianNoise, LinearChirp, ExponentialChirp };

struct ChirpParameters {
    double startHz         = 1.0;
    double stopHz          = 10.0;
    double durationSeconds = 1.0;
    bool   repeat          = true; // false: the output holds the offset once the chirp has ended
    double periodSeconds   = 0.0;  // 0: back to back
    double tukeyFraction   = 0.0;  // 0..1
};

// precondition: durationSeconds > 0; an exponential chirp also needs startHz > 0 and stopHz > 0
[[nodiscard]] inline double chirpPhase(SignalType type, double timeSeconds, const ChirpParameters& chirp, double phaseRad) noexcept {
    constexpr double kTwoPi = 2.0 * std::numbers::pi;
    if (type == SignalType::ExponentialChirp && chirp.startHz != chirp.stopHz) {
        const double rate = std::log(chirp.stopHz / chirp.startHz) / chirp.durationSeconds;
        return phaseRad + kTwoPi * chirp.startHz * std::expm1(rate * timeSeconds) / rate;
    }
    return phaseRad + kTwoPi * (chirp.startHz * timeSeconds + 0.5 * (chirp.stopHz - chirp.startHz) * timeSeconds * timeSeconds / chirp.durationSeconds);
}

/** F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform," Proc. IEEE, vol. 66, no. 1, pp. 51-83, 1978 */
[[nodiscard]] inline double tukeyWeight(double timeSeconds, double durationSeconds, double tukeyFraction) noexcept {
    const double edge     = 0.5 * tukeyFraction * durationSeconds;
    const double fromEdge = std::min(timeSeconds, durationSeconds - timeSeconds);
    return fromEdge >= edge ? 1.0 : gr::algorithm::BurstTaper<double>::computeRise(gr::algorithm::TaperType::RaisedCosine, fromEdge / edge, 1.0);
}

namespace detail {

template<typename T>
struct is_complex : std::false_type {};
template<typename F>
struct is_complex<std::complex<F>> : std::true_type {};
template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Computation float type: double for all scalar types, base type for complex
template<typename T>
struct compute_float {
    using type = double;
};
template<>
struct compute_float<std::complex<float>> {
    using type = float;
};
template<>
struct compute_float<std::complex<double>> {
    using type = double;
};
template<typename T>
using compute_float_t = typename compute_float<T>::type;

constexpr bool isToneType(SignalType t) noexcept { return static_cast<int>(t) <= static_cast<int>(SignalType::FastCos); }

constexpr ToneType toToneType(SignalType t) noexcept { return static_cast<ToneType>(static_cast<int>(t)); }

constexpr NoiseType toNoiseType(SignalType t) noexcept { return static_cast<NoiseType>(static_cast<int>(t) - static_cast<int>(SignalType::UniformNoise)); }

template<std::integral T, std::floating_point F>
constexpr T clampToInt(F value) noexcept {
    // Comparison is safe even when static_cast<F>(max) rounds up:
    // if value >= rounded_max, we return max; otherwise the cast is in range.
    if (value >= static_cast<F>(std::numeric_limits<T>::max())) {
        return std::numeric_limits<T>::max();
    }
    if (value <= static_cast<F>(std::numeric_limits<T>::min())) {
        return std::numeric_limits<T>::min();
    }
    return static_cast<T>(value);
}

} // namespace detail

/**
 * @brief Unified signal generation core dispatching to ToneGenerator or NoiseGenerator.
 *
 * Template parameter T is the output type (any GR4 fundamental type).
 * Computes in double for scalar output, or in the complex base type for complex output.
 * Integer output: truncate + clamp to [min, max]. Float/double: identity or precision cast.
 */
template<typename T>
struct SignalGeneratorCore {
    using F = detail::compute_float_t<T>;

    SignalType        _type = SignalType::Sin;
    ToneGenerator<F>  _tone;
    NoiseGenerator<F> _noise;
    std::uint64_t     _seed           = 0;
    double            _sampleRate     = 1000.0;
    std::uint64_t     nextChirpSample = 0U;
    ChirpParameters   chirp;

    void configure(SignalType type, double frequency, float sampleRate, float phase, float amplitude, float offset, std::uint64_t seed) noexcept {
        _type       = type;
        _seed       = seed;
        _sampleRate = static_cast<double>(sampleRate);
        if (detail::isToneType(type) || type == SignalType::LinearChirp || type == SignalType::ExponentialChirp) {
            const ToneType tone = detail::isToneType(type) ? detail::toToneType(type) : ToneType::Sin;
            _tone.configure(tone, static_cast<F>(frequency), static_cast<F>(sampleRate), static_cast<F>(phase), static_cast<F>(amplitude), static_cast<F>(offset));
        } else {
            _noise.configure(detail::toNoiseType(type), static_cast<F>(amplitude), static_cast<F>(offset), seed);
        }
    }

    void reset() noexcept {
        _tone.reset();
        _noise.reset(_seed);
        nextChirpSample = 0U;
    }

    [[nodiscard]] double secondsIntoSweep(std::uint64_t n) const noexcept {
        const double elapsed = static_cast<double>(n) / _sampleRate;
        const double period  = chirp.periodSeconds > 0.0 ? chirp.periodSeconds : chirp.durationSeconds;
        return chirp.repeat ? std::fmod(elapsed, period) : elapsed;
    }

    [[nodiscard]] std::optional<double> secondsOfSweepStartingAt(std::uint64_t n) const noexcept {
        const double time = secondsIntoSweep(n);
        if (n != 0U && time >= secondsIntoSweep(n - 1U)) {
            return std::nullopt;
        }
        return static_cast<double>(n) / _sampleRate - time;
    }

    [[nodiscard]] T generateSample() noexcept {
        if (_type == SignalType::LinearChirp || _type == SignalType::ExponentialChirp) {
            const double time   = secondsIntoSweep(nextChirpSample++);
            const bool   active = time < chirp.durationSeconds;
            const double angle  = chirpPhase(_type, time, chirp, static_cast<double>(_tone._phase));
            const F      weight = active ? _tone._amplitude * static_cast<F>(tukeyWeight(time, chirp.durationSeconds, chirp.tukeyFraction)) : F{};
            const F      real   = weight * static_cast<F>(std::sin(angle)) + _tone._offset;
            if constexpr (detail::is_complex_v<T>) {
                return T{real, -weight * static_cast<F>(std::cos(angle))};
            } else {
                return convert(real);
            }
        }
        if constexpr (detail::is_complex_v<T>) {
            if (detail::isToneType(_type)) {
                return _tone.generateComplexSample();
            } else {
                return _noise.generateComplexSample();
            }
        } else {
            const F raw = detail::isToneType(_type) ? _tone.generateSample() : _noise.generateSample();
            return convert(raw);
        }
    }

    void fill(std::span<T> out) noexcept {
        if (_type == SignalType::LinearChirp || _type == SignalType::ExponentialChirp) {
            for (T& sample : out) {
                sample = generateSample();
            }
            return;
        }
        if constexpr (detail::is_complex_v<T>) {
            if (detail::isToneType(_type)) {
                _tone.fillComplex(out);
            } else {
                _noise.fillComplex(out);
            }
        } else if constexpr (std::same_as<T, F>) {
            if (detail::isToneType(_type)) {
                _tone.fill(out);
            } else {
                _noise.fill(out);
            }
        } else {
            if (detail::isToneType(_type)) {
                for (auto& sample : out) {
                    sample = convert(_tone.generateSample());
                }
            } else {
                for (auto& sample : out) {
                    sample = convert(_noise.generateSample());
                }
            }
        }
    }

private:
    static constexpr T convert(F value) noexcept {
        if constexpr (std::same_as<T, F>) {
            return value;
        } else if constexpr (std::floating_point<T>) {
            return static_cast<T>(value);
        } else if constexpr (std::integral<T>) {
            return detail::clampToInt<T>(value);
        } else {
            static_assert(std::floating_point<T> || std::integral<T>, "unsupported output type");
        }
    }
};

} // namespace gr::signal

#endif // GNURADIO_ALGORITHM_SIGNAL_GENERATOR_CORE_HPP
