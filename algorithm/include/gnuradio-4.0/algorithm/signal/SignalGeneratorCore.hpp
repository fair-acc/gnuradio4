#ifndef GNURADIO_ALGORITHM_SIGNAL_GENERATOR_CORE_HPP
#define GNURADIO_ALGORITHM_SIGNAL_GENERATOR_CORE_HPP

#include <complex>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

namespace gr::signal {

enum class SignalType : int { Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos, UniformNoise, TriangularNoise, GaussianNoise };

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
    std::uint64_t     _seed = 0;

    void configure(SignalType type, float frequency, float sampleRate, float phase, float amplitude, float offset, std::uint64_t seed) noexcept {
        _type = type;
        _seed = seed;
        if (detail::isToneType(type)) {
            _tone.configure(detail::toToneType(type), static_cast<F>(frequency), static_cast<F>(sampleRate), static_cast<F>(phase), static_cast<F>(amplitude), static_cast<F>(offset));
        } else {
            _noise.configure(detail::toNoiseType(type), static_cast<F>(amplitude), static_cast<F>(offset), seed);
        }
    }

    void reset() noexcept {
        _tone.reset();
        _noise.reset(_seed);
    }

    [[nodiscard]] T generateSample() noexcept {
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
