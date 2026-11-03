#ifndef GNURADIO_FUNCTION_GENERATOR_CORE_HPP
#define GNURADIO_FUNCTION_GENERATOR_CORE_HPP

#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>
#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

#include <concepts>
#include <utility>

namespace gr::basic {

namespace function_generator {
enum class SignalType : int { Const, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse, UniformNoise, TriangularNoise, GaussianNoise, Sin, Cos, FastSin, FastCos };

using enum SignalType;

constexpr bool isToneType(SignalType type) noexcept { return type == Sin || type == Cos || type == FastSin || type == FastCos; }

constexpr gr::signal::ToneType toToneType(SignalType type) noexcept {
    switch (type) {
    case Sin: return gr::signal::ToneType::Sin;
    case Cos: return gr::signal::ToneType::Cos;
    case FastSin: return gr::signal::ToneType::FastSin;
    case FastCos: return gr::signal::ToneType::FastCos;
    default: std::unreachable();
    }
}
} // namespace function_generator

template<typename T>
struct FunctionGeneratorCore {
    struct Parameters {
        function_generator::SignalType signalType   = function_generator::Const;
        double                         startValue   = 0.;
        double                         finalValue   = 0.;
        double                         duration     = 0.; // in s, as are the times below
        double                         roundOffTime = 0.;
        double                         impulseTime0 = 0.;
        double                         impulseTime1 = 0.;
    };

    double elapsedSeconds      = 0.;
    double samplePeriodSeconds = 0.001;

    gr::signal::NoiseGenerator<double> noise;
    gr::signal::ToneGenerator<double>  tone;

    [[nodiscard]] T generateSample(const Parameters& parameters) noexcept {
        using enum function_generator::SignalType;

        if (parameters.signalType == UniformNoise || parameters.signalType == TriangularNoise || parameters.signalType == GaussianNoise) {
            elapsedSeconds += samplePeriodSeconds;
            if constexpr (gr::signal::detail::is_complex_v<T>) {
                using F           = typename T::value_type;
                const auto sample = noise.generateComplexSample();
                return T(static_cast<F>(sample.real()), static_cast<F>(sample.imag()));
            } else {
                return toOutputType(noise.generateSample());
            }
        }

        if (function_generator::isToneType(parameters.signalType)) {
            const double sv      = static_cast<double>(parameters.startValue);
            const double dur     = static_cast<double>(parameters.duration);
            const bool   expired = dur > 0. && elapsedSeconds > dur;
            elapsedSeconds += samplePeriodSeconds;
            if (expired) {
                if constexpr (gr::signal::detail::is_complex_v<T>) {
                    using F = typename T::value_type;
                    return T(static_cast<F>(sv), F(0));
                } else {
                    return toOutputType(sv);
                }
            }
            if constexpr (gr::signal::detail::is_complex_v<T>) {
                using F           = typename T::value_type;
                const auto sample = tone.generateComplexSample();
                return T(static_cast<F>(sample.real()), static_cast<F>(sample.imag()));
            } else {
                return toOutputType(tone.generateSample());
            }
        }

        const double sv  = static_cast<double>(parameters.startValue);
        const double fv  = static_cast<double>(parameters.finalValue);
        const double dur = static_cast<double>(parameters.duration);

        double value{};
        switch (parameters.signalType) {
        case Const: value = sv; break;
        case LinearRamp: value = elapsedSeconds > dur ? fv : sv + (fv - sv) * (elapsedSeconds / dur); break;
        case ParabolicRamp: value = calculateParabolicRamp(parameters); break;
        case CubicSpline: {
            const double t  = elapsedSeconds / dur;
            const double t2 = 3. * t * t;
            const double t3 = 2. * t * t * t;
            value           = elapsedSeconds > dur ? fv : (t3 - t2 + 1.) * sv + (-t3 + t2) * fv;
            break;
        }
        case ImpulseResponse: {
            const double it0 = static_cast<double>(parameters.impulseTime0);
            const double it1 = static_cast<double>(parameters.impulseTime1);
            value            = (elapsedSeconds < it0 || elapsedSeconds > it0 + it1) ? sv : fv;
            break;
        }
        default: value = 0.;
        }
        elapsedSeconds += samplePeriodSeconds;

        if constexpr (gr::signal::detail::is_complex_v<T>) {
            using F = typename T::value_type;
            return T(static_cast<F>(value), F(0));
        } else {
            return toOutputType(value);
        }
    }

private:
    [[nodiscard]] double calculateParabolicRamp(const Parameters& parameters) const noexcept {
        const double sv  = static_cast<double>(parameters.startValue);
        const double fv  = static_cast<double>(parameters.finalValue);
        const double dur = static_cast<double>(parameters.duration);
        const double rot = static_cast<double>(parameters.roundOffTime);

        const double linearLength = dur - 2. * rot;
        const double a            = (fv - sv) / (2. * rot * (linearLength + rot));
        const double ar2          = a * rot * rot;
        const double slope        = (fv - sv - 2. * ar2) / linearLength;

        if (elapsedSeconds > dur) {
            return fv;
        }
        if (elapsedSeconds < rot) {
            return sv + a * elapsedSeconds * elapsedSeconds;
        }
        if (elapsedSeconds < dur - rot) {
            return sv + ar2 + slope * (elapsedSeconds - rot);
        }
        const double shiftedTime = elapsedSeconds - (dur - rot);
        return fv - ar2 + slope * shiftedTime - a * shiftedTime * shiftedTime;
    }

    static constexpr T toOutputType(double value) noexcept {
        if constexpr (std::same_as<T, double>) {
            return value;
        } else if constexpr (std::floating_point<T>) {
            return static_cast<T>(value);
        } else if constexpr (std::integral<T>) {
            return gr::signal::detail::clampToInt<T>(value);
        } else {
            static_assert(std::floating_point<T> || std::integral<T>, "unsupported output type");
        }
    }
};

} // namespace gr::basic

#endif // GNURADIO_FUNCTION_GENERATOR_CORE_HPP
