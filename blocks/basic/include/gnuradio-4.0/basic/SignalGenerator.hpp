#ifndef GNURADIO_SIGNAL_GENERATOR_HPP
#define GNURADIO_SIGNAL_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <numbers>

namespace gr::basic {

using namespace gr;

namespace signal_generator {
enum class Type : int { Const, Sin, Cos, Square, Saw, Triangle };
using enum Type;
constexpr auto                                 TypeList  = magic_enum::enum_values<Type>();
inline static constexpr gr::meta::fixed_string TypeNames = "[Const, Sin, Cos, Square, Saw, Triangle]";

constexpr Type
parse(std::string_view name) {
    auto signalType = magic_enum::enum_cast<Type>(name, magic_enum::case_insensitive);
    if (!signalType.has_value()) {
        throw std::invalid_argument(fmt::format("unknown signal generator type '{}'", name));
    }
    return signalType.value();
}

} // namespace signal_generator

template<std::floating_point T>
struct SignalGenerator : public gr::Block<SignalGenerator<T>, BlockingIO<true>> {
    using Description = Doc<R""(
@brief The SignalGenerator class generates various types of signal waveforms, including sine, cosine, square, constant, saw, and triangle signals.
Users can set parameters such as amplitude, frequency, offset, and phase for the desired waveform.
Note that not all parameters are supported for all signals.

The following signal shapes are supported (A = amplitude, f = frequency, P = phase, O = offset, t = time):

For the Square, Saw and Triangle t corresponds to phase adjusted time t = t + P / (pi2 * f)

* Sine
This waveform represents a smooth periodic oscillation.
formula: s(t) = A * sin(2 * pi * f * t + P) + O

* Cosine
Similar to the sine signal but shifted by pi/2 radians.
formula: s(t) = A * cos(2 * pi * f * t + P) + O

* Square
Represents a signal that alternates between two levels: -amplitude and amplitude.
If timeWithinPeriod < halfPeriod, y(t) = A + O else, y(t) = -A + O

* Constant
A signal that always returns a constant value, irrespective of time.
s(t) = A + O

* Saw (Sawtooth)
This waveform ramps upward and then sharply drops. It's called a sawtooth due to its resemblance to the profile of a saw blade.
s(t) = 2 * A * (t * f - floor(t * f + 0.5)) + O

* Triangle
This waveform linearly increases from -amplitude to amplitude in the first half of its period and then decreases back to -amplitude in the second half, forming a triangle shape.
s(t) = A * (4 * abs(t * f - floor(t * f + 0.75) + 0.25) - 1) + O
)"">;
    PortIn<T>  in; // ClockSource input
    PortOut<T> out;

    Annotated<float, "sample_rate", Visible, Doc<"sample rate">>                      sample_rate = 1000.f;
    Annotated<std::string, "signal_type", Visible, Doc<"see signal_generator::Type">> signal_type = "Sin";
    Annotated<T, "frequency", Visible>                                                frequency   = T(1.);
    Annotated<T, "amplitude", Visible>                                                amplitude   = T(1.);
    Annotated<T, "offset", Visible>                                                   offset      = T(0.);
    Annotated<T, "phase", Visible, Doc<"in rad">>                                     phase       = T(0.);

    T _currentTime = T(0.);

private:
    signal_generator::Type _signalType = signal_generator::parse(signal_type);
    T                      _timeTick   = T(1.) / T(sample_rate);

public:
    void
    settingsChanged(const property_map & /*old_settings*/, const property_map & /*new_settings*/) {
        _signalType = signal_generator::parse(signal_type);
        _timeTick   = T(1.) / T(sample_rate);
    }

    [[nodiscard]] constexpr T
    processOne(T /*input*/) noexcept {
        using enum signal_generator::Type;

        constexpr T pi2 = T(2.) * std::numbers::pi_v<T>;
        T           value{};
        T           phaseAdjustedTime = _currentTime + phase / (pi2 * frequency);

        switch (_signalType) {
        case Sin: value = amplitude * std::sin(pi2 * frequency * phaseAdjustedTime) + offset; break;
        case Cos: value = amplitude * std::cos(pi2 * frequency * phaseAdjustedTime) + offset; break;
        case Const: value = amplitude + offset; break;
        case Square: {
            T halfPeriod       = T(0.5) / frequency;
            T timeWithinPeriod = std::fmod(phaseAdjustedTime, T(2.) * halfPeriod);
            value              = (timeWithinPeriod < halfPeriod) ? amplitude + offset : -amplitude + offset;
            break;
        }
        case Saw: value = amplitude * (T(2.) * (phaseAdjustedTime * frequency - std::floor(phaseAdjustedTime * frequency + T(0.5)))) + offset; break;
        case Triangle: {
            T tmp = phaseAdjustedTime * frequency;
            value = amplitude * (T(4.0) * std::abs(tmp - std::floor(tmp + T(0.75)) + T(0.25)) - T(1.0)) + offset;
            break;
        }
        default: value = T(0.);
        }

        _currentTime += _timeTick;

        return value;
    }
};

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE(gr::basic::SignalGenerator, in, out, sample_rate, signal_type, frequency, amplitude, offset, phase);
auto registerSignalGenerator = gr::registerBlock<gr::basic::SignalGenerator, double, float>(gr::globalBlockRegistry());

#endif // GNURADIO_SIGNAL_GENERATOR_HPP
