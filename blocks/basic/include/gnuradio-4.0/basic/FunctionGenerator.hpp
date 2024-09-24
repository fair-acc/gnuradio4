#ifndef GNURADIO_FUNCTION_GENERATOR_HPP
#define GNURADIO_FUNCTION_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

namespace gr::basic {

using namespace gr;

namespace function_generator {
enum class SignalType : int { Const, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse };
// TODO: Enum values should be capitalized CamelStyle, for the moment they are the same as FunctionGenerator class members
enum class ParameterType : int { signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1 };

using enum SignalType;
using enum ParameterType;
constexpr auto SignalTypeList = magic_enum::enum_values<SignalType>();

template<typename T>
requires std::is_same_v<T, SignalType> || std::is_same_v<T, ParameterType>
constexpr std::string toString(T type) {
    return std::string(magic_enum::enum_name(type));
}

template<typename EnumType, typename T>
requires std::is_same_v<EnumType, SignalType> || std::is_same_v<EnumType, ParameterType>
[[nodiscard]] std::pair<std::string, pmtv::pmt> createPropertyMapEntry(EnumType enumType, T value) {
    if constexpr (std::is_same_v<T, SignalType>) {
        return {toString(enumType), static_cast<pmtv::pmt>(toString(value))};
    } else {
        return {toString(enumType), static_cast<pmtv::pmt>(value)};
    }
}

template<typename T>
requires std::is_same_v<T, SignalType> || std::is_same_v<T, ParameterType>
constexpr T parse(std::string_view name) {
    auto type = magic_enum::enum_cast<T>(name, magic_enum::case_insensitive);
    if (!type.has_value()) {
        throw std::invalid_argument(fmt::format("parser error, unknown function_generator::{} - '{}'", meta::type_name<T>(), name));
    }
    return type.value();
}

template<typename T>
[[nodiscard]] property_map createConstPropertyMap(T startValue) {
    return {createPropertyMapEntry(signal_type, Const), createPropertyMapEntry(start_value, startValue)};
}

template<typename T>
[[nodiscard]] property_map createLinearRampPropertyMap(T startValue, T finalValue, T durationValue) {
    return {createPropertyMapEntry(signal_type, LinearRamp), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(duration, durationValue)};
}

template<typename T>
[[nodiscard]] property_map createParabolicRampPropertyMap(T startValue, T finalValue, T durationValue, T roundOffTime) {
    return {createPropertyMapEntry(signal_type, ParabolicRamp), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(duration, durationValue), createPropertyMapEntry(round_off_time, roundOffTime)};
}

template<typename T>
[[nodiscard]] property_map createCubicSplinePropertyMap(T startValue, T finalValue, T durationValue) {
    return {createPropertyMapEntry(signal_type, CubicSpline), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(duration, durationValue)};
}

template<typename T>
[[nodiscard]] property_map createImpulseResponsePropertyMap(T startValue, T finalValue, T time0, T time1) {
    return {createPropertyMapEntry(signal_type, ImpulseResponse), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(impulse_time0, time0), createPropertyMapEntry(impulse_time1, time1)};
}

} // namespace function_generator

template<typename T>
requires(std::floating_point<T>)
struct FunctionGenerator : public gr::Block<FunctionGenerator<T>, BlockingIO<true>> {
    using Description = Doc<R""(
@brief The `FunctionGenerator` class generates a variety of functions and their combinations.
It supports multiple function types, including Constant, Linear Ramp, Parabolic Ramp, Cubic Spline, and Impulse Response.
Each function type is configurable with specific parameters, enabling precise tailoring to meet the user's requirements.
Users can create sequences of various functions using tags.
These tags are generated through dedicated helper methods, named `create{FunctionName}Tag(...)`.
This method enables the flexible assembly of diverse function chains, offering significant versatility in signal generation.

Supported function types and their configurable parameters:

* Const
  - Constantly returns the same value, specified by 'startValue'.
  To create a `property_map` use `createConstPropertyMap(tagIndex, startValue)`.

* LinearRamp
  - Interpolates linearly between a start and a final value over a set duration.
  To create a `property_map` use `createLinearRampPropertyMap(tagIndex, startValue, finalValue, duration)`.

* ParabolicRamp
  - Produces a function with three sections: parabolic-linear-parabolic. The 'roundOffTime' parameter determines the length of the parabolic sections.
  To create a `property_map` use `createParabolicRampPropertyMap(tagIndex, startValue, finalValue, duration, roundOffTime)`.

* CubicSpline
  - Implements smooth cubic spline interpolation between start and final values over a specified duration.
  To create a `property_map` use `createCubicSplinePropertyMap(tagIndex, startValue, finalValue, duration)`.

* ImpulseResponse
  - Creates a function that spikes from 'time0' for a duration of 'time1'.
  To create a `property_map` use `createImpulseResponsePropertyMap(tagIndex, startValue, finalValue, time0, time1)`.

To create a chain of functions one can use `ClockSource`:
1) add time Tag entries, with time and command string:
@code
auto addTimeTagEntry = []<typename T>(gr::basic::ClockSource<T>& clockSource, std::uint64_t timeInNanoseconds, const std::string& value) {
    clockSource.tag_times.value.push_back(timeInNanoseconds);
    clockSource.tag_values.value.push_back(value);
};

auto &clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ gr::tag::SAMPLE_RATE(100.f), { "n_samples_max", 100 }, { "name", "ClockSource" } });
addTimeTagEntry(1'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=1");
addTimeTagEntry(10'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=2");
addTimeTagEntry(30'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=3");
@endcode

2) set parameters for all required contexts:
@code
funcGen.settings().set(createConstPropertyMap(5.f), SettingsCtx{now, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=1"});
funcGen.settings().set(createLinearRampPropertyMap(5.f, 30.f, .2f), SettingsCtx{now, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=2"});
funcGen.settings().set(createConstPropertyMap(30.f), SettingsCtx{now, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=3"});
@endcode

The parameters will automatically update when a Tag containing the "context" field arrives.
)"">;

    PortIn<T>  in; // ClockSource input
    PortOut<T> out;

    // TODO: Example of type aliases for sample_rate, make global and for all default Tags
    using SampleRate       = Annotated<float, "sample_rate", Visible, Doc<"stream sampling rate in [Hz]">>;
    SampleRate sample_rate = 1000.f;

    Annotated<std::string, "signal_type", Visible, Doc<"see function_generator::SignalType">> signal_type = function_generator::toString(function_generator::Const);

    // Parameters for different functions, not all parameters are used for all functions
    Annotated<T, "start_value">                                               start_value    = T(0.);
    Annotated<T, "final_value">                                               final_value    = T(0.);
    Annotated<T, "duration", Doc<"in sec">>                                   duration       = T(0.);
    Annotated<T, "round_off_time", Doc<"Specific to ParabolicRamp, in sec">>  round_off_time = T(0.);
    Annotated<T, "impulse_time0", Doc<"Specific to ImpulseResponse, in sec">> impulse_time0  = T(0.);
    Annotated<T, "impulse_time1", Doc<"Specific to ImpulseResponse, in sec">> impulse_time1  = T(0.);

    Annotated<property_map, "trigger_meta_info"> trigger_meta_info{};

    GR_MAKE_REFLECTABLE(FunctionGenerator, in, out, sample_rate, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, trigger_meta_info);

    T   _currentTime   = T(0.);
    int _sampleCounter = 0;

    function_generator::SignalType _signalType = function_generator::parse<function_generator::SignalType>(signal_type);
    T                              _timeTick   = T(1.) / static_cast<T>(sample_rate);

    void settingsChanged(const property_map& /*old_settings*/, const property_map& new_settings) {
        if (new_settings.contains(gr::tag::TRIGGER_META_INFO.shortKey()) || new_settings.contains(function_generator::toString(function_generator::signal_type))) {
            _currentTime = T(0.);
            _signalType  = function_generator::parse<function_generator::SignalType>(signal_type);
        }
        _timeTick = T(1.) / static_cast<T>(sample_rate);
    }

    [[nodiscard]] constexpr T processOne(T /*input*/) noexcept {
        _sampleCounter++;
        using enum function_generator::SignalType;
        T value{};

        switch (_signalType) {
        case Const: value = start_value; break;
        case LinearRamp: value = _currentTime > duration.value ? final_value.value : start_value + (final_value - start_value) * (_currentTime / duration); break;
        case ParabolicRamp: value = calculateParabolicRamp(); break;
        case CubicSpline: {
            const T normalizedTime  = _currentTime / duration;
            const T normalizedTime2 = T(3.) * std::pow(normalizedTime, T(2.));
            const T normalizedTime3 = T(2.) * std::pow(normalizedTime, T(3.));
            const T tmpValue        = (normalizedTime3 - normalizedTime2 + 1) * start_value + (-normalizedTime3 + normalizedTime2) * final_value;
            value                   = _currentTime > duration ? final_value.value : tmpValue;
            break;
        }
        case ImpulseResponse: value = (_currentTime < impulse_time0 || _currentTime > impulse_time0 + impulse_time1) ? start_value.value : final_value.value; break;
        default: value = T(0.);
        }
        _currentTime += _timeTick;
        return value;
    }

private:
    [[nodiscard]] constexpr T calculateParabolicRamp() {
        const T roundOnTime  = round_off_time; // assume that round ON and OFF times are equal
        const T linearLength = duration - (roundOnTime + round_off_time);
        const T a            = (final_value - start_value) / (T(2.) * roundOnTime * (linearLength + round_off_time));
        const T ar2          = a * std::pow(round_off_time, T(2.));
        const T slope        = (final_value - start_value - T(2.) * ar2) / linearLength;

        if (_currentTime > duration) {
            return final_value;
        }
        if (_currentTime < roundOnTime) {
            return start_value + a * std::pow(_currentTime, T(2.)); // first parabolic section
        }
        if (_currentTime < duration - round_off_time) {
            const T transitPoint1 = start_value + ar2;
            return transitPoint1 + slope * (_currentTime - round_off_time); // linear section
        }
        // second parabolic section
        const T shiftedTime   = _currentTime - (duration - round_off_time);
        const T transitPoint2 = final_value - ar2;
        return transitPoint2 + slope * shiftedTime - a * std::pow(shiftedTime, T(2.));
    }
};

} // namespace gr::basic

auto registerFunctionGenerator = gr::registerBlock<gr::basic::FunctionGenerator, float, double>(gr::globalBlockRegistry());

#endif // GNURADIO_FUNCTION_GENERATOR_HPP
