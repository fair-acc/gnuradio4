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
constexpr std::string
toString(T type) {
    return std::string(magic_enum::enum_name(type));
}

template<typename EnumType, typename T>
    requires std::is_same_v<EnumType, SignalType> || std::is_same_v<EnumType, ParameterType>
[[nodiscard]] std::pair<std::string, pmtv::pmt>
createPropertyMapEntry(EnumType enumType, T value) {
    if constexpr (std::is_same_v<T, SignalType>) {
        return { toString(enumType), static_cast<pmtv::pmt>(toString(value)) };
    } else {
        return { toString(enumType), static_cast<pmtv::pmt>(value) };
    }
}

template<typename T>
    requires std::is_same_v<T, SignalType> || std::is_same_v<T, ParameterType>
constexpr T
parse(std::string_view name) {
    auto type = magic_enum::enum_cast<T>(name, magic_enum::case_insensitive);
    if (!type.has_value()) {
        throw std::invalid_argument(fmt::format("parser error, unknown function_generator::{} - '{}'", meta::type_name<T>(), name));
    }
    return type.value();
}

template<typename T>
[[nodiscard]] property_map
createConstPropertyMap(T startValue) {
    return { createPropertyMapEntry(signal_type, Const), createPropertyMapEntry(start_value, startValue) };
}

template<typename T>
[[nodiscard]] property_map
createLinearRampPropertyMap(T startValue, T finalValue, T durationValue) {
    return { createPropertyMapEntry(signal_type, LinearRamp), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue),
             createPropertyMapEntry(duration, durationValue) };
}

template<typename T>
[[nodiscard]] property_map
createParabolicRampPropertyMap(T startValue, T finalValue, T durationValue, T roundOffTime) {
    return { createPropertyMapEntry(signal_type, ParabolicRamp), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue),
             createPropertyMapEntry(duration, durationValue), createPropertyMapEntry(round_off_time, roundOffTime) };
}

template<typename T>
[[nodiscard]] property_map
createCubicSplinePropertyMap(T startValue, T finalValue, T durationValue) {
    return { createPropertyMapEntry(signal_type, CubicSpline), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue),
             createPropertyMapEntry(duration, durationValue) };
}

template<typename T>
[[nodiscard]] property_map
createImpulseResponsePropertyMap(T startValue, T finalValue, T time0, T time1) {
    return { createPropertyMapEntry(signal_type, ImpulseResponse), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue),
             createPropertyMapEntry(impulse_time0, time0), createPropertyMapEntry(impulse_time1, time1) };
}

} // namespace function_generator

using FunctionGeneratorDoc = Doc<R""(
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

To create a chain of functions one can use `ClockSource`. There are 2 ways to do it:
1) Using times + property_map
Firstly, add time Tag entries, with time and command string:
@code
auto &clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ gr::tag::SAMPLE_RATE(100.f), { "n_samples_max", 100 }, { "name", "ClockSource" } });
clockSrc.addTimeTagEntry(1'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=1");
clockSrc.addTimeTagEntry(10'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=2");
clockSrc.addTimeTagEntry(30'000, "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=3");
@endcode

Secondly, create table to match command and functions:
@code
funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=1" } }, createConstPropertyMap(5.f));
funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=2" } }, createLinearRampPropertyMap(5.f, 30.f, .2f));
funcGen.addFunctionTableEntry({ { "context", "CMD_BP_START:FAIR.SELECTOR.C=1:S=1:P=3" } }, createConstPropertyMap(30.f));
@endcode

Lastly create a matching predicate with the following signature:
@code
std::function<std::optional<bool>(const property_map &, const property_map &, std::size_t)>;
@endcode


2) Using user defined ready-to-use Tags.
@code
auto &clockSrc = testGraph.emplaceBlock<gr::basic::ClockSource<float>>({ gr::tag::SAMPLE_RATE(100.f), { "n_samples_max", 100 }, { "name", "ClockSource" } });
clockSrc.tags = { Tag(0, createConstPropertyMap(5.f)),
                  Tag(100, createLinearRampPropertyMap(5.f, 30.f, .2f)),
                  Tag(300, createConstPropertyMap(30.f)),
                  Tag(350, createParabolicRampPropertyMap(30.f, 20.f, .1f, 0.02f)),
                  Tag(550, createConstPropertyMap(20.f)),
                  Tag(650, createCubicSplinePropertyMap(20.f, 10.f, .1f)),
                  Tag(800, createConstPropertyMap(10.f)),
                  Tag(850, createImpulseResponsePropertyMap(10.f, 20.f, .02f, .06f)) };
@endcode

)"">;

template<typename T>
    requires(std::floating_point<T>)
struct FunctionGenerator : public gr::Block<FunctionGenerator<T>, BlockingIO<true>, FunctionGeneratorDoc> {
    /**
     * TODO: Taken from CtxSettings
     * A predicate for matching two contexts
     * The third "attempt" parameter indicates the current round of matching being done.
     * This is useful for hierarchical matching schemes,
     * e.g. in the first round the predicate could look for almost exact matches only,
     * then in a a second round (attempt=1) it could be more forgiving, given that there are no exact matches available.
     *
     * The predicate will be called until it returns "true" (a match is found), or until it returns std::nullopt,
     * which indicates that no matches were found and there is no chance of matching anything in a further round.
     */
    using MatchPredicate = std::function<std::optional<bool>(const property_map &, const property_map &, std::size_t)>;

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
    // TODO; need to use 2 std::vectors, can not use property_map as std::map key.
    std::vector<property_map> function_keys{};   // matches the trigger tag map to the prestored settings
    std::vector<property_map> function_values{}; // matches the trigger tag map to the prestored settings

    MatchPredicate match_pred = [](auto, auto, auto) { return std::nullopt; };

    T   _currentTime  = T(0.);
    int sampleCounter = 0;

private:
    function_generator::SignalType _signalType = function_generator::parse<function_generator::SignalType>(signal_type);
    T                              _timeTick   = T(1.) / static_cast<T>(sample_rate);

public:
    void
    settingsChanged(const property_map & /*old_settings*/, const property_map &new_settings) {
        if (new_settings.contains(gr::tag::TRIGGER_META_INFO.shortKey())) {
            const auto funcSettings = bestMatch(trigger_meta_info);
            if (funcSettings.has_value()) {
                applyFunctionSettings(funcSettings.value());
                _currentTime = T(0.);
                _signalType  = function_generator::parse<function_generator::SignalType>(signal_type);
            }
        }
        if (new_settings.contains(function_generator::toString(function_generator::signal_type))) {
            _currentTime = T(0.);
            _signalType  = function_generator::parse<function_generator::SignalType>(signal_type);
        }
        _timeTick = T(1.) / static_cast<T>(sample_rate);
    }

    [[nodiscard]] constexpr T
    processOne(T /*input*/) noexcept {
        sampleCounter++;
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

    constexpr void
    addFunctionTableEntry(const property_map &key, const property_map &value) {
        function_keys.push_back(key);
        function_values.push_back(value);
    }

private:
    void
    applyFunctionSettings(const property_map &properties) {
        auto getProperty = [&properties]<typename U>(const property_map &m, function_generator::ParameterType paramType, U &&) -> std::optional<U> {
            const auto it = m.find(function_generator::toString(paramType));
            if (it == m.end()) {
                return std::nullopt;
            }
            try {
                return std::get<U>(it->second);
            } catch (const std::bad_variant_access &) {
                return std::nullopt;
            }
        };

        signal_type    = getProperty(properties, function_generator::signal_type, std::string("")).value_or(function_generator::toString(function_generator::Const));
        start_value    = getProperty(properties, function_generator::start_value, T{}).value_or(T(0.));
        final_value    = getProperty(properties, function_generator::final_value, T{}).value_or(T(0.));
        duration       = getProperty(properties, function_generator::duration, T{}).value_or(T(0.));
        round_off_time = getProperty(properties, function_generator::round_off_time, T{}).value_or(T(0.));
        impulse_time0  = getProperty(properties, function_generator::impulse_time0, T{}).value_or(T(0.));
        impulse_time1  = getProperty(properties, function_generator::impulse_time1, T{}).value_or(T(0.));
    }

    [[nodiscard]] constexpr T
    calculateParabolicRamp() {
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

    std::optional<property_map>
    bestMatch(const property_map &context) const {
        if (function_keys.size() != function_values.size()) {
            throw std::invalid_argument(fmt::format("function_keys size ({}) and function_values size ({}) must be equal", function_keys.size(), function_values.size()));
        }
        if (function_keys.empty() || function_values.empty()) {
            return std::nullopt;
        }
        // retry until we either get a match or std::nullopt
        for (std::size_t attempt = 0;; attempt++) {
            for (std::size_t i = 0; i < function_values.size(); i++) {
                const auto isMatched = match_pred(function_keys[i], context, attempt);
                if (!isMatched) {
                    return std::nullopt;
                } else if (*isMatched) {
                    return function_values[i];
                }
            }
        }
    }
};

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE(gr::basic::FunctionGenerator, in, out, sample_rate, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, trigger_meta_info);
GR_REGISTER_BLOCK(gr::globalBlockRegistry(), gr::basic::FunctionGenerator, float, double);

#endif // GNURADIO_FUNCTION_GENERATOR_HPP
