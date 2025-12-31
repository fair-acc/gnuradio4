#ifndef GNURADIO_FUNCTION_GENERATOR_HPP
#define GNURADIO_FUNCTION_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

namespace gr::basic {

using namespace gr;

namespace function_generator {
enum class SignalType : int { Const, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse };
enum class ParameterType : int { signal_trigger, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1 };

using enum SignalType;
using enum ParameterType;
constexpr auto SignalTypeList = magic_enum::enum_values<SignalType>();

template<typename T>
requires std::is_same_v<T, SignalType> || std::is_same_v<T, ParameterType>
constexpr std::pmr::string toString(T type) {
    return std::pmr::string(magic_enum::enum_name(type));
}

template<typename EnumType, typename T>
requires std::is_same_v<EnumType, SignalType> || std::is_same_v<EnumType, ParameterType>
[[nodiscard]] std::pair<std::pmr::string, pmt::Value> createPropertyMapEntry(EnumType enumType, T value) {
    if constexpr (std::is_same_v<T, SignalType>) {
        return {toString(enumType), static_cast<pmt::Value>(toString(value))};
    } else {
        return {toString(enumType), static_cast<pmt::Value>(value)};
    }
}

template<typename T>
requires std::is_same_v<T, SignalType> || std::is_same_v<T, ParameterType>
constexpr T parse(std::string_view name) {
    auto type = magic_enum::enum_cast<T>(name, magic_enum::case_insensitive);
    if (!type.has_value()) {
        throw std::invalid_argument(std::format("parser error, unknown function_generator::{} - '{}'", meta::type_name<T>(), name));
    }
    return type.value();
}

template<typename T>
[[nodiscard]] property_map createConstPropertyMap(std::string_view triggerName, T startValue) {
    return property_map{createPropertyMapEntry(signal_trigger, std::string(triggerName)), //
        createPropertyMapEntry(signal_type, Const),                                       //
        createPropertyMapEntry(start_value, startValue)};
}

template<typename T>
[[nodiscard]] property_map createLinearRampPropertyMap(std::string_view triggerName, T startValue, T finalValue, T durationValue) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, LinearRamp), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(duration, durationValue)};
}

template<typename T>
[[nodiscard]] property_map createParabolicRampPropertyMap(std::string_view triggerName, T startValue, T finalValue, T durationValue, T roundOffTime) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, ParabolicRamp), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(duration, durationValue), createPropertyMapEntry(round_off_time, roundOffTime)};
}

template<typename T>
[[nodiscard]] property_map createCubicSplinePropertyMap(std::string_view triggerName, T startValue, T finalValue, T durationValue) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, CubicSpline), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(duration, durationValue)};
}

template<typename T>
[[nodiscard]] property_map createImpulseResponsePropertyMap(std::string_view triggerName, T startValue, T finalValue, T time0, T time1) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, ImpulseResponse), createPropertyMapEntry(start_value, startValue), createPropertyMapEntry(final_value, finalValue), createPropertyMapEntry(impulse_time0, time0), createPropertyMapEntry(impulse_time1, time1)};
}

} // namespace function_generator

GR_REGISTER_BLOCK(gr::basic::FunctionGenerator, [T], [ float, double ])

template<typename T>
requires(std::floating_point<T>)
struct FunctionGenerator : Block<FunctionGenerator<T>>, BlockingSync<FunctionGenerator<T>> {
    using Description = Doc<R""(@brief generates function waveforms and their combinations via tag-based sequencing.

Supported functions: Constant, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse.

Operating modes:
  clk_in connected: generates one sample per clock input sample
  clk_in disconnected: free-running mode synchronised to wall-clock time
)"">;

    PortIn<std::uint8_t, Optional> clk_in;
    PortOut<T>                     out;

    Annotated<float, "sample_rate", Visible, Doc<"stream sampling rate in [Hz]">>                sample_rate = 1000.f;
    Annotated<gr::Size_t, "chunk_size", Visible, Doc<"samples per update in free-running mode">> chunk_size  = 100;

    Annotated<std::string, "signal_trigger", Visible, Doc<"required trigger name (empty -> ignored)">>           signal_trigger;
    Annotated<function_generator::SignalType, "signal_type", Visible, Doc<"see function_generator::SignalType">> signal_type = function_generator::Const;

    Annotated<T, "start_value">                                               start_value    = T(0.);
    Annotated<T, "final_value">                                               final_value    = T(0.);
    Annotated<T, "duration", Doc<"in sec">>                                   duration       = T(0.);
    Annotated<T, "round_off_time", Doc<"specific to ParabolicRamp, in sec">>  round_off_time = T(0.);
    Annotated<T, "impulse_time0", Doc<"specific to ImpulseResponse, in sec">> impulse_time0  = T(0.);
    Annotated<T, "impulse_time1", Doc<"specific to ImpulseResponse, in sec">> impulse_time1  = T(0.);

    Annotated<std::string, "trigger name">       trigger_name;
    Annotated<std::uint64_t, "trigger time">     trigger_time;
    Annotated<float, "trigger offset">           trigger_offset;
    Annotated<std::string, "context name">       context;
    Annotated<property_map, "trigger_meta_info"> trigger_meta_info{};

    GR_MAKE_REFLECTABLE(FunctionGenerator, clk_in, out, sample_rate, chunk_size, signal_trigger, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, //
        trigger_name, trigger_time, trigger_offset, context, trigger_meta_info);

    T   _currentTime   = T(0.);
    int _sampleCounter = 0;
    T   _timeTick      = T(1.) / static_cast<T>(sample_rate);

    void start() {
        _currentTime = T(0.);
        _timeTick    = T(1.) / static_cast<T>(sample_rate);
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains(convert_string_domain(function_generator::toString(function_generator::signal_type)))) {
            if (signal_trigger.value.empty()) {
                _currentTime = T(0.);
            } else if (newSettings.contains(gr::tag::TRIGGER_NAME.shortKey())) {
                std::string newTrigger = newSettings.at(gr::tag::TRIGGER_NAME.shortKey()).value_or(std::string());
                if (newTrigger == signal_trigger.value) {
                    _currentTime = T(0.);
                } else {
                    // trigger does not match required signal_trigger -- revert to previous
                    start_value    = oldSettings.at("start_value").value_or(T{});
                    final_value    = oldSettings.at("final_value").value_or(T{});
                    duration       = oldSettings.at("duration").value_or(T{});
                    round_off_time = oldSettings.at("round_off_time").value_or(T{});
                    impulse_time0  = oldSettings.at("impulse_time0").value_or(T{});
                    impulse_time1  = oldSettings.at("impulse_time1").value_or(T{});
                }
            }
        }
        _timeTick = T(1.) / static_cast<T>(sample_rate);
    }

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto nSamples = this->syncSamples(input, output);
        if (nSamples == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0; i < nSamples; ++i) {
            output[i] = generateSample();
        }

        std::ignore = input.consume(this->isFreeRunning() ? 0UZ : nSamples);
        output.publish(nSamples);
        return work::Status::OK;
    }

    [[nodiscard]] constexpr T generateSample() noexcept {
        _sampleCounter++;
        using enum function_generator::SignalType;
        T value{};

        switch (signal_type) {
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

    [[nodiscard]] constexpr T calculateParabolicRamp() {
        const T roundOnTime  = round_off_time;
        const T linearLength = duration - (roundOnTime + round_off_time);
        const T a            = (final_value - start_value) / (T(2.) * roundOnTime * (linearLength + round_off_time));
        const T ar2          = a * std::pow(round_off_time, T(2.));
        const T slope        = (final_value - start_value - T(2.) * ar2) / linearLength;

        if (_currentTime > duration) {
            return final_value;
        }
        if (_currentTime < roundOnTime) {
            return start_value + a * std::pow(_currentTime, T(2.));
        }
        if (_currentTime < duration - round_off_time) {
            const T transitPoint1 = start_value + ar2;
            return transitPoint1 + slope * (_currentTime - round_off_time);
        }
        const T shiftedTime   = _currentTime - (duration - round_off_time);
        const T transitPoint2 = final_value - ar2;
        return transitPoint2 + slope * shiftedTime - a * std::pow(shiftedTime, T(2.));
    }
};

} // namespace gr::basic

#endif // GNURADIO_FUNCTION_GENERATOR_HPP
