#ifndef GNURADIO_FUNCTION_GENERATOR_HPP
#define GNURADIO_FUNCTION_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/basic/FunctionGeneratorCore.hpp>
#include <gnuradio-4.0/basic/GenerateDeviceSamples.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

namespace gr::basic {

using namespace gr;

namespace function_generator {
enum class ParameterType : int { signal_trigger, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, frequency, phase };

using enum ParameterType;
constexpr auto SignalTypeList = magic_enum::enum_values<SignalType>();

template<typename T>
requires std::is_same_v<T, SignalType> || std::is_same_v<T, ParameterType>
constexpr std::pmr::string toString(T type) {
    return std::pmr::string(magic_enum::enum_name(type));
}

template<typename EnumType, typename T>
requires std::is_same_v<EnumType, SignalType> || std::is_same_v<EnumType, ParameterType>
[[nodiscard]] std::pair<std::pmr::string, Value> createPropertyMapEntry(EnumType enumType, T value) {
    if constexpr (std::is_same_v<T, SignalType>) {
        return {toString(enumType), static_cast<Value>(toString(value))};
    } else {
        return {toString(enumType), static_cast<Value>(value)};
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

template<typename T>
[[nodiscard]] property_map createUniformNoisePropertyMap(std::string_view triggerName, T amplitude) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, UniformNoise), createPropertyMapEntry(start_value, amplitude)};
}

template<typename T>
[[nodiscard]] property_map createTriangularNoisePropertyMap(std::string_view triggerName, T amplitude) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, TriangularNoise), createPropertyMapEntry(start_value, amplitude)};
}

template<typename T>
[[nodiscard]] property_map createGaussianNoisePropertyMap(std::string_view triggerName, T amplitude) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, GaussianNoise), createPropertyMapEntry(start_value, amplitude)};
}

template<typename T>
[[nodiscard]] property_map createTonePropertyMap(std::string_view triggerName, SignalType toneType, T freq, T amplitude, T phase, T offset, T durationValue) {
    return {createPropertyMapEntry(signal_trigger, std::string(triggerName)), createPropertyMapEntry(signal_type, toneType), createPropertyMapEntry(frequency, freq), createPropertyMapEntry(final_value, amplitude), createPropertyMapEntry(ParameterType::phase, phase), createPropertyMapEntry(start_value, offset), createPropertyMapEntry(duration, durationValue)};
}

template<typename T>
[[nodiscard]] property_map createSinPropertyMap(std::string_view triggerName, T freq, T amplitude, T phase = T(0), T offset = T(0), T durationValue = T(0)) {
    return createTonePropertyMap(triggerName, Sin, freq, amplitude, phase, offset, durationValue);
}

template<typename T>
[[nodiscard]] property_map createCosPropertyMap(std::string_view triggerName, T freq, T amplitude, T phase = T(0), T offset = T(0), T durationValue = T(0)) {
    return createTonePropertyMap(triggerName, Cos, freq, amplitude, phase, offset, durationValue);
}

template<typename T>
[[nodiscard]] property_map createFastSinPropertyMap(std::string_view triggerName, T freq, T amplitude, T phase = T(0), T offset = T(0), T durationValue = T(0)) {
    return createTonePropertyMap(triggerName, FastSin, freq, amplitude, phase, offset, durationValue);
}

template<typename T>
[[nodiscard]] property_map createFastCosPropertyMap(std::string_view triggerName, T freq, T amplitude, T phase = T(0), T offset = T(0), T durationValue = T(0)) {
    return createTonePropertyMap(triggerName, FastCos, freq, amplitude, phase, offset, durationValue);
}

} // namespace function_generator

GR_REGISTER_BLOCK(gr::basic::FunctionGenerator, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct FunctionGenerator : Block<FunctionGenerator<T>>, BlockingSync<FunctionGenerator<T>> {
    using Description = Doc<R""(@brief generates function waveforms and their combinations via tag-based sequencing.

Supported functions: Constant, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse,
                     UniformNoise, TriangularNoise, GaussianNoise,
                     Sin, Cos, FastSin, FastCos.

Operating modes:
  clk connected: generates one sample per clock input sample
  clk disconnected: free-running mode synchronised to wall-clock time
)"">;

    PortIn<std::uint8_t, Optional> clk;
    PortOut<T>                     out;

    gr::SampleRate                                                                      sample_rate = 1000.f;
    Annotated<gr::Size_t, "chunk_size", Doc<"samples per update in free-running mode">> chunk_size  = 100;

    Annotated<std::string, "signal_trigger", Visible, Doc<"required trigger name (empty -> ignored)">>           signal_trigger;
    Annotated<function_generator::SignalType, "signal_type", Visible, Doc<"see function_generator::SignalType">> signal_type = function_generator::Const;

    Annotated<float, "start_value">                                               start_value    = 0.f;
    Annotated<float, "final_value">                                               final_value    = 0.f;
    Annotated<float, "duration", Doc<"in sec">>                                   duration       = 0.f;
    Annotated<float, "round_off_time", Doc<"specific to ParabolicRamp, in sec">>  round_off_time = 0.f;
    Annotated<float, "impulse_time0", Doc<"specific to ImpulseResponse, in sec">> impulse_time0  = 0.f;
    Annotated<float, "impulse_time1", Doc<"specific to ImpulseResponse, in sec">> impulse_time1  = 0.f;
    gr::Frequency                                                                 frequency      = 0.f;
    Annotated<float, "phase", Visible, Doc<"in rad">>                             phase          = 0.f;

    gr::TriggerName                                                                                            trigger_name;
    gr::TriggerTime                                                                                            trigger_time;
    gr::TriggerOffset                                                                                          trigger_offset;
    gr::Context                                                                                                context;
    gr::TriggerMetaInfo                                                                                        trigger_meta_info{};
    Annotated<std::uint64_t, "seed", Doc<"PRNG seed for noise types (0 = fixed default for reproducibility)">> seed = 0ULL;

    GR_MAKE_REFLECTABLE(FunctionGenerator, clk, out, sample_rate, chunk_size, signal_trigger, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, //
        frequency, phase, trigger_name, trigger_time, trigger_offset, context, trigger_meta_info, seed);

    FunctionGeneratorCore<T> _core;

    void start() {
        _core.elapsedSeconds      = 0.;
        _core.samplePeriodSeconds = 1. / static_cast<double>(sample_rate);
        configureGeneratorsFromSettings();
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains(convert_string_domain(function_generator::toString(function_generator::signal_type)))) {
            if (signal_trigger.value.empty()) {
                _core.elapsedSeconds = 0.;
            } else if (newSettings.contains(gr::tag::kTriggerName.shortKey())) {
                std::string newTrigger = newSettings.value_or<std::string>(gr::tag::kTriggerName.shortKey(), std::string());
                if (newTrigger == signal_trigger.value) {
                    _core.elapsedSeconds = 0.;
                } else {
                    // trigger does not match required signal_trigger -- revert to previous
                    if (auto oldType = oldSettings.value_or<std::string_view>("signal_type", std::string_view{}); oldType.data() != nullptr) {
                        if (auto parsed = magic_enum::enum_cast<function_generator::SignalType>(oldType); parsed.has_value()) {
                            signal_type = parsed.value();
                        }
                    }
                    start_value    = oldSettings.value_or<float>("start_value", 0.f);
                    final_value    = oldSettings.value_or<float>("final_value", 0.f);
                    duration       = oldSettings.value_or<float>("duration", 0.f);
                    round_off_time = oldSettings.value_or<float>("round_off_time", 0.f);
                    impulse_time0  = oldSettings.value_or<float>("impulse_time0", 0.f);
                    impulse_time1  = oldSettings.value_or<float>("impulse_time1", 0.f);
                    frequency      = oldSettings.value_or<double>("frequency", 0.);
                    phase          = oldSettings.value_or<float>("phase", 0.f);
                    seed           = oldSettings.value_or<std::uint64_t>("seed", std::uint64_t(0));
                }
            }
        }
        _core.samplePeriodSeconds = 1. / static_cast<double>(sample_rate);
        configureGeneratorsFromSettings();
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

    [[nodiscard]] T generateSample() noexcept { return _core.generateSample(generationParameters()); }

    work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = this->syncSamples(input, output);
        if (count == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const auto        generate  = [parameters = generationParameters()] GR_DEVICE_LAMBDA(auto& state) { return state.generateSample(parameters); };
        const bool        generated = detail::generateSamples(*this, ctx, _core, std::span<T>{output.data(), count}, generate);
        const std::size_t produced  = generated ? count : 0UZ;
        std::ignore                 = input.consume(this->isFreeRunning() ? 0UZ : produced);
        output.publish(produced);
        return generated ? work::Status::OK : work::Status::ERROR;
    }

private:
    [[nodiscard]] typename FunctionGeneratorCore<T>::Parameters generationParameters() const noexcept { return {.signalType = signal_type.value, .startValue = static_cast<double>(start_value), .finalValue = static_cast<double>(final_value), .duration = static_cast<double>(duration), .roundOffTime = static_cast<double>(round_off_time), .impulseTime0 = static_cast<double>(impulse_time0), .impulseTime1 = static_cast<double>(impulse_time1)}; }

    void configureGeneratorsFromSettings() noexcept {
        using enum function_generator::SignalType;
        if (signal_type == UniformNoise || signal_type == TriangularNoise || signal_type == GaussianNoise) {
            const auto noiseType = static_cast<gr::signal::NoiseType>(static_cast<int>(signal_type.value) - static_cast<int>(UniformNoise));
            _core.noise.configure(noiseType, static_cast<double>(start_value), 0., seed); // start_value = noise amplitude
        } else if (function_generator::isToneType(signal_type)) {
            // for tone types: final_value = amplitude, start_value = offset (reused from ramp semantics)
            _core.tone.configure(function_generator::toToneType(signal_type), static_cast<double>(frequency), static_cast<double>(sample_rate), static_cast<double>(phase), static_cast<double>(final_value), static_cast<double>(start_value));
            _core.tone.reset();
        }
    }
};

} // namespace gr::basic

#endif // GNURADIO_FUNCTION_GENERATOR_HPP
