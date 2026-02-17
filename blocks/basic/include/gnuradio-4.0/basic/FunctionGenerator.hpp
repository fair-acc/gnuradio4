#ifndef GNURADIO_FUNCTION_GENERATOR_HPP
#define GNURADIO_FUNCTION_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>
#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

namespace gr::basic {

using namespace gr;

namespace function_generator {
enum class SignalType : int { Const, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse, UniformNoise, TriangularNoise, GaussianNoise, Sin, Cos, FastSin, FastCos };
enum class ParameterType : int { signal_trigger, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, frequency, phase };

using enum SignalType;
using enum ParameterType;
constexpr auto SignalTypeList = magic_enum::enum_values<SignalType>();

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
  clk_in connected: generates one sample per clock input sample
  clk_in disconnected: free-running mode synchronised to wall-clock time
)"">;

    PortIn<std::uint8_t, Optional> clk_in;
    PortOut<T>                     out;

    Annotated<float, "sample_rate", Visible, Doc<"stream sampling rate in [Hz]">>                sample_rate = 1000.f;
    Annotated<gr::Size_t, "chunk_size", Visible, Doc<"samples per update in free-running mode">> chunk_size  = 100;

    Annotated<std::string, "signal_trigger", Visible, Doc<"required trigger name (empty -> ignored)">>           signal_trigger;
    Annotated<function_generator::SignalType, "signal_type", Visible, Doc<"see function_generator::SignalType">> signal_type = function_generator::Const;

    Annotated<float, "start_value">                                               start_value    = 0.f;
    Annotated<float, "final_value">                                               final_value    = 0.f;
    Annotated<float, "duration", Doc<"in sec">>                                   duration       = 0.f;
    Annotated<float, "round_off_time", Doc<"specific to ParabolicRamp, in sec">>  round_off_time = 0.f;
    Annotated<float, "impulse_time0", Doc<"specific to ImpulseResponse, in sec">> impulse_time0  = 0.f;
    Annotated<float, "impulse_time1", Doc<"specific to ImpulseResponse, in sec">> impulse_time1  = 0.f;
    Annotated<float, "frequency", Visible, Doc<"in Hz">>                          frequency      = 0.f;
    Annotated<float, "phase", Visible, Doc<"in rad">>                             phase          = 0.f;

    Annotated<std::string, "trigger name">                                                                              trigger_name;
    Annotated<std::uint64_t, "trigger time">                                                                            trigger_time;
    Annotated<float, "trigger offset">                                                                                  trigger_offset;
    Annotated<std::string, "context name">                                                                              context;
    Annotated<property_map, "trigger_meta_info">                                                                        trigger_meta_info{};
    Annotated<std::uint64_t, "seed", Visible, Doc<"PRNG seed for noise types (0 = fixed default for reproducibility)">> seed = 0ULL;

    GR_MAKE_REFLECTABLE(FunctionGenerator, clk_in, out, sample_rate, chunk_size, signal_trigger, signal_type, start_value, final_value, duration, round_off_time, impulse_time0, impulse_time1, //
        frequency, phase, trigger_name, trigger_time, trigger_offset, context, trigger_meta_info, seed);

    double _currentTime   = 0.;
    int    _sampleCounter = 0;
    double _timeTick      = 1. / static_cast<double>(sample_rate);

    gr::signal::NoiseGenerator<double> _noise;
    gr::signal::ToneGenerator<double>  _tone;

    void start() {
        _currentTime = 0.;
        _timeTick    = 1. / static_cast<double>(sample_rate);
        configureGeneratorsFromSettings();
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains(convert_string_domain(function_generator::toString(function_generator::signal_type)))) {
            if (signal_trigger.value.empty()) {
                _currentTime = 0.;
            } else if (newSettings.contains(gr::tag::TRIGGER_NAME.shortKey())) {
                std::string newTrigger = newSettings.at(gr::tag::TRIGGER_NAME.shortKey()).value_or(std::string());
                if (newTrigger == signal_trigger.value) {
                    _currentTime = 0.;
                } else {
                    // trigger does not match required signal_trigger -- revert to previous
                    if (auto oldType = oldSettings.at("signal_type").value_or(std::string_view{}); oldType.data() != nullptr) {
                        if (auto parsed = magic_enum::enum_cast<function_generator::SignalType>(oldType); parsed.has_value()) {
                            signal_type = parsed.value();
                        }
                    }
                    start_value    = oldSettings.at("start_value").value_or(0.f);
                    final_value    = oldSettings.at("final_value").value_or(0.f);
                    duration       = oldSettings.at("duration").value_or(0.f);
                    round_off_time = oldSettings.at("round_off_time").value_or(0.f);
                    impulse_time0  = oldSettings.at("impulse_time0").value_or(0.f);
                    impulse_time1  = oldSettings.at("impulse_time1").value_or(0.f);
                    frequency      = oldSettings.at("frequency").value_or(0.f);
                    phase          = oldSettings.at("phase").value_or(0.f);
                    seed           = oldSettings.at("seed").value_or(std::uint64_t(0));
                }
            }
        }
        _timeTick = 1. / static_cast<double>(sample_rate);
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

    [[nodiscard]] T generateSample() noexcept {
        _sampleCounter++;
        using enum function_generator::SignalType;

        if (signal_type == UniformNoise || signal_type == TriangularNoise || signal_type == GaussianNoise) {
            _currentTime += _timeTick;
            if constexpr (gr::signal::detail::is_complex_v<T>) {
                using F           = typename T::value_type;
                const auto sample = _noise.generateComplexSample();
                return T(static_cast<F>(sample.real()), static_cast<F>(sample.imag()));
            } else {
                return toOutputType(_noise.generateSample());
            }
        }

        if (function_generator::isToneType(signal_type)) {
            const double sv      = static_cast<double>(start_value); // tone offset (start_value = offset for tone types)
            const double dur     = static_cast<double>(duration);
            const bool   expired = dur > 0. && _currentTime > dur;
            _currentTime += _timeTick;
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
                const auto sample = _tone.generateComplexSample();
                return T(static_cast<F>(sample.real()), static_cast<F>(sample.imag()));
            } else {
                return toOutputType(_tone.generateSample());
            }
        }

        const double sv  = static_cast<double>(start_value);
        const double fv  = static_cast<double>(final_value);
        const double dur = static_cast<double>(duration);

        double value{};
        switch (signal_type) {
        case Const: value = sv; break;
        case LinearRamp: value = _currentTime > dur ? fv : sv + (fv - sv) * (_currentTime / dur); break;
        case ParabolicRamp: value = calculateParabolicRamp(); break;
        case CubicSpline: {
            const double t  = _currentTime / dur;
            const double t2 = 3. * t * t;
            const double t3 = 2. * t * t * t;
            value           = _currentTime > dur ? fv : (t3 - t2 + 1.) * sv + (-t3 + t2) * fv;
            break;
        }
        case ImpulseResponse: {
            const double it0 = static_cast<double>(impulse_time0);
            const double it1 = static_cast<double>(impulse_time1);
            value            = (_currentTime < it0 || _currentTime > it0 + it1) ? sv : fv;
            break;
        }
        default: value = 0.;
        }
        _currentTime += _timeTick;

        if constexpr (gr::signal::detail::is_complex_v<T>) {
            using F = typename T::value_type;
            return T(static_cast<F>(value), F(0));
        } else {
            return toOutputType(value);
        }
    }

private:
    [[nodiscard]] double calculateParabolicRamp() const noexcept {
        const double sv  = static_cast<double>(start_value);
        const double fv  = static_cast<double>(final_value);
        const double dur = static_cast<double>(duration);
        const double rot = static_cast<double>(round_off_time);

        const double linearLength = dur - 2. * rot;
        const double a            = (fv - sv) / (2. * rot * (linearLength + rot));
        const double ar2          = a * rot * rot;
        const double slope        = (fv - sv - 2. * ar2) / linearLength;

        if (_currentTime > dur) {
            return fv;
        }
        if (_currentTime < rot) {
            return sv + a * _currentTime * _currentTime;
        }
        if (_currentTime < dur - rot) {
            return sv + ar2 + slope * (_currentTime - rot);
        }
        const double shiftedTime = _currentTime - (dur - rot);
        return fv - ar2 + slope * shiftedTime - a * shiftedTime * shiftedTime;
    }

    void configureGeneratorsFromSettings() noexcept {
        using enum function_generator::SignalType;
        if (signal_type == UniformNoise || signal_type == TriangularNoise || signal_type == GaussianNoise) {
            const auto noiseType = static_cast<gr::signal::NoiseType>(static_cast<int>(signal_type.value) - static_cast<int>(UniformNoise));
            _noise.configure(noiseType, static_cast<double>(start_value), 0., seed); // start_value = noise amplitude
        } else if (function_generator::isToneType(signal_type)) {
            // for tone types: final_value = amplitude, start_value = offset (reused from ramp semantics)
            _tone.configure(function_generator::toToneType(signal_type), static_cast<double>(frequency), static_cast<double>(sample_rate), static_cast<double>(phase), static_cast<double>(final_value), static_cast<double>(start_value));
            _tone.reset();
        }
    }

    static constexpr T toOutputType(double value) noexcept {
        if constexpr (std::same_as<T, double>) {
            return value;
        } else if constexpr (std::floating_point<T>) {
            return static_cast<T>(value);
        } else if constexpr (std::integral<T>) {
            return gr::signal::detail::clampToInt<T>(value);
        } else {
            return T(0); // unreachable for supported types
        }
    }
};

} // namespace gr::basic

#endif // GNURADIO_FUNCTION_GENERATOR_HPP
