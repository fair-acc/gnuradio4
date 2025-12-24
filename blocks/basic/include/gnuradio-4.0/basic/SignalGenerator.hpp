#ifndef GNURADIO_SIGNAL_GENERATOR_HPP
#define GNURADIO_SIGNAL_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>

#include <numbers>

namespace gr::basic {

using namespace gr;

namespace signal_generator {
enum class Type : int { Const, Sin, Cos, Square, Saw, Triangle };
using enum Type;
constexpr auto                                 TypeList  = magic_enum::enum_values<Type>();
inline static constexpr gr::meta::fixed_string TypeNames = "[Const, Sin, Cos, Square, Saw, Triangle]";

} // namespace signal_generator

GR_REGISTER_BLOCK(gr::basic::SignalGenerator, [T], [ float, double ])

template<std::floating_point T>
struct SignalGenerator : Block<SignalGenerator<T>>, BlockingSync<SignalGenerator<T>> {
    using Description = Doc<R""(@brief generates various signal waveforms (sine, cosine, square, saw, triangle, constant).

Operating modes:
  clk_in connected: generates one sample per clock input sample
  clk_in disconnected: free-running mode synchronised to wall-clock time

Signal equations (A = amplitude, f = frequency, P = phase, O = offset, t = time):
  Sine:     s(t) = A * sin(2π * f * t + P) + O
  Cosine:   s(t) = A * cos(2π * f * t + P) + O
  Square:   s(t) = A + O if t < halfPeriod else -A + O
  Constant: s(t) = A + O
  Saw:      s(t) = 2 * A * (t * f - floor(t * f + 0.5)) + O
  Triangle: s(t) = A * (4 * abs(t * f - floor(t * f + 0.75) + 0.25) - 1) + O
)"">;

    PortIn<std::uint8_t, Optional> clk_in;
    PortOut<T>                     out;

    Annotated<float, "sample_rate", Visible, Doc<"sample rate">>                                 sample_rate = 1000.f;
    Annotated<gr::Size_t, "chunk_size", Visible, Doc<"samples per update in free-running mode">> chunk_size  = 100;
    Annotated<signal_generator::Type, "signal_type", Visible, Doc<"see signal_generator::Type">> signal_type = signal_generator::Type::Sin;
    Annotated<T, "frequency", Visible>                                                           frequency   = T(1.);
    Annotated<T, "amplitude", Visible>                                                           amplitude   = T(1.);
    Annotated<T, "offset", Visible>                                                              offset      = T(0.);
    Annotated<T, "phase", Visible, Doc<"in rad">>                                                phase       = T(0.);

    GR_MAKE_REFLECTABLE(SignalGenerator, clk_in, out, sample_rate, chunk_size, signal_type, frequency, amplitude, offset, phase);

    T _currentTime = T(0.);
    T _timeTick    = T(1.) / T(sample_rate);

    void start() {
        _currentTime = T(0.);
        _timeTick    = T(1.) / T(sample_rate);
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    void settingsChanged(const property_map& /*old_settings*/, const property_map& /*new_settings*/) { _timeTick = T(1.) / T(sample_rate); }

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto nSamples = this->syncSamples(input, output);
        if (nSamples == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0; i < nSamples; ++i) {
            output[i] = generateSample();
        }

        std::ignore = input.consume(this->isFreeRunning() ? 0 : nSamples);
        output.publish(nSamples);
        return work::Status::OK;
    }

    [[nodiscard]] constexpr T generateSample() noexcept {
        using enum signal_generator::Type;

        constexpr T pi2 = T(2.) * std::numbers::pi_v<T>;
        T           value{};
        T           phaseAdjustedTime = _currentTime + phase / (pi2 * frequency);

        switch (signal_type) {
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

#endif // GNURADIO_SIGNAL_GENERATOR_HPP
