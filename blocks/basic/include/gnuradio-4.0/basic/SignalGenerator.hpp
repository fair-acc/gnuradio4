#ifndef GNURADIO_SIGNAL_GENERATOR_HPP
#define GNURADIO_SIGNAL_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>

#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>

namespace gr::basic {

using namespace gr;

namespace signal_generator {
using Type = gr::signal::SignalType; // same enum, reused from algorithm core
using enum Type;
constexpr auto                                 TypeList  = magic_enum::enum_values<Type>();
inline static constexpr gr::meta::fixed_string TypeNames = "[Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos, UniformNoise, TriangularNoise, GaussianNoise]";

} // namespace signal_generator

GR_REGISTER_BLOCK(gr::basic::SignalGenerator, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct SignalGenerator : Block<SignalGenerator<T>>, BlockingSync<SignalGenerator<T>> {
    using Description = Doc<R""(@brief generates signal waveforms: sine, cosine, square, saw, triangle, constant, fast sine/cosine, and noise.

Operating modes:
  clk_in connected: generates one sample per clock input sample
  clk_in disconnected: free-running mode synchronised to wall-clock time

Signal types (A = amplitude, f = frequency, P = phase, O = offset):
  Sin/Cos:          A * sin/cos(2π·f·t + P) + O  (std::sin/cos, high precision)
  FastSin/FastCos:  A * sin/cos(2π·f·t + P) + O  (recursive phasor, ~10x faster)
  Constant:         A + O
  Square/Saw/Triangle: standard periodic waveforms
  UniformNoise:     uniform random in [-A, +A) + O
  TriangularNoise:  triangular random in [-A, +A) + O
  GaussianNoise:    Gaussian N(0, A²) + O
)"">;

    PortIn<std::uint8_t, Optional> clk_in;
    PortOut<T>                     out;

    Annotated<float, "sample_rate", Visible, Doc<"sample rate">>                                                        sample_rate = 1000.f;
    Annotated<gr::Size_t, "chunk_size", Visible, Doc<"samples per update in free-running mode">>                        chunk_size  = 100;
    Annotated<signal_generator::Type, "signal_type", Visible, Doc<"see signal_generator::Type">>                        signal_type = signal_generator::Type::Sin;
    Annotated<float, "frequency", Visible>                                                                              frequency   = 1.f;
    Annotated<float, "amplitude", Visible>                                                                              amplitude   = 1.f;
    Annotated<float, "offset", Visible>                                                                                 offset      = 0.f;
    Annotated<float, "phase", Visible, Doc<"in rad">>                                                                   phase       = 0.f;
    Annotated<std::uint64_t, "seed", Visible, Doc<"PRNG seed for noise types (0 = fixed default for reproducibility)">> seed        = 0ULL;

    GR_MAKE_REFLECTABLE(SignalGenerator, clk_in, out, sample_rate, chunk_size, signal_type, frequency, amplitude, offset, phase, seed);

    gr::signal::SignalGeneratorCore<T> _core;

    void start() {
        _core.configure(signal_type.value, frequency, sample_rate, phase, amplitude, offset, seed);
        _core.reset();
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    void settingsChanged(const property_map& /*old_settings*/, const property_map& /*new_settings*/) { _core.configure(signal_type.value, frequency, sample_rate, phase, amplitude, offset, seed); }

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto nSamples = this->syncSamples(input, output);
        if (nSamples == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0; i < nSamples; ++i) {
            output[i] = _core.generateSample();
        }

        std::ignore = input.consume(this->isFreeRunning() ? 0 : nSamples);
        output.publish(nSamples);
        return work::Status::OK;
    }

    [[nodiscard]] T generateSample() noexcept { return _core.generateSample(); }
};

} // namespace gr::basic

#endif // GNURADIO_SIGNAL_GENERATOR_HPP
