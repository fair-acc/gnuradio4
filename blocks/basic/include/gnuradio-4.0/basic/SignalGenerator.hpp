#ifndef GNURADIO_SIGNAL_GENERATOR_HPP
#define GNURADIO_SIGNAL_GENERATOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>

#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>
#include <gnuradio-4.0/basic/GenerateDeviceSamples.hpp>

namespace gr::basic {

using namespace gr;

namespace signal_generator {
using Type = gr::signal::SignalType; // same enum, reused from algorithm core
using enum Type;
inline static constexpr gr::meta::fixed_string kTypeNames = "[Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos, UniformNoise, TriangularNoise, GaussianNoise, LinearChirp, ExponentialChirp]";

} // namespace signal_generator

GR_REGISTER_BLOCK(gr::basic::SignalGenerator, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct SignalGenerator : Block<SignalGenerator<T>>, BlockingSync<SignalGenerator<T>> {
    using Description = Doc<R""(@brief generates signal waveforms: sine, cosine, square, saw, triangle, constant, fast sine/cosine, noise and chirps.

Operating modes:
  clk connected: generates one sample per clock input sample
  clk disconnected: free-running mode synchronised to wall-clock time

Signal types (A = amplitude, f = frequency, P = phase, O = offset):
  Sin/Cos:          A * sin/cos(2π·f·t + P) + O  (std::sin/cos, high precision)
  FastSin/FastCos:  A * sin/cos(2π·f·t + P) + O  (recursive phasor, ~10x faster)
  Constant:         A + O
  Square/Saw/Triangle: standard periodic waveforms
  UniformNoise:     uniform random in [-A, +A) + O
  TriangularNoise:  triangular random in [-A, +A) + O
  GaussianNoise:    Gaussian N(0, A²) + O
  LinearChirp:      A * sin(2π·∫f(t)dt + P) + O, f(t) from f_start to f_stop linear in t over duration
  ExponentialChirp: as LinearChirp, f(t) = f_start·(f_stop/f_start)^(t/duration)
)"">;

    PortIn<std::uint8_t, Optional> clk;
    PortOut<T>                     out;

    gr::SampleRate                                                                                             sample_rate = 1000.f;
    Annotated<gr::Size_t, "chunk_size", Doc<"samples per update in free-running mode">>                        chunk_size  = 100;
    Annotated<signal_generator::Type, "signal_type", Visible, Doc<"see signal_generator::Type">>               signal_type = signal_generator::Type::Sin;
    gr::Frequency                                                                                              frequency   = 1.f;
    Annotated<double, "f_start", Unit<"Hz">, Doc<"chirp start frequency">, Visible>                            f_start     = gr::signal::ChirpParameters{}.startHz;
    Annotated<double, "f_stop", Unit<"Hz">, Doc<"chirp stop frequency">, Visible>                              f_stop      = gr::signal::ChirpParameters{}.stopHz;
    Annotated<double, "duration", Unit<"s">, Doc<"chirp duration">>                                            duration    = gr::signal::ChirpParameters{}.durationSeconds;
    Annotated<bool, "repeat", Doc<"repeat chirps; otherwise emit offset after duration">>                      repeat      = gr::signal::ChirpParameters{}.repeat;
    Annotated<float, "amplitude", Visible>                                                                     amplitude   = 1.f;
    Annotated<float, "offset", Visible>                                                                        offset      = 0.f;
    Annotated<float, "phase", Visible, Doc<"in rad">>                                                          phase       = 0.f;
    Annotated<std::uint64_t, "seed", Doc<"PRNG seed for noise types (0 = fixed default for reproducibility)">> seed        = 0ULL;

    GR_MAKE_REFLECTABLE(SignalGenerator, clk, out, sample_rate, chunk_size, signal_type, frequency, amplitude, offset, phase, seed, f_start, f_stop, duration, repeat);

    gr::signal::SignalGeneratorCore<T> _core;

    void start() {
        _core.configure(signal_type.value, frequency, sample_rate, phase, amplitude, offset, seed);
        configureChirp();
        _core.reset();
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    void settingsChanged(const property_map&, const property_map&) {
        _core.configure(signal_type.value, frequency, sample_rate, phase, amplitude, offset, seed);
        configureChirp();
    }

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

    work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = this->syncSamples(input, output);
        if (count == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const auto        generate  = [] GR_DEVICE_LAMBDA(auto& state) { return state.generateSample(); };
        const bool        generated = detail::generateSamples(*this, ctx, _core, std::span<T>{output.data(), count}, generate);
        const std::size_t produced  = generated ? count : 0UZ;
        std::ignore                 = input.consume(this->isFreeRunning() ? 0UZ : produced);
        output.publish(produced);
        return generated ? work::Status::OK : work::Status::ERROR;
    }

    [[nodiscard]] T generateSample() noexcept { return _core.generateSample(); }

private:
    [[gnu::noinline]] void configureChirp() {
        if (signal_type != signal_generator::Type::LinearChirp && signal_type != signal_generator::Type::ExponentialChirp) {
            return;
        }
        if (!(duration > 0.0) || !(sample_rate > 0.f) || (signal_type == signal_generator::Type::ExponentialChirp && (!(f_start > 0.0) || !(f_stop > 0.0)))) {
            this->emitErrorMessage("SignalGenerator::configureChirp", "chirps require positive duration and sample rate, and exponential chirps require positive frequencies");
            this->requestStop();
            return;
        }
        _core.chirp = {.startHz = f_start, .stopHz = f_stop, .durationSeconds = duration, .repeat = repeat};
    }
};

} // namespace gr::basic

#endif // GNURADIO_SIGNAL_GENERATOR_HPP
