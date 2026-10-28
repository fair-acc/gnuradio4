#ifndef GNURADIO_BASIC_SPEECH_SHAPED_NOISE_HPP
#define GNURADIO_BASIC_SPEECH_SHAPED_NOISE_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>
#include <gnuradio-4.0/algorithm/filter/DifferenceEquation.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>
#include <gnuradio-4.0/basic/GenerateDeviceSamples.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::SpeechShapedNoise)

struct SpeechShapedNoise : Block<SpeechShapedNoise>, BlockingSync<SpeechShapedNoise> {
    using Description = Doc<R"(@brief stationary speech-shaped noise: white noise through Butterworth HP 110 Hz, LP 2.6 kHz and LP 6 kHz, output RMS `amplitude`

Approximates the long-term average speech spectrum (LTASS) of ITU-T Recommendation P.50, "Artificial voices", and of
D. Byrne et al., "An international comparison of long-term average speech spectra", J. Acoust. Soc. Am., vol. 96, no. 4, pp. 2108-2120, 1994;
section corners after T. Renz, P. Leistner and A. Liebl, J. Acoust. Soc. Am., vol. 143, pp. EL212-EL217, 2018, doi:10.1121/1.5027765.
)">;

    static constexpr std::size_t kSections               = 3UZ;
    static constexpr std::size_t kCoefficientsPerSection = 3UZ;

    PortIn<std::uint8_t, Optional>                                                                                                        clk;
    PortOut<float>                                                                                                                        out;
    gr::SampleRate                                                                                                                        sample_rate = 48000.f;
    Annotated<float, "amplitude", Doc<"stationary digital output RMS, not acoustic SPL">, Limits<0.f, std::numeric_limits<float>::max()>> amplitude   = 0.1f;
    Annotated<std::uint64_t, "seed", Doc<"PRNG seed (0 = fixed default for reproducibility)">>                                            seed        = 0ULL;
    Annotated<gr::Size_t, "chunk_size", Doc<"samples per free-running update">>                                                           chunk_size  = 100U;
    GR_MAKE_REFLECTABLE(SpeechShapedNoise, clk, out, sample_rate, amplitude, seed, chunk_size);

    struct State {
        std::array<double, kCoefficientsPerSection * kSections>         b{};
        std::array<double, kCoefficientsPerSection * kSections>         a{};
        std::array<double, (kCoefficientsPerSection - 1UZ) * kSections> history{};
        gr::rng::Xoshiro256pp                                           rng;
        double                                                          gain = 1.0;

        [[nodiscard]] GR_DEVICE_FN float generateSample() noexcept { return static_cast<float>(gain * gr::algorithm::filter::Cascade<double>::step(rng.uniformM11<double>(), b, a, kSections, history)); }
    };
    State _state;

    void start() {
        configureSpectrum();
        this->blockingSyncStart();
    }
    void stop() { this->blockingSyncStop(); }

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = this->syncSamples(input, output);
        if (count == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = _state.generateSample();
        }
        std::ignore = input.consume(this->isFreeRunning() ? 0UZ : count);
        output.publish(count);
        return work::Status::OK;
    }

    work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = this->syncSamples(input, output);
        if (count == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const auto        generate  = [] GR_DEVICE_LAMBDA(State & state) { return state.generateSample(); };
        const bool        generated = detail::generateSamples(*this, ctx, _state, std::span<float>{output.data(), count}, generate);
        const std::size_t produced  = generated ? count : 0UZ;
        std::ignore                 = input.consume(this->isFreeRunning() ? 0UZ : produced);
        output.publish(produced);
        return generated ? work::Status::OK : work::Status::ERROR;
    }

    void settingsChanged(const property_map&, const property_map& newSettings) {
        if (newSettings.contains("sample_rate") || newSettings.contains("seed")) {
            configureSpectrum();
        } else if (newSettings.contains("amplitude")) {
            _state.gain = outputGain();
        }
    }

private:
    static constexpr std::array<double, kSections> kCutoffs{110.0, 2600.0, 6000.0};

    [[gnu::noinline]] void configureSpectrum() {
        constexpr float kMinSampleRate = static_cast<float>(2.0 * kCutoffs.back());
        if (!(sample_rate > kMinSampleRate) || !std::isfinite(sample_rate.value)) {
            this->emitErrorMessage("SpeechShapedNoise::configureSpectrum", std::format("sample_rate must exceed {} Hz", kMinSampleRate));
            this->requestStop();
            return;
        }
        _state = State{.rng = gr::rng::Xoshiro256pp{seed}};
        for (std::size_t section = 0UZ; section < kCutoffs.size(); ++section) {
            const gr::filter::FilterParameters parameters{.order = 2UZ, .fLow = kCutoffs[section], .fHigh = kCutoffs[section], .fs = static_cast<double>(sample_rate.value)};
            const auto                         response = section == 0UZ ? gr::filter::Type::HIGHPASS : gr::filter::Type::LOWPASS;
            const auto                         sections = gr::filter::iir::designFilter<double, 2UZ>(response, parameters, gr::filter::iir::Design::BUTTERWORTH);
            const auto                         offset   = static_cast<std::ptrdiff_t>(kCoefficientsPerSection * section);
            std::ranges::copy(sections.front().b, _state.b.begin() + offset);
            std::ranges::copy(sections.front().a, _state.a.begin() + offset);
        }
        _state.gain = outputGain();
    }

    /// the gain that gives white U[-1, 1] input an output RMS of `amplitude` through the designed sections
    [[nodiscard]] double outputGain() const {
        constexpr double  kImpulsePeriods    = 32.0; // of the 110 Hz high-pass: the impulse response has decayed by then
        constexpr double  kMinImpulseSamples = 4096.0;
        constexpr double  kMaxImpulseSamples = 1048576.0;
        constexpr double  kUniformVariance   = 1.0 / 3.0; // of U[-1, 1]
        auto              impulseHistory     = decltype(_state.history){};
        double            energy             = 0.0;
        const std::size_t length             = static_cast<std::size_t>(std::clamp(kImpulsePeriods * static_cast<double>(sample_rate) / kCutoffs.front(), kMinImpulseSamples, kMaxImpulseSamples));
        for (std::size_t i = 0UZ; i < length; ++i) {
            const double response = gr::algorithm::filter::Cascade<double>::step(i == 0UZ ? 1.0 : 0.0, _state.b, _state.a, kSections, impulseHistory);
            energy += response * response;
        }
        return static_cast<double>(amplitude) / std::sqrt(kUniformVariance * energy);
    }
};

} // namespace gr::basic
#endif // GNURADIO_BASIC_SPEECH_SHAPED_NOISE_HPP
