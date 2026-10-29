#ifndef GNURADIO_ANALOG_FMMODULATION_HPP
#define GNURADIO_ANALOG_FMMODULATION_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>
#include <numbers>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::FMModulator, [T], [ float, double ])

template<std::floating_point T>
struct FMModulator : Block<FMModulator<T>> {
    using Description = Doc<R""(@brief Frequency modulator: turns a real message into unit-amplitude complex baseband whose frequency follows the message.

An input of 1 shifts the carrier by frequency_deviation, so each sample advances the phase by 2 pi frequency_deviation x[n] / sample_rate.
Broadcast FM uses a peak deviation of 75 kHz, the default.
FMDemodulator with the same settings recovers the message.

 * ITU-R, "Transmission standards for FM sound broadcasting at VHF", Recommendation ITU-R BS.450-4, 2019.)"">;

    PortIn<T>                in;
    PortOut<std::complex<T>> out;

    gr::SampleRate                                                                                                sample_rate         = 240'000.f;
    Annotated<float, "frequency deviation", Unit<"Hz">, Visible, Doc<"carrier offset produced by an input of 1">> frequency_deviation = 75'000.f;

    GR_MAKE_REFLECTABLE(FMModulator, in, out, sample_rate, frequency_deviation);

    double _sensitivity{};
    double _phase{};

    void start() noexcept {
        updateSensitivity();
        reset();
    }

    void reset() noexcept { _phase = 0.; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        double phase = _phase;
        std::ranges::transform(input, output.begin(), [&phase, sensitivity = _sensitivity](T sample) {
            const double advanced = phase + sensitivity * static_cast<double>(sample);
            if (std::isfinite(advanced)) [[likely]] {
                phase = std::remainder(advanced, 2. * std::numbers::pi);
            }
            return std::polar(T{1}, static_cast<T>(phase));
        });
        _phase = phase;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate         = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validFrequencyDeviation = std::isfinite(frequency_deviation.value) && frequency_deviation.value > 0.f;
        if (!validSampleRate || !validFrequencyDeviation) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz and frequency_deviation {} Hz must be finite and positive, keeping the previous settings", sample_rate.value, frequency_deviation.value));
            sample_rate         = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            frequency_deviation = oldSettings.value_or<float>("frequency_deviation", frequency_deviation.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateSensitivity();
    }

    void updateSensitivity() noexcept { _sensitivity = 2. * std::numbers::pi * static_cast<double>(frequency_deviation.value) / static_cast<double>(sample_rate.value); }
};

GR_REGISTER_BLOCK(gr::analog::FMDemodulator, [T], [ float, double ])

template<std::floating_point T>
struct FMDemodulator : Block<FMDemodulator<T>> {
    using Description = Doc<R""(@brief Frequency demodulator: turns complex baseband into its instantaneous frequency, so a carrier offset of frequency_deviation gives 1.

It measures the phase step between neighbouring samples: y[n] = arg(x[n] conj(x[n-1])) sample_rate / (2 pi frequency_deviation).
The input amplitude does not matter, and offsets up to +/- sample_rate / 2 are unambiguous.
Broadcast FM uses a peak deviation of 75 kHz, the default.
Follow it with FMDeemphasis, or with FMStereoDecoder for stereo.

 * ITU-R, "Transmission standards for FM sound broadcasting at VHF", Recommendation ITU-R BS.450-4, 2019.)"">;

    PortIn<std::complex<T>> in;
    PortOut<T>              out;

    gr::SampleRate                                                                                                sample_rate         = 240'000.f;
    Annotated<float, "frequency deviation", Unit<"Hz">, Visible, Doc<"carrier offset that gives an output of 1">> frequency_deviation = 75'000.f;

    GR_MAKE_REFLECTABLE(FMDemodulator, in, out, sample_rate, frequency_deviation);

    T               _gain{};
    std::complex<T> _previous{};

    void start() noexcept {
        updateGain();
        reset();
    }

    void reset() noexcept { _previous = {}; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        std::complex<T> previous = _previous;
        std::ranges::transform(input, output.begin(), [&previous, gain = _gain](std::complex<T> sample) {
            const std::complex<T> product = sample * std::conj(previous);
            previous                      = sample;
            return product == std::complex<T>{} ? T{0} : gain * std::arg(product);
        });
        _previous = previous;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate         = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validFrequencyDeviation = std::isfinite(frequency_deviation.value) && frequency_deviation.value > 0.f;
        if (!validSampleRate || !validFrequencyDeviation) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz and frequency_deviation {} Hz must be finite and positive, keeping the previous settings", sample_rate.value, frequency_deviation.value));
            sample_rate         = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            frequency_deviation = oldSettings.value_or<float>("frequency_deviation", frequency_deviation.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateGain();
    }

    void updateGain() noexcept { _gain = static_cast<T>(static_cast<double>(sample_rate.value) / (2. * std::numbers::pi * static_cast<double>(frequency_deviation.value))); }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_FMMODULATION_HPP
