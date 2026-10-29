#ifndef GNURADIO_ANALOG_AMMODULATION_HPP
#define GNURADIO_ANALOG_AMMODULATION_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::AMModulator, [T], [ float, double ])

template<std::floating_point T>
struct AMModulator : Block<AMModulator<T>> {
    using Description = Doc<R""(@brief Amplitude modulator: turns a real message into complex baseband whose envelope follows the message.

The output is y[n] = 1 + modulation_index x[n], a carrier of amplitude 1 at 0 Hz plus the message.
AMDemodulator recovers the message while modulation_index x[n] stays above -1.
Below -1 the envelope folds over and the message is distorted (overmodulation).)"">;

    PortIn<T>                in;
    PortOut<std::complex<T>> out;

    Annotated<float, "modulation index", Visible, Doc<"relative envelope change produced by an input of 1">> modulation_index = 1.f;

    GR_MAKE_REFLECTABLE(AMModulator, in, out, modulation_index);

    [[nodiscard]] constexpr std::complex<T> processOne(T sample) const noexcept { return {T{1} + static_cast<T>(modulation_index.value) * sample, T{0}}; }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const bool validModulationIndex = std::isfinite(modulation_index.value) && modulation_index.value > 0.f;
        if (!validModulationIndex) {
            this->emitErrorMessage("settingsChanged()", std::format("modulation_index {} must be finite and positive, keeping the previous setting", modulation_index.value));
            modulation_index = oldSettings.value_or<float>("modulation_index", modulation_index.value);
        }
    }
};

GR_REGISTER_BLOCK(gr::analog::AMDemodulator, [T], [ float, double ])

template<std::floating_point T>
struct AMDemodulator : Block<AMDemodulator<T>> {
    using Description = Doc<R""(@brief Amplitude demodulator: recovers the message from the envelope of complex baseband.

The output is y[n] = (|x[n]| / c - 1) / modulation_index, where the carrier level c is the mean envelope over carrier_time_constant.
Carrier phase, frequency offset and input amplitude do not matter, so no carrier recovery is needed.
Level changes slower than carrier_time_constant, such as fading, count as carrier changes and do not reach the output.
While the input is zero, for example behind a closed PowerSquelch, c keeps its value.
It undoes AMModulator with the same modulation_index.)"">;

    PortIn<std::complex<T>> in;
    PortOut<T>              out;

    gr::SampleRate                                                                                            sample_rate           = 240'000.f;
    Annotated<float, "modulation index", Visible, Doc<"relative envelope change that gives an output of 1">>  modulation_index      = 1.f;
    Annotated<float, "carrier time constant", Unit<"s">, Visible, Doc<"averaging time of the carrier level">> carrier_time_constant = 0.1f;

    GR_MAKE_REFLECTABLE(AMDemodulator, in, out, sample_rate, modulation_index, carrier_time_constant);

    double _smoothingFactor{};
    double _carrierLevel{};

    void start() noexcept {
        updateSmoothingFactor();
        reset();
    }

    void reset() noexcept { _carrierLevel = 0.; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        double carrierLevel = _carrierLevel;
        std::ranges::transform(input, output.begin(), [this, &carrierLevel](std::complex<T> sample) {
            const double envelope = static_cast<double>(std::abs(sample));
            if (!std::isfinite(envelope)) [[unlikely]] {
                return T{0};
            }
            if (carrierLevel == 0.) {
                carrierLevel = envelope;
                return T{0};
            }
            const double relativeEnvelopeChange = envelope / carrierLevel - 1.;
            if (envelope > 0.) {
                carrierLevel += _smoothingFactor * (envelope - carrierLevel);
            }
            return static_cast<T>(relativeEnvelopeChange / static_cast<double>(modulation_index.value));
        });
        _carrierLevel = carrierLevel;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate          = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validModulationIndex     = std::isfinite(modulation_index.value) && modulation_index.value > 0.f;
        const bool validCarrierTimeConstant = std::isfinite(carrier_time_constant.value) && carrier_time_constant.value > 0.f;
        if (!validSampleRate || !validModulationIndex || !validCarrierTimeConstant) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz, modulation_index {} and carrier_time_constant {} s must be finite and positive, keeping the previous settings", sample_rate.value, modulation_index.value, carrier_time_constant.value));
            sample_rate           = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            modulation_index      = oldSettings.value_or<float>("modulation_index", modulation_index.value);
            carrier_time_constant = oldSettings.value_or<float>("carrier_time_constant", carrier_time_constant.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateSmoothingFactor();
    }

    void updateSmoothingFactor() noexcept { _smoothingFactor = -std::expm1(-1. / (static_cast<double>(sample_rate.value) * static_cast<double>(carrier_time_constant.value))); }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_AMMODULATION_HPP
