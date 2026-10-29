#ifndef GNURADIO_ANALOG_POWERSQUELCH_HPP
#define GNURADIO_ANALOG_POWERSQUELCH_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::PowerSquelch, [T], [ float, double ])

template<std::floating_point T>
struct PowerSquelch : Block<PowerSquelch<T>> {
    using Description = Doc<R""(@brief Power squelch: passes complex baseband while its mean power is above threshold, and outputs 0 otherwise.

The power is averaged over time_constant; threshold and hysteresis are in dB, where 0 dB is a power of 1.
The squelch opens at threshold and closes at threshold - hysteresis, so it does not flicker near the threshold.
Before FMDemodulator, it mutes the noise between stations.)"">;

    PortIn<std::complex<T>>  in;
    PortOut<std::complex<T>> out;

    gr::SampleRate                                                                                                     sample_rate   = 240'000.f;
    Annotated<float, "threshold", Unit<"dB">, Visible, Doc<"mean power that opens the squelch, 0 dB is a power of 1">> threshold     = -50.f;
    Annotated<float, "hysteresis", Unit<"dB">, Visible, Doc<"drop below threshold that closes the squelch">>           hysteresis    = 3.f;
    Annotated<float, "time constant", Unit<"s">, Doc<"averaging time of the mean power">>                              time_constant = 0.01f;

    GR_MAKE_REFLECTABLE(PowerSquelch, in, out, sample_rate, threshold, hysteresis, time_constant);

    double _smoothingFactor{};
    double _openingPower{};
    double _closingPower{};
    double _meanPower{};
    bool   _open{};

    void start() noexcept {
        updateFromSettings();
        reset();
    }

    void reset() noexcept {
        _meanPower = 0.;
        _open      = false;
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        double meanPower = _meanPower;
        bool   open      = _open;
        std::ranges::transform(input, output.begin(), [this, &meanPower, &open](std::complex<T> sample) {
            const double power = static_cast<double>(std::norm(sample));
            if (std::isfinite(power)) [[likely]] {
                meanPower += _smoothingFactor * (power - meanPower);
            }
            open = meanPower >= (open ? _closingPower : _openingPower);
            return open ? sample : std::complex<T>{};
        });
        _meanPower = meanPower;
        _open      = open;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate   = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validThreshold    = std::isfinite(threshold.value);
        const bool validHysteresis   = std::isfinite(hysteresis.value) && hysteresis.value >= 0.f;
        const bool validTimeConstant = std::isfinite(time_constant.value) && time_constant.value > 0.f;
        if (!validSampleRate || !validThreshold || !validHysteresis || !validTimeConstant) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz and time_constant {} s must be finite and positive, threshold {} dB finite and hysteresis {} dB finite and non-negative, keeping the previous settings", sample_rate.value, time_constant.value, threshold.value, hysteresis.value));
            sample_rate   = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            threshold     = oldSettings.value_or<float>("threshold", threshold.value);
            hysteresis    = oldSettings.value_or<float>("hysteresis", hysteresis.value);
            time_constant = oldSettings.value_or<float>("time_constant", time_constant.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateFromSettings();
    }

    void updateFromSettings() noexcept {
        const auto powerFromDecibel = [](double decibel) { return std::pow(10., decibel / 10.); };
        _smoothingFactor            = -std::expm1(-1. / (static_cast<double>(sample_rate.value) * static_cast<double>(time_constant.value)));
        _openingPower               = powerFromDecibel(static_cast<double>(threshold.value));
        _closingPower               = powerFromDecibel(static_cast<double>(threshold.value) - static_cast<double>(hysteresis.value));
    }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_POWERSQUELCH_HPP
