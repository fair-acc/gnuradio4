#ifndef GNURADIO_ANALOG_AUTOMATICGAINCONTROL_HPP
#define GNURADIO_ANALOG_AUTOMATICGAINCONTROL_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::AutomaticGainControl, [T], [ float, double ])

template<std::floating_point T>
struct AutomaticGainControl : Block<AutomaticGainControl<T>> {
    using Description = Doc<R""(@brief Automatic gain control (AGC): scales complex baseband to the constant RMS amplitude target_amplitude.

The gain is target_amplitude / sqrt(P), at most max_gain, and the phase is not changed.
P is the larger of two mean powers, a short-term one over attack_time and a long-term one over decay_time.
So the gain drops within attack_time when the signal gets stronger; when it gets weaker, the gain rises within attack_time as far as the long-term mean allows, and from there over decay_time.
Use it before blocks that need a known level, such as CostasLoop or ConstellationDemapper for QAM16.
At one sample per symbol, an attack_time of at least 100 symbols keeps the level within about 1 %.)"">;

    PortIn<std::complex<T>>  in;
    PortOut<std::complex<T>> out;

    gr::SampleRate                                                                                                   sample_rate      = 240'000.f;
    Annotated<float, "target amplitude", Visible, Doc<"RMS amplitude of the output">>                                target_amplitude = 1.f;
    Annotated<float, "max gain", Unit<"dB">, Visible, Doc<"largest gain, reached for weak or missing signals">>      max_gain         = 60.f;
    Annotated<float, "attack time", Unit<"s">, Doc<"averaging time of the short-term power, which lowers the gain">> attack_time      = 1e-3f;
    Annotated<float, "decay time", Unit<"s">, Doc<"averaging time of the long-term power, which raises the gain">>   decay_time       = 0.1f;

    GR_MAKE_REFLECTABLE(AutomaticGainControl, in, out, sample_rate, target_amplitude, max_gain, attack_time, decay_time);

    double _attackFactor{};
    double _decayFactor{};
    double _maxGain{};
    double _shortTermPower{};
    double _longTermPower{};

    void start() noexcept {
        updateFromSettings();
        reset();
    }

    void reset() noexcept {
        _shortTermPower = 0.;
        _longTermPower  = 0.;
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        double shortTermPower = _shortTermPower;
        double longTermPower  = _longTermPower;
        std::ranges::transform(input, output.begin(), [this, &shortTermPower, &longTermPower](std::complex<T> sample) {
            const double power = static_cast<double>(std::norm(sample));
            if (std::isfinite(power)) [[likely]] {
                if (longTermPower == 0.) {
                    shortTermPower = power;
                    longTermPower  = power;
                } else {
                    shortTermPower += _attackFactor * (power - shortTermPower);
                    longTermPower += _decayFactor * (power - longTermPower);
                }
            }
            const double meanPower = std::max(shortTermPower, longTermPower);
            const double gain      = meanPower > 0. ? std::min(static_cast<double>(target_amplitude.value) / std::sqrt(meanPower), _maxGain) : _maxGain;
            return static_cast<T>(gain) * sample;
        });
        _shortTermPower = shortTermPower;
        _longTermPower  = longTermPower;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate      = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validTargetAmplitude = std::isfinite(target_amplitude.value) && target_amplitude.value > 0.f;
        const bool validMaxGain         = std::isfinite(max_gain.value);
        const bool validAttackTime      = std::isfinite(attack_time.value) && attack_time.value > 0.f;
        const bool validDecayTime       = std::isfinite(decay_time.value) && decay_time.value > 0.f;
        if (!validSampleRate || !validTargetAmplitude || !validMaxGain || !validAttackTime || !validDecayTime) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz, target_amplitude {}, attack_time {} s and decay_time {} s must be finite and positive and max_gain {} dB finite, keeping the previous settings", sample_rate.value, target_amplitude.value, attack_time.value, decay_time.value, max_gain.value));
            sample_rate      = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            target_amplitude = oldSettings.value_or<float>("target_amplitude", target_amplitude.value);
            max_gain         = oldSettings.value_or<float>("max_gain", max_gain.value);
            attack_time      = oldSettings.value_or<float>("attack_time", attack_time.value);
            decay_time       = oldSettings.value_or<float>("decay_time", decay_time.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateFromSettings();
    }

    void updateFromSettings() noexcept {
        const auto smoothingFactor = [this](float timeConstant) { return -std::expm1(-1. / (static_cast<double>(sample_rate.value) * static_cast<double>(timeConstant))); };
        _attackFactor              = smoothingFactor(attack_time.value);
        _decayFactor               = smoothingFactor(decay_time.value);
        _maxGain                   = std::pow(10., static_cast<double>(max_gain.value) / 20.);
    }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_AUTOMATICGAINCONTROL_HPP
