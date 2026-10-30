#ifndef GNURADIO_ANALOG_PMMODULATION_HPP
#define GNURADIO_ANALOG_PMMODULATION_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::PMModulator, [T], [ float, double ])

template<std::floating_point T>
struct PMModulator : Block<PMModulator<T>> {
    using Description = Doc<R""(@brief Phase modulator: turns a real message into unit-amplitude complex baseband whose phase follows the message.

The output is y[n] = exp(j phase_deviation x[n]), so an input of 1 turns the phase by phase_deviation radians.
PMDemodulator recovers the message while |phase_deviation x[n]| < pi.)"">;

    PortIn<T>                in;
    PortOut<std::complex<T>> out;

    Annotated<float, "phase deviation", Unit<"rad">, Visible, Doc<"carrier phase shift produced by an input of 1">> phase_deviation = 1.f;

    GR_MAKE_REFLECTABLE(PMModulator, in, out, phase_deviation);

    [[nodiscard]] std::complex<T> processOne(T sample) const noexcept { return std::polar(T{1}, static_cast<T>(phase_deviation.value) * sample); }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const bool validPhaseDeviation = std::isfinite(phase_deviation.value) && phase_deviation.value > 0.f;
        if (!validPhaseDeviation) {
            this->emitErrorMessage("settingsChanged()", std::format("phase_deviation {} rad must be finite and positive, keeping the previous setting", phase_deviation.value));
            phase_deviation = oldSettings.value_or<float>("phase_deviation", phase_deviation.value);
        }
    }
};

GR_REGISTER_BLOCK(gr::analog::PMDemodulator, [T], [ float, double ])

template<std::floating_point T>
struct PMDemodulator : Block<PMDemodulator<T>> {
    using Description = Doc<R""(@brief Phase demodulator: turns complex baseband into its phase, so a phase of phase_deviation gives 1.

The output is y[n] = arg(x[n]) / phase_deviation, and the input amplitude does not matter.
It needs the carrier at 0 Hz with zero phase, as PhaseLockedLoop delivers.
A remaining frequency offset shows up as a phase ramp that wraps at +/- pi.)"">;

    PortIn<std::complex<T>> in;
    PortOut<T>              out;

    Annotated<float, "phase deviation", Unit<"rad">, Visible, Doc<"carrier phase that gives an output of 1">> phase_deviation = 1.f;

    GR_MAKE_REFLECTABLE(PMDemodulator, in, out, phase_deviation);

    [[nodiscard]] T processOne(std::complex<T> sample) const noexcept { return sample == std::complex<T>{} ? T{0} : std::atan2(sample.imag(), sample.real()) / static_cast<T>(phase_deviation.value); }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const bool validPhaseDeviation = std::isfinite(phase_deviation.value) && phase_deviation.value > 0.f;
        if (!validPhaseDeviation) {
            this->emitErrorMessage("settingsChanged()", std::format("phase_deviation {} rad must be finite and positive, keeping the previous setting", phase_deviation.value));
            phase_deviation = oldSettings.value_or<float>("phase_deviation", phase_deviation.value);
        }
    }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_PMMODULATION_HPP
