#ifndef GNURADIO_ANALOG_PHASELOCKEDLOOP_HPP
#define GNURADIO_ANALOG_PHASELOCKEDLOOP_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <format>
#include <numbers>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/signal/PhaseLockedLoopCore.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::PhaseLockedLoop, [T], [ float, double ])

template<std::floating_point T>
struct PhaseLockedLoop : Block<PhaseLockedLoop<T>> {
    using Description = Doc<R""(@brief Phase-locked loop (PLL): locks onto the carrier of complex baseband and removes it.

A second-order loop adjusts its phase p[n] so that the carrier in out[n] = x[n] exp(-j p[n]) sits at 0 Hz with zero phase, as PMDemodulator needs.
It follows phase and frequency steps without remaining error, while modulation much faster than loop_bandwidth stays in out.
The loop frequency stays between min_frequency and max_frequency, and within +/- sample_rate / 2.
The carrier output exp(j p[n]) and the frequency output in Hz may stay unconnected.

 * M. Rice, "Digital Communications: A Discrete-Time Approach". Upper Saddle River, NJ: Pearson Prentice Hall, 2009, app. C.)"">;

    PortIn<std::complex<T>>            in;
    PortOut<std::complex<T>>           out;
    PortOut<std::complex<T>, Optional> carrier;
    PortOut<T, Optional>               frequency;

    gr::SampleRate                                                                                            sample_rate    = 240'000.f;
    Annotated<float, "loop bandwidth", Unit<"Hz">, Visible, Doc<"noise bandwidth of the loop">>               loop_bandwidth = 100.f;
    Annotated<float, "min frequency", Unit<"Hz">, Visible, Doc<"lowest carrier frequency the loop follows">>  min_frequency  = -1'000.f;
    Annotated<float, "max frequency", Unit<"Hz">, Visible, Doc<"highest carrier frequency the loop follows">> max_frequency  = 1'000.f;

    GR_MAKE_REFLECTABLE(PhaseLockedLoop, in, out, carrier, frequency, sample_rate, loop_bandwidth, min_frequency, max_frequency);

    gr::signal::PhaseLockedLoopCore _core;

    void start() noexcept {
        configureCore();
        reset();
    }

    void reset() noexcept { _core.reset(); }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& trackedOut, OutputViewLike auto& carrierOut, OutputViewLike auto& frequencyOut) noexcept {
        gr::signal::PhaseLockedLoopCore core       = _core;
        const double                    sampleRate = static_cast<double>(sample_rate.value);
        for (std::size_t n = 0UZ; n < input.size(); ++n) {
            const std::complex<T> regeneratedCarrier = std::polar(T{1}, static_cast<T>(core.phase));
            const std::complex<T> tracked            = input[n] * std::conj(regeneratedCarrier);
            const bool            hasPhase           = tracked != std::complex<T>{} && std::isfinite(tracked.real()) && std::isfinite(tracked.imag());
            trackedOut[n]                            = tracked;
            carrierOut[n]                            = regeneratedCarrier;
            frequencyOut[n]                          = static_cast<T>(core.phaseStep * sampleRate / (2. * std::numbers::pi));
            core.advance(hasPhase ? static_cast<double>(std::arg(tracked)) : 0.);
        }
        _core = core;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate    = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validLoopBandwidth = std::isfinite(loop_bandwidth.value) && loop_bandwidth.value > 0.f;
        const bool validRange         = std::isfinite(min_frequency.value) && std::isfinite(max_frequency.value) && min_frequency.value <= max_frequency.value;
        if (!validSampleRate || !validLoopBandwidth || !validRange) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz and loop_bandwidth {} Hz must be finite and positive and min_frequency {} Hz <= max_frequency {} Hz finite, keeping the previous settings", sample_rate.value, loop_bandwidth.value, min_frequency.value, max_frequency.value));
            sample_rate    = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            loop_bandwidth = oldSettings.value_or<float>("loop_bandwidth", loop_bandwidth.value);
            min_frequency  = oldSettings.value_or<float>("min_frequency", min_frequency.value);
            max_frequency  = oldSettings.value_or<float>("max_frequency", max_frequency.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        _core.phaseStep *= static_cast<double>(oldSettings.value_or<float>("sample_rate", sample_rate.value)) / static_cast<double>(sample_rate.value);
        configureCore();
    }

    void configureCore() noexcept {
        const double sampleRate        = static_cast<double>(sample_rate.value);
        const double phaseStepPerHertz = 2. * std::numbers::pi / sampleRate;
        _core.configure(static_cast<double>(loop_bandwidth.value) / sampleRate, static_cast<double>(min_frequency.value) * phaseStepPerHertz, static_cast<double>(max_frequency.value) * phaseStepPerHertz);
    }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_PHASELOCKEDLOOP_HPP
