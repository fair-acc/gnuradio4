#ifndef GNURADIO_DIGITAL_COSTASLOOP_HPP
#define GNURADIO_DIGITAL_COSTASLOOP_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <format>
#include <numbers>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/signal/PhaseLockedLoopCore.hpp>
#include <gnuradio-4.0/digital/Constellation.hpp>

namespace gr::digital {

GR_REGISTER_BLOCK(gr::digital::CostasLoop, [T], [ float, double ])

template<std::floating_point T>
struct CostasLoop : Block<CostasLoop<T>> {
    using Description = Doc<R""(@brief Costas loop: removes the remaining carrier phase and frequency offset from received symbols.

A second-order loop turns the symbols onto the points of constellation, steered by the angle between each symbol and its nearest point.
It runs at one sample per symbol after SymbolSync; a frequency offset far beyond loop_bandwidth needs a coarse correction first.
QAM16 needs unit mean power, for example from an AutomaticGainControl.
Like any carrier recovery from the symbols alone, it may lock with the constellation turned onto itself, for BPSK by half a turn; differential coding resolves that.
The frequency output, the offset in Hz, may stay unconnected.

 * J. P. Costas, "Synchronous communications", Proc. IRE, vol. 44, no. 12, pp. 1713-1718, 1956.
 * M. Rice, "Digital Communications: A Discrete-Time Approach". Upper Saddle River, NJ: Pearson Prentice Hall, 2009, ch. 7 and app. C.)"">;

    PortIn<std::complex<T>>  in;
    PortOut<std::complex<T>> out;
    PortOut<T, Optional>     frequency;

    Annotated<Constellation, "constellation", Visible, Doc<"symbol alphabet the loop locks onto">>         constellation  = Constellation::BPSK;
    gr::SampleRate                                                                                         sample_rate    = 240'000.f;
    Annotated<float, "loop bandwidth", Unit<"Hz">, Visible, Doc<"noise bandwidth of the loop">>            loop_bandwidth = 1'000.f;
    Annotated<float, "min frequency", Unit<"Hz">, Visible, Doc<"lowest carrier offset the loop follows">>  min_frequency  = -10'000.f;
    Annotated<float, "max frequency", Unit<"Hz">, Visible, Doc<"highest carrier offset the loop follows">> max_frequency  = 10'000.f;

    GR_MAKE_REFLECTABLE(CostasLoop, in, out, frequency, constellation, sample_rate, loop_bandwidth, min_frequency, max_frequency);

    gr::signal::PhaseLockedLoopCore _core;
    std::vector<std::complex<T>>    _points;

    void start() {
        updateFromSettings();
        reset();
    }

    void reset() noexcept { _core.reset(); }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& derotatedOut, OutputViewLike auto& frequencyOut) noexcept {
        gr::signal::PhaseLockedLoopCore core       = _core;
        const double                    sampleRate = static_cast<double>(sample_rate.value);
        for (std::size_t n = 0UZ; n < input.size(); ++n) {
            const std::complex<T> derotated = input[n] * std::polar(T{1}, static_cast<T>(-core.phase));
            const bool            hasPhase  = derotated != std::complex<T>{} && std::isfinite(derotated.real()) && std::isfinite(derotated.imag());
            const std::complex<T> decision  = _points[nearestLabel<T>(_points, derotated)];
            derotatedOut[n]                 = derotated;
            frequencyOut[n]                 = static_cast<T>(core.phaseStep * sampleRate / (2. * std::numbers::pi));
            core.advance(hasPhase ? static_cast<double>(std::arg(derotated * std::conj(decision))) : 0.);
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
            constellation  = gr::meta::parseEnum<Constellation>(oldSettings.value_or<std::string_view>("constellation", "")).value_or(constellation.value);
            sample_rate    = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            loop_bandwidth = oldSettings.value_or<float>("loop_bandwidth", loop_bandwidth.value);
            min_frequency  = oldSettings.value_or<float>("min_frequency", min_frequency.value);
            max_frequency  = oldSettings.value_or<float>("max_frequency", max_frequency.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        _core.phaseStep *= static_cast<double>(oldSettings.value_or<float>("sample_rate", sample_rate.value)) / static_cast<double>(sample_rate.value);
        updateFromSettings();
    }

    void updateFromSettings() {
        _points                        = constellationPoints<T>(constellation.value);
        const double sampleRate        = static_cast<double>(sample_rate.value);
        const double phaseStepPerHertz = 2. * std::numbers::pi / sampleRate;
        _core.configure(static_cast<double>(loop_bandwidth.value) / sampleRate, static_cast<double>(min_frequency.value) * phaseStepPerHertz, static_cast<double>(max_frequency.value) * phaseStepPerHertz);
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_COSTASLOOP_HPP
