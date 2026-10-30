#ifndef GNURADIO_ANALOG_FMSTEREODECODER_HPP
#define GNURADIO_ANALOG_FMSTEREODECODER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>
#include <numbers>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/signal/PhaseLockedLoopCore.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::FMStereoDecoder, [T], [ float, double ])

template<std::floating_point T>
struct FMStereoDecoder : Block<FMStereoDecoder<T>> {
    using Description = Doc<R""(@brief FM stereo decoder: splits the multiplex signal from FMDemodulator, sampled at 106 kHz or more, into left and right audio.

A phase-locked loop locks onto the 19 kHz pilot, and twice its phase demodulates the left-minus-right signal at 38 kHz.
Left is the mono signal plus this difference, right is the mono signal minus it.
Only the pilot in phase with the loop counts: below pilot_threshold the output is mono and above twice that it is full stereo, so it stays mono until the loop has locked.
At 75 kHz deviation, a broadcast pilot is about 0.08 to 0.1.
Each output still needs a 15 kHz low-pass, which also removes the pilot, and FMDeemphasis.

 * ITU-R, "Transmission standards for FM sound broadcasting at VHF", Recommendation ITU-R BS.450-4, 2019.
 * M. Rice, "Digital Communications: A Discrete-Time Approach". Upper Saddle River, NJ: Pearson Prentice Hall, 2009, app. C.)"">;

    static constexpr double kPilotFrequency    = 19'000.;
    static constexpr double kPilotTolerance    = 10.;
    static constexpr double kLoopBandwidth     = 10.;
    static constexpr double kPilotFilterCorner = 100.;
    static constexpr double kMinSampleRate     = 2. * 53'000.;

    PortIn<T>  in;
    PortOut<T> left;
    PortOut<T> right;

    gr::SampleRate                                                                                      sample_rate     = 240'000.f;
    Annotated<float, "pilot threshold", Visible, Doc<"pilot amplitude below which the output is mono">> pilot_threshold = 0.02f;

    GR_MAKE_REFLECTABLE(FMStereoDecoder, in, left, right, sample_rate, pilot_threshold);

    gr::signal::PhaseLockedLoopCore _pilotLoop;
    double                          _pilotFilterFactor{};
    std::complex<double>            _roughPilotPhasor{};
    std::complex<double>            _pilotPhasor{};

    void start() noexcept {
        updateFromSettings();
        reset();
    }

    void reset() noexcept {
        _pilotLoop.reset();
        _roughPilotPhasor = {};
        _pilotPhasor      = {};
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& leftOut, OutputViewLike auto& rightOut) noexcept {
        gr::signal::PhaseLockedLoopCore pilotLoop        = _pilotLoop;
        std::complex<double>            roughPilotPhasor = _roughPilotPhasor;
        std::complex<double>            pilotPhasor      = _pilotPhasor;
        const double                    pilotThreshold   = static_cast<double>(pilot_threshold.value);
        for (std::size_t n = 0UZ; n < input.size(); ++n) {
            const double multiplex  = static_cast<double>(input[n]);
            const double difference = 2. * multiplex * std::sin(2. * pilotLoop.phase);
            if (std::isfinite(multiplex)) [[likely]] {
                const std::complex<double> sineReference{std::sin(pilotLoop.phase), std::cos(pilotLoop.phase)};
                roughPilotPhasor += _pilotFilterFactor * (multiplex * sineReference - roughPilotPhasor);
                pilotPhasor += _pilotFilterFactor * (roughPilotPhasor - pilotPhasor);
            }
            const double inPhasePilotAmplitude = 2. * pilotPhasor.real();
            const double blend                 = std::clamp(inPhasePilotAmplitude / pilotThreshold - 1., 0., 1.);
            pilotLoop.advance(pilotPhasor != std::complex<double>{} ? std::arg(pilotPhasor) : 0.);
            leftOut[n]  = static_cast<T>(multiplex + blend * difference);
            rightOut[n] = static_cast<T>(multiplex - blend * difference);
        }
        _pilotLoop        = pilotLoop;
        _roughPilotPhasor = roughPilotPhasor;
        _pilotPhasor      = pilotPhasor;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate     = std::isfinite(sample_rate.value) && static_cast<double>(sample_rate.value) >= kMinSampleRate;
        const bool validPilotThreshold = std::isfinite(pilot_threshold.value) && pilot_threshold.value > 0.f;
        if (!validSampleRate || !validPilotThreshold) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz must be finite and at least {} Hz and pilot_threshold {} finite and positive, keeping the previous settings", sample_rate.value, kMinSampleRate, pilot_threshold.value));
            sample_rate     = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            pilot_threshold = oldSettings.value_or<float>("pilot_threshold", pilot_threshold.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        _pilotLoop.phaseStep *= static_cast<double>(oldSettings.value_or<float>("sample_rate", sample_rate.value)) / static_cast<double>(sample_rate.value);
        updateFromSettings();
    }

    void updateFromSettings() noexcept {
        const double sampleRate        = static_cast<double>(sample_rate.value);
        const double phaseStepPerHertz = 2. * std::numbers::pi / sampleRate;
        _pilotFilterFactor             = -std::expm1(-kPilotFilterCorner * phaseStepPerHertz);
        _pilotLoop.configure(kLoopBandwidth / sampleRate, (kPilotFrequency - kPilotTolerance) * phaseStepPerHertz, (kPilotFrequency + kPilotTolerance) * phaseStepPerHertz);
    }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_FMSTEREODECODER_HPP
