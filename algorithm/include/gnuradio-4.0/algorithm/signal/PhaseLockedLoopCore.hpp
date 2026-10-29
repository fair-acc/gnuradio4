#ifndef GNURADIO_ALGORITHM_PHASE_LOCKED_LOOP_CORE_HPP
#define GNURADIO_ALGORITHM_PHASE_LOCKED_LOOP_CORE_HPP

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

namespace gr::signal {

/**
 * @brief Second-order phase-locked loop core: a proportional-integral loop filter and a phase accumulator that follow phase and frequency steps without remaining error.
 * The caller measures the phase error between its input and `phase` with a detector of its choice and passes it to advance().
 * The gains give a noise bandwidth of normalisedLoopBandwidth times the sample rate and a damping of 1/sqrt(2).
 * phaseStep, the loop frequency in rad per sample, stays within [minPhaseStep, maxPhaseStep] and +/- pi; reset() starts it in the middle with zero phase.
 *
 * M. Rice, "Digital Communications: A Discrete-Time Approach". Upper Saddle River, NJ: Pearson Prentice Hall, 2009, app. C.
 */
struct PhaseLockedLoopCore {
    struct LoopGains {
        double proportional;
        double integral;
    };

    static constexpr double kDamping = std::numbers::sqrt2 / 2.;

    double phase     = 0.;
    double phaseStep = 0.;

    double _proportionalGain = 0.;
    double _integralGain     = 0.;
    double _minPhaseStep     = -std::numbers::pi;
    double _maxPhaseStep     = std::numbers::pi;

    [[nodiscard]] static constexpr LoopGains loopGains(double normalisedLoopBandwidth) noexcept {
        const double halfNaturalFrequency = normalisedLoopBandwidth / (kDamping + 1. / (4. * kDamping));
        const double denominator          = 1. + 2. * kDamping * halfNaturalFrequency + halfNaturalFrequency * halfNaturalFrequency;
        return {.proportional = 4. * kDamping * halfNaturalFrequency / denominator, .integral = 4. * halfNaturalFrequency * halfNaturalFrequency / denominator};
    }

    void configure(double normalisedLoopBandwidth, double minPhaseStep, double maxPhaseStep) noexcept {
        const LoopGains gains = loopGains(normalisedLoopBandwidth);
        _proportionalGain     = gains.proportional;
        _integralGain         = gains.integral;
        _minPhaseStep         = std::clamp(minPhaseStep, -std::numbers::pi, std::numbers::pi);
        _maxPhaseStep         = std::clamp(maxPhaseStep, -std::numbers::pi, std::numbers::pi);
        phaseStep             = std::clamp(phaseStep, _minPhaseStep, _maxPhaseStep);
    }

    void reset() noexcept {
        phase     = 0.;
        phaseStep = std::midpoint(_minPhaseStep, _maxPhaseStep);
    }

    void advance(double phaseError) noexcept {
        phaseStep = std::clamp(phaseStep + _integralGain * phaseError, _minPhaseStep, _maxPhaseStep);
        phase     = std::remainder(phase + phaseStep + _proportionalGain * phaseError, 2. * std::numbers::pi);
    }
};

} // namespace gr::signal

#endif // GNURADIO_ALGORITHM_PHASE_LOCKED_LOOP_CORE_HPP
