#ifndef GNURADIO_SAMPLE_RATE_ESTIMATOR_HPP
#define GNURADIO_SAMPLE_RATE_ESTIMATOR_HPP

#include <cstddef>

#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>

namespace gr::algorithm {

/**
 * @brief Estimates the true sample rate from timestamped chunk arrivals.
 *
 * Each update() call provides a wall-clock timestamp and the number of samples
 * delivered. The instantaneous period per sample (dt / nSamples) is fed through
 * a configurable IIR low-pass filter (Butterworth, default 2nd order) to smooth
 * out transport jitter. The estimated rate is 1 / filteredPeriod.
 *
 * Usage: call update(tObsSeconds, nSamples) once per chunk delivery.
 * Query estimatedRate() or estimatedPpm() at any time.
 * resetPhase() after a retune/reconfiguration preserves the filter state.
 *
 * Inspired by:
 *   Adriaensen, "Using a DLL to filter time" (2005)
 *   https://kokkinizita.linuxaudio.org/papers/usingdll.pdf
 *   Adriaensen, "Adaptive resampling"
 *   https://kokkinizita.linuxaudio.org/papers/adapt-resamp.pdf
 */
struct SampleRateEstimator {
    float       filter_cutoff_hz = 0.1f; // LP cutoff for period smoothing
    std::size_t filter_order     = 2UZ;  // Butterworth order
    float       ppm_initial      = 0.0f;

    double                 _periodEst   = 0.0; // filtered period per sample (s/sample)
    double                 _nominalRate = 0.0;
    double                 _tPrev       = 0.0;
    bool                   _initialised = false;
    bool                   _hasPrev     = false;
    filter::Filter<double> _lpFilter;
    double                 _updateRate = 0.0; // estimated chunk delivery rate for filter design

    void reset(double nominalRate, double expectedUpdateRateHz = 250.0) {
        _nominalRate = nominalRate;
        _periodEst   = (nominalRate > 0.0) ? (1.0 / nominalRate) : 0.0;
        if (ppm_initial != 0.0f) {
            _periodEst *= (1.0 + static_cast<double>(ppm_initial) * 1e-6);
        }
        _tPrev       = 0.0;
        _initialised = false;
        _hasPrev     = false;
        _updateRate  = expectedUpdateRateHz;
        rebuildFilter();
    }

    void resetPhase() {
        _tPrev   = 0.0;
        _hasPrev = false;
    }

    void update(double tObs, std::size_t nSamples) {
        if (_nominalRate <= 0.0 || nSamples == 0) {
            return;
        }

        if (!_hasPrev) {
            _tPrev   = tObs;
            _hasPrev = true;
            return;
        }

        double dt = tObs - _tPrev;
        _tPrev    = tObs;

        if (dt <= 0.0) {
            return;
        }

        double measuredPeriod = dt / static_cast<double>(nSamples);
        _periodEst            = _lpFilter.processOne(measuredPeriod);
        _initialised          = true;
    }

    [[nodiscard]] double estimatedRate() const noexcept {
        if (!_initialised || _periodEst <= 0.0 || _nominalRate <= 0.0) {
            return _nominalRate;
        }
        return 1.0 / _periodEst;
    }

    [[nodiscard]] float estimatedPpm() const noexcept {
        if (!_initialised || _periodEst <= 0.0 || _nominalRate <= 0.0) {
            return 0.0f;
        }
        double estRate = 1.0 / _periodEst;
        return static_cast<float>((estRate - _nominalRate) / _nominalRate * 1e6);
    }

    void rebuildFilter() {
        if (_updateRate > 0.0 && filter_cutoff_hz > 0.f) {
            double cutoff = std::min(static_cast<double>(filter_cutoff_hz), _updateRate * 0.4);
            auto   coeffs = filter::iir::designFilter<double>(filter::Type::LOWPASS, filter::FilterParameters{.order = filter_order, .fLow = cutoff, .fs = _updateRate}, filter::iir::Design::BUTTERWORTH);
            _lpFilter     = filter::Filter<double>(coeffs);
            _lpFilter.reset(_periodEst);
        }
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_SAMPLE_RATE_ESTIMATOR_HPP
