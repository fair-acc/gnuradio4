#ifndef GNURADIO_SAMPLE_RATE_ESTIMATOR_HPP
#define GNURADIO_SAMPLE_RATE_ESTIMATOR_HPP

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>

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
            _tPrev       = tObs;
            _hasPrev     = true;
            _initialised = true;
            return;
        }

        double dt = tObs - _tPrev;
        _tPrev    = tObs;

        if (dt <= 0.0) {
            return;
        }

        double measuredPeriod = dt / static_cast<double>(nSamples);
        _periodEst            = _lpFilter.processOne(measuredPeriod);
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

/**
 * @brief Compensates small clock drift between two domains by inserting or dropping
 * individual frames with linear interpolation at the splice boundary.
 *
 * Tracks a fractional sample accumulator. Each call adds the drift error for the
 * current chunk. When the accumulator crosses ±1.0, one frame is inserted (interpolated
 * midpoint of the two boundary frames) or dropped (with lerp blend at the splice point).
 * The accumulator is clamped to prevent burst corrections after silence gaps.
 */
template<typename T>
    requires std::is_arithmetic_v<T>
struct DriftCompensator {
    static constexpr double kMaxAccumulator = 2.0;

    double fractionalAccumulator{0.0};

    void reset() { fractionalAccumulator = 0.0; }

    std::size_t compensateSource(std::span<T> output, std::size_t nProduced, double estimatedRate, double nominalRate, std::size_t channelCount) {
        if (estimatedRate <= 0.0 || nominalRate <= 0.0 || nProduced == 0U || channelCount == 0U) {
            return nProduced;
        }

        const double ratio = estimatedRate / nominalRate;
        fractionalAccumulator += static_cast<double>(nProduced / channelCount) * (ratio - 1.0);
        fractionalAccumulator = std::clamp(fractionalAccumulator, -kMaxAccumulator, kMaxAccumulator);

        if (fractionalAccumulator >= 1.0 && nProduced + channelCount <= output.size()) {
            const std::size_t insertAt = nProduced;
            for (std::size_t ch = 0U; ch < channelCount; ++ch) {
                const T prev          = nProduced >= 2U * channelCount ? output[nProduced - 2U * channelCount + ch] : T{};
                const T curr          = nProduced >= channelCount ? output[nProduced - channelCount + ch] : T{};
                output[insertAt + ch] = linearInterpolate(prev, curr, 0.5f);
            }
            fractionalAccumulator -= 1.0;
            return nProduced + channelCount;
        }

        if (fractionalAccumulator <= -1.0 && nProduced >= 2U * channelCount) {
            for (std::size_t ch = 0U; ch < channelCount; ++ch) {
                const T kept    = output[nProduced - 2U * channelCount + ch];
                const T dropped = output[nProduced - channelCount + ch];
                output[nProduced - 2U * channelCount + ch] = linearInterpolate(kept, dropped, 0.5f);
            }
            fractionalAccumulator += 1.0;
            return nProduced - channelCount;
        }

        return nProduced;
    }

    std::size_t compensateSink(std::span<const T> input, std::span<T> adjusted, std::size_t nAvailable, double estimatedRate, double nominalRate, std::size_t channelCount) {
        if (estimatedRate <= 0.0 || nominalRate <= 0.0 || nAvailable == 0U || channelCount == 0U) {
            std::copy_n(input.begin(), static_cast<std::ptrdiff_t>(nAvailable), adjusted.begin());
            return nAvailable;
        }

        const double ratio = nominalRate / estimatedRate;
        fractionalAccumulator += static_cast<double>(nAvailable / channelCount) * (ratio - 1.0);
        fractionalAccumulator = std::clamp(fractionalAccumulator, -kMaxAccumulator, kMaxAccumulator);

        std::copy_n(input.begin(), static_cast<std::ptrdiff_t>(nAvailable), adjusted.begin());

        if (fractionalAccumulator >= 1.0 && nAvailable + channelCount <= adjusted.size()) {
            for (std::size_t ch = 0U; ch < channelCount; ++ch) {
                const T prev              = nAvailable >= 2U * channelCount ? adjusted[nAvailable - 2U * channelCount + ch] : T{};
                const T curr              = nAvailable >= channelCount ? adjusted[nAvailable - channelCount + ch] : T{};
                adjusted[nAvailable + ch] = linearInterpolate(prev, curr, 0.5f);
            }
            fractionalAccumulator -= 1.0;
            return nAvailable + channelCount;
        }

        if (fractionalAccumulator <= -1.0 && nAvailable >= 2U * channelCount) {
            for (std::size_t ch = 0U; ch < channelCount; ++ch) {
                const T kept    = adjusted[nAvailable - 2U * channelCount + ch];
                const T dropped = adjusted[nAvailable - channelCount + ch];
                adjusted[nAvailable - 2U * channelCount + ch] = linearInterpolate(kept, dropped, 0.5f);
            }
            fractionalAccumulator += 1.0;
            return nAvailable - channelCount;
        }

        return nAvailable;
    }

private:
    [[nodiscard]] static T linearInterpolate(T a, T b, float t) {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(static_cast<double>(a) + (static_cast<double>(b) - static_cast<double>(a)) * static_cast<double>(t));
        } else {
            return static_cast<T>(static_cast<float>(a) + (static_cast<float>(b) - static_cast<float>(a)) * t);
        }
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_SAMPLE_RATE_ESTIMATOR_HPP
