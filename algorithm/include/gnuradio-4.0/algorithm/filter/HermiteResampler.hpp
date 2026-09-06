#ifndef GNURADIO_ALGORITHM_HERMITE_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_HERMITE_RESAMPLER_HPP

#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>

namespace gr::algorithm::filter {

/**
 * @brief Variable-rate resampling by four-point cubic Hermite interpolation.
 *
 * Unlike a rational L/M resampler the ratio need not be a fraction and may drift while the stream runs, which
 * is what clock-drift compensation between two independent domains needs. The price is that the output count
 * per window is not fixed, so this cannot be declared as a window and cannot be run one window per work item;
 * it stays sequential, and belongs on a device only for residency.
 *
 * The kernel matches the one `DriftCompensator` has used for its splice interpolation, extracted so a block can
 * resample continuously rather than only at insert/drop boundaries. Follows the adaptive-resampling approach of
 * Fons Adriaensen, "Using a DLL to filter time" (LAC 2005) and "Controlling adaptive resampling" (LAC 2012).
 */
template<typename T>
requires std::floating_point<T>
struct HermiteResampler {
    /// value at fractional position `t` in [0,1) between y1 and y2, using y0 and y3 for the slopes
    [[nodiscard]] static constexpr double interpolate(double y0, double y1, double y2, double y3, double t) noexcept {
        const double a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
        const double b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        const double c = -0.5 * y0 + 0.5 * y2;
        return ((a * t + b) * t + c) * t + y1;
    }

    /// four samples centred on `index`, clamped at the ends so the caller need not pad
    [[nodiscard]] static constexpr double sampleAt(std::span<const T> signal, std::size_t index, double fraction) noexcept {
        if (signal.empty()) {
            return 0.0;
        }
        const std::size_t last = signal.size() - 1UZ;
        const auto        at   = [&](std::size_t i) { return static_cast<double>(signal[i > last ? last : i]); };
        const double      y0   = index == 0UZ ? at(0UZ) : at(index - 1UZ);
        return interpolate(y0, at(index), at(index + 1UZ), at(index + 2UZ), fraction);
    }

    struct Progress {
        std::size_t produced = 0UZ; // output samples written
        std::size_t consumed = 0UZ; // input samples the caller may release
    };

    /**
     * @brief Resamples at `ratio` output samples per input sample, carrying the fractional phase across calls.
     *
     * `phase` is the read position within `signal` and is left holding only the fraction the next call resumes
     * from, so a caller that consumes `consumed` samples continues seamlessly.
     */
    [[nodiscard]] static Progress resample(std::span<const T> signal, std::span<T> output, double ratio, double& phase) noexcept {
        if (signal.empty() || output.empty() || ratio <= 0.0) {
            return {};
        }
        const double step     = 1.0 / ratio; // input samples consumed per output sample
        std::size_t  produced = 0UZ;
        while (produced < output.size()) {
            const auto index = static_cast<std::size_t>(phase);
            if (index + 1UZ >= signal.size()) {
                break; // the next output would need a sample this window does not have
            }
            output[produced] = static_cast<T>(sampleAt(signal, index, phase - static_cast<double>(index)));
            ++produced;
            phase += step;
        }
        const auto consumed = static_cast<std::size_t>(phase);
        phase -= static_cast<double>(consumed);
        return {.produced = produced, .consumed = consumed};
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_HERMITE_RESAMPLER_HPP
