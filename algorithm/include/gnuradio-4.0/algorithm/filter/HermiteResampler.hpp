#ifndef GNURADIO_ALGORITHM_HERMITE_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_HERMITE_RESAMPLER_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>

#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm::filter {

/**
 * Variable-rate resampling by four-point cubic Hermite interpolation (Catmull-Rom).
 *
 * The ratio need not be rational and may drift while the stream runs, so the output count per window is not
 * fixed and this cannot be declared as a window. The positions themselves are not sequential, though:
 * output `m` reads `phase + m / ratio`, so a caller that knows its own count can evaluate them in parallel.
 *
 * E. Catmull and R. Rom, "A class of local interpolating splines", in Computer Aided Geometric Design,
 * R. E. Barnhill and R. F. Riesenfeld, Eds. New York: Academic Press, 1974, pp. 317-326.
 */
template<typename T>
requires(std::floating_point<T> || gr::meta::complex_like<T>)
struct HermiteResampler {
    /// the sample may be complex; the position between samples never is
    using Real = gr::meta::fundamental_base_value_type_t<T>;

    /// value at fractional position `t` in [0,1) between y1 and y2, using y0 and y3 for the slopes
    [[nodiscard]] GR_DEVICE_FN static constexpr T interpolate(T y0, T y1, T y2, T y3, Real t) noexcept {
        const T a = Real{-0.5} * y0 + Real{1.5} * y1 - Real{1.5} * y2 + Real{0.5} * y3;
        const T b = y0 - Real{2.5} * y1 + Real{2.0} * y2 - Real{0.5} * y3;
        const T c = Real{-0.5} * y0 + Real{0.5} * y2;
        return ((a * t + b) * t + c) * t + y1;
    }

    /// four samples centred on `index`, clamped at the ends so the caller need not pad
    [[nodiscard]] GR_DEVICE_FN static constexpr T sampleAt(std::span<const T> signal, std::size_t index, Real fraction) noexcept {
        if (signal.empty()) {
            return T{};
        }
        const std::size_t last = signal.size() - 1UZ;
        const auto        at   = [&](std::size_t i) { return signal[i > last ? last : i]; };
        const T           y0   = index == 0UZ ? at(0UZ) : at(index - 1UZ);
        return interpolate(y0, at(index), at(index + 1UZ), at(index + 2UZ), fraction);
    }

    struct Progress {
        std::size_t produced = 0UZ; // output samples written
        std::size_t consumed = 0UZ; // input samples the caller may release
    };

    /**
     * Resamples at `ratio` output samples per input sample, carrying the fractional phase across calls.
     *
     * `phase` is the read position rebased onto the sample the caller is left holding: release `consumed`, carry
     * `phase`, and the next call resumes seamlessly. Not a bare fraction -- a lead-in sample is held back so the
     * interpolant keeps its left neighbour, so `phase` normally lands in [1, 2).
     */
    [[nodiscard]] static Progress resample(std::span<const T> signal, std::span<T> output, double ratio, double& phase) noexcept {
        if (signal.empty() || output.empty() || ratio <= 0.0) {
            return {};
        }
        const double step     = 1.0 / ratio; // input samples consumed per output sample
        std::size_t  produced = 0UZ;
        while (produced < output.size()) {
            const auto index = static_cast<std::size_t>(phase);
            if (index + 2UZ >= signal.size()) {
                break; // the next output would need a sample this window does not have
            }
            output[produced] = sampleAt(signal, index, static_cast<Real>(phase - static_cast<double>(index)));
            ++produced;
            phase += step;
        }
        // clamped because the position may stop a step past the span; the lead-in because `sampleAt` reads
        // `index - 1`, so releasing through the index would drop the neighbour and silently substitute the edge
        const auto reached  = std::min(static_cast<std::size_t>(phase), signal.size());
        const auto consumed = reached == 0UZ ? 0UZ : reached - 1UZ;
        phase -= static_cast<double>(consumed);
        return {.produced = produced, .consumed = consumed};
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_HERMITE_RESAMPLER_HPP
