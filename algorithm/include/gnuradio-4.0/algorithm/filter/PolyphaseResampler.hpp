#ifndef GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/window.hpp>

namespace gr::algorithm::filter {

/**
 * @brief Rational L/M sample-rate conversion by polyphase decomposition.
 *
 * A prototype low-pass running at the interpolated rate is split into `interpolation` phases, so each output
 * sample costs one phase rather than the whole prototype. Output n takes phase `(n * decimation) % interpolation`
 * over the input history ending at `(n * decimation) / interpolation`.
 *
 * `sampleAt` is a pure function of the window it is given and carries no state, so a caller may evaluate any
 * number of outputs concurrently — which is what lets a block wrap it without writing a kernel.
 */
template<typename T>
requires std::floating_point<T>
struct PolyphaseResampler {
    /// windowed-sinc prototype at the interpolated rate; `cutoff` is normalised to that rate
    [[nodiscard]] static std::vector<T> designPrototype(std::size_t nTaps, T cutoff, window::Type windowType = window::Type::Kaiser) {
        std::vector<T> taps(nTaps);
        const auto     shape  = window::create<T>(windowType, nTaps);
        const T        centre = static_cast<T>(nTaps - 1UZ) / T{2};
        T              sum{};
        for (std::size_t k = 0UZ; k < nTaps; ++k) {
            const T x    = static_cast<T>(k) - centre;
            const T arg  = T{2} * cutoff * x;
            const T sinc = std::abs(arg) < std::numeric_limits<T>::epsilon() ? T{1} : std::sin(std::numbers::pi_v<T> * arg) / (std::numbers::pi_v<T> * arg);
            taps[k]      = T{2} * cutoff * sinc * shape[k];
            sum += taps[k];
        }
        for (T& tap : taps) { // unity gain at DC, so the conversion does not change the signal level
            tap /= sum;
        }
        return taps;
    }

    /// phase-major: phase p, tap j at `[p * phaseLength(nTaps, interpolation) + j]`, zero-padded to a common length
    [[nodiscard]] static std::vector<T> decompose(std::span<const T> prototype, std::size_t interpolation) {
        const std::size_t length = phaseLength(prototype.size(), interpolation);
        std::vector<T>    phases(interpolation * length, T{});
        for (std::size_t k = 0UZ; k < prototype.size(); ++k) {
            phases[(k % interpolation) * length + k / interpolation] = prototype[k];
        }
        // each phase must sum to one on its own: an output takes exactly one phase, so unequal phase sums
        // beat at the output rate and turn a constant into a ripple. A unit-sum prototype does not imply it.
        for (std::size_t p = 0UZ; p < interpolation; ++p) {
            const std::span<T> phase{phases.data() + p * length, length};
            const T            sum = std::accumulate(phase.begin(), phase.end(), T{});
            if (std::abs(sum) > std::numeric_limits<T>::epsilon()) {
                for (T& tap : phase) {
                    tap /= sum;
                }
            }
        }
        return phases;
    }

    [[nodiscard]] static constexpr std::size_t phaseLength(std::size_t nTaps, std::size_t interpolation) noexcept { return (nTaps + interpolation - 1UZ) / interpolation; }

    /// how many input samples a window of `nOutputs` consecutive outputs needs
    [[nodiscard]] static constexpr std::size_t windowLength(std::size_t nOutputs, std::size_t interpolation, std::size_t decimation, std::size_t phaseLen) noexcept { return (nOutputs * decimation + interpolation - 1UZ) / interpolation + phaseLen; }

    /// output `n` of the window, whose sample 0 is the oldest input any output in the window reaches back to
    [[nodiscard]] static constexpr T sampleAt(std::span<const T> window, std::span<const T> phases, std::size_t phaseLen, std::size_t interpolation, std::size_t decimation, std::size_t n) noexcept {
        const std::size_t phase  = (n * decimation) % interpolation;
        const std::size_t newest = (n * decimation) / interpolation + phaseLen - 1UZ;
        T                 acc{};
        for (std::size_t j = 0UZ; j < phaseLen; ++j) {
            if (newest >= j && newest - j < window.size()) {
                acc += phases[phase * phaseLen + j] * window[newest - j];
            }
        }
        return acc;
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP
