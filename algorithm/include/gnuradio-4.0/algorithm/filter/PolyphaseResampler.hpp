#ifndef GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseBank.hpp>

namespace gr::algorithm::filter {

/**
 * Rational L/M sample-rate conversion by polyphase decomposition.
 *
 * A prototype low-pass running at the interpolated rate is split by `PolyphaseBank` into `interpolation` phases, so each output
 * sample costs one phase rather than the whole prototype. Output n takes phase `(n * decimation) % interpolation`
 * over the input history ending at `(n * decimation) / interpolation`.
 *
 * `sampleAt` is a pure function of the window it is given and carries no state, so a caller may evaluate any
 * number of outputs concurrently — which is what lets a block wrap it without writing a kernel.
 *
 * R. E. Crochiere and L. R. Rabiner, "Interpolation and decimation of digital signals — a tutorial review",
 * Proc. IEEE, vol. 69, no. 3, pp. 300-331, 1981.
 */
template<typename T>
requires std::floating_point<T>
struct PolyphaseResampler : PolyphaseBank<T> {
    using Bank = PolyphaseBank<T>;

    /// how many input samples a window of `nOutputs` consecutive outputs needs
    [[nodiscard]] static constexpr std::size_t windowLength(std::size_t nOutputs, std::size_t interpolation, std::size_t decimation, std::size_t phaseLen) noexcept { return (nOutputs * decimation + interpolation - 1UZ) / interpolation + phaseLen; }

    /// output `n` of the window, whose sample 0 is the oldest input any output in the window reaches back to
    /// the taps are always real; the samples they weight may be complex
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample sampleAt(std::span<const TSample> window, std::span<const T> phases, std::size_t phaseLen, std::size_t interpolation, std::size_t decimation, std::size_t n) noexcept {
        const std::size_t phase  = (n * decimation) % interpolation;
        const std::size_t newest = (n * decimation) / interpolation + phaseLen - 1UZ;
        return Bank::template armAt<TSample>(window, phases.data() + phase * phaseLen, phaseLen, newest);
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP
