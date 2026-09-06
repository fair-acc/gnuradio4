#ifndef GNURADIO_ALGORITHM_POLYPHASE_ARBITRARY_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_ARBITRARY_RESAMPLER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseBank.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::algorithm::filter {

/**
 * @brief Resampling at an arbitrary, not necessarily rational, ratio.
 *
 * `PolyphaseResampler` needs the ratio as L/M, and a large L means a large bank. This instead builds a fixed
 * bank of `nPhases` arms and reads BETWEEN them: output `n` sits at input position `n / ratio`, whose
 * fractional part selects an arm, and the remainder interpolates linearly between that arm and its neighbour.
 * The bank is then a lookup table for the continuum of sub-sample delays rather than an exact factorisation.
 *
 * `nPhases` sets the residual interpolation error: doubling the arms halves the distance between them.
 *
 * f. j. harris, "Multirate Signal Processing for Communication Systems", ch. 7.
 */
template<typename T>
requires std::floating_point<T>
struct PolyphaseArbitraryResampler : PolyphaseBank<T> {
    using Bank = PolyphaseBank<T>;

    /// the prototype runs at the interpolated rate, so its cutoff follows the arm count
    [[nodiscard]] static std::vector<T> designPrototype(std::size_t nTaps, std::size_t nPhases, window::Type windowType = window::Type::Kaiser) { //
        return Bank::designPrototype(nTaps, T{0.5} / static_cast<T>(nPhases), windowType);
    }

    /// what one `resample` call got through
    struct Progress {
        std::size_t consumed = 0UZ;
        std::size_t produced = 0UZ;
    };

    /// output at a fractional input `position`, interpolating between the two arms either side of it
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample sampleAt(std::span<const TSample> input, std::span<const T> phases, std::size_t phaseLen, std::size_t nPhases, double position) noexcept {
        const std::size_t base     = static_cast<std::size_t>(position);
        const double      fraction = position - static_cast<double>(base);

        const double      armPosition = fraction * static_cast<double>(nPhases);
        const std::size_t arm         = static_cast<std::size_t>(armPosition);
        const T           alpha       = static_cast<T>(armPosition - static_cast<double>(arm));

        const std::size_t newest = base + phaseLen - 1UZ;
        const std::size_t armLo  = arm < nPhases ? arm : nPhases - 1UZ;
        // past the last arm the neighbour is arm 0 of the NEXT sample: holding armLo would flatten the top
        // 1/nPhases of every interval, and the same position reached as (base, 0.999) or (base + 1, 0.0) would differ
        const bool        wraps    = armLo + 1UZ >= nPhases;
        const std::size_t armHi    = wraps ? 0UZ : armLo + 1UZ;
        const std::size_t newestHi = wraps ? newest + 1UZ : newest;

        const TSample lo = Bank::template armAt<TSample>(input, phases.data() + armLo * phaseLen, phaseLen, newest);
        const TSample hi = Bank::template armAt<TSample>(input, phases.data() + armHi * phaseLen, phaseLen, newestHi);
        return lo * (T{1} - alpha) + hi * alpha;
    }

    /**
     * @brief Fill `output` from `input`, reporting what was consumed.
     *
     * An arbitrary ratio consumes a FRACTIONAL number of inputs per output, so the block cannot state the rate
     * as a chunk pair: it must consume what was actually used and carry the remainder in `phase`. Getting that
     * wrong shows up as ripple on a constant, because the sub-sample offset drifts a little every call.
     *
     * `phase` is the fractional input position of the next output, relative to `input[0]`, and carries between
     * calls. Stops early when the remaining input cannot fill another window.
     */
    template<typename TSample = T>
    [[nodiscard]] static Progress resample(std::span<const TSample> input, std::span<TSample> output, std::span<const T> phases, std::size_t phaseLen, std::size_t nPhases, double ratio, double& phase) noexcept {
        if (phaseLen == 0UZ || nPhases == 0UZ || ratio <= 0.0) {
            return {};
        }
        const double step     = 1.0 / ratio;
        double       position = phase;
        std::size_t  produced = 0UZ;
        while (produced < output.size()) {
            const std::size_t newest = static_cast<std::size_t>(position) + phaseLen - 1UZ;
            if (newest + 1UZ >= input.size()) {
                break; // one sample of headroom: interpolating past the last arm reaches into the next window
            }
            output[produced++] = sampleAt<TSample>(input, phases, phaseLen, nPhases, position);
            position += step;
        }
        // everything before the next output's window base is finished with, clamped because a step wider than the
        // window leaves `position` past the span
        const std::size_t consumed = std::min(static_cast<std::size_t>(position), input.size());
        phase                      = position - static_cast<double>(consumed);
        return {.consumed = consumed, .produced = produced};
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_ARBITRARY_RESAMPLER_HPP
