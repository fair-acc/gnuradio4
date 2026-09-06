#ifndef GNURADIO_ALGORITHM_POLYPHASE_CHANNELIZER_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_CHANNELIZER_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseBank.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::algorithm::filter {

/**
 * @brief Analysis filterbank: one wideband stream in, `nChannels` critically sampled channels out.
 *
 * The prototype keeps 1/nChannels of the band and is dealt out phase-major by `PolyphaseBank`. Arm `p` then
 * filters the stream commutated at `m * nChannels - p`, and a DFT across the arms separates the channels —
 * so the filtering happens once at the OUTPUT rate rather than once per channel at the input rate. That is
 * the whole reason a channelizer is cheaper than `nChannels` independent mixer-and-filter chains.
 *
 * Two passes, both index-independent so either may run as a flat parallel loop: `armAt` per (output, arm),
 * then `channelAt` per (output, channel) over the arms the first wrote. That split keeps the arm filtering
 * O(nChannels · phaseLen) per output rather than the O(nChannels² · phaseLen) a per-channel form would cost.
 *
 * f. j. harris, C. Dick and M. Rice, "Digital receivers and transmitters using polyphase filter banks for
 * wireless communications", IEEE Trans. Microw. Theory Techn., vol. 51, no. 4, pp. 1395-1412, 2003.
 */
template<typename T>
requires std::floating_point<T>
struct PolyphaseChannelizer : PolyphaseBank<T> {
    using Bank    = PolyphaseBank<T>;
    using Complex = gr::complex<T>; // gr::complex, not std::: its multiply survives a device JIT (see Complex.hpp)

    /// each channel keeps 1/nChannels of the band, so the prototype's cutoff is half of that
    [[nodiscard]] static std::vector<T> designPrototype(std::size_t nTaps, std::size_t nChannels, window::Type windowType = window::Type::Kaiser) { //
        return Bank::designPrototype(nTaps, T{0.5} / static_cast<T>(nChannels), windowType);
    }

    /// input samples needed for `nOutputs` consecutive output sets
    [[nodiscard]] static constexpr std::size_t windowLength(std::size_t nOutputs, std::size_t nChannels, std::size_t phaseLen) noexcept { return (nOutputs + phaseLen - 1UZ) * nChannels; }

    /// output set `m` spans raw indices `[(phaseLen-1+m)*nChannels, +nChannels)`; the commutator hands arm `p`
    /// the newest of those minus `p`, so index 0 is the oldest sample of the leading history
    [[nodiscard]] GR_DEVICE_FN static constexpr std::size_t armNewest(std::size_t m, std::size_t nChannels, std::size_t phaseLen, std::size_t p) noexcept {
        const std::size_t newestOfSet = (phaseLen - 1UZ + m) * nChannels + nChannels - 1UZ;
        return newestOfSet >= p ? newestOfSet - p : 0UZ;
    }

    /// arm `p` of output set `m`, read straight off the raw stream at stride `nChannels` -- pass 1
    template<typename TSample = Complex>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample armOf(std::span<const TSample> window, std::span<const T> phases, std::size_t phaseLen, std::size_t nChannels, std::size_t m, std::size_t p) noexcept {
        return Bank::template armAt<TSample>(window, phases.data() + p * phaseLen, phaseLen, armNewest(m, nChannels, phaseLen, p), nChannels);
    }

    /// the twiddles `channelAt` reads; POSITIVE, because the commutator hands arm `p` the sample `p` steps
    /// into the past, so the arms are already reversed and a negative sign would put channel k at nChannels-k
    [[nodiscard]] static std::vector<Complex> designTwiddles(std::size_t nChannels) { return Bank::designTwiddles(nChannels, +1); }

    /// channel `k` from the arms of one output set — pass 2, an `nChannels`-point transform
    /// evaluated directly: at these channel counts a radix-2 FFT measured 1.4x at 8 and 2.7x at 16, and only
    /// becomes decisive above 32
    template<typename TSample = Complex>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample channelAt(const TSample* arms, const TSample* twiddles, std::size_t nChannels, std::size_t k) noexcept {
        return Bank::template transformAt<TSample>(arms, twiddles, nChannels, k);
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_CHANNELIZER_HPP
