#ifndef GNURADIO_ALGORITHM_POLYPHASE_SYNTHESIZER_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_SYNTHESIZER_HPP

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
 * Synthesis filterbank: `nChannels` critically sampled channels in, one wideband stream out.
 *
 * The mirror of `PolyphaseChannelizer`, and deliberately built from the same `PolyphaseBank`: a transform
 * across the channels produces one value per arm, each arm filters its own slow stream, and the commutator
 * interleaves the arm outputs back up to the full rate.
 *
 * Two passes, both index-independent so either may run as a flat parallel loop:
 *
 *   1. `armInputAt` for every (set, arm) pair — the transform across channels
 *   2. `outputAt` for every (set, arm) pair — each arm's filter over the history pass 1 wrote
 *
 * The arm inputs are stored flattened as `[set * nChannels + arm]`, so pass 2 walks one arm's history at
 * stride `nChannels` — the same strided dot product the analysis side uses, from the same bank.
 *
 * f. j. harris, C. Dick and M. Rice, "Digital receivers and transmitters using polyphase filter banks for
 * wireless communications", IEEE Trans. Microw. Theory Techn., vol. 51, no. 4, pp. 1395-1412, 2003.
 */
template<typename T>
requires std::floating_point<T>
struct PolyphaseSynthesizer : PolyphaseBank<T> {
    using Bank    = PolyphaseBank<T>;
    using Complex = gr::complex<T>; // gr::complex, not std::: its multiply survives a device JIT (see Complex.hpp)

    /// the same prototype the analysis side uses: each channel occupies 1/nChannels of the band
    [[nodiscard]] static std::vector<T> designPrototype(std::size_t nTaps, std::size_t nChannels, window::Type windowType = window::Type::Kaiser) { //
        return Bank::designPrototype(nTaps, T{0.5} / static_cast<T>(nChannels), windowType);
    }

    /// arm-input sets needed to produce `nOutputs` output sets
    [[nodiscard]] static constexpr std::size_t windowLength(std::size_t nOutputs, std::size_t phaseLen) noexcept { return nOutputs + phaseLen - 1UZ; }

    /// the twiddles `armInputAt` reads; NEGATIVE, mirroring `PolyphaseChannelizer`, and already carrying the
    /// 1/nChannels that makes the two a unity round trip rather than one that grows by the channel count
    [[nodiscard]] static std::vector<Complex> designTwiddles(std::size_t nChannels) {
        auto table = Bank::designTwiddles(nChannels, -1);
        for (auto& entry : table) {
            entry *= T{1} / static_cast<T>(nChannels);
        }
        return table;
    }

    /// arm `p` of set `m`, from that set's `nChannels` channel samples — pass 1
    template<typename TSample = Complex>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample armInputAt(const TSample* channelsOfSet, const TSample* twiddles, std::size_t nChannels, std::size_t p) noexcept {
        return Bank::template transformAt<TSample>(channelsOfSet, twiddles, nChannels, p);
    }

    /// the output sample arm `p` contributes to set `m` — pass 2
    /// `armInputs` is flattened `[set * nChannels + arm]`, so one arm's history is at stride `nChannels`
    template<typename TSample = Complex>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample outputAt(std::span<const TSample> armInputs, std::span<const T> phases, std::size_t phaseLen, std::size_t nChannels, std::size_t m, std::size_t p) noexcept {
        const std::size_t newest = (m + phaseLen - 1UZ) * nChannels + p;
        return Bank::template armAt<TSample>(armInputs, phases.data() + p * phaseLen, phaseLen, newest, nChannels);
    }

    /// where arm `p`'s sample lands in the output stream: the commutator runs backwards, as on the analysis side
    [[nodiscard]] GR_DEVICE_FN static constexpr std::size_t outputIndex(std::size_t m, std::size_t nChannels, std::size_t p) noexcept { return m * nChannels + (nChannels - 1UZ - p); }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_SYNTHESIZER_HPP
