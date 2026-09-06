#ifndef GNURADIO_ALGORITHM_POLYPHASE_BANK_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_BANK_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/filter/DifferenceEquation.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::algorithm::filter {

/**
 * A prototype low-pass split into phases — the structure every polyphase block shares.
 *
 * A resampler, a channelizer and a synthesizer differ in what they do with the phases, not in how the phases
 * are formed: all three design one prototype and deal it out phase-major. That common part lives here so it
 * has one implementation and one set of tests.
 *
 * R. E. Crochiere and L. R. Rabiner, "Interpolation and decimation of digital signals — a tutorial review",
 * Proc. IEEE, vol. 69, no. 3, pp. 300-331, 1981.
 */
template<typename T>
requires std::floating_point<T>
struct PolyphaseBank {
    /// e^{sign*2*pi*i*j/nChannels} for j in [0, nChannels)
    /// nChannels entries cover the whole grid because the twiddle for (channel, arm) depends only on
    /// (k*p) mod nChannels. Tabulated because evaluating the angle in the inner loop instead measured about
    /// six times the rest of the transform put together.
    [[nodiscard]] static std::vector<gr::complex<T>> designTwiddles(std::size_t nChannels, int sign) {
        std::vector<gr::complex<T>> table(nChannels);
        for (std::size_t j = 0UZ; j < nChannels; ++j) {
            const T angle = static_cast<T>(sign) * T{2} * std::numbers::pi_v<T> * static_cast<T>(j) / static_cast<T>(nChannels);
            table[j]      = gr::complex<T>{std::cos(angle), std::sin(angle)};
        }
        return table;
    }

    /// the largest channel count for which the planar table is built: it costs 2*nChannels^2 reals, so 256
    /// channels is 512 kB and anything beyond that would cost more in cache than the vectors win
    static constexpr std::size_t kMaxPlanarChannels = 256UZ;

    /// the twiddles as two REAL matrices indexed [arm * nChannels + channel]
    /// planar and channel-inner because that shape is a contiguous multiply-accumulate; one channel at a time
    /// over interleaved complex is a reduction with a shuffle per element and does not vectorise. The
    /// nChannels-entry stepped table stays the right answer for a single channel and for a device.
    static void designPlanarTwiddles(std::size_t nChannels, int sign, std::pmr::vector<T>& real, std::pmr::vector<T>& imag) {
        if (nChannels == 0UZ || nChannels > kMaxPlanarChannels) {
            real.clear();
            imag.clear();
            return;
        }
        real.assign(nChannels * nChannels, T{});
        imag.assign(nChannels * nChannels, T{});
        for (std::size_t p = 0UZ; p < nChannels; ++p) {
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                const T angle           = static_cast<T>(sign) * T{2} * std::numbers::pi_v<T> * static_cast<T>((p * k) % nChannels) / static_cast<T>(nChannels);
                real[p * nChannels + k] = std::cos(angle);
                imag[p * nChannels + k] = std::sin(angle);
            }
        }
    }

    /// every channel of one set at once: `out[k] = sum_p value[p] * twiddle[p][k]`, `k` innermost so every
    /// array is contiguous and it vectorises without a reduction and without fast-math
    static void transformSetPlanar(const T* valueReal, const T* valueImag, const T* twiddleReal, const T* twiddleImag, std::size_t nChannels, T* outReal, T* outImag) noexcept {
        std::fill_n(outReal, nChannels, T{});
        std::fill_n(outImag, nChannels, T{});
        for (std::size_t p = 0UZ; p < nChannels; ++p) {
            const T  vr = valueReal[p];
            const T  vi = valueImag[p];
            const T* wr = twiddleReal + p * nChannels;
            const T* wi = twiddleImag + p * nChannels;
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                outReal[k] += vr * wr[k] - vi * wi[k];
                outImag[k] += vr * wi[k] + vi * wr[k];
            }
        }
    }

    /// seat a designed table in a caller's buffer, converting to whatever complex type its ports carry
    /// the algorithm designs in gr::complex; a block's ports are usually std::complex, and every block that
    /// holds a twiddle table needs exactly this, so it lives here rather than once per block
    template<typename TSample>
    static void seatTwiddles(std::pmr::vector<TSample>& target, const std::vector<gr::complex<T>>& table) {
        target.resize(table.size());
        std::ranges::transform(table, target.begin(), [](const gr::complex<T>& entry) { return TSample{entry.real(), entry.imag()}; });
    }

    /// windowed-sinc prototype at the un-decimated rate; `cutoff` is normalised to that rate
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

    /// phase-major: phase p, tap j at `[p * phaseLength(nTaps, nPhases) + j]`, zero-padded to a common length
    [[nodiscard]] static std::vector<T> decompose(std::span<const T> prototype, std::size_t nPhases) {
        const std::size_t length = phaseLength(prototype.size(), nPhases);
        std::vector<T>    phases(nPhases * length, T{});
        for (std::size_t k = 0UZ; k < prototype.size(); ++k) {
            phases[(k % nPhases) * length + k / nPhases] = prototype[k];
        }
        // each phase must sum to one on its own: an output takes exactly one phase, so unequal phase sums
        // beat at the output rate and turn a constant into a ripple. A unit-sum prototype does not imply it.
        for (std::size_t p = 0UZ; p < nPhases; ++p) {
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

    [[nodiscard]] static constexpr std::size_t phaseLength(std::size_t nTaps, std::size_t nPhases) noexcept { return (nTaps + nPhases - 1UZ) / nPhases; }

    /// one phase's dot product against the history ending at `newest`, walking back in steps of `stride`
    /// a resampler's arm is contiguous (stride 1); a channelizer's reads the raw stream at stride nChannels
    /// the taps are always real; the samples they weight may be complex
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample armAt(std::span<const TSample> window, const T* phaseTaps, std::size_t phaseLen, std::size_t newest, std::size_t stride = 1UZ) noexcept {
        return Fir<T>::template dotAt<TSample>(window, phaseTaps, phaseLen, newest, stride);
    }

    /// sum_p arms[p] * twiddles[(k*p) mod nChannels], stepping the index rather than taking a modulo
    /// accumulated in gr::complex whatever the caller's type is, so the multiply survives a device JIT
    template<typename TSample>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample transformAt(const TSample* values, const TSample* twiddles, std::size_t nChannels, std::size_t k) noexcept {
        gr::complex<T> acc{};
        std::size_t    index = 0UZ;
        for (std::size_t p = 0UZ; p < nChannels; ++p) {
            acc += gr::complex<T>{values[p].real(), values[p].imag()} * gr::complex<T>{twiddles[index].real(), twiddles[index].imag()};
            index += k; // k and index are both below nChannels, so one subtraction re-normalises
            if (index >= nChannels) {
                index -= nChannels;
            }
        }
        return TSample{acc.real(), acc.imag()};
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_BANK_HPP
