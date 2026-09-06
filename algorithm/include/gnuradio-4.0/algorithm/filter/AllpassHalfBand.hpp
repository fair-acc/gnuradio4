#ifndef GNURADIO_ALGORITHM_ALLPASS_HALF_BAND_HPP
#define GNURADIO_ALGORITHM_ALLPASS_HALF_BAND_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::algorithm::filter {

/**
 * @brief A half-band split into complementary low and high bands by two allpass branches.
 *
 * The one IIR structure that DOES decompose polyphase-style:
 *
 *     H(z) = 1/2 * [ A0(z^2) + z^-1 * A1(z^2) ]
 *     lowpass = A0 + z^-1 A1,   highpass = A0 - z^-1 A1
 *
 * Both branches see only every other input sample, so each evaluates at the DECIMATED rate and `z^2` becomes
 * plain `z` there — a first-order allpass per section. That is why a steep transition costs a handful of
 * coefficients rather than a long tap set, and why an IIR tree is viable where a DFT-modulated bank is not:
 * the branches are independent, whereas a DFT bank's arms would each need the next one's state.
 *
 * The low/high pair is POWER COMPLEMENTARY by construction — |H_lp|^2 + |H_hp|^2 = 1 for any allpass A0, A1 —
 * so that identity tests the structure, while the coefficients only set how sharp the transition is.
 *
 * R. Ansari and B. Liu, "Multirate signal processing", in Advanced Topics in Signal Processing, 1988.
 */
template<typename T>
requires std::floating_point<T>
struct AllpassHalfBand {
    /// allpass coefficients of a half-band whose analog prototype `FilterTool` designs.
    ///
    /// Bilinear-transforming with the cutoff at pi/2 makes the prewarp unity, so the analog poles map straight
    /// onto the unit disc. A half-band's digital poles then sit at `+/- j*sqrt(alpha)`, giving `alpha = -p^2`
    /// real, and the remaining pole at the origin is the branch's `z^-1`.
    ///
    /// The poles come from `gr::filter::iir::designAnalogFilter`, so this is the same design path `BasicFilter`
    /// uses rather than a second Butterworth implementation. Only designs whose digital poles come out purely
    /// imaginary factor into two allpass branches — Butterworth at pi/2 does; a design that does not yields a
    /// non-real `alpha`, which is reported by returning an empty set rather than a wrong filter.
    [[nodiscard]] static std::vector<T> designAllpass(std::size_t nSections, gr::filter::iir::Design design = gr::filter::iir::Design::BUTTERWORTH) {
        gr::filter::FilterParameters params;
        params.order = 2UZ * nSections + 1UZ;
        params.fLow  = 0.25; // half-band: the crossover sits at a quarter of the sample rate
        params.fs    = 1.0;

        const auto analog = gr::filter::iir::designAnalogFilter(gr::filter::Type::LOWPASS, params, design);
        // the designer returns poles already scaled to the analog cutoff in rad/s, so they are normalised back
        // onto the unit circle here; the bilinear step below then has unit prewarp, which is what puts the
        // digital poles at +/- j*sqrt(alpha)
        const double   cutoff = 2.0 * std::numbers::pi * params.fLow / params.fs;
        std::vector<T> alpha;
        alpha.reserve(nSections);
        for (const std::complex<double>& s : analog.poles) {
            if (s.imag() <= 1e-12 * cutoff) {
                continue; // one of each conjugate pair; the real pole contributes the branch delay instead
            }
            const std::complex<double> sNorm = s / cutoff;                    // back onto the unit circle before the bilinear
            const std::complex<double> z     = (1.0 + sNorm) / (1.0 - sNorm); // bilinear, unit prewarp at pi/2
            const std::complex<double> zSq   = z * z;
            if (std::abs(zSq.imag()) > 1e-9) {
                return {}; // this design does not factor into two allpass branches
            }
            alpha.push_back(static_cast<T>(-zSq.real()));
        }
        std::ranges::sort(alpha);
        return alpha;
    }

    /// how many state values a branch cascade needs
    [[nodiscard]] static constexpr std::size_t stateSize(std::size_t nSections) noexcept { return 2UZ * nSections; }

    /// one first-order allpass at the decimated rate: `y = a*x + x[-1] - a*y[-1]`
    /// `state` is `{xPrev, yPrev}` for this section and carries between samples
    /// the coefficient is always real; the sample it weights may be complex
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample allpassStep(TSample x, T alpha, TSample* state) noexcept {
        const TSample y = alpha * x + state[0] - alpha * state[1];
        state[0]        = x;
        state[1]        = y;
        return y;
    }

    /// a branch is a cascade of the sections it owns
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample branchStep(TSample x, std::span<const T> alphas, TSample* state) noexcept {
        TSample value = x;
        for (std::size_t s = 0UZ; s < alphas.size(); ++s) {
            value = allpassStep<TSample>(value, alphas[s], state + 2UZ * s);
        }
        return value;
    }

    /// one branch with its section count known at compile time: the coefficients and state become locals and
    /// the loop unrolls, so nothing in the recursion's dependency chain goes through memory. The same
    /// specialisation `Iir::filterInRegisters` makes, for the same reason and with the same expected payoff.
    template<std::size_t kSections, typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample branchStepFixed(TSample x, const T* alphas, TSample* state) noexcept {
        TSample value = x;
        for (std::size_t s = 0UZ; s < kSections; ++s) {
            value = allpassStep<TSample>(value, alphas[s], state + 2UZ * s);
        }
        return value;
    }

    /// the split with both branch lengths compile-time known
    template<std::size_t kEven, std::size_t kOdd, typename TSample = T>
    GR_DEVICE_FN static void splitFixed(std::span<const TSample> input, const T* alphasEven, const T* alphasOdd, TSample* stateEven, TSample* stateOdd, std::span<TSample> low, std::span<TSample> high) noexcept {
        const std::size_t nPairs = std::min({(input.size() - 1UZ) / 2UZ, low.size(), high.size()});
        for (std::size_t m = 0UZ; m < nPairs; ++m) {
            const TSample even = branchStepFixed<kEven, TSample>(input[2UZ * m + 1UZ], alphasEven, stateEven);
            const TSample odd  = branchStepFixed<kOdd, TSample>(input[2UZ * m], alphasOdd, stateOdd);
            low[m]             = T{0.5} * (even + odd);
            high[m]            = T{0.5} * (even - odd);
        }
    }

    /// input samples needed for `nPairs` output pairs: the odd branch reaches one sample further back
    [[nodiscard]] static constexpr std::size_t windowLength(std::size_t nPairs) noexcept { return 2UZ * nPairs + 1UZ; }

    /**
     * @brief Split `input` into its low and high halves, each at half the input rate.
     *
     * Pair `m` feeds `input[2m + 1]` to the even branch and `input[2m]` to the odd one -- the odd branch takes
     * the EARLIER sample, which is what the structure's `z^-1` means. Feeding it the later sample instead
     * inverts the relative branch phase and the response stops being a lowpass at all: measured, it then
     * oscillates, passing f=0.2 into the HIGH band and f=0.3 into the low one.
     *
     * `stateEven` and `stateOdd` carry between calls and are `stateSize(...)` long for their own branch.
     */
    template<typename TSample = T>
    GR_DEVICE_FN static void split(std::span<const TSample> input, std::span<const T> alphasEven, std::span<const T> alphasOdd, std::span<TSample> stateEven, std::span<TSample> stateOdd, std::span<TSample> low, std::span<TSample> high) noexcept {
        if (input.size() < 2UZ) {
            return;
        }
        // a designed tree uses one to four sections, and at those the whole recursion fits in registers
        const std::size_t nEven = alphasEven.size();
        const std::size_t nOdd  = alphasOdd.size();
        if (nEven == 1UZ && nOdd == 0UZ) {
            splitFixed<1UZ, 0UZ, TSample>(input, alphasEven.data(), alphasOdd.data(), stateEven.data(), stateOdd.data(), low, high);
            return;
        }
        if (nEven == 1UZ && nOdd == 1UZ) {
            splitFixed<1UZ, 1UZ, TSample>(input, alphasEven.data(), alphasOdd.data(), stateEven.data(), stateOdd.data(), low, high);
            return;
        }
        if (nEven == 2UZ && nOdd == 1UZ) {
            splitFixed<2UZ, 1UZ, TSample>(input, alphasEven.data(), alphasOdd.data(), stateEven.data(), stateOdd.data(), low, high);
            return;
        }
        if (nEven == 2UZ && nOdd == 2UZ) {
            splitFixed<2UZ, 2UZ, TSample>(input, alphasEven.data(), alphasOdd.data(), stateEven.data(), stateOdd.data(), low, high);
            return;
        }

        const std::size_t nPairs = std::min({(input.size() - 1UZ) / 2UZ, low.size(), high.size()});
        for (std::size_t m = 0UZ; m < nPairs; ++m) { // longer designs run where the coefficients lie
            const TSample even = branchStep<TSample>(input[2UZ * m + 1UZ], alphasEven, stateEven.data());
            const TSample odd  = branchStep<TSample>(input[2UZ * m], alphasOdd, stateOdd.data());
            low[m]             = T{0.5} * (even + odd);
            high[m]            = T{0.5} * (even - odd);
        }
    }

    /// how the designed coefficients divide between the two branches
    static void splitCoefficients(std::span<const T> alphas, std::vector<T>& even, std::vector<T>& odd) {
        even.clear();
        odd.clear();
        for (std::size_t i = 0UZ; i < alphas.size(); ++i) {
            (i % 2UZ == 0UZ ? even : odd).push_back(alphas[i]);
        }
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_ALLPASS_HALF_BAND_HPP
