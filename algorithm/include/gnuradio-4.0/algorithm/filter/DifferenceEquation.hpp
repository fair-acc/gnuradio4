#ifndef GNURADIO_ALGORITHM_DIFFERENCE_EQUATION_HPP
#define GNURADIO_ALGORITHM_DIFFERENCE_EQUATION_HPP

#include <vir/simd.h>

#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <span>

#include <gnuradio-4.0/algorithm/filter/FilterForms.hpp>
#include <gnuradio-4.0/meta/ExecutionPolicy.hpp>

/**
 * @brief The two difference equations a filter evaluates, as functions of the state they are given.
 *
 * A FIR is the non-recursive equation `y[n] = Σ b[k] x[n-k]`; an IIR is the recursive one that adds
 * `- Σ a[k] y[n-k]`. Both appear in several blocks, so they live here once rather than once per block.
 *
 * Neither owns storage. Coefficients and state arrive as spans, so the same code runs on a host and
 * inside a kernel, and each caller keeps its state in whatever form suits it -- a reflected `Tensor` for
 * a block that travels to a device, a `HistoryBuffer` for a streaming filter, plain locals for a test.
 * `PolyphaseResampler::sampleAt` strikes the same bargain for the same reason.
 *
 * A. V. Oppenheim and R. W. Schafer, "Discrete-Time Signal Processing", 3rd ed. Upper Saddle River, NJ:
 * Prentice Hall, 2009, ch. 6, for the direct forms and what each keeps between samples.
 *
 * ```cpp
 * std::array<float, 3> state{};                              // one biquad section
 * Iir<float>::filter(input, b, a, state, output);            // state carries on to the next call
 * ```
 */

/// where an IIR keeps its state. The forms compute the same response and differ in what they hold
/// between samples, which is what makes one preferable in fixed point and another in floating point.

/// `span.data() + 1` on an empty span forms a pointer past a null one, which is undefined even where the count
/// below clamps every read away. UBSan's pointer-overflow check fires on it.
namespace stdx = vir::stdx;

namespace gr::algorithm::filter {

template<typename T, std::size_t kMinSize = 0UZ>
[[nodiscard]] constexpr const T* afterFirst(std::span<const T> span) noexcept {
    if constexpr (kMinSize > 1UZ) {
        return span.data() + 1UZ; // the caller's span is compile-time long enough to have a tail
    } else {
        return span.size() > 1UZ ? span.data() + 1UZ : nullptr;
    }
}

template<typename T>
requires std::floating_point<T>
struct Fir {
    /// outputs accumulated at once, with each tap held broadcast across the batch so that no horizontal reduction
    /// enters the inner loop. 64 floats fill eight AVX2 accumulators and still leave registers for the taps.
    static constexpr std::size_t kOutputsPerBatch = 64UZ;

    /// `y[n] = Σ b[k] x[n-k]` for as many outputs as the spans allow. `input` is the window, so the first output
    /// needs `taps.size()` samples and the span yields `input.size() + 1 - taps.size()` of them.
    static void convolve(std::span<const T> input, std::span<const T> taps, std::span<T> output) noexcept {
        const std::size_t nTaps  = std::max(std::size_t{1}, taps.size());
        const std::size_t nOut   = input.size() + 1UZ >= nTaps ? std::min(output.size(), input.size() + 1UZ - nTaps) : 0UZ;
        const T*          weight = taps.data();
        const T*          sample = input.data();

        std::size_t n = 0UZ;
        for (; n + kOutputsPerBatch <= nOut; n += kOutputsPerBatch) {
            std::array<T, kOutputsPerBatch> batch{};
            for (std::size_t k = 0UZ; k < taps.size(); ++k) { // b[0] weights the newest sample, so the window is read from its end
                const T           tap    = weight[k];
                const std::size_t oldest = n + taps.size() - 1UZ - k;
                for (std::size_t j = 0UZ; j < kOutputsPerBatch; ++j) {
                    batch[j] += tap * sample[oldest + j];
                }
            }
            std::ranges::copy(batch, output.begin() + static_cast<std::ptrdiff_t>(n));
        }
        for (; n < nOut; ++n) {
            output[n] = sampleAt(input.subspan(n, nTaps), taps);
        }
    }

    /// `y[n] = Σ b[k] x[n-k]` over a span that does not carry its own lead-in, `history` holding the newest
    /// `taps.size() - 1` samples of the previous call, newest first. One output per input, so a caller need not
    /// declare a window; the outputs that straddle the seam are taken a sample at a time, the rest go through the
    /// batched body, and `history` is left holding this span's own tail.
    static void convolveStreaming(std::span<const T> input, std::span<const T> taps, std::span<T> history, std::span<T> output) noexcept {
        const std::size_t nLead    = taps.size() > 0UZ ? taps.size() - 1UZ : 0UZ;
        const std::size_t nSamples = std::min(input.size(), output.size());
        if (taps.empty()) {
            std::ranges::fill(output.first(nSamples), T{0}); // the caller publishes this span either way
            return;
        }
        // a history the caller has not sized is the stream's start: the windows that reach back read zeros, and
        // there is nowhere to carry this span's tail to
        const bool carries = history.size() >= nLead;

        // the windows that reach back before this span began, split at the seam so that neither run needs a test per
        // tap: a branch there is what makes this the expensive part when the lead-in is long
        const std::size_t nStraddling = std::min(nLead, nSamples);
        for (std::size_t n = 0UZ; n < nStraddling; ++n) {
            T sum{0};
            for (std::size_t k = 0UZ; k <= n; ++k) { // b[k] weights the sample k back from the newest
                sum += taps[k] * input[n - k];
            }
            if (carries) {
                for (std::size_t k = n + 1UZ; k < taps.size(); ++k) {
                    sum += taps[k] * history[k - n - 1UZ];
                }
            }
            output[n] = sum;
        }
        if (nSamples > nLead) { // the rest lies wholly inside this span
            convolve(input.first(nSamples), taps, output.subspan(nLead, nSamples - nLead));
        }

        // keep the newest samples, newest first. Walked backwards because a span shorter than the lead-in shifts the
        // older entries up, and going forwards would read slots this loop had already written
        if (carries) {
            for (std::size_t k = nLead; k-- > 0UZ;) {
                history[k] = k < nSamples ? input[nSamples - 1UZ - k] : history[k - nSamples];
            }
        }
    }

    /// `y = Σ taps[j] * window[newest - j*stride]`, the dot product a device can also run
    ///
    /// The device-callable form of `sampleAt`: no execution policy and no reverse iterators, because neither
    /// survives a kernel JIT. Taps that would reach before the window begins are dropped rather than wrapped,
    /// so a caller may hand it the stream's start. `stride` is 1 for a contiguous filter; a polyphase arm
    /// reads the raw stream at the channel count.
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample dotAt(std::span<const TSample> window, const T* taps, std::size_t nTaps, std::size_t newest, std::size_t stride = 1UZ) noexcept {
        TSample acc{};
        for (std::size_t j = 0UZ; j < nTaps; ++j) {
            const std::size_t back = j * stride;
            if (back > newest) {
                break; // reaches before the window begins: those taps weight samples that do not exist
            }
            if (const std::size_t index = newest - back; index < window.size()) {
                acc += taps[j] * window[index];
            }
        }
        return acc;
    }

    /// `y = Σ tapsReversed[j] * window[j]` — both walks run FORWARD, which is what lets it vectorise
    ///
    /// `window` points at the OLDEST sample of the window, and the taps are stored newest-weight-last. The
    /// reversal is done once where the taps are designed rather than per output: reading one of the two
    /// backwards is what stops a compiler vectorising the dot product at all.
    ///
    /// Device-callable and scalar. `dotReversedSimd` is the host counterpart over the same coefficients.
    template<typename TSample = T>
    [[nodiscard]] GR_DEVICE_FN static constexpr TSample dotReversed(const TSample* window, const T* tapsReversed, std::size_t nTaps) noexcept {
        TSample acc{};
        for (std::size_t j = 0UZ; j < nTaps; ++j) {
            acc += tapsReversed[j] * window[j];
        }
        return acc;
    }

    /// the same sum with explicit vectors and four accumulators
    ///
    /// Four, not one: a single `acc +=` chain runs at the latency of the FP add (4-5 cycles per tap) rather
    /// than its throughput, and that -- not the multiply -- is what a long FIR spends its time on. A plain
    /// unrolled scalar loop does NOT get vectorised here, so the vectors are written out.
    [[nodiscard]] static T dotReversedSimd(const T* window, const T* tapsReversed, std::size_t nTaps) noexcept {
        using V                        = stdx::native_simd<T>;
        constexpr std::size_t kWidth   = V::size();
        constexpr std::size_t kUnroll  = 4UZ;
        constexpr std::size_t kPerStep = kWidth * kUnroll;

        V           acc0{}, acc1{}, acc2{}, acc3{};
        std::size_t j = 0UZ;
        for (; j + kPerStep <= nTaps; j += kPerStep) {
            acc0 += V(tapsReversed + j, stdx::element_aligned) * V(window + j, stdx::element_aligned);
            acc1 += V(tapsReversed + j + kWidth, stdx::element_aligned) * V(window + j + kWidth, stdx::element_aligned);
            acc2 += V(tapsReversed + j + 2UZ * kWidth, stdx::element_aligned) * V(window + j + 2UZ * kWidth, stdx::element_aligned);
            acc3 += V(tapsReversed + j + 3UZ * kWidth, stdx::element_aligned) * V(window + j + 3UZ * kWidth, stdx::element_aligned);
        }
        for (; j + kWidth <= nTaps; j += kWidth) {
            acc0 += V(tapsReversed + j, stdx::element_aligned) * V(window + j, stdx::element_aligned);
        }
        T sum = stdx::reduce(acc0 + acc1 + acc2 + acc3);
        for (; j < nTaps; ++j) { // the tail, when the length is not a whole number of vectors
            sum += tapsReversed[j] * window[j];
        }
        return sum;
    }

    /// one output from the window ending at its newest sample
    [[nodiscard]] static T sampleAt(std::span<const T> window, std::span<const T> taps) noexcept {
        const auto newestTapFirst = std::make_reverse_iterator(taps.end());
        const auto oldestTapLast  = std::make_reverse_iterator(taps.begin());
        return std::transform_reduce(std::execution::unseq, newestTapFirst, oldestTapLast, window.data(), T{0}, std::plus<>{}, std::multiplies<>{});
    }
};

template<typename T, IIRForm form = IIRForm::DF_II>
requires std::floating_point<T>
struct Iir {
    /// how many histories the form keeps between samples: direct form II folds both into one, the others keep two
    static constexpr std::size_t kHistories = form == IIRForm::DF_II ? 1UZ : 2UZ;

    /// how long the state span must be for these coefficients: `kHistories` histories, each long enough for
    /// whichever side reads furthest back
    [[nodiscard]] static constexpr std::size_t stateSize(std::size_t nFeedforward, std::size_t nFeedback) noexcept { return kHistories * std::max(nFeedforward, nFeedback); }

    /// the state a pass can hold in locals. Beyond this it runs in the caller's span instead -- correct, merely
    /// slower. A cascade of biquads is what a high order should be built from and never comes near it.
    static constexpr std::size_t kMaxLocalState = 32UZ;

    /// `state` is `stateSize(b.size(), a.size() - 1)` long, carries between calls, and is the caller's to reset when
    /// the coefficients change meaning.
    static void filter(std::span<const T> input, std::span<const T> b, std::span<const T> a, std::span<T> state, std::span<T> output) noexcept {
        const std::size_t nSamples = std::min(input.size(), output.size());
        const std::size_t nState   = state.size() / kHistories;
        if (nState == 0UZ || nSamples == 0UZ) {
            std::ranges::fill(output.first(nSamples), T{0}); // the caller publishes this span either way
            return;
        }

        // A section is written at second or third order, and at those the whole recursion fits in registers:
        // specialising to the order makes state and coefficients compile-time sized, so every loop unrolls and nothing
        // in the dependency chain goes through memory. Worth 1.25x on a host and 6x on 'gpu:sycl', where the
        // alternative leaves a dependent read of device memory in the chain of every sample.
        {
            switch (nState) {
            case 1UZ: filterInRegisters<1UZ>(input, b, a, state, output); return;
            case 2UZ: filterInRegisters<2UZ>(input, b, a, state, output); return;
            case 3UZ: filterInRegisters<3UZ>(input, b, a, state, output); return;
            case 4UZ: filterInRegisters<4UZ>(input, b, a, state, output); return;
            default: break;
            }
        }

        if (nState > kMaxLocalState) { // too much state for locals: run it where it lies rather than truncate it
            const std::span<T> primary   = state.first(nState);
            const std::span<T> secondary = kHistories == 2UZ ? state.subspan(nState, nState) : primary;
            for (std::size_t i = 0UZ; i < nSamples; ++i) {
                output[i] = step(input[i], b, a, primary, secondary);
            }
            return;
        }

        std::array<T, kMaxLocalState> primary{};
        std::array<T, kMaxLocalState> secondary{};
        std::copy_n(state.begin(), nState, primary.begin());
        if constexpr (kHistories == 2UZ) {
            std::copy_n(state.begin() + static_cast<std::ptrdiff_t>(nState), nState, secondary.begin());
        }

        const std::span<T> primaryLocal{primary.data(), nState};
        const std::span<T> secondaryLocal{secondary.data(), nState};
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            output[i] = step(input[i], b, a, primaryLocal, secondaryLocal);
        }

        std::copy_n(primary.begin(), nState, state.begin());
        if constexpr (kHistories == 2UZ) {
            std::copy_n(secondary.begin(), nState, state.begin() + static_cast<std::ptrdiff_t>(nState));
        }
    }

    /**
     * @brief One sample of the recursion, for a caller that has a sample rather than a span.
     *
     * The histories are newest-first; a form keeping one history reads only `primary`. Same arithmetic as `filter`,
     * exposed for the streaming blocks whose surroundings are written a sample at a time.
     *
     * `kFixed` is the history length where the caller knows it at compile time and 0 where the spans carry
     * their own. It makes every bound below a constant, which is what lets the pass run in registers; the
     * coefficients are zero-padded past the history there, so a shorter set stays exact rather than merely
     * harmless -- a zero weight contributes nothing to the sum it appears in.
     */
    template<std::size_t kFixed = 0UZ>
    [[nodiscard]] static T step(T input, std::span<const T> b, std::span<const T> a, std::span<T> primary, std::span<T> secondary) noexcept {
        constexpr std::size_t kCoefficientExtent = kFixed == 0UZ ? 0UZ : kFixed + 1UZ; // zero-padded to this on the fixed path
        const std::size_t     nFeedforward       = std::min(b.size(), primary.size());
        const std::size_t     nFeedback          = std::min(a.size() > 0UZ ? a.size() - 1UZ : 0UZ, secondary.size());

        if constexpr (form == IIRForm::DF_I) {
            // y[n] = b[0] x[n] + b[1] x[n-1] + ... - a[1] y[n-1] - a[2] y[n-2] - ...
            push<kFixed>(primary, input);
            const T output = weightedSum<kFixed>(b.data(), nFeedforward, primary.data()) - weightedSum<kFixed>(afterFirst<T, kCoefficientExtent>(a), nFeedback, secondary.data());
            push<kFixed>(secondary, output);
            return output;
        } else if constexpr (form == IIRForm::DF_II) {
            // w[n] = x[n] - a[1] w[n-1] - ... ; y[n] = b[0] w[n] + b[1] w[n-1] + ...
            const T w = input - weightedSum<kFixed>(afterFirst<T, kCoefficientExtent>(a), std::min(a.size() > 0UZ ? a.size() - 1UZ : 0UZ, primary.size()), primary.data());
            push<kFixed>(primary, w);
            return weightedSum<kFixed>(b.data(), nFeedforward, primary.data());
        } else if constexpr (form == IIRForm::DF_I_TRANSPOSED) {
            const T v0 = input - weightedSum<kFixed>(afterFirst<T, kCoefficientExtent>(a), nFeedback, secondary.data());
            push<kFixed>(secondary, v0);
            return weightedSum<kFixed>(b.data(), std::min(b.size(), secondary.size()), secondary.data());
        } else {                                                                                                                                                     // DF_II_TRANSPOSED: y[n] = b[0] x[n] + Σ (b[k] x[n-k] - a[k] y[n-k])
            const T output = (b.empty() ? T{0} : b[0] * input)                                                                                                       //
                             + weightedSum<kFixed>(afterFirst<T, kCoefficientExtent>(b), std::min(b.empty() ? 0UZ : b.size() - 1UZ, primary.size()), primary.data()) //
                             - weightedSum<kFixed>(afterFirst<T, kCoefficientExtent>(a), nFeedback, secondary.data());
            push<kFixed>(primary, input);
            push<kFixed>(secondary, output);
            return output;
        }
    }

private:
    /// the same pass with every length known at compile time. The coefficients are copied into locals too: leaving
    /// them in the caller's memory would keep a load in the recursion's chain, which is the whole cost being removed.
    template<std::size_t kState>
    static void filterInRegisters(std::span<const T> input, std::span<const T> b, std::span<const T> a, std::span<T> state, std::span<T> output) noexcept {
        std::array<T, kState + 1UZ> feedforward{};
        std::array<T, kState + 1UZ> feedback{};
        std::copy_n(b.begin(), std::min(b.size(), kState + 1UZ), feedforward.begin());
        std::copy_n(a.begin(), std::min(a.size(), kState + 1UZ), feedback.begin());

        std::array<T, kState> primary{};
        std::array<T, kState> secondary{};
        std::copy_n(state.begin(), kState, primary.begin());
        if constexpr (kHistories == 2UZ) {
            std::copy_n(state.begin() + static_cast<std::ptrdiff_t>(kState), kState, secondary.begin());
        }

        const std::size_t nSamples = std::min(input.size(), output.size());
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            output[i] = step<kState>(input[i], feedforward, feedback, primary, secondary);
        }

        std::copy_n(primary.begin(), kState, state.begin());
        if constexpr (kHistories == 2UZ) {
            std::copy_n(secondary.begin(), kState, state.begin() + static_cast<std::ptrdiff_t>(kState));
        }
    }

    /// `kFixed` bounds the loop with a constant where the caller knows the order, and with the run-time count
    /// where it does not; that choice is the only thing separating the two storage paths.
    template<std::size_t kFixed>
    [[nodiscard]] static T weightedSum(const T* coefficient, std::size_t nCoefficients, const T* history) noexcept {
        T sum{0};
        if constexpr (kFixed != 0UZ) {
            for (std::size_t k = kFixed; k-- > 0UZ;) {
                sum += coefficient[k] * history[k];
            }
        } else {
            for (std::size_t k = nCoefficients; k-- > 0UZ;) {
                sum += coefficient[k] * history[k];
            }
        }
        return sum;
    }

    /// newest first, oldest off the end. At the orders a section is worth writing -- two or three coefficients -- the
    /// shift is fewer instructions than the index arithmetic a ring needs, and it leaves the history addressable.
    template<std::size_t kFixed>
    static void push(std::span<T> history, T value) noexcept {
        if constexpr (kFixed != 0UZ) {
            for (std::size_t k = kFixed; k-- > 1UZ;) {
                history[k] = history[k - 1UZ];
            }
            history[0] = value;
        } else {
            for (std::size_t k = history.size(); k-- > 1UZ;) {
                history[k] = history[k - 1UZ];
            }
            if (!history.empty()) {
                history[0] = value;
            }
        }
    }
};

/**
 * @brief A cascade of sections, each in transposed direct form II.
 *
 * What a designed filter actually is: biquads chained, not one long section. Each section keeps one accumulator per
 * state and updates them in place, so there is no history to shift -- the transposed form's advantage.
 *
 * `b` and `a` hold `nSections` rows of `nCoefficients`, row-major and zero-padded to a common length; `state` holds
 * `nSections * (nCoefficients - 1)` accumulators and carries between calls.
 */
template<typename T>
requires std::floating_point<T>
struct Cascade {
    /// the accumulators a pass may lift into locals -- eight biquads' worth. Past it the pass runs in the caller's
    /// span, which is correct and merely slower.
    static constexpr std::size_t kMaxLocalState = 64UZ;

    [[nodiscard]] GR_DEVICE_FN static constexpr std::size_t stateSize(std::size_t nSections, std::size_t nCoefficients) noexcept { return nSections * (nCoefficients > 0UZ ? nCoefficients - 1UZ : 0UZ); }

    /// every `decimation`-th filtered sample is published; the recursion still sees them all, because a cascade's
    /// state depends on every input it was given
    static void filter(std::span<const T> input, std::span<const T> b, std::span<const T> a, std::size_t nCoefficients, std::span<T> state, std::span<T> output, std::size_t decimation = 1UZ) noexcept {
        const std::size_t nSections = nCoefficients > 0UZ ? std::min(b.size(), a.size()) / nCoefficients : 0UZ;
        const std::size_t decim     = std::max(std::size_t{1}, decimation);
        const std::size_t nOut      = std::min(input.size() / decim, output.size());
        if (nSections == 0UZ || nOut == 0UZ) {
            std::ranges::fill(output.first(nOut), T{0}); // the caller publishes this span either way
            return;
        }
        // an undersized state is not something the local-array path can rescue: `step` indexes it per section and
        // would run past the end, so the design is refused rather than half-applied
        if (state.size() < stateSize(nSections, nCoefficients)) {
            std::ranges::fill(output.first(nOut), T{0});
            return;
        }

        if (state.size() > kMaxLocalState) {
            run(input, b, a, nCoefficients, state, output, decim, nOut);
            return;
        }
        std::array<T, kMaxLocalState> local{};
        std::copy_n(state.begin(), state.size(), local.begin());
        run(input, b, a, nCoefficients, std::span<T>{local.data(), state.size()}, output, decim, nOut);
        std::copy_n(local.begin(), state.size(), state.begin());
    }

    /// one sample through the cascade, for a caller holding a sample rather than a span
    /// device-callable, so a kernel that must run the recursion sequentially can still do it in place
    [[nodiscard]] GR_DEVICE_FN static T step(T input, std::span<const T> b, std::span<const T> a, std::size_t nCoefficients, std::span<T> state) noexcept {
        const std::size_t nStates   = nCoefficients > 0UZ ? nCoefficients - 1UZ : 0UZ;
        const std::size_t nSections = nCoefficients > 0UZ ? std::min(b.size(), a.size()) / nCoefficients : 0UZ;

        T sample = input;
        for (std::size_t section = 0UZ; section < nSections; ++section) {
            const T* bRow  = b.data() + section * nCoefficients;
            const T* aRow  = a.data() + section * nCoefficients;
            T*       slots = state.data() + section * nStates;

            if (nStates == 0UZ) { // a bare gain has nothing to remember
                sample = bRow[0] * sample;
                continue;
            }
            const T output = bRow[0] * sample + slots[0];
            for (std::size_t j = 0UZ; j + 1UZ < nStates; ++j) {
                slots[j] = bRow[j + 1UZ] * sample - aRow[j + 1UZ] * output + slots[j + 1UZ];
            }
            slots[nStates - 1UZ] = bRow[nStates] * sample - aRow[nStates] * output;
            sample               = output;
        }
        return sample;
    }

private:
    static void run(std::span<const T> input, std::span<const T> b, std::span<const T> a, std::size_t nCoefficients, std::span<T> state, std::span<T> output, std::size_t decim, std::size_t nOut) noexcept {
        std::size_t outIndex = 0UZ;
        for (std::size_t i = 0UZ; i < nOut * decim; ++i) {
            const T filtered = step(input[i], b, a, nCoefficients, state);
            if (i % decim == 0UZ) {
                output[outIndex++] = filtered;
            }
        }
    }
};

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_DIFFERENCE_EQUATION_HPP
