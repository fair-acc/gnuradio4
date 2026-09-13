#ifndef GNURADIO_ALGORITHM_DIFFERENCE_EQUATION_HPP
#define GNURADIO_ALGORITHM_DIFFERENCE_EQUATION_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <span>

namespace gr::algorithm::filter {

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
 * ```cpp
 * std::array<float, 3> state{};                              // one biquad section
 * Iir<float>::filter(input, b, a, state, output);            // state carries on to the next call
 * ```
 */

/// where an IIR keeps its state. The forms compute the same response and differ in what they hold
/// between samples, which is what makes one preferable in fixed point and another in floating point.
enum class IIRForm {
    DF_I,  /// direct form I: preferred for fixed-point arithmetics (e.g. no overflow)
    DF_II, /// direct form II: preferred for floating-point arithmetics (less operations)
    DF_I_TRANSPOSED,
    DF_II_TRANSPOSED,
};

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
    /// `taps.size() - 1` samples of the previous call. One output per input, so a caller need not declare a window; the
    /// outputs that straddle the seam are taken a sample at a time and the rest go through the batched body.
    /// whether a span is long enough for a streaming convolution to be worth it. Its seam costs one window per
    /// lead-in sample, so the cost is quadratic in the tap count and fixed per call: it pays only once the span it
    /// amortises over is several filter lengths long. Shorter than that, a recursion costing one window per sample wins.
    [[nodiscard]] static constexpr bool streamingPays(std::size_t nSamples, std::size_t nTaps) noexcept { return nSamples >= 8UZ * nTaps; }

    static void convolveStreaming(std::span<const T> input, std::span<const T> taps, std::span<T> history, std::span<T> output) noexcept {
        const std::size_t nLead    = taps.size() > 0UZ ? taps.size() - 1UZ : 0UZ;
        const std::size_t nSamples = std::min(input.size(), output.size());
        if (taps.empty() || history.size() < nLead) {
            return;
        }

        // the windows that reach back before this span began, split at the seam so that neither run needs a test per
        // tap: a branch there is what makes this the expensive part when the lead-in is long
        const std::size_t nStraddling = std::min(nLead, nSamples);
        for (std::size_t n = 0UZ; n < nStraddling; ++n) {
            T sum{0};
            for (std::size_t k = 0UZ; k <= n; ++k) { // b[k] weights the sample k back from the newest
                sum += taps[k] * input[n - k];
            }
            for (std::size_t k = n + 1UZ; k < taps.size(); ++k) {
                sum += taps[k] * history[k - n - 1UZ];
            }
            output[n] = sum;
        }
        if (nSamples > nLead) { // the rest lies wholly inside this span
            convolve(input.first(nSamples), taps, output.subspan(nLead, nSamples - nLead));
        }

        // keep the newest samples, newest first. Walked backwards because a span shorter than the lead-in shifts the
        // older entries up, and going forwards would read slots this loop had already written
        for (std::size_t k = nLead; k-- > 0UZ;) {
            history[k] = k < nSamples ? input[nSamples - 1UZ - k] : history[k - nSamples];
        }
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
     */
    [[nodiscard]] static T step(T input, std::span<const T> b, std::span<const T> a, std::span<T> primary, std::span<T> secondary) noexcept {
        const std::size_t nFeedforward = std::min(b.size(), primary.size());
        const std::size_t nFeedback    = std::min(a.size() > 0UZ ? a.size() - 1UZ : 0UZ, secondary.size());

        if constexpr (form == IIRForm::DF_I) {
            // y[n] = b[0] x[n] + b[1] x[n-1] + ... - a[1] y[n-1] - a[2] y[n-2] - ...
            pushFront(primary, input);
            const T output = weightedSumOldestFirst(b.data(), nFeedforward, primary.data()) - weightedSumOldestFirst(a.data() + 1, nFeedback, secondary.data());
            pushFront(secondary, output);
            return output;
        } else if constexpr (form == IIRForm::DF_II) {
            // w[n] = x[n] - a[1] w[n-1] - ... ; y[n] = b[0] w[n] + b[1] w[n-1] + ...
            const T w = input - weightedSumOldestFirst(a.data() + 1, std::min(a.size() > 0UZ ? a.size() - 1UZ : 0UZ, primary.size()), primary.data());
            pushFront(primary, w);
            return weightedSumOldestFirst(b.data(), nFeedforward, primary.data());
        } else if constexpr (form == IIRForm::DF_I_TRANSPOSED) {
            const T v0 = input - weightedSumOldestFirst(a.data() + 1, nFeedback, secondary.data());
            pushFront(secondary, v0);
            return weightedSumOldestFirst(b.data(), std::min(b.size(), secondary.size()), secondary.data());
        } else {                                                                                                                     // DF_II_TRANSPOSED: y[n] = b[0] x[n] + Σ (b[k] x[n-k] - a[k] y[n-k])
            const T output = (b.empty() ? T{0} : b[0] * input)                                                                       //
                             + weightedSumOldestFirst(b.data() + 1, std::min(b.empty() ? 0UZ : b.size() - 1UZ, primary.size()), primary.data()) //
                             - weightedSumOldestFirst(a.data() + 1, nFeedback, secondary.data());
            pushFront(primary, input);
            pushFront(secondary, output);
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
            output[i] = stepInRegisters<kState>(input[i], feedforward, feedback, primary, secondary);
        }

        std::copy_n(primary.begin(), kState, state.begin());
        if constexpr (kHistories == 2UZ) {
            std::copy_n(secondary.begin(), kState, state.begin() + static_cast<std::ptrdiff_t>(kState));
        }
    }

    /// one sample with every bound a constant. The zero padding makes a shorter coefficient set exact rather than
    /// merely harmless: a zero weight contributes nothing to the sum it appears in.
    template<std::size_t kState>
    [[nodiscard]] static T stepInRegisters(T input, const std::array<T, kState + 1UZ>& b, const std::array<T, kState + 1UZ>& a, std::array<T, kState>& primary, std::array<T, kState>& secondary) noexcept {
        if constexpr (form == IIRForm::DF_I) {
            shiftFixed(primary, input);
            const T output = weightedSumFixedOldestFirst<kState>(b.data(), primary.data()) - weightedSumFixedOldestFirst<kState>(a.data() + 1, secondary.data());
            shiftFixed(secondary, output);
            return output;
        } else if constexpr (form == IIRForm::DF_II) {
            const T w = input - weightedSumFixedOldestFirst<kState>(a.data() + 1, primary.data());
            shiftFixed(primary, w);
            return weightedSumFixedOldestFirst<kState>(b.data(), primary.data());
        } else if constexpr (form == IIRForm::DF_I_TRANSPOSED) {
            const T v0 = input - weightedSumFixedOldestFirst<kState>(a.data() + 1, secondary.data());
            shiftFixed(secondary, v0);
            return weightedSumFixedOldestFirst<kState>(b.data(), secondary.data());
        } else { // DF_II_TRANSPOSED
            const T output = b[0] * input + weightedSumFixedOldestFirst<kState>(b.data() + 1, primary.data()) - weightedSumFixedOldestFirst<kState>(a.data() + 1, secondary.data());
            shiftFixed(primary, input);
            shiftFixed(secondary, output);
            return output;
        }
    }

    template<std::size_t kCount>
    [[nodiscard]] static T weightedSumFixedOldestFirst(const T* coefficient, const T* history) noexcept {
        T sum{0};
        for (std::size_t k = kCount; k-- > 0UZ;) {
            sum += coefficient[k] * history[k];
        }
        return sum;
    }

    template<std::size_t kState>
    static void shiftFixed(std::array<T, kState>& history, T value) noexcept {
        for (std::size_t k = kState; k-- > 1UZ;) {
            history[k] = history[k - 1UZ];
        }
        if constexpr (kState > 0UZ) {
            history[0] = value;
        }
    }

    [[nodiscard]] static T weightedSumOldestFirst(const T* coefficient, std::size_t nCoefficients, const T* history) noexcept {
        T sum{0};
        for (std::size_t k = nCoefficients; k-- > 0UZ;) {
            sum += coefficient[k] * history[k];
        }
        return sum;
    }

    /// newest first, oldest off the end. At the orders a section is worth writing -- two or three coefficients -- the
    /// shift is fewer instructions than the index arithmetic a ring needs, and it leaves the history addressable.
    static void pushFront(std::span<T> history, T value) noexcept {
        for (std::size_t k = history.size(); k-- > 1UZ;) {
            history[k] = history[k - 1UZ];
        }
        if (!history.empty()) {
            history[0] = value;
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

    [[nodiscard]] static constexpr std::size_t stateSize(std::size_t nSections, std::size_t nCoefficients) noexcept { return nSections * (nCoefficients > 0UZ ? nCoefficients - 1UZ : 0UZ); }

    /// every `decimation`-th filtered sample is published; the recursion still sees them all, because a cascade's
    /// state depends on every input it was given
    static void filter(std::span<const T> input, std::span<const T> b, std::span<const T> a, std::size_t nCoefficients, std::span<T> state, std::span<T> output, std::size_t decimation = 1UZ) noexcept {
        const std::size_t nSections = nCoefficients > 0UZ ? std::min(b.size(), a.size()) / nCoefficients : 0UZ;
        const std::size_t decim     = std::max(std::size_t{1}, decimation);
        const std::size_t nOut      = std::min(input.size() / decim, output.size());
        if (nSections == 0UZ || nOut == 0UZ) {
            return;
        }

        if (state.size() < stateSize(nSections, nCoefficients) || state.size() > kMaxLocalState) {
            run(input, b, a, nCoefficients, state, output, decim, nOut);
            return;
        }
        std::array<T, kMaxLocalState> local{};
        std::copy_n(state.begin(), state.size(), local.begin());
        run(input, b, a, nCoefficients, std::span<T>{local.data(), state.size()}, output, decim, nOut);
        std::copy_n(local.begin(), state.size(), state.begin());
    }

    /// one sample through the cascade, for a caller holding a sample rather than a span
    [[nodiscard]] static T step(T input, std::span<const T> b, std::span<const T> a, std::size_t nCoefficients, std::span<T> state) noexcept {
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
