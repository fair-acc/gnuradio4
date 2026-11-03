#ifndef GNURADIO_FILTER_HILBERT_HPP
#define GNURADIO_FILTER_HILBERT_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

enum class HilbertOutput : std::uint8_t { complex, realImag, magnitude };

namespace detail {

/** One output sample per work item: the in-phase branch is the input delayed to the filter's centre `c`, the quadrature branch folds the antisymmetric taps, `taps[c + k] == -taps[c - k]` for odd `k` and zero for even `k`. */
template<std::floating_point T, HilbertOutput kOutput>
struct HilbertKernel {
    const T*    samples;
    const T*    leadIn;
    const T*    taps;
    T*          first;
    T*          second;
    std::size_t nTaps;

    /** a vector `V` reads the span only: it serves `n >= nTaps - 1`, where every tap reaches back inside it */
    template<typename V>
    [[nodiscard]] GR_DEVICE_FN V at(std::size_t n, std::size_t back) const noexcept {
        if constexpr (std::same_as<V, T>) {
            return n >= back ? samples[n - back] : leadIn[nTaps - 1UZ + n - back];
        } else {
            return V(samples + n - back, stdx::element_aligned);
        }
    }

    template<typename V = T>
    GR_DEVICE_FN void operator()(std::size_t n) const noexcept {
        const std::size_t centre = nTaps / 2UZ;
        V                 quadrature{T{0}};
        for (std::size_t k = 1UZ; k <= centre; k += 2UZ) {
            quadrature += V(taps[centre + k]) * (at<V>(n, centre + k) - at<V>(n, centre - k));
        }
        write(n, at<V>(n, centre), quadrature);
    }

    template<typename V>
    GR_DEVICE_FN void write(std::size_t n, const V& inPhase, const V& quadrature) const noexcept {
        constexpr bool kScalar = std::same_as<V, T>;
        if constexpr (kOutput == HilbertOutput::magnitude) {
            using std::sqrt;
            const V magnitude = sqrt(inPhase * inPhase + quadrature * quadrature);
            if constexpr (kScalar) {
                first[n] = magnitude;
            } else {
                magnitude.copy_to(first + n, stdx::element_aligned);
            }
        } else if constexpr (kOutput == HilbertOutput::realImag && kScalar) {
            first[n]  = inPhase;
            second[n] = quadrature;
        } else if constexpr (kOutput == HilbertOutput::realImag) {
            inPhase.copy_to(first + n, stdx::element_aligned);
            quadrature.copy_to(second + n, stdx::element_aligned);
        } else if constexpr (kScalar) {
            first[2UZ * n]       = inPhase;
            first[2UZ * n + 1UZ] = quadrature;
        } else {
            for (std::size_t lane = 0UZ; lane < V::size(); ++lane) {
                first[2UZ * (n + lane)]       = inPhase[lane];
                first[2UZ * (n + lane) + 1UZ] = quadrature[lane];
            }
        }
    }
};

/**
 * The design, the convolution and the tag delay the three Hilbert blocks share, over the block's own `_taps`, `_leadIn`, `_leadInNext` and `_tagsWithOutputIndex`.
 * They live on the block, not in a member struct, because every reflected member must be a settings type.
 */
template<std::floating_point T>
void designHilbertFor(auto& block, const property_map& changed) {
    if (!changed.contains("n_taps") && !block._taps.empty()) {
        return;
    }
    const std::vector<T> designed = gr::filter::fir::designHilbert<T>(block.n_taps.value);
    block._taps.assign(designed.begin(), designed.end());
    block._leadIn.assign(designed.size() - 1UZ, T{0});
    block._leadInNext.assign(designed.size() - 1UZ, T{0});
    block._tagsWithOutputIndex.clear();
}

inline constexpr std::size_t kMinTaps     = 3UZ;
inline constexpr std::size_t kMaxTaps     = 65537UZ;
inline constexpr gr::Size_t  kDefaultTaps = 65U;

template<HilbertOutput kOutput, std::floating_point T>
void convolveHilbert(auto& block, gr::device::DeviceContext* ctx, const T* samples, std::size_t nOut, T* first, T* second = nullptr) {
    const HilbertKernel<T, kOutput> kernel{.samples = samples, .leadIn = block._leadIn.data(), .taps = block._taps.data(), .first = first, .second = second, .nTaps = block._taps.size()};
    const std::size_t               lead = kernel.nTaps - 1UZ;
    if (ctx != nullptr) {
        gr::device::parallelFor(*ctx, nOut, [kernel] GR_DEVICE_LAMBDA(std::size_t n) { kernel(n); }, false);
    } else {
        using V       = stdx::native_simd<T>;
        std::size_t n = 0UZ;
        for (; n < std::min(lead, nOut); ++n) {
            kernel(n);
        }
        for (; n + V::size() <= nOut; n += V::size()) {
            kernel.template operator()<V>(n);
        }
        for (; n < nOut; ++n) {
            kernel(n);
        }
    }
    T* const next = block._leadInNext.data();
    gr::device::parallelFor(ctx != nullptr ? *ctx : gr::device::hostBackend(), lead, [kernel, next, nOut] GR_DEVICE_LAMBDA(std::size_t i) { next[i] = kernel.template at<T>(nOut, kernel.nTaps - 1UZ - i); });
    std::swap(block._leadIn, block._leadInNext);
}

template<typename TBlock>
void forwardHilbertTags(TBlock& block, auto& input, auto&& outputs, std::size_t nOut) {
    block.forwardDelayedTags(block._tagsWithOutputIndex, input, outputs, nOut, block._taps.size() / 2UZ);
}

} // namespace detail

GR_REGISTER_BLOCK("gr::filter::HilbertTransform", gr::filter::Hilbert, ([T], gr::filter::HilbertOutput::complex), [ float, double ])
GR_REGISTER_BLOCK("gr::filter::HilbertEnvelope", gr::filter::Hilbert, ([T], gr::filter::HilbertOutput::magnitude), [ float, double ])

template<std::floating_point T, HilbertOutput kOutput>
requires(kOutput != HilbertOutput::realImag)
struct Hilbert : Block<Hilbert<T, kOutput>, NoTagPropagation> {
    using Description = std::conditional_t<kOutput == HilbertOutput::complex, Doc<R""(@brief the analytic signal `x + j H{x}` of a real stream, by a windowed type-III FIR Hilbert transformer [1].

The real part is the input delayed by the group delay `(n_taps - 1) / 2`; tags move with their samples, and a tag within that delay of the stream's end is dropped.
DC and Nyquist cannot pass: a longer filter holds the quadrature closer to either.

[1] L. R. Rabiner and B. Gold, "Theory and Application of Digital Signal Processing", Englewood Cliffs, NJ: Prentice-Hall, 1975, sec. 3.25.)"">,
        Doc<"@brief the envelope of a real stream: the magnitude of its `HilbertTransform`, delayed with its tags by `(n_taps - 1) / 2`">>;
    using TOut        = std::conditional_t<kOutput == HilbertOutput::complex, std::complex<T>, T>;

    PortIn<T>     in;
    PortOut<TOut> out;

    Annotated<gr::Size_t, "n_taps", Doc<"FIR length, rounded up to odd">, Limits<detail::kMinTaps, detail::kMaxTaps>> n_taps = detail::kDefaultTaps;

    std::pmr::vector<T> _taps;
    std::pmr::vector<T> _leadIn;
    std::pmr::vector<T> _leadInNext;

    GR_MAKE_REFLECTABLE(Hilbert, in, out, n_taps, _taps, _leadIn, _leadInNext);

    DelayedTags _tagsWithOutputIndex;

    void start() { detail::designHilbertFor<T>(*this, {}); }
    void reset() {
        _taps.clear();
        detail::designHilbertFor<T>(*this, {});
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<TOut> output) {
        detail::convolveHilbert<kOutput>(*this, nullptr, input.data(), output.size(), reinterpret_cast<T*>(output.data()));
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        detail::forwardHilbertTags(*this, input, std::tie(output), output.size());
        return processBulk(std::span<const T>(input), std::span<TOut>(output));
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t nOut = std::min(input.size(), output.size());
        detail::convolveHilbert<kOutput>(*this, &ctx, input.data(), nOut, reinterpret_cast<T*>(output.data()));
        detail::forwardHilbertTags(*this, input, std::tie(output), nOut);
        std::ignore = input.consume(nOut);
        output.publish(nOut);
        return nOut == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) { detail::designHilbertFor<T>(*this, newSettings); }
};

template<std::floating_point T>
using HilbertTransform = Hilbert<T, HilbertOutput::complex>;

template<std::floating_point T>
using HilbertEnvelope = Hilbert<T, HilbertOutput::magnitude>;

GR_REGISTER_BLOCK(gr::filter::HilbertTransformRealImag, [T], [ float, double ])

template<std::floating_point T>
struct HilbertTransformRealImag : Block<HilbertTransformRealImag<T>, NoTagPropagation> {
    using Description = Doc<"@brief the analytic signal of `HilbertTransform` with its real and imaginary parts on separate ports">;

    PortIn<T>  in;
    PortOut<T> real;
    PortOut<T> imag;

    Annotated<gr::Size_t, "n_taps", Doc<"FIR length, rounded up to odd">, Limits<detail::kMinTaps, detail::kMaxTaps>> n_taps = detail::kDefaultTaps;

    std::pmr::vector<T> _taps;
    std::pmr::vector<T> _leadIn;
    std::pmr::vector<T> _leadInNext;

    GR_MAKE_REFLECTABLE(HilbertTransformRealImag, in, real, imag, n_taps, _taps, _leadIn, _leadInNext);

    DelayedTags _tagsWithOutputIndex;

    void start() { detail::designHilbertFor<T>(*this, {}); }
    void reset() {
        _taps.clear();
        detail::designHilbertFor<T>(*this, {});
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> realOut, std::span<T> imagOut) {
        detail::convolveHilbert<HilbertOutput::realImag>(*this, nullptr, input.data(), realOut.size(), realOut.data(), imagOut.data());
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& realOut, OutputSpanLike auto& imagOut) {
        detail::forwardHilbertTags(*this, input, std::tie(realOut, imagOut), realOut.size());
        return processBulk(std::span<const T>(input), std::span<T>(realOut), std::span<T>(imagOut));
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& realOut, OutputSpanLike auto& imagOut) {
        const std::size_t nOut = std::min({input.size(), realOut.size(), imagOut.size()});
        detail::convolveHilbert<HilbertOutput::realImag>(*this, &ctx, input.data(), nOut, realOut.data(), imagOut.data());
        detail::forwardHilbertTags(*this, input, std::tie(realOut, imagOut), nOut);
        std::ignore = input.consume(nOut);
        realOut.publish(nOut);
        imagOut.publish(nOut);
        return nOut == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) { detail::designHilbertFor<T>(*this, newSettings); }
};

} // namespace gr::filter

#endif // GNURADIO_FILTER_HILBERT_HPP
