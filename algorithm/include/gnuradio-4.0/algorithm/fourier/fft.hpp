#ifndef GNURADIO_ALGORITHM_FFT_HPP
#define GNURADIO_ALGORITHM_FFT_HPP

#include <bit>
#include <complex>
#include <execution>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm {

namespace detail {

template<gr::meta::complex_like T>
constexpr T complex_mult(T a, T b) {
    return a * b; // SIMD optimisation candidate
}

template<gr::meta::complex_like C, std::size_t N = std::dynamic_extent>
constexpr void fft_stage_kernel(C* data, const C* twiddles, std::size_t halfsize = 0UZ) {
    using ValueType = typename C::value_type;

    if constexpr (N == 2UZ) {
        const C a = data[0];
        const C b = data[1];

        data[0] = a + b;
        data[1] = a - b;
    } else if constexpr (N == 4UZ) {
        constexpr std::array<C, 2> twiddles4 = {
            C{1, 0},  // W_4^0
            C{0, -1}, // W_4^1 == -j
        };
        const C a = data[0];
        const C b = data[1];
        const C c = complex_mult(data[2], twiddles4[0]); // W_4^0 == 1
        const C d = complex_mult(data[3], twiddles4[1]); // W_4^1 == -j

        data[0] = a + c;
        data[1] = b + d;
        data[2] = a - c;
        data[3] = b - d;
    } else if constexpr (N == 8UZ) {
        constexpr auto inv_sqrt2 = static_cast<ValueType>(1 / std::numbers::sqrt2_v<ValueType>); // std::sqrt(0.5f)

        constexpr std::array<C, 4> twiddles8 = {//
            C{1, 0},                            //
            C{inv_sqrt2, -inv_sqrt2},           //
            C{0, -1},                           //
            C{-inv_sqrt2, -inv_sqrt2}};

        const C a0 = data[0];
        const C a1 = data[1];
        const C a2 = data[2];
        const C a3 = data[3];

        const C b0 = complex_mult(data[4], twiddles8[0]);
        const C b1 = complex_mult(data[5], twiddles8[1]);
        const C b2 = complex_mult(data[6], twiddles8[2]);
        const C b3 = complex_mult(data[7], twiddles8[3]);

        data[0] = a0 + b0;
        data[1] = a1 + b1;
        data[2] = a2 + b2;
        data[3] = a3 + b3;

        data[4] = a0 - b0;
        data[5] = a1 - b1;
        data[6] = a2 - b2;
        data[7] = a3 - b3;
    } else if constexpr (N == std::dynamic_extent) {
        for (std::size_t j = 0; j < halfsize; ++j) {
            const auto temp    = complex_mult(data[j + halfsize], twiddles[j]);
            data[j + halfsize] = data[j] - temp;
            data[j] += temp;
        }
    } else {
        static_assert(gr::meta::always_false<C>, "unimplemented power N of 2, 4, or 8");
    }
}

} // namespace detail

template<typename TInput, gr::meta::complex_like TOutput = std::conditional_t<gr::meta::complex_like<TInput>, TInput, std::complex<TInput>>>
requires((gr::meta::complex_like<TInput> || std::floating_point<TInput>))
struct FFT {
    using ValueType = typename TOutput::value_type;

    std::vector<std::vector<TOutput>>                     stageTwiddles{};
    std::vector<TOutput, gr::allocator::Aligned<TOutput>> bluesteinExpTable{};
    std::vector<TOutput, gr::allocator::Aligned<TOutput>> bluesteinChirpFFT{};
    std::vector<std::size_t>                              bitReverseTable{};
    std::size_t                                           fftSize{0};

    constexpr static Transform kTransform = gr::meta::complex_like<TInput> && gr::meta::complex_like<TOutput> ? Transform::Complex : Transform::Real;
    constexpr static Direction kDirection = kTransform == Transform::Complex ? Direction::Forward : std::is_arithmetic_v<TInput> && gr::meta::complex_like<TOutput> ? algorithm::Direction::Forward : algorithm::Direction::Backward;

    mutable SimdFFT<ValueType, kTransform>                            simdFFT{};
    mutable std::vector<ValueType, gr::allocator::Aligned<ValueType>> alignedInputBuffer{};
    mutable std::vector<ValueType, gr::allocator::Aligned<ValueType>> alignedOutputBuffer{};
    bool                                                              useSimdFFT{true};

    void initAll() {
        precomputeTwiddleFactors();
        precomputeBitReversal();
    }

    auto compute(const std::ranges::input_range auto& in, std::ranges::output_range<TOutput> auto&& out) {
        if constexpr (requires { out.resize(in.size()); }) {
            if (out.size() != in.size()) {
                out.resize(in.size());
            }
        }

        const auto size = in.size();
        if (size == 0) {
            return out;
        }

        if (fftSize != size) {
            fftSize = size;
            initAll();
        }
        if (!std::has_single_bit(size) && bluesteinExpTable.size() != size) {
            precomputeBluesteinTable(size); // added
        }

        if (useSimdFFT && SimdFFT<ValueType, kTransform>::canProcessSize(size, Order::Ordered) && trySimdFFT(in, out)) { // use SimdFFT if enabled and size is supported
            return out;
        }

        // fallback to original implementation
        std::ranges::transform(in, out.begin(), [](auto v) {
            if constexpr (std::floating_point<TInput>) {
                return TOutput(ValueType(v), 0);
            } else {
                return static_cast<TOutput>(v);
            }
        });

        if (std::has_single_bit(size)) {
            transformRadix2(out);
        } else {
            transformBluestein(out);
        }

        return out;
    }

    auto compute(const std::ranges::input_range auto& in) {
        using input_container_t = std::remove_cvref_t<decltype(in)>;
        using output_alloc_t    = gr::allocator::detail::deduce_output_allocator_t<input_container_t, TOutput>;
        return compute(in, std::vector<TOutput, output_alloc_t>(in.size()));
    }

    template<typename InRange, typename OutRange>
    bool trySimdFFT(const InRange& in, OutRange&& out) const {
        if (simdFFT.size() != in.size()) {
            simdFFT.resize(in.size());
        }
        if constexpr (kTransform == Transform::Complex) {
            return trySimdFFT_C2C(in, out, in.size());
        } else if constexpr (kDirection == Direction::Forward) {
            return trySimdFFT_R2C(in, out, in.size());
        } else if constexpr (kDirection == Direction::Backward) {
            return trySimdFFT_C2R(in, out, in.size());
        }
        return false;
    }

private:
    bool trySimdFFT_C2C(const auto& in, auto&& out, std::size_t N) const {
        using InputValueType        = typename std::remove_cvref_t<decltype(in)>::value_type::value_type;
        const std::size_t nElements = 2UZ * N;
        static_assert(sizeof(std::complex<ValueType>) == 2 * sizeof(ValueType), "SimdFFT backend expects interleaved scalars; complex<T> must be 2*T bytes.");

        if constexpr (std::is_same_v<InputValueType, ValueType>) {
            if (gr::allocator::isAligned(in.data(), 64UZ) && gr::allocator::isAligned(out.data(), 64UZ)) { // zero-copy path (same precision and aligned)
                std::span<const ValueType> inputSpan(reinterpret_cast<const ValueType*>(in.data()), nElements);
                std::span<ValueType>       outputSpan(reinterpret_cast<ValueType*>(out.data()), nElements);
                simdFFT.template transform<kDirection, Order::Ordered>(inputSpan, outputSpan);
                return true;
            } // future else: copy and avoid reinterpret_cast (slightly UB)
        }

        // buffered path for non-aligned or different precision
        if (alignedInputBuffer.size() != nElements) {
            alignedInputBuffer.resize(nElements);
        }
        if (alignedOutputBuffer.size() != nElements) {
            alignedOutputBuffer.resize(nElements);
        }

        // copy input with type conversion if needed
        const auto* inputPtr = reinterpret_cast<const InputValueType*>(in.data());
        for (std::size_t i = 0; i < nElements; ++i) {
            alignedInputBuffer[i] = static_cast<ValueType>(inputPtr[i]);
        }

        simdFFT.template transform<kDirection, Order::Ordered>(alignedInputBuffer, alignedOutputBuffer);

        // copy output with type conversion if needed
        auto* outputPtr = reinterpret_cast<ValueType*>(out.data());
        std::memcpy(outputPtr, alignedOutputBuffer.data(), nElements * sizeof(ValueType));

        return true;
    }

    bool trySimdFFT_R2C(const auto& in, auto&& out, std::size_t N) const {
        using InputValueType = typename std::remove_cvref_t<decltype(in)>::value_type;
        static_assert(std::is_trivially_copyable_v<InputValueType>);
        static_assert(std::is_trivially_copyable_v<ValueType>);
        assert(std::size(in) >= N);
        assert(std::size(out) >= N);

        if (alignedOutputBuffer.size() != N) {
            alignedOutputBuffer.resize(N); // packed real: [DC, Nyq, re1, im1, ...]
        }

        if constexpr (std::is_same_v<InputValueType, ValueType>) {
            if (gr::allocator::isAligned(in.data(), 64UZ)) { // input is cacheline-aligned
                simdFFT.template transform<kDirection, Order::Ordered>(std::span<const ValueType>{in.data(), N}, alignedOutputBuffer);
            } else { // not cacheline-aligned -> copy to aligned scratch
                if (alignedInputBuffer.size() != N) {
                    alignedInputBuffer.resize(N);
                }
                std::memcpy(alignedInputBuffer.data(), in.data(), N * sizeof(ValueType));
                simdFFT.template transform<kDirection, Order::Ordered>(std::span<const ValueType>{alignedInputBuffer.data(), N}, alignedOutputBuffer);
            }
        } else { // type conversion needed
            if (alignedInputBuffer.size() != N) {
                alignedInputBuffer.resize(N);
            }
            for (std::size_t i = 0; i < N; ++i) { // element-wise conversion (memcpy is invalid across types)
                alignedInputBuffer[i] = static_cast<ValueType>(in[i]);
            }
            simdFFT.template transform<kDirection, Order::Ordered>(std::span<const ValueType>{alignedInputBuffer.data(), N}, alignedOutputBuffer);
        }

        // unpack to full spectrum: [DC, Nyquist, re1, im1, re2, im2, ...] → N complex values
        out[0] = TOutput(static_cast<ValueType>(alignedOutputBuffer[0]), 0); // DC component at bin 0
        for (std::size_t k = 1; k < N / 2; ++k) {                            // positive frequencies (bins 1 to N/2-1)
            out[k] = TOutput(static_cast<ValueType>(alignedOutputBuffer[2 * k]), static_cast<ValueType>(alignedOutputBuffer[2 * k + 1]));
        }
        out[N / 2] = TOutput(static_cast<ValueType>(alignedOutputBuffer[1]), 0); // nyquist component at bin N/2

        for (std::size_t k = N / 2 + 1; k < N; ++k) { // negative frequencies (bins N/2+1 to N-1) - Hermitian symmetry -> complex conjugates
            const std::size_t mirrorIdx = N - k;
            out[k]                      = TOutput(static_cast<ValueType>(alignedOutputBuffer[2 * mirrorIdx]), -static_cast<ValueType>(alignedOutputBuffer[2 * mirrorIdx + 1])); // Conjugate
        }

        return true;
    }

    bool trySimdFFT_C2R(const auto& in, auto&& out, std::size_t N) const {
        using InComplex = typename std::remove_cvref_t<decltype(in)>::value_type;  // std::complex<Tin>
        using OutScalar = typename std::remove_cvref_t<decltype(out)>::value_type; // e.g. float/double
        static_assert(std::is_trivially_copyable_v<InComplex>);
        static_assert(std::is_trivially_copyable_v<ValueType>);

        assert((N % 2) == 0);
        assert(std::size(in) >= N);
        assert(std::size(out) >= N);

        if (alignedInputBuffer.size() != N) {
            alignedInputBuffer.resize(N); // ValueType
        }
        if (alignedOutputBuffer.size() != N) {
            alignedOutputBuffer.resize(N); // ValueType
        }

        // --- pack: N complex → [DC, Nyquist, re1, im1, re2, im2, ...] as ValueType ---
        alignedInputBuffer[0] = static_cast<ValueType>(in[0].real());
        alignedInputBuffer[1] = static_cast<ValueType>(in[N / 2].real());
        for (std::size_t k = 1; k < N / 2; ++k) {
            alignedInputBuffer[2 * k + 0] = static_cast<ValueType>(in[k].real());
            alignedInputBuffer[2 * k + 1] = static_cast<ValueType>(in[k].imag());
        }

        // --- choose output target for the backend ---
        ValueType* outUse    = nullptr;
        bool       directOut = false;

        if constexpr (std::is_same_v<OutScalar, ValueType>) {
            if (gr::allocator::isAligned(out.data(), 64UZ)) { // can write directly into the user buffer
                outUse    = out.data();                       // zero-copy output
                directOut = true;
            } else {
                outUse = alignedOutputBuffer.data(); // align-required scratch
            }
        } else { // different scalar type -> need to write to output scratch buffer, then convert
            outUse = alignedOutputBuffer.data();
        }

        simdFFT.template transform<kDirection, Order::Ordered>(std::span<const ValueType>{alignedInputBuffer.data(), N}, std::span<ValueType>{outUse, N}); // transform: N packed (ValueType) → N real (ValueType)

        if constexpr (std::is_same_v<OutScalar, ValueType>) {
            if (!directOut) {
                // same type → memcpy is optimal; alignment not required for byte copy
                std::memcpy(out.data(), alignedOutputBuffer.data(), N * sizeof(ValueType));
            }
        } else { // type conversion needed
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = static_cast<OutScalar>(alignedOutputBuffer[i]);
            }
        }

        return true;
    }

    void transformRadix2(std::ranges::input_range auto& inPlace) const {
        const std::size_t N = inPlace.size();
        if (!std::has_single_bit(N)) {
            throw std::invalid_argument(std::format("Input data must be power-of-two, input size: {}", inPlace.size()));
        }

        for (std::size_t i = 0UZ; i < N; ++i) {
            const std::size_t j = bitReverseTable[i];
            if (j > i) {
                std::swap(inPlace[i], inPlace[j]);
            }
        }

        const std::size_t nStages = static_cast<std::size_t>(std::countr_zero(N));
        for (std::size_t stage = 0, size = 2; stage < nStages; size *= 2, ++stage) {
            const auto&       twiddles = stageTwiddles[stage];
            const std::size_t halfsize = size / 2;

            for (std::size_t i = 0; i < N; i += size) {
                TOutput* block = &inPlace[i];

                switch (size) { // optimised non-branching fft sub kernels
                case 2: detail::fft_stage_kernel<TOutput, 2>(block, twiddles.data()); break;
                case 4: detail::fft_stage_kernel<TOutput, 4>(block, twiddles.data()); break;
                case 8: detail::fft_stage_kernel<TOutput, 8>(block, twiddles.data()); break;
                default: // generic case
                    detail::fft_stage_kernel<TOutput>(block, twiddles.data(), halfsize);
                    break;
                }
            }
        }
    }

    mutable std::unique_ptr<FFT<TOutput, TOutput>>                fftCache;
    mutable std::vector<TOutput, gr::allocator::Aligned<TOutput>> aCache{};
    mutable std::vector<TOutput, gr::allocator::Aligned<TOutput>> bCache{};

    void transformBluestein(std::ranges::input_range auto& inPlace) const {
        const std::size_t n = inPlace.size();
        const std::size_t m = std::bit_ceil(2 * n + 1);

        using input_container_t = std::remove_cvref_t<decltype(inPlace)>;
        using output_alloc_t    = gr::allocator::detail::deduce_output_allocator_t<input_container_t, TOutput>;
        std::vector<TOutput, output_alloc_t> a(m);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = detail::complex_mult(inPlace[i], bluesteinExpTable[i]);
        }

        // convolve input with chirp function
        if (a.size() != bluesteinChirpFFT.size()) {
            throw std::domain_error("mismatched lengths for convolution");
        }
        if (!fftCache) {
            fftCache = std::make_unique<FFT<TOutput, TOutput>>();
        }

        aCache = fftCache->compute(a, aCache); // forward FFT

        // pointwise multiply with chirp spectrum
        std::transform(aCache.begin(), aCache.end(), bluesteinChirpFFT.begin(), aCache.begin(), std::multiplies<>{});
        bCache = fftCache->compute(aCache, bCache); // inverse FFT

        const ValueType scale = ValueType(1) / ValueType(bCache.size());                          // normalise FFT and scale by 1/N
        std::transform(bCache.begin(), std::next(bCache.begin(), static_cast<std::ptrdiff_t>(n)), // restrict to signal size (N.B. m > n)
            bluesteinExpTable.begin(), inPlace.begin(), [scale](auto v, auto w) { return detail::complex_mult(v * scale, w); });
    }

    void precomputeTwiddleFactors(bool inverse = false) {
        stageTwiddles.clear();
        const auto minus2Pi = ValueType(inverse ? 2 : -2) * std::numbers::pi_v<ValueType>;
        for (std::size_t size = 2UZ; size <= fftSize; size *= 2UZ) {
            const std::size_t    m{size / 2};
            const TOutput        w{std::exp(TOutput(0., minus2Pi / static_cast<ValueType>(size)))};
            std::vector<TOutput> twiddles;
            if (size == 2) {
                twiddles.push_back(TOutput{1.0, 0.0});
            } else if (size == 8) {
                twiddles.emplace_back(1.0, 0.0);                         // W_8^0
                twiddles.emplace_back(std::sqrt(0.5), -std::sqrt(0.5));  // W_8^1
                twiddles.emplace_back(0.0, -1.0);                        // W_8^2
                twiddles.emplace_back(-std::sqrt(0.5), -std::sqrt(0.5)); // W_8^3
            } else {
                TOutput wk{1., 0.};
                for (std::size_t j = 0UZ; j < m; ++j) {
                    twiddles.push_back(wk);
                    wk *= w;
                }
            }
            stageTwiddles.push_back(std::move(twiddles));
        }
    }

    void precomputeBluesteinTable(std::size_t n, bool inverse = false) {
        bluesteinExpTable.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uintmax_t tmp   = static_cast<std::uintmax_t>(i) * i % (2 * n);
            const ValueType      angle = (inverse ? -1 : 1) * std::numbers::pi_v<ValueType> * static_cast<ValueType>(tmp) / static_cast<ValueType>(n);
            bluesteinExpTable[i]       = std::polar<ValueType>(1.0, angle);
        }

        const std::size_t                                     m = std::bit_ceil(2 * n + 1);
        std::vector<TOutput, gr::allocator::Aligned<TOutput>> b(m);
        b[0] = bluesteinExpTable[0];
        for (std::size_t i = 1; i < n; ++i) {
            b[i] = b[m - i] = std::conj(bluesteinExpTable[i]);
        }

        FFT<TOutput, TOutput> fft{}; // always power-of-two
        bluesteinChirpFFT = fft.compute(b);
    }

    void precomputeBitReversal() {
        const std::size_t width = static_cast<std::size_t>(std::countr_zero(fftSize));
        bitReverseTable.resize(fftSize);
        for (std::size_t i = 0; i < fftSize; ++i) {
            std::size_t val = i, result = 0;
            for (std::size_t j = 0; j < width; ++j, val >>= 1) {
                result = (result << 1) | (val & 1U);
            }
            bitReverseTable[i] = result;
        }
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_FFT_HPP
