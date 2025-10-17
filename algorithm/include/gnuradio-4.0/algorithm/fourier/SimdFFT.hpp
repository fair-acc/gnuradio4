/* SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2025  Matthias Kretz (m.kretz@gsi.de) &
                    Ralph J. Steinhagen (r.steinhagen@gsi.de)
                    GSI Helmholtz Centre for Heavy Ion Research, &
                    FAIR - Facility for Antiproton & Ion Research,
                    Darmstadt, Germany
Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )
Copyright (c) 2019  Hayati Ayguen ( h_ayguen@web.de )
Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

Copyright (c) 2004 the University Corporation for Atmospheric
Research ("UCAR"). All rights reserved. Developed by NCAR's
Computational and Information Systems Laboratory, UCAR,
www.cisl.ucar.edu.

Redistribution and use of the Software in source and binary forms,
with or without modification, is permitted provided that the
following conditions are met:

- Neither the names of NCAR's Computational and Information Systems
Laboratory, the University Corporation for Atmospheric Research,
nor the names of its sponsors or contributors may be used to
endorse or promote products derived from this Software without
specific prior written permission.

- Redistributions of source code must retain the above copyright
notices, this list of conditions, and the disclaimer below.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions, and the disclaimer below in the
documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
*/
#ifndef SIMD_FFT_HPP
#define SIMD_FFT_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <source_location>
#include <span>

#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Message.hpp>

#if defined(__clang__) || defined(__GNUC__)
#define ALWAYS_INLINE(return_type) inline return_type __attribute__((always_inline))
#define NEVER_INLINE(return_type)  return_type __attribute__((noinline))
#define RESTRICT                   __restrict
#elif defined(COMPILER_MSVC)
#define ALWAYS_INLINE(return_type) __forceinline return_type
#define NEVER_INLINE(return_type)  __declspec(noinline) return_type
#define RESTRICT                   __restrict
#endif

#ifndef __cpp_aligned_new
#error
#endif

#include <vir/simd.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"          // warning/error in vir/simd
#pragma GCC diagnostic ignored "-Wsign-conversion" // warning/error in vir/simd
#pragma GCC diagnostic ignored "-Wconversion"      // warning/error in vir/simd
#endif

#include <vir/simd_execution.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace stdx = vir::stdx;

namespace gr::algorithm {
template<std::floating_point T, int N = 4> // inspired by future C++26 definition
using vec = stdx::simd<T, stdx::simd_abi::deduce_t<T, static_cast<std::size_t>(N)>>;

namespace details {
template<typename T>
constexpr ALWAYS_INLINE(void) store_unchecked(const T& v, typename T::value_type* ptr, auto) {
    v.copy_to(ptr, stdx::vector_aligned);
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) blend_lo2_hi1(const Vec& a, const Vec& b) noexcept {
    // takes low from 2nd arg, high from 1st (compact, FFT convention)
    // equivalent for lane 'i': out[i] = (i < 2) ? b[i] : a[i];
    constexpr int size = Vec::size();
    return vir::simd_permute<size>(stdx::concat(b, a), [](int i) { return i < 2 ? i : i + size; });
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) interleave(const Vec& in1, const Vec& in2, Vec& out1, Vec& out2) noexcept {
    constexpr int size   = Vec::size();
    std::tie(out1, out2) = stdx::split<Vec>(vir::simd_permute(stdx::concat(in1, in2), [](int i) { return (i >> 1) + size * (i & 1); }));
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) uninterleave(const Vec& in1, const Vec& in2, Vec& out1, Vec& out2) noexcept {
    constexpr int size   = Vec::size();
    std::tie(out1, out2) = stdx::split<Vec>(vir::simd_permute(stdx::concat(in1, in2), [](int i) noexcept -> int { return (i % size) * 2 + (i / size); }));
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) transpose(Vec& x0, Vec& x1, Vec& x2, Vec& x3) noexcept {
    constexpr int size          = Vec::size();
    const auto [y0, y1, y2, y3] = stdx::split<Vec>(vir::simd_permute(stdx::concat(x0, x1, x2, x3), [](int i) noexcept -> int { return (i % size) * size + (i / size); }));
    x0                          = y0;
    x1                          = y1;
    x2                          = y2;
    x3                          = y3;
}

/* shortcuts for complex multiplications */
template<typename Vec, typename T>
constexpr static ALWAYS_INLINE(void) complex_multiply(Vec& ar, Vec& ai, const T& br, const T& bi) noexcept {
    const Vec tmp = ar * bi;
    ar            = ar * br - ai * bi;
    ai            = ai * br + tmp;
}

template<typename Vec, typename T>
constexpr static ALWAYS_INLINE(void) complex_multiply_conj(Vec& ar, Vec& ai, const T& br, const T& bi) noexcept {
    const Vec tmp = ar * bi;
    ar            = ar * br + ai * bi;
    ai            = ai * br - tmp;
}

template<std::size_t Align, typename T>
[[nodiscard]] constexpr bool isAligned(const T* p) noexcept {
    return std::bit_cast<std::uintptr_t>(std::to_address(p)) % Align == 0UZ;
}

template<typename T>
[[nodiscard]] constexpr bool isAligned(const T* p, std::size_t alignment) noexcept {
    return std::bit_cast<std::uintptr_t>(std::to_address(p)) % alignment == 0UZ;
}
} // namespace details

/// @brief data layout ordering for FFT input/output
enum class Order {
    Ordered,  /// R2C: [DC, Nyquist, Re(1), Im(1), Re(2), Im(2), …, Re(N/2-1), Im(N/2-1)] (Nyquist at index 1)
              ///      ONLY for N % 32 == 0 (power-of-2 aligned)
              /// C2C: [X[0]=DC, … X[N/2]=Nyquist, X[N/2+1]=-Nyquist, …, X[N-1]=left of DC] (natural DFT order)
              ///      for any N factoring into {2, 3, 4, 5}
    Unordered /// R2C: SIMD-tiled with mixed-radix {2,3,4,5} bit-reversal permutation
              ///      - for any N factoring into {2, 3, 4, 5}
              ///      - organized in 8*L blocks: [Re(k)…Re(k+3), Im(k)…Im(k+3)] (L=4)
              ///      - bins NOT sequential, follow radix order (e.g. N=48: [0-3], [6,11,10,9], [12-15], [18,23,22,21])
              ///      - DC at pos 0, Nyquist at pos L
              /// C2C: [Re0,Re4,Re8,Re12, Im0,Im4,Im8,Im12, Re1,Re5,Re9,Re13, Im1,Im5,Im9,Im13,…] (L=4)
              ///      gor any N factoring into {2, 3, 4, 5}
              /// ~20% faster; REQUIRED for non-power-of-2 R2C; preferred for convolution
};

/// @brief Transform direction for FFT operations
/// @note Transforms are unnormalized: backward(forward(x)) = N*x
enum class Direction {
    Forward, /// Time → frequency domain (R2C or C2C forward)
    Backward /// Frequency → time domain (C2R or C2C inverse)
};

/// @brief Type of Fourier transform
enum class Transform {
    Real,   /// real-valued input/output (exploits Hermitian symmetry)
    Complex /// complex-valued input/output (general DFT)
};

template<class R, class T>
concept InBuf = std::ranges::borrowed_range<R> && std::ranges::contiguous_range<R> && std::convertible_to<decltype(std::ranges::data(std::declval<R&>())), const T*>;
template<class R, class T>
concept OutBuf = std::ranges::borrowed_range<R> && std::ranges::contiguous_range<R> && std::same_as<decltype(std::ranges::data(std::declval<R&>())), T*>;

// clang-format off
/// @brief Tag types for direction/ordering dispatch
struct forward_t   { static constexpr Direction value = Direction::Forward; };
struct backward_t  { static constexpr Direction value = Direction::Backward; };
struct ordered_t   { static constexpr Order value = Order::Ordered; };
struct unordered_t { static constexpr Order value = Order::Unordered; };
// clang-format on

/// @brief tag instances for transform dispatch
/// @example fft.transform(forward, unordered, input, output);
inline constexpr forward_t   forward{};
inline constexpr backward_t  backward{};
inline constexpr ordered_t   ordered{};
inline constexpr unordered_t unordered{};

namespace details { // forward declaration, implementations below

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix2(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles);
template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix3(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2);
template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix4(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3);
template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix5(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3, std::span<const T> twiddles4);

template<std::floating_point T>
constexpr void complexFinalise(std::size_t Ncvec, std::span<T> input, std::span<T> output, std::span<const T> butterflyTwiddles);
template<std::floating_point T>
constexpr void complexPreprocess(std::size_t Ncvec, std::span<const T> input, std::span<T> output, std::span<const T> butterflyTwiddles);

template<Direction dir, std::floating_point T>
static NEVER_INLINE(void) realRadix2(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles);
template<Direction dir, std::floating_point T>
static NEVER_INLINE(void) realRadix3(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2);
template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) realRadix4(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3);
template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) realRadix5(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3, std::span<const T> twiddles4);

template<std::array<std::size_t, 5UZ> ntryh>
static constexpr std::size_t decompose(std::size_t n, std::span<std::size_t> radixPlan);

template<Direction dir, Transform transform, std::floating_point T>
void dispatchRadix(std::size_t radix, std::size_t stride, std::size_t nGroups, std::span<T> in, std::span<T> out, std::span<const T> twiddles, std::size_t offset);

template<Direction dir, Transform transform, std::floating_point T>
static NEVER_INLINE(std::span<T>) fftStages(std::size_t nVectors, std::span<const T> input, std::span<T> workBuffer1, std::span<T> workBuffer2, std::span<const T> twiddles, std::span<const std::size_t, 15> radixPlan);

template<std::floating_point T>
static void reversed_copy(std::size_t N, const vec<T>* in, std::size_t in_stride, vec<T>* out);
template<std::floating_point T>
void unreversed_copy(std::size_t N, const vec<T>* in, vec<T>* out, int out_stride);

template<std::floating_point T>
constexpr ALWAYS_INLINE(void) realFinalise_4x4(const T* RESTRICT in0, const T* RESTRICT in1, const T* RESTRICT in, std::span<const T> eSpan, T* RESTRICT out);
template<std::floating_point T>
constexpr void realFinalise(std::size_t Ncvec, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<const T> e);
template<std::floating_point T>
constexpr ALWAYS_INLINE(void) realPreprocess_4x4(const T* RESTRICT in, std::span<const T> eSpan, T* RESTRICT out, std::size_t first);
template<std::floating_point T>
constexpr void realPreprocess(std::size_t Ncvec, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<const T> e);

} // namespace details

/**
 * @brief SIMD-optimized FFT for real or complex transforms
 * @tparam T float or double
 * @tparam fftTransform Transform::Real or Transform::Complex
 * @tparam N Size at compile-time, or std::dynamic_extent for runtime sizing
 *
 * Supports mixed-radix {2,3,4,5} factorizations. Real transforms use Hermitian symmetry.
 * Transforms are unnormalized: backward(forward(x)) = N*x (scale by 1/N for identity).
 *
 * @example
 * SimdFFT<float, Transform::Real, 1024> fft;
 * // alt: SimdFFT<float, Transform::Real> fft(1025);
 * fft.transform(forward, unordered, input, output);
 * // Scale for inverse: for(auto& x : output) x /= 1024;
 *
 * or:
 * @example
 * SimdFFT<float, Transform::Complex, 1024> fft;
 * fft.transform(forward, unordered, input, output);
 */
template<std::floating_point T, Transform fftTransform, std::size_t N = std::dynamic_extent>
struct SimdFFT {
    using value_type   = T;
    using vector_type  = vec<T, 4>;
    using V            = vector_type;
    using IsRealValued = std::conditional_t<fftTransform == Transform::Real, std::true_type, std::false_type>;
    using IsDynamic    = std::conditional_t<N == std::dynamic_extent, std::true_type, std::false_type>;

    static constexpr std::size_t kAlignment = 64UZ;
    static constexpr std::size_t ceil_div(std::size_t x, std::size_t y) { return (x + y - 1) / y; }
    static constexpr std::size_t kTwiddleCount = 2 * (V::size() - 1) * V::size() * ceil_div((IsRealValued::value ? (N / 2) : N) / V::size(), V::size());

    using WorkStorage             = std::conditional_t<IsDynamic::value, std::vector<T, gr::allocator::Aligned<T, kAlignment>>, std::array<T, IsRealValued::value ? N : 2 * N>>;
    using StageTwiddleStorage     = std::conditional_t<IsDynamic::value, std::vector<T, gr::allocator::Aligned<T, kAlignment>>, std::array<T, kTwiddleCount>>;
    using ButterflyTwiddleStorage = std::conditional_t<IsDynamic::value, std::vector<T, gr::allocator::Aligned<T, kAlignment>>, std::array<T, 2 * (V::size() - 1) * ((IsRealValued::value ? (N / 2) : N) / V::size())>>; // 2*(L-1)*SV

    std::size_t                   _N = N;
    std::array<std::size_t, 15UZ> _radixPlan{}; // [0]: unused (FFTPACK legacy), [1]: number of stages nf, [2 .. 1 + nf] stage radices in execution ordering (2,3,4,5,...)

    alignas(64) WorkStorage _scratch{};                       // used as local data storage
    alignas(64) StageTwiddleStorage _stageTwiddles{};         // stage-level twiddles
    alignas(64) ButterflyTwiddleStorage _butterflyTwiddles{}; // butterfly simd-level twiddles

    /// @brief Construct compile-time sized FFT
    /// N must be valid, can be checked using `canProcessSize(std::size_t) -> bool`
    constexpr SimdFFT()
    requires(!IsDynamic::value)
    {
        static_assert(!canProcessSize(size()), "cannot process this size: min>=16C2C (32: R2C) & radix-2, -3, -5 & 'x min' compatible");
        computeTwiddles();
    }

    /// @brief construct runtime-sized FFT handler
    /// @throws gr::exception if n is incompatible (must factor into {2,3,4,5} and >= minSize()), can be checked using `canProcessSize(std::size_t) -> bool`
    explicit SimdFFT(std::size_t n = 64UZ, std::source_location loc = std::source_location::current())
    requires(IsDynamic::value)
        : _N(n) {
        resize(n, loc);
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        if constexpr (IsDynamic::value) {
            return _N;
        } else {
            return N;
        }
    }
    constexpr void resize(std::size_t n, std::source_location loc = std::source_location::current())
    requires(IsDynamic::value)
    {
        _N = n;
        if (!canProcessSize(size())) {
            throw gr::exception(std::format("incompatible sizes for {}2C: N ({}) must be multiple of 2,3,4,5 and >{}", fftTransform == Transform::Real ? "R" : "C", size(), minSize()), loc);
        }
        computeTwiddles();
    }
    [[nodiscard]] static constexpr std::size_t simdSize() noexcept { return vector_type::size(); }
    [[nodiscard]] static constexpr std::size_t minSize() {
        constexpr std::size_t L = simdSize();
        if constexpr (fftTransform == Transform::Real) {
            return 2UZ * L * L; // min size is N = 32 (SIMD-limit)
        } else {                // transform == Transform::Complex
            return L * L;       // min size is N = 16
        }
    }
    [[nodiscard]] static constexpr bool canProcessSize(std::size_t n, Order ordering = Order::Unordered) {
        if (n < minSize()) {
            return false;
        }

        constexpr std::size_t L     = simdSize();
        constexpr std::size_t N_min = minSize(); /// 16 for complex, 32 for real

        if (ordering == Order::Ordered && fftTransform == Transform::Real) {
            if (n % (2 * L * L) != 0) {
                return false;
            }
        }

        // validation: must reduce exactly to N_min
        std::size_t R = n;
        while (R >= 5 * N_min && (R % 5) == 0) {
            R /= 5;
        }
        while (R >= 3 * N_min && (R % 3) == 0) {
            R /= 3;
        }
        while (R >= 2 * N_min && (R % 2) == 0) {
            R /= 2;
        }

        return (R == N_min); // Must reduce exactly to minimum size!
    }
    [[nodiscard]] constexpr std::size_t simdVectorSize() const noexcept {
        return (IsRealValued::value ? (size() / 2) : size()) / simdSize(); // simdVectorSize = number of complex SIMD vectors (N/4 if complex, N/8 if real for lanes=4)
    }

    [[nodiscard]] std::span<const T> butterflyTwiddles() const noexcept { return _butterflyTwiddles; }
    [[nodiscard]] std::span<const T> stageTwiddles() const noexcept { return _stageTwiddles; }
    [[nodiscard]] std::span<T>       scratch() noexcept { return _scratch; }
    [[nodiscard]] std::span<const T> scratch() const noexcept { return _scratch; }

    void computeTwiddles() {
        // not too performance critical, computed usually only once
        constexpr std::size_t L       = simdSize();
        const std::size_t     nScalar = ceil_div(size(), L);
        if constexpr (IsDynamic::value) {
            constexpr std::size_t kGuard = 2UZ;
            _stageTwiddles.resize(2UZ * nScalar + kGuard);
            _butterflyTwiddles.resize(2UZ * (L - 1UZ) * L * ceil_div(simdVectorSize(), L));
            _scratch.resize(std::max(2UZ * simdVectorSize() * L, 2UZ * size())); // see if this max size can be optimised
        }

        // compute stage twiddles & radix plan
        // radix preference order: Real prefers 4 first, Complex prefers 5
        [[maybe_unused]] constexpr auto radixOrderOriginal = (fftTransform == Transform::Real) ? std::array{4UZ, 2UZ, 3UZ, 5UZ, 0UZ} : std::array{5UZ, 3UZ, 4UZ, 2UZ, 0UZ};
        [[maybe_unused]] constexpr auto radixOrder         = radixOrderOriginal; // N.B. this is potentially a platform-specific tuning parameter

        std::size_t       n         = size() / L;
        const std::size_t numStages = details::decompose<radixOrder>(n, _radixPlan);
        const T           argh      = (2 * std::numbers::pi_v<T>) / static_cast<T>(n);

        std::size_t       twiddlePos = fftTransform == Transform::Real ? 0 : 1;
        std::size_t       l1         = 1;
        const std::size_t loopEnd    = fftTransform == Transform::Real ? (numStages - 1) : numStages;
        for (std::size_t k1 = 1UZ; k1 <= loopEnd; ++k1) {
            const std::size_t radix  = _radixPlan[k1 + 1];
            std::size_t       ld     = 0;
            const std::size_t l2     = l1 * radix;
            const std::size_t stride = n / l2;
            const std::size_t radixm = radix - 1;

            for (std::size_t j = 1; j <= radixm; ++j) {
                ld += l1;
                const T argld = static_cast<T>(ld) * argh;

                if constexpr (fftTransform == Transform::Real) { // simple stride-based indexing
                    std::size_t twiddleIdx = twiddlePos;
                    for (std::size_t fi = 1, ii = 3; ii <= stride; ii += 2, ++fi) {
                        twiddleIdx += 2UZ;
                        _stageTwiddles[twiddleIdx - 2] = std::cos(T(fi) * argld);
                        _stageTwiddles[twiddleIdx - 1] = std::sin(T(fi) * argld);
                    }
                    twiddlePos += stride;
                } else { // fftTransform == Transform::Complex -- more complex indexing with special cases
                    const std::size_t startPos     = twiddlePos;
                    _stageTwiddles[twiddlePos - 1] = 1;
                    _stageTwiddles[twiddlePos]     = 0;

                    const std::size_t complexStride = stride + stride + 2;
                    for (std::size_t fi = 1, ii = 4; ii <= complexStride; ii += 2, ++fi) {
                        twiddlePos += 2;
                        _stageTwiddles[twiddlePos - 1] = std::cos(T(fi) * argld);
                        _stageTwiddles[twiddlePos]     = std::sin(T(fi) * argld);
                    }

                    // special handling for large radices
                    if (radix > 5) {
                        _stageTwiddles[startPos - 1] = _stageTwiddles[twiddlePos - 1];
                        _stageTwiddles[startPos]     = _stageTwiddles[twiddlePos];
                    }
                }
            }
            l1 = l2;
        }

        // butterfly “rotation” scalars in SoA layout
        const std::size_t SV   = simdVectorSize();
        const T           base = (-T{2} * std::numbers::pi_v<T>) / static_cast<T>(size());
        for (std::size_t k = 0UZ; k < SV; ++k) {
            const T           kf = base * static_cast<T>(k);
            const std::size_t i  = k / L;
            const std::size_t j  = k % L;
            for (std::size_t m = 0; m < L - 1; ++m) {
                const T A                                               = kf * static_cast<T>(m + 1);
                _butterflyTwiddles[(2 * (i * (L - 1) + m) + 0) * L + j] = std::cos(A);
                _butterflyTwiddles[(2 * (i * (L - 1) + m) + 1) * L + j] = std::sin(A);
            }
        }

        // factorization check: product(radices) == size()/L
        std::size_t prod = 1;
        for (std::size_t k = 0; k < _radixPlan[1]; ++k) {
            prod *= _radixPlan[2 + k];
        }
        if (prod != nScalar) {
            throw gr::exception(std::format("{} transform (N={}) factorization mismatch: prod(radices={})={} != size()/SIMD_width={} (N={}, SIMD_width={})", //
                fftTransform, size(), _radixPlan, prod, nScalar, size(), vector_type::size()));
        }
    }

    /// @brief Perform forward/backward FFT with optional reordering
    /// @param direction forward or backward tag
    /// @param ordering ordered (canonical interleaved) or unordered (SIMD-optimized, ~20% faster)
    /// @param in Input buffer (real: N samples, complex: 2*N samples)
    /// @param out Output buffer (same size as input), may alias input
    /// @throws gr::exception on size mismatch or unsupported configuration
    ///
    /// @example real-valued transform, unordered (fastest, usually used for convolution):
    /// fft.transform(forward, unordered, input, spectrum);
    ///
    /// @example complex-valued transform, ordered (canonical frequency bins):
    /// fft.transform(forward, ordered, timeDomain, freqDomain)
    void transform(forward_t, ordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Forward, Order::Ordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }
    void transform(backward_t, ordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Backward, Order::Ordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }
    void transform(forward_t, unordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Forward, Order::Unordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }
    void transform(backward_t, unordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Backward, Order::Unordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }

    template<Direction direction, Order ordering, InBuf<T> Rin, OutBuf<T> Rout>
    void transform(Rin&& in, Rout&& out, std::source_location loc = std::source_location::current()) {
        const auto inputSpan  = std::span<const T>(std::ranges::data(in), std::ranges::size(in));
        auto       outputSpan = std::span<T>(std::ranges::data(out), std::ranges::size(out));

        const std::size_t need = (fftTransform == Transform::Real) ? size() : 2 * size();
        if (inputSpan.size() < need || outputSpan.size() < need) {
            throw gr::exception(std::format("size mismatch: input({}) output({}) setup({})", inputSpan.size(), outputSpan.size(), need), loc);
        }
        if (!SimdFFT<T, fftTransform, N>::canProcessSize(size(), ordering)) {
            if constexpr (ordering == Order::Ordered && fftTransform == Transform::Real) {
                throw gr::exception(std::format("{} {} FFT not supported for N={} (requires N % 32 == 0)", ordering, fftTransform, size()), loc);
            } else {
                throw gr::exception(std::format("{} {} FFT with N={} not supported (must factor into {{2,3,4,5}} and >={})", ordering, fftTransform, size(), minSize()), loc);
            }
        }

        if (!details::isAligned<kAlignment>(inputSpan.data())) {
            throw gr::exception(std::format("input is not {}-bytes aligned", kAlignment), loc);
        }
        if (!details::isAligned<kAlignment>(outputSpan.data())) {
            throw gr::exception(std::format("output is not {}-bytes aligned", kAlignment), loc);
        }

        transformInternal<direction, ordering>(inputSpan, outputSpan, scratch());
    }

    /// @brief reorder between SIMD-tiled and canonical interleaved format
    /// @note real-valued transforms: only supported for N % 32 == 0
    /// @note Input and output must not alias
    template<Direction direction>
    constexpr void simdReordering(std::span<const T> input, std::span<T> output) const {
        constexpr std::size_t L     = vector_type::size();
        const std::size_t     Ncvec = simdVectorSize();

        assert(input.data() != output.data());

        if constexpr (IsRealValued::value) {
            assert(canProcessSize(size(), Order::Ordered)); // non-multiple of 32 (while they can be computed) are very hard to bit-reverse
            if constexpr (direction == Direction::Forward) {
                const V* vin  = reinterpret_cast<const V*>(input.data());
                V*       vout = reinterpret_cast<V*>(output.data());

                const std::size_t dk = size() / 32; // For N=48: dk=1

                for (std::size_t k = 0; k < dk; ++k) {
                    V out0_0, out0_1, out2_0, out2_1;

                    // INTERLEAVE2(vin[k*8+0], vin[k*8+1], vout[2*(0*dk+k)+0], vout[2*(0*dk+k)+1])
                    details::interleave(vin[k * 8 + 0], vin[k * 8 + 1], out0_0, out0_1);
                    vout[2 * (0 * dk + k) + 0] = out0_0;
                    vout[2 * (0 * dk + k) + 1] = out0_1;

                    // INTERLEAVE2(vin[k*8+4], vin[k*8+5], vout[2*(2*dk+k)+0], vout[2*(2*dk+k)+1])
                    details::interleave(vin[k * 8 + 4], vin[k * 8 + 5], out2_0, out2_1);
                    vout[2 * (2 * dk + k) + 0] = out2_0;
                    vout[2 * (2 * dk + k) + 1] = out2_1;
                }

                // reversed_copy(dk, vin+2, 8, (v4sf*)(out + N/2))
                details::reversed_copy(dk, vin + 2, 8, reinterpret_cast<V*>(output.data() + size() / 2));

                // reversed_copy(dk, vin+6, 8, (v4sf*)(out + N))
                details::reversed_copy(dk, vin + 6, 8, reinterpret_cast<V*>(output.data() + size()));
            } else { // Backward
                const V* vin  = reinterpret_cast<const V*>(input.data());
                V*       vout = reinterpret_cast<V*>(output.data());

                const std::size_t dk = size() / 32;

                for (std::size_t k = 0; k < dk; ++k) {
                    V out0_0, out0_1, out4_0, out4_1;

                    details::uninterleave(vin[2 * (0 * dk + k) + 0], vin[2 * (0 * dk + k) + 1], out0_0, out0_1);
                    vout[k * 8 + 0] = out0_0;
                    vout[k * 8 + 1] = out0_1;

                    details::uninterleave(vin[2 * (2 * dk + k) + 0], vin[2 * (2 * dk + k) + 1], out4_0, out4_1);
                    vout[k * 8 + 4] = out4_0;
                    vout[k * 8 + 5] = out4_1;
                }

                details::unreversed_copy(dk, reinterpret_cast<const V*>(input.data() + size() / 4), reinterpret_cast<V*>(output.data() + size() - 6 * L), -8);
                details::unreversed_copy(dk, reinterpret_cast<const V*>(input.data() + 3 * size() / 4), reinterpret_cast<V*>(output.data() + size() - 2 * L), -8);
            }
            return;
        }

        // Complex FFT - this part was already correct
        const T* inP  = input.data();
        T*       outP = output.data();

        if constexpr (direction == Direction::Forward) {
            for (std::size_t k = 0; k < Ncvec; ++k) {
                const std::size_t kk = (k / 4) + (k % 4) * (Ncvec / 4);
                V                 lo{}, hi{};
                details::interleave(V(inP + (2 * k + 0) * L, stdx::vector_aligned), V(inP + (2 * k + 1) * L, stdx::vector_aligned), lo, hi);
                details::store_unchecked(lo, outP + (2 * kk + 0) * L, stdx::vector_aligned);
                details::store_unchecked(hi, outP + (2 * kk + 1) * L, stdx::vector_aligned);
            }
        } else {
            for (std::size_t k = 0; k < Ncvec; ++k) {
                const std::size_t kk = (k / 4) + (k % 4) * (Ncvec / 4);
                V                 re{}, im{};
                details::uninterleave(V(inP + (2 * kk + 0) * L, stdx::vector_aligned), V(inP + (2 * kk + 1) * L, stdx::vector_aligned), re, im);
                details::store_unchecked(re, outP + (2 * k + 0) * L, stdx::vector_aligned);
                details::store_unchecked(im, outP + (2 * k + 1) * L, stdx::vector_aligned);
            }
        }
    }

private:
    template<Direction direction, Order ordering>
    constexpr void transformInternal(std::span<const T> inputSpan, std::span<T> outputSpan, std::span<T> scratch) {
        if constexpr (fftTransform == Transform::Real) {
            inputSpan  = std::span{inputSpan.data(), size()};
            outputSpan = std::span{outputSpan.data(), size()};
        }

        std::span<T>  buff[2]       = {outputSpan, scratch};
        constexpr int orderinged    = (ordering == Order::Ordered) ? 1 : 0;
        const int     numStages_odd = _radixPlan[1] & 1;

        std::size_t ib = (numStages_odd ^ orderinged) ? 1 : 0;

        const std::size_t Ncvec = simdVectorSize();
        const std::size_t nVecs = IsRealValued::value ? (Ncvec * 2) : Ncvec;

        // complex-valued FFT inverts the stage direction
        constexpr Direction stagesDir           = (IsRealValued::value == (direction == Direction::Forward)) ? Direction::Forward : Direction::Backward;
        auto                processInterleaving = [&](auto operation, std::span<const T> src, std::span<T> dst) {
            assert(details::isAligned<64>(src.data()));
            assert(details::isAligned<64>(dst.data()));
            const T* RESTRICT pSrc = std::assume_aligned<64>(src.data());
            T* RESTRICT       pDst = std::assume_aligned<64>(dst.data());

            for (std::size_t k = 0UZ; k < Ncvec; ++k) {
                const std::size_t k2 = 2UZ * k * V::size();
                V                 out0, out1;
                operation(V(pSrc + k2, stdx::vector_aligned), V(pSrc + k2 + V::size(), stdx::vector_aligned), out0, out1);
                details::store_unchecked(out0, pDst + k2, stdx::vector_aligned);
                details::store_unchecked(out1, pDst + k2 + V::size(), stdx::vector_aligned);
            }
        };

        if constexpr (direction == Direction::Forward) {
            ib                             = !ib;
            std::span<const T> stagesInput = inputSpan; // default to input
            if constexpr (!IsRealValued::value) {
                processInterleaving(details::uninterleave<V>, inputSpan, buff[ib]);
                stagesInput = buff[ib];
            }
            std::span<T> outp = details::fftStages<stagesDir, fftTransform, T>(nVecs, stagesInput, buff[ib], buff[!ib], stageTwiddles(), _radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            // finalise: butterfly twiddle application
            if constexpr (IsRealValued::value) {
                details::realFinalise<T>(Ncvec, buff[ib], buff[!ib], butterflyTwiddles());
            } else {
                details::complexFinalise(Ncvec, buff[ib], buff[!ib], butterflyTwiddles());
            }

            if constexpr (ordering == Order::Ordered) {
                simdReordering<Direction::Forward>(std::span<const T>{buff[!ib]}, buff[ib]);
            } else {
                ib = !ib;
            }

        } else { // Direction::Backward
            if (inputSpan.data() == buff[ib].data()) {
                ib = !ib;
            }

            if constexpr (ordering == Order::Ordered) {
                simdReordering<Direction::Backward>(inputSpan, buff[ib]);
                inputSpan = buff[ib];
                ib        = !ib;
            }

            // preprocess: inverse butterfly twiddles
            if constexpr (IsRealValued::value) {
                details::realPreprocess<T>(Ncvec, inputSpan, buff[ib], butterflyTwiddles());
            } else {
                details::complexPreprocess(Ncvec, inputSpan, buff[ib], butterflyTwiddles());
            }

            std::span<T> outp = details::fftStages<stagesDir, fftTransform, T>(nVecs, buff[ib], buff[0], buff[1], stageTwiddles(), _radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            if constexpr (!IsRealValued::value) {
                processInterleaving(details::interleave<V>, buff[ib], buff[ib]);
            }
        }

        if (buff[ib].data() != outputSpan.data()) { // final copy -- only if needed
            std::memcpy(std::assume_aligned<64>(outputSpan.data()), std::assume_aligned<64>(buff[ib].data()), Ncvec * 2 * V::size() * sizeof(T));
            ib = !ib;
        }
        assert(buff[ib].data() == outputSpan.data());
    }
};

namespace details {

/**************************************************************************************************************/
/********************************** complex FFT processing C2C ************************************************/
/**************************************************************************************************************/

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix2(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles) {
    using V                    = vec<T, 4>;
    constexpr std::size_t L    = V::size();
    constexpr T           sign = direction == Direction::Forward ? T{1} : T{-1};

    const std::size_t totalStride = nGroups * stride;
    assert(input.size() >= 2UZ * totalStride);
    assert(output.size() >= 2UZ * totalStride);
    assert(isAligned<64UZ>(input.data()));
    assert(isAligned<64UZ>(output.data()));
    assert(stride >= 2UZ && "Radix-2 requires stride >= 2 for SIMD path");

    const T* RESTRICT pInput  = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput = std::assume_aligned<64>(output.data());
    if (stride <= 2UZ) { // fast path: no twiddle factors needed
        for (std::size_t k = 0UZ; k < totalStride; k += stride) {
            // butterfly: [in0, in1] + [in2, in3] and [in0, in1] - [in2, in3]
            const size_t k2 = k * 2;
            V            re0(pInput + (k2 + 0) * L, stdx::vector_aligned);
            V            im0(pInput + (k2 + 1) * L, stdx::vector_aligned);
            V            re1(pInput + (k2 + stride) * L, stdx::vector_aligned);
            V            im1(pInput + (k2 + stride + 1) * L, stdx::vector_aligned);

            (re0 + re1).copy_to(pOutput + (k + 0) * L, stdx::vector_aligned);
            (im0 + im1).copy_to(pOutput + (k + 1) * L, stdx::vector_aligned);
            (re0 - re1).copy_to(pOutput + (k + 0 + totalStride) * L, stdx::vector_aligned);
            (im0 - im1).copy_to(pOutput + (k + 1 + totalStride) * L, stdx::vector_aligned);
        }
    } else { // general path: apply twiddle factor rotation for each element within stride
        assert(isAligned<64UZ>(twiddles.data()));
        const T* RESTRICT pTwiddle = std::assume_aligned<64UZ>(twiddles.data());
        for (std::size_t k = 0UZ; k < totalStride; k += stride) {
            for (std::size_t i = 0UZ; i < stride - 1UZ; i += 2UZ) { // N.B. +2 for Re/Im
                std::size_t idxIn  = (k * 2 + i) * L;
                std::size_t idxOut = (k + i) * L;

                V re0(pInput + idxIn, stdx::vector_aligned);
                V im0(pInput + idxIn + L, stdx::vector_aligned);
                V re1(pInput + idxIn + stride * L, stdx::vector_aligned);
                V im1(pInput + idxIn + stride * L + L, stdx::vector_aligned);

                V diffRe = re0 - re1;
                V diffIm = im0 - im1;
                V wr(pTwiddle[i]);
                V wi = T{sign} * V(pTwiddle[i + 1]);

                (re0 + re1).copy_to(pOutput + idxOut, stdx::vector_aligned);
                (im0 + im1).copy_to(pOutput + idxOut + L, stdx::vector_aligned);

                complex_multiply(diffRe, diffIm, wr, wi); // apply phase rotation: (tr2 + i*ti2) * (wr + i*wi)
                diffRe.copy_to(pOutput + idxOut + totalStride * L, stdx::vector_aligned);
                diffIm.copy_to(pOutput + idxOut + totalStride * L + L, stdx::vector_aligned);
            }
        }
    }
}

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix3(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2) {
    using V                      = vec<T, 4>;
    constexpr std::size_t L      = V::size();
    constexpr T           sign   = direction == Direction::Forward ? T{1} : T{-1};
    constexpr T           cos120 = T{-0.5};                                  // cos(2π/3)
    constexpr T           sin120 = T{0.5} * std::numbers::sqrt3_v<T> * sign; // ±sin(2π/3)

    const std::size_t totalStride = nGroups * stride;
    assert(input.size() >= 2UZ * totalStride);
    assert(output.size() >= 2UZ * totalStride);
    assert(isAligned<64UZ>(input.data()));
    assert(isAligned<64UZ>(output.data()));
    assert(stride >= 3UZ && "Radix-3 requires stride >= 3 for SIMD path");

    const T* RESTRICT pInput  = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput = std::assume_aligned<64>(output.data());

    for (std::size_t k = 0UZ; k < totalStride; k += stride) {
        const std::size_t k3     = k * 3UZ; // input:  3 values per group
        const std::size_t idxIn  = k3 * L;
        const std::size_t idxOut = k * L;

        for (std::size_t i = 0UZ; i < stride - 1UZ; i += 2UZ) { // N.B. +2 for Re/Im
            // load 3 complex inputs
            V re0(pInput + idxIn + i * L, stdx::vector_aligned);
            V im0(pInput + idxIn + (i + 1) * L, stdx::vector_aligned);
            V re1(pInput + idxIn + (i + stride) * L, stdx::vector_aligned);
            V im1(pInput + idxIn + (i + stride + 1) * L, stdx::vector_aligned);
            V re2(pInput + idxIn + (i + 2 * stride) * L, stdx::vector_aligned);
            V im2(pInput + idxIn + (i + 2 * stride + 1) * L, stdx::vector_aligned);

            // Radix-3 butterfly: sum and difference of inputs 1 & 2
            const V sumRe = re1 + re2;
            const V sumIm = im1 + im2;

            // output[0]: DC component (sum of all inputs)
            (re0 + sumRe).copy_to(pOutput + idxOut + i * L, stdx::vector_aligned);
            (im0 + sumIm).copy_to(pOutput + idxOut + (i + 1) * L, stdx::vector_aligned);

            // apply 120° rotation for outputs 1 & 2
            const V rotSumRe = re0 + cos120 * sumRe;
            const V rotSumIm = im0 + cos120 * sumIm;

            const V scaledDiffRe = sin120 * (re1 - re2);
            const V scaledDiffIm = sin120 * (im1 - im2);

            V out1Re = rotSumRe - scaledDiffIm; // +120° phase shift
            V out1Im = rotSumIm + scaledDiffRe;
            V out2Re = rotSumRe + scaledDiffIm; // -120° phase shift
            V out2Im = rotSumIm - scaledDiffRe;

            const T wr1 = twiddles1[i], wi1 = sign * twiddles1[i + 1];
            const T wr2 = twiddles2[i], wi2 = sign * twiddles2[i + 1];

            complex_multiply(out1Re, out1Im, V(wr1), V(wi1));
            out1Re.copy_to(pOutput + idxOut + (i + totalStride) * L, stdx::vector_aligned);
            out1Im.copy_to(pOutput + idxOut + (i + totalStride + 1) * L, stdx::vector_aligned);

            complex_multiply(out2Re, out2Im, V(wr2), V(wi2));
            out2Re.copy_to(pOutput + idxOut + (i + 2 * totalStride) * L, stdx::vector_aligned);
            out2Im.copy_to(pOutput + idxOut + (i + 2 * totalStride + 1) * L, stdx::vector_aligned);
        }
    }
}

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix4(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3) {
    using V                    = vec<T, 4>;
    constexpr std::size_t L    = V::size();
    constexpr T           sign = direction == Direction::Forward ? T{1} : T{-1};

    const std::size_t totalStride = nGroups * stride;
    assert(input.size() >= 2UZ * totalStride);
    assert(output.size() >= 2UZ * totalStride);
    assert(isAligned<64UZ>(input.data()));
    assert(isAligned<64UZ>(output.data()));
    assert(stride >= 2UZ && "Radix-4 requires stride >= 2 for SIMD path");

    const T* RESTRICT pInput  = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput = std::assume_aligned<64>(output.data());

    if (stride == 2UZ) { // fast path: no twiddle factors needed
        for (std::size_t k = 0UZ; k < totalStride; k += stride) {
            const std::size_t k4     = k * 4UZ; // input: 4 values per group
            const std::size_t idxIn  = k4 * L;
            const std::size_t idxOut = k * L;

            // load 4 complex inputs
            V re0(pInput + idxIn, stdx::vector_aligned);
            V im0(pInput + idxIn + L, stdx::vector_aligned);
            V re1(pInput + idxIn + stride * L, stdx::vector_aligned);
            V im1(pInput + idxIn + (stride + 1) * L, stdx::vector_aligned);
            V re2(pInput + idxIn + 2 * stride * L, stdx::vector_aligned);
            V im2(pInput + idxIn + (2 * stride + 1) * L, stdx::vector_aligned);
            V re3(pInput + idxIn + 3 * stride * L, stdx::vector_aligned);
            V im3(pInput + idxIn + (3 * stride + 1) * L, stdx::vector_aligned);

            // Radix-4 butterfly with 90° rotations
            const V sumRe_02  = re0 + re2;
            const V diffRe_02 = re0 - re2;
            const V sumIm_02  = im0 + im2;
            const V diffIm_02 = im0 - im2;

            const V sumRe_13  = re1 + re3;
            const V diffRe_13 = (im3 - im1) * sign; // 90° rotation: ±i(in1 - in3)
            const V sumIm_13  = im1 + im3;
            const V diffIm_13 = (re1 - re3) * sign; // 90° rotation

            // store 4 outputs
            store_unchecked(sumRe_02 + sumRe_13, pOutput + idxOut, stdx::vector_aligned);
            store_unchecked(sumIm_02 + sumIm_13, pOutput + idxOut + L, stdx::vector_aligned);
            store_unchecked(diffRe_02 + diffRe_13, pOutput + idxOut + totalStride * L, stdx::vector_aligned);
            store_unchecked(diffIm_02 + diffIm_13, pOutput + idxOut + (totalStride + 1) * L, stdx::vector_aligned);
            store_unchecked(sumRe_02 - sumRe_13, pOutput + idxOut + 2 * totalStride * L, stdx::vector_aligned);
            store_unchecked(sumIm_02 - sumIm_13, pOutput + idxOut + (2 * totalStride + 1) * L, stdx::vector_aligned);
            store_unchecked(diffRe_02 - diffRe_13, pOutput + idxOut + 3 * totalStride * L, stdx::vector_aligned);
            store_unchecked(diffIm_02 - diffIm_13, pOutput + idxOut + (3 * totalStride + 1) * L, stdx::vector_aligned);
        }
    } else { // general path: apply twiddle factors
        for (std::size_t k = 0UZ; k < totalStride; k += stride) {
            const std::size_t k4     = k * 4UZ;
            const std::size_t idxIn  = k4 * L;
            const std::size_t idxOut = k * L;

            for (std::size_t i = 0UZ; i < stride - 1UZ; i += 2UZ) { // N.B. +2 for Re/Im
                // load 4 complex inputs
                V re0(pInput + idxIn + i * L, stdx::vector_aligned);
                V im0(pInput + idxIn + (i + 1) * L, stdx::vector_aligned);
                V re1(pInput + idxIn + (i + stride) * L, stdx::vector_aligned);
                V im1(pInput + idxIn + (i + stride + 1) * L, stdx::vector_aligned);
                V re2(pInput + idxIn + (i + 2 * stride) * L, stdx::vector_aligned);
                V im2(pInput + idxIn + (i + 2 * stride + 1) * L, stdx::vector_aligned);
                V re3(pInput + idxIn + (i + 3 * stride) * L, stdx::vector_aligned);
                V im3(pInput + idxIn + (i + 3 * stride + 1) * L, stdx::vector_aligned);

                // Radix-4 butterfly with 90° rotations
                const V sumRe_02  = re0 + re2;
                const V diffRe_02 = re0 - re2;
                const V sumIm_02  = im0 + im2;
                const V diffIm_02 = im0 - im2;

                const V sumRe_13  = re1 + re3;
                const V diffRe_13 = (im3 - im1) * sign; // 90° rotation: ±i(in1 - in3)
                const V sumIm_13  = im1 + im3;
                const V diffIm_13 = (re1 - re3) * sign; // 90° rotation

                // output[0]: DC component (no twiddle needed)
                store_unchecked(sumRe_02 + sumRe_13, pOutput + idxOut + i * L, stdx::vector_aligned);
                store_unchecked(sumIm_02 + sumIm_13, pOutput + idxOut + (i + 1) * L, stdx::vector_aligned);

                // prepare outputs 1-3 for twiddle application
                V out1Re = diffRe_02 + diffRe_13;
                V out1Im = diffIm_02 + diffIm_13;
                V out2Re = sumRe_02 - sumRe_13;
                V out2Im = sumIm_02 - sumIm_13;
                V out3Re = diffRe_02 - diffRe_13;
                V out3Im = diffIm_02 - diffIm_13;

                const T wr1 = twiddles1[i], wi1 = sign * twiddles1[i + 1];
                const T wr2 = twiddles2[i], wi2 = sign * twiddles2[i + 1];
                const T wr3 = twiddles3[i], wi3 = sign * twiddles3[i + 1];

                complex_multiply(out1Re, out1Im, V(wr1), V(wi1));
                complex_multiply(out2Re, out2Im, V(wr2), V(wi2));
                complex_multiply(out3Re, out3Im, V(wr3), V(wi3));

                store_unchecked(out1Re, pOutput + idxOut + (i + totalStride) * L, stdx::vector_aligned);
                store_unchecked(out1Im, pOutput + idxOut + (i + totalStride + 1) * L, stdx::vector_aligned);
                store_unchecked(out2Re, pOutput + idxOut + (i + 2 * totalStride) * L, stdx::vector_aligned);
                store_unchecked(out2Im, pOutput + idxOut + (i + 2 * totalStride + 1) * L, stdx::vector_aligned);
                store_unchecked(out3Re, pOutput + idxOut + (i + 3 * totalStride) * L, stdx::vector_aligned);
                store_unchecked(out3Im, pOutput + idxOut + (i + 3 * totalStride + 1) * L, stdx::vector_aligned);
            }
        }
    }
}

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) complexRadix5(std::size_t stride, std::size_t nGroups, std::span<const T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3, std::span<const T> twiddles4) {
    using V                      = vec<T, 4>;
    constexpr std::size_t L      = V::size();
    constexpr T           sign   = direction == Direction::Forward ? T{1} : T{-1};
    constexpr T           cos72  = static_cast<T>(0.3090169943749474241022934171828191L);        // cos(2π/5)
    constexpr T           sin72  = static_cast<T>(0.9510565162951535721164393333793821L) * sign; // ±sin(2π/5)
    constexpr T           cos144 = static_cast<T>(-0.8090169943749474241022934171828191L);       // cos(4π/5)
    constexpr T           sin144 = static_cast<T>(0.5877852522924731291687059546390728L) * sign; // ±sin(4π/5)

    const std::size_t totalStride = nGroups * stride;
    assert(input.size() >= 2UZ * totalStride);
    assert(output.size() >= 2UZ * totalStride);
    assert(isAligned<64UZ>(input.data()));
    assert(isAligned<64UZ>(output.data()));
    assert(stride > 2UZ && "Radix-5 requires stride > 2 for SIMD path");

    const T* RESTRICT pInput  = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput = std::assume_aligned<64>(output.data());

    for (std::size_t k = 0UZ; k < nGroups; ++k) {
        const std::size_t k5     = k * 5UZ; // input: 5 values per group
        const std::size_t idxIn  = k5 * stride * L;
        const std::size_t idxOut = k * stride * L;

        for (std::size_t i = 0UZ; i < stride - 1UZ; i += 2UZ) { // N.B. +2 for Re/Im
            // load 5 complex inputs
            V re0(pInput + idxIn + i * L, stdx::vector_aligned);
            V im0(pInput + idxIn + (i + 1) * L, stdx::vector_aligned);
            V re1(pInput + idxIn + (i + stride) * L, stdx::vector_aligned);
            V im1(pInput + idxIn + (i + stride + 1) * L, stdx::vector_aligned);
            V re2(pInput + idxIn + (i + 2 * stride) * L, stdx::vector_aligned);
            V im2(pInput + idxIn + (i + 2 * stride + 1) * L, stdx::vector_aligned);
            V re3(pInput + idxIn + (i + 3 * stride) * L, stdx::vector_aligned);
            V im3(pInput + idxIn + (i + 3 * stride + 1) * L, stdx::vector_aligned);
            V re4(pInput + idxIn + (i + 4 * stride) * L, stdx::vector_aligned);
            V im4(pInput + idxIn + (i + 4 * stride + 1) * L, stdx::vector_aligned);

            // Radix-5 butterfly: sums and differences with conjugate symmetry
            const V sumRe_14  = re1 + re4; // inputs 1 & 4 (conjugate pair)
            const V diffRe_14 = re1 - re4;
            const V sumIm_14  = im1 + im4;
            const V diffIm_14 = im1 - im4;

            const V sumRe_23  = re2 + re3; // inputs 2 & 3 (conjugate pair)
            const V diffRe_23 = re2 - re3;
            const V sumIm_23  = im2 + im3;
            const V diffIm_23 = im2 - im3;

            // output[0]: DC component (sum of all inputs)
            store_unchecked(re0 + (sumRe_14 + sumRe_23), pOutput + idxOut + i * L, stdx::vector_aligned);
            store_unchecked(im0 + (sumIm_14 + sumIm_23), pOutput + idxOut + (i + 1) * L, stdx::vector_aligned);

            // apply 72° and 144° rotations
            const V rot72Re  = re0 + (cos72 * sumRe_14 + cos144 * sumRe_23);
            const V rot72Im  = im0 + (cos72 * sumIm_14 + cos144 * sumIm_23);
            const V rot144Re = re0 + (cos144 * sumRe_14 + cos72 * sumRe_23);
            const V rot144Im = im0 + (cos144 * sumIm_14 + cos72 * sumIm_23);

            const V cross72Re  = (sin72 * diffRe_14) + (sin144 * diffRe_23);
            const V cross72Im  = (sin72 * diffIm_14) + (sin144 * diffIm_23);
            const V cross144Re = (sin144 * diffRe_14) - (sin72 * diffRe_23);
            const V cross144Im = (sin144 * diffIm_14) - (sin72 * diffIm_23);

            // combine rotations to form outputs 1-4
            V out1Re = rot72Re - cross72Im; // 72° phase
            V out1Im = rot72Im + cross72Re;
            V out2Re = rot144Re - cross144Im; // 144° phase
            V out2Im = rot144Im + cross144Re;
            V out3Re = rot144Re + cross144Im; // 216° phase
            V out3Im = rot144Im - cross144Re;
            V out4Re = rot72Re + cross72Im; // 288° phase
            V out4Im = rot72Im - cross72Re;

            const T wr1 = twiddles1[i], wi1 = sign * twiddles1[i + 1];
            const T wr2 = twiddles2[i], wi2 = sign * twiddles2[i + 1];
            const T wr3 = twiddles3[i], wi3 = sign * twiddles3[i + 1];
            const T wr4 = twiddles4[i], wi4 = sign * twiddles4[i + 1];

            complex_multiply(out1Re, out1Im, V(wr1), V(wi1));
            store_unchecked(out1Re, pOutput + idxOut + (i + totalStride) * L, stdx::vector_aligned);
            store_unchecked(out1Im, pOutput + idxOut + (i + totalStride + 1) * L, stdx::vector_aligned);

            complex_multiply(out2Re, out2Im, V(wr2), V(wi2));
            store_unchecked(out2Re, pOutput + idxOut + (i + 2 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(out2Im, pOutput + idxOut + (i + 2 * totalStride + 1) * L, stdx::vector_aligned);

            complex_multiply(out3Re, out3Im, V(wr3), V(wi3));
            store_unchecked(out3Re, pOutput + idxOut + (i + 3 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(out3Im, pOutput + idxOut + (i + 3 * totalStride + 1) * L, stdx::vector_aligned);

            complex_multiply(out4Re, out4Im, V(wr4), V(wi4));
            store_unchecked(out4Re, pOutput + idxOut + (i + 4 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(out4Im, pOutput + idxOut + (i + 4 * totalStride + 1) * L, stdx::vector_aligned);
        }
    }
}

template<std::floating_point T>
constexpr void complexFinalise(std::size_t Ncvec, std::span<T> input, std::span<T> output, std::span<const T> butterflyTwiddles) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    assert(input.data() != output.data());
    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(output.data()));
    assert(isAligned<64>(butterflyTwiddles.data()));

    const T* RESTRICT pInput   = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput  = std::assume_aligned<64>(output.data());
    const T* RESTRICT pTwiddle = std::assume_aligned<64>(butterflyTwiddles.data());

    const std::size_t nBlocks = Ncvec / L; // number of 4×4 matrix blocks

    for (std::size_t block = 0; block < nBlocks; ++block) {
        const std::size_t baseIdx = 8 * block * L;

        // load 4 complex values (8 SIMD vectors)
        V re0(pInput + baseIdx, stdx::vector_aligned);
        V im0(pInput + baseIdx + L, stdx::vector_aligned);
        V re1(pInput + baseIdx + 2 * L, stdx::vector_aligned);
        V im1(pInput + baseIdx + 3 * L, stdx::vector_aligned);
        V re2(pInput + baseIdx + 4 * L, stdx::vector_aligned);
        V im2(pInput + baseIdx + 5 * L, stdx::vector_aligned);
        V re3(pInput + baseIdx + 6 * L, stdx::vector_aligned);
        V im3(pInput + baseIdx + 7 * L, stdx::vector_aligned);

        // transpose 4×4 matrices (convert between SoA layouts)
        transpose(re0, re1, re2, re3);
        transpose(im0, im1, im2, im3);

        // load butterfly twiddle factors (3 complex = 6 real values)
        const std::size_t twiddleBase = block * 6 * L;
        V                 twiddleRe0(pTwiddle + twiddleBase, stdx::vector_aligned);
        V                 twiddleIm0(pTwiddle + twiddleBase + L, stdx::vector_aligned);
        V                 twiddleRe1(pTwiddle + twiddleBase + 2 * L, stdx::vector_aligned);
        V                 twiddleIm1(pTwiddle + twiddleBase + 3 * L, stdx::vector_aligned);
        V                 twiddleRe2(pTwiddle + twiddleBase + 4 * L, stdx::vector_aligned);
        V                 twiddleIm2(pTwiddle + twiddleBase + 5 * L, stdx::vector_aligned);

        // apply twiddle rotations to values 1-3 (value 0 unchanged)
        complex_multiply(re1, im1, twiddleRe0, twiddleIm0);
        complex_multiply(re2, im2, twiddleRe1, twiddleIm1);
        complex_multiply(re3, im3, twiddleRe2, twiddleIm2);

        // Radix-4 butterfly (similar to complexRadix4)
        const V sumRe_02  = re0 + re2;
        const V diffRe_02 = re0 - re2;
        const V sumRe_13  = re1 + re3;
        const V diffRe_13 = re1 - re3;

        const V sumIm_02  = im0 + im2;
        const V diffIm_02 = im0 - im2;
        const V sumIm_13  = im1 + im3;
        const V diffIm_13 = im1 - im3;

        re0 = sumRe_02 + sumRe_13;
        im0 = sumIm_02 + sumIm_13;
        re1 = diffRe_02 + diffIm_13; // 90° rotation
        im1 = diffIm_02 - diffRe_13;
        re2 = sumRe_02 - sumRe_13;
        im2 = sumIm_02 - sumIm_13;
        re3 = diffRe_02 - diffIm_13; // 90° rotation
        im3 = diffIm_02 + diffRe_13;

        // store 4 complex outputs (8 SIMD vectors)
        store_unchecked(re0, pOutput + baseIdx, stdx::vector_aligned);
        store_unchecked(im0, pOutput + baseIdx + L, stdx::vector_aligned);
        store_unchecked(re1, pOutput + baseIdx + 2 * L, stdx::vector_aligned);
        store_unchecked(im1, pOutput + baseIdx + 3 * L, stdx::vector_aligned);
        store_unchecked(re2, pOutput + baseIdx + 4 * L, stdx::vector_aligned);
        store_unchecked(im2, pOutput + baseIdx + 5 * L, stdx::vector_aligned);
        store_unchecked(re3, pOutput + baseIdx + 6 * L, stdx::vector_aligned);
        store_unchecked(im3, pOutput + baseIdx + 7 * L, stdx::vector_aligned);
    }
}

template<std::floating_point T>
constexpr void complexPreprocess(std::size_t Ncvec, std::span<const T> input, std::span<T> output, std::span<const T> butterflyTwiddles) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    assert(input.data() != output.data());
    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(output.data()));
    assert(isAligned<64>(butterflyTwiddles.data()));

    const T* RESTRICT pInput   = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput  = std::assume_aligned<64>(output.data());
    const T* RESTRICT pTwiddle = std::assume_aligned<64>(butterflyTwiddles.data());

    const std::size_t nBlocks = Ncvec / L; // number of 4×4 matrix blocks

    for (std::size_t block = 0; block < nBlocks; ++block) {
        const std::size_t baseIdx = 8 * block * L;

        // load 4 complex values (8 SIMD vectors)
        V re0(pInput + baseIdx, stdx::vector_aligned);
        V im0(pInput + baseIdx + L, stdx::vector_aligned);
        V re1(pInput + baseIdx + 2 * L, stdx::vector_aligned);
        V im1(pInput + baseIdx + 3 * L, stdx::vector_aligned);
        V re2(pInput + baseIdx + 4 * L, stdx::vector_aligned);
        V im2(pInput + baseIdx + 5 * L, stdx::vector_aligned);
        V re3(pInput + baseIdx + 6 * L, stdx::vector_aligned);
        V im3(pInput + baseIdx + 7 * L, stdx::vector_aligned);

        // inverse radix-4 butterfly (reverse of complexFinalise)
        const V sumRe_02  = re0 + re2;
        const V diffRe_02 = re0 - re2;
        const V sumRe_13  = re1 + re3;
        const V diffRe_13 = re1 - re3;

        const V sumIm_02  = im0 + im2;
        const V diffIm_02 = im0 - im2;
        const V sumIm_13  = im1 + im3;
        const V diffIm_13 = im1 - im3;

        re0 = sumRe_02 + sumRe_13;
        im0 = sumIm_02 + sumIm_13;
        re1 = diffRe_02 - diffIm_13; // 90° rotation (inverse)
        im1 = diffIm_02 + diffRe_13;
        re2 = sumRe_02 - sumRe_13;
        im2 = sumIm_02 - sumIm_13;
        re3 = diffRe_02 + diffIm_13; // 90° rotation (inverse)
        im3 = diffIm_02 - diffRe_13;

        // load butterfly twiddle factors (3 complex = 6 real values)
        const std::size_t twiddleBase = block * 6 * L;
        V                 twiddleRe0(pTwiddle + twiddleBase, stdx::vector_aligned);
        V                 twiddleIm0(pTwiddle + twiddleBase + L, stdx::vector_aligned);
        V                 twiddleRe1(pTwiddle + twiddleBase + 2 * L, stdx::vector_aligned);
        V                 twiddleIm1(pTwiddle + twiddleBase + 3 * L, stdx::vector_aligned);
        V                 twiddleRe2(pTwiddle + twiddleBase + 4 * L, stdx::vector_aligned);
        V                 twiddleIm2(pTwiddle + twiddleBase + 5 * L, stdx::vector_aligned);

        // apply conjugate twiddle rotations to values 1-3 (value 0 unchanged)
        complex_multiply_conj(re1, im1, twiddleRe0, twiddleIm0);
        complex_multiply_conj(re2, im2, twiddleRe1, twiddleIm1);
        complex_multiply_conj(re3, im3, twiddleRe2, twiddleIm2);

        // transpose 4×4 matrices (convert between SoA layouts)
        transpose(re0, re1, re2, re3);
        transpose(im0, im1, im2, im3);

        // store 4 complex outputs (8 SIMD vectors)
        store_unchecked(re0, pOutput + baseIdx, stdx::vector_aligned);
        store_unchecked(im0, pOutput + baseIdx + L, stdx::vector_aligned);
        store_unchecked(re1, pOutput + baseIdx + 2 * L, stdx::vector_aligned);
        store_unchecked(im1, pOutput + baseIdx + 3 * L, stdx::vector_aligned);
        store_unchecked(re2, pOutput + baseIdx + 4 * L, stdx::vector_aligned);
        store_unchecked(im2, pOutput + baseIdx + 5 * L, stdx::vector_aligned);
        store_unchecked(re3, pOutput + baseIdx + 6 * L, stdx::vector_aligned);
        store_unchecked(im3, pOutput + baseIdx + 7 * L, stdx::vector_aligned);
    }
}

/**************************************************************************************************************/
/********************************** real-valued FFT R2C & C2R *************************************************/
/**************************************************************************************************************/
/********************************** danger territory starts here **********************************************/

template<Direction dir, std::floating_point T>
static NEVER_INLINE(void) realRadix2(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    const std::size_t totalStride = nGroups * stride;

    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(output.data()));

    T* RESTRICT       pInput   = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput  = std::assume_aligned<64>(output.data());
    const T* RESTRICT pTwiddle = std::assume_aligned<64>(twiddles.data());

    // direction-specific complex multiply (Forward: conjugate, Backward: normal)
    constexpr auto applyTwiddle = []<typename Vec, typename Scalar>(Vec& re, Vec& im, const Scalar& wr, const Scalar& wi) {
        if constexpr (dir == Direction::Forward) {
            complex_multiply_conj(re, im, wr, wi);
        } else {
            complex_multiply(re, im, wr, wi);
        }
    };

    // pass 1: DC components (i=0)
    for (std::size_t k = 0; k < totalStride; k += stride) {
        if constexpr (dir == Direction::Forward) { // Real → Hermitian
            V val0(pInput + k * L, stdx::vector_aligned);
            V val1(pInput + (k + totalStride) * L, stdx::vector_aligned);

            store_unchecked(val0 + val1, pOutput + 2 * k * L, stdx::vector_aligned);
            store_unchecked(val0 - val1, pOutput + (2 * (k + stride) - 1) * L, stdx::vector_aligned);
        } else { // Hermitian → Real
            V val0(pInput + 2 * k * L, stdx::vector_aligned);
            V val1(pInput + (2 * (k + stride) - 1) * L, stdx::vector_aligned);

            store_unchecked(val0 + val1, pOutput + k * L, stdx::vector_aligned);
            store_unchecked(val0 - val1, pOutput + (k + totalStride) * L, stdx::vector_aligned);
        }
    }

    if (stride < 2) {
        return;
    }

    // pass 2: general frequencies (0 < i < stride/2)
    if (stride > 2) {
        for (std::size_t k = 0; k < totalStride; k += stride) {
            for (std::size_t i = 2; i < stride; i += 2) {
                if constexpr (dir == Direction::Forward) { // Real → Hermitian
                    V re0(pInput + (i - 1 + k) * L, stdx::vector_aligned);
                    V im0(pInput + (i + k) * L, stdx::vector_aligned);
                    V re1(pInput + (i - 1 + k + totalStride) * L, stdx::vector_aligned);
                    V im1(pInput + (i + k + totalStride) * L, stdx::vector_aligned);

                    // apply conjugate twiddle to the second input
                    applyTwiddle(re1, im1, V(pTwiddle[i - 2]), V(pTwiddle[i - 1]));

                    // store Hermitian-symmetric pairs: X[i] and X[-i] = X*[i]
                    store_unchecked(im0 + im1, pOutput + (i + 2 * k) * L, stdx::vector_aligned);
                    store_unchecked(im1 - im0, pOutput + (2 * (k + stride) - i) * L, stdx::vector_aligned);
                    store_unchecked(re0 + re1, pOutput + (i - 1 + 2 * k) * L, stdx::vector_aligned);
                    store_unchecked(re0 - re1, pOutput + (2 * (k + stride) - i - 1) * L, stdx::vector_aligned);
                } else { // Hermitian → Real
                    V re_pos(pInput + (i - 1 + 2 * k) * L, stdx::vector_aligned);
                    V re_neg(pInput + (2 * (k + stride) - i - 1) * L, stdx::vector_aligned);
                    V im_pos(pInput + (i + 2 * k) * L, stdx::vector_aligned);
                    V im_neg(pInput + (2 * (k + stride) - i) * L, stdx::vector_aligned);

                    // unpack Hermitian pairs
                    V re0 = re_pos + re_neg;
                    V re1 = re_pos - re_neg;
                    V im0 = im_pos - im_neg;
                    V im1 = im_pos + im_neg;

                    store_unchecked(re0, pOutput + (i - 1 + k) * L, stdx::vector_aligned);
                    store_unchecked(im0, pOutput + (i + k) * L, stdx::vector_aligned);

                    // apply twiddle to the second output
                    applyTwiddle(re1, im1, V(pTwiddle[i - 2]), V(pTwiddle[i - 1]));

                    store_unchecked(re1, pOutput + (i - 1 + k + totalStride) * L, stdx::vector_aligned);
                    store_unchecked(im1, pOutput + (i + k + totalStride) * L, stdx::vector_aligned);
                }
            }
        }
        if (stride % 2 == 1) {
            return;
        }
    }

    // pass 3: Nyquist frequency (i = stride/2, only when stride is even) ===
    // Nyquist is real-valued for real signals, requires special handling
    for (std::size_t k = 0; k < totalStride; k += stride) {
        if constexpr (dir == Direction::Forward) { // Real → Hermitian: apply π phase shift
            V nyquist0(pInput + (k + stride - 1) * L, stdx::vector_aligned);
            V nyquist1(pInput + (k + stride - 1 + totalStride) * L, stdx::vector_aligned);

            store_unchecked(nyquist0, pOutput + (2 * k + stride - 1) * L, stdx::vector_aligned);
            store_unchecked(T{-1} * nyquist1, pOutput + (2 * k + stride) * L, stdx::vector_aligned);
        } else { // Hermitian → Real: scale by 2 (unpack)
            V nyquist0(pInput + (2 * k + stride - 1) * L, stdx::vector_aligned);
            V nyquist1(pInput + (2 * k + stride) * L, stdx::vector_aligned);

            store_unchecked(nyquist0 + nyquist0, pOutput + (k + stride - 1) * L, stdx::vector_aligned);
            store_unchecked(T{-2} * nyquist1, pOutput + (k + stride - 1 + totalStride) * L, stdx::vector_aligned);
        }
    }
}

template<Direction dir, std::floating_point T>
static NEVER_INLINE(void) realRadix3(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    constexpr T sign   = dir == Direction::Forward ? T{1} : T{-1};
    constexpr T cos120 = T{-0.5};
    constexpr T sin120 = T{0.5} * std::numbers::sqrt3_v<T> * sign;

    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(output.data()));

    T* RESTRICT       pInput    = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput   = std::assume_aligned<64>(output.data());
    const T* RESTRICT pTwiddle1 = twiddles1.data();
    const T* RESTRICT pTwiddle2 = twiddles2.data();

    // direction-specific complex multiply
    constexpr auto applyTwiddle = []<typename Vec, typename Scalar>(Vec& re, Vec& im, const Scalar& wr, const Scalar& wi) {
        if constexpr (dir == Direction::Forward) {
            complex_multiply_conj(re, im, wr, wi);
        } else {
            complex_multiply(re, im, wr, wi);
        }
    };

    // pass 1: DC components (i=0)
    for (std::size_t k = 0; k < nGroups; ++k) {
        if constexpr (dir == Direction::Forward) { // Real → Hermitian
            V val0(pInput + (k * stride) * L, stdx::vector_aligned);
            V val1(pInput + ((k + nGroups) * stride) * L, stdx::vector_aligned);
            V val2(pInput + ((k + 2 * nGroups) * stride) * L, stdx::vector_aligned);

            V sum12 = val1 + val2;

            store_unchecked(val0 + sum12, pOutput + (3 * k * stride) * L, stdx::vector_aligned);
            store_unchecked(val0 + cos120 * sum12, pOutput + (stride - 1 + (3 * k + 1) * stride) * L, stdx::vector_aligned);
            store_unchecked(sin120 * (val2 - val1), pOutput + ((3 * k + 2) * stride) * L, stdx::vector_aligned);
        } else { // Hermitian → Real
            V val0(pInput + (3 * k * stride) * L, stdx::vector_aligned);
            V val1(pInput + (stride - 1 + (3 * k + 1) * stride) * L, stdx::vector_aligned);
            V val2(pInput + ((3 * k + 2) * stride) * L, stdx::vector_aligned);

            val1     = val1 + val1;            // unpack Hermitian: double real component
            val2     = val2 * (T{2} * sin120); // unpack: scale imaginary
            V rotSum = cos120 * val1 + val0;

            store_unchecked(val0 + val1, pOutput + (k * stride) * L, stdx::vector_aligned);
            store_unchecked(rotSum - val2, pOutput + ((k + nGroups) * stride) * L, stdx::vector_aligned);
            store_unchecked(rotSum + val2, pOutput + ((k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
        }
    }

    if (stride == 1) {
        return;
    }

    // pass 2: General frequencies (0 < i < stride/2)
    for (std::size_t k = 0; k < nGroups; ++k) {
        for (std::size_t i = 2; i < stride; i += 2) {
            const std::size_t ic = stride - i; // conjugate index for Hermitian symmetry

            if constexpr (dir == Direction::Forward) { // Real → Hermitian
                V re0(pInput + ((i - 1) + k * stride) * L, stdx::vector_aligned);
                V im0(pInput + (i + k * stride) * L, stdx::vector_aligned);

                V re1(pInput + ((i - 1) + (k + nGroups) * stride) * L, stdx::vector_aligned);
                V im1(pInput + (i + (k + nGroups) * stride) * L, stdx::vector_aligned);
                applyTwiddle(re1, im1, V(pTwiddle1[i - 2]), V(pTwiddle1[i - 1]));

                V re2(pInput + ((i - 1) + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
                V im2(pInput + (i + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
                applyTwiddle(re2, im2, V(pTwiddle2[i - 2]), V(pTwiddle2[i - 1]));

                V sumRe = re1 + re2;
                V sumIm = im1 + im2;

                // output[0]: DC-like (sum of all)
                store_unchecked(re0 + sumRe, pOutput + ((i - 1) + (3 * k) * stride) * L, stdx::vector_aligned);
                store_unchecked(im0 + sumIm, pOutput + (i + (3 * k) * stride) * L, stdx::vector_aligned);

                // apply 120° rotations
                V rotSumRe = re0 + cos120 * sumRe;
                V rotSumIm = im0 + cos120 * sumIm;
                V crossRe  = sin120 * (im1 - im2);
                V crossIm  = sin120 * (re2 - re1);

                // store Hermitian-symmetric pairs: X[i] and X[-i]
                store_unchecked(rotSumRe + crossRe, pOutput + ((i - 1) + (3 * k + 2) * stride) * L, stdx::vector_aligned);
                store_unchecked(rotSumRe - crossRe, pOutput + ((ic - 1) + (3 * k + 1) * stride) * L, stdx::vector_aligned);
                store_unchecked(rotSumIm + crossIm, pOutput + (i + (3 * k + 2) * stride) * L, stdx::vector_aligned);
                store_unchecked(crossIm - rotSumIm, pOutput + (ic + (3 * k + 1) * stride) * L, stdx::vector_aligned);
            } else { // Hermitian → Real
                V re_pos(pInput + ((i - 1) + 3 * k * stride) * L, stdx::vector_aligned);
                V im_pos(pInput + (i + 3 * k * stride) * L, stdx::vector_aligned);
                V re_i2(pInput + ((i - 1) + (3 * k + 2) * stride) * L, stdx::vector_aligned);
                V im_i2(pInput + (i + (3 * k + 2) * stride) * L, stdx::vector_aligned);
                V re_neg(pInput + ((ic - 1) + (3 * k + 1) * stride) * L, stdx::vector_aligned);
                V im_neg(pInput + (ic + (3 * k + 1) * stride) * L, stdx::vector_aligned);

                // unpack Hermitian pairs
                V sumRe    = re_i2 + re_neg;
                V rotSumRe = cos120 * sumRe + re_pos;
                V diffRe   = sin120 * (re_i2 - re_neg);

                V sumIm    = im_i2 - im_neg;
                V rotSumIm = cos120 * sumIm + im_pos;
                V diffIm   = sin120 * (im_i2 + im_neg);

                // output[0]
                store_unchecked(re_pos + sumRe, pOutput + ((i - 1) + k * stride) * L, stdx::vector_aligned);
                store_unchecked(im_pos + sumIm, pOutput + (i + k * stride) * L, stdx::vector_aligned);

                // output[1] and [2] with twiddles
                V out1Re = rotSumRe - diffIm;
                V out1Im = rotSumIm + diffRe;
                V out2Re = rotSumRe + diffIm;
                V out2Im = rotSumIm - diffRe;

                applyTwiddle(out1Re, out1Im, V(pTwiddle1[i - 2]), V(pTwiddle1[i - 1]));
                store_unchecked(out1Re, pOutput + ((i - 1) + (k + nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(out1Im, pOutput + (i + (k + nGroups) * stride) * L, stdx::vector_aligned);

                applyTwiddle(out2Re, out2Im, V(pTwiddle2[i - 2]), V(pTwiddle2[i - 1]));
                store_unchecked(out2Re, pOutput + ((i - 1) + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(out2Im, pOutput + (i + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
            }
        }
    }
}

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) realRadix4(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3) {
    using V                           = vec<T, 4>;
    constexpr std::size_t L           = V::size();
    const std::size_t     totalStride = nGroups * stride;

    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(output.data()));

    T* RESTRICT       pInput    = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput   = std::assume_aligned<64>(output.data());
    const T* RESTRICT pTwiddle1 = twiddles1.data();
    const T* RESTRICT pTwiddle2 = twiddles2.data();
    const T* RESTRICT pTwiddle3 = twiddles3.data();

    constexpr auto applyTwiddle = []<typename Vec, typename Scalar>(Vec& ar, Vec& ai, const Scalar& br, const Scalar& bi) {
        if constexpr (direction == Direction::Forward) {
            complex_multiply_conj(ar, ai, br, bi);
        } else {
            complex_multiply(ar, ai, br, bi);
        }
    };

    if constexpr (direction == Direction::Forward) {
        // -1/√2 used for 45° phase shifts at Nyquist frequency
        constexpr T minus_inv_sqrt2 = T{-1} / std::numbers::sqrt2_v<T>;

        // i=0 case: DC components
        for (std::size_t k = 0; k < totalStride; k += stride) {
            const std::size_t idxIn  = k * L;
            const std::size_t idxOut = (4 * k) * L;

            // load 4 input samples
            V a0(pInput + idxIn, stdx::vector_aligned);
            V a1(pInput + idxIn + totalStride * L, stdx::vector_aligned);
            V a2(pInput + idxIn + 2 * totalStride * L, stdx::vector_aligned);
            V a3(pInput + idxIn + 3 * totalStride * L, stdx::vector_aligned);

            // 4-point DFT butterfly:
            // DC (k=0): sum all
            // π/2 (k=1): a0-a2 (90° rotation)
            // π (k=2): a1-a3 (180° rotation)
            // 3π/2 (k=3): difference of sums
            V sum13 = a1 + a3;
            V sum02 = a0 + a2;

            store_unchecked(sum13 + sum02, pOutput + idxOut, stdx::vector_aligned);
            store_unchecked(a0 - a2, pOutput + idxOut + (2 * stride - 1) * L, stdx::vector_aligned);
            store_unchecked(a3 - a1, pOutput + idxOut + (2 * stride) * L, stdx::vector_aligned);
            store_unchecked(sum02 - sum13, pOutput + idxOut + (4 * stride - 1) * L, stdx::vector_aligned);
        }

        if (stride < 2) {
            return;
        }

        // general case: apply twiddle factors
        if (stride != 2) {
            for (std::size_t k = 0; k < totalStride; k += stride) {
                for (std::size_t i = 2; i < stride; i += 2) {
                    const std::size_t ic = stride - i; // conjugate index

                    // Load first input (no twiddle)
                    V re0(pInput + ((i - 1) + k) * L, stdx::vector_aligned);
                    V im0(pInput + ((i) + k) * L, stdx::vector_aligned);

                    // Apply twiddle W^1 to second input
                    V re1(pInput + ((i - 1) + k + totalStride) * L, stdx::vector_aligned);
                    V im1(pInput + ((i) + k + totalStride) * L, stdx::vector_aligned);
                    applyTwiddle(re1, im1, V(pTwiddle1[i - 2]), V(pTwiddle1[i - 1]));

                    // Apply twiddle W^2 to third input
                    V re2(pInput + ((i - 1) + k + 2 * totalStride) * L, stdx::vector_aligned);
                    V im2(pInput + ((i) + k + 2 * totalStride) * L, stdx::vector_aligned);
                    applyTwiddle(re2, im2, V(pTwiddle2[i - 2]), V(pTwiddle2[i - 1]));

                    // Apply twiddle W^3 to fourth input
                    V re3(pInput + ((i - 1) + k + 3 * totalStride) * L, stdx::vector_aligned);
                    V im3(pInput + ((i) + k + 3 * totalStride) * L, stdx::vector_aligned);
                    applyTwiddle(re3, im3, V(pTwiddle3[i - 2]), V(pTwiddle3[i - 1]));

                    // 4-point butterfly with 90° rotations
                    V sumRe13  = re1 + re3; // Real: input[1] + input[3]
                    V diffRe13 = re3 - re1; // Real: input[3] - input[1] (90° phase)
                    V sumRe02  = re0 + re2; // Real: input[0] + input[2]
                    V diffRe02 = re0 - re2; // Real: input[0] - input[2]

                    V sumIm13  = im1 + im3; // Imag: input[1] + input[3]
                    V diffIm13 = im1 - im3; // Imag: input[1] - input[3] (90° phase)
                    V sumIm02  = im0 + im2; // Imag: input[0] + input[2]
                    V diffIm02 = im0 - im2; // Imag: input[0] - input[2]

                    // store Hermitian-symmetric pairs
                    const std::size_t outBase = 4 * k * L;
                    store_unchecked(sumRe02 + sumRe13, pOutput + outBase + (i - 1) * L, stdx::vector_aligned);
                    store_unchecked(sumRe02 - sumRe13, pOutput + outBase + (ic - 1 + 3 * stride) * L, stdx::vector_aligned);
                    store_unchecked(sumIm02 + sumIm13, pOutput + outBase + i * L, stdx::vector_aligned);
                    store_unchecked(sumIm13 - sumIm02, pOutput + outBase + (ic + 3 * stride) * L, stdx::vector_aligned);

                    store_unchecked(diffRe02 + diffIm13, pOutput + outBase + (i - 1 + 2 * stride) * L, stdx::vector_aligned);
                    store_unchecked(diffRe02 - diffIm13, pOutput + outBase + (ic - 1 + stride) * L, stdx::vector_aligned);
                    store_unchecked(diffRe13 + diffIm02, pOutput + outBase + (i + 2 * stride) * L, stdx::vector_aligned);
                    store_unchecked(diffRe13 - diffIm02, pOutput + outBase + (ic + stride) * L, stdx::vector_aligned);
                }
            }
            if (stride % 2 == 1) {
                return;
            }
        }

        // pass 3: Nyquist frequency (i = stride/2, only when stride is even) ===
        // requires 45° phase shift: multiply by (1-i)/√2 = -1/√2 * (1+i)
        for (std::size_t k = 0; k < totalStride; k += stride) {
            V val0(pInput + (stride - 1 + k) * L, stdx::vector_aligned);
            V val1(pInput + (stride - 1 + k + totalStride) * L, stdx::vector_aligned);
            V val2(pInput + (stride - 1 + k + 2 * totalStride) * L, stdx::vector_aligned);
            V val3(pInput + (stride - 1 + k + 3 * totalStride) * L, stdx::vector_aligned);

            V nyquistRe = minus_inv_sqrt2 * (val3 - val1);
            V nyquistIm = minus_inv_sqrt2 * (val1 + val3);

            const std::size_t outBase = 4 * k * L;
            store_unchecked(val0 + nyquistRe, pOutput + outBase + (stride - 1) * L, stdx::vector_aligned);
            store_unchecked(val0 - nyquistRe, pOutput + outBase + (stride - 1 + 2 * stride) * L, stdx::vector_aligned);
            store_unchecked(nyquistIm - val2, pOutput + outBase + stride * L, stdx::vector_aligned);
            store_unchecked(nyquistIm + val2, pOutput + outBase + 3 * stride * L, stdx::vector_aligned);
        }

    } else { // Direction::Backward
        constexpr T minus_sqrt2 = T{-1} * std::numbers::sqrt2_v<T>;

        // pass 1: DC components (i=0)
        for (std::size_t k = 0; k < totalStride; k += stride) {
            const std::size_t inBase = 4 * k * L;

            V val0(pInput + inBase, stdx::vector_aligned);
            V val1(pInput + inBase + (4 * stride - 1) * L, stdx::vector_aligned);
            V val2(pInput + inBase + (2 * stride) * L, stdx::vector_aligned);
            V val3(pInput + inBase + (2 * stride - 1) * L, stdx::vector_aligned);

            // unpack Hermitian: double conjugate-symmetric components
            val2     = val2 + val2;
            val3     = val3 + val3;
            V sum01  = val0 + val1;
            V diff01 = val0 - val1;

            store_unchecked(sum01 + val3, pOutput + k * L, stdx::vector_aligned);
            store_unchecked(diff01 - val2, pOutput + (k + totalStride) * L, stdx::vector_aligned);
            store_unchecked(sum01 - val3, pOutput + (k + 2 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(diff01 + val2, pOutput + (k + 3 * totalStride) * L, stdx::vector_aligned);
        }

        if (stride < 2) {
            return;
        }

        // general case
        if (stride != 2) {
            for (std::size_t k = 0; k < totalStride; k += stride) {
                for (std::size_t i = 2; i < stride; i += 2) {
                    const std::size_t inBase  = (4 * k) * L;
                    const std::size_t outBase = (k)*L;

                    // read Hermitian-symmetric input (already correct)
                    V re_pos(pInput + inBase + (i - 1) * L, stdx::vector_aligned);
                    V im_pos(pInput + inBase + (i)*L, stdx::vector_aligned);
                    V re_neg(pInput + inBase + (4 * stride - i - 1) * L, stdx::vector_aligned);
                    V im_neg(pInput + inBase + (4 * stride - i) * L, stdx::vector_aligned);

                    V tr1 = re_pos - re_neg;
                    V tr2 = re_pos + re_neg;

                    V v_2i(pInput + inBase + (2 * stride + i - 1) * L, stdx::vector_aligned);
                    V v_2i1(pInput + inBase + (2 * stride + i) * L, stdx::vector_aligned);
                    V v_2im(pInput + inBase + (2 * stride - i - 1) * L, stdx::vector_aligned);
                    V v_2im1(pInput + inBase + (2 * stride - i) * L, stdx::vector_aligned);

                    V ti4 = v_2i - v_2im;
                    V tr3 = v_2i + v_2im;

                    store_unchecked(tr2 + tr3, pOutput + outBase + (i - 1) * L, stdx::vector_aligned);
                    V cr3 = tr2 - tr3;

                    V ti3 = v_2i1 - v_2im1;
                    V tr4 = v_2i1 + v_2im1;

                    V ti1 = im_pos + im_neg;
                    V ti2 = im_pos - im_neg;

                    store_unchecked(ti2 + ti3, pOutput + outBase + (i)*L, stdx::vector_aligned);

                    // inverse butterfly with 90° rotations
                    V ci3 = ti2 - ti3;
                    V cr2 = tr1 - tr4;
                    V cr4 = tr1 + tr4;
                    V ci2 = ti1 + ti4;
                    V ci4 = ti1 - ti4;

                    // apply inverse twiddle factors
                    applyTwiddle(cr2, ci2, V(pTwiddle1[i - 2]), V(pTwiddle1[i - 1]));
                    store_unchecked(cr2, pOutput + outBase + (i - 1 + totalStride) * L, stdx::vector_aligned);
                    store_unchecked(ci2, pOutput + outBase + (i + totalStride) * L, stdx::vector_aligned);

                    applyTwiddle(cr3, ci3, V(pTwiddle2[i - 2]), V(pTwiddle2[i - 1]));
                    store_unchecked(cr3, pOutput + outBase + (i - 1 + 2 * totalStride) * L, stdx::vector_aligned);
                    store_unchecked(ci3, pOutput + outBase + (i + 2 * totalStride) * L, stdx::vector_aligned);

                    applyTwiddle(cr4, ci4, V(pTwiddle3[i - 2]), V(pTwiddle3[i - 1]));
                    store_unchecked(cr4, pOutput + outBase + (i - 1 + 3 * totalStride) * L, stdx::vector_aligned);
                    store_unchecked(ci4, pOutput + outBase + (i + 3 * totalStride) * L, stdx::vector_aligned);
                }
            }
            if (stride % 2 == 1) {
                return;
            }
        }

        // Nyquist frequency
        for (std::size_t k = 0; k < totalStride; k += stride) {
            const std::size_t i0 = 4 * k + stride;

            V c(pInput + (i0 - 1) * L, stdx::vector_aligned);
            V d(pInput + (i0 + 2 * stride - 1) * L, stdx::vector_aligned);
            V a(pInput + (i0 + 0) * L, stdx::vector_aligned);
            V b(pInput + (i0 + 2 * stride) * L, stdx::vector_aligned);

            V tr1 = c - d;
            V tr2 = c + d;
            V ti1 = b + a;
            V ti2 = b - a;

            // unpack with -√2 scaling for 45° phase correction
            store_unchecked(tr2 + tr2, pOutput + (stride - 1 + k + 0 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(minus_sqrt2 * (ti1 - tr1), pOutput + (stride - 1 + k + 1 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(ti2 + ti2, pOutput + (stride - 1 + k + 2 * totalStride) * L, stdx::vector_aligned);
            store_unchecked(minus_sqrt2 * (ti1 + tr1), pOutput + (stride - 1 + k + 3 * totalStride) * L, stdx::vector_aligned);
        }
    }
}

template<Direction direction, std::floating_point T>
static NEVER_INLINE(void) realRadix5(std::size_t stride, std::size_t nGroups, std::span<T> input, std::span<T> output, std::span<const T> twiddles1, std::span<const T> twiddles2, std::span<const T> twiddles3, std::span<const T> twiddles4) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    constexpr T cos72  = static_cast<T>(0.30901699437494745L);  // cos(2π/5)
    constexpr T cos144 = static_cast<T>(-0.80901699437494745L); // cos(4π/5)

    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(output.data()));

    T* RESTRICT       pInput    = std::assume_aligned<64>(input.data());
    T* RESTRICT       pOutput   = std::assume_aligned<64>(output.data());
    const T* RESTRICT pTwiddle1 = twiddles1.data();
    const T* RESTRICT pTwiddle2 = twiddles2.data();
    const T* RESTRICT pTwiddle3 = twiddles3.data();
    const T* RESTRICT pTwiddle4 = twiddles4.data();

    // direction-specific complex multiply
    constexpr auto applyTwiddle = []<typename Vec, typename Scalar>(Vec& re, Vec& im, const Scalar& wr, const Scalar& wi) {
        if constexpr (direction == Direction::Forward) {
            complex_multiply_conj(re, im, wr, wi);
        } else {
            complex_multiply(re, im, wr, wi);
        }
    };

    if constexpr (direction == Direction::Forward) {
        constexpr T sin72  = static_cast<T>(0.95105651629515357L); // sin(2π/5)
        constexpr T sin144 = static_cast<T>(0.58778525229247313L); // sin(4π/5)

        // pass 1: DC components (i=0)
        for (std::size_t k = 0; k < nGroups; ++k) {
            V val0(pInput + (k + 0 * nGroups) * stride * L, stdx::vector_aligned);
            V val1(pInput + (k + 1 * nGroups) * stride * L, stdx::vector_aligned);
            V val2(pInput + (k + 2 * nGroups) * stride * L, stdx::vector_aligned);
            V val3(pInput + (k + 3 * nGroups) * stride * L, stdx::vector_aligned);
            V val4(pInput + (k + 4 * nGroups) * stride * L, stdx::vector_aligned);

            // group conjugate pairs
            V sumRe_14  = val4 + val1; // inputs 1 & 4 (conjugate pair)
            V diffRe_14 = val4 - val1;
            V sumRe_23  = val3 + val2; // inputs 2 & 3 (conjugate pair)
            V diffRe_23 = val3 - val2;

            // output[0] - DC component (sum of all)
            store_unchecked(val0 + (sumRe_14 + sumRe_23), pOutput + (5 * k + 0) * stride * L, stdx::vector_aligned);

            // outputs with 72° and 144° rotations
            store_unchecked(val0 + (cos72 * sumRe_14 + cos144 * sumRe_23), pOutput + (stride - 1 + (5 * k + 1) * stride) * L, stdx::vector_aligned);
            store_unchecked(sin72 * diffRe_14 + sin144 * diffRe_23, pOutput + (5 * k + 2) * stride * L, stdx::vector_aligned);
            store_unchecked(val0 + (cos144 * sumRe_14 + cos72 * sumRe_23), pOutput + (stride - 1 + (5 * k + 3) * stride) * L, stdx::vector_aligned);
            store_unchecked(sin144 * diffRe_14 - sin72 * diffRe_23, pOutput + (5 * k + 4) * stride * L, stdx::vector_aligned);
        }

        if (stride == 1) {
            return;
        }

        // pass 2: general frequencies (0 < i < stride/2) ===
        const std::size_t strideP2 = stride + 2;
        for (std::size_t k = 0; k < nGroups; ++k) {
            for (std::size_t i = 2; i < stride; i += 2) {
                const std::size_t ic = strideP2 - i - 1; // Conjugate index

                // load first input (no twiddle)
                V re0(pInput + (i - 1 + (k + 0 * nGroups) * stride) * L, stdx::vector_aligned);
                V im0(pInput + (i + (k + 0 * nGroups) * stride) * L, stdx::vector_aligned);

                // load and apply twiddles to inputs 1-4
                V re1(pInput + (i - 1 + (k + 1 * nGroups) * stride) * L, stdx::vector_aligned);
                V im1(pInput + (i + (k + 1 * nGroups) * stride) * L, stdx::vector_aligned);
                applyTwiddle(re1, im1, V(pTwiddle1[i - 2]), V(pTwiddle1[i - 1]));

                V re2(pInput + (i - 1 + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
                V im2(pInput + (i + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
                applyTwiddle(re2, im2, V(pTwiddle2[i - 2]), V(pTwiddle2[i - 1]));

                V re3(pInput + (i - 1 + (k + 3 * nGroups) * stride) * L, stdx::vector_aligned);
                V im3(pInput + (i + (k + 3 * nGroups) * stride) * L, stdx::vector_aligned);
                applyTwiddle(re3, im3, V(pTwiddle3[i - 2]), V(pTwiddle3[i - 1]));

                V re4(pInput + (i - 1 + (k + 4 * nGroups) * stride) * L, stdx::vector_aligned);
                V im4(pInput + (i + (k + 4 * nGroups) * stride) * L, stdx::vector_aligned);
                applyTwiddle(re4, im4, V(pTwiddle4[i - 2]), V(pTwiddle4[i - 1]));

                // 5-point butterfly with conjugate pairs
                V sumRe_14   = re1 + re4;
                V diffRe_14  = re4 - re1;
                V crossRe_14 = im1 - im4;
                V sumIm_14   = im1 + im4;

                V sumRe_23   = re2 + re3;
                V diffRe_23  = re3 - re2;
                V crossRe_23 = im2 - im3;
                V sumIm_23   = im2 + im3;

                // output[0] - DC-like (sum with conjugate symmetry)
                store_unchecked(re0 + (sumRe_14 + sumRe_23), pOutput + (i - 1 + (5 * k + 0) * stride) * L, stdx::vector_aligned);
                store_unchecked(im0 - (sumIm_14 + sumIm_23), pOutput + (i + (5 * k + 0) * stride) * L, stdx::vector_aligned);

                // apply 72° and 144° rotations
                V rot72Re  = re0 + (cos72 * sumRe_14 + cos144 * sumRe_23);
                V rot72Im  = im0 - (cos72 * sumIm_14 + cos144 * sumIm_23);
                V rot144Re = re0 + (cos144 * sumRe_14 + cos72 * sumRe_23);
                V rot144Im = im0 - (cos144 * sumIm_14 + cos72 * sumIm_23);

                V cross72Re  = sin72 * crossRe_14 + sin144 * crossRe_23;
                V cross72Im  = sin72 * diffRe_14 + sin144 * diffRe_23;
                V cross144Re = sin144 * crossRe_14 - sin72 * crossRe_23;
                V cross144Im = sin144 * diffRe_14 - sin72 * diffRe_23;

                // store Hermitian-symmetric pairs: X[i] and X[-i]
                store_unchecked(rot72Re - cross72Re, pOutput + (i - 1 + (5 * k + 2) * stride) * L, stdx::vector_aligned);
                store_unchecked(rot72Re + cross72Re, pOutput + (ic + (5 * k + 1) * stride) * L, stdx::vector_aligned);
                store_unchecked(rot72Im + cross72Im, pOutput + (i + (5 * k + 2) * stride) * L, stdx::vector_aligned);
                store_unchecked(cross72Im - rot72Im, pOutput + (ic + 1 + (5 * k + 1) * stride) * L, stdx::vector_aligned);

                store_unchecked(rot144Re - cross144Re, pOutput + (i - 1 + (5 * k + 4) * stride) * L, stdx::vector_aligned);
                store_unchecked(rot144Re + cross144Re, pOutput + (ic + (5 * k + 3) * stride) * L, stdx::vector_aligned);
                store_unchecked(rot144Im + cross144Im, pOutput + (i + (5 * k + 4) * stride) * L, stdx::vector_aligned);
                store_unchecked(cross144Im - rot144Im, pOutput + (ic + 1 + (5 * k + 3) * stride) * L, stdx::vector_aligned);
            }
        }

    } else { // Direction::Backward
        constexpr T sin72  = static_cast<T>(0.95105651629515357L);
        constexpr T sin144 = static_cast<T>(0.58778525229247313L);

        // pass 1: DC components (i=0)
        for (std::size_t k = 0; k < nGroups; ++k) {
            V val0(pInput + (5 * k + 0) * stride * L, stdx::vector_aligned);
            V val1(pInput + (stride - 1 + (5 * k + 1) * stride) * L, stdx::vector_aligned);
            V val2(pInput + (5 * k + 2) * stride * L, stdx::vector_aligned);
            V val3(pInput + (stride - 1 + (5 * k + 3) * stride) * L, stdx::vector_aligned);
            V val4(pInput + (5 * k + 4) * stride * L, stdx::vector_aligned);

            // unpack Hermitian: double real components
            val1 = val1 + val1;
            val2 = val2 + val2;
            val3 = val3 + val3;
            val4 = val4 + val4;

            // inverse 5-point DFT
            V rot72    = val0 + (cos72 * val1 + cos144 * val3);
            V rot144   = val0 + (cos144 * val1 + cos72 * val3);
            V cross72  = sin72 * val2 + sin144 * val4;
            V cross144 = sin144 * val2 + sin72 * val4;

            store_unchecked(val0 + (val1 + val3), pOutput + (k + 0 * nGroups) * stride * L, stdx::vector_aligned);
            store_unchecked(rot72 - cross72, pOutput + (k + 1 * nGroups) * stride * L, stdx::vector_aligned);
            store_unchecked(rot144 - cross144, pOutput + (k + 2 * nGroups) * stride * L, stdx::vector_aligned);
            store_unchecked(rot144 + cross144, pOutput + (k + 3 * nGroups) * stride * L, stdx::vector_aligned);
            store_unchecked(rot72 + cross72, pOutput + (k + 4 * nGroups) * stride * L, stdx::vector_aligned);
        }

        if (stride == 1) {
            return;
        }

        // pass 2: general frequencies (0 < i < stride/2)
        const std::size_t strideP2 = stride + 2;
        for (std::size_t k = 0; k < nGroups; ++k) {
            for (std::size_t i = 2; i < stride; i += 2) {
                const std::size_t ic = strideP2 - i - 1;

                // load Hermitian-symmetric input from both X[i] and X[-i]
                V re_pos(pInput + (i - 1 + (5 * k + 0) * stride) * L, stdx::vector_aligned);
                V im_pos(pInput + (i + (5 * k + 0) * stride) * L, stdx::vector_aligned);
                V re_i2(pInput + (i - 1 + (5 * k + 2) * stride) * L, stdx::vector_aligned);
                V im_i2(pInput + (i + (5 * k + 2) * stride) * L, stdx::vector_aligned);
                V re_neg1(pInput + (ic + (5 * k + 1) * stride) * L, stdx::vector_aligned);
                V im_neg1(pInput + (ic + 1 + (5 * k + 1) * stride) * L, stdx::vector_aligned);
                V re_i4(pInput + (i - 1 + (5 * k + 4) * stride) * L, stdx::vector_aligned);
                V im_i4(pInput + (i + (5 * k + 4) * stride) * L, stdx::vector_aligned);
                V re_neg3(pInput + (ic + (5 * k + 3) * stride) * L, stdx::vector_aligned);
                V im_neg3(pInput + (ic + 1 + (5 * k + 3) * stride) * L, stdx::vector_aligned);

                // unpack Hermitian pairs
                V sumRe_14  = re_i2 + re_neg1;
                V diffRe_14 = re_i2 - re_neg1;
                V sumRe_23  = re_i4 + re_neg3;
                V diffRe_23 = re_i4 - re_neg3;
                V sumIm_14  = im_i2 - re_neg1;
                V diffIm_14 = im_i2 + re_neg1;
                V sumIm_23  = im_i4 - re_neg3;
                V diffIm_23 = im_i4 + re_neg3;

                V crossRe_14 = re_i2 - im_neg1;
                V crossRe_23 = re_i4 - im_neg3;

                // output[0]
                store_unchecked(re_pos + (sumRe_14 + sumRe_23), pOutput + (i - 1 + (k + 0 * nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(im_pos + (diffRe_14 + diffRe_23), pOutput + (i + (k + 0 * nGroups) * stride) * L, stdx::vector_aligned);

                // inverse butterfly with rotations
                V rot72Re  = re_pos + (cos72 * sumRe_14 + cos144 * sumRe_23);
                V rot72Im  = im_pos + (cos72 * diffRe_14 + cos144 * diffRe_23);
                V rot144Re = re_pos + (cos144 * sumRe_14 + cos72 * sumRe_23);
                V rot144Im = im_pos + (cos144 * diffRe_14 + cos72 * diffRe_23);

                V cross72Re  = sin72 * crossRe_14 + sin144 * crossRe_23;
                V cross72Im  = sin72 * sumIm_14 + sin144 * sumIm_23;
                V cross144Re = sin144 * crossRe_14 - sin72 * crossRe_23;
                V cross144Im = sin144 * sumIm_14 - sin72 * sumIm_23;

                // prepare outputs with twiddles
                V out1Re = rot72Re + cross72Im;
                V out1Im = rot72Im - cross72Re;
                V out2Re = rot144Re + cross144Im;
                V out2Im = rot144Im - cross144Re;
                V out3Re = rot144Re - cross144Im;
                V out3Im = rot144Im + cross144Re;
                V out4Re = rot72Re - cross72Im;
                V out4Im = rot72Im + cross72Re;

                applyTwiddle(out1Re, out1Im, V(pTwiddle1[i - 2]), V(pTwiddle1[i - 1]));
                store_unchecked(out1Re, pOutput + (i - 1 + (k + 1 * nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(out1Im, pOutput + (i + (k + 1 * nGroups) * stride) * L, stdx::vector_aligned);

                applyTwiddle(out2Re, out2Im, V(pTwiddle2[i - 2]), V(pTwiddle2[i - 1]));
                store_unchecked(out2Re, pOutput + (i - 1 + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(out2Im, pOutput + (i + (k + 2 * nGroups) * stride) * L, stdx::vector_aligned);

                applyTwiddle(out3Re, out3Im, V(pTwiddle3[i - 2]), V(pTwiddle3[i - 1]));
                store_unchecked(out3Re, pOutput + (i - 1 + (k + 3 * nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(out3Im, pOutput + (i + (k + 3 * nGroups) * stride) * L, stdx::vector_aligned);

                applyTwiddle(out4Re, out4Im, V(pTwiddle4[i - 2]), V(pTwiddle4[i - 1]));
                store_unchecked(out4Re, pOutput + (i - 1 + (k + 4 * nGroups) * stride) * L, stdx::vector_aligned);
                store_unchecked(out4Im, pOutput + (i + (k + 4 * nGroups) * stride) * L, stdx::vector_aligned);
            }
        }
    }
}

/********************************** danger territory ends here ************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/

template<std::array<std::size_t, 5UZ> ntryh>
static constexpr std::size_t decompose(std::size_t n, std::span<std::size_t> radixPlan) {
    std::size_t nl = n, numStages = 0;
    for (std::size_t j = 0; ntryh[j]; ++j) {
        const std::size_t ntry = ntryh[j];
        while (nl != 1) {
            std::size_t nq = nl / ntry;
            std::size_t nr = nl - ntry * nq;
            if (nr == 0) {
                radixPlan[2 + numStages++] = ntry;
                nl                         = nq;
                if (ntry == 2 && numStages != 1) {
                    for (std::size_t i = 2; i <= numStages; ++i) {
                        std::size_t ib      = numStages - i + 2;
                        radixPlan[ib + 1UZ] = radixPlan[ib];
                    }
                    radixPlan[2UZ] = 2;
                }
            } else {
                break;
            }
        }
    }
    radixPlan[0] = n;
    radixPlan[1] = numStages;
    return numStages;
}

template<Direction dir, Transform transform, std::floating_point T>
void dispatchRadix(std::size_t radix, std::size_t stride, std::size_t nGroups, std::span<T> in, std::span<T> out, std::span<const T> twiddles, std::size_t offset) {

    constexpr auto getRadixFn = []<std::size_t R>() {
        if constexpr (transform == Transform::Real) {
            if constexpr (R == 2) {
                return &realRadix2<dir, T>;
            } else if constexpr (R == 3) {
                return &realRadix3<dir, T>;
            } else if constexpr (R == 4) {
                return &realRadix4<dir, T>;
            } else if constexpr (R == 5) {
                return &realRadix5<dir, T>;
            }
        } else {
            if constexpr (R == 2) {
                return &complexRadix2<dir, T>;
            } else if constexpr (R == 3) {
                return &complexRadix3<dir, T>;
            } else if constexpr (R == 4) {
                return &complexRadix4<dir, T>;
            } else if constexpr (R == 5) {
                return &complexRadix5<dir, T>;
            }
        }
    };

    switch (radix) {
    case 2: getRadixFn.template operator()<2>()(stride, nGroups, in, out, twiddles.subspan(offset)); break;
    case 3: {
        const std::size_t   off2 = offset + stride;
        getRadixFn.template operator()<3>()(stride, nGroups, in, out, twiddles.subspan(offset), twiddles.subspan(off2));
    } break;
    case 4: {
        const std::size_t   off2 = offset + stride;
        const std::size_t   off3 = off2 + stride;
        getRadixFn.template operator()<4>()(stride, nGroups, in, out, twiddles.subspan(offset), twiddles.subspan(off2), twiddles.subspan(off3));
    } break;
    case 5: {
        const std::size_t   off2 = offset + stride;
        const std::size_t   off3 = off2 + stride;
        const std::size_t   off4 = off3 + stride;
        getRadixFn.template operator()<5>()(stride, nGroups, in, out, twiddles.subspan(offset), twiddles.subspan(off2), twiddles.subspan(off3), twiddles.subspan(off4));
    } break;
    default: std::unreachable();
    }
}

template<Direction dir, Transform transform, std::floating_point T>
static NEVER_INLINE(std::span<T>) fftStages(std::size_t nVectors, std::span<const T> input, std::span<T> workBuffer1, std::span<T> workBuffer2, std::span<const T> twiddles, std::span<const std::size_t, 15> radixPlan) {
    assert(isAligned<64>(input.data()));
    assert(isAligned<64>(workBuffer1.data()));
    assert(isAligned<64>(workBuffer2.data()));
    assert(isAligned<64>(twiddles.data()));

    const T* RESTRICT inputAligned = std::assume_aligned<64>(input.data());
    std::span<T>      bufferIn{const_cast<T*>(inputAligned), input.size()};
    std::span<T>      bufferOut = (bufferIn.data() == workBuffer2.data()) ? workBuffer1 : workBuffer2;
    if constexpr (transform == Transform::Real && dir == Direction::Backward) {
        assert(bufferIn.data() != bufferOut.data());
    }

    constexpr bool isRealForward = (transform == Transform::Real && dir == Direction::Forward);
    constexpr bool isComplex     = (transform == Transform::Complex);
    std::size_t    groupSize     = isRealForward ? nVectors : 1;
    std::size_t    twiddleOffset = isRealForward ? (nVectors - 1) : 0;

    const std::size_t numStages = radixPlan[1];
    auto              getRadix  = [&](std::size_t k1) -> std::size_t {
        if constexpr (isRealForward) {
            return radixPlan[numStages - k1 + 2];
        } else if constexpr (isComplex) {
            return radixPlan[k1];
        } else {
            return radixPlan[k1 + 1];
        }
    };

    const auto [loopStart, loopEnd] = isComplex ? std::pair{2UZ, numStages + 1} : std::pair{1UZ, numStages};
    for (std::size_t k1 = loopStart; k1 <= loopEnd; ++k1) {
        const std::size_t radix           = getRadix(k1);
        const std::size_t nextGroupSize   = isRealForward ? groupSize / radix : groupSize * radix;
        const std::size_t stride          = nVectors / (isRealForward ? groupSize : nextGroupSize);
        const std::size_t effectiveStride = isComplex ? (stride + stride) : stride;

        // update twiddle offset before dispatch for real-forward
        if constexpr (isRealForward) {
            twiddleOffset -= (radix - 1) * effectiveStride;
        }

        dispatchRadix<dir, transform>(radix, effectiveStride, isRealForward ? nextGroupSize : groupSize, bufferIn, bufferOut, twiddles, twiddleOffset);

        groupSize = nextGroupSize;

        if constexpr (!isRealForward) {
            twiddleOffset += (radix - 1) * effectiveStride;
        }

        if (bufferOut.data() == workBuffer2.data()) {
            bufferOut = workBuffer1;
            bufferIn  = workBuffer2;
        } else {
            bufferOut = workBuffer2;
            bufferIn  = workBuffer1;
        }
    }

    return bufferIn;
}

/* [0 0 1 2 3 4 5 6 7 8] -> [0 8 7 6 5 4 3 2 1] */
template<std::floating_point T>
static void reversed_copy(std::size_t N, const vec<T>* in, std::size_t in_stride, vec<T>* out) {
    vec<T> g0, g1;
    interleave(in[0], in[1], g0, g1);
    in += in_stride;

    *--out = blend_lo2_hi1(g0, g1); /* [g0l, g0h], [g1l g1h] -> [g1l, g0h] */
    for (std::size_t k = 1UZ; k < N; ++k) {
        vec<T> h0, h1;
        interleave(in[0], in[1], h0, h1);
        in += in_stride;
        *--out = blend_lo2_hi1(g1, h0);
        *--out = blend_lo2_hi1(h0, h1);
        g1     = h1;
    }
    *--out = blend_lo2_hi1(g1, g0);
}

template<std::floating_point T>
void unreversed_copy(std::size_t N, const vec<T>* in, vec<T>* out, int out_stride) {
    const vec<T> g0 = in[0];
    vec<T>       g1 = in[0];
    ++in;
    vec<T> h0, h1;
    for (std::size_t k = 1; k < N; ++k) {
        h0 = *in++;
        h1 = *in++;
        g1 = blend_lo2_hi1(g1, h0);
        h0 = blend_lo2_hi1(h0, h1);
        uninterleave(h0, g1, out[0], out[1]);
        out += out_stride;
        g1 = h1;
    }
    h0 = *in++;
    h1 = g0;
    g1 = blend_lo2_hi1(g1, h0);
    h0 = blend_lo2_hi1(h0, h1);
    uninterleave(h0, g1, out[0], out[1]);
}

template<std::floating_point T>
constexpr ALWAYS_INLINE(void) realFinalise_4x4(const T* RESTRICT in0, const T* RESTRICT in1, const T* RESTRICT in, std::span<const T> eSpan, T* RESTRICT out) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();
    const T* RESTRICT     e = std::assume_aligned<64>(eSpan.data());

    V r0(in0, stdx::vector_aligned);
    V i0(in1, stdx::vector_aligned);
    V r1(in, stdx::vector_aligned);
    in += L;
    V i1(in, stdx::vector_aligned);
    in += L;
    V r2(in, stdx::vector_aligned);
    in += L;
    V i2(in, stdx::vector_aligned);
    in += L;
    V r3(in, stdx::vector_aligned);
    in += L;
    V i3(in, stdx::vector_aligned);

    transpose(r0, r1, r2, r3);
    transpose(i0, i1, i2, i3);

    V e0(e + 0UZ * L, stdx::vector_aligned);
    V e1(e + 1UZ * L, stdx::vector_aligned);
    V e2(e + 2UZ * L, stdx::vector_aligned);
    V e3(e + 3UZ * L, stdx::vector_aligned);
    V e4(e + 4UZ * L, stdx::vector_aligned);
    V e5(e + 5UZ * L, stdx::vector_aligned);

    complex_multiply(r1, i1, e0, e1);
    complex_multiply(r2, i2, e2, e3);
    complex_multiply(r3, i3, e4, e5);

    V sr0 = r0 + r2, dr0 = r0 - r2;
    V sr1 = r1 + r3, dr1 = r3 - r1;
    V si0 = i0 + i2, di0 = i0 - i2;
    V si1 = i1 + i3, di1 = i3 - i1;

    r0 = sr0 + sr1;
    r3 = sr0 - sr1;
    i0 = si0 + si1;
    i3 = si1 - si0;
    r1 = dr0 + di1;
    r2 = dr0 - di1;
    i1 = dr1 - di0;
    i2 = dr1 + di0;

    store_unchecked(r0, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i0, out, stdx::vector_aligned);
    out += L;
    store_unchecked(r1, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i1, out, stdx::vector_aligned);
    out += L;
    store_unchecked(r2, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i2, out, stdx::vector_aligned);
    out += L;
    store_unchecked(r3, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i3, out, stdx::vector_aligned);
}

template<std::floating_point T>
constexpr void realFinalise(std::size_t Ncvec, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<const T> e) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    assert(inputSpan.data() != outputSpan.data());
    assert(isAligned<64>(inputSpan.data()));
    assert(isAligned<64>(outputSpan.data()));
    const T* RESTRICT input  = std::assume_aligned<64>(inputSpan.data());
    T* RESTRICT       output = std::assume_aligned<64>(outputSpan.data());

    V cr(input, stdx::vector_aligned);
    V ci(input + (Ncvec * 2 - 1) * L, stdx::vector_aligned);

    alignas(64UZ) T zero_storage[L] = {};
    realFinalise_4x4(zero_storage, zero_storage, input + L, e.subspan(0, 6 * L), output);

    // special handling for DC and Nyquist
    constexpr T s     = std::numbers::sqrt2_v<T> / 2;      // Keep this for forward
    output[0 * L + 0] = (cr[0] + cr[2]) + (cr[1] + cr[3]); // DC
    output[1 * L + 0] = (cr[0] + cr[2]) - (cr[1] + cr[3]); // Nyquist
    output[4 * L + 0] = (cr[0] - cr[2]);
    output[5 * L + 0] = (cr[3] - cr[1]);
    output[2 * L + 0] = ci[0] + s * (ci[1] - ci[3]);
    output[3 * L + 0] = -ci[2] - s * (ci[1] + ci[3]);
    output[6 * L + 0] = ci[0] - s * (ci[1] - ci[3]);
    output[7 * L + 0] = ci[2] - s * (ci[1] + ci[3]);

    const std::size_t dx = Ncvec / L;
    V                 save(input + 7 * L, stdx::vector_aligned);

    for (std::size_t k = 1; k < dx; ++k) {
        V               save_next(input + (8 * k + 7) * L, stdx::vector_aligned);
        alignas(64UZ) T save_storage[L];
        store_unchecked(save, save_storage, stdx::vector_aligned);
        realFinalise_4x4(save_storage, input + 8 * k * L, input + (8 * k + 1) * L, e.subspan(k * 6 * L, 6 * L), output + k * 8 * L);
        save = save_next;
    }
}

template<std::floating_point T>
constexpr ALWAYS_INLINE(void) realPreprocess_4x4(const T* RESTRICT in, std::span<const T> eSpan, T* RESTRICT out, std::size_t first) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    V r0(in, stdx::vector_aligned);
    V i0(in + L, stdx::vector_aligned);
    V r1(in + 2 * L, stdx::vector_aligned);
    V i1(in + 3 * L, stdx::vector_aligned);
    V r2(in + 4 * L, stdx::vector_aligned);
    V i2(in + 5 * L, stdx::vector_aligned);
    V r3(in + 6 * L, stdx::vector_aligned);
    V i3(in + 7 * L, stdx::vector_aligned);

    V sr0 = r0 + r3, dr0 = r0 - r3;
    V sr1 = r1 + r2, dr1 = r1 - r2;
    V si0 = i0 + i3, di0 = i0 - i3;
    V si1 = i1 + i2, di1 = i1 - i2;

    r0 = sr0 + sr1;
    r2 = sr0 - sr1;
    r1 = dr0 - si1;
    r3 = dr0 + si1;
    i0 = di0 - di1;
    i2 = di0 + di1;
    i1 = si0 - dr1;
    i3 = si0 + dr1;

    const T* RESTRICT e = std::assume_aligned<64>(eSpan.data());

    V e0(e, stdx::vector_aligned);
    V e1(e + L, stdx::vector_aligned);
    V e2(e + 2 * L, stdx::vector_aligned);
    V e3(e + 3 * L, stdx::vector_aligned);
    V e4(e + 4 * L, stdx::vector_aligned);
    V e5(e + 5 * L, stdx::vector_aligned);

    complex_multiply_conj(r1, i1, e0, e1);
    complex_multiply_conj(r2, i2, e2, e3);
    complex_multiply_conj(r3, i3, e4, e5);

    transpose(r0, r1, r2, r3);
    transpose(i0, i1, i2, i3);

    if (!first) {
        store_unchecked(r0, out, stdx::vector_aligned);
        out += L;
        store_unchecked(i0, out, stdx::vector_aligned);
        out += L;
    }
    store_unchecked(r1, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i1, out, stdx::vector_aligned);
    out += L;
    store_unchecked(r2, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i2, out, stdx::vector_aligned);
    out += L;
    store_unchecked(r3, out, stdx::vector_aligned);
    out += L;
    store_unchecked(i3, out, stdx::vector_aligned);
}

template<std::floating_point T>
constexpr void realPreprocess(std::size_t Ncvec, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<const T> e) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    assert(inputSpan.data() != outputSpan.data());
    assert(isAligned<64>(inputSpan.data()));
    assert(isAligned<64>(outputSpan.data()));
    const T* RESTRICT input  = std::assume_aligned<64>(inputSpan.data());
    T* RESTRICT       output = std::assume_aligned<64>(outputSpan.data());

    V Xr, Xi;
    for (std::size_t k = 0; k < 4; ++k) {
        Xr[k] = input[8 * k];     // positions 0, 8, 16, 24
        Xi[k] = input[8 * k + 4]; // positions 4, 12, 20, 28
    }

    realPreprocess_4x4(inputSpan.data(), e.subspan(0, 6 * L), output + L, 1);

    const std::size_t dk = Ncvec / L;
    for (std::size_t k = 1; k < dk; ++k) {
        realPreprocess_4x4(inputSpan.data() + 8 * k * L, e.subspan(k * 6 * L, 6 * L), output + (k * 8 - 1) * L, 0);
    }

    // DC & Nyquist writeback
    output[0 * L + 0]               = (Xr[0] + Xi[0]) + 2 * Xr[2];
    output[0 * L + 1]               = (Xr[0] - Xi[0]) - 2 * Xi[2];
    output[0 * L + 2]               = (Xr[0] + Xi[0]) - 2 * Xr[2];
    output[0 * L + 3]               = (Xr[0] - Xi[0]) + 2 * Xi[2];
    output[(2 * Ncvec - 1) * L + 0] = 2 * (Xr[1] + Xr[3]);
    output[(2 * Ncvec - 1) * L + 1] = std::numbers::sqrt2_v<T> * (Xr[1] - Xr[3]) - std::numbers::sqrt2_v<T> * (Xi[1] + Xi[3]);
    output[(2 * Ncvec - 1) * L + 2] = 2 * (Xi[3] - Xi[1]);
    output[(2 * Ncvec - 1) * L + 3] = -std::numbers::sqrt2_v<T> * (Xr[1] - Xr[3]) - std::numbers::sqrt2_v<T> * (Xi[1] + Xi[3]);
}

} // namespace details

} // namespace gr::algorithm

#ifdef ALWAYS_INLINE
#undef ALWAYS_INLINE
#endif
#ifdef NEVER_INLINE
#undef NEVER_INLINE
#endif
#ifdef RESTRICT
#undef RESTRICT
#endif

#endif /* SIMD_FFT_HPP */
