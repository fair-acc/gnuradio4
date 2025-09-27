#include "pffft.h"

#include "gnuradio-4.0/Message.hpp"

/* detect compiler flavour */
#if defined(_MSC_VER)
#define COMPILER_MSVC
#elif defined(__GNUC__)
#define COMPILER_GCC
#endif

#include <gnuradio-4.0/MemoryAllocators.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <span>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"          // error in vir/simd
#pragma GCC diagnostic ignored "-Wsign-conversion" // error in vir/simd
#endif

#include <vir/simd_execution.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if defined(COMPILER_GCC)
#define ALWAYS_INLINE(return_type)                    inline return_type __attribute__((always_inline))
#define NEVER_INLINE(return_type)                     return_type __attribute__((noinline))
#define RESTRICT                                      __restrict
#define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ varname__[size__];
#elif defined(COMPILER_MSVC)
#define ALWAYS_INLINE(return_type)                    __forceinline return_type
#define NEVER_INLINE(return_type)                     __declspec(noinline) return_type
#define RESTRICT                                      __restrict
#define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__* varname__ = static_cast<type__*>(_alloca(size__ * sizeof(type__)))
#endif

#ifdef COMPILER_MSVC
#pragma warning(disable : 4244 4305 4204 4456)
#endif

#include <vir/simd.h>

namespace stdx = vir::stdx;

template<std::floating_point T, int N = 4> // inspired by future C++26 definition
using vec = stdx::simd<T, stdx::simd_abi::deduce_t<T, static_cast<std::size_t>(N)>>;

template<typename T>
[[nodiscard]] constexpr ALWAYS_INLINE(T) load_unchecked(const typename T::value_type* ptr, auto flags = stdx::element_aligned) {
    return T(ptr, flags);
}

template<typename T>
constexpr ALWAYS_INLINE(void) store_unchecked(const T& v, typename T::value_type* ptr, auto flags = stdx::element_aligned) {
    v.copy_to(ptr, flags);
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) VSWAPHL(const Vec& a, const Vec& b) noexcept { // find a better name
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

template<std::floating_point T>
struct FFTConstants {
    static constexpr T taur        = T{-0.5};
    static constexpr T taui        = T{0.5} * std::numbers::sqrt3_v<T>; // sqrt(3)/2
    static constexpr T taui_2      = std::numbers::sqrt3_v<T>;
    static constexpr T tr11        = T{0.309016994374947};  // cos(2*pi/5)
    static constexpr T ti11        = T{0.951056516295154};  // sin(2*pi/5)
    static constexpr T tr12        = T{-0.809016994374947}; // cos(4*pi/5)
    static constexpr T ti12        = T{0.587785252292473};  // sin(4*pi/5)
    static constexpr T minus_hsqt2 = T{-0.7071067811865475};
    // static constexpr T inv_sqrt2   = T{0.5} * std::numbers::sqrt2_v<T>;
};

template<std::floating_point T>
constexpr bool pffft_is_valid_size(std::size_t N, Transform cplx) {
    const std::size_t N_min = pffft_min_fft_size<T>(cplx); // checks for Radix-5, -3, and -2
    for (std::size_t factor : {5UZ, 3UZ, 2UZ}) {
        while (N >= factor * N_min && N % factor == 0UZ) {
            N /= factor;
        }
    }
    return N == N_min;
}

template<std::floating_point T>
constexpr std::size_t pffft_nearest_transform_size(std::size_t N, Transform cplx, bool higher) {
    const std::size_t N_min = pffft_min_fft_size<T>(cplx);
    if (N < N_min) {
        N = N_min;
    }
    const std::size_t d = higher ? N_min : -N_min;
    if (d > 0) {
        N = N_min * ((N + N_min - 1) / N_min); /* round up */
    } else {
        N = N_min * (N / N_min); /* round down */
    }

    for (;; N += d) {
        if (pffft_is_valid_size<T>(N, cplx)) {
            return N;
        }
    }
}

/*
  passf2 and passb2 has been merged here, fsign = -1 for passf2, +1 for passb2
*/
template<int sign, std::floating_point T>
static NEVER_INLINE(void) passf2_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1) {
    using V                                  = vec<T, 4>;
    [[maybe_unused]] constexpr std::size_t L = V::size();

    if (ido <= 2) {
        const std::size_t l1ido = l1 * ido;
        for (std::size_t k = 0; k < l1ido; k += ido) {
            std::size_t base_cc = k * 2 * L; // FIXED: multiply by 2
            std::size_t base_ch = k * L;

            V c0 = load_unchecked<V>(cc.data() + base_cc, stdx::vector_aligned);
            V c1 = load_unchecked<V>(cc.data() + base_cc + L, stdx::vector_aligned);
            V c2 = load_unchecked<V>(cc.data() + base_cc + ido * L, stdx::vector_aligned);
            V c3 = load_unchecked<V>(cc.data() + base_cc + ido * L + L, stdx::vector_aligned);

            store_unchecked(c0 + c2, ch.data() + base_ch, stdx::vector_aligned);
            store_unchecked(c1 + c3, ch.data() + base_ch + L, stdx::vector_aligned);
            store_unchecked(c0 - c2, ch.data() + base_ch + l1ido * L, stdx::vector_aligned);
            store_unchecked(c1 - c3, ch.data() + base_ch + l1ido * L + L, stdx::vector_aligned);

            // REMOVED the manual increments - k handles the iteration
        }
    } else {
        const std::size_t l1ido = l1 * ido;
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 0; i < ido - 1; i += 2) {
                std::size_t idx_cc = (k * 2 + i) * L; // This is already correct
                std::size_t idx_ch = (k + i) * L;     // This is already correct

                // Rest of the code remains the same
                V cc0 = load_unchecked<V>(cc.data() + idx_cc, stdx::vector_aligned);
                V cc1 = load_unchecked<V>(cc.data() + idx_cc + L, stdx::vector_aligned);
                V cc2 = load_unchecked<V>(cc.data() + idx_cc + ido * L, stdx::vector_aligned);
                V cc3 = load_unchecked<V>(cc.data() + idx_cc + ido * L + L, stdx::vector_aligned);

                V tr2 = cc0 - cc2;
                V ti2 = cc1 - cc3;
                V wr(wa1[i]);
                V wi = T{sign} * V(wa1[i + 1]);

                store_unchecked(cc0 + cc2, ch.data() + idx_ch, stdx::vector_aligned);
                store_unchecked(cc1 + cc3, ch.data() + idx_ch + L, stdx::vector_aligned);

                complex_multiply(tr2, ti2, wr, wi);
                store_unchecked(tr2, ch.data() + idx_ch + l1ido * L, stdx::vector_aligned);
                store_unchecked(ti2, ch.data() + idx_ch + l1ido * L + L, stdx::vector_aligned);
            }
        }
    }
}

template<int sign, std::floating_point T>
static NEVER_INLINE(void) passf3_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2) {
    using V                                     = vec<T, 4>;
    [[maybe_unused]] constexpr std::size_t L    = V::size();
    constexpr T                            taui = 0.866025403784439f * T(sign);

    assert(ido > 2);
    const std::size_t l1ido = l1 * ido;

    for (std::size_t k = 0; k < l1ido; k += ido) {
        std::size_t cc_base = k * 3 * L; // cc advances by 3*ido per iteration
        std::size_t ch_base = k * L;     // ch advances by ido per iteration

        for (std::size_t i = 0; i < ido - 1; i += 2) {
            // Load from cc array
            V cc_r0 = load_unchecked<V>(cc.data() + cc_base + i * L, stdx::vector_aligned);
            V cc_i0 = load_unchecked<V>(cc.data() + cc_base + (i + 1) * L, stdx::vector_aligned);
            V cc_r1 = load_unchecked<V>(cc.data() + cc_base + (i + ido) * L, stdx::vector_aligned);
            V cc_i1 = load_unchecked<V>(cc.data() + cc_base + (i + ido + 1) * L, stdx::vector_aligned);
            V cc_r2 = load_unchecked<V>(cc.data() + cc_base + (i + 2 * ido) * L, stdx::vector_aligned);
            V cc_i2 = load_unchecked<V>(cc.data() + cc_base + (i + 2 * ido + 1) * L, stdx::vector_aligned);

            V tr2 = cc_r1 + cc_r2;
            V cr2 = cc_r0 + FFTConstants<T>::taur * tr2;
            V ti2 = cc_i1 + cc_i2;
            V ci2 = cc_i0 + FFTConstants<T>::taur * ti2;

            store_unchecked(cc_r0 + tr2, ch.data() + ch_base + i * L, stdx::vector_aligned);
            store_unchecked(cc_i0 + ti2, ch.data() + ch_base + (i + 1) * L, stdx::vector_aligned);

            V cr3 = taui * (cc_r1 - cc_r2);
            V ci3 = taui * (cc_i1 - cc_i2);
            V dr2 = cr2 - ci3;
            V dr3 = cr2 + ci3;
            V di2 = ci2 + cr3;
            V di3 = ci2 - cr3;

            const T wr1 = wa1[i], wi1 = T(sign) * wa1[i + 1];
            const T wr2 = wa2[i], wi2 = T(sign) * wa2[i + 1];

            complex_multiply(dr2, di2, V(wr1), V(wi1));
            store_unchecked(dr2, ch.data() + ch_base + (i + l1ido) * L, stdx::vector_aligned);
            store_unchecked(di2, ch.data() + ch_base + (i + l1ido + 1) * L, stdx::vector_aligned);

            complex_multiply(dr3, di3, V(wr2), V(wi2));
            store_unchecked(dr3, ch.data() + ch_base + (i + 2 * l1ido) * L, stdx::vector_aligned);
            store_unchecked(di3, ch.data() + ch_base + (i + 2 * l1ido + 1) * L, stdx::vector_aligned);
        }
    }
}

template<int sign, std::floating_point T>
static NEVER_INLINE(void) passf4_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2, std::span<const T> wa3) {
    using V                                      = vec<T, 4>;
    [[maybe_unused]] constexpr std::size_t L     = V::size();
    std::size_t                            l1ido = l1 * ido;

    if (ido == 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            std::size_t cc_base = k * 4 * L;
            std::size_t ch_base = k * L;

            V cc0 = load_unchecked<V>(cc.data() + cc_base, stdx::vector_aligned);
            V cc1 = load_unchecked<V>(cc.data() + cc_base + L, stdx::vector_aligned);
            V cc2 = load_unchecked<V>(cc.data() + cc_base + 2 * ido * L, stdx::vector_aligned);
            V cc3 = load_unchecked<V>(cc.data() + cc_base + (2 * ido + 1) * L, stdx::vector_aligned);
            V cc4 = load_unchecked<V>(cc.data() + cc_base + ido * L, stdx::vector_aligned);
            V cc5 = load_unchecked<V>(cc.data() + cc_base + (ido + 1) * L, stdx::vector_aligned);
            V cc6 = load_unchecked<V>(cc.data() + cc_base + 3 * ido * L, stdx::vector_aligned);
            V cc7 = load_unchecked<V>(cc.data() + cc_base + (3 * ido + 1) * L, stdx::vector_aligned);

            const V tr1 = cc0 - cc2;
            const V tr2 = cc0 + cc2;
            const V ti1 = cc1 - cc3;
            const V ti2 = cc1 + cc3;
            const V ti4 = (cc4 - cc6) * T{sign};
            const V tr4 = (cc7 - cc5) * T{sign};
            const V tr3 = cc4 + cc6;
            const V ti3 = cc5 + cc7;

            store_unchecked(tr2 + tr3, ch.data() + ch_base, stdx::vector_aligned);
            store_unchecked(ti2 + ti3, ch.data() + ch_base + L, stdx::vector_aligned);
            store_unchecked(tr1 + tr4, ch.data() + ch_base + l1ido * L, stdx::vector_aligned);
            store_unchecked(ti1 + ti4, ch.data() + ch_base + (l1ido + 1) * L, stdx::vector_aligned);
            store_unchecked(tr2 - tr3, ch.data() + ch_base + 2 * l1ido * L, stdx::vector_aligned);
            store_unchecked(ti2 - ti3, ch.data() + ch_base + (2 * l1ido + 1) * L, stdx::vector_aligned);
            store_unchecked(tr1 - tr4, ch.data() + ch_base + 3 * l1ido * L, stdx::vector_aligned);
            store_unchecked(ti1 - ti4, ch.data() + ch_base + (3 * l1ido + 1) * L, stdx::vector_aligned);
        }
    } else {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            std::size_t cc_base = k * 4 * L;
            std::size_t ch_base = k * L;

            for (std::size_t i = 0; i < ido - 1; i += 2) {
                std::size_t cc_idx = cc_base + i * L;
                std::size_t ch_idx = ch_base + i * L;

                // Load raw values from cc - DON'T apply twiddle factors yet
                V cc0 = load_unchecked<V>(cc.data() + cc_idx, stdx::vector_aligned);
                V cc1 = load_unchecked<V>(cc.data() + cc_idx + L, stdx::vector_aligned);

                V cc2_r = load_unchecked<V>(cc.data() + cc_idx + ido * L, stdx::vector_aligned);
                V cc2_i = load_unchecked<V>(cc.data() + cc_idx + ido * L + L, stdx::vector_aligned);

                V cc3_r = load_unchecked<V>(cc.data() + cc_idx + 2 * ido * L, stdx::vector_aligned);
                V cc3_i = load_unchecked<V>(cc.data() + cc_idx + 2 * ido * L + L, stdx::vector_aligned);

                V cc4_r = load_unchecked<V>(cc.data() + cc_idx + 3 * ido * L, stdx::vector_aligned);
                V cc4_i = load_unchecked<V>(cc.data() + cc_idx + 3 * ido * L + L, stdx::vector_aligned);

                // Butterfly operations on RAW values
                const V tr1 = cc0 - cc3_r;
                const V tr2 = cc0 + cc3_r;
                const V ti1 = cc1 - cc3_i;
                const V ti2 = cc1 + cc3_i;
                const V tr4 = (cc4_i - cc2_i) * T{sign}; // imaginary parts
                const V ti4 = (cc2_r - cc4_r) * T{sign}; // real parts
                const V tr3 = cc2_r + cc4_r;
                const V ti3 = cc2_i + cc4_i;

                // store first output (no twiddle needed)
                store_unchecked(tr2 + tr3, ch.data() + ch_idx, stdx::vector_aligned);
                store_unchecked(ti2 + ti3, ch.data() + ch_idx + L, stdx::vector_aligned);

                // compute butterfly results
                V cr2 = tr1 + tr4;
                V ci2 = ti1 + ti4;
                V cr3 = tr2 - tr3;
                V ci3 = ti2 - ti3;
                V cr4 = tr1 - tr4;
                V ci4 = ti1 - ti4;

                // apply twiddle factors
                T wr1 = wa1[i], wi1 = T{sign} * wa1[i + 1];
                T wr2 = wa2[i], wi2 = T{sign} * wa2[i + 1];
                T wr3 = wa3[i], wi3 = T{sign} * wa3[i + 1];

                complex_multiply(cr2, ci2, V(wr1), V(wi1));
                complex_multiply(cr3, ci3, V(wr2), V(wi2));
                complex_multiply(cr4, ci4, V(wr3), V(wi3));

                // Store results
                store_unchecked(cr2, ch.data() + ch_idx + l1ido * L, stdx::vector_aligned);
                store_unchecked(ci2, ch.data() + ch_idx + l1ido * L + L, stdx::vector_aligned);
                store_unchecked(cr3, ch.data() + ch_idx + 2 * l1ido * L, stdx::vector_aligned);
                store_unchecked(ci3, ch.data() + ch_idx + 2 * l1ido * L + L, stdx::vector_aligned);
                store_unchecked(cr4, ch.data() + ch_idx + 3 * l1ido * L, stdx::vector_aligned);
                store_unchecked(ci4, ch.data() + ch_idx + 3 * l1ido * L + L, stdx::vector_aligned);
            }
        }
    }
}

template<int sign, std::floating_point T>
static NEVER_INLINE(void) passf5_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2, std::span<const T> wa3, std::span<const T> wa4) {
    using V                                  = vec<T, 4>;
    [[maybe_unused]] constexpr std::size_t L = V::size();
    assert(ido > 2);
    constexpr T tr11 = static_cast<T>(0.30901699437494745L);
    constexpr T ti11 = static_cast<T>(0.95105651629515357L) * T{sign};
    constexpr T tr12 = static_cast<T>(-0.80901699437494745L);
    constexpr T ti12 = static_cast<T>(0.58778525229247313L) * T{sign};

    for (std::size_t k = 0; k < l1; ++k) {
        std::size_t cc_base = k * 5 * ido * L;
        std::size_t ch_base = k * ido * L;

        for (std::size_t i = 0; i < ido - 1; i += 2) {
            // Load cc values - using direct indexing
            V cc_r0 = load_unchecked<V>(cc.data() + cc_base + i * L, stdx::vector_aligned);
            V cc_i0 = load_unchecked<V>(cc.data() + cc_base + (i + 1) * L, stdx::vector_aligned);
            V cc_r1 = load_unchecked<V>(cc.data() + cc_base + (ido + i) * L, stdx::vector_aligned);
            V cc_i1 = load_unchecked<V>(cc.data() + cc_base + (ido + i + 1) * L, stdx::vector_aligned);
            V cc_r2 = load_unchecked<V>(cc.data() + cc_base + (2 * ido + i) * L, stdx::vector_aligned);
            V cc_i2 = load_unchecked<V>(cc.data() + cc_base + (2 * ido + i + 1) * L, stdx::vector_aligned);
            V cc_r3 = load_unchecked<V>(cc.data() + cc_base + (3 * ido + i) * L, stdx::vector_aligned);
            V cc_i3 = load_unchecked<V>(cc.data() + cc_base + (3 * ido + i + 1) * L, stdx::vector_aligned);
            V cc_r4 = load_unchecked<V>(cc.data() + cc_base + (4 * ido + i) * L, stdx::vector_aligned);
            V cc_i4 = load_unchecked<V>(cc.data() + cc_base + (4 * ido + i + 1) * L, stdx::vector_aligned);

            V ti5 = cc_i1 - cc_i4;
            V ti2 = cc_i1 + cc_i4;
            V ti4 = cc_i2 - cc_i3;
            V ti3 = cc_i2 + cc_i3;
            V tr5 = cc_r1 - cc_r4;
            V tr2 = cc_r1 + cc_r4;
            V tr4 = cc_r2 - cc_r3;
            V tr3 = cc_r2 + cc_r3;

            store_unchecked(cc_r0 + (tr2 + tr3), ch.data() + ch_base + i * L, stdx::vector_aligned);
            store_unchecked(cc_i0 + (ti2 + ti3), ch.data() + ch_base + (i + 1) * L, stdx::vector_aligned);

            V cr2 = cc_r0 + (tr11 * tr2 + tr12 * tr3);
            V ci2 = cc_i0 + (tr11 * ti2 + tr12 * ti3);
            V cr3 = cc_r0 + (tr12 * tr2 + tr11 * tr3);
            V ci3 = cc_i0 + (tr12 * ti2 + tr11 * ti3);
            V cr5 = (ti11 * tr5) + ti12 * tr4;
            V ci5 = (ti11 * ti5) + ti12 * ti4;
            V cr4 = (ti12 * tr5) - ti11 * tr4;
            V ci4 = (ti12 * ti5) - ti11 * ti4;

            V dr3 = cr3 - ci4;
            V dr4 = cr3 + ci4;
            V di3 = ci3 + cr4;
            V di4 = ci3 - cr4;
            V dr5 = cr2 + ci5;
            V dr2 = cr2 - ci5;
            V di5 = ci2 - cr5;
            V di2 = ci2 + cr5;

            const T wr1 = wa1[i], wi1 = T{sign} * wa1[i + 1];
            const T wr2 = wa2[i], wi2 = T{sign} * wa2[i + 1];
            const T wr3 = wa3[i], wi3 = T{sign} * wa3[i + 1];
            const T wr4 = wa4[i], wi4 = T{sign} * wa4[i + 1];

            complex_multiply(dr2, di2, V(wr1), V(wi1));
            store_unchecked(dr2, ch.data() + ch_base + (l1 * ido + i) * L, stdx::vector_aligned);
            store_unchecked(di2, ch.data() + ch_base + (l1 * ido + i + 1) * L, stdx::vector_aligned);

            complex_multiply(dr3, di3, V(wr2), V(wi2));
            store_unchecked(dr3, ch.data() + ch_base + (2 * l1 * ido + i) * L, stdx::vector_aligned);
            store_unchecked(di3, ch.data() + ch_base + (2 * l1 * ido + i + 1) * L, stdx::vector_aligned);

            complex_multiply(dr4, di4, V(wr3), V(wi3));
            store_unchecked(dr4, ch.data() + ch_base + (3 * l1 * ido + i) * L, stdx::vector_aligned);
            store_unchecked(di4, ch.data() + ch_base + (3 * l1 * ido + i + 1) * L, stdx::vector_aligned);

            complex_multiply(dr5, di5, V(wr4), V(wi4));
            store_unchecked(dr5, ch.data() + ch_base + (4 * l1 * ido + i) * L, stdx::vector_aligned);
            store_unchecked(di5, ch.data() + ch_base + (4 * l1 * ido + i + 1) * L, stdx::vector_aligned);
        }
    }
}

template<std::floating_point T>
static NEVER_INLINE(void) radf2_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1) {
    using V = vec<T, 4>;

    constexpr std::size_t L         = V::size();
    constexpr T           minus_one = T{-1};
    std::size_t           l1ido     = l1 * ido;

    for (std::size_t k = 0; k < l1ido; k += ido) {
        V a = load_unchecked<V>(cc.data() + k * L, stdx::vector_aligned);
        V b = load_unchecked<V>(cc.data() + (k + l1ido) * L, stdx::vector_aligned);
        store_unchecked(a + b, ch.data() + 2 * k * L, stdx::vector_aligned);
        store_unchecked(a - b, ch.data() + (2 * (k + ido) - 1) * L, stdx::vector_aligned);
    }

    if (ido < 2) {
        return;
    }

    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 2; i < ido; i += 2) {
                V br  = load_unchecked<V>(cc.data() + (i - 1 + k) * L, stdx::vector_aligned);
                V bi  = load_unchecked<V>(cc.data() + (i + k) * L, stdx::vector_aligned);
                V tr2 = load_unchecked<V>(cc.data() + (i - 1 + k + l1ido) * L, stdx::vector_aligned);
                V ti2 = load_unchecked<V>(cc.data() + (i + k + l1ido) * L, stdx::vector_aligned);

                const T wr = wa1[i - 2];
                const T wi = wa1[i - 1];
                complex_multiply_conj(tr2, ti2, V(wr), V(wi));

                store_unchecked(bi + ti2, ch.data() + (i + 2 * k) * L, stdx::vector_aligned);
                store_unchecked(ti2 - bi, ch.data() + (2 * (k + ido) - i) * L, stdx::vector_aligned);
                store_unchecked(br + tr2, ch.data() + (i - 1 + 2 * k) * L, stdx::vector_aligned);
                store_unchecked(br - tr2, ch.data() + (2 * (k + ido) - i - 1) * L, stdx::vector_aligned);
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }

    for (std::size_t k = 0; k < l1ido; k += ido) {
        V cc_last    = load_unchecked<V>(cc.data() + (k + ido - 1) * L, stdx::vector_aligned);
        V cc_last_l1 = load_unchecked<V>(cc.data() + (ido - 1 + k + l1ido) * L, stdx::vector_aligned);
        store_unchecked(minus_one * cc_last_l1, ch.data() + (2 * k + ido) * L, stdx::vector_aligned);
        store_unchecked(cc_last, ch.data() + (2 * k + ido - 1) * L, stdx::vector_aligned);
    }
}

template<std::floating_point T>
static NEVER_INLINE(void) radb2_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1) {
    using V = vec<T, 4>;

    constexpr std::size_t L         = V::size();
    constexpr T           minus_two = T{-2};
    std::size_t           l1ido     = l1 * ido;

    for (std::size_t k = 0; k < l1ido; k += ido) {
        V a = load_unchecked<V>(cc.data() + 2 * k * L, stdx::vector_aligned);
        V b = load_unchecked<V>(cc.data() + (2 * (k + ido) - 1) * L, stdx::vector_aligned);
        store_unchecked(a + b, ch.data() + k * L, stdx::vector_aligned);
        store_unchecked(a - b, ch.data() + (k + l1ido) * L, stdx::vector_aligned);
    }

    if (ido < 2) {
        return;
    }

    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 2; i < ido; i += 2) {
                V a = load_unchecked<V>(cc.data() + (i - 1 + 2 * k) * L, stdx::vector_aligned);
                V b = load_unchecked<V>(cc.data() + (2 * (k + ido) - i - 1) * L, stdx::vector_aligned);
                V c = load_unchecked<V>(cc.data() + (i + 2 * k) * L, stdx::vector_aligned);
                V d = load_unchecked<V>(cc.data() + (2 * (k + ido) - i) * L, stdx::vector_aligned);

                store_unchecked(a + b, ch.data() + (i - 1 + k) * L, stdx::vector_aligned);
                V tr2 = a - b;
                store_unchecked(c - d, ch.data() + (i + k) * L, stdx::vector_aligned);
                V ti2 = c + d;

                const T wr = wa1[i - 2];
                const T wi = wa1[i - 1];
                complex_multiply(tr2, ti2, V(wr), V(wi));

                store_unchecked(tr2, ch.data() + (i - 1 + k + l1ido) * L, stdx::vector_aligned);
                store_unchecked(ti2, ch.data() + (i + k + l1ido) * L, stdx::vector_aligned);
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }

    for (std::size_t k = 0; k < l1ido; k += ido) {
        V a = load_unchecked<V>(cc.data() + (2 * k + ido - 1) * L, stdx::vector_aligned);
        V b = load_unchecked<V>(cc.data() + (2 * k + ido) * L, stdx::vector_aligned);
        store_unchecked(a + a, ch.data() + (k + ido - 1) * L, stdx::vector_aligned);
        store_unchecked(minus_two * b, ch.data() + (k + ido - 1 + l1ido) * L, stdx::vector_aligned);
    }
}

template<std::floating_point T>
static void radf3_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2) {
    using V                                     = vec<T, 4>;
    [[maybe_unused]] constexpr std::size_t L    = V::size();
    constexpr T                            taui = std::numbers::sqrt3_v<T> / T{2};

    // k-loop, i==0 (special) lanes
    for (std::size_t k = 0; k < l1; ++k) {
        V cc0 = load_unchecked<V>(cc.data() + (k * ido) * L, stdx::vector_aligned);
        V cc1 = load_unchecked<V>(cc.data() + ((k + l1) * ido) * L, stdx::vector_aligned);
        V cc2 = load_unchecked<V>(cc.data() + ((k + 2 * l1) * ido) * L, stdx::vector_aligned);

        V cr2 = cc1 + cc2;
        store_unchecked(cc0 + cr2, ch.data() + (3 * k * ido) * L, stdx::vector_aligned);
        store_unchecked(taui * (cc2 - cc1), ch.data() + ((3 * k + 2) * ido) * L, stdx::vector_aligned);
        store_unchecked(cc0 + FFTConstants<T>::taur * cr2, ch.data() + ((ido - 1 + (3 * k + 1) * ido) * L), stdx::vector_aligned);
    }
    if (ido == 1) {
        return;
    }

    // general i>0
    for (std::size_t k = 0; k < l1; ++k) {
        for (std::size_t i = 2; i < ido; i += 2) {
            const std::size_t ic = ido - i;

            V dr2   = load_unchecked<V>(cc.data() + ((i - 1) + (k + l1) * ido) * L, stdx::vector_aligned);
            V di2   = load_unchecked<V>(cc.data() + ((i) + (k + l1) * ido) * L, stdx::vector_aligned);
            T wr1_s = wa1[i - 2];
            T wi1_s = wa1[i - 1];
            V wr1(wr1_s), wi1(wi1_s);
            complex_multiply_conj(dr2, di2, wr1, wi1);

            V dr3   = load_unchecked<V>(cc.data() + ((i - 1) + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            V di3   = load_unchecked<V>(cc.data() + ((i) + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            T wr2_s = wa2[i - 2];
            T wi2_s = wa2[i - 1];
            V wr2(wr2_s), wi2(wi2_s);
            complex_multiply_conj(dr3, di3, wr2, wi2);

            V cc_r = load_unchecked<V>(cc.data() + ((i - 1) + k * ido) * L, stdx::vector_aligned);
            V cc_i = load_unchecked<V>(cc.data() + ((i) + k * ido) * L, stdx::vector_aligned);

            V cr2 = dr2 + dr3;
            V ci2 = di2 + di3;

            store_unchecked(cc_r + cr2, ch.data() + ((i - 1) + (3 * k * ido)) * L, stdx::vector_aligned);
            store_unchecked(cc_i + ci2, ch.data() + ((i) + (3 * k * ido)) * L, stdx::vector_aligned);

            V tr2 = cc_r + FFTConstants<T>::taur * cr2;
            V ti2 = cc_i + FFTConstants<T>::taur * ci2;
            V tr3 = taui * (di2 - di3);
            V ti3 = taui * (dr3 - dr2);

            store_unchecked(tr2 + tr3, ch.data() + ((i - 1) + (3 * k + 2) * ido) * L, stdx::vector_aligned);
            store_unchecked(tr2 - tr3, ch.data() + ((ic - 1) + (3 * k + 1) * ido) * L, stdx::vector_aligned);
            store_unchecked(ti2 + ti3, ch.data() + ((i) + (3 * k + 2) * ido) * L, stdx::vector_aligned);
            store_unchecked(ti3 - ti2, ch.data() + ((ic) + (3 * k + 1) * ido) * L, stdx::vector_aligned);
        }
    }
}

template<std::floating_point T>
static void radb3_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2) {
    using V = vec<T, 4>;

    constexpr std::size_t L      = V::size();
    constexpr T           taur   = T{-0.5};
    constexpr T           taui   = std::numbers::sqrt3_v<T> / T{2};
    constexpr T           taui_2 = T{2} * taui;

    // k-loop, i==0 (special) lanes
    for (std::size_t k = 0; k < l1; ++k) {
        V tr2 = load_unchecked<V>(cc.data() + ((ido - 1) + (3 * k + 1) * ido) * L, stdx::vector_aligned);
        tr2   = tr2 + tr2;

        V cc0 = load_unchecked<V>(cc.data() + (3 * k * ido) * L, stdx::vector_aligned);
        V cr2 = taur * tr2 + cc0;

        store_unchecked(cc0 + tr2, ch.data() + (k * ido) * L, stdx::vector_aligned);

        V ci3 = taui_2 * load_unchecked<V>(cc.data() + ((3 * k + 2) * ido) * L, stdx::vector_aligned);
        store_unchecked(cr2 - ci3, ch.data() + ((k + l1) * ido) * L, stdx::vector_aligned);
        store_unchecked(cr2 + ci3, ch.data() + ((k + 2 * l1) * ido) * L, stdx::vector_aligned);
    }
    if (ido == 1) {
        return;
    }

    // general i>0
    for (std::size_t k = 0; k < l1; ++k) {
        for (std::size_t i = 2; i < ido; i += 2) {
            const std::size_t ic = ido - i;

            V cc_r0 = load_unchecked<V>(cc.data() + ((i - 1) + 3 * k * ido) * L, stdx::vector_aligned);
            V cc_i0 = load_unchecked<V>(cc.data() + ((i) + 3 * k * ido) * L, stdx::vector_aligned);
            V cc_r1 = load_unchecked<V>(cc.data() + ((i - 1) + (3 * k + 2) * ido) * L, stdx::vector_aligned);
            V cc_i1 = load_unchecked<V>(cc.data() + ((i) + (3 * k + 2) * ido) * L, stdx::vector_aligned);
            V cc_r2 = load_unchecked<V>(cc.data() + ((ic - 1) + (3 * k + 1) * ido) * L, stdx::vector_aligned);
            V cc_i2 = load_unchecked<V>(cc.data() + ((ic) + (3 * k + 1) * ido) * L, stdx::vector_aligned);

            V tr2 = cc_r1 + cc_r2;
            V cr2 = taur * tr2 + cc_r0;
            store_unchecked(cc_r0 + tr2, ch.data() + ((i - 1) + k * ido) * L, stdx::vector_aligned);

            V ti2 = cc_i1 - cc_i2;
            V ci2 = taur * ti2 + cc_i0;
            store_unchecked(cc_i0 + ti2, ch.data() + ((i) + k * ido) * L, stdx::vector_aligned);

            V cr3 = taui * (cc_r1 - cc_r2);
            V ci3 = taui * (cc_i1 + cc_i2);

            V dr2 = cr2 - ci3;
            V dr3 = cr2 + ci3;
            V di2 = ci2 + cr3;
            V di3 = ci2 - cr3;

            T wr1_s = wa1[i - 2];
            T wi1_s = wa1[i - 1];
            V wr1(wr1_s), wi1(wi1_s);
            complex_multiply(dr2, di2, wr1, wi1);
            store_unchecked(dr2, ch.data() + ((i - 1) + (k + l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(di2, ch.data() + ((i) + (k + l1) * ido) * L, stdx::vector_aligned);

            T wr2_s = wa2[i - 2];
            T wi2_s = wa2[i - 1];
            V wr2(wr2_s), wi2(wi2_s);
            complex_multiply(dr3, di3, wr2, wi2);
            store_unchecked(dr3, ch.data() + ((i - 1) + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(di3, ch.data() + ((i) + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
        }
    }
}

template<std::floating_point T>
static NEVER_INLINE(void) radf4_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2, std::span<const T> wa3) {
    using V                               = vec<T, 4>;
    constexpr std::size_t L               = V::size();
    constexpr T           minus_inv_sqrt2 = T{-1} / std::numbers::sqrt2_v<T>;
    const std::size_t     l1ido           = l1 * ido;

    // i == 0 lanes
    for (std::size_t k = 0; k < l1ido; k += ido) {
        const std::size_t cc_idx = k * L;
        const std::size_t ch_idx = (4 * k) * L;

        V a0 = load_unchecked<V>(cc.data() + cc_idx, stdx::vector_aligned);
        V a1 = load_unchecked<V>(cc.data() + cc_idx + l1ido * L, stdx::vector_aligned);
        V a2 = load_unchecked<V>(cc.data() + cc_idx + 2 * l1ido * L, stdx::vector_aligned);
        V a3 = load_unchecked<V>(cc.data() + cc_idx + 3 * l1ido * L, stdx::vector_aligned);

        V tr1 = a1 + a3;
        V tr2 = a0 + a2;

        store_unchecked(tr1 + tr2, ch.data() + ch_idx, stdx::vector_aligned);
        store_unchecked(a0 - a2, ch.data() + ch_idx + (2 * ido - 1) * L, stdx::vector_aligned);
        store_unchecked(a3 - a1, ch.data() + ch_idx + (2 * ido) * L, stdx::vector_aligned);
        store_unchecked(tr2 - tr1, ch.data() + ch_idx + (4 * ido - 1) * L, stdx::vector_aligned);
    }

    if (ido < 2) {
        return;
    }

    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 2; i < ido; i += 2) {
                const std::size_t ic = ido - i;

                V pc0 = load_unchecked<V>(cc.data() + ((i - 1) + k) * L, stdx::vector_aligned);
                V pc1 = load_unchecked<V>(cc.data() + ((i) + k) * L, stdx::vector_aligned);

                V cr2 = load_unchecked<V>(cc.data() + ((i - 1) + k + l1ido) * L, stdx::vector_aligned);
                V ci2 = load_unchecked<V>(cc.data() + ((i) + k + l1ido) * L, stdx::vector_aligned);
                T wr1_s;
                wr1_s = wa1[i - 2];
                T wi1_s;
                wi1_s = wa1[i - 1];
                V wr1(wr1_s), wi1(wi1_s);
                complex_multiply_conj(cr2, ci2, wr1, wi1);

                V cr3 = load_unchecked<V>(cc.data() + ((i - 1) + k + 2 * l1ido) * L, stdx::vector_aligned);
                V ci3 = load_unchecked<V>(cc.data() + ((i) + k + 2 * l1ido) * L, stdx::vector_aligned);
                T wr2_s;
                wr2_s = wa2[i - 2];
                T wi2_s;
                wi2_s = wa2[i - 1];
                V wr2(wr2_s), wi2(wi2_s);
                complex_multiply_conj(cr3, ci3, wr2, wi2);

                V cr4 = load_unchecked<V>(cc.data() + ((i - 1) + k + 3 * l1ido) * L, stdx::vector_aligned);
                V ci4 = load_unchecked<V>(cc.data() + ((i) + k + 3 * l1ido) * L, stdx::vector_aligned);
                T wr3_s;
                wr3_s = wa3[i - 2];
                T wi3_s;
                wi3_s = wa3[i - 1];
                V wr3(wr3_s), wi3(wi3_s);
                complex_multiply_conj(cr4, ci4, wr3, wi3);

                V tr1 = cr2 + cr4;
                V tr4 = cr4 - cr2;
                V tr2 = pc0 + cr3;
                V tr3 = pc0 - cr3;

                V ti1 = ci2 + ci4;
                V ti4 = ci2 - ci4;
                V ti2 = pc1 + ci3;
                V ti3 = pc1 - ci3;

                store_unchecked(tr1 + tr2, ch.data() + ((i - 1) + (4 * k)) * L, stdx::vector_aligned);
                store_unchecked(tr2 - tr1, ch.data() + ((ic - 1) + (4 * k + 3 * ido)) * L, stdx::vector_aligned);
                store_unchecked(ti4 + tr3, ch.data() + ((i - 1) + (4 * k + 2 * ido)) * L, stdx::vector_aligned);
                store_unchecked(tr3 - ti4, ch.data() + ((ic - 1) + (4 * k + 1 * ido)) * L, stdx::vector_aligned);
                store_unchecked(ti1 + ti2, ch.data() + ((i) + (4 * k)) * L, stdx::vector_aligned);
                store_unchecked(ti1 - ti2, ch.data() + ((ic) + (4 * k + 3 * ido)) * L, stdx::vector_aligned);
                store_unchecked(tr4 + ti3, ch.data() + ((i) + (4 * k + 2 * ido)) * L, stdx::vector_aligned);
                store_unchecked(tr4 - ti3, ch.data() + ((ic) + (4 * k + 1 * ido)) * L, stdx::vector_aligned);
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }

    // i == ido (even tail)
    for (std::size_t k = 0; k < l1ido; k += ido) {
        V a = load_unchecked<V>(cc.data() + (ido - 1 + k + l1ido) * L, stdx::vector_aligned);
        V b = load_unchecked<V>(cc.data() + (ido - 1 + k + 3 * l1ido) * L, stdx::vector_aligned);
        V c = load_unchecked<V>(cc.data() + (ido - 1 + k) * L, stdx::vector_aligned);
        V d = load_unchecked<V>(cc.data() + (ido - 1 + k + 2 * l1ido) * L, stdx::vector_aligned);

        V ti1 = minus_inv_sqrt2 * (a + b);
        V tr1 = minus_inv_sqrt2 * (b - a);

        store_unchecked(tr1 + c, ch.data() + (ido - 1 + 4 * k) * L, stdx::vector_aligned);
        store_unchecked(c - tr1, ch.data() + (ido - 1 + 4 * k + 2 * ido) * L, stdx::vector_aligned);
        store_unchecked(ti1 - d, ch.data() + (4 * k + 1 * ido) * L, stdx::vector_aligned);
        store_unchecked(ti1 + d, ch.data() + (4 * k + 3 * ido) * L, stdx::vector_aligned);
    }
}

template<std::floating_point T>
static NEVER_INLINE(void) radb4_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2, std::span<const T> wa3) {
    using V                           = vec<T, 4>;
    constexpr std::size_t L           = V::size();
    constexpr T           minus_sqrt2 = T{-1} * std::numbers::sqrt2_v<T>;
    constexpr T           two         = T{2};
    const std::size_t     l1ido       = l1 * ido;

    // i == 0 lanes
    for (std::size_t k = 0; k < l1ido; k += ido) {
        const std::size_t base = (4 * k) * L;

        V a = load_unchecked<V>(cc.data() + base + 0 * L, stdx::vector_aligned);
        V b = load_unchecked<V>(cc.data() + base + (4 * ido - 1) * L, stdx::vector_aligned);
        V c = load_unchecked<V>(cc.data() + base + (2 * ido) * L, stdx::vector_aligned);
        V d = load_unchecked<V>(cc.data() + base + (2 * ido - 1) * L, stdx::vector_aligned);

        V tr3 = two * d;
        V tr2 = a + b;
        V tr1 = a - b;
        V tr4 = two * c;

        store_unchecked(tr2 + tr3, ch.data() + (k + 0 * l1ido) * L, stdx::vector_aligned);
        store_unchecked(tr2 - tr3, ch.data() + (k + 2 * l1ido) * L, stdx::vector_aligned);
        store_unchecked(tr1 - tr4, ch.data() + (k + 1 * l1ido) * L, stdx::vector_aligned);
        store_unchecked(tr1 + tr4, ch.data() + (k + 3 * l1ido) * L, stdx::vector_aligned);
    }

    if (ido < 2) {
        return;
    }

    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 2; i < ido; i += 2) {
                const std::size_t base_cc = (4 * k) * L;
                const std::size_t base_ch = (k)*L;

                V pc0 = load_unchecked<V>(cc.data() + base_cc + (i)*L, stdx::vector_aligned);
                V pc1 = load_unchecked<V>(cc.data() + base_cc + (i + 1) * L, stdx::vector_aligned);
                V pc2 = load_unchecked<V>(cc.data() + base_cc + (4 * ido - i) * L, stdx::vector_aligned);
                V pc3 = load_unchecked<V>(cc.data() + base_cc + (4 * ido - i + 1) * L, stdx::vector_aligned);

                V tr1 = pc0 - pc2;
                V tr2 = pc0 + pc2;

                V v_2i   = load_unchecked<V>(cc.data() + base_cc + (2 * ido + i) * L, stdx::vector_aligned);
                V v_2im  = load_unchecked<V>(cc.data() + base_cc + (2 * ido - i) * L, stdx::vector_aligned);
                V v_2i1  = load_unchecked<V>(cc.data() + base_cc + (2 * ido + i + 1) * L, stdx::vector_aligned);
                V v_2im1 = load_unchecked<V>(cc.data() + base_cc + (2 * ido - i + 1) * L, stdx::vector_aligned);

                V ti4 = v_2i - v_2im;
                V tr3 = v_2i + v_2im;

                store_unchecked(tr2 + tr3, ch.data() + base_ch + (i)*L, stdx::vector_aligned);
                V cr3 = tr2 - tr3;

                V ti3 = v_2i1 - v_2im1;
                V tr4 = v_2i1 + v_2im1;

                V ti1 = pc1 + pc3;
                V ti2 = pc1 - pc3;

                store_unchecked(ti2 + ti3, ch.data() + base_ch + (i + 1) * L, stdx::vector_aligned);

                V ci3 = ti2 - ti3;
                V cr2 = tr1 - tr4;
                V cr4 = tr1 + tr4;
                V ci2 = ti1 + ti4;
                V ci4 = ti1 - ti4;

                T wr1_s;
                wr1_s = wa1[i - 2];
                T wi1_s;
                wi1_s = wa1[i - 1];
                V wr1(wr1_s), wi1(wi1_s);
                complex_multiply(cr2, ci2, wr1, wi1);
                store_unchecked(cr2, ch.data() + base_ch + (i + l1ido) * L, stdx::vector_aligned);
                store_unchecked(ci2, ch.data() + base_ch + (i + 1 + l1ido) * L, stdx::vector_aligned);

                T wr2_s;
                wr2_s = wa2[i - 2];
                T wi2_s;
                wi2_s = wa2[i - 1];
                V wr2(wr2_s), wi2(wi2_s);
                complex_multiply(cr3, ci3, wr2, wi2);
                store_unchecked(cr3, ch.data() + base_ch + (i + 2 * l1ido) * L, stdx::vector_aligned);
                store_unchecked(ci3, ch.data() + base_ch + (i + 1 + 2 * l1ido) * L, stdx::vector_aligned);

                T wr3_s;
                wr3_s = wa3[i - 2];
                T wi3_s;
                wi3_s = wa3[i - 1];
                V wr3(wr3_s), wi3(wi3_s);
                complex_multiply(cr4, ci4, wr3, wi3);
                store_unchecked(cr4, ch.data() + base_ch + (i + 3 * l1ido) * L, stdx::vector_aligned);
                store_unchecked(ci4, ch.data() + base_ch + (i + 1 + 3 * l1ido) * L, stdx::vector_aligned);
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }

    // even tail
    for (std::size_t k = 0; k < l1ido; k += ido) {
        const std::size_t i0 = 4 * k + ido;

        V c = load_unchecked<V>(cc.data() + (i0 - 1) * L, stdx::vector_aligned);
        V d = load_unchecked<V>(cc.data() + (i0 + 2 * ido - 1) * L, stdx::vector_aligned);
        V a = load_unchecked<V>(cc.data() + (i0 + 0) * L, stdx::vector_aligned);
        V b = load_unchecked<V>(cc.data() + (i0 + 2 * ido) * L, stdx::vector_aligned);

        V tr1 = c - d;
        V tr2 = c + d;
        V ti1 = b + a;
        V ti2 = b - a;

        store_unchecked(tr2 + tr2, ch.data() + (ido - 1 + k + 0 * l1ido) * L, stdx::vector_aligned);
        store_unchecked(minus_sqrt2 * (ti1 - tr1), ch.data() + (ido - 1 + k + 1 * l1ido) * L, stdx::vector_aligned);
        store_unchecked(ti2 + ti2, ch.data() + (ido - 1 + k + 2 * l1ido) * L, stdx::vector_aligned);
        store_unchecked(minus_sqrt2 * (ti1 + tr1), ch.data() + (ido - 1 + k + 3 * l1ido) * L, stdx::vector_aligned);
    }
}

template<std::floating_point T>
static void radf5_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2, std::span<const T> wa3, std::span<const T> wa4) {
    using V                                     = vec<T, 4>;
    [[maybe_unused]] constexpr std::size_t L    = V::size();
    constexpr T                            tr11 = static_cast<T>(0.3090169943749474241022934171828191L);
    constexpr T                            ti11 = static_cast<T>(0.9510565162951535721164393333793821L);
    constexpr T                            tr12 = static_cast<T>(-0.8090169943749474241022934171828191L);
    constexpr T                            ti12 = static_cast<T>(0.5877852522924731291687059546390728L);

    // k-loop, i==0 lanes
    for (std::size_t k = 0; k < l1; ++k) {
        V cc1 = load_unchecked<V>(cc.data() + (k + 0 * l1) * ido * L, stdx::vector_aligned);
        V cc2 = load_unchecked<V>(cc.data() + (k + 1 * l1) * ido * L, stdx::vector_aligned);
        V cc3 = load_unchecked<V>(cc.data() + (k + 2 * l1) * ido * L, stdx::vector_aligned);
        V cc4 = load_unchecked<V>(cc.data() + (k + 3 * l1) * ido * L, stdx::vector_aligned);
        V cc5 = load_unchecked<V>(cc.data() + (k + 4 * l1) * ido * L, stdx::vector_aligned);

        V cr2 = cc5 + cc2;
        V ci5 = cc5 - cc2;
        V cr3 = cc4 + cc3;
        V ci4 = cc4 - cc3;

        store_unchecked(cc1 + (cr2 + cr3), ch.data() + (5 * k + 0) * ido * L, stdx::vector_aligned);
        store_unchecked(cc1 + (tr11 * cr2 + tr12 * cr3), ch.data() + (ido - 1 + (5 * k + 1) * ido) * L, stdx::vector_aligned);
        store_unchecked(ti11 * ci5 + ti12 * ci4, ch.data() + (5 * k + 2) * ido * L, stdx::vector_aligned);
        store_unchecked(cc1 + (tr12 * cr2 + tr11 * cr3), ch.data() + (ido - 1 + (5 * k + 3) * ido) * L, stdx::vector_aligned);
        store_unchecked(ti12 * ci5 - ti11 * ci4, ch.data() + (5 * k + 4) * ido * L, stdx::vector_aligned);
    }
    if (ido == 1) {
        return;
    }

    const std::size_t idp2 = ido + 2;
    for (std::size_t k = 0; k < l1; ++k) {
        for (std::size_t i = 2; i < ido; i += 2) {
            const std::size_t ic = idp2 - i - 1;

            V dr2 = load_unchecked<V>(cc.data() + (i - 1 + (k + 1 * l1) * ido) * L, stdx::vector_aligned);
            V di2 = load_unchecked<V>(cc.data() + (i + (k + 1 * l1) * ido) * L, stdx::vector_aligned);
            T wr1_s;
            wr1_s = wa1[i - 2];
            T wi1_s;
            wi1_s = wa1[i - 1];
            complex_multiply_conj(dr2, di2, V(wr1_s), V(wi1_s));

            V dr3 = load_unchecked<V>(cc.data() + (i - 1 + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            V di3 = load_unchecked<V>(cc.data() + (i + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            T wr2_s;
            wr2_s = wa2[i - 2];
            T wi2_s;
            wi2_s = wa2[i - 1];
            complex_multiply_conj(dr3, di3, V(wr2_s), V(wi2_s));

            V dr4 = load_unchecked<V>(cc.data() + (i - 1 + (k + 3 * l1) * ido) * L, stdx::vector_aligned);
            V di4 = load_unchecked<V>(cc.data() + (i + (k + 3 * l1) * ido) * L, stdx::vector_aligned);
            T wr3_s;
            wr3_s = wa3[i - 2];
            T wi3_s;
            wi3_s = wa3[i - 1];
            complex_multiply_conj(dr4, di4, V(wr3_s), V(wi3_s));

            V dr5 = load_unchecked<V>(cc.data() + (i - 1 + (k + 4 * l1) * ido) * L, stdx::vector_aligned);
            V di5 = load_unchecked<V>(cc.data() + (i + (k + 4 * l1) * ido) * L, stdx::vector_aligned);
            T wr4_s;
            wr4_s = wa4[i - 2];
            T wi4_s;
            wi4_s = wa4[i - 1];
            complex_multiply_conj(dr5, di5, V(wr4_s), V(wi4_s));

            V cc_r = load_unchecked<V>(cc.data() + (i - 1 + (k + 0 * l1) * ido) * L, stdx::vector_aligned);
            V cc_i = load_unchecked<V>(cc.data() + (i + (k + 0 * l1) * ido) * L, stdx::vector_aligned);

            V cr2 = dr2 + dr5;
            V ci5 = dr5 - dr2;
            V cr5 = di2 - di5;
            V ci2 = di2 + di5;
            V cr3 = dr3 + dr4;
            V ci4 = dr4 - dr3;
            V cr4 = di3 - di4;
            V ci3 = di3 + di4;

            store_unchecked(cc_r + (cr2 + cr3), ch.data() + (i - 1 + (5 * k + 0) * ido) * L, stdx::vector_aligned);
            store_unchecked(cc_i - (ci2 + ci3), ch.data() + (i + (5 * k + 0) * ido) * L, stdx::vector_aligned);

            V tr2 = cc_r + (tr11 * cr2 + tr12 * cr3);
            V ti2 = cc_i - (tr11 * ci2 + tr12 * ci3);
            V tr3 = cc_r + (tr12 * cr2 + tr11 * cr3);
            V ti3 = cc_i - (tr12 * ci2 + tr11 * ci3);
            V tr5 = ti11 * cr5 + ti12 * cr4;
            V ti5 = ti11 * ci5 + ti12 * ci4;
            V tr4 = ti12 * cr5 - ti11 * cr4;
            V ti4 = ti12 * ci5 - ti11 * ci4;

            store_unchecked(tr2 - tr5, ch.data() + (i - 1 + (5 * k + 2) * ido) * L, stdx::vector_aligned);
            store_unchecked(tr2 + tr5, ch.data() + (ic + (5 * k + 1) * ido) * L, stdx::vector_aligned);
            store_unchecked(ti2 + ti5, ch.data() + (i + (5 * k + 2) * ido) * L, stdx::vector_aligned);
            store_unchecked(ti5 - ti2, ch.data() + (ic + 1 + (5 * k + 1) * ido) * L, stdx::vector_aligned);

            store_unchecked(tr3 - tr4, ch.data() + (i - 1 + (5 * k + 4) * ido) * L, stdx::vector_aligned);
            store_unchecked(tr3 + tr4, ch.data() + (ic + (5 * k + 3) * ido) * L, stdx::vector_aligned);
            store_unchecked(ti3 + ti4, ch.data() + (i + (5 * k + 4) * ido) * L, stdx::vector_aligned);
            store_unchecked(ti4 - ti3, ch.data() + (ic + 1 + (5 * k + 3) * ido) * L, stdx::vector_aligned);
        }
    }
}

template<std::floating_point T>
static void radb5_ps(std::size_t ido, std::size_t l1, std::span<const T> cc, std::span<T> ch, std::span<const T> wa1, std::span<const T> wa2, std::span<const T> wa3, std::span<const T> wa4) {
    using V                    = vec<T, 4>;
    constexpr std::size_t L    = V::size();
    constexpr T           tr11 = static_cast<T>(0.3090169943749474241022934171828191L);
    constexpr T           ti11 = static_cast<T>(0.9510565162951535721164393333793821L);
    constexpr T           tr12 = static_cast<T>(-0.8090169943749474241022934171828191L);
    constexpr T           ti12 = static_cast<T>(0.5877852522924731291687059546390728L);

    // k-loop, i==0 lanes
    for (std::size_t k = 0; k < l1; ++k) {
        V ti5 = load_unchecked<V>(cc.data() + (5 * k + 2) * ido * L, stdx::vector_aligned);
        V ti4 = load_unchecked<V>(cc.data() + (5 * k + 4) * ido * L, stdx::vector_aligned);
        V tr2 = load_unchecked<V>(cc.data() + (ido - 1 + (5 * k + 1) * ido) * L, stdx::vector_aligned);
        V tr3 = load_unchecked<V>(cc.data() + (ido - 1 + (5 * k + 3) * ido) * L, stdx::vector_aligned);
        V cc1 = load_unchecked<V>(cc.data() + (5 * k + 0) * ido * L, stdx::vector_aligned);

        ti5 = ti5 + ti5;
        ti4 = ti4 + ti4;
        tr2 = tr2 + tr2;
        tr3 = tr3 + tr3;

        store_unchecked(cc1 + (tr2 + tr3), ch.data() + (k + 0 * l1) * ido * L, stdx::vector_aligned);
        V cr2 = cc1 + (tr11 * tr2 + tr12 * tr3);
        V cr3 = cc1 + (tr12 * tr2 + tr11 * tr3);
        V ci5 = ti11 * ti5 + ti12 * ti4;
        V ci4 = ti12 * ti5 + ti11 * ti4; // NOTE: '+' (matches original), not '-'

        store_unchecked(cr2 - ci5, ch.data() + (k + 1 * l1) * ido * L, stdx::vector_aligned);
        store_unchecked(cr3 - ci4, ch.data() + (k + 2 * l1) * ido * L, stdx::vector_aligned);
        store_unchecked(cr3 + ci4, ch.data() + (k + 3 * l1) * ido * L, stdx::vector_aligned);
        store_unchecked(cr2 + ci5, ch.data() + (k + 4 * l1) * ido * L, stdx::vector_aligned);
    }
    if (ido == 1) {
        return;
    }

    const std::size_t idp2 = ido + 2;
    for (std::size_t k = 0; k < l1; ++k) {
        for (std::size_t i = 2; i < ido; i += 2) {
            const std::size_t ic = idp2 - i - 1;

            V cc_r0 = load_unchecked<V>(cc.data() + (i - 1 + (5 * k + 0) * ido) * L, stdx::vector_aligned);
            V cc_i0 = load_unchecked<V>(cc.data() + (i + (5 * k + 0) * ido) * L, stdx::vector_aligned);
            V cc_r2 = load_unchecked<V>(cc.data() + (i - 1 + (5 * k + 2) * ido) * L, stdx::vector_aligned);
            V cc_i2 = load_unchecked<V>(cc.data() + (i + (5 * k + 2) * ido) * L, stdx::vector_aligned);
            V cc_r1 = load_unchecked<V>(cc.data() + (ic + (5 * k + 1) * ido) * L, stdx::vector_aligned);
            V cc_i1 = load_unchecked<V>(cc.data() + (ic + 1 + (5 * k + 1) * ido) * L, stdx::vector_aligned);
            V cc_r4 = load_unchecked<V>(cc.data() + (i - 1 + (5 * k + 4) * ido) * L, stdx::vector_aligned);
            V cc_i4 = load_unchecked<V>(cc.data() + (i + (5 * k + 4) * ido) * L, stdx::vector_aligned);
            V cc_r3 = load_unchecked<V>(cc.data() + (ic + (5 * k + 3) * ido) * L, stdx::vector_aligned);
            V cc_i3 = load_unchecked<V>(cc.data() + (ic + 1 + (5 * k + 3) * ido) * L, stdx::vector_aligned);

            V ti5 = cc_i2 + cc_r1;
            V ti2 = cc_i2 - cc_r1;
            V ti4 = cc_i4 + cc_r3;
            V ti3 = cc_i4 - cc_r3;
            V tr5 = cc_r2 - cc_i1;
            V tr2 = cc_r2 + cc_i1;
            V tr4 = cc_r4 - cc_i3;
            V tr3 = cc_r4 + cc_i3;

            store_unchecked(cc_r0 + (tr2 + tr3), ch.data() + (i - 1 + (k + 0 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(cc_i0 + (ti2 + ti3), ch.data() + (i + (k + 0 * l1) * ido) * L, stdx::vector_aligned);

            V cr2 = cc_r0 + (tr11 * tr2 + tr12 * tr3);
            V ci2 = cc_i0 + (tr11 * ti2 + tr12 * ti3);
            V cr3 = cc_r0 + (tr12 * tr2 + tr11 * tr3);
            V ci3 = cc_i0 + (tr12 * ti2 + tr11 * ti3);
            V cr5 = ti11 * tr5 + ti12 * tr4;
            V ci5 = ti11 * ti5 + ti12 * ti4;
            V cr4 = ti12 * tr5 - ti11 * tr4;
            V ci4 = ti12 * ti5 - ti11 * ti4;

            V dr3 = cr3 - ci4;
            V dr4 = cr3 + ci4;
            V di3 = ci3 + cr4;
            V di4 = ci3 - cr4;
            V dr5 = cr2 + ci5;
            V dr2 = cr2 - ci5;
            V di5 = ci2 - cr5;
            V di2 = ci2 + cr5;

            T wr1_s;
            wr1_s = wa1[i - 2];
            T wi1_s;
            wi1_s = wa1[i - 1];
            complex_multiply(dr2, di2, V(wr1_s), V(wi1_s));
            T wr2_s;
            wr2_s = wa2[i - 2];
            T wi2_s;
            wi2_s = wa2[i - 1];
            complex_multiply(dr3, di3, V(wr2_s), V(wi2_s));
            T wr3_s;
            wr3_s = wa3[i - 2];
            T wi3_s;
            wi3_s = wa3[i - 1];
            complex_multiply(dr4, di4, V(wr3_s), V(wi3_s));
            T wr4_s;
            wr4_s = wa4[i - 2];
            T wi4_s;
            wi4_s = wa4[i - 1];
            complex_multiply(dr5, di5, V(wr4_s), V(wi4_s));

            store_unchecked(dr2, ch.data() + (i - 1 + (k + 1 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(di2, ch.data() + (i + (k + 1 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(dr3, ch.data() + (i - 1 + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(di3, ch.data() + (i + (k + 2 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(dr4, ch.data() + (i - 1 + (k + 3 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(di4, ch.data() + (i + (k + 3 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(dr5, ch.data() + (i - 1 + (k + 4 * l1) * ido) * L, stdx::vector_aligned);
            store_unchecked(di5, ch.data() + (i + (k + 4 * l1) * ido) * L, stdx::vector_aligned);
        }
    }
}

template<std::floating_point T>
static NEVER_INLINE(std::span<T>) rfftf1_ps(std::size_t n_vecs, std::span<const T> input_span, std::span<T> work1_span, std::span<T> work2_span, std::span<const T> wa_span, std::span<const std::size_t, 15> radixPlan) {
    const T* RESTRICT input_aligned = std::assume_aligned<64>(input_span.data());
    std::span<T>      in{const_cast<T*>(input_aligned), input_span.size()};
    std::span<T>      out = (in.data() == work2_span.data()) ? work1_span : work2_span;

    const std::size_t nf = radixPlan[1];
    std::size_t       l2 = n_vecs; // n_vecs is the number of vec<T> units
    std::size_t       iw = n_vecs - 1UZ;

    for (std::size_t k1 = 1; k1 <= nf; ++k1) {
        const std::size_t kh  = nf - k1;
        const std::size_t ip  = radixPlan[kh + 2];
        const std::size_t l1  = l2 / ip;
        const std::size_t ido = n_vecs / l2;
        iw -= (ip - 1) * ido;

        switch (ip) {
        case 5: {
            std::size_t ix2 = iw + ido;
            std::size_t ix3 = ix2 + ido;
            std::size_t ix4 = ix3 + ido;
            radf5_ps<T>(ido, l1, in, out, wa_span.subspan(iw), wa_span.subspan(ix2), wa_span.subspan(ix3), wa_span.subspan(ix4));
        } break;
        case 4: {
            std::size_t ix2 = iw + ido;
            std::size_t ix3 = ix2 + ido;
            radf4_ps<T>(ido, l1, in, out, wa_span.subspan(iw), wa_span.subspan(ix2), wa_span.subspan(ix3));
        } break;
        case 3: {
            std::size_t ix2 = iw + ido;
            radf3_ps<T>(ido, l1, in, out, wa_span.subspan(iw), wa_span.subspan(ix2));
        } break;
        case 2: radf2_ps<T>(ido, l1, in, out, wa_span.subspan(iw)); break;
        }

        l2 = l1;
        if (out.data() == work2_span.data()) {
            out = work1_span;
            in  = work2_span;
        } else {
            out = work2_span;
            in  = work1_span;
        }
    }
    return in;
}

template<std::floating_point T>
static NEVER_INLINE(std::span<T>) rfftb1_ps(std::size_t n_vecs, std::span<const T> input_span, std::span<T> work1_span, std::span<T> work2_span, std::span<const T> wa_span, std::span<const std::size_t, 15> radixPlan) {
    const T* RESTRICT input_aligned = std::assume_aligned<64>(input_span.data());
    std::span<T>      input{const_cast<T*>(input_aligned), input_span.size()};
    std::span<T>      output = (input.data() == work2_span.data()) ? work1_span : work2_span;

    std::size_t nf = radixPlan[1];
    std::size_t l1 = 1;
    std::size_t iw = 0;
    assert(input.data() != output.data());

    for (std::size_t k1 = 1; k1 <= nf; k1++) {
        std::size_t       ip  = radixPlan[k1 + 1];
        std::size_t       l2  = ip * l1;
        std::size_t       ido = n_vecs / l2;
        const std::size_t ix2 = iw + ido;

        switch (ip) {
        case 5: {
            std::size_t ix3 = ix2 + ido;
            std::size_t ix4 = ix3 + ido;
            radb5_ps<T>(ido, l1, input, output, wa_span.subspan(iw), wa_span.subspan(ix2), wa_span.subspan(ix3), wa_span.subspan(ix4));
        } break;
        case 4: {
            std::size_t ix3 = ix2 + ido;
            radb4_ps<T>(ido, l1, input, output, wa_span.subspan(iw), wa_span.subspan(ix2), wa_span.subspan(ix3));
        } break;
        case 3: {
            radb3_ps<T>(ido, l1, input, output, wa_span.subspan(iw), wa_span.subspan(ix2));
        } break;
        case 2: radb2_ps<T>(ido, l1, input, output, wa_span.subspan(iw)); break;
        default: assert(0); break;
        }

        l1 = l2;
        iw += (ip - 1) * ido;

        if (output.data() == work2_span.data()) {
            output = work1_span;
            input  = work2_span;
        } else {
            output = work2_span;
            input  = work1_span;
        }
    }
    return input;
}

template<std::array<std::size_t, 5UZ> ntryh>
static constexpr std::size_t decompose(std::size_t n, std::span<std::size_t> radixPlan) {
    std::size_t nl = n, nf = 0;
    for (std::size_t j = 0; ntryh[j]; ++j) {
        const std::size_t ntry = ntryh[j];
        while (nl != 1) {
            std::size_t nq = nl / ntry;
            std::size_t nr = nl - ntry * nq;
            if (nr == 0) {
                radixPlan[2 + nf++] = ntry;
                nl                  = nq;
                if (ntry == 2 && nf != 1) {
                    for (std::size_t i = 2; i <= nf; ++i) {
                        std::size_t ib      = nf - i + 2;
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
    radixPlan[1] = nf;
    return nf;
}

template<std::floating_point T>
inline void rffti1_ps(std::size_t n, std::span<T> wa, std::span<std::size_t, 15> radixPlan) {
    std::size_t nf   = decompose<{4, 2, 3, 5, 0}>(n, radixPlan);
    T           argh = (2 * std::numbers::pi_v<T>) / static_cast<T>(n);
    std::size_t is   = 0;
    std::size_t nfm1 = nf - 1;
    std::size_t l1   = 1;

    for (std::size_t k1 = 1; k1 <= nfm1; k1++) {
        std::size_t ip  = radixPlan[k1 + 1];
        std::size_t ld  = 0;
        std::size_t l2  = l1 * ip;
        std::size_t ido = n / l2;
        std::size_t ipm = ip - 1;

        for (std::size_t j = 1; j <= ipm; ++j) {
            T           argld;
            std::size_t i = is, fi = 0;
            ld += l1;
            argld = static_cast<T>(ld) * argh;
            for (std::size_t ii = 3; ii <= ido; ii += 2) {
                i += 2;
                fi += 1;
                wa[i - 2] = std::cos(T(fi) * argld);
                wa[i - 1] = std::sin(T(fi) * argld);
            }
            is += ido;
        }
        l1 = l2;
    }
}

template<std::floating_point T>
inline void cffti1_ps(std::size_t n, std::span<T> wa, std::span<std::size_t, 15> radixPlan) {
    const std::size_t nf   = decompose<{5, 3, 4, 2, 0}>(n, radixPlan);
    const T           argh = (2 * std::numbers::pi_v<T>) / static_cast<T>(n);
    std::size_t       i    = 1;
    std::size_t       l1   = 1;

    for (std::size_t k1 = 1; k1 <= nf; k1++) {
        std::size_t ip   = radixPlan[k1 + 1];
        std::size_t ld   = 0UZ;
        std::size_t l2   = l1 * ip;
        std::size_t ido  = n / l2;
        std::size_t idot = ido + ido + 2;
        std::size_t ipm  = ip - 1;

        for (std::size_t j = 1; j <= ipm; j++) {
            std::size_t i1 = i, fi = 0;
            wa[i - 1] = 1;
            wa[i]     = 0;
            ld += l1;
            const T argld = static_cast<T>(ld) * argh;
            for (std::size_t ii = 4; ii <= idot; ii += 2) {
                i += 2;
                fi += 1;
                wa[i - 1] = std::cos(T(fi) * argld);
                wa[i]     = std::sin(T(fi) * argld);
            }
            if (ip > 5) {
                wa[i1 - 1] = wa[i - 1];
                wa[i1]     = wa[i];
            }
        }
        l1 = l2;
    }
}

template<int isign, std::floating_point T>
static std::span<T> cfftf1_ps(std::size_t n_vecs, std::span<const T> input_span, std::span<T> work1_span, std::span<T> work2_span, std::span<const T> wa_span, std::span<const std::size_t, 15> radixPlan) {
    const T* RESTRICT input_aligned = std::assume_aligned<64>(input_span.data());
    std::span<T>      in{const_cast<T*>(input_aligned), input_span.size()};
    std::span<T>      out = (in.data() == work2_span.data()) ? work1_span : work2_span;

    std::size_t nf = radixPlan[1];
    std::size_t l1 = 1;
    std::size_t iw = 0;
    for (std::size_t k1 = 2; k1 <= nf + 1; k1++) {
        const std::size_t ip   = radixPlan[k1];
        const std::size_t l2   = ip * l1;
        const std::size_t ido  = n_vecs / l2;
        const std::size_t idot = ido + ido;

        switch (ip) {
        case 5: {
            std::size_t ix2 = iw + idot;
            std::size_t ix3 = ix2 + idot;
            std::size_t ix4 = ix3 + idot;
            passf5_ps<isign, T>(idot, l1, in, out, wa_span.subspan(iw), wa_span.subspan(ix2), wa_span.subspan(ix3), wa_span.subspan(ix4));
        } break;
        case 4: {
            std::size_t ix2 = iw + idot;
            std::size_t ix3 = ix2 + idot;
            passf4_ps<isign, T>(idot, l1, in, out, wa_span.subspan(iw), wa_span.subspan(ix2), wa_span.subspan(ix3));
        } break;
        case 2: {
            passf2_ps<isign, T>(idot, l1, in, out, wa_span.subspan(iw));
        } break;
        case 3: {
            std::size_t ix2 = iw + idot;
            passf3_ps<isign, T>(idot, l1, in, out, wa_span.subspan(iw), wa_span.subspan(ix2));
        } break;
        default: assert(0);
        }

        l1 = l2;
        iw += (ip - 1) * idot;

        if (out.data() == work2_span.data()) {
            out = work1_span;
            in  = work2_span;
        } else {
            out = work2_span;
            in  = work1_span;
        }
    }

    return in; // this is in fact the output (in-place operation)
}

template<std::floating_point T, Transform transform, std::size_t N>
struct PFFFT_Setup {
    using value_type   = T;
    using vector_type  = vec<T, 4>;
    using V            = vector_type;
    using IsRealValued = std::conditional_t<transform == Transform::Real, std::true_type, std::false_type>;
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

    constexpr void checkAlgorithmConstraints() const {
        constexpr std::size_t L = simdSize();
        if ((size() % (L * L)) || size() == 0 || size() < minSize()) {
            std::println(stderr, "incompatible sizes for {}2C: N ({}) must be multiple of {} and >{}", //
                transform == Transform::Real ? "R" : "C", size(), 2 * L * L, minSize());
            std::exit(EXIT_FAILURE);
        }
    }

    constexpr PFFFT_Setup()
    requires(!IsDynamic::value)
    {
        checkAlgorithmConstraints();
        computeTwiddle();
    }

    explicit PFFFT_Setup(std::size_t N_)
    requires(IsDynamic::value)
        : _N(N_) {
        checkAlgorithmConstraints();
        computeTwiddle();
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        if constexpr (IsDynamic::value) {
            return _N;
        } else {
            return N;
        }
    }
    [[nodiscard]] static constexpr std::size_t simdSize() noexcept { return vector_type::size(); }
    [[nodiscard]] static constexpr std::size_t minSize() {
        constexpr std::size_t L = simdSize();
        if constexpr (transform == Transform::Real) {
            return 2UZ * L * L; // min size is N = 32 (SIMD-limit)
        } else { // transform == Transform::Complex
            return L * L; // min size is N = 16
        }
    }
    [[nodiscard]] constexpr std::size_t simdVectorSize() const noexcept {
        return (IsRealValued::value ? (size() / 2) : size()) / simdSize(); // simdVectorSize = number of complex SIMD vectors (N/4 if complex, N/8 if real for lanes=4)
    }

    [[nodiscard]] std::span<const T> butterflyTwiddles() const noexcept { return _butterflyTwiddles; }
    [[nodiscard]] std::span<const T> stageTwiddles() const noexcept { return _stageTwiddles; }
    [[nodiscard]] std::span<T>       scratch() noexcept { return _scratch; }
    [[nodiscard]] std::span<const T> scratch() const noexcept { return _scratch; }

    void computeTwiddle() {
        // not too performance critical, computed usually only once
        constexpr std::size_t L       = simdSize();
        const std::size_t     nScalar = ceil_div(size(), L);
        if constexpr (IsDynamic::value) {
            constexpr std::size_t kGuard = 2UZ;
            _stageTwiddles.resize(2UZ * nScalar + kGuard);
            _butterflyTwiddles.resize(2UZ * (L - 1UZ) * L * ceil_div(simdVectorSize(), L));
            _scratch.resize(std::max(2UZ * simdVectorSize() * L, 8UZ * L));
        }

        // compute stage twiddles & radix plan
        if constexpr (IsRealValued::value) {
            rffti1_ps<T>(nScalar, _stageTwiddles, _radixPlan);
        } else {
            cffti1_ps<T>(nScalar, _stageTwiddles, _radixPlan);
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
            std::println(stderr, "factorization mismatch: prod(radices)={} != size()/SIMD_width={} (N={}, SIMD_width={})", prod, nScalar, size(), vector_type::size());
            std::abort();
        }
    }
};

/* [0 0 1 2 3 4 5 6 7 8] -> [0 8 7 6 5 4 3 2 1] */
template<std::floating_point T>
static void reversed_copy(std::size_t N, const vec<T>* in, std::size_t in_stride, vec<T>* out) {
    vec<T> g0, g1;
    interleave(in[0], in[1], g0, g1);
    in += in_stride;

    *--out = VSWAPHL(g0, g1); /* [g0l, g0h], [g1l g1h] -> [g1l, g0h] */
    for (std::size_t k = 1UZ; k < N; ++k) {
        vec<T> h0, h1;
        interleave(in[0], in[1], h0, h1);
        in += in_stride;
        *--out = VSWAPHL(g1, h0);
        *--out = VSWAPHL(h0, h1);
        g1     = h1;
    }
    *--out = VSWAPHL(g1, g0);
}

template<std::floating_point T>
static void unreversed_copy(std::size_t N, const vec<T>* in, vec<T>* out, int out_stride) {
    const vec<T> g0 = in[0];
    vec<T>       g1 = in[0];
    ++in;
    vec<T> h0, h1;
    for (std::size_t k = 1; k < N; ++k) {
        h0 = *in++;
        h1 = *in++;
        g1 = VSWAPHL(g1, h0);
        h0 = VSWAPHL(h0, h1);
        uninterleave(h0, g1, out[0], out[1]);
        out += out_stride;
        g1 = h1;
    }
    h0 = *in++;
    h1 = g0;
    g1 = VSWAPHL(g1, h0);
    h0 = VSWAPHL(h0, h1);
    uninterleave(h0, g1, out[0], out[1]);
}

template<Direction direction, std::floating_point T, Transform transform, std::size_t N_>
void pffft_zreordering(PFFFT_Setup<T, transform, N_>& setup, std::span<const T> input, std::span<T> output) {
    using V                 = vec<T>;
    constexpr std::size_t L = V::size();

    const std::size_t N     = setup.size();           // scalars
    const std::size_t Ncvec = setup.simdVectorSize(); // complex vectors (each has Re/Im V’s)
    const std::size_t Nvec  = N / L;                  // total V’s

    assert(input.data() != output.data());

    if constexpr (PFFFT_Setup<T, transform, N_>::IsRealValued::value) {
        // ---------------- R2C/C2R (halfcomplex externally) ----------------
        const std::size_t dk   = N / 32; // blocks in V-units per quarter-band
        const T*          inP  = input.data();
        T*                outP = output.data();

        if constexpr (direction == Direction::Forward) {
            // Two interleave streams
            for (std::size_t k = 0; k < dk; ++k) {
                // stream 0: (0,1) -> out[2*k + {0,1}]
                {
                    const V a = load_unchecked<V>(inP + (k * 8 + 0) * L, stdx::vector_aligned);
                    const V b = load_unchecked<V>(inP + (k * 8 + 1) * L, stdx::vector_aligned);
                    V       lo{}, hi{};
                    interleave(a, b, lo, hi);
                    store_unchecked(lo, outP + (2 * k + 0) * L, stdx::vector_aligned);
                    store_unchecked(hi, outP + (2 * k + 1) * L, stdx::vector_aligned);
                }
                // stream 1: (4,5) -> out[(4*dk + 2*k) + {0,1}]
                {
                    const V a = load_unchecked<V>(inP + (k * 8 + 4) * L, stdx::vector_aligned);
                    const V b = load_unchecked<V>(inP + (k * 8 + 5) * L, stdx::vector_aligned);
                    V       lo{}, hi{};
                    interleave(a, b, lo, hi);
                    const std::size_t base = 4 * dk + 2 * k;
                    store_unchecked(lo, outP + (base + 0) * L, stdx::vector_aligned);
                    store_unchecked(hi, outP + (base + 1) * L, stdx::vector_aligned);
                }
            }
            // Two reversed tails into [N/2 - dk .. N/2) and [N - dk .. N)
            const std::size_t mid = (N / 2) / L; // one-past midpoint in V’s
            for (std::size_t m = 0; m < dk; ++m) {
                const V v0 = load_unchecked<V>(inP + (2 + m * 8) * L, stdx::vector_aligned);
                const V v1 = load_unchecked<V>(inP + (6 + m * 8) * L, stdx::vector_aligned);
                store_unchecked(v0, outP + ((mid - 1) - m) * L, stdx::vector_aligned);
                store_unchecked(v1, outP + ((Nvec - 1) - m) * L, stdx::vector_aligned);
            }
        } else {
            // Un-interleave the two streams
            for (std::size_t k = 0; k < dk; ++k) {
                // stream 0: out[2*k + {0,1}] -> (0,1)
                {
                    const V in0 = load_unchecked<V>(inP + (2 * k + 0) * L, stdx::vector_aligned);
                    const V in1 = load_unchecked<V>(inP + (2 * k + 1) * L, stdx::vector_aligned);
                    V       a{}, b{};
                    uninterleave(in0, in1, a, b);
                    store_unchecked(a, outP + (k * 8 + 0) * L, stdx::vector_aligned);
                    store_unchecked(b, outP + (k * 8 + 1) * L, stdx::vector_aligned);
                }
                // stream 1: out[(4*dk + 2*k) + {0,1}] -> (4,5)
                {
                    const std::size_t base = 4 * dk + 2 * k;
                    const V           in0  = load_unchecked<V>(inP + (base + 0) * L, stdx::vector_aligned);
                    const V           in1  = load_unchecked<V>(inP + (base + 1) * L, stdx::vector_aligned);
                    V                 a{}, b{};
                    uninterleave(in0, in1, a, b);
                    store_unchecked(a, outP + (k * 8 + 4) * L, stdx::vector_aligned);
                    store_unchecked(b, outP + (k * 8 + 5) * L, stdx::vector_aligned);
                }
            }
            // Un-reverse: sources start at N/4 and 3N/4, go forward; destinations stride backwards
            const std::size_t src0 = (N / 4) / L;
            const std::size_t src1 = (3 * N / 4) / L;
            for (std::size_t m = 0; m < dk; ++m) {
                const V v0 = load_unchecked<V>(inP + (src0 + m) * L, stdx::vector_aligned);
                const V v1 = load_unchecked<V>(inP + (src1 + m) * L, stdx::vector_aligned);
                store_unchecked(v0, outP + (Nvec - 6 - 8 * m) * L, stdx::vector_aligned);
                store_unchecked(v1, outP + (Nvec - 2 - 8 * m) * L, stdx::vector_aligned);
            }
        }
    } else {
        // ---------------- C2C (interleaved externally) ----------------
        const std::size_t rows = Ncvec / 4; // R: number of 4-wide row groups
        const T*          inP  = input.data();
        T*                outP = output.data();

        if constexpr (direction == Direction::Forward) {
            // k = 4*r + c ; kk = c*R + r
            for (std::size_t c = 0; c < 4; ++c) {
                const std::size_t kk_base = c * rows;
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t k  = (r << 2) | c;
                    const std::size_t kk = kk_base + r;

                    const V re = load_unchecked<V>(inP + (2 * k + 0) * L, stdx::vector_aligned);
                    const V im = load_unchecked<V>(inP + (2 * k + 1) * L, stdx::vector_aligned);
                    V       lo{}, hi{};
                    interleave(re, im, lo, hi);
                    store_unchecked(lo, outP + (2 * kk + 0) * L, stdx::vector_aligned);
                    store_unchecked(hi, outP + (2 * kk + 1) * L, stdx::vector_aligned);
                }
            }
        } else {
            for (std::size_t c = 0; c < 4; ++c) {
                const std::size_t kk_base = c * rows;
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t k  = (r << 2) | c;
                    const std::size_t kk = kk_base + r;

                    const V in0 = load_unchecked<V>(inP + (2 * kk + 0) * L, stdx::vector_aligned);
                    const V in1 = load_unchecked<V>(inP + (2 * kk + 1) * L, stdx::vector_aligned);
                    V       re{}, im{};
                    uninterleave(in0, in1, re, im);
                    store_unchecked(re, outP + (2 * k + 0) * L, stdx::vector_aligned);
                    store_unchecked(im, outP + (2 * k + 1) * L, stdx::vector_aligned);
                }
            }
        }
    }
}

template<std::floating_point T>
static inline void pffft_cplx_finalize(std::size_t Ncvec, std::span<T> input, std::span<T> output, std::span<const T> e) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    const std::size_t dk = Ncvec / L;
    for (std::size_t k = 0; k < dk; ++k) {
        std::size_t base = 8 * k * L;
        V           r0   = load_unchecked<V>(input.data() + base, stdx::vector_aligned);
        V           i0   = load_unchecked<V>(input.data() + base + L, stdx::vector_aligned);
        V           r1   = load_unchecked<V>(input.data() + base + 2 * L, stdx::vector_aligned);
        V           i1   = load_unchecked<V>(input.data() + base + 3 * L, stdx::vector_aligned);
        V           r2   = load_unchecked<V>(input.data() + base + 4 * L, stdx::vector_aligned);
        V           i2   = load_unchecked<V>(input.data() + base + 5 * L, stdx::vector_aligned);
        V           r3   = load_unchecked<V>(input.data() + base + 6 * L, stdx::vector_aligned);
        V           i3   = load_unchecked<V>(input.data() + base + 7 * L, stdx::vector_aligned);

        transpose(r0, r1, r2, r3);
        transpose(i0, i1, i2, i3);

        V e0 = load_unchecked<V>(e.data() + k * 6 * L, stdx::vector_aligned);
        V e1 = load_unchecked<V>(e.data() + (k * 6 + 1) * L, stdx::vector_aligned);
        V e2 = load_unchecked<V>(e.data() + (k * 6 + 2) * L, stdx::vector_aligned);
        V e3 = load_unchecked<V>(e.data() + (k * 6 + 3) * L, stdx::vector_aligned);
        V e4 = load_unchecked<V>(e.data() + (k * 6 + 4) * L, stdx::vector_aligned);
        V e5 = load_unchecked<V>(e.data() + (k * 6 + 5) * L, stdx::vector_aligned);

        complex_multiply(r1, i1, e0, e1);
        complex_multiply(r2, i2, e2, e3);
        complex_multiply(r3, i3, e4, e5);

        // Butterfly operations...
        V sr0 = r0 + r2, dr0 = r0 - r2;
        V sr1 = r1 + r3, dr1 = r1 - r3;
        V si0 = i0 + i2, di0 = i0 - i2;
        V si1 = i1 + i3, di1 = i1 - i3;

        r0 = sr0 + sr1;
        i0 = si0 + si1;
        r1 = dr0 + di1;
        i1 = di0 - dr1;
        r2 = sr0 - sr1;
        i2 = si0 - si1;
        r3 = dr0 - di1;
        i3 = di0 + dr1;

        std::size_t out_idx = 0;
        store_unchecked(r0, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(i0, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(r1, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(i1, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(r2, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(i2, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(r3, output.data() + base + out_idx++ * L, stdx::vector_aligned);
        store_unchecked(i3, output.data() + base + out_idx++ * L, stdx::vector_aligned);
    }
}

template<std::floating_point T>
static inline void pffft_cplx_preprocess(std::size_t Ncvec, std::span<const T> input, std::span<T> output, std::span<const T> e) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    assert(input.data() != output.data());
    const std::size_t dk = Ncvec / L;

    for (std::size_t k = 0; k < dk; ++k) {
        std::size_t base = 8 * k * L;

        V r0 = load_unchecked<V>(input.data() + base, stdx::vector_aligned);
        V i0 = load_unchecked<V>(input.data() + base + L, stdx::vector_aligned);
        V r1 = load_unchecked<V>(input.data() + base + 2 * L, stdx::vector_aligned);
        V i1 = load_unchecked<V>(input.data() + base + 3 * L, stdx::vector_aligned);
        V r2 = load_unchecked<V>(input.data() + base + 4 * L, stdx::vector_aligned);
        V i2 = load_unchecked<V>(input.data() + base + 5 * L, stdx::vector_aligned);
        V r3 = load_unchecked<V>(input.data() + base + 6 * L, stdx::vector_aligned);
        V i3 = load_unchecked<V>(input.data() + base + 7 * L, stdx::vector_aligned);

        V sr0 = r0 + r2, dr0 = r0 - r2;
        V sr1 = r1 + r3, dr1 = r1 - r3;
        V si0 = i0 + i2, di0 = i0 - i2;
        V si1 = i1 + i3, di1 = i1 - i3;

        r0 = sr0 + sr1;
        i0 = si0 + si1;
        r1 = dr0 - di1;
        i1 = di0 + dr1;
        r2 = sr0 - sr1;
        i2 = si0 - si1;
        r3 = dr0 + di1;
        i3 = di0 - dr1;

        V e0 = load_unchecked<V>(e.data() + k * 6 * L, stdx::vector_aligned);
        V e1 = load_unchecked<V>(e.data() + (k * 6 + 1) * L, stdx::vector_aligned);
        V e2 = load_unchecked<V>(e.data() + (k * 6 + 2) * L, stdx::vector_aligned);
        V e3 = load_unchecked<V>(e.data() + (k * 6 + 3) * L, stdx::vector_aligned);
        V e4 = load_unchecked<V>(e.data() + (k * 6 + 4) * L, stdx::vector_aligned);
        V e5 = load_unchecked<V>(e.data() + (k * 6 + 5) * L, stdx::vector_aligned);

        complex_multiply_conj(r1, i1, e0, e1);
        complex_multiply_conj(r2, i2, e2, e3);
        complex_multiply_conj(r3, i3, e4, e5);

        transpose(r0, r1, r2, r3);
        transpose(i0, i1, i2, i3);

        std::size_t out_idx = base;
        store_unchecked(r0, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(i0, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(r1, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(i1, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(r2, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(i2, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(r3, output.data() + out_idx, stdx::vector_aligned);
        out_idx += L;
        store_unchecked(i3, output.data() + out_idx, stdx::vector_aligned);
    }
}

template<std::floating_point T>
static ALWAYS_INLINE(void) pffft_real_finalize_4x4(const T* RESTRICT in0, const T* RESTRICT in1, const T* RESTRICT in, std::span<const T> eSpan, T* RESTRICT out) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();
    const T* RESTRICT     e = std::assume_aligned<64>(eSpan.data());

    V r0 = load_unchecked<V>(in0, stdx::vector_aligned);
    V i0 = load_unchecked<V>(in1, stdx::vector_aligned);
    V r1 = load_unchecked<V>(in, stdx::vector_aligned);
    in += L;
    V i1 = load_unchecked<V>(in, stdx::vector_aligned);
    in += L;
    V r2 = load_unchecked<V>(in, stdx::vector_aligned);
    in += L;
    V i2 = load_unchecked<V>(in, stdx::vector_aligned);
    in += L;
    V r3 = load_unchecked<V>(in, stdx::vector_aligned);
    in += L;
    V i3 = load_unchecked<V>(in, stdx::vector_aligned);

    transpose(r0, r1, r2, r3);
    transpose(i0, i1, i2, i3);

    V e0 = load_unchecked<V>(e + 0UZ * L, stdx::vector_aligned);
    V e1 = load_unchecked<V>(e + 1UZ * L, stdx::vector_aligned);
    V e2 = load_unchecked<V>(e + 2UZ * L, stdx::vector_aligned);
    V e3 = load_unchecked<V>(e + 3UZ * L, stdx::vector_aligned);
    V e4 = load_unchecked<V>(e + 4UZ * L, stdx::vector_aligned);
    V e5 = load_unchecked<V>(e + 5UZ * L, stdx::vector_aligned);

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
static inline void pffft_real_finalize(std::size_t Ncvec, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<const T> e) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    assert(inputSpan.data() != outputSpan.data());
    assert(isAligned<64>(inputSpan.data()));
    assert(isAligned<64>(outputSpan.data()));
    const T* RESTRICT input  = std::assume_aligned<64>(inputSpan.data());
    T* RESTRICT       output = std::assume_aligned<64>(outputSpan.data());

    V cr = load_unchecked<V>(input, stdx::vector_aligned);
    V ci = load_unchecked<V>(input + (Ncvec * 2 - 1) * L, stdx::vector_aligned);

    constexpr T inv_sqrt2 = std::numbers::sqrt2_v<T> / 2;

    alignas(64) T zero_storage[L] = {};
    pffft_real_finalize_4x4(zero_storage, zero_storage, input + L, e.subspan(0, 6 * L), output);

    // special handling for DC and Nyquist
    output[0 * L + 0] = (cr[0] + cr[2]) + (cr[1] + cr[3]); // DC
    output[1 * L + 0] = (cr[0] + cr[2]) - (cr[1] + cr[3]); // Nyquist
    output[4 * L + 0] = (cr[0] - cr[2]);
    output[5 * L + 0] = (cr[3] - cr[1]);
    output[2 * L + 0] = ci[0] + inv_sqrt2 * (ci[1] - ci[3]);
    output[3 * L + 0] = -ci[2] - inv_sqrt2 * (ci[1] + ci[3]);
    output[6 * L + 0] = ci[0] - inv_sqrt2 * (ci[1] - ci[3]);
    output[7 * L + 0] = ci[2] - inv_sqrt2 * (ci[1] + ci[3]);

    const std::size_t dx   = Ncvec / L;
    V                 save = load_unchecked<V>(input + 7 * L, stdx::vector_aligned);

    for (std::size_t k = 1; k < dx; ++k) {
        V             save_next = load_unchecked<V>(input + (8 * k + 7) * L, stdx::vector_aligned);
        alignas(64) T save_storage[L];
        store_unchecked(save, save_storage, stdx::vector_aligned);
        pffft_real_finalize_4x4(save_storage, input + 8 * k * L, input + (8 * k + 1) * L, e.subspan(k * 6 * L, 6 * L), output + k * 8 * L);
        save = save_next;
    }
}

template<std::floating_point T>
static ALWAYS_INLINE(void) pffft_real_preprocess_4x4(const T* RESTRICT in, std::span<const T> eSpan, T* RESTRICT out, std::size_t first) {
    using V                 = vec<T, 4>;
    constexpr std::size_t L = V::size();

    V r0 = load_unchecked<V>(in, stdx::vector_aligned);
    V i0 = load_unchecked<V>(in + L, stdx::vector_aligned);
    V r1 = load_unchecked<V>(in + 2 * L, stdx::vector_aligned);
    V i1 = load_unchecked<V>(in + 3 * L, stdx::vector_aligned);
    V r2 = load_unchecked<V>(in + 4 * L, stdx::vector_aligned);
    V i2 = load_unchecked<V>(in + 5 * L, stdx::vector_aligned);
    V r3 = load_unchecked<V>(in + 6 * L, stdx::vector_aligned);
    V i3 = load_unchecked<V>(in + 7 * L, stdx::vector_aligned);

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

    V e0 = load_unchecked<V>(e, stdx::vector_aligned);
    V e1 = load_unchecked<V>(e + L, stdx::vector_aligned);
    V e2 = load_unchecked<V>(e + 2 * L, stdx::vector_aligned);
    V e3 = load_unchecked<V>(e + 3 * L, stdx::vector_aligned);
    V e4 = load_unchecked<V>(e + 4 * L, stdx::vector_aligned);
    V e5 = load_unchecked<V>(e + 5 * L, stdx::vector_aligned);

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
static inline void pffft_real_preprocess(std::size_t Ncvec, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<const T> e) {
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

    pffft_real_preprocess_4x4(inputSpan.data(), e.subspan(0, 6 * L), output + L, 1);

    const std::size_t dk = Ncvec / L;
    for (std::size_t k = 1; k < dk; ++k) {
        pffft_real_preprocess_4x4(inputSpan.data() + 8 * k * L, e.subspan(k * 6 * L, 6 * L), output + (k * 8 - 1) * L, 0);
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

template<Direction direction, Order ordering, std::floating_point T, Transform transform, std::size_t N_>
constexpr void transformInternal(PFFFT_Setup<T, transform, N_>& setup, std::span<const T> inputSpan, std::span<T> outputSpan, std::span<T> scratch) {
    using V = PFFFT_Setup<T, transform, N_>::vector_type;

    if constexpr (transform == Transform::Real) {
        inputSpan  = std::span{inputSpan.data(), setup.size()};
        outputSpan = std::span{outputSpan.data(), setup.size()};
    }

    std::span<T>  buff[2]    = {outputSpan, scratch};
    constexpr int orderinged = (ordering == Order::Ordered) ? 1 : 0;
    const int     nf_odd     = setup._radixPlan[1] & 1;

    std::size_t ib = (nf_odd ^ orderinged) ? 1 : 0;

    // Use Ncvec * 2 for real FFTs (number of vec<T> units), Ncvec for complex
    constexpr std::size_t L           = V::size();
    const std::size_t     Ncvec       = setup.simdVectorSize();
    const std::size_t     n_vecs_real = Ncvec * 2; // Number of vec<T> units for real FFT
    const std::size_t     n_vecs_cplx = Ncvec;     // Number of vec<T> units for complex FFT

    std::span<const T> twiddle_span = setup.stageTwiddles();
    std::span<const T> e_span       = setup.butterflyTwiddles();

    if constexpr (direction == Direction::Forward) {
        ib = !ib;
        if constexpr (PFFFT_Setup<T, transform, N_>::IsRealValued::value) {
            std::span<T> outp = rfftf1_ps<T>(n_vecs_real, inputSpan, buff[ib], buff[!ib], twiddle_span, setup._radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            pffft_real_finalize<T>(Ncvec, buff[ib], buff[!ib], e_span);
        } else {
            // deinterleave
            const T* RESTRICT pInput  = std::assume_aligned<64>(inputSpan.data());
            T* RESTRICT       pBuffer = std::assume_aligned<64>(buff[ib].data());
            for (std::size_t k = 0; k < Ncvec; ++k) {
                const std::size_t k2 = 2 * k * L;

                V v0 = load_unchecked<V>(pInput + k2, stdx::vector_aligned);
                V v1 = load_unchecked<V>(pInput + k2 + L, stdx::vector_aligned);
                V r, i;
                uninterleave(v0, v1, r, i);
                store_unchecked(r, pBuffer + k2, stdx::vector_aligned);
                store_unchecked(i, pBuffer + k2 + L, stdx::vector_aligned);
            }

            std::span<T> outp = cfftf1_ps<-1, T>(n_vecs_cplx, buff[ib], buff[!ib], buff[ib], twiddle_span, setup._radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            pffft_cplx_finalize(Ncvec, buff[ib], buff[!ib], e_span);
        }

        if constexpr (ordering == Order::Ordered) {
            pffft_zreordering<Direction::Forward>(setup, std::span<const T>{buff[!ib]}, buff[ib]);
        } else {
            ib = !ib;
        }
    } else {
        if (inputSpan.data() == buff[ib].data()) {
            ib = !ib;
        }

        if constexpr (ordering == Order::Ordered) {
            pffft_zreordering<Direction::Backward>(setup, inputSpan, buff[ib]);
            inputSpan = buff[ib];
            ib        = !ib;
        }

        if constexpr (PFFFT_Setup<T, transform, N_>::IsRealValued::value) {
            pffft_real_preprocess<T>(Ncvec, inputSpan, buff[ib], e_span);

            std::span<T> outp = rfftb1_ps<T>(n_vecs_real, buff[ib], buff[0], buff[1], twiddle_span, setup._radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;
        } else {
            pffft_cplx_preprocess(Ncvec, inputSpan, buff[ib], e_span);

            std::span<T> outp = cfftf1_ps<+1, T>(n_vecs_cplx, buff[ib], buff[0], buff[1], twiddle_span, setup._radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            // interleave
            T* RESTRICT pBuffer = std::assume_aligned<64>(buff[ib].data());
            for (std::size_t k = 0; k < Ncvec; ++k) {
                const std::size_t k2 = 2 * k * L;

                V r = load_unchecked<V>(pBuffer + k2, stdx::vector_aligned);
                V i = load_unchecked<V>(pBuffer + k2 + L, stdx::vector_aligned);
                V v0, v1;
                interleave(r, i, v0, v1);
                store_unchecked(v0, pBuffer + k2, stdx::vector_aligned);
                store_unchecked(v1, pBuffer + k2 + L, stdx::vector_aligned);
            }
        }
    }

    if (buff[ib].data() != outputSpan.data()) {
        T* RESTRICT pBuffer = std::assume_aligned<64>(buff[ib].data());
        T* RESTRICT pOutput = std::assume_aligned<64>(outputSpan.data());
        std::memcpy(pBuffer, pOutput, Ncvec * 2 * L * sizeof(T));
        ib = !ib;
    }
    assert(buff[ib].data() == outputSpan.data());
}

template<std::floating_point T, Transform transform, std::size_t N_>
void zconvolve_accumulate(PFFFT_Setup<T, transform, N_>& s, const T* a, const T* b, T* ab, T scaling) {
    std::size_t Ncvec = s.simdVectorSize();

    const T ar  = a[0];
    const T ai  = a[4];
    const T br  = b[0];
    const T bi  = b[4];
    const T abr = ab[0];
    const T abi = ab[4];

    /* default routine, works fine for non-arm cpus with current compilers */
    const vec<T>       vscal = scaling;
    std::span<const T> sa(a, Ncvec * 8);
    std::span<const T> sb(b, Ncvec * 8);
    std::span<T>       sab(ab, Ncvec * 8);
    vir::transform(vir::execution::simd.prefer_size<8UZ>().unroll_by<2UZ>(), std::views::zip(sa, sb, sab), sab, [=](const auto& tup) {
        const auto& [va, vb, vab] = tup;
        if constexpr (va.size() == 8UZ) {
            auto [ar_, ai_]   = split<4, 4>(va);
            auto [br_, bi_]   = split<4, 4>(vb);
            auto [abr_, abi_] = split<4, 4>(vab);
            complex_multiply(ar_, ai_, br_, bi_);
            return concat((ar * vscal + abr_), (ai * vscal + abi_));
        } else {
            __builtin_trap(); // this should be impossible
            return vab;       // to get the expected return type
        }
    });

    if constexpr (PFFFT_Setup<T, transform, N_>::IsRealValued::value) {
        ab[0] = abr + ar * br * scaling;
        ab[4] = abi + ai * bi * scaling;
    }
}

template<std::floating_point T, Transform transform, std::size_t N_>
void pffft_zconvolve_no_accu(PFFFT_Setup<T, transform, N_>& s, const T* a, const T* b, T* ab, T scaling) {
    const vec<T>      vscal       = scaling;
    const std::size_t NcvecMulTwo = 2 * s.simdVectorSize(); /* std::size_t Ncvec = s.simdVectorSize(); */

    const T sar = a[0];
    const T sai = a[vec<T>::size()];
    const T sbr = b[0];
    const T sbi = b[vec<T>::size()];

    /* default routine, works fine for non-arm cpus with current compilers */
    for (std::size_t k = 0; k < NcvecMulTwo; k += 4) {
        vec<T> var = load_unchecked<vec<T>>(a + (k + 0) * vec<T>::size(), stdx::vector_aligned);
        vec<T> vai = load_unchecked<vec<T>>(a + (k + 1) * vec<T>::size(), stdx::vector_aligned);
        vec<T> vbr = load_unchecked<vec<T>>(b + (k + 0) * vec<T>::size(), stdx::vector_aligned);
        vec<T> vbi = load_unchecked<vec<T>>(b + (k + 1) * vec<T>::size(), stdx::vector_aligned);
        complex_multiply(var, vai, vbr, vbi);
        store_unchecked(var * vscal, ab + (k + 0) * vec<T>::size(), stdx::vector_aligned);
        store_unchecked(vai * vscal, ab + (k + 1) * vec<T>::size(), stdx::vector_aligned);
        var = load_unchecked<vec<T>>(a + (k + 2) * vec<T>::size(), stdx::vector_aligned);
        vai = load_unchecked<vec<T>>(a + (k + 3) * vec<T>::size(), stdx::vector_aligned);
        vbr = load_unchecked<vec<T>>(b + (k + 2) * vec<T>::size(), stdx::vector_aligned);
        vbi = load_unchecked<vec<T>>(b + (k + 3) * vec<T>::size(), stdx::vector_aligned);
        complex_multiply(var, vai, vbr, vbi);
        store_unchecked(var * vscal, ab + (k + 2) * vec<T>::size(), stdx::vector_aligned);
        store_unchecked(vai * vscal, ab + (k + 3) * vec<T>::size(), stdx::vector_aligned);
    }

    if constexpr (PFFFT_Setup<T, transform, N_>::IsRealValued::value) {
        ab[0]              = sar * sbr * scaling;
        ab[vec<T>::size()] = sai * sbi * scaling;
    }
}

template<Direction direction, Order ordering, std::floating_point T, Transform transform, std::size_t N, InBuf<T> Rin, OutBuf<T> Rout>
void pffft_transform(PFFFT_Setup<T, transform, N>& setup, Rin&& in, Rout&& out, std::source_location loc) {
    constexpr std::size_t kAlignment = PFFFT_Setup<T, transform, N>::kAlignment;
    const auto            inputSpan  = std::span<const T>(std::ranges::data(in), std::ranges::size(in));
    auto                  outputSpan = std::span<T>(std::ranges::data(out), std::ranges::size(out));

    const std::size_t need = (transform == Transform::Real) ? setup.size() : 2 * setup.size();
    if (inputSpan.size() < need || outputSpan.size() < need) {
        throw gr::exception(std::format("size mismatch: input({}) output({}) setup({})", inputSpan.size(), outputSpan.size(), need), loc);
    }

    if (!isAligned<kAlignment>(inputSpan.data())) {
        throw gr::exception(std::format("input is not {}-bytes aligned", kAlignment), loc);
    }
    if (!isAligned<kAlignment>(outputSpan.data())) {
        throw gr::exception(std::format("output is not {}-bytes aligned", kAlignment), loc);
    }

    transformInternal<direction, ordering>(setup, inputSpan, outputSpan, setup.scratch());
}