#include "pffft.h"

/* detect compiler flavour */
#if defined(_MSC_VER)
#define COMPILER_MSVC
#elif defined(__GNUC__)
#define COMPILER_GCC
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

#include "pf_stdx_simd.h"

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
    static constexpr T inv_sqrt2   = T{0.5} * std::numbers::sqrt2_v<T>;
};

template<std::floating_point T>
consteval std::size_t pffft_simd_size(void) {
    return SIMD_SZ;
}

template<std::floating_point T>
std::size_t pffft_min_fft_size(fft_transform_t transform) {
    /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
       and 32 for real FFTs -- a lot of stuff would need to be rewritten to
       handle other cases (or maybe just switch to a scalar fft, I don't know..) */
    constexpr std::size_t simdSz = pffft_simd_size<T>();
    if (transform == fft_transform_t::Real) {
        return (2UZ * simdSz * simdSz);
    } else if (transform == fft_transform_t::Complex) {
        return (simdSz * simdSz);
    } else {
        return 1UZ;
    }
}

template<std::floating_point T>
constexpr bool pffft_is_valid_size(std::size_t N, fft_transform_t cplx) {
    const std::size_t N_min = pffft_min_fft_size<T>(cplx); // checks for Radix-5, -3, and -2
    for (std::size_t factor : {5UZ, 3UZ, 2UZ}) {
        while (N >= factor * N_min && N % factor == 0UZ) {
            N /= factor;
        }
    }
    return N == N_min;
}

template<std::floating_point T>
constexpr std::size_t pffft_nearest_transform_size(std::size_t N, fft_transform_t cplx, bool higher) {
    std::size_t       d;
    const std::size_t N_min = pffft_min_fft_size<T>(cplx);
    if (N < N_min) {
        N = N_min;
    }
    d = higher ? N_min : -N_min;
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
template<std::floating_point T>
static NEVER_INLINE(void) passf2_ps(std::size_t ido, std::size_t l1, const v4sf<T>* cc, v4sf<T>* ch, const T* wa1, T fsign) {
    std::size_t l1ido = l1 * ido;
    if (ido <= 2) {
        for (std::size_t k = 0; k < l1ido; k += ido, ch += ido, cc += 2 * ido) {
            ch[0]         = cc[0] + cc[ido + 0];
            ch[l1ido]     = cc[0] - cc[ido + 0];
            ch[1]         = cc[1] + cc[ido + 1];
            ch[l1ido + 1] = cc[1] - cc[ido + 1];
        }
    } else {
        for (std::size_t k = 0UZ; k < l1ido; k += ido, ch += ido, cc += 2 * ido) {
            for (std::size_t i = 0UZ; i < ido - 1UZ; i += 2UZ) {
                v4sf<T> tr2 = cc[i + 0] - cc[i + ido + 0];
                v4sf<T> ti2 = cc[i + 1] - cc[i + ido + 1];
                v4sf<T> wr = wa1[i], wi = fsign * wa1[i + 1];
                ch[i]     = cc[i + 0] + cc[i + ido + 0];
                ch[i + 1] = cc[i + 1] + cc[i + ido + 1];
                VCPLXMUL(tr2, ti2, wr, wi);
                ch[i + l1ido]     = tr2;
                ch[i + l1ido + 1] = ti2;
            }
        }
    }
}

/*
  passf3 and passb3 has been merged here, fsign = -1 for passf3, +1 for passb3
*/
template<std::floating_point T>
static NEVER_INLINE(void) passf3_ps(std::size_t ido, std::size_t l1, const v4sf<T>* cc, v4sf<T>* ch, const T* wa1, const T* wa2, T fsign) {
    const T     taui = 0.866025403784439f * fsign;
    v4sf<T>     tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
    std::size_t l1ido = l1 * ido;
    T           wr1, wi1, wr2, wi2;
    assert(ido > 2);
    for (std::size_t k = 0UZ; k < l1ido; k += ido, cc += 3 * ido, ch += ido) {
        for (std::size_t i = 0UZ; i < ido - 1; i += 2) {
            tr2       = cc[i + ido] + cc[i + 2 * ido];
            cr2       = cc[i] + FFTConstants<T>::taur * tr2;
            ch[i]     = cc[i] + tr2;
            ti2       = cc[i + ido + 1] + cc[i + 2 * ido + 1];
            ci2       = cc[i + 1] + FFTConstants<T>::taur * ti2;
            ch[i + 1] = cc[i + 1] + ti2;
            cr3       = taui * (cc[i + ido] - cc[i + 2 * ido]);
            ci3       = taui * (cc[i + ido + 1] - cc[i + 2 * ido + 1]);
            dr2       = cr2 - ci3;
            dr3       = cr2 + ci3;
            di2       = ci2 + cr3;
            di3       = ci2 - cr3;
            wr1 = wa1[i], wi1 = fsign * wa1[i + 1], wr2 = wa2[i], wi2 = fsign * wa2[i + 1];
            VCPLXMUL(dr2, di2, wr1, wi1);
            ch[i + l1ido]     = dr2;
            ch[i + l1ido + 1] = di2;
            VCPLXMUL(dr3, di3, wr2, wi2);
            ch[i + 2 * l1ido]     = dr3;
            ch[i + 2 * l1ido + 1] = di3;
        }
    }
} /* passf3 */

template<std::floating_point T>
static NEVER_INLINE(void) passf4_ps(std::size_t ido, std::size_t l1, const v4sf<T>* cc, v4sf<T>* ch, const T* wa1, const T* wa2, const T* wa3, T fsign) {
    /* isign == -1 for forward transform and +1 for backward transform */
    v4sf<T>     ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    std::size_t l1ido = l1 * ido;
    if (ido == 2) {
        for (std::size_t k = 0; k < l1ido; k += ido, ch += ido, cc += 4 * ido) {
            tr1 = cc[0] - cc[2 * ido + 0];
            tr2 = cc[0] + cc[2 * ido + 0];
            ti1 = cc[1] - cc[2 * ido + 1];
            ti2 = cc[1] + cc[2 * ido + 1];
            ti4 = (cc[1 * ido + 0] - cc[3 * ido + 0]) * fsign;
            tr4 = (cc[3 * ido + 1] - cc[1 * ido + 1]) * fsign;
            tr3 = cc[ido + 0] + cc[3 * ido + 0];
            ti3 = cc[ido + 1] + cc[3 * ido + 1];

            ch[0 * l1ido + 0] = tr2 + tr3;
            ch[0 * l1ido + 1] = ti2 + ti3;
            ch[1 * l1ido + 0] = tr1 + tr4;
            ch[1 * l1ido + 1] = ti1 + ti4;
            ch[2 * l1ido + 0] = tr2 - tr3;
            ch[2 * l1ido + 1] = ti2 - ti3;
            ch[3 * l1ido + 0] = tr1 - tr4;
            ch[3 * l1ido + 1] = ti1 - ti4;
        }
    } else {
        for (std::size_t k = 0; k < l1ido; k += ido, ch += ido, cc += 4 * ido) {
            for (std::size_t i = 0; i < ido - 1; i += 2) {
                T wr1, wi1, wr2, wi2, wr3, wi3;
                tr1 = cc[i + 0] - cc[i + 2 * ido + 0];
                tr2 = cc[i + 0] + cc[i + 2 * ido + 0];
                ti1 = cc[i + 1] - cc[i + 2 * ido + 1];
                ti2 = cc[i + 1] + cc[i + 2 * ido + 1];
                tr4 = (cc[i + 3 * ido + 1] - cc[i + 1 * ido + 1]) * fsign;
                ti4 = (cc[i + 1 * ido + 0] - cc[i + 3 * ido + 0]) * fsign;
                tr3 = cc[i + ido + 0] + cc[i + 3 * ido + 0];
                ti3 = cc[i + ido + 1] + cc[i + 3 * ido + 1];

                ch[i]     = tr2 + tr3;
                cr3       = tr2 - tr3;
                ch[i + 1] = ti2 + ti3;
                ci3       = ti2 - ti3;

                cr2 = tr1 + tr4;
                cr4 = tr1 - tr4;
                ci2 = ti1 + ti4;
                ci4 = ti1 - ti4;
                wr1 = wa1[i], wi1 = fsign * wa1[i + 1];
                VCPLXMUL(cr2, ci2, wr1, wi1);
                wr2 = wa2[i], wi2 = fsign * wa2[i + 1];
                ch[i + l1ido]     = cr2;
                ch[i + l1ido + 1] = ci2;

                VCPLXMUL(cr3, ci3, wr2, wi2);
                wr3 = wa3[i], wi3 = fsign * wa3[i + 1];
                ch[i + 2 * l1ido]     = cr3;
                ch[i + 2 * l1ido + 1] = ci3;

                VCPLXMUL(cr4, ci4, wr3, wi3);
                ch[i + 3 * l1ido]     = cr4;
                ch[i + 3 * l1ido + 1] = ci4;
            }
        }
    }
} /* passf4 */

/*
  passf5 and passb5 has been merged here, fsign = -1 for passf5, +1 for passb5
*/
template<std::floating_point T>
static NEVER_INLINE(void) passf5_ps(std::size_t ido, std::size_t l1, const v4sf<T>* cc, v4sf<T>* ch, const T* wa1, const T* wa2, const T* wa3, const T* wa4, T fsign) {
    constexpr T tr11 = .309016994374947f;
    const T     ti11 = .951056516295154f * fsign;
    constexpr T tr12 = -.809016994374947f;
    const T     ti12 = .587785252292473f * fsign;

    /* Local variables */
    v4sf<T> ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;

    T wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

    constexpr auto cc_ref = [](const v4sf<T>* cc_, std::size_t ido_, std::size_t a_1, std::size_t a_2) -> const v4sf<T>& { return cc_[(a_2 - 1) * ido_ + a_1 + 1]; };

    auto ch_ref = [](v4sf<T>* ch_, std::size_t l1_, std::size_t ido_, std::size_t a_1, std::size_t a_3) -> v4sf<T>& { return ch_[(a_3 - 1) * l1_ * ido_ + a_1 + 1]; };

    assert(ido > 2);
    for (std::size_t k = 0; k < l1; ++k, cc += 5 * ido, ch += ido) {
        for (std::size_t i = 0; i < ido - 1; i += 2) {
            ti5                           = cc_ref(cc, ido, i, 2) - cc_ref(cc, ido, i, 5);
            ti2                           = cc_ref(cc, ido, i, 2) + cc_ref(cc, ido, i, 5);
            ti4                           = cc_ref(cc, ido, i, 3) - cc_ref(cc, ido, i, 4);
            ti3                           = cc_ref(cc, ido, i, 3) + cc_ref(cc, ido, i, 4);
            tr5                           = cc_ref(cc, ido, i - 1, 2) - cc_ref(cc, ido, i - 1, 5);
            tr2                           = cc_ref(cc, ido, i - 1, 2) + cc_ref(cc, ido, i - 1, 5);
            tr4                           = cc_ref(cc, ido, i - 1, 3) - cc_ref(cc, ido, i - 1, 4);
            tr3                           = cc_ref(cc, ido, i - 1, 3) + cc_ref(cc, ido, i - 1, 4);
            ch_ref(ch, l1, ido, i - 1, 1) = cc_ref(cc, ido, i - 1, 1) + (tr2 + tr3);
            ch_ref(ch, l1, ido, i, 1)     = cc_ref(cc, ido, i, 1) + (ti2 + ti3);
            cr2                           = cc_ref(cc, ido, i - 1, 1) + (tr11 * tr2 + tr12 * tr3);
            ci2                           = cc_ref(cc, ido, i, 1) + (tr11 * ti2 + tr12 * ti3);
            cr3                           = cc_ref(cc, ido, i - 1, 1) + (tr12 * tr2 + tr11 * tr3);
            ci3                           = cc_ref(cc, ido, i, 1) + (tr12 * ti2 + tr11 * ti3);
            cr5                           = (ti11 * tr5) + ti12 * tr4;
            ci5                           = (ti11 * ti5) + ti12 * ti4;
            cr4                           = (ti12 * tr5) - ti11 * tr4;
            ci4                           = (ti12 * ti5) - ti11 * ti4;
            dr3                           = cr3 - ci4;
            dr4                           = cr3 + ci4;
            di3                           = ci3 + cr4;
            di4                           = ci3 - cr4;
            dr5                           = cr2 + ci5;
            dr2                           = cr2 - ci5;
            di5                           = ci2 - cr5;
            di2                           = ci2 + cr5;
            wr1 = wa1[i], wi1 = fsign * wa1[i + 1], wr2 = wa2[i], wi2 = fsign * wa2[i + 1];
            wr3 = wa3[i], wi3 = fsign * wa3[i + 1], wr4 = wa4[i], wi4 = fsign * wa4[i + 1];
            VCPLXMUL(dr2, di2, wr1, wi1);
            ch_ref(ch, l1, ido, i - 1, 2) = dr2;
            ch_ref(ch, l1, ido, i, 2)     = di2;
            VCPLXMUL(dr3, di3, wr2, wi2);
            ch_ref(ch, l1, ido, i - 1, 3) = dr3;
            ch_ref(ch, l1, ido, i, 3)     = di3;
            VCPLXMUL(dr4, di4, wr3, wi3);
            ch_ref(ch, l1, ido, i - 1, 4) = dr4;
            ch_ref(ch, l1, ido, i, 4)     = di4;
            VCPLXMUL(dr5, di5, wr4, wi4);
            ch_ref(ch, l1, ido, i - 1, 5) = dr5;
            ch_ref(ch, l1, ido, i, 5)     = di5;
        }
    }
}

template<std::floating_point T>
static NEVER_INLINE(void) radf2_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* wa1) {
    constexpr T minus_one = -1.f;
    std::size_t l1ido     = l1 * ido;
    for (std::size_t k = 0; k < l1ido; k += ido) {
        v4sf<T> a = cc[k], b = cc[k + l1ido];
        ch[2 * k]             = a + b;
        ch[2 * (k + ido) - 1] = a - b;
    }
    if (ido < 2) {
        return;
    }
    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 2; i < ido; i += 2) {
                v4sf<T> tr2 = cc[i - 1 + k + l1ido], ti2 = cc[i + k + l1ido];
                v4sf<T> br = cc[i - 1 + k], bi = cc[i + k];
                VCPLXMULCONJ(tr2, ti2, wa1[i - 2], wa1[i - 1]);
                ch[i + 2 * k]             = bi + ti2;
                ch[2 * (k + ido) - i]     = ti2 - bi;
                ch[i - 1 + 2 * k]         = br + tr2;
                ch[2 * (k + ido) - i - 1] = br - tr2;
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }
    for (std::size_t k = 0; k < l1ido; k += ido) {
        ch[2 * k + ido]     = minus_one * cc[ido - 1 + k + l1ido];
        ch[2 * k + ido - 1] = cc[k + ido - 1];
    }
} /* radf2 */

template<std::floating_point T>
static NEVER_INLINE(void) radb2_ps(std::size_t ido, std::size_t l1, const v4sf<T>* cc, v4sf<T>* ch, const T* wa1) {
    constexpr T minus_two = -2;
    std::size_t l1ido     = l1 * ido;
    v4sf<T>     a, b, c, d, tr2, ti2;
    for (std::size_t k = 0; k < l1ido; k += ido) {
        a             = cc[2 * k];
        b             = cc[2 * (k + ido) - 1];
        ch[k]         = a + b;
        ch[k + l1ido] = a - b;
    }
    if (ido < 2) {
        return;
    }
    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            for (std::size_t i = 2; i < ido; i += 2) {
                a             = cc[i - 1 + 2 * k];
                b             = cc[2 * (k + ido) - i - 1];
                c             = cc[i + 0 + 2 * k];
                d             = cc[2 * (k + ido) - i + 0];
                ch[i - 1 + k] = a + b;
                tr2           = a - b;
                ch[i + 0 + k] = c - d;
                ti2           = c + d;
                VCPLXMUL(tr2, ti2, wa1[i - 2], wa1[i - 1]);
                ch[i - 1 + k + l1ido] = tr2;
                ch[i + 0 + k + l1ido] = ti2;
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }
    for (std::size_t k = 0; k < l1ido; k += ido) {
        a                       = cc[2 * k + ido - 1];
        b                       = cc[2 * k + ido];
        ch[k + ido - 1]         = a + a;
        ch[k + ido - 1 + l1ido] = minus_two * b;
    }
} /* radb2 */

template<std::floating_point T>
static void radf3_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* wa1, const T* wa2) {
    constexpr T taui = 0.866025403784439f;
    std::size_t ic;
    v4sf<T>     ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3, wr1, wi1, wr2, wi2;
    for (std::size_t k = 0; k < l1; k++) {
        cr2                             = cc[(k + l1) * ido] + cc[(k + 2 * l1) * ido];
        ch[3 * k * ido]                 = cc[k * ido] + cr2;
        ch[(3 * k + 2) * ido]           = taui * (cc[(k + l1 * 2) * ido] - cc[(k + l1) * ido]);
        ch[ido - 1 + (3 * k + 1) * ido] = cc[k * ido] + FFTConstants<T>::taur * cr2;
    }
    if (ido == 1) {
        return;
    }
    for (std::size_t k = 0; k < l1; k++) {
        for (std::size_t i = 2; i < ido; i += 2) {
            ic  = ido - i;
            wr1 = wa1[i - 2];
            wi1 = wa1[i - 1];
            dr2 = cc[i - 1 + (k + l1) * ido];
            di2 = cc[i + (k + l1) * ido];
            VCPLXMULCONJ(dr2, di2, wr1, wi1);

            wr2 = wa2[i - 2];
            wi2 = wa2[i - 1];
            dr3 = cc[i - 1 + (k + l1 * 2) * ido];
            di3 = cc[i + (k + l1 * 2) * ido];
            VCPLXMULCONJ(dr3, di3, wr2, wi2);

            cr2                            = dr2 + dr3;
            ci2                            = di2 + di3;
            ch[i - 1 + 3 * k * ido]        = cc[i - 1 + k * ido] + cr2;
            ch[i + 3 * k * ido]            = cc[i + k * ido] + ci2;
            tr2                            = cc[i - 1 + k * ido] + FFTConstants<T>::taur * cr2;
            ti2                            = cc[i + k * ido] + FFTConstants<T>::taur * ci2;
            tr3                            = taui * (di2 - di3);
            ti3                            = taui * (dr3 - dr2);
            ch[i - 1 + (3 * k + 2) * ido]  = tr2 + tr3;
            ch[ic - 1 + (3 * k + 1) * ido] = tr2 - tr3;
            ch[i + (3 * k + 2) * ido]      = ti2 + ti3;
            ch[ic + (3 * k + 1) * ido]     = ti3 - ti2;
        }
    }
} /* radf3 */

template<std::floating_point T>
static void radb3_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* wa1, const T* wa2) {
    constexpr T taur   = -0.5f;
    constexpr T taui   = 0.866025403784439f;
    constexpr T taui_2 = 0.866025403784439f * 2;
    std::size_t ic;
    v4sf<T>     ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    for (std::size_t k = 0; k < l1; k++) {
        tr2                    = cc[ido - 1 + (3 * k + 1) * ido];
        tr2                    = tr2 + tr2;
        cr2                    = taur * tr2 + cc[3 * k * ido];
        ch[k * ido]            = cc[3 * k * ido] + tr2;
        ci3                    = taui_2 * cc[(3 * k + 2) * ido];
        ch[(k + l1) * ido]     = cr2 - ci3;
        ch[(k + 2 * l1) * ido] = cr2 + ci3;
    }
    if (ido == 1) {
        return;
    }
    for (std::size_t k = 0; k < l1; k++) {
        for (std::size_t i = 2; i < ido; i += 2) {
            ic                  = ido - i;
            tr2                 = cc[i - 1 + (3 * k + 2) * ido] + cc[ic - 1 + (3 * k + 1) * ido];
            cr2                 = taur * tr2 + cc[i - 1 + 3 * k * ido];
            ch[i - 1 + k * ido] = cc[i - 1 + 3 * k * ido] + tr2;
            ti2                 = cc[i + (3 * k + 2) * ido] - cc[ic + (3 * k + 1) * ido];
            ci2                 = taur * ti2 + cc[i + 3 * k * ido];
            ch[i + k * ido]     = cc[i + 3 * k * ido] + ti2;
            cr3                 = taui * (cc[i - 1 + (3 * k + 2) * ido] - cc[ic - 1 + (3 * k + 1) * ido]);
            ci3                 = taui * (cc[i + (3 * k + 2) * ido] + cc[ic + (3 * k + 1) * ido]);
            dr2                 = cr2 - ci3;
            dr3                 = cr2 + ci3;
            di2                 = ci2 + cr3;
            di3                 = ci2 - cr3;
            VCPLXMUL(dr2, di2, wa1[i - 2], wa1[i - 1]);
            ch[i - 1 + (k + l1) * ido] = dr2;
            ch[i + (k + l1) * ido]     = di2;
            VCPLXMUL(dr3, di3, wa2[i - 2], wa2[i - 1]);
            ch[i - 1 + (k + 2 * l1) * ido] = dr3;
            ch[i + (k + 2 * l1) * ido]     = di3;
        }
    }
} /* radb3 */

template<std::floating_point T>
static NEVER_INLINE(void) radf4_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* RESTRICT wa1, const T* RESTRICT wa2, const T* RESTRICT wa3) {
    constexpr T minus_hsqt2 = -0.7071067811865475f;
    std::size_t l1ido       = l1 * ido;
    {
        const v4sf<T>*RESTRICT cc_ = cc, *RESTRICT cc_end = cc + l1ido;
        v4sf<T>* RESTRICT ch_ = ch;
        while (cc < cc_end) {
            /* this loop represents between 25% and 40% of total radf4_ps cost ! */
            v4sf<T> a0 = cc[0], a1 = cc[l1ido];
            v4sf<T> a2 = cc[2 * l1ido], a3 = cc[3 * l1ido];
            v4sf<T> tr1     = a1 + a3;
            v4sf<T> tr2     = a0 + a2;
            ch[2 * ido - 1] = a0 - a2;
            ch[2 * ido]     = a3 - a1;
            ch[0]           = tr1 + tr2;
            ch[4 * ido - 1] = tr2 - tr1;
            cc += ido;
            ch += 4 * ido;
        }
        cc = cc_;
        ch = ch_;
    }
    if (ido < 2) {
        return;
    }
    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            const v4sf<T>* RESTRICT pc = const_cast<v4sf<T>*>(cc + 1 + k);
            for (std::size_t i = 2; i < ido; i += 2, pc += 2) {
                std::size_t ic = ido - i;
                v4sf<T>     wr, wi, cr2, ci2, cr3, ci3, cr4, ci4;
                v4sf<T>     tr1, ti1, tr2, ti2, tr3, ti3, tr4, ti4;

                cr2 = pc[1 * l1ido + 0];
                ci2 = pc[1 * l1ido + 1];
                wr  = wa1[i - 2];
                wi  = wa1[i - 1];
                VCPLXMULCONJ(cr2, ci2, wr, wi);

                cr3 = pc[2 * l1ido + 0];
                ci3 = pc[2 * l1ido + 1];
                wr  = wa2[i - 2];
                wi  = wa2[i - 1];
                VCPLXMULCONJ(cr3, ci3, wr, wi);

                cr4 = pc[3 * l1ido];
                ci4 = pc[3 * l1ido + 1];
                wr  = wa3[i - 2];
                wi  = wa3[i - 1];
                VCPLXMULCONJ(cr4, ci4, wr, wi);

                /* at this point, on SSE, five of "cr2 cr3 cr4 ci2 ci3 ci4" should be loaded in registers */

                tr1                          = cr2 + cr4;
                tr4                          = cr4 - cr2;
                tr2                          = pc[0] + cr3;
                tr3                          = pc[0] - cr3;
                ch[i - 1 + 4 * k]            = tr1 + tr2;
                ch[ic - 1 + 4 * k + 3 * ido] = tr2 - tr1; /* at this point tr1 and tr2 can be disposed */
                ti1                          = ci2 + ci4;
                ti4                          = ci2 - ci4;
                ch[i - 1 + 4 * k + 2 * ido]  = ti4 + tr3;
                ch[ic - 1 + 4 * k + 1 * ido] = tr3 - ti4; /* dispose tr3, ti4 */
                ti2                          = pc[1] + ci3;
                ti3                          = pc[1] - ci3;
                ch[i + 4 * k]                = ti1 + ti2;
                ch[ic + 4 * k + 3 * ido]     = ti1 - ti2;
                ch[i + 4 * k + 2 * ido]      = tr4 + ti3;
                ch[ic + 4 * k + 1 * ido]     = tr4 - ti3;
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }
    for (std::size_t k = 0; k < l1ido; k += ido) {
        v4sf<T> a = cc[ido - 1 + k + l1ido], b = cc[ido - 1 + k + 3 * l1ido];
        v4sf<T> c = cc[ido - 1 + k], d = cc[ido - 1 + k + 2 * l1ido];
        v4sf<T> ti1                   = minus_hsqt2 * (a + b);
        v4sf<T> tr1                   = minus_hsqt2 * (b - a);
        ch[ido - 1 + 4 * k]           = tr1 + c;
        ch[ido - 1 + 4 * k + 2 * ido] = c - tr1;
        ch[4 * k + 1 * ido]           = ti1 - d;
        ch[4 * k + 3 * ido]           = ti1 + d;
    }
} /* radf4 */

template<std::floating_point T>
static NEVER_INLINE(void) radb4_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* RESTRICT wa1, const T* RESTRICT wa2, const T* RESTRICT wa3) {
    constexpr T minus_sqrt2 = -1.414213562373095f;
    constexpr T two         = 2.f;
    std::size_t l1ido       = l1 * ido;
    v4sf<T>     ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    {
        const v4sf<T>*RESTRICT cc_ = cc, *RESTRICT ch_end = ch + l1ido;
        v4sf<T>* ch_ = ch;
        while (ch < ch_end) {
            v4sf<T> a = cc[0], b = cc[4 * ido - 1];
            v4sf<T> c = cc[2 * ido], d = cc[2 * ido - 1];
            tr3           = two * d;
            tr2           = a + b;
            tr1           = a - b;
            tr4           = two * c;
            ch[0 * l1ido] = tr2 + tr3;
            ch[2 * l1ido] = tr2 - tr3;
            ch[1 * l1ido] = tr1 - tr4;
            ch[3 * l1ido] = tr1 + tr4;

            cc += 4 * ido;
            ch += ido;
        }
        cc = cc_;
        ch = ch_;
    }
    if (ido < 2) {
        return;
    }
    if (ido != 2) {
        for (std::size_t k = 0; k < l1ido; k += ido) {
            const v4sf<T>* RESTRICT pc = reinterpret_cast<const v4sf<T>*>(cc - 1 + 4 * k);
            v4sf<T>* RESTRICT       ph = reinterpret_cast<v4sf<T>*>(ch + k + 1);
            for (std::size_t i = 2; i < ido; i += 2) {

                tr1   = pc[i] - pc[4 * ido - i];
                tr2   = pc[i] + pc[4 * ido - i];
                ti4   = pc[2 * ido + i] - pc[2 * ido - i];
                tr3   = pc[2 * ido + i] + pc[2 * ido - i];
                ph[0] = tr2 + tr3;
                cr3   = tr2 - tr3;

                ti3 = pc[2 * ido + i + 1] - pc[2 * ido - i + 1];
                tr4 = pc[2 * ido + i + 1] + pc[2 * ido - i + 1];
                cr2 = tr1 - tr4;
                cr4 = tr1 + tr4;

                ti1 = pc[i + 1] + pc[4 * ido - i + 1];
                ti2 = pc[i + 1] - pc[4 * ido - i + 1];

                ph[1] = ti2 + ti3;
                ph += l1ido;
                ci3 = ti2 - ti3;
                ci2 = ti1 + ti4;
                ci4 = ti1 - ti4;
                VCPLXMUL(cr2, ci2, wa1[i - 2], wa1[i - 1]);
                ph[0] = cr2;
                ph[1] = ci2;
                ph += l1ido;
                VCPLXMUL(cr3, ci3, wa2[i - 2], wa2[i - 1]);
                ph[0] = cr3;
                ph[1] = ci3;
                ph += l1ido;
                VCPLXMUL(cr4, ci4, wa3[i - 2], wa3[i - 1]);
                ph[0] = cr4;
                ph[1] = ci4;
                ph    = ph - 3 * l1ido + 2;
            }
        }
        if (ido % 2 == 1) {
            return;
        }
    }
    for (std::size_t k = 0; k < l1ido; k += ido) {
        std::size_t i0 = 4 * k + ido;
        v4sf<T>     c = cc[i0 - 1], d = cc[i0 + 2 * ido - 1];
        v4sf<T>     a = cc[i0 + 0], b = cc[i0 + 2 * ido + 0];
        tr1                         = c - d;
        tr2                         = c + d;
        ti1                         = b + a;
        ti2                         = b - a;
        ch[ido - 1 + k + 0 * l1ido] = tr2 + tr2;
        ch[ido - 1 + k + 1 * l1ido] = minus_sqrt2 * (ti1 - tr1);
        ch[ido - 1 + k + 2 * l1ido] = ti2 + ti2;
        ch[ido - 1 + k + 3 * l1ido] = minus_sqrt2 * (ti1 + tr1);
    }
} /* radb4 */

template<std::floating_point T>
static void radf5_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* wa1, const T* wa2, const T* wa3, const T* wa4) {
    constexpr T tr11 = .309016994374947f;
    constexpr T ti11 = .951056516295154f;
    constexpr T tr12 = -.809016994374947f;
    constexpr T ti12 = .587785252292473f;

    /* system generated locals */
    std::size_t cc_offset, ch_offset;

    /* Local variables */
    std::size_t ic;
    v4sf<T>     ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, dr4, dr5, cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
    std::size_t idp2;

    constexpr auto cc_ref = [](const v4sf<T>* cc_, std::size_t ido_, std::size_t l1_, std::size_t cc_offset_, std::size_t a_1, std::size_t a_2, std::size_t a_3) -> const v4sf<T>& { return cc_[cc_offset_ - 1 + ((a_3)*l1_ + (a_2)) * ido_ + a_1]; };

    auto ch_ref = [](v4sf<T>* ch_, std::size_t ido_, std::size_t ch_offset_, std::size_t a_1, std::size_t a_2, std::size_t a_3) -> v4sf<T>& { return ch_[ch_offset_ - 1 + ((a_3) * 5 + (a_2)) * ido_ + a_1]; };

    /* Parameter adjustments */
    ch_offset = 1 + ido * 6;
    ch -= ch_offset;
    cc_offset = 1 + ido * (1 + l1);
    cc -= cc_offset;

    /* Function Body */
    for (std::size_t k = 1; k <= l1; ++k) {
        cr2                                   = cc_ref(cc, ido, l1, cc_offset, 1, k, 5) + cc_ref(cc, ido, l1, cc_offset, 1, k, 2);
        ci5                                   = cc_ref(cc, ido, l1, cc_offset, 1, k, 5) - cc_ref(cc, ido, l1, cc_offset, 1, k, 2);
        cr3                                   = cc_ref(cc, ido, l1, cc_offset, 1, k, 4) + cc_ref(cc, ido, l1, cc_offset, 1, k, 3);
        ci4                                   = cc_ref(cc, ido, l1, cc_offset, 1, k, 4) - cc_ref(cc, ido, l1, cc_offset, 1, k, 3);
        ch_ref(ch, ido, ch_offset, 1, 1, k)   = cc_ref(cc, ido, l1, cc_offset, 1, k, 1) + (cr2 + cr3);
        ch_ref(ch, ido, ch_offset, ido, 2, k) = cc_ref(cc, ido, l1, cc_offset, 1, k, 1) + (tr11 * cr2 + tr12 * cr3);
        ch_ref(ch, ido, ch_offset, 1, 3, k)   = ti11 * ci5 + ti12 * ci4;
        ch_ref(ch, ido, ch_offset, ido, 4, k) = cc_ref(cc, ido, l1, cc_offset, 1, k, 1) + (tr12 * cr2 + tr11 * cr3);
        ch_ref(ch, ido, ch_offset, 1, 5, k)   = ti12 * ci5 - ti11 * ci4;
    }
    if (ido == 1) {
        return;
    }
    idp2 = ido + 2;
    for (std::size_t k = 1; k <= l1; ++k) {
        for (std::size_t i = 3; i <= ido; i += 2) {
            ic  = idp2 - i;
            dr2 = wa1[i - 3];
            di2 = wa1[i - 2];
            dr3 = wa2[i - 3];
            di3 = wa2[i - 2];
            dr4 = wa3[i - 3];
            di4 = wa3[i - 2];
            dr5 = wa4[i - 3];
            di5 = wa4[i - 2];
            VCPLXMULCONJ(dr2, di2, cc_ref(cc, ido, l1, cc_offset, i - 1, k, 2), cc_ref(cc, ido, l1, cc_offset, i, k, 2));
            VCPLXMULCONJ(dr3, di3, cc_ref(cc, ido, l1, cc_offset, i - 1, k, 3), cc_ref(cc, ido, l1, cc_offset, i, k, 3));
            VCPLXMULCONJ(dr4, di4, cc_ref(cc, ido, l1, cc_offset, i - 1, k, 4), cc_ref(cc, ido, l1, cc_offset, i, k, 4));
            VCPLXMULCONJ(dr5, di5, cc_ref(cc, ido, l1, cc_offset, i - 1, k, 5), cc_ref(cc, ido, l1, cc_offset, i, k, 5));
            cr2                                      = dr2 + dr5;
            ci5                                      = dr5 - dr2;
            cr5                                      = di2 - di5;
            ci2                                      = di2 + di5;
            cr3                                      = dr3 + dr4;
            ci4                                      = dr4 - dr3;
            cr4                                      = di3 - di4;
            ci3                                      = di3 + di4;
            ch_ref(ch, ido, ch_offset, i - 1, 1, k)  = cc_ref(cc, ido, l1, cc_offset, i - 1, k, 1) + (cr2 + cr3);
            ch_ref(ch, ido, ch_offset, i, 1, k)      = cc_ref(cc, ido, l1, cc_offset, i, k, 1) - (ci2 + ci3);
            tr2                                      = cc_ref(cc, ido, l1, cc_offset, i - 1, k, 1) + (tr11 * cr2 + tr12 * cr3);
            ti2                                      = cc_ref(cc, ido, l1, cc_offset, i, k, 1) - (tr11 * ci2 + tr12 * ci3);
            tr3                                      = cc_ref(cc, ido, l1, cc_offset, i - 1, k, 1) + (tr12 * cr2 + tr11 * cr3);
            ti3                                      = cc_ref(cc, ido, l1, cc_offset, i, k, 1) - (tr12 * ci2 + tr11 * ci3);
            tr5                                      = ti11 * cr5 + ti12 * cr4;
            ti5                                      = ti11 * ci5 + ti12 * ci4;
            tr4                                      = ti12 * cr5 - ti11 * cr4;
            ti4                                      = ti12 * ci5 - ti11 * ci4;
            ch_ref(ch, ido, ch_offset, i - 1, 3, k)  = tr2 - tr5;
            ch_ref(ch, ido, ch_offset, ic - 1, 2, k) = tr2 + tr5;
            ch_ref(ch, ido, ch_offset, i, 3, k)      = ti2 + ti5;
            ch_ref(ch, ido, ch_offset, ic, 2, k)     = ti5 - ti2;
            ch_ref(ch, ido, ch_offset, i - 1, 5, k)  = tr3 - tr4;
            ch_ref(ch, ido, ch_offset, ic - 1, 4, k) = tr3 + tr4;
            ch_ref(ch, ido, ch_offset, i, 5, k)      = ti3 + ti4;
            ch_ref(ch, ido, ch_offset, ic, 4, k)     = ti4 - ti3;
        }
    }
} /* radf5 */

template<std::floating_point T>
static void radb5_ps(std::size_t ido, std::size_t l1, const v4sf<T>* RESTRICT cc, v4sf<T>* RESTRICT ch, const T* wa1, const T* wa2, const T* wa3, const T* wa4) {
    constexpr T tr11 = .309016994374947f;
    constexpr T ti11 = .951056516295154f;
    constexpr T tr12 = -.809016994374947f;
    constexpr T ti12 = .587785252292473f;

    std::size_t cc_offset, ch_offset;

    /* Local variables */
    std::size_t ic;
    v4sf<T>     ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
    std::size_t idp2;

    constexpr auto cc_ref = [](const v4sf<T>* cc_, std::size_t ido_, std::size_t cc_offset_, std::size_t a_1, std::size_t a_2, std::size_t a_3) -> const v4sf<T>& { return cc_[cc_offset_ - 1 + ((a_3) * 5 + (a_2)) * ido_ + a_1]; };

    auto ch_ref = [](v4sf<T>* ch_, std::size_t ido_, std::size_t l1_, std::size_t ch_offset_, std::size_t a_1, std::size_t a_2, std::size_t a_3) -> v4sf<T>& { return ch_[ch_offset_ - 1 + ((a_3)*l1_ + (a_2)) * ido_ + a_1]; };

    /* Parameter adjustments */
    ch_offset = 1 + ido * (1 + l1);
    ch -= ch_offset;
    cc_offset = 1 + ido * 6;
    cc -= cc_offset;

    /* Function Body */
    for (std::size_t k = 1; k <= l1; ++k) {
        ti5                                     = cc_ref(cc, ido, cc_offset, 1, 3, k) + cc_ref(cc, ido, cc_offset, 1, 3, k);
        ti4                                     = cc_ref(cc, ido, cc_offset, 1, 5, k) + cc_ref(cc, ido, cc_offset, 1, 5, k);
        tr2                                     = cc_ref(cc, ido, cc_offset, ido, 2, k) + cc_ref(cc, ido, cc_offset, ido, 2, k);
        tr3                                     = cc_ref(cc, ido, cc_offset, ido, 4, k) + cc_ref(cc, ido, cc_offset, ido, 4, k);
        ch_ref(ch, ido, l1, ch_offset, 1, k, 1) = cc_ref(cc, ido, cc_offset, 1, 1, k) + (tr2 + tr3);
        cr2                                     = cc_ref(cc, ido, cc_offset, 1, 1, k) + (tr11 * tr2 + tr12 * tr3);
        cr3                                     = cc_ref(cc, ido, cc_offset, 1, 1, k) + (tr12 * tr2 + tr11 * tr3);
        ci5                                     = ti11 * ti5 + ti12 * ti4;
        ci4                                     = ti12 * ti5 + ti11 * ti4;
        ch_ref(ch, ido, l1, ch_offset, 1, k, 2) = cr2 - ci5;
        ch_ref(ch, ido, l1, ch_offset, 1, k, 3) = cr3 - ci4;
        ch_ref(ch, ido, l1, ch_offset, 1, k, 4) = cr3 + ci4;
        ch_ref(ch, ido, l1, ch_offset, 1, k, 5) = cr2 + ci5;
    }
    if (ido == 1) {
        return;
    }
    idp2 = ido + 2;
    for (std::size_t k = 1; k <= l1; ++k) {
        for (std::size_t i = 3; i <= ido; i += 2) {
            ic                                          = idp2 - i;
            ti5                                         = cc_ref(cc, ido, cc_offset, i, 3, k) + cc_ref(cc, ido, cc_offset, ic, 2, k);
            ti2                                         = cc_ref(cc, ido, cc_offset, i, 3, k) - cc_ref(cc, ido, cc_offset, ic, 2, k);
            ti4                                         = cc_ref(cc, ido, cc_offset, i, 5, k) + cc_ref(cc, ido, cc_offset, ic, 4, k);
            ti3                                         = cc_ref(cc, ido, cc_offset, i, 5, k) - cc_ref(cc, ido, cc_offset, ic, 4, k);
            tr5                                         = cc_ref(cc, ido, cc_offset, i - 1, 3, k) - cc_ref(cc, ido, cc_offset, ic - 1, 2, k);
            tr2                                         = cc_ref(cc, ido, cc_offset, i - 1, 3, k) + cc_ref(cc, ido, cc_offset, ic - 1, 2, k);
            tr4                                         = cc_ref(cc, ido, cc_offset, i - 1, 5, k) - cc_ref(cc, ido, cc_offset, ic - 1, 4, k);
            tr3                                         = cc_ref(cc, ido, cc_offset, i - 1, 5, k) + cc_ref(cc, ido, cc_offset, ic - 1, 4, k);
            ch_ref(ch, ido, l1, ch_offset, i - 1, k, 1) = cc_ref(cc, ido, cc_offset, i - 1, 1, k) + (tr2 + tr3);
            ch_ref(ch, ido, l1, ch_offset, i, k, 1)     = cc_ref(cc, ido, cc_offset, i, 1, k) + (ti2 + ti3);
            cr2                                         = cc_ref(cc, ido, cc_offset, i - 1, 1, k) + (tr11 * tr2 + tr12 * tr3);
            ci2                                         = cc_ref(cc, ido, cc_offset, i, 1, k) + (tr11 * ti2 + tr12 * ti3);
            cr3                                         = cc_ref(cc, ido, cc_offset, i - 1, 1, k) + (tr12 * tr2 + tr11 * tr3);
            ci3                                         = cc_ref(cc, ido, cc_offset, i, 1, k) + (tr12 * ti2 + tr11 * ti3);
            cr5                                         = ti11 * tr5 + ti12 * tr4;
            ci5                                         = ti11 * ti5 + ti12 * ti4;
            cr4                                         = ti12 * tr5 - ti11 * tr4;
            ci4                                         = ti12 * ti5 - ti11 * ti4;
            dr3                                         = cr3 - ci4;
            dr4                                         = cr3 + ci4;
            di3                                         = ci3 + cr4;
            di4                                         = ci3 - cr4;
            dr5                                         = cr2 + ci5;
            dr2                                         = cr2 - ci5;
            di5                                         = ci2 - cr5;
            di2                                         = ci2 + cr5;
            VCPLXMUL(dr2, di2, wa1[i - 3], wa1[i - 2]);
            VCPLXMUL(dr3, di3, wa2[i - 3], wa2[i - 2]);
            VCPLXMUL(dr4, di4, wa3[i - 3], wa3[i - 2]);
            VCPLXMUL(dr5, di5, wa4[i - 3], wa4[i - 2]);

            ch_ref(ch, ido, l1, ch_offset, i - 1, k, 2) = dr2;
            ch_ref(ch, ido, l1, ch_offset, i, k, 2)     = di2;
            ch_ref(ch, ido, l1, ch_offset, i - 1, k, 3) = dr3;
            ch_ref(ch, ido, l1, ch_offset, i, k, 3)     = di3;
            ch_ref(ch, ido, l1, ch_offset, i - 1, k, 4) = dr4;
            ch_ref(ch, ido, l1, ch_offset, i, k, 4)     = di4;
            ch_ref(ch, ido, l1, ch_offset, i - 1, k, 5) = dr5;
            ch_ref(ch, ido, l1, ch_offset, i, k, 5)     = di5;
        }
    }
} /* radb5_ps */

template<std::floating_point T>
static NEVER_INLINE(v4sf<T>*) rfftf1_ps(std::size_t n, const v4sf<T>* input_readonly, v4sf<T>* work1, v4sf<T>* work2, const T* wa, const std::size_t* ifac) {
    v4sf<T>*    in  = const_cast<v4sf<T>*>(input_readonly);
    v4sf<T>*    out = in == work2 ? work1 : work2;
    std::size_t nf  = ifac[1];
    std::size_t l2  = n;
    std::size_t iw  = n - 1UZ;
    assert(in != out && work1 != work2);
    for (std::size_t k1 = 1; k1 <= nf; ++k1) {
        std::size_t kh  = nf - k1;
        std::size_t ip  = ifac[kh + 2];
        std::size_t l1  = l2 / ip;
        std::size_t ido = n / l2;
        iw -= (ip - 1) * ido;
        switch (ip) {
        case 5: {
            std::size_t ix2 = iw + ido;
            std::size_t ix3 = ix2 + ido;
            std::size_t ix4 = ix3 + ido;
            radf5_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
        } break;
        case 4: {
            std::size_t ix2 = iw + ido;
            std::size_t ix3 = ix2 + ido;
            radf4_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3]);
        } break;
        case 3: {
            std::size_t ix2 = iw + ido;
            radf3_ps(ido, l1, in, out, &wa[iw], &wa[ix2]);
        } break;
        case 2: radf2_ps(ido, l1, in, out, &wa[iw]); break;
        default: assert(0); break;
        }
        l2 = l1;
        if (out == work2) {
            out = work1;
            in  = work2;
        } else {
            out = work2;
            in  = work1;
        }
    }
    return in; /* this is in fact the output .. */
} /* rfftf1 */

template<std::floating_point T>
static NEVER_INLINE(v4sf<T>*) rfftb1_ps(std::size_t n, const v4sf<T>* input_readonly, v4sf<T>* work1, v4sf<T>* work2, const T* wa, const std::size_t* ifac) {
    v4sf<T>*    in  = const_cast<v4sf<T>*>(input_readonly);
    v4sf<T>*    out = in == work2 ? work1 : work2;
    std::size_t nf  = ifac[1];
    std::size_t l1  = 1;
    std::size_t iw  = 0;
    assert(in != out);
    for (std::size_t k1 = 1; k1 <= nf; k1++) {
        std::size_t ip  = ifac[k1 + 1];
        std::size_t l2  = ip * l1;
        std::size_t ido = n / l2;
        switch (ip) {
        case 5: {
            std::size_t ix2 = iw + ido;
            std::size_t ix3 = ix2 + ido;
            std::size_t ix4 = ix3 + ido;
            radb5_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
        } break;
        case 4: {
            std::size_t ix2 = iw + ido;
            std::size_t ix3 = ix2 + ido;
            radb4_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3]);
        } break;
        case 3: {
            std::size_t ix2 = iw + ido;
            radb3_ps(ido, l1, in, out, &wa[iw], &wa[ix2]);
        } break;
        case 2: radb2_ps(ido, l1, in, out, &wa[iw]); break;
        default: assert(0); break;
        }
        l1 = l2;
        iw += (ip - 1) * ido;

        if (out == work2) {
            out = work1;
            in  = work2;
        } else {
            out = work2;
            in  = work1;
        }
    }
    return in; /* this is in fact the output .. */
}

static std::size_t decompose(std::size_t n, std::size_t* ifac, const std::size_t* ntryh) {
    std::size_t nl = n, nf = 0;
    for (std::size_t j = 0; ntryh[j]; ++j) {
        std::size_t ntry = ntryh[j];
        while (nl != 1) {
            std::size_t nq = nl / ntry;
            std::size_t nr = nl - ntry * nq;
            if (nr == 0) {
                ifac[2 + nf++] = ntry;
                nl             = nq;
                if (ntry == 2 && nf != 1) {
                    for (std::size_t i = 2; i <= nf; ++i) {
                        std::size_t ib = nf - i + 2;
                        ifac[ib + 1UZ] = ifac[ib];
                    }
                    ifac[2UZ] = 2;
                }
            } else {
                break;
            }
        }
    }
    ifac[0] = n;
    ifac[1] = nf;
    return nf;
}

template<typename T>
static void rffti1_ps(std::size_t n, T* wa, std::size_t* ifac) {
    constexpr std::size_t ntryh[] = {4, 2, 3, 5, 0};

    std::size_t nf   = decompose(n, ifac, ntryh);
    T           argh = (2 * std::numbers::pi_v<T>) / static_cast<T>(n);
    std::size_t is   = 0;
    std::size_t nfm1 = nf - 1;
    std::size_t l1   = 1;
    for (std::size_t k1 = 1; k1 <= nfm1; k1++) {
        std::size_t ip  = ifac[k1 + 1];
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
} /* rffti1 */

template<typename T>
static void cffti1_ps(std::size_t n, T* wa, std::size_t* ifac) {
    constexpr std::size_t ntryh[] = {5, 3, 4, 2, 0};

    std::size_t nf   = decompose(n, ifac, ntryh);
    T           argh = (2 * std::numbers::pi_v<T>) / static_cast<T>(n);
    std::size_t i    = 1;
    std::size_t l1   = 1;
    for (std::size_t k1 = 1; k1 <= nf; k1++) {
        std::size_t ip   = ifac[k1 + 1];
        std::size_t ld   = 0UZ;
        std::size_t l2   = l1 * ip;
        std::size_t ido  = n / l2;
        std::size_t idot = ido + ido + 2;
        std::size_t ipm  = ip - 1;
        for (std::size_t j = 1; j <= ipm; j++) {
            T           argld;
            std::size_t i1 = i, fi = 0;
            wa[i - 1] = 1;
            wa[i]     = 0;
            ld += l1;
            argld = static_cast<T>(ld) * argh;
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
} /* cffti1 */

template<std::floating_point T>
static v4sf<T>* cfftf1_ps(std::size_t n, const v4sf<T>* input_readonly, v4sf<T>* work1, v4sf<T>* work2, const T* wa, const std::size_t* ifac, bool isign) {
    v4sf<T>*    in  = const_cast<v4sf<T>*>(input_readonly);
    v4sf<T>*    out = in == work2 ? work1 : work2;
    std::size_t nf  = ifac[1];
    std::size_t l1  = 1;
    std::size_t iw  = 0;
    assert(in != out && work1 != work2);
    for (std::size_t k1 = 2; k1 <= nf + 1; k1++) {
        std::size_t ip   = ifac[k1];
        std::size_t l2   = ip * l1;
        std::size_t ido  = n / l2;
        std::size_t idot = ido + ido;
        switch (ip) {
        case 5: {
            std::size_t ix2 = iw + idot;
            std::size_t ix3 = ix2 + idot;
            std::size_t ix4 = ix3 + idot;
            passf5_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], static_cast<T>(isign));
        } break;
        case 4: {
            std::size_t ix2 = iw + idot;
            std::size_t ix3 = ix2 + idot;
            passf4_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], static_cast<T>(isign));
        } break;
        case 2: {
            passf2_ps(idot, l1, in, out, &wa[iw], static_cast<T>(isign));
        } break;
        case 3: {
            std::size_t ix2 = iw + idot;
            passf3_ps(idot, l1, in, out, &wa[iw], &wa[ix2], static_cast<T>(isign));
        } break;
        default: assert(0);
        }
        l1 = l2;
        iw += (ip - 1) * idot;
        if (out == work2) {
            out = work1;
            in  = work2;
        } else {
            out = work2;
            in  = work1;
        }
    }

    return in; /* this is in fact the output .. */
}

template<std::floating_point T, fft_transform_t transform_>
struct PFFFT_Setup {
    using value_type                           = T;
    static constexpr fft_transform_t transform = transform_;
    std::size_t                      N;
    std::size_t                      Ncvec; /* nb of complex simd vectors (N/4 if PFFFT_COMPLEX, N/8 if PFFFT_REAL) */
    std::size_t                      ifac[15UZ];
    v4sf<T>*                         data;    /* allocated room for twiddle coefs */
    T*                               e;       /* points into 'data', N/4*3 elements */
    T*                               twiddle; /* points into 'data', N/4 elements */
};

template<std::floating_point T, fft_transform_t transform>
PFFFT_Setup<T, transform>* pffft_new_setup(std::size_t N) {
    PFFFT_Setup<T, transform>* s = nullptr;
    /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
       and 32 for real FFTs -- a lot of stuff would need to be rewritten to
       handle other cases (or maybe just switch to a scalar fft, I don't know..) */
    if constexpr (transform == fft_transform_t::Real) {
        if ((N % (2 * SIMD_SZ * SIMD_SZ)) || N <= 0) {
            return s;
        }
    }
    if constexpr (transform == fft_transform_t::Complex) {
        if ((N % (SIMD_SZ * SIMD_SZ)) || N <= 0) {
            return s;
        }
    }
    // s = static_cast<PFFFT_Setup<T, transform>*>(malloc(sizeof(PFFFT_Setup<T, transform>)));
    s = new PFFFT_Setup<T, transform>();
    /* assert((N % 32) == 0); */
    s->N = N;
    // s->transform = transform;
    /* nb of complex simd vectors */
    s->Ncvec   = (PFFFT_Setup<T, transform>::transform == fft_transform_t::Real ? N / 2 : N) / SIMD_SZ;
    s->data    = pffft_aligned_malloc<v4sf<T>>(2 * s->Ncvec * sizeof(v4sf<T>));
    s->e       = reinterpret_cast<T*>(s->data);
    s->twiddle = reinterpret_cast<T*>(s->data + (2 * s->Ncvec * (SIMD_SZ - 1)) / SIMD_SZ);

    if constexpr (transform == fft_transform_t::Real) {
        for (std::size_t k = 0; k < s->Ncvec; ++k) {
            std::size_t i = k / SIMD_SZ;
            std::size_t j = k % SIMD_SZ;
            for (std::size_t m = 0; m < SIMD_SZ - 1; ++m) {
                T A                                       = -2 * std::numbers::pi_v<T> * static_cast<T>(m + 1) * static_cast<T>(k) / static_cast<T>(N);
                s->e[(2 * (i * 3 + m) + 0) * SIMD_SZ + j] = std::cos(A);
                s->e[(2 * (i * 3 + m) + 1) * SIMD_SZ + j] = std::sin(A);
            }
        }
        rffti1_ps(N / SIMD_SZ, s->twiddle, s->ifac);
    } else {
        for (std::size_t k = 0; k < s->Ncvec; ++k) {
            std::size_t i = k / SIMD_SZ;
            std::size_t j = k % SIMD_SZ;
            for (std::size_t m = 0; m < SIMD_SZ - 1; ++m) {
                T A                                       = -2 * std::numbers::pi_v<T> * static_cast<T>(m + 1UZ) * static_cast<T>(k) / static_cast<T>(N);
                s->e[(2 * (i * 3 + m) + 0) * SIMD_SZ + j] = std::cos(A);
                s->e[(2 * (i * 3 + m) + 1) * SIMD_SZ + j] = std::sin(A);
            }
        }
        cffti1_ps(N / SIMD_SZ, s->twiddle, s->ifac);
    }

    /* check that N is decomposable with allowed prime factors */
    std::size_t m = 1;
    for (std::size_t k = 0; k < s->ifac[1]; ++k) {
        m *= s->ifac[2 + k];
    }
    if (m != N / SIMD_SZ) {
        pffft_destroy_setup(s);
        s = nullptr;
    }

    return s;
}

template<std::floating_point T, fft_transform_t transform>
void pffft_destroy_setup(PFFFT_Setup<T, transform>* s) {
    if (!s) {
        return;
    }
    pffft_aligned_free(s->data);
    delete s;
}

/* [0 0 1 2 3 4 5 6 7 8] -> [0 8 7 6 5 4 3 2 1] */
template<std::floating_point T>
static void reversed_copy(std::size_t N, const v4sf<T>* in, std::size_t in_stride, v4sf<T>* out) {
    v4sf<T> g0, g1;
    INTERLEAVE2(in[0], in[1], g0, g1);
    in += in_stride;

    *--out = VSWAPHL(g0, g1); /* [g0l, g0h], [g1l g1h] -> [g1l, g0h] */
    for (std::size_t k = 1UZ; k < N; ++k) {
        v4sf<T> h0, h1;
        INTERLEAVE2(in[0], in[1], h0, h1);
        in += in_stride;
        *--out = VSWAPHL(g1, h0);
        *--out = VSWAPHL(h0, h1);
        g1     = h1;
    }
    *--out = VSWAPHL(g1, g0);
}

template<std::floating_point T>
static void unreversed_copy(std::size_t N, const v4sf<T>* in, v4sf<T>* out, int out_stride) {
    v4sf<T> g0, g1, h0, h1;
    g0 = g1 = in[0];
    ++in;
    for (std::size_t k = 1; k < N; ++k) {
        h0 = *in++;
        h1 = *in++;
        g1 = VSWAPHL(g1, h0);
        h0 = VSWAPHL(h0, h1);
        UNINTERLEAVE2(h0, g1, out[0], out[1]);
        out += out_stride;
        g1 = h1;
    }
    h0 = *in++;
    h1 = g0;
    g1 = VSWAPHL(g1, h0);
    h0 = VSWAPHL(h0, h1);
    UNINTERLEAVE2(h0, g1, out[0], out[1]);
}

template<fft_direction_t direction, std::floating_point T, fft_transform_t transform>
void pffft_zreorder(PFFFT_Setup<T, transform>* setup, const T* in, T* out) {
    std::size_t    N = setup->N, Ncvec = setup->Ncvec;
    const v4sf<T>* vin  = reinterpret_cast<const v4sf<T>*>(in);
    v4sf<T>*       vout = reinterpret_cast<v4sf<T>*>(out);
    assert(in != out);
    if constexpr (PFFFT_Setup<T, transform>::transform == fft_transform_t::Real) {
        std::size_t k, dk = N / 32;
        if constexpr (direction == fft_direction_t::Forward) {
            for (k = 0; k < dk; ++k) {
                INTERLEAVE2(vin[k * 8 + 0], vin[k * 8 + 1], vout[2 * (0 * dk + k) + 0], vout[2 * (0 * dk + k) + 1]);
                INTERLEAVE2(vin[k * 8 + 4], vin[k * 8 + 5], vout[2 * (2 * dk + k) + 0], vout[2 * (2 * dk + k) + 1]);
            }
            reversed_copy(dk, vin + 2, 8, reinterpret_cast<v4sf<T>*>(out + N / 2));
            reversed_copy(dk, vin + 6, 8, reinterpret_cast<v4sf<T>*>(out + N));
        } else {
            for (k = 0; k < dk; ++k) {
                UNINTERLEAVE2(vin[2 * (0 * dk + k) + 0], vin[2 * (0 * dk + k) + 1], vout[k * 8 + 0], vout[k * 8 + 1]);
                UNINTERLEAVE2(vin[2 * (2 * dk + k) + 0], vin[2 * (2 * dk + k) + 1], vout[k * 8 + 4], vout[k * 8 + 5]);
            }
            unreversed_copy(dk, reinterpret_cast<const v4sf<T>*>(in + N / 4), reinterpret_cast<v4sf<T>*>(out + N - 6 * SIMD_SZ), -8);
            unreversed_copy(dk, reinterpret_cast<const v4sf<T>*>(in + 3 * N / 4), reinterpret_cast<v4sf<T>*>(out + N - 2 * SIMD_SZ), -8);
        }
    } else {
        if constexpr (direction == fft_direction_t::Forward) {
            for (std::size_t k = 0UZ; k < Ncvec; ++k) {
                std::size_t kk = (k / 4) + (k % 4) * (Ncvec / 4);
                INTERLEAVE2(vin[k * 2], vin[k * 2 + 1], vout[kk * 2], vout[kk * 2 + 1]);
            }
        } else {
            for (std::size_t k = 0UZ; k < Ncvec; ++k) {
                std::size_t kk = (k / 4) + (k % 4) * (Ncvec / 4);
                UNINTERLEAVE2(vin[kk * 2], vin[kk * 2 + 1], vout[k * 2], vout[k * 2 + 1]);
            }
        }
    }
}

template<std::floating_point T>
void pffft_cplx_finalize(std::size_t Ncvec, const v4sf<T>* in, v4sf<T>* out, const v4sf<T>* e) {
    std::size_t dk = Ncvec / SIMD_SZ; /* number of 4x4 matrix blocks */
    v4sf<T>     r0, i0, r1, i1, r2, i2, r3, i3;
    v4sf<T>     sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    assert(in != out);
    for (std::size_t k = 0; k < dk; ++k) {
        r0 = in[8 * k + 0];
        i0 = in[8 * k + 1];
        r1 = in[8 * k + 2];
        i1 = in[8 * k + 3];
        r2 = in[8 * k + 4];
        i2 = in[8 * k + 5];
        r3 = in[8 * k + 6];
        i3 = in[8 * k + 7];
        VTRANSPOSE4(r0, r1, r2, r3);
        VTRANSPOSE4(i0, i1, i2, i3);
        VCPLXMUL(r1, i1, e[k * 6 + 0], e[k * 6 + 1]);
        VCPLXMUL(r2, i2, e[k * 6 + 2], e[k * 6 + 3]);
        VCPLXMUL(r3, i3, e[k * 6 + 4], e[k * 6 + 5]);

        sr0 = r0 + r2;
        dr0 = r0 - r2;
        sr1 = r1 + r3;
        dr1 = r1 - r3;
        si0 = i0 + i2;
        di0 = i0 - i2;
        si1 = i1 + i3;
        di1 = i1 - i3;

        r0 = sr0 + sr1;
        i0 = si0 + si1;
        r1 = dr0 + di1;
        i1 = di0 - dr1;
        r2 = sr0 - sr1;
        i2 = si0 - si1;
        r3 = dr0 - di1;
        i3 = di0 + dr1;

        *out++ = r0;
        *out++ = i0;
        *out++ = r1;
        *out++ = i1;
        *out++ = r2;
        *out++ = i2;
        *out++ = r3;
        *out++ = i3;
    }
}

template<std::floating_point T>
void pffft_cplx_preprocess(std::size_t Ncvec, const v4sf<T>* in, v4sf<T>* out, const v4sf<T>* e) {
    std::size_t dk = Ncvec / SIMD_SZ; /* number of 4x4 matrix blocks */
    v4sf<T>     r0, i0, r1, i1, r2, i2, r3, i3;
    v4sf<T>     sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    assert(in != out);
    for (std::size_t k = 0; k < dk; ++k) {
        r0 = in[8 * k + 0];
        i0 = in[8 * k + 1];
        r1 = in[8 * k + 2];
        i1 = in[8 * k + 3];
        r2 = in[8 * k + 4];
        i2 = in[8 * k + 5];
        r3 = in[8 * k + 6];
        i3 = in[8 * k + 7];

        sr0 = r0 + r2;
        dr0 = r0 - r2;
        sr1 = r1 + r3;
        dr1 = r1 - r3;
        si0 = i0 + i2;
        di0 = i0 - i2;
        si1 = i1 + i3;
        di1 = i1 - i3;

        r0 = sr0 + sr1;
        i0 = si0 + si1;
        r1 = dr0 - di1;
        i1 = di0 + dr1;
        r2 = sr0 - sr1;
        i2 = si0 - si1;
        r3 = dr0 + di1;
        i3 = di0 - dr1;

        VCPLXMULCONJ(r1, i1, e[k * 6 + 0], e[k * 6 + 1]);
        VCPLXMULCONJ(r2, i2, e[k * 6 + 2], e[k * 6 + 3]);
        VCPLXMULCONJ(r3, i3, e[k * 6 + 4], e[k * 6 + 5]);

        VTRANSPOSE4(r0, r1, r2, r3);
        VTRANSPOSE4(i0, i1, i2, i3);

        *out++ = r0;
        *out++ = i0;
        *out++ = r1;
        *out++ = i1;
        *out++ = r2;
        *out++ = i2;
        *out++ = r3;
        *out++ = i3;
    }
}

template<std::floating_point T>
static ALWAYS_INLINE(void) pffft_real_finalize_4x4(const v4sf<T>* in0, const v4sf<T>* in1, const v4sf<T>* in, const v4sf<T>* e, v4sf<T>* out) {
    v4sf<T> r0, i0, r1, i1, r2, i2, r3, i3;
    v4sf<T> sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    r0 = *in0;
    i0 = *in1;
    r1 = *in++;
    i1 = *in++;
    r2 = *in++;
    i2 = *in++;
    r3 = *in++;
    i3 = *in++;
    VTRANSPOSE4(r0, r1, r2, r3);
    VTRANSPOSE4(i0, i1, i2, i3);

    VCPLXMUL(r1, i1, e[0], e[1]);
    VCPLXMUL(r2, i2, e[2], e[3]);
    VCPLXMUL(r3, i3, e[4], e[5]);

    sr0 = r0 + r2;
    dr0 = r0 - r2;
    sr1 = r1 + r3;
    dr1 = r3 - r1;
    si0 = i0 + i2;
    di0 = i0 - i2;
    si1 = i1 + i3;
    di1 = i3 - i1;

    r0 = sr0 + sr1;
    r3 = sr0 - sr1;
    i0 = si0 + si1;
    i3 = si1 - si0;
    r1 = dr0 + di1;
    r2 = dr0 - di1;
    i1 = dr1 - di0;
    i2 = dr1 + di0;

    *out++ = r0;
    *out++ = i0;
    *out++ = r1;
    *out++ = i1;
    *out++ = r2;
    *out++ = i2;
    *out++ = r3;
    *out++ = i3;
}

template<std::floating_point T>
static NEVER_INLINE(void) pffft_real_finalize(std::size_t Ncvec, const v4sf<T>* in, v4sf<T>* out, const v4sf<T>* e) {
    std::size_t dk = Ncvec / SIMD_SZ; /* number of 4x4 matrix blocks */

    v4sf_union<T> cr, ci, *uout = reinterpret_cast<v4sf_union<T>*>(out);
    v4sf<T>       save = in[7];
    v4sf<T>       zero = {};
    T             xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3;

    cr.v = in[0];
    ci.v = in[Ncvec * 2 - 1];
    assert(in != out);
    pffft_real_finalize_4x4(&zero, &zero, in + 1, e, out);

    xr0          = (cr.f[0] + cr.f[2]) + (cr.f[1] + cr.f[3]);
    uout[0].f[0] = xr0;
    xi0          = (cr.f[0] + cr.f[2]) - (cr.f[1] + cr.f[3]);
    uout[1].f[0] = xi0;
    xr2          = (cr.f[0] - cr.f[2]);
    uout[4].f[0] = xr2;
    xi2          = (cr.f[3] - cr.f[1]);
    uout[5].f[0] = xi2;
    xr1          = ci.f[0] + FFTConstants<T>::inv_sqrt2 * (ci.f[1] - ci.f[3]);
    uout[2].f[0] = xr1;
    xi1          = -ci.f[2] - FFTConstants<T>::inv_sqrt2 * (ci.f[1] + ci.f[3]);
    uout[3].f[0] = xi1;
    xr3          = ci.f[0] - FFTConstants<T>::inv_sqrt2 * (ci.f[1] - ci.f[3]);
    uout[6].f[0] = xr3;
    xi3          = ci.f[2] - FFTConstants<T>::inv_sqrt2 * (ci.f[1] + ci.f[3]);
    uout[7].f[0] = xi3;

    for (std::size_t k = 1; k < dk; ++k) {
        v4sf<T> save_next = in[8 * k + 7];
        pffft_real_finalize_4x4(&save, &in[8 * k + 0], in + 8 * k + 1, e + k * 6, out + k * 8);
        save = save_next;
    }
}

template<std::floating_point T>
static ALWAYS_INLINE(void) pffft_real_preprocess_4x4(const v4sf<T>* in, const v4sf<T>* e, v4sf<T>* out, std::size_t first) {
    v4sf<T> r0 = in[0], i0 = in[1], r1 = in[2], i1 = in[3], r2 = in[4], i2 = in[5], r3 = in[6], i3 = in[7];

    v4sf<T> sr0 = r0 + r3, dr0 = r0 - r3;
    v4sf<T> sr1 = r1 + r2, dr1 = r1 - r2;
    v4sf<T> si0 = i0 + i3, di0 = i0 - i3;
    v4sf<T> si1 = i1 + i2, di1 = i1 - i2;

    r0 = sr0 + sr1;
    r2 = sr0 - sr1;
    r1 = dr0 - si1;
    r3 = dr0 + si1;
    i0 = di0 - di1;
    i2 = di0 + di1;
    i1 = si0 - dr1;
    i3 = si0 + dr1;

    VCPLXMULCONJ(r1, i1, e[0], e[1]);
    VCPLXMULCONJ(r2, i2, e[2], e[3]);
    VCPLXMULCONJ(r3, i3, e[4], e[5]);

    VTRANSPOSE4(r0, r1, r2, r3);
    VTRANSPOSE4(i0, i1, i2, i3);

    if (!first) {
        *out++ = r0;
        *out++ = i0;
    }
    *out++ = r1;
    *out++ = i1;
    *out++ = r2;
    *out++ = i2;
    *out++ = r3;
    *out++ = i3;
}

template<std::floating_point T>
static NEVER_INLINE(void) pffft_real_preprocess(std::size_t Ncvec, const v4sf<T>* in, v4sf<T>* out, const v4sf<T>* e) {
    std::size_t dk = Ncvec / SIMD_SZ; /* number of 4x4 matrix blocks */

    v4sf_union<T> Xr, Xi, *uout = reinterpret_cast<v4sf_union<T>*>(out);
    T             cr0, ci0, cr1, ci1, cr2, ci2, cr3, ci3;
    assert(in != out);
    for (std::size_t k = 0; k < 4; ++k) {
        Xr.f[k] = reinterpret_cast<const T*>(in)[8 * k];
        Xi.f[k] = reinterpret_cast<const T*>(in)[8 * k + 4];
    }

    pffft_real_preprocess_4x4(in, e, out + 1, 1); /* will write only 6 values */

    for (std::size_t k = 1; k < dk; ++k) {
        pffft_real_preprocess_4x4(in + 8 * k, e + k * 6, out - 1 + k * 8, 0);
    }

    cr0                      = (Xr.f[0] + Xi.f[0]) + 2 * Xr.f[2];
    uout[0].f[0]             = cr0;
    cr1                      = (Xr.f[0] - Xi.f[0]) - 2 * Xi.f[2];
    uout[0].f[1]             = cr1;
    cr2                      = (Xr.f[0] + Xi.f[0]) - 2 * Xr.f[2];
    uout[0].f[2]             = cr2;
    cr3                      = (Xr.f[0] - Xi.f[0]) + 2 * Xi.f[2];
    uout[0].f[3]             = cr3;
    ci0                      = 2 * (Xr.f[1] + Xr.f[3]);
    uout[2 * Ncvec - 1].f[0] = ci0;
    ci1                      = std::numbers::sqrt2_v<T> * (Xr.f[1] - Xr.f[3]) - std::numbers::sqrt2_v<T> * (Xi.f[1] + Xi.f[3]);
    uout[2 * Ncvec - 1].f[1] = ci1;
    ci2                      = 2 * (Xi.f[3] - Xi.f[1]);
    uout[2 * Ncvec - 1].f[2] = ci2;
    ci3                      = -std::numbers::sqrt2_v<T> * (Xr.f[1] - Xr.f[3]) - std::numbers::sqrt2_v<T> * (Xi.f[1] + Xi.f[3]);
    uout[2 * Ncvec - 1].f[3] = ci3;
}

template<fft_direction_t direction, std::floating_point T, fft_transform_t transform>
void pffft_transform_internal(PFFFT_Setup<T, transform>* setup, const T* finput, T* foutput, v4sf<T>* scratch, int ordered) {
    std::size_t k, Ncvec = setup->Ncvec;
    int         nf_odd = setup->ifac[1] & 1;

    /* temporary buffer is allocated on the stack if the scratch pointer is NULL */
    std::size_t stack_allocate = scratch == nullptr ? Ncvec * 2 : 1;
    VLA_ARRAY_ON_STACK(v4sf<T>, scratch_on_stack, stack_allocate); // TODO: this is ugly and not portable (however, an efficiency hack -> local stack storage)
    // alignas(64) std::array<v4sf<T>, 2048UZ> scratch_storage; // max reasonable size
    // v4sf<T>* scratch_ptr = scratch ? scratch : scratch_storage.data();

    const v4sf<T>* vinput  = reinterpret_cast<const v4sf<T>*>(finput);
    v4sf<T>*       voutput = reinterpret_cast<v4sf<T>*>(foutput);
    v4sf<T>*       buff[2] = {voutput, scratch ? scratch : scratch_on_stack};
    std::size_t    ib      = nf_odd ^ ordered ? 1 : 0;

    assert(VALIGNED(finput) && VALIGNED(foutput));

    /* assert(finput != foutput); */
    if constexpr (direction == fft_direction_t::Forward) {
        ib = !ib;
        if constexpr (PFFFT_Setup<T, transform>::transform == fft_transform_t::Real) {
            ib = (rfftf1_ps(Ncvec * 2, vinput, buff[ib], buff[!ib], setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
            pffft_real_finalize<T>(Ncvec, buff[ib], buff[!ib], reinterpret_cast<v4sf<T>*>(setup->e));
        } else {
            v4sf<T>* tmp = buff[ib];
            for (k = 0; k < Ncvec; ++k) {
                UNINTERLEAVE2(vinput[k * 2], vinput[k * 2 + 1], tmp[k * 2], tmp[k * 2 + 1]);
            }
            ib = (cfftf1_ps(Ncvec, buff[ib], buff[!ib], buff[ib], setup->twiddle, &setup->ifac[0], -1) == buff[0] ? 0 : 1);
            pffft_cplx_finalize(Ncvec, buff[ib], buff[!ib], reinterpret_cast<v4sf<T>*>(setup->e));
        }
        if (ordered) {
            pffft_zreorder<fft_direction_t::Forward>(setup, reinterpret_cast<T*>(buff[!ib]), reinterpret_cast<T*>(buff[ib]));
        } else {
            ib = !ib;
        }
    } else {
        if (vinput == buff[ib]) {
            ib = !ib; /* may happen when finput == foutput */
        }
        if (ordered) {
            pffft_zreorder<fft_direction_t::Backward>(setup, reinterpret_cast<const T*>(vinput), reinterpret_cast<T*>(buff[ib]));
            vinput = buff[ib];
            ib     = !ib;
        }
        if constexpr (PFFFT_Setup<T, transform>::transform == fft_transform_t::Real) {
            pffft_real_preprocess<T>(Ncvec, vinput, buff[ib], reinterpret_cast<v4sf<T>*>(setup->e));
            ib = (rfftb1_ps(Ncvec * 2, buff[ib], buff[0], buff[1], setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
        } else {
            pffft_cplx_preprocess(Ncvec, vinput, buff[ib], reinterpret_cast<v4sf<T>*>(setup->e));
            ib = (cfftf1_ps(Ncvec, buff[ib], buff[0], buff[1], setup->twiddle, &setup->ifac[0], +1) == buff[0] ? 0 : 1);
            for (k = 0; k < Ncvec; ++k) {
                INTERLEAVE2(buff[ib][k * 2], buff[ib][k * 2 + 1], buff[ib][k * 2], buff[ib][k * 2 + 1]);
            }
        }
    }

    if (buff[ib] != voutput) {
        /* extra copy required -- this situation should only happen when finput == foutput */
        assert(finput == foutput);
        for (k = 0; k < Ncvec; ++k) {
            v4sf<T> a = buff[ib][2 * k], b = buff[ib][2 * k + 1];
            voutput[2 * k]     = a;
            voutput[2 * k + 1] = b;
        }
        ib = !ib;
    }
    assert(buff[ib] == voutput);
}

template<std::floating_point T, fft_transform_t transform>
void pffft_zconvolve_accumulate(PFFFT_Setup<T, transform>* s, const T* a, const T* b, T* ab, T scaling) {
    std::size_t Ncvec = s->Ncvec;

    assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
    const T ar  = a[0];
    const T ai  = a[4];
    const T br  = b[0];
    const T bi  = b[4];
    const T abr = ab[0];
    const T abi = ab[4];

    /* default routine, works fine for non-arm cpus with current compilers */
    const v4sf<T>      vscal = scaling;
    std::span<const T> sa(a, Ncvec * 8);
    std::span<const T> sb(b, Ncvec * 8);
    std::span<T>       sab(ab, Ncvec * 8);
    vir::transform(vir::execution::simd.prefer_size<8UZ>().unroll_by<2UZ>(), std::views::zip(sa, sb, sab), sab, [=](const auto& tup) {
        const auto& [va, vb, vab] = tup;
        if constexpr (va.size() == 8UZ) {
            auto [ar_, ai_]   = split<4, 4>(va);
            auto [br_, bi_]   = split<4, 4>(vb);
            auto [abr_, abi_] = split<4, 4>(vab);
            VCPLXMUL(ar_, ai_, br_, bi_);
            return concat((ar * vscal + abr_), (ai * vscal + abi_));
        } else {
            __builtin_trap(); // this should be impossible
            return vab;       // to get the expected return type
        }
    });

    if constexpr (PFFFT_Setup<T, transform>::transform == fft_transform_t::Real) {
        ab[0] = abr + ar * br * scaling;
        ab[4] = abi + ai * bi * scaling;
    }
}

template<std::floating_point T, fft_transform_t transform>
void pffft_zconvolve_no_accu(PFFFT_Setup<T, transform>* s, const T* a, const T* b, T* ab, T scaling) {
    v4sf<T>                 vscal = scaling;
    const v4sf<T>* RESTRICT va    = reinterpret_cast<const v4sf<T>*>(a);
    const v4sf<T>* RESTRICT vb    = reinterpret_cast<const v4sf<T>*>(b);
    v4sf<T>* RESTRICT       vab   = reinterpret_cast<v4sf<T>*>(ab);
    T                       sar, sai, sbr, sbi;
    const std::size_t       NcvecMulTwo = 2 * s->Ncvec; /* std::size_t Ncvec = s->Ncvec; */

    assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
    sar = reinterpret_cast<const v4sf_union<T>*>(va)[0].f[0];
    sai = reinterpret_cast<const v4sf_union<T>*>(va)[1].f[0];
    sbr = reinterpret_cast<const v4sf_union<T>*>(vb)[0].f[0];
    sbi = reinterpret_cast<const v4sf_union<T>*>(vb)[1].f[0];

    /* default routine, works fine for non-arm cpus with current compilers */
    for (std::size_t k = 0; k < NcvecMulTwo; k += 4) {
        v4sf<T> var, vai, vbr, vbi;
        var = va[k + 0];
        vai = va[k + 1];
        vbr = vb[k + 0];
        vbi = vb[k + 1];
        VCPLXMUL(var, vai, vbr, vbi);
        vab[k + 0] = var * vscal;
        vab[k + 1] = vai * vscal;
        var        = va[k + 2];
        vai        = va[k + 3];
        vbr        = vb[k + 2];
        vbi        = vb[k + 3];
        VCPLXMUL(var, vai, vbr, vbi);
        vab[k + 2] = var * vscal;
        vab[k + 3] = vai * vscal;
    }

    if constexpr (PFFFT_Setup<T, transform>::transform == fft_transform_t::Real) {
        reinterpret_cast<v4sf_union<T>*>(vab)[0].f[0] = sar * sbr * scaling;
        reinterpret_cast<v4sf_union<T>*>(vab)[1].f[0] = sai * sbi * scaling;
    }
}

template<fft_direction_t direction, std::floating_point T, fft_transform_t transform>
void pffft_transform(PFFFT_Setup<T, transform>* setup, const T* input, T* output, T* work) {
    pffft_transform_internal<direction>(setup, input, output, reinterpret_cast<v4sf<T>*>(work), 0);
}

template<fft_direction_t direction, std::floating_point T, fft_transform_t transform>
void pffft_transform_ordered(PFFFT_Setup<T, transform>* setup, const T* input, T* output, T* work) {
    pffft_transform_internal<direction>(setup, input, output, reinterpret_cast<v4sf<T>*>(work), 1);
}
