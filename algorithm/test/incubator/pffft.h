#ifndef PFFFT_H
#define PFFFT_H

#include <bit>
#include <cstddef>
#include <new>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"          // error in vir/simd
#pragma GCC diagnostic ignored "-Wsign-conversion" // error in vir/simd
#endif

#include "pf_stdx_simd.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifndef __cpp_aligned_new
#error
#endif

/* direction of the transform */
enum class fft_direction_t { Forward, Backward };

/* type of transform */
enum class fft_transform_t { Real, Complex };

/* struct holding internal stuff (precomputed twiddle factors)
   this struct can be shared by many threads as it contains only
   read-only data.
*/
template<std::floating_point T, fft_transform_t transform_, std::size_t N = std::dynamic_extent>
struct PFFFT_Setup;

/*
   Perform a Fourier transform , The z-domain data is stored in the
   most efficient order for transforming it back, or using it for
   convolution. If you need to have its content sorted in the
   "usual" way, that is as an array of interleaved complex numbers,
   either use pffft_transform_ordered , or call pffft_zreorder after
   the forward fft, and before the backward fft.

   Transforms are not scaled: PFFFT_BACKWARD(PFFFT_FORWARD(x)) = N*x.
   Typically you will want to scale the backward transform by 1/N.

   The 'work' pointer should point to an area of N (2*N for complex
   fft) floats, properly aligned. If 'work' is NULL, then stack will
   be used instead (this is probably the best strategy for small
   FFTs, say for N < 16384). Threads usually have a small stack, that
   there's no sufficient amount of memory, usually leading to a crash!
   Use the heap with pffft_aligned_malloc() in this case.

   For a real forward transform (PFFFT_REAL | PFFFT_FORWARD) with real
   input with input(=transformation) length N, the output array is
   'mostly' complex:
     index k in 1 .. N/2 -1  corresponds to frequency k * Samplerate / N
     index k == 0 is a special case:
       the real() part contains the result for the DC frequency 0,
       the imag() part contains the result for the Nyquist frequency Samplerate/2
   both 0-frequency and half frequency components, which are real,
   are assembled in the first entry as  F(0)+i*F(N/2).
   With the output size N/2 complex values (=N real/imag values), it is
   obvious, that the result for negative frequencies are not output,
   cause of symmetry.

   input and output may alias.
*/
template<fft_direction_t direction, std::floating_point T, fft_transform_t transform, std::size_t N>
void pffft_transform(PFFFT_Setup<T, transform, N>* setup, const T* input, T* output, T* work);

/*
   Similar to pffft_transform, but makes sure that the output is
   ordered as expected (interleaved complex numbers).  This is
   similar to calling pffft_transform and then pffft_zreorder.

   input and output may alias.
*/
template<fft_direction_t direction, std::floating_point T, fft_transform_t transform, std::size_t N>
void pffft_transform_ordered(PFFFT_Setup<T, transform, N>& setup, const T* input, T* output, T* work);

/*
   call pffft_zreorder(.., PFFFT_FORWARD) after pffft_transform(...,
   PFFFT_FORWARD) if you want to have the frequency components in
   the correct "canonical" order, as interleaved complex numbers.

   (for real transforms, both 0-frequency and half frequency
   components, which are real, are assembled in the first entry as
   F(0)+i*F(n/2+1). Note that the original fftpack did place
   F(n/2+1) at the end of the arrays).

   input and output should not alias.
*/
template<fft_direction_t direction, std::floating_point T, fft_transform_t transform>
void pffft_zreorder(PFFFT_Setup<T, transform>* setup, const T* input, T* output);

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and accumulate them into dft_ab. The arrays should have
   been obtained with pffft_transform(.., PFFFT_FORWARD) and should
   *not* have been reordered with pffft_zreorder (otherwise just
   perform the operation yourself as the dft coefs are stored as
   interleaved complex numbers).

   the operation performed is: dft_ab += (dft_a * fdt_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
template<std::floating_point T, fft_transform_t transform>
void pffft_zconvolve_accumulate(PFFFT_Setup<T, transform>* setup, const T* dft_a, const T* dft_b, T* dft_ab, T scaling);

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and put result in dft_ab. The arrays should have
   been obtained with pffft_transform(.., PFFFT_FORWARD) and should
   *not* have been reordered with pffft_zreorder (otherwise just
   perform the operation yourself as the dft coefs are stored as
   interleaved complex numbers).

   the operation performed is: dft_ab = (dft_a * fdt_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
template<std::floating_point T, fft_transform_t transform>
void pffft_zconvolve_no_accu(PFFFT_Setup<T, transform>* setup, const T* dft_a, const T* dft_b, T* dft_ab, T scaling);

/* return 4 or 1 wether support SSE/NEON/Altivec instructions was enabled when building pffft.c */
template<std::floating_point T>
consteval std::size_t pffft_simd_size(void);

/* return string identifier of used architecture (SSE/NEON/Altivec/..) */
const char* pffft_simd_arch(void);

/* following functions are identical to the pffftd_ functions */

/* simple helper to get minimum possible fft size */
template<std::floating_point T>
std::size_t pffft_min_fft_size(fft_transform_t transform);

/* simple helper to determine size N is valid
   - factorizable to pffft_min_fft_size() with factors 2, 3, 5
   returns bool
*/
template<std::floating_point T>
constexpr bool pffft_is_valid_size(std::size_t N, fft_transform_t cplx);

/* determine nearest valid transform size  (by brute-force testing)
   - factorizable to pffft_min_fft_size() with factors 2, 3, 5.
   higher: bool-flag to find nearest higher value; else lower.
*/
template<std::floating_point T>
constexpr std::size_t pffft_nearest_transform_size(std::size_t N, fft_transform_t cplx, bool higher);

/*
  the float buffers must have the correct alignment (16-byte boundary
  on intel and powerpc). This function may be used to obtain such
  correctly aligned buffers.
*/
template<typename T>
T* pffft_aligned_malloc(size_t nb_bytes) {
    return static_cast<T*>(operator new(nb_bytes, std::align_val_t(64)));
}

inline void pffft_aligned_free(void* ptr) { operator delete(ptr, std::align_val_t(64)); }

#endif /* PFFFT_H */
