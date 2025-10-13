#include "SimdFFT.hpp"

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
