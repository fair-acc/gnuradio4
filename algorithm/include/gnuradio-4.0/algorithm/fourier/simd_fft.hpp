#ifndef GNURADIO_ALGORITHM_FFTSIMD_HPP
#define GNURADIO_ALGORITHM_FFTSIMD_HPP

/**
Copyright (c) 2025  Matthias Kretz & Ralph J. Steinhagen
GSI Helmholtzzentrum fuer Schwerionenforschung GmbH, Darmstadt, Germany
FAIR - Facility for Antiproton & Ion Research, Darmstadt, Germany

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

// TODO: simdized version (WIP)
#include <vir/simd.h>
#include <complex>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <span>

namespace simd_fft {

namespace stdx = vir::stdx;

// Core FFT class template
template<typename T>
requires std::is_floating_point_v<T>
class SimdFFT {
public:
    // SIMD type aliases
    using value_type = T;
    using simd_type = stdx::native_simd<T>;
    using complex_type = std::complex<T>;
    static constexpr size_t simd_size = simd_type::size();

    // Transform direction enum
    enum class Direction {
        Forward = 1,
        Backward = -1
    };

    // Constructor - initialize FFT for given size
    explicit SimdFFT(size_t n)
        : n_(n)
        , log2_n_(calculate_log2(n))
        , twiddle_factors_(n / 2)
    {
        if (!is_power_of_two(n)) {
            throw std::invalid_argument("FFT size must be a power of 2");
        }

        if (n < simd_size * 4) {
            throw std::invalid_argument("FFT size too small for SIMD operations");
        }

        initialize_twiddle_factors();
    }

    // Main transform functions
    void transform_complex(std::span<complex_type> data, Direction dir) {
        if (data.size() != n_) {
            throw std::invalid_argument("Data size mismatch");
        }

        // Bit reversal using SIMD operations
        bit_reverse_simd(data);

        // Cooley-Tukey FFT with SIMD butterflies
        size_t m = 2;
        while (m <= n_) {
            const size_t half_m = m / 2;

            // Process with SIMD-width parallelism
            for (size_t k = 0; k < n_; k += m) {
                butterfly_stage_simd(data, k, half_m, dir);
            }

            m *= 2;
        }

        // Normalize for backward transform
        if (dir == Direction::Backward) {
            normalize_simd(data);
        }
    }

    // Real-to-complex transform (using half-size complex transform)
    void transform_real(std::span<T> real_data,
                         std::span<complex_type> complex_data,
                         Direction dir) {
        if (real_data.size() != n_) {
            throw std::invalid_argument("Real data size mismatch");
        }

        if (complex_data.size() != n_ / 2 + 1) {
            throw std::invalid_argument("Complex data size mismatch");
        }

        if (dir == Direction::Forward) {
            real_to_complex_forward_simd(real_data, complex_data);
        } else {
            complex_to_real_backward_simd(complex_data, real_data);
        }
    }

private:
    size_t n_;
    size_t log2_n_;
    std::vector<complex_type> twiddle_factors_;

    // Initialize twiddle factors
    void initialize_twiddle_factors() {
        const T angle_step = -2.0 * M_PI / n_;

        for (size_t i = 0; i < n_ / 2; ++i) {
            const T angle = angle_step * i;
            twiddle_factors_[i] = complex_type(std::cos(angle), std::sin(angle));
        }
    }

    // SIMD butterfly operation
    void butterfly_stage_simd(std::span<complex_type> data,
                              size_t offset,
                              size_t half_m,
                              Direction dir) {
        const size_t twiddle_step = n_ / (2 * half_m);

        // Process multiple butterflies in parallel using SIMD
        for (size_t j = 0; j < half_m; j += simd_size) {
            [[maybe_unused]] const size_t chunk_size = std::min(simd_size, half_m - j);

            // Load data into SIMD registers
            simd_type real_a, imag_a, real_b, imag_b;
            load_complex_simd(&data[offset + j], real_a, imag_a);
            load_complex_simd(&data[offset + j + half_m], real_b, imag_b);

            // Apply twiddle factors
            if (twiddle_step * j > 0) {
                apply_twiddle_simd(real_b, imag_b, j * twiddle_step, dir);
            }

            // Butterfly computation
            const simd_type real_sum = real_a + real_b;
            const simd_type imag_sum = imag_a + imag_b;
            const simd_type real_diff = real_a - real_b;
            const simd_type imag_diff = imag_a - imag_b;

            // Store results
            store_complex_simd(&data[offset + j], real_sum, imag_sum);
            store_complex_simd(&data[offset + j + half_m], real_diff, imag_diff);
        }
    }

    // SIMD-optimized bit reversal
    void bit_reverse_simd(std::span<complex_type> data) {
        // Process multiple elements in parallel where possible
        for (size_t i = 0; i < n_; ++i) {
            const size_t rev = bit_reverse(i, log2_n_);
            if (i < rev) {
                std::swap(data[i], data[rev]);
            }
        }
    }

    // Load complex data into SIMD registers
    void load_complex_simd(const complex_type* ptr,
                           simd_type& real_out,
                           simd_type& imag_out) {
        // Deinterleave complex data for SIMD processing
        alignas(stdx::memory_alignment_v<simd_type>) T real_temp[simd_size];
        alignas(stdx::memory_alignment_v<simd_type>) T imag_temp[simd_size];

        for (size_t i = 0; i < simd_size; ++i) {
            real_temp[i] = ptr[i].real();
            imag_temp[i] = ptr[i].imag();
        }

        real_out = simd_type(real_temp, stdx::element_aligned);
        imag_out = simd_type(imag_temp, stdx::element_aligned);
    }

    // Store SIMD registers to complex data
    void store_complex_simd(complex_type* ptr,
                            const simd_type& real_in,
                            const simd_type& imag_in) {
        alignas(stdx::memory_alignment_v<simd_type>) T real_temp[simd_size];
        alignas(stdx::memory_alignment_v<simd_type>) T imag_temp[simd_size];

        real_in.copy_to(real_temp, stdx::element_aligned);
        imag_in.copy_to(imag_temp, stdx::element_aligned);

        for (size_t i = 0; i < simd_size; ++i) {
            ptr[i] = complex_type(real_temp[i], imag_temp[i]);
        }
    }

    // Apply twiddle factors using SIMD
    void apply_twiddle_simd(simd_type& real,
                            simd_type& imag,
                            size_t twiddle_idx,
                            Direction dir) {
        // Load twiddle factors
        alignas(stdx::memory_alignment_v<simd_type>) T tw_real[simd_size];
        alignas(stdx::memory_alignment_v<simd_type>) T tw_imag[simd_size];

        for (size_t i = 0; i < simd_size; ++i) {
            const size_t idx = (twiddle_idx + i) % (n_ / 2);
            tw_real[i] = twiddle_factors_[idx].real();
            tw_imag[i] = (dir == Direction::Forward) ?
                         twiddle_factors_[idx].imag() :
                         -twiddle_factors_[idx].imag();
        }

        const simd_type tw_r(tw_real, stdx::element_aligned);
        const simd_type tw_i(tw_imag, stdx::element_aligned);

        // Complex multiplication: (a + bi) * (c + di)
        const simd_type temp_real = real * tw_r - imag * tw_i;
        const simd_type temp_imag = real * tw_i + imag * tw_r;

        real = temp_real;
        imag = temp_imag;
    }

    // Normalize using SIMD
    void normalize_simd(std::span<complex_type> data) {
        const simd_type scale(T(1.0) / n_);

        for (size_t i = 0; i < n_; i += simd_size) {
            simd_type real, imag;
            load_complex_simd(&data[i], real, imag);
            real *= scale;
            imag *= scale;
            store_complex_simd(&data[i], real, imag);
        }
    }

    // Real-to-complex forward transform using SIMD
    void real_to_complex_forward_simd(std::span<T> real_data,
                                      std::span<complex_type> complex_data) {
        // Pack real data as complex for half-size FFT
        std::vector<complex_type> packed(n_ / 2);

        for (size_t i = 0; i < n_ / 2; i += simd_size) {
            const simd_type r1(real_data.data() + 2 * i, stdx::element_aligned);
            const simd_type r2(real_data.data() + 2 * i + simd_size, stdx::element_aligned);
            store_complex_simd(&packed[i], r1, r2);
        }

        // Half-size complex FFT
        SimdFFT<T> half_fft(n_ / 2);
        half_fft.transform_complex(packed, Direction::Forward);

        // Post-process to get full spectrum using SIMD
        post_process_real_fft_simd(packed, complex_data);
    }

    // Complex-to-real backward transform using SIMD
    void complex_to_real_backward_simd(std::span<complex_type> complex_data,
                                        std::span<T> real_data) {
        // Pre-process spectrum
        std::vector<complex_type> packed(n_ / 2);
        pre_process_real_ifft_simd(complex_data, packed);

        // Half-size complex IFFT
        SimdFFT<T> half_fft(n_ / 2);
        half_fft.transform_complex(packed, Direction::Backward);

        // Unpack to real data using SIMD
        for (size_t i = 0; i < n_ / 2; i += simd_size) {
            simd_type r1, r2;
            load_complex_simd(&packed[i], r1, r2);
            r1.copy_to(real_data.data() + 2 * i, stdx::element_aligned);
            r2.copy_to(real_data.data() + 2 * i + simd_size, stdx::element_aligned);
        }
    }

    // Post-process for real FFT
    void post_process_real_fft_simd(const std::vector<complex_type>& packed,
                                    std::span<complex_type> output) {
        // DC and Nyquist components
        output[0] = complex_type(
            packed[0].real() + packed[0].imag(),
            0
        );
        output[n_ / 2] = complex_type(
            packed[0].real() - packed[0].imag(),
            0
        );

        // Process remaining bins with SIMD
        const T angle_step = -2.0 * M_PI / n_;

        for (size_t k = 1; k < n_ / 4; k += simd_size) {
            // Use SIMD for symmetric processing
            process_real_fft_bins_simd(packed, output, k, angle_step);
        }
    }

    // Pre-process for real IFFT
    void pre_process_real_ifft_simd(std::span<complex_type> input,
                                    std::vector<complex_type>& packed) {
        // Inverse of post-processing
        packed[0] = complex_type(
            (input[0].real() + input[n_ / 2].real()) * T(0.5),
            (input[0].real() - input[n_ / 2].real()) * T(0.5)
        );

        const T angle_step = 2.0 * M_PI / n_;

        for (size_t k = 1; k < n_ / 4; k += simd_size) {
            process_real_ifft_bins_simd(input, packed, k, angle_step);
        }
    }

    // Helper for processing real FFT bins with SIMD
    void process_real_fft_bins_simd(const std::vector<complex_type>& packed,
                                   std::span<complex_type> output,
                                   size_t k,
                                   T angle_step) {
        // Implementation using SIMD operations for bin processing
        // This involves complex arithmetic with twiddle factors
        // Details omitted for brevity but follow standard real FFT post-processing
    }

    // Helper for processing real IFFT bins with SIMD
    void process_real_ifft_bins_simd(std::span<complex_type> input,
                                    std::vector<complex_type>& packed,
                                    size_t k,
                                    T angle_step) {
        // Implementation using SIMD operations for bin processing
        // Inverse of process_real_fft_bins_simd
    }

    // Utility functions
    static constexpr bool is_power_of_two(size_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    static constexpr size_t calculate_log2(size_t n) {
        size_t log2 = 0;
        while ((n >>= 1) != 0) {
            ++log2;
        }
        return log2;
    }

    static size_t bit_reverse(size_t x, size_t bits) {
        size_t result = 0;
        for (size_t i = 0; i < bits; ++i) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
};

// Convenience aliases
using SimdFFT_float = SimdFFT<float>;
using SimdFFT_double = SimdFFT<double>;

} // namespace simd_fft
/**************************************************************************
 *  pfft.cpp -- SIMD prototype implementation
 *************************************************************************/

// TODO: incorporate into new interface
namespace gr::algorithm {

template<typename TInput, gr::meta::complex_like TOutput = std::conditional<gr::meta::complex_like<TInput>, TInput, std::complex<typename TInput::value_type>>>
requires((gr::meta::complex_like<TInput> || std::floating_point<TInput>))
struct FFT {
    using ValueType = typename TOutput::value_type;

    std::vector<std::vector<TOutput>> stageTwiddles{};
    std::vector<TOutput>              bluesteinExpTable{};
    std::vector<TOutput>              bluesteinChirpFFT{};
    std::vector<std::size_t>          bitReverseTable{};
    std::size_t                       fftSize{0};

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

    auto compute(const std::ranges::input_range auto& in) { return compute(in, std::vector<TOutput>(in.size())); }

private:
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
                    break;
                }
            }
        }
    }

    mutable std::unique_ptr<FFT<TOutput, TOutput>> fftCache;
    mutable std::vector<TOutput>                   aCache{};
    mutable std::vector<TOutput>                   bCache{};

    void transformBluestein(std::ranges::input_range auto& inPlace) const {
        const std::size_t n = inPlace.size();
        const std::size_t m = std::bit_ceil(2 * n + 1);

        std::vector<TOutput> a(m);
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

        const std::size_t    m = std::bit_ceil(2 * n + 1);
        std::vector<TOutput> b(m);
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

}

#endif // GNURADIO_ALGORITHM_FFTSIMD_HPP
