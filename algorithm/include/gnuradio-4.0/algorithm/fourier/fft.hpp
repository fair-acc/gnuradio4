#ifndef GNURADIO_ALGORITHM_FFT_HPP
#define GNURADIO_ALGORITHM_FFT_HPP

#include <bit>
#include <complex>
#include <execution>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <vector>

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

        // stdx::simd-fication attempt
        // namespace stdx = vir::stdx;
        // auto* raw    = reinterpret_cast<ValueType*>(data); // [re0, im0, re1, im1]
        // using simd_t = stdx::fixed_size_simd<ValueType, 2>;;
        // simd_t a(&raw[0], stdx::element_aligned);
        // simd_t b(&raw[2], stdx::element_aligned);
        // simd_t sum  = a + b;
        // simd_t diff = a - b;
        //
        // sum.copy_to(&raw[0], stdx::element_aligned);  // data[0] = sum
        // diff.copy_to(&raw[2], stdx::element_aligned); // data[1] = diff
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

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_FFT_HPP
