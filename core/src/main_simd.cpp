#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <format>
#include <immintrin.h> // AVX/AVX2 intrinsics
#include <iostream>
#include <numbers>
#include <print>
#include <vector>

#include <gnuradio-4.0/meta/utils.hpp>

#if defined(__GNUC__) && !defined(__clang__) && !defined(__EMSCRIPTEN__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <pffft.hpp>
#if defined(__GNUC__) && !defined(__clang__) && !defined(__EMSCRIPTEN__)
#pragma GCC diagnostic pop
#endif

namespace stdx = vir::stdx;

namespace gr {

using simd_complex = __m128d;

template<typename T>
struct complex {
    T re;
    T im;

    constexpr complex(T r = T{}, T i = T{}) : re(r), im(i) {}

    complex operator+(const complex& other) const { return {re + other.re, im + other.im}; }
    complex operator-(const complex& other) const { return {re - other.re, im - other.im}; }
    complex operator*(const complex& other) const { return {re * other.re - im * other.im, re * other.im + im * other.re}; }

    constexpr T real() const { return re; }
    constexpr T imag() const { return im; }
};

namespace algorithm {

template<typename T>
requires std::floating_point<T>
struct FFT {
    using complex_t = gr::complex<T>;

    std::size_t            fftSize{0};
    std::vector<complex_t> twiddleFactors;
    std::vector<complex_t> scratch;

    FFT() = default;

    void init(std::size_t N) {
        if (!std::has_single_bit(N)) {
            throw std::invalid_argument(std::format("FFT size must be power of 2, got: {}", N));
        }
        fftSize = N;
        twiddleFactors.resize(fftSize);
        scratch.resize(fftSize / 2);

        const T minus2PiOverN = -static_cast<T>(2) * std::numbers::pi_v<T> / static_cast<T>(fftSize);
        for (std::size_t k = 0; k < fftSize; ++k) {
            T angle           = minus2PiOverN * static_cast<T>(k);
            twiddleFactors[k] = complex_t{std::cos(angle), std::sin(angle)};
        }
    }

    std::size_t bitReverse(std::size_t x, int bits) const {
        std::size_t r = 0;
        for (int i = 0; i < bits; ++i) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        return r;
    }

    void bitReversalPermutation(std::vector<complex_t>& vec) const {
        std::size_t n    = vec.size();
        const int   bits = std::countr_zero(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = bitReverse(i, bits);
            if (i < j) {
                std::swap(vec[i], vec[j]);
            }
        }
    }

    void compute1(std::vector<complex_t>& data) {
        if (data.size() != fftSize) {
            init(data.size());
        }
        bitReversalPermutation(data);

        for (std::size_t s = 2; s <= fftSize; s *= 2) {
            std::size_t half   = s / 2;
            std::size_t stride = fftSize / s;
            for (std::size_t k = 0; k < fftSize; k += s) {
                for (std::size_t j = 0; j < half; j += 1) {
                    __m128d o = _mm_load_pd(reinterpret_cast<double*>(&data[k + j + half]));

                    const complex_t w  = twiddleFactors[j * stride];
                    const T         cc = w.re;
                    const T         ss = w.im;

                    __m128d wr = _mm_load1_pd(&cc);
                    __m128d wi = _mm_set_pd(ss, -ss);

                    wr         = _mm_mul_pd(o, wr);
                    __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                    wi         = _mm_mul_pd(n1, wi);
                    n1         = _mm_add_pd(wr, wi);

                    __m128d u    = _mm_load_pd(reinterpret_cast<double*>(&data[k + j]));
                    __m128d sum  = _mm_add_pd(u, n1);
                    __m128d diff = _mm_sub_pd(u, n1);

                    _mm_store_pd(reinterpret_cast<double*>(&data[k + j]), sum);
                    _mm_store_pd(reinterpret_cast<double*>(&data[k + j + half]), diff);
                }
            }
        }
    }

    constexpr void separate(complex_t* a, std::size_t n) {
        for (std::size_t i = 0; i < n / 2; i++) {
            scratch[i] = a[i * 2 + 1];
        }
        for (std::size_t i = 0; i < n / 2; i++) {
            a[i] = a[i * 2];
        }
        for (std::size_t i = 0; i < n / 2; i++) {
            a[i + n / 2] = scratch[i];
        }
    }

    constexpr void fft2_simd(complex_t* X, std::size_t N, std::size_t stride = 1) {
        if (N < 2) {
            return;
        }

        separate(X, N);
        fft2_simd(X, N / 2, stride * 2);
        fft2_simd(X + N / 2, N / 2, stride * 2);

        for (std::size_t k = 0; k < N / 2; k++) {
            const complex_t& w  = twiddleFactors[k * stride];
            double           cc = w.re;
            double           ss = w.im;

            __m128d xo          = _mm_load_pd(reinterpret_cast<double*>(&X[k + N / 2]));
            __m128d wr          = _mm_load1_pd(&cc);
            __m128d wi          = _mm_set_pd(ss, -ss);
            __m128d t1          = _mm_mul_pd(xo, wr);
            __m128d xoi         = _mm_shuffle_pd(xo, xo, _MM_SHUFFLE2(0, 1));
            __m128d t2          = _mm_mul_pd(xoi, wi);
            __m128d twiddle_mul = _mm_add_pd(t1, t2);

            __m128d xu   = _mm_load_pd(reinterpret_cast<double*>(&X[k]));
            __m128d sum  = _mm_add_pd(xu, twiddle_mul);
            __m128d diff = _mm_sub_pd(xu, twiddle_mul);

            _mm_store_pd(reinterpret_cast<double*>(&X[k]), sum);
            _mm_store_pd(reinterpret_cast<double*>(&X[k + N / 2]), diff);
        }
    }

    void compute2(std::vector<complex_t>& data) {
        if (data.size() != fftSize) {
            init(data.size());
        }
        fft2_simd(data.data(), fftSize);
    }

    void compute3(std::vector<complex_t>& data) {
        if (data.size() != fftSize) {
            init(data.size());
        }
        bitReversalPermutation(data);

        constexpr std::size_t stride = 2; // AVX: 2 doubles == 1 complex

        for (std::size_t s = 2; s <= fftSize; s *= 2) {
            std::size_t half = s / 2;
            std::size_t step = fftSize / s;
            for (std::size_t k = 0; k < fftSize; k += s) {
                std::size_t j = 0;
                for (; j + stride <= half; j += stride) {
                    const complex_t& w0 = twiddleFactors[(j + 0) * step];
                    const complex_t& w1 = twiddleFactors[(j + 1) * step];

                    __m256d lo = _mm256_loadu_pd(reinterpret_cast<double*>(&data[k + j]));
                    __m256d hi = _mm256_loadu_pd(reinterpret_cast<double*>(&data[k + j + half]));

                    __m256d wre = _mm256_set_pd(w1.re, w1.re, w0.re, w0.re);
                    __m256d wim = _mm256_set_pd(-w1.im, w1.im, -w0.im, w0.im);

                    __m256d hi_shuf = _mm256_shuffle_pd(hi, hi, 0b0101);
                    __m256d t       = _mm256_add_pd(_mm256_mul_pd(hi, wre), _mm256_mul_pd(hi_shuf, wim));

                    _mm256_storeu_pd(reinterpret_cast<double*>(&data[k + j]), _mm256_add_pd(lo, t));
                    _mm256_storeu_pd(reinterpret_cast<double*>(&data[k + j + half]), _mm256_sub_pd(lo, t));
                }
                for (; j < half; ++j) {
                    auto w             = twiddleFactors[j * stride];
                    auto t             = w * data[k + j + half];
                    auto u             = data[k + j];
                    data[k + j]        = u + t;
                    data[k + j + half] = u - t;
                }
            }
        }
    }

    void compute4_rec(complex_t* data, std::size_t n, std::size_t stride) {
        if (n == 1) {
            return;
        }
        compute4_rec(data, n / 2, 2 * stride);
        compute4_rec(data + stride, n / 2, 2 * stride);
        for (std::size_t k = 0; k < n / 2; k += 2) {
            complex_t w0 = twiddleFactors[(k + 0) * fftSize / n];
            complex_t w1 = twiddleFactors[(k + 1) * fftSize / n];

            __m256d lo = _mm256_loadu_pd(reinterpret_cast<double*>(&data[stride * (2 * k)]));
            __m256d hi = _mm256_loadu_pd(reinterpret_cast<double*>(&data[stride * (2 * k + 1)]));

            __m256d wre = _mm256_set_pd(w1.re, w1.re, w0.re, w0.re);
            __m256d wim = _mm256_set_pd(-w1.im, w1.im, -w0.im, w0.im);

            __m256d hi_shuf = _mm256_shuffle_pd(hi, hi, 0b0101);
            __m256d t       = _mm256_add_pd(_mm256_mul_pd(hi, wre), _mm256_mul_pd(hi_shuf, wim));

            _mm256_storeu_pd(reinterpret_cast<double*>(&data[stride * (2 * k)]), _mm256_add_pd(lo, t));
            _mm256_storeu_pd(reinterpret_cast<double*>(&data[stride * (2 * k + 1)]), _mm256_sub_pd(lo, t));
        }
    }

    void compute4(std::vector<complex_t>& data) { compute4_rec(data.data(), fftSize, 1); }

    void compute5(std::vector<complex_t>& data) {
        bitReversalPermutation(data);
        std::size_t N = fftSize;
        for (std::size_t len = 2; len <= N; len *= 2) {
            const std::size_t half = len / 2;
            const std::size_t step = fftSize / len;
            for (std::size_t i = 0; i < N; i += len) {
                for (std::size_t j = 0; j < half; j += 4) {
                    if (j + 3 >= half) {
                        for (std::size_t jj = j; jj < half; ++jj) {
                            auto w              = twiddleFactors[jj * step];
                            auto t              = w * data[i + jj + half];
                            auto u              = data[i + jj];
                            data[i + jj]        = u + t;
                            data[i + jj + half] = u - t;
                        }
                        break;
                    }
                    __m256d lo      = _mm256_loadu_pd(reinterpret_cast<double*>(&data[i + j]));
                    __m256d hi      = _mm256_loadu_pd(reinterpret_cast<double*>(&data[i + j + half]));
                    auto    w0      = twiddleFactors[(j + 0) * step];
                    auto    w1      = twiddleFactors[(j + 1) * step];
                    auto    w2      = twiddleFactors[(j + 2) * step];
                    auto    w3      = twiddleFactors[(j + 3) * step];
                    __m256d wre     = _mm256_set_pd(w3.re, w2.re, w1.re, w0.re);
                    __m256d wim     = _mm256_set_pd(-w3.im, -w2.im, -w1.im, -w0.im);
                    __m256d hi_shuf = _mm256_shuffle_pd(hi, hi, 0b0101);
                    __m256d t       = _mm256_add_pd(_mm256_mul_pd(hi, wre), _mm256_mul_pd(hi_shuf, wim));
                    _mm256_storeu_pd(reinterpret_cast<double*>(&data[i + j]), _mm256_add_pd(lo, t));
                    _mm256_storeu_pd(reinterpret_cast<double*>(&data[i + j + half]), _mm256_sub_pd(lo, t));
                }
            }
        }
    }

    using U = std::complex<float>;
    using FFT_t = pffft::Fft<U>;
    FFT_t prettyFastFFT = FFT_t(static_cast<int>(fftSize), 4096);
    pffft::AlignedVector<U> input = prettyFastFFT.valueVector();
    pffft::AlignedVector<U> output = prettyFastFFT.spectrumVector();

    void compute6(std::vector<complex_t>& data) {

        if (static_cast<int>(fftSize) != prettyFastFFT.getLength()) {
            prettyFastFFT.prepareLength(static_cast<int>(fftSize));
            input = prettyFastFFT.valueVector();
            output = prettyFastFFT.spectrumVector();
        }

        // copy input data to aligned input vector
        for (std::size_t k = 0; k < fftSize; ++k) {
            input[k] = U(static_cast<float>(data[k].re), static_cast<float>(data[k].im));
        }
        // do the forward transform; write complex spectrum result into Y
        prettyFastFFT.forward(input, output);

        for (std::size_t k = 0; k < fftSize; ++k) {
            data[k].re = output[k].real();
            data[k].im = output[k].imag();
        }
    }

    void compute(std::vector<complex_t>& data, std::size_t version = 0) {
        switch (version) {
        case 0: compute1(data); break;
        case 1: compute2(data); break;
        case 2: compute3(data); break;
        case 3: compute4(data); break;
        case 4: compute5(data); break;
        case 5: compute6(data); break;
        default: throw std::invalid_argument(std::format("FFT version {} not implemented (yet)", version));
        }
    }
};

} // namespace algorithm
} // namespace gr

int main() {
    using T         = double;
    using FFT       = gr::algorithm::FFT<T>;
    using complex_t = gr::complex<T>;

    constexpr std::size_t  N = 8192;
    std::vector<complex_t> signal(N);
    std::vector<complex_t> output(N);

    for (std::size_t i = 0; i < N; ++i) {
        T angle   = 2 * std::numbers::pi_v<T> * static_cast<T>(i) / static_cast<T>(N) * static_cast<T>(5);
        signal[i] = complex_t{std::sin(angle), 0}; // peak at bin 5
    }

    FFT fft;
    fft.init(N);
    std::copy(signal.begin(), signal.end(), output.begin());
    fft.compute(output); // warm up
    std::copy(signal.begin(), signal.end(), output.begin());

    constexpr bool debug = false;
    for (std::size_t version = 0; version < 6; ++version) {
        fft.compute(output, version); // warm up
        std::copy(signal.begin(), signal.end(), output.begin());

        using Clock          = std::chrono::high_resolution_clock;
        Clock::time_point t1 = Clock::now();
        fft.compute(output, version);
        Clock::time_point t2   = Clock::now();
        const float       usec = std::chrono::duration<float, std::micro>(t2 - t1).count();

        std::println("[SIMD_V{}] FFT (N={}): {:7.3f} us - {:.3f} MS/s", version + 1, N, usec, static_cast<float>(N) / usec);

        if (debug) {
            for (std::size_t i = 0; i < std::min(N / 2, 10UL); ++i) {
                std::println("  bin[{:2}] = {:.3f} + {:.3f}j", i, output[i].real(), output[i].imag());
            }
            std::println("  ...");
            for (std::size_t i = N - std::min(N / 2, 10UL); i < N; ++i) {
                std::println("  bin[{:2}] = {:.3f} + {:.3f}j", i, output[i].real(), output[i].imag());
            }
        }
    }
}
