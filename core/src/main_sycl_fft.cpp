#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <execution>
#include <functional>
#include <numbers>
#include <numeric>
#include <print>
#include <span>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <sycl/sycl.hpp>

enum class BenchmarkTarget { SYCL, GR_FFT, GR_FFTW };

namespace gr {

template<typename T>
struct SyclShared {
    sycl::queue& q;
    T*           ptr;

    SyclShared(sycl::queue& q, std::size_t n) : q(q), ptr(sycl::malloc_shared<T>(n, q)) {}

    ~SyclShared() {
        if (ptr) {
            sycl::free(ptr, q);
        }
    }

    T* get() { return ptr; }
    T& operator[](std::size_t i) { return ptr[i]; }
    operator T*() { return ptr; }
};

struct complexf {
    float re{}, im{};

    constexpr complexf() = default;
    constexpr complexf(float r, float i = 0.0f) : re(r), im(i) {}

    friend complexf operator+(const complexf& a, const complexf& b) { return {a.re + b.re, a.im + b.im}; }
    friend complexf operator-(const complexf& a, const complexf& b) { return {a.re - b.re, a.im - b.im}; }
    // friend complexf operator*(const complexf& a, const complexf& b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

    complexf operator*(const complexf& other) const { return {re * other.re - im * other.im, re * other.im + im * other.re}; }
    complexf operator*(const std::complex<float>& other) const { return {re * other.real() - im * other.imag(), re * other.imag() + im * other.real()}; }

    float real() const { return re; }
    float imag() const { return im; }
};

inline complexf polar(float r, float theta) { return {r * sycl::cos(theta), r * sycl::sin(theta)}; }

inline complexf exp(float real_part, float imag_part) {
    float magnitude = std::exp(real_part);
    return {magnitude * sycl::cos(imag_part), magnitude * sycl::sin(imag_part)};
}

template<typename T>
void swap(T& a, T& b) noexcept {
    T tmp = a;
    a     = b;
    b     = tmp;
}

template<typename T>
struct TwiddleCache {
    using allocator_t = sycl::usm_allocator<complexf, sycl::usm::alloc::shared>;
    using vector_t    = std::vector<complexf, allocator_t>;

    static TwiddleCache& instance(sycl::queue& q, bool inverse = false) {
        static TwiddleCache forward{q, false};
        static TwiddleCache inverse_{q, true};
        return inverse ? inverse_ : forward;
    }

    const vector_t& get(std::size_t N) {
        std::scoped_lock lock(_mutex);
        auto [it, inserted] = _cache.try_emplace(N, vector_t{_alloc});
        if (inserted) {
            auto& twiddles = it->second;
            twiddles.resize(N / 2);
            const T sign       = _inverse ? 1.f : -1.f;
            const T theta_base = sign * 2 * std::numbers::pi_v<T> / static_cast<T>(N);
            for (std::size_t k = 0; k < N / 2; ++k) {
                T theta     = theta_base * static_cast<T>(k);
                twiddles[k] = polar(1.f, theta);
            }
        }
        return it->second;
    }

    void prime(std::size_t N) { (void)get(N); }
    void clear() {
        std::scoped_lock lock(_mutex);
        _cache.clear();
    }

private:
    TwiddleCache(sycl::queue& q, bool inverse) : _queue(q), _inverse(inverse), _alloc(q) {}

    sycl::queue&                              _queue;
    const bool                                _inverse;
    allocator_t                               _alloc;
    std::unordered_map<std::size_t, vector_t> _cache;
    std::mutex                                _mutex;
};

template<typename T>
struct BitReversalCache {
    using allocator_t = sycl::usm_allocator<std::size_t, sycl::usm::alloc::shared>;
    using index_t     = std::vector<std::size_t, allocator_t>;

    static BitReversalCache& instance(sycl::queue& q) {
        static BitReversalCache cache{q};
        return cache;
    }

    const index_t& get(std::size_t N) {
        std::scoped_lock lock(_mutex);
        auto [it, inserted] = _cache.try_emplace(N, index_t{_alloc});

        if (inserted) {
            auto& bit_rev = it->second;
            bit_rev.resize(N);

            // Precompute bit-reversal table
            const std::size_t nStages = static_cast<std::size_t>(std::countr_zero(N));
            for (std::size_t i = 0; i < N; ++i) {
                std::size_t rev = 0;
                for (std::size_t k = 0; k < nStages; ++k) {
                    rev = (rev << 1) | ((i >> k) & 1);
                }
                bit_rev[i] = rev;
            }
        }
        return it->second;
    }

    void prime(std::size_t N) { (void)get(N); }

    void clear() {
        std::scoped_lock lock(_mutex);
        _cache.clear();
    }

private:
    BitReversalCache(sycl::queue& q) : _queue(q), _alloc(q) {}

    sycl::queue&                             _queue;
    allocator_t                              _alloc;
    std::unordered_map<std::size_t, index_t> _cache;
    std::mutex                               _mutex;
};

// --- SYCL-based Cooley-Tukey FFT
void compute_fft_inplace_orgiginal(sycl::queue& q, complexf* data, std::size_t N, bool inverse = false) {
    const std::size_t nStages     = static_cast<std::size_t>(std::countr_zero(N));
    auto&             cache       = TwiddleCache<float>::instance(q, inverse);
    const auto&       twiddle_vec = cache.get(N);
    const complexf*   twiddles    = twiddle_vec.data();

    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
             std::size_t j = i[0], rev = 0;
             for (std::size_t k = 0; k < nStages; ++k) {
                 rev = (rev << 1) | ((j >> k) & 1);
             }
             if (j < rev) {
                 swap(data[j], data[rev]);
             }
         });
     }).wait();

    for (std::size_t stage = 1; stage <= nStages; ++stage) {
        std::size_t m = 1 << stage, m2 = m >> 1;
        std::size_t stride = N / m; // twiddle stride for this stage

        q.submit([&](sycl::handler& h) {
             h.parallel_for(sycl::range<1>{N / m}, [=](sycl::id<1> grp) {
                 std::size_t base = grp[0] * m;
                 for (std::size_t k = 0; k < m2; ++k) {
                     // auto w              = polar(1.f, sign * 2.f * pi * k / m);
                     // auto t              = w * data[base + k + m2];
                     // auto u              = data[base + k];
                     std::size_t twid_idx = stride * k;
                     complexf    w        = twiddles[twid_idx];
                     complexf    t        = w * data[base + k + m2];
                     complexf    u        = data[base + k];
                     data[base + k]       = u + t;
                     data[base + k + m2]  = u - t;
                 }
             });
         }).wait();
    }

    if (inverse) {
        q.submit([&](sycl::handler& h) {
             h.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
                 data[i].re /= static_cast<float>(N);
                 data[i].im /= static_cast<float>(N);
             });
         }).wait();
    }
}

// Twiddle factor exponentiation function (adjust for float precision)
template<typename T = complexf>
inline constexpr T exp_1_8(float tid) { return {sycl::cos(-2.0f * std::numbers::pi_v<float> * tid / 8), sycl::sin(-2.0f * std::numbers::pi_v<float> * tid / 8)}; }

template<typename T = complexf>
inline constexpr T exp_1_4(float tid) { return {sycl::cos(-2.0f * std::numbers::pi_v<float> * tid / 4), sycl::sin(-2.0f * std::numbers::pi_v<float> * tid / 4)}; }

template<typename T = complexf>
inline constexpr T exp_3_8(float tid) { return {sycl::cos(-6.0f * std::numbers::pi_v<float> * tid / 8), sycl::sin(-6.0f * std::numbers::pi_v<float> * tid / 8)}; }

// Define constants for scaling
// constexpr float m_sqrt1_2 = 1.0f / std::sqrt(2.0f);

// FFT2 function (for 2-point FFT)
template<typename T>
void FFT2(T* a0, T* a1) {
    T c0 = *a0;
    *a0         = *a0 + *a1;
    *a1         = c0 - *a1;
}

// FFT4 function (for 4-point FFT)
template<typename T>
void FFT4(T* a0, T* a1, T* a2, T* a3) {
    FFT2(a0, a2);
    FFT2(a1, a3);
    *a3 = *a3 * exp_1_4<T>(0);
    FFT2(a0, a1);
    FFT2(a2, a3);
}

// FFT8 function (for 8-point FFT)
template<typename T>
void FFT8(T a[8]) {
    // Stage 1: Compute FFT2 for pairs
    FFT2(&a[0], &a[4]);
    FFT2(&a[1], &a[5]);
    FFT2(&a[2], &a[6]);
    FFT2(&a[3], &a[7]);

    // Apply twiddle factors and scaling
    a[5] = *(&a[5]) * exp_1_8<T>(5);
    a[6] = a[6] * exp_1_4<T>(6);
    a[7] = a[7] * exp_3_8<T>(7);

    // Stage 2: Compute FFT4 for pairs
    FFT4(&a[0], &a[1], &a[2], &a[3]);
    FFT4(&a[4], &a[5], &a[6], &a[7]);
}

void compute_fft_inplace(sycl::queue& q, complexf* data_in, std::size_t N) {
    const std::size_t nThreadsTotal = N;
    constexpr std::size_t nThreadsPerBlock = 64;  // Threads per block
    [[maybe_unused]] const std::size_t nBlocks = nThreadsTotal / nThreadsPerBlock; // Number of blocks

    auto fftKernel = [&](sycl::handler& handler) {
        // Allocate local memory for 8 data points and intermediate stages for FFT
        sycl::local_accessor<float, 1> smem(sycl::range<1>(nThreadsPerBlock * 9), handler); // 64 * 9 for intermediate data

        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(nThreadsTotal), sycl::range<1>(nThreadsPerBlock)), [=](sycl::nd_item<1> item) {
            std::size_t tid = item.get_local_id(0);
            std::size_t gid = item.get_group(0);
            std::size_t blockIdx = gid * nThreadsPerBlock; // Starting index for each block

            // Ensure we don't exceed bounds of data array
            assert(blockIdx + 7 < N);

            complexf data[8];                               // Array to store 8 complex numbers for FFT computation
            const int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7}; // Bit-reversal lookup

            // Load data from global memory (input data) into local array
            for (std::size_t i = 0; i < 8; i++) {
                assert(blockIdx + i < N);        // Ensure valid memory access
                data[i] = data_in[blockIdx + i]; // Load data from input array
            }

            // Apply FFT8 (radix-8 FFT)
            FFT8(data);

            // Apply twiddle factors for the first stage (using tid)
            const float tidf = static_cast<float>(tid);
            data[1] = data[1] * exp_1_8(tidf);
            data[2] = data[2] * exp_1_4(tidf);
            data[3] = data[3] * exp_3_8(tidf);
            data[4] = data[4] * exp_1_4(tidf);
            data[5] = data[5] * exp_1_8(tidf);
            data[6] = data[6] * exp_1_4(tidf);
            data[7] = data[7] * exp_3_8(tidf);

            // Split tid into higher and lower bits (hi and lo) for the transposition
            std::size_t hi = tid >> 3; // higher part of the tid (block-wide part)
            std::size_t lo = tid & 7;  // lower part of the tid (within block)

            // Transpose and store the result in shared memory
            for (std::size_t i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 66] = data[reversed[i]].re; // Store real part
            }

            item.barrier(sycl::access::fence_space::local_space); // Synchronize the threads

            // After transposition, move data back to global memory
            for (std::size_t i = 0; i < 8; i++) {
                data[reversed[i]].re = smem[lo * 66 + hi + i * 8]; // Access real part
            }
            item.barrier(sycl::access::fence_space::local_space);

            for (std::size_t i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 66] = data[reversed[i]].im; // Store imaginary part
            }

            item.barrier(sycl::access::fence_space::local_space); // Synchronize

            // Move the y component (imaginary part) back after transposition
            for (std::size_t i = 0; i < 8; i++) {
                data[reversed[i]].im = smem[lo * 66 + hi + i * 8]; // Access imaginary part
            }

            item.barrier(sycl::access::fence_space::local_space); // Synchronize

            // Apply second FFT8 on transposed data
            FFT8(data);

            // Apply twiddle factors for the second stage (using hi)
            const float hif = static_cast<float>(hi);
            data[1] = data[1] * exp_1_8(hif);
            data[2] = data[2] * exp_1_4(hif);
            data[3] = data[3] * exp_3_8(hif);
            data[4] = data[4] * exp_1_4(hif);
            data[5] = data[5] * exp_1_8(hif);
            data[6] = data[6] * exp_1_4(hif);
            data[7] = data[7] * exp_3_8(hif);

            // Transpose the results back to the global memory
            for (std::size_t i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 72] = data[reversed[i]].re; // Store real part
            }

            item.barrier(sycl::access::fence_space::local_space);

            for (std::size_t i = 0; i < 8; i++) {
                data[reversed[i]].re = smem[hi * 72 + lo + i * 8]; // Access real part
            }

            item.barrier(sycl::access::fence_space::local_space);

            for (std::size_t i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 72] = data[reversed[i]].im; // Store imaginary part
            }

            item.barrier(sycl::access::fence_space::local_space);

            for (std::size_t i = 0; i < 8; i++) {
                data[reversed[i]].im = smem[hi * 72 + lo + i * 8]; // Access imaginary part
            }

            // Store the final result back to global memory
            for (std::size_t i = 0; i < 8; i++) {
                data_in[blockIdx + i] = smem[tid * 8 + i]; // Only store real part
            }
        });
    };

    q.submit(fftKernel);
    q.wait();
}


#define exp_1_8   (sycl::float2){  1, -1 } // requires post-multiply by 1/sqrt(2)
#define exp_1_4   (sycl::float2){  0, -1 }
#define exp_3_8   (sycl::float2){ -1, -1 } // requires post-multiply by 1/sqrt(2)

sycl::float2 exp_i(float phi) {
    return sycl::float2{ sycl::cos(phi), sycl::sin(phi) };
}

sycl::float2 cmplx_mul(sycl::float2 a, sycl::float2 b) {
    return sycl::float2{ a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x() };
}

sycl::float2 cm_fl_mul(sycl::float2 a, float b) {
    return sycl::float2{ b * a.x(), b * a.y() };
}

sycl::float2 cmplx_add(sycl::float2 a, sycl::float2 b) {
    return sycl::float2{ a.x() + b.x(), a.y() + b.y() };
}

sycl::float2 cmplx_sub(sycl::float2 a, sycl::float2 b) {
    return sycl::float2{ a.x() - b.x(), a.y() - b.y() };
}

void compute_fft(sycl::queue &q, sycl::float2 *data_in, std::size_t ) {
    const size_t N = 512;  // Single 512-point FFT

    // FFT kernel
    auto fft_kernel = [&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::float2, 1> smem(sycl::range<1>(8 * 8 * 9), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(64)), [=](sycl::nd_item<1> item) {
            std::size_t tid = item.get_local_id(0);
            std::size_t blockIdx = item.get_group(0) * 512 + tid;
            sycl::float2 data[8];
            const int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

            // Load data from input
            for (std::size_t i = 0; i < 8; i++) {
                data[i] = data_in[blockIdx + i * 64];  // Use data
            }

            // Apply FFT8
            FFT8(data);

            // Twiddle factor adjustment for FFT8
            for (std::size_t j = 1; j < 8; j++) {
                data[j] = cmplx_mul(data[j], exp_i((-2.0f * std::numbers::pi_v<float> * static_cast<float>(reversed[j]) / static_cast<float>(512)) * static_cast<float>(tid)));
            }

            // Transposition and store the result back into memory
            for (std::size_t i = 0; i < 8; i++) {
                smem[tid * 8 + i] = data[reversed[i]];
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Move data back to global memory
            for (std::size_t i = 0; i < 8; i++) {
                data_in[blockIdx + i * 64] = smem[tid * 8 + i];
            }
        });
    };

    q.submit(fft_kernel);
    q.wait();
}

} // namespace gr

template<BenchmarkTarget Target>
void run_fft_benchmark(std::size_t N, std::optional<sycl::queue> queue = std::nullopt) {
    std::vector<std::complex<float>> input(N);
    std::vector<std::complex<float>> output(N);
    for (std::size_t i = 0UZ; i < N; ++i) {
        input[i] = std::sin(2 * std::numbers::pi_v<float> * 5.f * static_cast<float>(i) / static_cast<float>(N));
    }

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t1;
    Clock::time_point t2;
    if constexpr (Target == BenchmarkTarget::SYCL) {
        gr::TwiddleCache<float>::instance(queue.value()).prime(N);
        auto data_in  = gr::SyclShared<sycl::float2>(queue.value(), N);
        auto data_out = gr::SyclShared<sycl::float2>(queue.value(), N);
        gr::compute_fft(queue.value(), data_in.get(), N); // warm up cache

        t1 = Clock::now();
        std::memcpy(data_in.get(), input.data(), N * sizeof(sycl::float2));
        gr::compute_fft(queue.value(), data_in.get(), N);
        std::memcpy(output.data(), data_in.get(), N * sizeof(sycl::float2));
        t2 = Clock::now();
    } else if constexpr (Target == BenchmarkTarget::GR_FFT) {
        static gr::algorithm::FFT<std::complex<float>, std::complex<float>> fft;
        output = fft.compute(input); // warm up cache/twiddlefactors
        t1     = Clock::now();
        output = fft.compute(input);
        t2     = Clock::now();
    } else {
        static gr::algorithm::FFTw<std::complex<float>, std::complex<float>> fft;
        output = fft.compute(input); // warm up cache/twiddlefactors
        t1     = Clock::now();
        output = fft.compute(input);
        t2     = Clock::now();
    }
    const float usec        = std::chrono::duration<float, std::micro>(t2 - t1).count();
    std::string device_name = queue ? queue->get_device().get_info<sycl::info::device::name>() : "CPU (default)";

    std::println("[{}] FFT (N={}): {:.3f} us - {:.3f} MS/s - on: {}", Target, N, usec, static_cast<float>(N) / usec, device_name);

    for (std::size_t i = 0; i < std::min(N / 2, 10UL); ++i) {
        std::println("  bin[{:2}] = {:.3f} + {:.3f}j", i, output[i].real(), output[i].imag());
    }
    std::println("  ...");
    for (std::size_t i = N - std::min(N / 2, 10UL); i < N; ++i) {
        std::println("  bin[{:2}] = {:.3f} + {:.3f}j", i, output[i].real(), output[i].imag());
    }
}

int main() {
    constexpr std::size_t N = 512;
    for (const auto& dev : sycl::device::get_devices()) {
        // if (dev.is_cpu()) {
        //     run_fft_benchmark<BenchmarkTarget::SYCL>(N, sycl::queue{dev});
        // }
        if (dev.is_gpu()) {
            run_fft_benchmark<BenchmarkTarget::SYCL>(N, sycl::queue{dev});
        }
    }

    // run_fft_benchmark<BenchmarkTarget::GR_FFT>(N);
    run_fft_benchmark<BenchmarkTarget::GR_FFTW>(N);
}
