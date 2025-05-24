#include <chrono>
#include <cmath>
#include <complex>
#include <format>
#include <print>

#include <sycl/sycl.hpp>

template<typename T = sycl::float2>
constexpr T exp_1_8() {
    return T{+1.0f, -1.0f};
}
template<typename T = sycl::float2>
constexpr T exp_1_4() {
    return T{+0.0f, -1.0f};
}
template<typename T = sycl::float2>
constexpr T exp_3_8() {
    return T{-1.0f, -1.0f};
}

constexpr sycl::float2 cmplx_mul(sycl::float2 a, sycl::float2 b) { return (sycl::float2){a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x()}; }
constexpr sycl::float2 cm_fl_mul(sycl::float2 a, float b) { return (sycl::float2){b * a.x(), b * a.y()}; }
constexpr sycl::float2 cmplx_add(sycl::float2 a, sycl::float2 b) { return (sycl::float2){a.x() + b.x(), a.y() + b.y()}; }
constexpr sycl::float2 cmplx_sub(sycl::float2 a, sycl::float2 b) { return (sycl::float2){a.x() - b.x(), a.y() - b.y()}; }
constexpr sycl::float2 exp_i(float phi) { return sycl::float2{sycl::cos(phi), sycl::sin(phi)}; }

// FFT2 function (for 2-point FFT)
template<typename T = sycl::float2>
constexpr void FFT2(T* a0, T* a1) {
    T c0 = *a0;
    *a0  = *a0 + *a1;
    *a1  = c0 - *a1;
}

// FFT4 function (for 4-point FFT)
template<typename T = sycl::float2>
constexpr void FFT4(T* a0, T* a1, T* a2, T* a3) {
    FFT2(a0, a2);
    FFT2(a1, a3);
    *a3 = cmplx_mul(*a3, exp_1_4<T>());
    FFT2(a0, a1);
    FFT2(a2, a3);
}

// FFT8 function (for 8-point FFT)
template<typename T>
constexpr void FFT8(T a[8]) {
    // Stage 1: Compute FFT2 for pairs
    FFT2(&a[0], &a[4]);
    FFT2(&a[1], &a[5]);
    FFT2(&a[2], &a[6]);
    FFT2(&a[3], &a[7]);

    // Apply twiddle factors and scaling
    a[5] = cm_fl_mul(cmplx_mul(a[5], exp_1_8<T>()), 1.0f / std::sqrt(2.0f));
    a[6] = cmplx_mul(a[6], exp_1_4<T>());
    a[7] = cm_fl_mul(cmplx_mul(a[7], exp_3_8<T>()), 1.0f / std::sqrt(2.0f));

    // Stage 2: Compute FFT4 for pairs
    FFT4(&a[0], &a[1], &a[2], &a[3]);
    FFT4(&a[4], &a[5], &a[6], &a[7]);
}

constexpr sycl::float2 compute_twiddle(int k, int N, float index) {
    float angle = -2.0f * std::numbers::pi_v<float> * static_cast<float>(k) / static_cast<float>(N) * index;
    return sycl::float2{sycl::cos(angle), sycl::sin(angle)};
}

// SYCL kernel for FFT computation
void compute_fft(sycl::queue& q, sycl::float2* work, std::size_t N = 512) {
    const std::size_t                  nThreadsTotal    = N;
    constexpr std::size_t              nThreadsPerBlock = 64;                               // Threads per block
    [[maybe_unused]] const std::size_t nBlocks          = nThreadsTotal / nThreadsPerBlock; // Number of blocks
    const size_t                       globalsz         = nThreadsPerBlock;

    auto fft_kernel = [&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(8 * 8 * 9), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(globalsz), sycl::range<1>(nThreadsPerBlock)), [=](sycl::nd_item<1> item) {
            std::size_t  tid      = item.get_local_id(0);
            std::size_t  blockIdx = item.get_group(0) + tid;
            std::size_t  hi       = tid >> 3;
            std::size_t  lo       = tid & 7;
            sycl::float2 data[8];
            //__local T smem[8*8*9];
            constexpr std::size_t reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

            // starting index of data to/from global memory
            // globalLoads8(data, work, nThreadsPerBlock)
            for (int i = 0; i < 8; i++) {
                data[i] = work[blockIdx + i * nThreadsPerBlock];
            }

            FFT8(data);

            // twiddle8( data, tid, N );
            for (int j = 1; j < 8; j++) {
                data[j] = cmplx_mul(data[j], compute_twiddle(reversed[j], N, static_cast<float>(tid)));
            }

            // transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
            for (int i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 66] = data[reversed[i]].x();
            }
            item.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < 8; i++) {
                data[i].x() = smem[lo * 66 + hi + i * 8];
            }
            item.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 66] = data[reversed[i]].y();
            }
            item.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < 8; i++) {
                data[i].y() = smem[lo * 66 + hi + i * 8];
            }
            item.barrier(sycl::access::fence_space::local_space);

            FFT8(data);

            // twiddle8( data, hi, nThreadsPerBlock );
            for (int j = 1; j < 8; j++) {
                data[j] = cmplx_mul(data[j], compute_twiddle(reversed[j], nThreadsPerBlock, static_cast<float>(hi)));
            }

            // transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
            for (int i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 72] = data[reversed[i]].x();
            }
            item.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < 8; i++) {
                data[i].x() = smem[hi * 72 + lo + i * 8];
            }
            item.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < 8; i++) {
                smem[hi * 8 + lo + i * 72] = data[reversed[i]].y();
            }
            item.barrier(sycl::access::fence_space::local_space);
            for (int i = 0; i < 8; i++) {
                data[i].y() = smem[hi * 72 + lo + i * 8];
            }

            FFT8(data);

            // globalStores8(data, work, nThreadsPerBlock);
            for (int i = 0; i < 8; i++) {
                work[blockIdx + i * nThreadsPerBlock] = data[reversed[i]];
            }
        });
    };

    q.submit(fft_kernel).wait();
}

template<typename T>
void run_fft_benchmark(std::size_t N, sycl::queue& q) {
    std::vector<T> input(N);
    std::vector<T> output(N);

    // Generate a sine wave signal
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = std::sin(2 * std::numbers::pi_v<float> * 5.f * static_cast<float>(i) / static_cast<float>(N));
    }

    // Timing the FFT computation
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t1, t2;

    // Allocate memory for SYCL
    sycl::float2* data = sycl::malloc_device<sycl::float2>(N, q);

    // Copy input data to device
    q.memcpy(data, input.data(), N * sizeof(sycl::float2)).wait();

    t1 = Clock::now();
    // Compute FFT
    compute_fft(q, data, N);
    t2 = Clock::now();

    // Copy results back to host
    q.memcpy(output.data(), data, N * sizeof(sycl::float2)).wait();

    // Print timing results
    const float usec        = std::chrono::duration<float, std::micro>(t2 - t1).count();
    std::string device_name = q.get_device().get_info<sycl::info::device::name>();

    std::println("[SYCL] FFT (N={}): {:.3f} us - {:.3f} MS/s - on: {}", N, usec, static_cast<float>(N) / usec, device_name);

    for (std::size_t i = 0; i < std::min(N / 2, 10UL); ++i) {
        std::println("  bin[{:2}] = {:.3f} + {:.3f}j", i, output[i].real(), output[i].imag());
    }
    std::println("  ...");
    for (std::size_t i = N - std::min(N / 2, 10UL); i < N; ++i) {
        std::println("  bin[{:2}] = {:.3f} + {:.3f}j", i, output[i].real(), output[i].imag());
    }

    sycl::free(data, q); // Free device memory
}

int main() {
    sycl::queue q(sycl::gpu_selector{});

    const size_t N = 512; // Set FFT size
    run_fft_benchmark<std::complex<float>>(N, q);

    return 0;
}
