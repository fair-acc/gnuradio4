#include <chrono>
#include <cmath>
#include <complex>
#include <numbers>
#include <print>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fftw.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <sycl/sycl.hpp>

enum class BenchmarkTarget { SYCL, GR_FFT, GR_FFTW };

namespace gr {

struct complexf {
    float re{}, im{};
    constexpr complexf() = default;
    constexpr complexf(float r, float i = 0.0f) : re(r), im(i) {}

    friend complexf operator+(const complexf& a, const complexf& b) { return {a.re + b.re, a.im + b.im}; }
    friend complexf operator-(const complexf& a, const complexf& b) { return {a.re - b.re, a.im - b.im}; }
    friend complexf operator*(const complexf& a, const complexf& b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

    float real() const { return re; }
    float imag() const { return im; }
};

inline complexf polar(float r, float theta) { return {r * sycl::cos(theta), r * sycl::sin(theta)}; }

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

// --- SYCL-based Cooley-Tukey FFT
void compute_fft_inplace(sycl::queue& q, complexf* data, std::size_t N, bool inverse = false) {
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
        auto* data = sycl::malloc_shared<gr::complexf>(N, queue.value());
        gr::compute_fft_inplace(queue.value(), data, N); // warm up cache

        t1 = Clock::now();
        std::memcpy(data, input.data(), N * sizeof(gr::complexf));
        gr::compute_fft_inplace(queue.value(), data, N);
        std::memcpy(output.data(), data, N * sizeof(gr::complexf));
        t2 = Clock::now();
        sycl::free(data, queue.value());
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
    constexpr std::size_t N = 8192;
    for (const auto& dev : sycl::device::get_devices()) {
        if (dev.is_cpu()) {
            run_fft_benchmark<BenchmarkTarget::SYCL>(N, sycl::queue{dev});
        }
        if (dev.is_gpu()) {
            run_fft_benchmark<BenchmarkTarget::SYCL>(N, sycl::queue{dev});
        }
    }

    run_fft_benchmark<BenchmarkTarget::GR_FFT>(N);
    run_fft_benchmark<BenchmarkTarget::GR_FFTW>(N);
}
