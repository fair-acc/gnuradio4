#include <algorithm>
#include <chrono>
#include <execution>
#include <functional>
#include <iostream>
#include <numeric>
#include <print>
#include <vector>

#include <sycl/sycl.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>

namespace gr {

enum class ExecutionTarget { CPU, GPU };

template<typename T>
using DeviceAllocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template<typename T>
using HostAllocator = std::allocator<T>;

class SyclEnv {
public:
    static SyclEnv& instance() {
        static SyclEnv env;
        return env;
    }

    sycl::queue& cpuQueue() { return _cpuQueue; }
    sycl::queue& gpuQueue() { return _gpuQueue; }

private:
    SyclEnv() : _cpuQueue(select_device(sycl::info::device_type::cpu)), _gpuQueue(select_device(sycl::info::device_type::gpu)) {}

    sycl::queue select_device(sycl::info::device_type type) {
        auto devs = sycl::device::get_devices(type);
        if (devs.empty()) {
            throw std::runtime_error("No SYCL device for type");
        }
        return sycl::queue{devs.front()};
    }

    sycl::queue _cpuQueue, _gpuQueue;
};

template<typename T, typename Allocator = std::allocator<T>>
struct vector {
    std::vector<T, Allocator> data;

    constexpr explicit vector(std::initializer_list<T> init, const Allocator& alloc) : data(init, alloc) {}

    constexpr vector(std::initializer_list<T> init)
    requires std::default_initializable<Allocator>
        : data(init) {}

    constexpr vector()
    requires std::default_initializable<Allocator>
        : data(Allocator{}) {}

    // Transparent forwarding
    auto        begin() const { return data.begin(); }
    auto        end() const { return data.end(); }
    auto        cbegin() const { return data.begin(); }
    auto        cend() const { return data.end(); }
    auto        size() const { return data.size(); }
    auto&       operator[](std::size_t i) { return data[i]; }
    const auto& operator[](std::size_t i) const { return data[i]; }

    operator std::span<T>() { return {data}; }
};

template<typename T, ExecutionTarget target>
using vector_for = std::conditional_t<target == ExecutionTarget::GPU, gr::vector<T, DeviceAllocator<T>>, gr::vector<T, HostAllocator<T>>>;

template<typename T, ExecutionTarget target>
vector_for<T, target> make_vector(std::initializer_list<T> init) {
    if constexpr (target == ExecutionTarget::GPU) {
        return vector<T, DeviceAllocator<T>>(init, DeviceAllocator<T>(SyclEnv::instance().gpuQueue()));
    } else {
        return vector<T, HostAllocator<T>>(init); // default allocator is fine
    }
}

// FIR filter with static coefficients
template<typename T, ExecutionTarget target = ExecutionTarget::CPU>
requires std::floating_point<T>
struct fir_filter : Block<fir_filter<T>> {
    using Description = Doc<R""(@brief Finite Impulse Response (FIR) filter class

    The transfer function of an FIR filter is given by:
    H(z) = b[0] + b[1]*z^-1 + b[2]*z^-2 + ... + b[N]*z^-N
    )"">;
    PortIn<T>                 in;
    PortOut<T>                out;
    gr::vector_for<T, target> b = make_vector<T, target>({T{1}, T{1}, T{1}, T{1}});
    // gr::vector<T> b{T{1}, T{1}, T{1}, T{1}}; // feedforward coefficients
    // std::array<T, 4> b{T{1}, T{1}, T{1}, T{1}}; // feedforward coefficients

    GR_MAKE_REFLECTABLE(fir_filter, in, out /*, b*/); // gr::vector not reflectable/supported (yet)

    HistoryBuffer<T, 32> inputHistory;

    void settingsChanged(const property_map& /*old_settings*/, const property_map& new_settings) noexcept {
        if (new_settings.contains("b") && b.size() > inputHistory.capacity()) {
            // inputHistory = HistoryBuffer<T>(std::bit_ceil(b.size()));
        }
    }

    constexpr T processOne(T input) noexcept {
        inputHistory.push_front(input);
        return std::transform_reduce(std::execution::unseq, b.cbegin(), b.cend(), inputHistory.cbegin(), T{0}, std::plus<>{}, std::multiplies<>{});
        // T acc = 0;
        // for (std::size_t j = 0; j < b.size(); ++j) {
        //     acc += b[j] * inputHistory[j];
        // }
        // return acc;
    }
};

// static_assert(std::is_trivially_copyable_v<fir_filter<float>>, "FIR block is not trivially copyable");
//  static_assert(std::is_trivially_copyable_v<std::vector<float>>, "föpat");
static_assert(std::is_trivially_copyable_v<gr::HistoryBuffer<float, 32>>, "HistoryBuffer");

// IIR filter with feedback state
template<typename T, ExecutionTarget target = ExecutionTarget::CPU>
struct IirFilter {
    float state = 0.f;
    float a{0.7f}, b{0.3f};

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V in) noexcept {
        float y = a * in + b * state;
        state   = y;
        return y;
    }
};

static_assert(std::is_trivially_copyable_v<IirFilter<float>>, "IIR block is not trivially copyable");

} // namespace gr

// Run FIR + IIR filter graph on given device
template<gr::ExecutionTarget target, bool debug = false>
void run_filter(const sycl::device& dev, const std::vector<float>& input) {
    std::size_t N = input.size();

    std::vector<float> output(N, 0.0f);
    sycl::queue        q{dev};

    float* d_input  = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);
    q.memcpy(d_input, input.data(), sizeof(float) * input.size()).wait();

    const auto t0 = std::chrono::steady_clock::now();

    gr::IirFilter<float, target>* iir = sycl::malloc_shared<gr::IirFilter<float, target>>(1, q);
    new (iir) gr::IirFilter<float, target>(); // optional if trivially constructible
    gr::fir_filter<float, target>* fir = sycl::malloc_shared<gr::fir_filter<float, target>>(1, q);
    new (fir) gr::fir_filter<float, target>(); // optional if trivially constructible

    q.submit([&](sycl::handler& h) {
         h.single_task([=]() {
             for (size_t i = 0; i < N; ++i) {
                 float fir_result = fir->processOne(d_input[i]);
                 float iir_result = iir->processOne(fir_result);
                 d_output[i]      = iir_result;
             }
         });
     }).wait();
    const auto t1 = std::chrono::steady_clock::now();

    std::copy(d_output, d_output + N, output.begin());

    if constexpr (debug) {
        std::println("Results from: {}", dev.get_info<sycl::info::device::name>());
        for (std::size_t i = 0; i < std::min<std::size_t>(20, output.size()); ++i) {
            std::println("  y[{:2}] = {:.3f}", i, d_output[i]);
        }
    }

    std::println("[Sequential FIR/IIR] results from '{}' (shared RAM): {} MS/s", dev.get_info<sycl::info::device::name>(), static_cast<float>(N) / static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(iir, q);
    sycl::free(fir, q);
}

// FIR filter applied with parallel_for
template<gr::ExecutionTarget target, bool debug = false>
void run_filter_parallel_fir(const sycl::device& dev, const std::vector<float>& input) {
    std::size_t N = input.size();
    sycl::queue q{dev};

    float* in_dev  = sycl::malloc_device<float>(N, q);
    float* out_dev = sycl::malloc_device<float>(N, q);
    q.memcpy(in_dev, input.data(), N * sizeof(float)).wait();

    // precompute impulse response of IIR as FIR (e.g. 64 taps)
    constexpr std::size_t NTAPS = 64;
    std::vector<float>    iir_ir(NTAPS);
    {
        float s = 0.f;
        for (std::size_t i = 0; i < NTAPS; ++i) {
            s         = 0.7f * (i == 0 ? 1.0f : 0.f) + 0.3f * s;
            iir_ir[i] = s;
        }
    }
    float* fir_taps = sycl::malloc_device<float>(NTAPS, q);
    q.memcpy(fir_taps, iir_ir.data(), NTAPS * sizeof(float)).wait();

    const auto t0 = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<1>{N - NTAPS}, [=](sycl::id<1> i) {
             float acc = 0.f;
             for (std::size_t k = 0; k < NTAPS; ++k) {
                 acc += fir_taps[k] * in_dev[i + NTAPS - 1 - k];
             }
             out_dev[i] = acc;
         });
     }).wait();

    const auto t1 = std::chrono::steady_clock::now();

    std::vector<float> output(N);
    q.memcpy(output.data(), out_dev, N * sizeof(float)).wait();

    if constexpr (debug) {
        std::println("[Parallel FIR + IIR approx] Results from: {}", dev.get_info<sycl::info::device::name>());
        for (std::size_t i = 0; i < std::min<std::size_t>(20, output.size()); ++i) {
            std::println("  y[{:2}] = {:.3f}", i, output[i]);
        }
    }

    std::println("[Parallel FIR/IIR] results from '{}' (device-only RAM): {} MS/s", dev.get_info<sycl::info::device::name>(), static_cast<float>(N) / static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

    sycl::free(in_dev, q);
    sycl::free(out_dev, q);
    sycl::free(fir_taps, q);
}

template<gr::ExecutionTarget target>
void run_filter_device_only(const sycl::device& dev, const std::vector<float>& input) {
    std::size_t N = input.size();
    sycl::queue q{dev};

    float* d_input  = sycl::malloc_device<float>(N, q);
    float* d_output = sycl::malloc_device<float>(N, q);
    q.memcpy(d_input, input.data(), sizeof(float) * input.size()).wait();

    gr::IirFilter<float, target>*  iir = sycl::malloc_device<gr::IirFilter<float, target>>(1, q);
    gr::fir_filter<float, target>* fir = sycl::malloc_device<gr::fir_filter<float, target>>(1, q);

    gr::IirFilter<float, target>  iir_host;
    gr::fir_filter<float, target> fir_host;
    q.memcpy(iir, &iir_host, sizeof(iir_host)).wait();
    q.memcpy(fir, &fir_host, sizeof(fir_host)).wait();

    const auto t0 = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler& h) {
         h.single_task([=]() {
             for (std::size_t i = 0; i < N; ++i) {
                 float y     = iir->processOne(fir->processOne(d_input[i]));
                 d_output[i] = y;
             }
         });
     }).wait();
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<float> output(N);
    q.memcpy(output.data(), d_output, N * sizeof(float)).wait();

    std::println("[Device-only] Results from '{}' (device-only RAM): {} MS/s", dev.get_info<sycl::info::device::name>(), static_cast<float>(N) / static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(iir, q);
    sycl::free(fir, q);
}

int main() {
    constexpr std::size_t N = 32'000'000UZ;
    std::vector<float>    input(N);
    std::iota(input.begin(), input.end(), 0.0f);

    // List available devices
    const auto all_devices = sycl::device::get_devices();
    std::println("=== SYCL devices detected ===");
    for (const auto& dev : all_devices) {
        std::println("Device: {}", dev.get_info<sycl::info::device::name>());
        std::println("  Max compute units     : {}", dev.get_info<sycl::info::device::max_compute_units>());
        std::println("  Max work-group size   : {}", dev.get_info<sycl::info::device::max_work_group_size>());
        // std::println("  Max work-items[0]     : {}", dev.template get_info<sycl::info::device::max_work_item_sizes>()[0]);
        std::println("  Preferred vector width: {}", dev.get_info<sycl::info::device::preferred_vector_width_float>());

        std::size_t wg_size = dev.get_info<sycl::info::device::max_work_group_size>();
        std::size_t cu      = dev.get_info<sycl::info::device::max_compute_units>();

        std::println("Theoretical occupancy:");
        std::println("  Work-groups          : {}", (N + wg_size - 1) / wg_size);
        std::println("  Compute units        : {}", cu);
        std::println("  Threads per CU (est.): {}", (N + cu - 1) / cu);
    }
    std::println("=============================\n");

    // Run on first CPU (likely OpenMP)

    for (const auto& dev : all_devices) {
        if (dev.is_cpu()) {
            std::println(">>> Running on CPU: {}", dev.get_info<sycl::info::device::name>());
            run_filter<gr::ExecutionTarget::CPU>(dev, input);
            run_filter_device_only<gr::ExecutionTarget::CPU>(dev, input);
            run_filter_parallel_fir<gr::ExecutionTarget::CPU>(dev, input);
        }

        if (dev.is_gpu()) {
            std::println(">>> Running on GPU: {}", dev.get_info<sycl::info::device::name>());
            run_filter<gr::ExecutionTarget::GPU>(dev, input);
            run_filter_device_only<gr::ExecutionTarget::GPU>(dev, input);
            run_filter_parallel_fir<gr::ExecutionTarget::GPU>(dev, input);
        }
    }

    return 0;
}
