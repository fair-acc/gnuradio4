#ifndef GNURADIO_NULLSOURCES_HPP
#define GNURADIO_NULLSOURCES_HPP

#include <gnuradio-4.0/AtomicRef.hpp>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::testing {

GR_REGISTER_BLOCK(gr::testing::NullSource, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
struct NullSource : Block<NullSource<T>> {
    using Description = Doc<R""(A source block that emits zeros or type-specified default value continuously.
Used mainly for testing and benchmarking where a consistent and predictable output is essential.
Ideal for scenarios that require a simple, low-overhead source of consistent values.)"">;

    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(NullSource, out);

    [[nodiscard]] constexpr T processOne() const noexcept { return T{}; }

    [[nodiscard]] constexpr auto processOne(meta::constexpr_value auto N) const noexcept
    requires std::is_arithmetic_v<T>
    {
        return meta::simdize<T, N>{};
    }
};

static_assert(gr::BlockLike<NullSource<float>>);

GR_REGISTER_BLOCK(gr::testing::ConstantSource, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
struct ConstantSource : Block<ConstantSource<T>> {
    using value_t     = std::conditional_t<std::is_same_v<T, std::string>, std::string, meta::fundamental_base_value_type_t<T>>; // use base-type for types not supported by settings
    using Description = Doc<R""(A source block that emits a constant default value for each output sample.
This block counts the number of samples emitted and optionally halts after reaching a specified maximum.
With `clk` connected it emits one sample per clock sample, otherwise as many as the output buffer takes.
Commonly used for testing and simulations where consistent output and finite execution are required.)"">;

    gr::PortIn<std::uint8_t, Optional> clk;
    gr::PortOut<T>                     out;

    Annotated<value_t, "default value", Doc<"default value for each sample">>                     default_value{};
    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    GR_MAKE_REFLECTABLE(ConstantSource, clk, out, default_value, n_samples_max, count);

    void reset() { count = 0U; }

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t n = samplesToEmit(input.size(), output.size());
        std::ranges::fill_n(output.data(), static_cast<std::ptrdiff_t>(n), T(default_value.value));
        return publishSamples(input, output, n);
    }

    [[nodiscard]] work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output)
    requires(std::is_arithmetic_v<T> || meta::complex_like<T>)
    {
        const auto failed = [&] {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return work::Status::ERROR;
        };
        const std::size_t n = samplesToEmit(input.size(), output.size());
        if (n == 0UZ) {
            return publishSamples(input, output, 0UZ);
        }
        const T                  value(default_value.value);
        const bool               stage   = !ctx.isDeviceAccessible(output.data());
        gr::device::DeviceBuffer storage = stage ? ctx.allocateShared<T>(n) : gr::device::DeviceBuffer{};
        const gr::on_scope_exit  release{[&] {
            if (!ctx.peekDeviceError()) { // a poisoned context is never touched again (AsyncErrorState)
                ctx.deallocate(storage);
            }
        }};
        T* const                 samples = stage ? storage.template devicePointer<T>() : output.data();
        if (samples == nullptr) [[unlikely]] {
            if (ctx.peekDeviceError() || !this->deviceFallbackIsAllowed("could not allocate staging samples")) {
                return failed();
            }
            std::ranges::fill_n(output.data(), static_cast<std::ptrdiff_t>(n), value);
            return publishSamples(input, output, n);
        }
        gr::device::parallelFor(ctx, n, [samples, value] GR_DEVICE_LAMBDA(std::size_t i) { samples[i] = value; });
        if (stage && !ctx.pollDeviceError()) {
            ctx.copy(output.data(), samples, n * sizeof(T));
        }
        if (ctx.pollDeviceError()) [[unlikely]] {
            return failed();
        }
        return publishSamples(input, output, n);
    }

private:
    [[nodiscard]] std::size_t samplesToEmit(std::size_t nClock, std::size_t nOutput) const noexcept {
        const std::size_t available = clk.isConnected() ? std::min(nClock, nOutput) : nOutput;
        const std::size_t remaining = n_samples_max == 0U ? available : static_cast<std::size_t>(n_samples_max.value - std::min(count.value, n_samples_max.value));
        return std::min(available, remaining);
    }

    [[nodiscard]] work::Status publishSamples(InputSpanLike auto& input, OutputSpanLike auto& output, std::size_t n) {
        count += static_cast<gr::Size_t>(n);
        std::ignore = input.consume(clk.isConnected() ? n : 0UZ);
        output.publish(n);
        if (n_samples_max > 0U && count >= n_samples_max) {
            return work::Status::DONE;
        }
        return n == 0UZ ? work::Status::INSUFFICIENT_INPUT_ITEMS : work::Status::OK;
    }
};

static_assert(gr::BlockLike<ConstantSource<float>>);

GR_REGISTER_BLOCK(gr::testing::SlowSource, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
struct SlowSource : Block<SlowSource<T>> {
    using value_t     = std::conditional_t<std::is_same_v<T, std::string>, std::string, meta::fundamental_base_value_type_t<T>>; // use base-type for types not supported by settings
    using Description = Doc<R""(A source block that emits a constant default value every n miliseconds)"">;

    gr::PortOut<T>                                                                              out;
    Annotated<value_t, "default value", Doc<"default value for each sample">>                   default_value{};
    Annotated<gr::Size_t, "delay", Unit<"ms">, Doc<"how many milliseconds between each value">> delay = 100U;

    GR_MAKE_REFLECTABLE(SlowSource, out, default_value, delay);

    std::optional<std::chrono::time_point<std::chrono::system_clock>> lastEventAt;

    [[nodiscard]] gr::work::Status processBulk(OutputSpanLike auto& output) {
        if (!lastEventAt || std::chrono::system_clock::now() - *lastEventAt > std::chrono::milliseconds(delay)) {
            lastEventAt = std::chrono::system_clock::now();

            output[0] = T(default_value.value);
            output.publish(1UZ);
        } else {
            output.publish(0UZ);
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::testing::CountingSource, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct CountingSource : Block<CountingSource<T>> {
    using Description = Doc<R""(A source block that emits an increasing sequence starting from a specified default value.
This block counts the number of samples emitted and optionally halts after reaching a specified maximum.
With `clk` connected it emits one sample per clock sample, otherwise as many as the output buffer takes.
Commonly used for testing and simulations where consistent output and finite execution are required.)"">;

    gr::PortOut<T> out;

    Annotated<T, "default value", Doc<"default value for each sample">>                           default_value{};
    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    GR_MAKE_REFLECTABLE(CountingSource, out, default_value, n_samples_max, count);

    void reset() { count = 0U; }

    [[nodiscard]] constexpr T processOne() noexcept {
        count++;
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
        return static_cast<T>(default_value.value) + static_cast<T>(count);
#pragma GCC diagnostic pop
    }

    [[nodiscard]] constexpr auto processOne(meta::constexpr_value auto N) noexcept
    requires std::is_arithmetic_v<T>
    {
        meta::simdize<T, N> result;
        const auto          base = static_cast<T>(default_value.value);
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = base + static_cast<T>(count + 1 + i);
        }
        count += static_cast<gr::Size_t>(result.size());
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
        return result;
    }
};

GR_REGISTER_BLOCK(gr::testing::Copy, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
struct Copy : Block<Copy<T>> {
    using Description = Doc<R""(A block that passes/copies input samples directly to its output without modification.
Commonly used used to isolate parts of a flowgraph, manage buffer sizes, or simply duplicate the signal path.)"">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(Copy, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V input) const noexcept {
        return input;
    }
};

GR_REGISTER_BLOCK(gr::testing::HeadBlock, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
struct HeadBlock : Block<HeadBlock<T>> {
    using Description = Doc<R""(Limits the number of output samples by copying the first N items from the input to the output and then signaling completion.
Commonly used to control data flow in systems where precise sample counts are critical, such as in file recording or when executing flow graphs without a GUI.)"">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    GR_MAKE_REFLECTABLE(HeadBlock, in, out, n_samples_max, count);

    void reset() { count = 0U; }

    [[nodiscard]] constexpr auto processOne(T input) noexcept {
        count++;
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
        return input;
    }
};

GR_REGISTER_BLOCK(gr::testing::NullSink, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>, gr::DataSet<uint16_t> ])

template<typename T>
struct NullSink : Block<NullSink<T>> {
    using Description = Doc<R""(A sink block that consumes and discards all input samples without producing any output.
Used primarily for absorbing data in a flow graph where output processing is unnecessary.
Commonly used for testing, performance benchmarking, and in scenarios where signal flow needs to be terminated without external effects.)"">;

    gr::PortIn<T> in;
    GR_MAKE_REFLECTABLE(NullSink, in);

    template<gr::meta::t_or_simd<T> V>
    void processOne(V) const noexcept { /* do nothing */ }
};

template<typename T, bool AtomicCount = false>
struct CountingSinkImpl : Block<CountingSinkImpl<T, AtomicCount>> {
    using Description = Doc<R""(A sink block that consumes and discards a fixed number of input samples, after which it signals the flow graph to halt.
This block is used to control execution in systems requiring precise input processing without data output, similar to how a 'HeadBlock' manages output samples.
Commonly used for testing scenarios and signal termination where output is unnecessary but precise input count control is needed.)"">;

    gr::PortIn<T> in;

    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    GR_MAKE_REFLECTABLE(CountingSinkImpl, in, n_samples_max, count);

    void reset() {
        if constexpr (AtomicCount) {
            gr::atomic_ref(count.value).store_release(0U);
        } else {
            count = 0U;
        }
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input) noexcept {
        const auto       nAvailable = static_cast<gr::Size_t>(input.size());
        const gr::Size_t nCounted   = loadCount();
        if (n_samples_max > 0 && nCounted + nAvailable >= n_samples_max) {
            std::ignore = input.consume(static_cast<std::size_t>(n_samples_max - nCounted));
            storeCount(n_samples_max.value);
            return gr::work::Status::DONE;
        }
        storeCount(nCounted + nAvailable);
        return gr::work::Status::OK;
    }

    void storeCount(gr::Size_t newValue) noexcept {
        if constexpr (AtomicCount) {
            gr::atomic_ref(count.value).store_release(newValue);
        } else {
            count = newValue;
        }
    }

    [[nodiscard]] gr::Size_t loadCount() const noexcept {
        if constexpr (AtomicCount) {
            return gr::atomic_ref(const_cast<gr::Size_t&>(count.value)).load_acquire();
        } else {
            return count.value;
        }
    }
};

GR_REGISTER_BLOCK(gr::testing::CountingSink, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
using CountingSink = CountingSinkImpl<T, false>;

GR_REGISTER_BLOCK(gr::testing::AtomicCountingSink, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
using AtomicCountingSink = CountingSinkImpl<T, true>;

GR_REGISTER_BLOCK(gr::testing::SimCompute, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> ])

template<typename T>
struct SimCompute : Block<SimCompute<T>> {
    using Description = Doc<R""(Block to simulate compute delay and model latency effects for a given complexity order

Delay depends on work size: delay_seconds = (N / reference_work_size)^complexity_order * (reference_work_size / target_throughput)

Default: linear O(N) delay that limits the block processing to 100 MS/s)"">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    Annotated<float, "target_throughput", Unit<"S/s">, Doc<"Target throughput in samples per second at reference_work_size.">>               target_throughput   = 100e6f;
    Annotated<gr::Size_t, "reference_work_size", Unit<"[N]">, Doc<"Reference work size N_r (samples) for which target_throughput applies.">> reference_work_size = 1024U;
    Annotated<float, "complexity_order", Doc<"compute complexity: 1.0 = linear, 2.0 = quadratic, 3.0 = cubic.">>                             complexity_order    = 1.0f;
    Annotated<bool, "busy_wait", Doc<"true: busy loop (CPU burn), False: sleep loop (lower power, less accurate).">>                         busy_wait           = true;

    GR_MAKE_REFLECTABLE(SimCompute, in, out, target_throughput, reference_work_size, complexity_order, busy_wait);

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        using Clock_t    = std::chrono::high_resolution_clock;
        const auto start = Clock_t::now();

        std::ranges::copy(input, output.begin());

        const std::chrono::duration<double> delay_seconds = compute_delay_seconds(input.size());
        const std::chrono::nanoseconds      delay_ns      = std::chrono::duration_cast<std::chrono::nanoseconds>(delay_seconds);
        while (Clock_t::now() - start < delay_ns) {
            if (!busy_wait) {
                std::this_thread::sleep_for(std::chrono::microseconds(10)); // avoid hard spin
            }
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] constexpr std::chrono::duration<double> compute_delay_seconds(std::size_t N) const noexcept {
        const auto N_r        = static_cast<double>(reference_work_size);
        const auto throughput = static_cast<double>(target_throughput);
        const auto k          = static_cast<double>(complexity_order);

        const double delay_s = std::pow(static_cast<double>(N) / N_r, k) * (N_r / throughput);

        return std::chrono::duration<double>(delay_s);
    }
};

} // namespace gr::testing

#endif // GNURADIO_NULLSOURCES_HPP
