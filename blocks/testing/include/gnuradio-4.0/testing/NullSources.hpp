#ifndef GNURADIO_NULLSOURCES_HPP
#define GNURADIO_NULLSOURCES_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/reflection.hpp>

namespace gr::testing {

template<typename T>
struct NullSource : public gr::Block<NullSource<T>> {
    using Description = Doc<R""(A source block that emits zeros or type-specified default value continuously.
Used mainly for testing and benchmarking where a consistent and predictable output is essential.
Ideal for scenarios that require a simple, low-overhead source of consistent values.)"">;

    gr::PortOut<T> out;

    [[nodiscard]] constexpr T
    processOne() const noexcept {
        return T{};
    }
};

static_assert(gr::BlockLike<NullSource<float>>);

template<typename T>
struct ConstantSource : public gr::Block<ConstantSource<T>> {
    using Description = Doc<R""(A source block that emits a constant default value for each output sample.
This block counts the number of samples emitted and optionally halts after reaching a specified maximum.
Commonly used for testing and simulations where consistent output and finite execution are required.)"">;

    gr::PortOut<T> out;

    Annotated<T, "default value", Visible, Doc<"default value for each sample">>                  default_value{};
    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    void
    reset() {
        count = 0U;
    }

    [[nodiscard]] constexpr T
    processOne() noexcept {
        count++;
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
        return default_value;
    }
};

static_assert(gr::BlockLike<ConstantSource<float>>);

template<typename T>
    requires(std::is_arithmetic_v<T>)
struct CountingSource : public gr::Block<CountingSource<T>> {
    using Description = Doc<R""(A source block that emits an increasing sequence starting from a specified default value.
This block counts the number of samples emitted and optionally halts after reaching a specified maximum.
Commonly used for testing and simulations where consistent output and finite execution are required.)"">;

    gr::PortOut<T> out;

    Annotated<T, "default value", Visible, Doc<"default value for each sample">>                  default_value{};
    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    void
    reset() {
        count = 0U;
    }

    [[nodiscard]] constexpr T
    processOne() noexcept {
        count++;
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
        return default_value + T(count);
    }
};

static_assert(gr::BlockLike<CountingSource<float>>);

template<typename T>
struct Copy : public gr::Block<Copy<T>> {
    using Description = Doc<R""(A block that passes/copies input samples directly to its output without modification.
Commonly used used to isolate parts of a flowgraph, manage buffer sizes, or simply duplicate the signal path.)"">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V input) const noexcept {
        return input;
    }
};

static_assert(gr::BlockLike<Copy<float>>);

template<typename T>
struct HeadBlock : public gr::Block<HeadBlock<T>> { // TODO confirm naming: while known in GR3, the semantic name seems to be odd. (Maybe add an alias?!)
    using Description = Doc<R""(Limits the number of output samples by copying the first N items from the input to the output and then signaling completion.
Commonly used to control data flow in systems where precise sample counts are critical, such as in file recording or when executing flow graphs without a GUI.)"">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    void
    reset() {
        count = 0U;
    }

    [[nodiscard]] constexpr auto
    processOne(T input) noexcept {
        count++;
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
        return input;
    }
};

static_assert(gr::BlockLike<HeadBlock<float>>);

template<typename T>
struct NullSink : public gr::Block<NullSink<T>> {
    using Description = Doc<R""(A sink block that consumes and discards all input samples without producing any output.
Used primarily for absorbing data in a flow graph where output processing is unnecessary.
Commonly used for testing, performance benchmarking, and in scenarios where signal flow needs to be terminated without external effects.)"">;

    gr::PortIn<T> in;

    template<gr::meta::t_or_simd<T> V>
    void
    processOne(V) const noexcept {}
};

static_assert(gr::BlockLike<NullSink<float>>);

template<typename T>
struct CountingSink : public gr::Block<CountingSink<T>> {
    using Description = Doc<R""(A sink block that consumes and discards a fixed number of input samples, after which it signals the flow graph to halt.
This block is used to control execution in systems requiring precise input processing without data output, similar to how a 'HeadBlock' manages output samples.
Commonly used for testing scenarios and signal termination where output is unnecessary but precise input count control is needed.)"">;

    gr::PortIn<T> in;

    Annotated<gr::Size_t, "max samples", Doc<"count>n_samples_max -> signal DONE (0: infinite)">> n_samples_max = 0U;
    Annotated<gr::Size_t, "count", Doc<"sample count (diagnostics only)">>                        count         = 0U;

    void
    reset() {
        count = 0U;
    }

    template<gr::meta::t_or_simd<T> V>
    void
    processOne(V) noexcept {
        if constexpr (stdx::is_simd_v<V>) {
            count += V::size();
        } else {
            count++;
        }
        if (n_samples_max > 0 && count >= n_samples_max) {
            this->requestStop();
        }
    }
};

static_assert(gr::BlockLike<CountingSink<float>>);

} // namespace gr::testing

ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::NullSource, out);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::ConstantSource, out, n_samples_max, count);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::CountingSource, out, n_samples_max, count);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::Copy, in, out);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::HeadBlock, in, out, n_samples_max, count);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::NullSink, in);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::CountingSink, in, n_samples_max, count);

const inline auto registerNullSources
        = gr::registerBlock<gr::testing::NullSource, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string,
                            gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry())
        | gr::registerBlock<gr::testing::ConstantSource, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>,
                            std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry())
        | gr::registerBlock<gr::testing::CountingSource, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>(gr::globalBlockRegistry())
        | gr::registerBlock<gr::testing::Copy, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string,
                            gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry())
        | gr::registerBlock<gr::testing::HeadBlock, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string,
                            gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry())
        | gr::registerBlock<gr::testing::NullSink, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string,
                            gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry())
        | gr::registerBlock<gr::testing::CountingSink, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string,
                            gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double>>(gr::globalBlockRegistry());

#endif // GNURADIO_NULLSOURCES_HPP
