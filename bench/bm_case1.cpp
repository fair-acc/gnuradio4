#include "benchmark.hpp"

#include <algorithm>
#include <boost/ut.hpp>
#include <functional>

#include "bm_test_helper.hpp"

#include <graph.hpp>
#include <node_traits.hpp>
#include <scheduler.hpp>

#include <vir/simd.h>

namespace fg                        = fair::graph;

inline constexpr std::size_t N_ITER = 10;
// inline constexpr std::size_t N_SAMPLES = gr::util::round_up(1'000'000, 1024);
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(10'000, 1024);

template<typename T, char op>
struct math_op : public fg::node<math_op<T, op>, fg::IN<T, 0, N_MAX, "in">, fg::OUT<T, 0, N_MAX, "out">> {
    T factor = static_cast<T>(1.0f);

    // public:
    //     math_op() = delete;
    //
    //     explicit math_op(T factor, std::string name = fair::graph::this_source_location()) : factor(factor) { this->set_name(name); }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(const V &a) const noexcept {
        if constexpr (op == '*') {
            return a * factor;
        } else if constexpr (op == '/') {
            return a / factor;
        } else if constexpr (op == '+') {
            return a + factor;
        } else if constexpr (op == '-') {
            return a - factor;
        } else {
            static_assert(fair::meta::always_false<T>, "unknown op");
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, char op), (math_op<T, op>), factor);

template<typename T>
using multiply = math_op<T, '*'>;
template<typename T>
using divide = math_op<T, '/'>;
// template<typename T> using add = math_op<T, '+'>;
// template<typename T> using sub = math_op<T, '-'>;

#if !DISABLE_SIMD
static_assert(fg::traits::node::can_process_one_simd<multiply<float>>);
#endif

template<typename T, char op>
class math_bulk_op : public fg::node<math_bulk_op<T, op>, fg::IN<T, 0, N_MAX, "in">, fg::OUT<T, 0, N_MAX, "out">> {
    T _factor = static_cast<T>(1.0f);

public:
    math_bulk_op() = delete;

    explicit math_bulk_op(T factor, std::string name_ = fair::graph::this_source_location()) : _factor(factor) { this->name = name_; }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(const V &a) const noexcept {
        if constexpr (op == '*') {
            return a * _factor;
        } else if constexpr (op == '/') {
            return a / _factor;
        } else if constexpr (op == '+') {
            return a + _factor;
        } else if constexpr (op == '-') {
            return a - _factor;
        } else {
            static_assert(fair::meta::always_false<T>, "unknown op");
        }
    }
};

template<typename T>
using multiply_bulk = math_bulk_op<T, '*'>;
template<typename T>
using divide_bulk = math_bulk_op<T, '/'>;
template<typename T>
using add_bulk = math_bulk_op<T, '+'>;
template<typename T>
using sub_bulk = math_bulk_op<T, '-'>;

//
// This defines a new node type that has only type template parameters.
//
// It defines its ports as member variables, so it needs to be
// enabled for reflection. Since it has no non-type template parameters,
// this can be achieved using the ENABLE_REFLECTION_FOR_TEMPLATE
// macro as can be seen immediately after the class is defined.
//
template<typename T, typename R = T>
class converting_multiply : public fg::node<converting_multiply<T, R>> {
    T _factor = static_cast<T>(1.0f);

public:
    fg::IN<T>  in;
    fg::OUT<R> out;

    converting_multiply() = delete;

    explicit converting_multiply(T factor, std::string name_ = fair::graph::this_source_location()) : _factor(factor) { this->name = name_; }

    [[nodiscard]] constexpr auto
    process_one(T a) const noexcept {
        return static_cast<R>(a * _factor);
    }

    [[nodiscard]] constexpr auto
    process_one(const fair::meta::any_simd<T> auto &a) const noexcept {
        return vir::stdx::static_simd_cast<R>(a * _factor);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(converting_multiply, in, out);
#if !DISABLE_SIMD
static_assert(fg::traits::node::can_process_one_simd<converting_multiply<float, double>>);
#endif

//
// This defines a new node type that is parametrised on several
// template parameters, some of which are non-type template parameters.
//
// It defines its ports as member variables. It needs to be
// enabled for reflection using the ENABLE_REFLECTION_FOR_TEMPLATE_FULL
// macro because it contains non-type template parameters.
//
template<typename T, int addend>
class add : public fg::node<add<T, addend>> {
public:
    fg::IN<T>  in;
    fg::OUT<T> out;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    process_one(const V &a) const noexcept {
        return a + addend;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, int addend), (add<T, addend>), in, out);
#if !DISABLE_SIMD
static_assert(fg::traits::node::can_process_one_simd<add<float, 1>>);
#endif

//
// This defines a new node type that which doesn't define ports
// as member variables, but as template parameters to the fg::node
// base class template.
//
// It doesn't need to be enabled for reflection.
//
template<typename T, char op>
class gen_operation_SIMD : public fg::node<gen_operation_SIMD<T, op>, fg::IN<T, 0, N_MAX, "in">, fg::OUT<T, 0, N_MAX, "out">> {
    T _value = static_cast<T>(1.0f);

public:
    gen_operation_SIMD() = delete;

    explicit gen_operation_SIMD(T value, std::string name_ = fair::graph::this_source_location()) : _value(value) { this->name = name_; }

    fair::graph::work_return_t
    work(std::size_t requested_work) noexcept {
        auto      &out_port   = output_port<0>(this);
        auto      &in_port    = input_port<0>(this);

        auto      &reader     = in_port.streamReader();
        auto      &writer     = out_port.streamWriter();
        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
        if (n_readable == 0) {
            return { requested_work, 0UL, fair::graph::work_return_status_t::INSUFFICIENT_INPUT_ITEMS };
        } else if (n_writable == 0) {
            return { requested_work, 0UL, fair::graph::work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
        }
        const std::size_t n_to_publish = std::min(n_readable, n_writable);

        writer.publish( //
                [&reader, n_to_publish, this](std::span<T> output) {
                    const auto input = reader.get();
                    // #### N.B. later high-level user-function starts here

                    using namespace vir::stdx;
                    using V           = native_simd<T>;
                    std::size_t i     = 0;
                    const auto  value = _value;
                    for (; i + V::size() <= n_to_publish; i += V::size()) {
                        V in(&input[i], element_aligned);
                        if constexpr (op == '*') {
                            in *= value;
                        } else if constexpr (op == '/') {
                            in /= value;
                        } else if constexpr (op == '+') {
                            in += value;
                        } else if constexpr (op == '-') {
                            in -= value;
                        } else {
                            static_assert(fair::meta::always_false<T>, "operation not implemented");
                        }
                        in.copy_to(&output[i], element_aligned);
                    }

                    // #### N.B. later high-level user-function finishes here

                    // epilogue handling the samples not fitting into a SIMD vector
                    for (; i < n_to_publish; i++) {
                        if constexpr (op == '*') {
                            output[i] = input[i] * value;
                        } else if constexpr (op == '/') {
                            output[i] = input[i] / value;
                        } else if constexpr (op == '+') {
                            output[i] = input[i] + value;
                        } else if constexpr (op == '-') {
                            output[i] = input[i] - value;
                        } else {
                            static_assert(fair::meta::always_false<T>, "operation not implemented");
                        }
                    }
                },
                n_to_publish);

        if (!reader.consume(n_to_publish)) {
            return { requested_work, n_to_publish, fair::graph::work_return_status_t::ERROR };
        }
        return { requested_work, n_to_publish, fair::graph::work_return_status_t::OK };
    }
};

// no reflection needed because ports are defined via the fg::node<...> base class

// gen_operation_SIMD has built-in SIMD-enabled work function, that means
// we don't see it as a SIMD-enabled node as we can not do simd<simd<something>>
static_assert(not fg::traits::node::can_process_one_simd<gen_operation_SIMD<float, '*'>>);

template<typename T>
using multiply_SIMD = gen_operation_SIMD<T, '*'>;

template<typename T>
using add_SIMD = gen_operation_SIMD<T, '+'>;

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX, bool use_bulk_operation = false, bool use_memcopy = true>
class copy : public fg::node<copy<T, N_MIN, N_MAX, use_bulk_operation, use_memcopy>> {
public:
    fg::IN<T, N_MIN, N_MAX>  in;
    fg::OUT<T, N_MIN, N_MAX> out;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    process_one(const V &a) const noexcept {
        return a;
    }

    fair::graph::work_return_t
    work(std::size_t requested_work) noexcept { // TODO - make this an alternate version to 'process_one'
        auto      &out_port   = out;
        auto      &in_port    = in;

        auto      &reader     = in_port.streamReader();
        auto      &writer     = out_port.streamWriter();
        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
        if (n_readable == 0) {
            return { requested_work, 0UL, fair::graph::work_return_status_t::DONE };
        } else if (n_writable == 0) {
            return { requested_work, 0UL, fair::graph::work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
        }
        const std::size_t n_to_publish = std::min(n_readable, n_writable);

        if constexpr (use_memcopy) {
            // fmt::print("n_to_publish {} - {} {}\n", n_to_publish, use_bulk_operation, use_memcopy);
            writer.publish( //
                    [&reader, n_to_publish](std::span<T> output) { std::memcpy(output.data(), reader.get().data(), n_to_publish * sizeof(T)); }, n_to_publish);
        } else {
            writer.publish( //
                    [&reader, n_to_publish](std::span<T> output) {
                        const auto input = reader.get();
                        for (std::size_t i = 0; i < n_to_publish; i++) {
                            output[i] = input[i];
                        }
                    },
                    n_to_publish);
        }
        if (!reader.consume(n_to_publish)) {
            return { requested_work, 0UL, fair::graph::work_return_status_t::ERROR };
        }
        return { requested_work, 0UL, fair::graph::work_return_status_t::OK };
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t N_MIN, std::size_t N_MAX, bool use_bulk_operation, bool use_memcopy), (copy<T, N_MIN, N_MAX, use_bulk_operation, use_memcopy>), in, out);

namespace detail {
template<typename T>
constexpr std::size_t
simd_size() noexcept {
    namespace stdx = vir::stdx;
    if constexpr (stdx::is_simd_v<T>) {
        return T::size();
    } else {
        return 1LU;
    }
}
} // namespace detail

namespace stdx = vir::stdx;

template<typename From, typename To, std::size_t N_MIN = 0 /* SIMD size */, std::size_t N_MAX = N_MAX>
class convert : public fg::node<convert<From, To, N_MIN, N_MAX>, fg::IN<From, N_MIN, N_MAX, "in">, fg::OUT<To, N_MIN, N_MAX, "out">> {
    static_assert(stdx::is_simd_v<From> != stdx::is_simd_v<To>, "either input xor output must be SIMD capable");
    constexpr static std::size_t from_simd_size = detail::simd_size<From>();
    constexpr static std::size_t to_simd_size   = detail::simd_size<To>();
    constexpr static std::size_t simd_size      = std::max(from_simd_size, to_simd_size);

public:
    fair::graph::work_return_status_t
    work() noexcept {
        using namespace stdx;
        auto      &out_port   = output_port<"out">(this);
        auto      &in_port    = input_port<"in">(this);

        auto      &reader     = in_port.streamReader();
        auto      &writer     = out_port.streamWriter();
        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
        if (n_readable < to_simd_size) {
            return fair::graph::work_return_status_t::INSUFFICIENT_INPUT_ITEMS;
        } else if (n_writable < from_simd_size) {
            return fair::graph::work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS;
        }
        const auto n_readable_scalars = n_readable * from_simd_size;
        const auto n_writable_scalars = n_writable * to_simd_size;
        const auto n_simd_to_convert  = std::min(n_readable_scalars, n_writable_scalars) / simd_size;
        const auto scalars_to_convert = n_simd_to_convert * simd_size;
        const auto objects_to_write   = stdx::is_simd_v<To> ? n_simd_to_convert : scalars_to_convert;
        const auto objects_to_read    = stdx::is_simd_v<From> ? n_simd_to_convert : scalars_to_convert;

        auto       return_value       = fair::graph::work_return_status_t::OK;
        writer.publish( //
                [&](std::span<To> output) {
                    const auto input = reader.get();
                    if constexpr (stdx::is_simd_v<To>) {
                        // convert from T to SIMD<T>
                        for (std::size_t i = 0; i < n_simd_to_convert; ++i) {
                            output[i] = To(&input[i * simd_size], element_aligned);
                        }
                    } else {
                        // convert from SIMD<T> to T
                        for (std::size_t i = 0; i < n_simd_to_convert; ++i) {
                            input[i].copy_to(&output[i * simd_size], element_aligned);
                        }
                    }
                    if (!reader.consume(objects_to_read)) {
                        return_value = fair::graph::work_return_status_t::ERROR;
                        return;
                    }
                },
                objects_to_write);
        return return_value;
    }
};

void
loop_over_process_one(auto &node) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
#if DISABLE_SIMD
    for (std::size_t i = 0; i < N_SAMPLES; i++) {
        node.process_one(i);
    }
#else
    constexpr int N = 32;
    for (std::size_t i = 0; i < N_SAMPLES / N; i++) {
        node.template process_one_simd(i, std::integral_constant<std::size_t, N>{});
    }
#endif
    expect(eq(test::n_samples_produced, N_SAMPLES)) << "produced too many/few samples";
    expect(eq(test::n_samples_consumed, N_SAMPLES)) << "consumed too many/few samples";
}

void
loop_over_work(auto &node) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
    while (test::n_samples_consumed < N_SAMPLES) {
        std::ignore = node.work(std::numeric_limits<std::size_t>::max());
    }
    expect(eq(test::n_samples_produced, N_SAMPLES)) << "produced too many/few samples";
    expect(eq(test::n_samples_consumed, N_SAMPLES)) << "consumed too many/few samples";
}

inline const boost::ut::suite _constexpr_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;
    using fair::graph::merge_by_index;
    using fair::graph::merge;

    {
        auto merged_node                                            = merge<"out", "in">(test::source<float>(N_SAMPLES), test::sink<float>());
        "merged src->sink work"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_work(merged_node); };
    }

    {
        auto merged_node = merge<"out", "in">(merge<"out", "in">(test::source<float>(N_SAMPLES), copy<float>()), test::sink<float>());
#if !DISABLE_SIMD
        static_assert(fair::graph::traits::node::can_process_one_simd<copy<float>>);
        static_assert(fair::graph::traits::node::can_process_one_simd<test::sink<float>>);
#endif
        "merged src->copy->sink"_benchmark.repeat<N_ITER>(N_SAMPLES)      = [&merged_node]() { loop_over_process_one(merged_node); };
        "merged src->copy->sink work"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_work(merged_node); };
    }

    {
        auto merged_node = merge<"out", "in">(merge<"out", "in">(test::source<float>(N_SAMPLES), test::cascade<10, copy<float>>(copy<float>())), test::sink<float>());
        "merged src->copy^10->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_process_one(merged_node); };
    }

    {
        auto merged_node = merge<"out", "in">(merge<"out", "in">(merge<"out", "in">(merge<"out", "in">(test::source<float, 1024, 1024>(N_SAMPLES), copy<float, 0, 128>()), copy<float, 0, 1024>()),
                                                                 copy<float, 32, 128>()),
                                              test::sink<float>());
        "merged src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_process_one(merged_node); };
    }

    constexpr auto templated_cascaded_test = []<typename T>(T factor, const char *test_name) {
        auto gen_mult_block = [&factor] { return merge<"out", "in">(multiply<T>({ { { "factor", factor } } }), merge<"out", "in">(divide<T>({ { { "factor", factor } } }), add<T, -1>())); };
        auto merged_node    = merge<"out", "in">(merge<"out", "in">(test::source<T>(N_SAMPLES), gen_mult_block()), test::sink<T>());
        ::benchmark::benchmark<1LU>{ test_name }.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_process_one(merged_node); };
    };
    templated_cascaded_test(static_cast<float>(2.0), "merged src->mult(2.0)->divide(2.0)->add(-1)->sink - float");
    templated_cascaded_test(static_cast<int>(2.0), "merged src->mult(2.0)->divide(2.0)->add(-1)->sink - int");

    constexpr auto templated_cascaded_test_10 = []<typename T>(T factor, const char *test_name) {
        auto gen_mult_block = [&factor] { return merge<"out", "in">(multiply<T>({ { { "factor", factor } } }), merge<"out", "in">(divide<T>({ { { "factor", factor } } }), add<T, -1>())); };
        auto merged_node    = merge<"out", "in">(merge<"out", "in">(test::source<T>(N_SAMPLES), //
                                                                 test::cascade<10, decltype(gen_mult_block())>(gen_mult_block(), gen_mult_block)),
                                              test::sink<T>());
        ::benchmark::benchmark<1LU>{ test_name }.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_process_one(merged_node); };
    };
    templated_cascaded_test_10(static_cast<float>(2.0), "merged src->(mult(2.0)->div(2.0)->add(-1))^10->sink - float");
    templated_cascaded_test_10(static_cast<int>(2.0), "merged src->(mult(2.0)->div(2.0)->add(-1))^10->sink - int");
};

void
invoke_work(auto &sched) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
    sched.run_and_wait();
    expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
    expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
}

inline const boost::ut::suite _runtime_tests = [] {
    using namespace boost::ut;
    using namespace benchmark;

    {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto     &sink = flow_graph.make_node<test::sink<float>>();
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->sink overhead"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto     &sink = flow_graph.make_node<test::sink<float>>();
        auto     &cpy  = flow_graph.make_node<copy<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect(src, &test::source<float>::out).to<"in">(cpy)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(cpy).to(sink, &test::sink<float>::in)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->copy->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto     &sink = flow_graph.make_node<test::sink<float>>();

        using copy     = ::copy<float, 0, N_MAX, true, true>;
        std::vector<copy *> cpy(10);
        for (std::size_t i = 0; i < cpy.size(); i++) {
            cpy[i] = std::addressof(flow_graph.make_node<copy>({ { "name", fmt::format("copy {} at {}", i, fair::graph::this_source_location()) } }));

            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(*cpy[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect(*cpy[i - 1], &copy::out).to(*cpy[i], &copy::in)));
            }
        }

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*cpy[cpy.size() - 1]).to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->copy^10->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<float, 0, 1024>>(N_SAMPLES);
        auto     &b1   = flow_graph.make_node<copy<float, 0, 128>>();
        auto     &b2   = flow_graph.make_node<copy<float, 1024, 1024>>();
        auto     &b3   = flow_graph.make_node<copy<float, 32, 128>>();
        auto     &sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(b1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(b1).to<"in">(b2)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(b2).to<"in">(b3)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(b3).to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    constexpr auto templated_cascaded_test = []<typename T>(T factor, const char *test_name) {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<T>>(N_SAMPLES);
        auto     &mult = flow_graph.make_node<multiply<T>>({ { { "factor", factor } } });
        auto     &div  = flow_graph.make_node<divide<T>>({ { { "factor", factor } } });
        auto     &add1 = flow_graph.make_node<add<T, -1>>();
        auto     &sink = flow_graph.make_node<test::sink<T>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).template to<"in">(mult)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(mult).template to<"in">(div)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(div).template to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(add1).template to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        ::benchmark::benchmark<1LU>{ test_name }.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    };
    templated_cascaded_test(static_cast<float>(2.0), "runtime   src->mult(2.0)->div(2.0)->add(-1)->sink - float");
    templated_cascaded_test(static_cast<int>(2.0), "runtime   src->mult(2.0)->div(2.0)->add(-1)->sink - int");

    constexpr auto templated_cascaded_test_10 = []<typename T>(T factor, const char *test_name) {
        fg::graph                  flow_graph;
        auto                      &src  = flow_graph.make_node<test::source<T>>(N_SAMPLES);
        auto                      &sink = flow_graph.make_node<test::sink<T>>();

        std::vector<multiply<T> *> mult1;
        std::vector<divide<T> *>   div1;
        std::vector<add<T, -1> *>  add1;
        for (std::size_t i = 0; i < 10; i++) {
            mult1.emplace_back(std::addressof(flow_graph.make_node<multiply<T>>({ { "factor", factor }, { "name", fmt::format("mult1.{}", i) } })));
            div1.emplace_back(std::addressof(flow_graph.make_node<divide<T>>({ { "factor", factor }, { "name", fmt::format("div1.{}", i) } })));
            add1.emplace_back(std::addressof(flow_graph.make_node<add<T, -1>>()));
        }

        for (std::size_t i = 0; i < add1.size(); i++) {
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).template to<"in">(*mult1[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*add1[i - 1]).template to<"in">(*mult1[i])));
            }
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*mult1[i]).template to<"in">(*div1[i])));
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*div1[i]).template to<"in">(*add1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*add1[add1.size() - 1]).template to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        ::benchmark::benchmark<1LU>{ test_name }.repeat<N_ITER>(N_SAMPLES) = [&sched]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            sched.run_and_wait();
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    };
    templated_cascaded_test_10(static_cast<float>(2.0), "runtime   src->(mult(2.0)->div(2.0)->add(-1))^10->sink - float");
    templated_cascaded_test_10(static_cast<int>(2.0), "runtime   src->(mult(2.0)->div(2.0)->add(-1))^10->sink - int");
};

inline const boost::ut::suite _simd_tests = [] {
    using namespace boost::ut;
    using namespace benchmark;

    {
        fg::graph flow_graph;
        auto     &src   = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto     &mult1 = flow_graph.make_node<multiply_SIMD<float>>(2.0f);
        auto     &mult2 = flow_graph.make_node<multiply_SIMD<float>>(0.5f);
        auto     &add1  = flow_graph.make_node<add_SIMD<float>>(-1.0f);
        auto     &sink  = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(mult1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(mult1).to<"in">(mult2)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(mult2).to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(add1).to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->mult(2.0)->mult(0.5)->add(-1)->sink (SIMD)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            sched.run_and_wait();
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph                           flow_graph;
        auto                               &src  = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto                               &sink = flow_graph.make_node<test::sink<float>>();

        std::vector<multiply_SIMD<float> *> mult1;
        std::vector<multiply_SIMD<float> *> mult2;
        std::vector<add_SIMD<float> *>      add1;
        for (std::size_t i = 0; i < 10; i++) {
            mult1.emplace_back(std::addressof(flow_graph.make_node<multiply_SIMD<float>>(2.0f, fmt::format("mult1.{}", i))));
            mult2.emplace_back(std::addressof(flow_graph.make_node<multiply_SIMD<float>>(0.5f, fmt::format("mult2.{}", i))));
            add1.emplace_back(std::addressof(flow_graph.make_node<add_SIMD<float>>(-1.0f, fmt::format("add.{}", i))));
        }

        for (std::size_t i = 0; i < add1.size(); i++) {
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(*mult1[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*add1[i - 1]).to<"in">(*mult1[i])));
            }
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*mult1[i]).to<"in">(*mult2[i])));
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*mult2[i]).to<"in">(*add1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*add1[add1.size() - 1]).to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->(mult(2.0)->mult(0.5)->add(-1))^10->sink (SIMD)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            sched.run_and_wait();
            sched.reset();
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }
};

inline const boost::ut::suite _sample_by_sample_vs_bulk_access_tests = [] {
    using namespace boost::ut;
    using namespace benchmark;

    constexpr auto templated_cascaded_test = []<typename T>(T factor, const char *test_name) {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<T>>(N_SAMPLES);
        auto     &mult = flow_graph.make_node<multiply<T>>({ { { "factor", factor } } });
        auto     &div  = flow_graph.make_node<divide<T>>({ { { "factor", factor } } });
        auto     &add1 = flow_graph.make_node<add<T, -1>>();
        auto     &sink = flow_graph.make_node<test::sink<T>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).template to<"in">(mult)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(mult).template to<"in">(div)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(div).template to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(add1).template to<"in">(sink)));

        ::benchmark::benchmark<1LU>{ test_name }.repeat<N_ITER>(N_SAMPLES) = [&flow_graph]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            fg::scheduler::simple sched{ std::move(flow_graph) };
            sched.run_and_wait();
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    };
    templated_cascaded_test(static_cast<float>(2.0), "runtime   src->mult(2.0)->div(2.0)->add(-1)->sink - float single");
    templated_cascaded_test(static_cast<int>(2.0), "runtime   src->mult(2.0)->div(2.0)->add(-1)->sink - int single");

    constexpr auto templated_cascaded_test_bulk = []<typename T>(T factor, const char *test_name) {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<test::source<T>>(N_SAMPLES);
        auto     &mult = flow_graph.make_node<multiply_bulk<T>>(factor);
        auto     &div  = flow_graph.make_node<divide_bulk<T>>(factor);
        auto     &add1 = flow_graph.make_node<add_bulk<T>>(static_cast<T>(-1.f));
        auto     &sink = flow_graph.make_node<test::sink<T>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).template to<"in">(mult)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(mult).template to<"in">(div)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(div).template to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(add1).template to<"in">(sink)));

        fg::scheduler::simple sched{ std::move(flow_graph) };

        ::benchmark::benchmark<1LU>{ test_name }.repeat<N_ITER>(N_SAMPLES) = [&sched]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            sched.run_and_wait();
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    };
    templated_cascaded_test_bulk(static_cast<float>(2.0), "runtime   src->mult(2.0)->div(2.0)->add(-1)->sink - float bulk");
    templated_cascaded_test_bulk(static_cast<int>(2.0), "runtime   src->mult(2.0)->div(2.0)->add(-1)->sink - int bulk");
};

int
main() { /* not needed by the UT framework */
}
