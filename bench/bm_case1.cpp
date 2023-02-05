#include "benchmark.hpp"

#include <algorithm>
#include <boost/ut.hpp>
#include <functional>

#include "bm_test_helper.hpp"
#include <graph.hpp>
#include <vir/simd.h>

namespace fg = fair::graph;

inline constexpr std::size_t N_ITER = 10;
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(1'000'000, 1024);

template<typename T, int addend, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class add
        : public fg::node<add<T, addend, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>, fg::limits<N_MIN, N_MAX>> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    process_one(V a) const noexcept {
        return a + addend;
    }
};

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class multiply
        : public fg::node<multiply<T, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>, fg::limits<N_MIN, N_MAX>> {
    T _factor = static_cast<T>(1.0f);

public:
    multiply() = delete;

    explicit multiply(T factor, std::string name = fair::graph::this_source_location()) : _factor(
            factor) { this->set_name(name); }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    process_one(V a) const noexcept {
        return a * _factor;
    }
};

template<typename T, char op, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class gen_operation_SIMD
        : public fg::node<gen_operation_SIMD<T, op, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>, fg::limits<N_MIN, N_MAX>> {
    T _value = static_cast<T>(1.0f);

public:
    gen_operation_SIMD() = delete;

    explicit gen_operation_SIMD(T value, std::string name = fair::graph::this_source_location()) : _value(
            value) { this->set_name(name); }

    fair::graph::work_return_t
    work() noexcept {
        auto &out_port = output_port<"out">(this);
        auto &in_port = input_port<"in">(this);

        auto &reader = in_port.reader();
        auto &writer = out_port.writer();
        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
        if (n_readable == 0) {
            return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
        } else if (n_writable == 0) {
            return fair::graph::work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
        }
        const std::size_t n_to_publish = std::min(n_readable, n_writable);

        writer.publish( //
                [&reader, n_to_publish, this](std::span<T> output) {
                    const auto input = reader.get();
                    // #### N.B. later high-level user-function starts here

                    using namespace vir::stdx;
                    using V = native_simd<T>;
                    std::size_t i = 0;
                    const auto value = _value;
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
            return fair::graph::work_return_t::ERROR;
        }
        return fair::graph::work_return_t::OK;
    }
};

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
using multiply_SIMD = gen_operation_SIMD<T, '*', N_MIN, N_MAX>;

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
using add_SIMD = gen_operation_SIMD<T, '+', N_MIN, N_MAX>;

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX, bool use_bulk_operation = false, bool use_memcopy = true>
class copy
        : public fg::node<copy<T, N_MIN, N_MAX, use_bulk_operation, use_memcopy>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>, fg::limits<N_MIN, N_MAX>> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr T
    process_one(V a) const noexcept {
        return a;
    }

    fair::graph::work_return_t
    work() noexcept { // TODO - make this an alternate version to 'process_one'
        auto &out_port = output_port<"out">(this);
        auto &in_port = input_port<"in">(this);

        auto &reader = in_port.reader();
        auto &writer = out_port.writer();
        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
        if (n_readable == 0) {
            return fair::graph::work_return_t::DONE;
        } else if (n_writable == 0) {
            return fair::graph::work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
        }
        const std::size_t n_to_publish = std::min(n_readable, n_writable);

        if constexpr (use_memcopy) {
            // fmt::print("n_to_publish {} - {} {}\n", n_to_publish, use_bulk_operation, use_memcopy);
            writer.publish( //
                    [&reader, n_to_publish](std::span<T> output) {
                        std::memcpy(output.data(), reader.get().data(), n_to_publish * sizeof(T));
                    }, n_to_publish);
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
            return fair::graph::work_return_t::ERROR;
        }
        return fair::graph::work_return_t::OK;
    }
};


namespace detail {
    template<typename T>
    constexpr std::size_t simd_size() noexcept {
        namespace stdx = vir::stdx;
        if constexpr (stdx::is_simd_v<T>) {
            return T::size();
        } else {
            return 1LU;
        }
    }
}

namespace stdx = vir::stdx;

template<typename From, typename To, std::size_t N_MIN = 0 /* SIMD size */, std::size_t N_MAX = N_MAX>
class convert : public fg::node<convert<From, To, N_MIN, N_MAX>, fg::IN<From, "in", N_MIN, N_MAX>,
                                fg::OUT<To, "out", N_MIN, N_MAX>, fg::limits<N_MIN, N_MAX>> {
    static_assert(stdx::is_simd_v<From> != stdx::is_simd_v<To>, "either input xor output must be SIMD capable");
    constexpr static std::size_t from_simd_size  = detail::simd_size<From>();
    constexpr static std::size_t to_simd_size = detail::simd_size<To>();
    constexpr static std::size_t simd_size = std::max(from_simd_size, to_simd_size);

public:
    fair::graph::work_return_t
    work() noexcept {
        using namespace stdx;
        auto      &out_port   = output_port<"out">(this);
        auto      &in_port    = input_port<"in">(this);

        auto      &reader     = in_port.reader();
        auto      &writer     = out_port.writer();
        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
        if (n_readable < to_simd_size) {
            return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
        } else if (n_writable < from_simd_size) {
            return fair::graph::work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
        }
        const auto n_readable_scalars = n_readable * from_simd_size;
        const auto n_writable_scalars = n_writable * to_simd_size;
        const auto n_simd_to_convert = std::min(n_readable_scalars, n_writable_scalars) / simd_size;
        const auto scalars_to_convert = n_simd_to_convert * simd_size;
        const auto objects_to_write = stdx::is_simd_v<To> ? n_simd_to_convert : scalars_to_convert;
        const auto objects_to_read = stdx::is_simd_v<From> ? n_simd_to_convert : scalars_to_convert;

        auto       return_value    = fair::graph::work_return_t::OK;
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
                        return_value = fair::graph::work_return_t::ERROR;
                        return;
                    }
                },
                objects_to_write);
        return return_value;
    }
};

inline const boost::ut::suite _constexpr_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;
    using fair::graph::merge_by_index;
    using fair::graph::merge;

    {
        auto merged_node = merge<"out", "in">(test::source<float>(N_SAMPLES), test::sink<float>());
        //
        "constexpr src->sink overhead V1"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.process_one();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };

        "constexpr src->sink overhead V2"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.work();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };
    }

    {
        auto merged_node = merge<"out", "in">(merge<"out", "in">(test::source<float>(N_SAMPLES), copy<float>()),
                                              test::sink<float>());
        //
        "constexpr src->copy->sink V1"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.process_one();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };

        "constexpr src->copy->sink V2"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.work();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };
    }

    {
        auto merged_node = merge<"out", "in">(
                merge<"out", "in">(test::source<float>(N_SAMPLES), test::cascade<10, copy<float>>(copy<float>())),
                test::sink<float>());
        "constexpr src->copy^10->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.process_one();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };
    }

    {
        auto merged_node = merge<"out", "in">(merge<"out", "in">(merge<"out", "in">(
                                                                         merge<"out", "in">(test::source<float, 1024, 1024>(N_SAMPLES), copy<float, 0, 128>()),
                                                                         copy<float, 0, 1024>()),
                                                                 copy<float, 32, 128>()),
                                              test::sink<float>());
        "constexpr src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.process_one();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };
    }

    {
        auto gen_mult_block_float = [] {
            return merge<"out", "in">(multiply<float>(2.0f), merge<"out", "in">(multiply<float>(0.5f),
                                                                                add<float, -1>()));
        };
        auto merged_node = merge<"out", "in">(
                merge<"out", "in">(test::source<float>(N_SAMPLES), gen_mult_block_float()), test::sink<float>());
        "constexpr src->mult(2.0)->mult(0.5)->add(-1)->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.process_one();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };
    }

    {
        auto gen_mult_block_float = [] {
            return merge<"out", "in">(multiply<float>(2.0f), merge<"out", "in">(multiply<float>(0.5f),
                                                                                add<float, -1>()));
        };
        auto merged_node = merge<"out", "in">(merge<"out", "in">(test::source<float>(N_SAMPLES), //
                                                                 test::cascade<10, decltype(gen_mult_block_float())>(
                                                                         gen_mult_block_float(), gen_mult_block_float)),
                                              test::sink<float>());
        "constexpr src->(mult(2.0)->mult(0.5)->add(-1))^10->sink"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&merged_node]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            for (std::size_t i = 0; i < N_SAMPLES; i++) {
                merged_node.process_one();
            }
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough samples";
        };
    }
};

inline const boost::ut::suite _runtime_tests = [] {
    using namespace boost::ut;
    using namespace benchmark;

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(sink)));

        "runtime   src->sink overhead"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&flow_graph]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            auto token = flow_graph.init();
            expect(token);
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *sink = flow_graph.make_node<test::sink<float>>();
        auto *cpy = flow_graph.make_node<copy<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(cpy)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(cpy).to<"in">(sink)));

        "runtime   src->copy->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&flow_graph]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            auto token = flow_graph.init();
            expect(token);
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *sink = flow_graph.make_node<test::sink<float>>();

        std::vector<copy<float, 0, N_MAX, true, true>*> cpy(10);
        for (std::size_t i = 0; i < cpy.size(); i++) {
            cpy[i] = flow_graph.make_node<copy<float, 0, N_MAX, true, true>>();
            cpy[i]->set_name(fmt::format("copy {} at {}", i, fair::graph::this_source_location()));

            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(cpy[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(cpy[i - 1]).to<"in">(cpy[i])));
            }
        }

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(cpy[cpy.size() - 1]).to<"in">(sink)));

        "runtime   src->copy^10->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&flow_graph]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            auto token = flow_graph.init();
            expect(token);
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float, 0, 1024>>(N_SAMPLES);
        auto *b1 = flow_graph.make_node<copy<float, 0, 128>>();
        auto *b2 = flow_graph.make_node<copy<float, 1024, 1024>>();
        auto *b3 = flow_graph.make_node<copy<float, 32, 128>>();
        auto *sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(b1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(b1).to<"in">(b2)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(b2).to<"in">(b3)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(b3).to<"in">(sink)));

        "runtime   src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&flow_graph]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            auto token = flow_graph.init();
            expect(token);
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *mult1 = flow_graph.make_node<multiply<float>>(2.0f);
        auto *mult2 = flow_graph.make_node<multiply<float>>(0.5f);
        auto *add1 = flow_graph.make_node<add<float, -1>>();
        auto *sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(mult1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult1).to<"in">(mult2)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult2).to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1).to<"in">(sink)));

        "runtime   src->mult(2.0)->mult(0.5)->add(-1)->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&flow_graph]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            auto token = flow_graph.init();
            expect(token);
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *sink = flow_graph.make_node<test::sink<float>>();

        std::vector<multiply<float>*> mult1;
        std::vector<multiply<float>*> mult2;
        std::vector<add<float, -1>*> add1;
        for (std::size_t i = 0; i < 10; i++) {
            mult1.emplace_back(flow_graph.make_node<multiply<float>>(2.0f, fmt::format("mult1.{}", i)));
            mult2.emplace_back(flow_graph.make_node<multiply<float>>(0.5f, fmt::format("mult2.{}", i)));
            add1.emplace_back(flow_graph.make_node<add<float, -1>>());
        }

        for (std::size_t i = 0; i < add1.size(); i++) {
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(mult1[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1[i - 1]).to<"in">(mult1[i])));
            }
            expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult1[i]).to<"in">(mult2[i])));
            expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult2[i]).to<"in">(add1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1[add1.size() - 1]).to<"in">(sink)));

        auto token = flow_graph.init();
        expect(token);
        "runtime   src->(mult(2.0)->mult(0.5)->add(-1))^10->sink"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&flow_graph, &token]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *mult1 = flow_graph.make_node<multiply_SIMD<float>>(2.0f);
        auto *mult2 = flow_graph.make_node<multiply_SIMD<float>>(0.5f);
        auto *add1 = flow_graph.make_node<add_SIMD<float>>(-1.0f);
        auto *sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(mult1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult1).to<"in">(mult2)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult2).to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1).to<"in">(sink)));

        auto token = flow_graph.init();
        expect(token);
        "runtime   src->mult(2.0)->mult(0.5)->add(-1)->sink (SIMD)"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&flow_graph, &token]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *sink = flow_graph.make_node<test::sink<float>>();

        std::vector<multiply_SIMD<float>*> mult1;
        std::vector<multiply_SIMD<float>*> mult2;
        std::vector<add_SIMD<float>*> add1;
        for (std::size_t i = 0; i < 10; i++) {
            mult1.emplace_back(flow_graph.make_node<multiply_SIMD<float>>(2.0f, fmt::format("mult1.{}", i)));
            mult2.emplace_back(flow_graph.make_node<multiply_SIMD<float>>(0.5f, fmt::format("mult2.{}", i)));
            add1.emplace_back(flow_graph.make_node<add_SIMD<float>>(-1.0f, fmt::format("add.{}", i)));
        }

        for (std::size_t i = 0; i < add1.size(); i++) {
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(mult1[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1[i - 1]).to<"in">(mult1[i])));
            }
            expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult1[i]).to<"in">(mult2[i])));
            expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult2[i]).to<"in">(add1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1[add1.size() - 1]).to<"in">(sink)));

        auto token = flow_graph.init();
        expect(token);
        "runtime   src->(mult(2.0)->mult(0.5)->add(-1))^10->sink (SIMD)"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&flow_graph, &token]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        using namespace stdx;
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *convert1 = flow_graph.make_node<convert<float, stdx::native_simd<float>>>();
        auto *mult1 = flow_graph.make_node<multiply<stdx::native_simd<float>>>(2.0f);
        auto *mult2 = flow_graph.make_node<multiply<stdx::native_simd<float>>>(0.5f);
        auto *add1 = flow_graph.make_node<add<stdx::native_simd<float>, -1>>();
        auto *convert2 = flow_graph.make_node<convert<stdx::native_simd<float>, float>>();
        auto *sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(convert1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(convert1).to<"in">(mult1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult1).to<"in">(mult2)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult2).to<"in">(add1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1).to<"in">(convert2)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(convert2).to<"in">(sink)));

        fmt::print("start bm: {} - simd_size: {}\n", "runtime   src->mult(2.0)->mult(0.5)->add(-1)->sink (SIMD-alt)",
                   ::detail::simd_size<stdx::native_simd<float>>());
        auto token = flow_graph.init();
        expect(token);
        "runtime   src->mult(2.0)->mult(0.5)->add(-1)->sink (SIMD-alt)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&flow_graph, &token]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }

    {
        fg::graph flow_graph;
        auto *src = flow_graph.make_node<test::source<float>>(N_SAMPLES);
        auto *convert1 = flow_graph.make_node<convert<float, stdx::native_simd<float>>>();
        auto *convert2 = flow_graph.make_node<convert<stdx::native_simd<float>, float>>();
        auto *sink = flow_graph.make_node<test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(src).to<"in">(convert1)));
        expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(convert2).to<"in">(sink)));

        std::vector<multiply<stdx::native_simd<float>>*> mult1;
        std::vector<multiply<stdx::native_simd<float>>*> mult2;
        std::vector<add<stdx::native_simd<float>, -1>*> add1;
        for (std::size_t i = 0; i < 10; i++) {
            mult1.emplace_back(flow_graph.make_node<multiply<stdx::native_simd<float>>>(2.0f, fmt::format("mult1.{}", i)));
            mult2.emplace_back(flow_graph.make_node<multiply<stdx::native_simd<float>>>(0.5f, fmt::format("mult2.{}", i)));
            add1.emplace_back(flow_graph.make_node<add<stdx::native_simd<float>, -1>>());
        }

        for (std::size_t i = 0; i < add1.size(); i++) {
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(convert1).to<"in">(mult1[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(add1[i - 1]).to<"in">(mult1[i])));
            }
            expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult1[i]).to<"in">(mult2[i])));
            expect(eq(fg::connection_result_t::SUCCESS, connect<"out">(mult2[i]).to<"in">(add1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS,
                  connect<"out">(add1[add1.size() - 1]).to<"in">(convert2)));

        fmt::print("start bm: {} - simd_size: {}\n",
                   "runtime   src->(mult(2.0)->mult(0.5)->add(-1))^10->sink (SIMD alt)",
                   ::detail::simd_size<stdx::native_simd<float>>());
        auto token = flow_graph.init();
        expect(token);
        "runtime   src->(mult(2.0)->mult(0.5)->add(-1))^10->sink (SIMD alt)"_benchmark.repeat<N_ITER>(
                N_SAMPLES) = [&flow_graph, &token]() {
            test::n_samples_produced = 0LU;
            test::n_samples_consumed = 0LU;
            flow_graph.work(token);
            expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
            expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
        };
    }
};

int
main() { /* not needed by the UT framework */
}
