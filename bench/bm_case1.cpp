#include "benchmark.hpp"
#include <boost/ut.hpp>
#include <functional>

#include <graph.hpp>

//#define RUN_SIMD_TESTS

namespace fg = fair::graph;

inline constexpr std::size_t N_MAX = std::numeric_limits<std::size_t>::max();

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class test_src : public fg::node<test_src<T, N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>> {
public:
    [[nodiscard]] constexpr T
    process_one() const noexcept {
        return T{};
    }
};

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class test_sink : public fg::node<test_sink<T, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>> {

public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        benchmark::do_not_optimize(a);
    }
};

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class copy : public fg::node<copy<T, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr T
    process_one(V a) const noexcept {
        return a;
    }
};

template<typename T1, typename T2, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class convert
        : public fg::node<convert<T1, T2, N_MIN, N_MAX>, fg::IN<T1, "in", N_MIN, N_MAX>, fg::OUT<T2, "out", N_MIN, N_MAX>> {
public:
    template<fair::meta::t_or_simd<T1> V>
    [[nodiscard]] constexpr T2
    process_one(V a) const noexcept {
        return a;
    }
};

template<typename T, int addend, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class add
        : public fg::node<add<T, addend, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr T
    process_one(V a) const noexcept {
        return a + addend;
    }
};

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class multiply
        : public fg::node<multiply<T, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::OUT<T, "out", N_MIN, N_MAX>> {
    T _factor = static_cast<T>(1.0f);

public:
    multiply() = delete;

    explicit multiply(T factor) : _factor(factor) {}

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr T
    process_one(V a) noexcept {
        return a * _factor;
    }
};

template<std::size_t N, typename base, typename aggregate>
constexpr auto
cascade(
        aggregate &&src, std::function<base()> generator = [] { return base(); }) {
    if constexpr (N <= 1) {
        return src;
    } else {
        return cascade<N - 1, base>(fair::graph::merge_by_index<0, 0>(std::forward<aggregate>(src),
                                                                      generator()),
                                    generator);
    }
}

inline const boost::ut::suite _constexpr_bm = [] {
    constexpr std::size_t N_ITER = 100'000;
    using namespace benchmark;
    using fair::graph::merge_by_index;
    using fair::graph::merge;

    auto src_sink_merged = merge<"out", "in">(test_src<float>(), test_sink<float>());
    "constexpr src->sink overhead"_benchmark.repeat<N_ITER>() = [&src_sink_merged]() {
        src_sink_merged.process_one();
    };

    auto src_copy_sink = merge<"out", "in">(merge<"out", "in">(test_src<float>(),
                                                               copy<float>()),
                                            test_sink<float>());
    "constexpr src->copy->sink"_benchmark.repeat<N_ITER>() = [&src_copy_sink]() {
        src_copy_sink.process_one();
    };

    auto src_copy_n_sink = merge<"out", "in">(merge<"out", "in">(test_src<float>(),
                                                                 cascade<10, copy<float>>(
                                                                         copy<float>())),
                                              test_sink<float>());
    "constexpr src->copy^10->sink"_benchmark.repeat<N_ITER>() = [&src_copy_n_sink]() {
        src_copy_n_sink.process_one();
    };

#ifdef RUN_SIMD_TESTS
    using std::experimental::fixed_size_simd;
    using std::experimental::simd;
    using std::experimental::native_simd;
    using float_simd = native_simd<float>;
    fmt::print(stderr, "float_simd::size() = {}\n", float_simd::size());
    auto src_copy_sink_simd = merge<"out", "in">(
            merge<"out", "in">(merge<"out", "in">(test_src<float>(), convert<float, float_simd>()),
                               copy<float_simd>()),
            test_sink<float_simd>());
    "constexpr src->copy->sink (SIMD)"_benchmark.repeat<N_ITER>() = [&src_copy_sink]() {
        src_copy_sink.process_one();
    };
    auto src_copy_n_sink_simd = merge<"out", "in">(merge<"out", "in">(test_src<float_simd>(),
                                                                      cascade<10, copy<float_simd>>(
                                                                              copy<float_simd>())),
                                                   test_sink<float_simd>());
    "constexpr src->copy^10->sink (SIMD)"_benchmark.repeat<N_ITER>() = [&src_copy_n_sink_simd]() {
        src_copy_n_sink_simd.process_one();
    };
#endif // - #RUN_SIMD_TESTS

    auto src_constraint_sink = merge<"out", "in">(
            merge<"out",
                    "in">(merge<"out", "in">(merge<"out", "in">(test_src<float, 1024, 1024>(),
                                                                copy<float, 0, 128>()),
                                             copy<float, 0, 1024>()),
                          copy<float, 32, 128>()),
            test_sink<float>());
    "constexpr src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink"_benchmark.repeat<N_ITER>() =
            [&src_constraint_sink]() { src_constraint_sink.process_one(); };

    auto gen_mult_block_float = [] {
        return merge<"out", "in">(multiply<float>(2.0f),
                                  merge<"out", "in">(multiply<float>(0.5f), add<float, -1>()));
    };
    auto src_mult_sink = merge<"out", "in">(merge<"out", "in">(test_src<float>(),
                                                               gen_mult_block_float()),
                                            test_sink<float>());
    "constexpr src->mult(2.0)->mult(0.5)->add(-1)->sink"_benchmark.repeat<N_ITER>() =
            [&src_mult_sink]() { src_mult_sink.process_one(); };

    auto src_mult_n_sink = merge<"out", "in">(
            merge<"out", "in">(test_src<float>(),
                               cascade<10, decltype(gen_mult_block_float())>(gen_mult_block_float(),
                                                                             gen_mult_block_float)),
            test_sink<float>());
    "constexpr src->(mult(2.0)->mult(0.5)->add(-1))^10->sink"_benchmark.repeat<N_ITER>() =
            [&src_mult_n_sink]() { src_mult_n_sink.process_one(); };
#ifdef RUN_SIMD_TESTS
    auto src_constraint_sink_SIMD = merge<"out", "in">(
            merge<"out", "in">(merge<"out", "in">(merge<"out", "in">(test_src<float_simd,
                                                                             1024, 1024>(),
                                                                     copy<float_simd, 0, 128>()),
                                                  copy<float_simd, 0, 1024>()),
                               copy<float_simd, 32, 128>()),
            test_sink<float_simd>());
    "constexpr src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink (SIMD)"_benchmark
            .repeat<N_ITER>()
            = [&src_constraint_sink_SIMD]() { src_constraint_sink_SIMD.process_one(); };

    auto gen_mult_block_simd_float = [] {
        return merge<"out", "in">(multiply<float_simd>(2.0f),
                                  merge<"out", "in">(multiply<float_simd>(0.5f),
                                                     add<float_simd, -1>()));
    };

    auto src_mult_sink_SIMD = merge<"out", "in">(merge<"out", "in">(test_src<float_simd>(),
                                                                    gen_mult_block_simd_float()),
                                                 test_sink<float_simd>());
    "constexpr src->mult(2.0)->mult(0.5)->add(-1)->sink (SIMD)"_benchmark.repeat<N_ITER>() =
            [&src_mult_sink_SIMD]() { src_mult_sink_SIMD.process_one(); };

    auto src_mult_n_sink_SIMD = merge<"out", "in">(
            merge<"out", "in">(test_src<float_simd>(),
                               cascade<10, decltype(gen_mult_block_simd_float())>(
                                       gen_mult_block_simd_float(), gen_mult_block_simd_float)),
            test_sink<float_simd>());
    "constexpr src->(mult(2.0)->mult(0.5)->add(-1))^10->sink (SIMD)"_benchmark.repeat<N_ITER>() =
            [&src_mult_n_sink_SIMD]() { src_mult_n_sink_SIMD.process_one(); };

#endif // - #RUN_SIMD_TESTS
};

inline const boost::ut::suite _runtime_tests = [] {
    constexpr std::size_t N_ITER = 100'000;
    using namespace boost::ut;
    using namespace benchmark;

    {
        test_src<float> src;
        test_sink<float> sink;

        fg::graph flow_graph;
        flow_graph.register_node(src);
        flow_graph.register_node(sink);
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(src, sink)));

        "runtime   src->sink overhead"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

    {
        test_src<float> src;
        test_sink<float> sink;
        copy<float> cpy;

        fg::graph flow_graph;
        flow_graph.register_node(src);
        flow_graph.register_node(cpy);
        flow_graph.register_node(sink);
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(src, cpy)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(cpy, sink)));
        "runtime   src->copy->sink"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

    {
        test_src<float> src;
        test_sink<float> sink;

        fg::graph flow_graph;
        flow_graph.register_node(src);

        std::vector<copy<float>> cpy(10);
        for (std::size_t i = 0; i < cpy.size(); i++) {
            flow_graph.register_node(cpy[i]);
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS,
                          flow_graph.connect<"out", "in">(src, cpy[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS,
                          flow_graph.connect<"out", "in">(cpy[i], cpy[i - 1])));
            }
        }

        expect(eq(fg::connection_result_t::SUCCESS,
                  flow_graph.connect<"out", "in">(cpy[cpy.size() - 1], sink)));

        "runtime   src->copy^10->sink"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

#ifdef RUN_SIMD_TESTS
    using std::experimental::fixed_size_simd;
    using std::experimental::simd;
    using std::experimental::native_simd;
    using float_simd = native_simd<float>;

    {
        test_src<float_simd> src;
        test_sink<float_simd> sink;
        copy<float_simd> cpy;

        fg::graph flow_graph;
        flow_graph.register_node(src);
        flow_graph.register_node(cpy);
        flow_graph.register_node(sink);
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(src, cpy)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(cpy, sink)));
        "runtime   src->copy->sink SIMD"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

    {
        test_src<float_simd> src;
        test_sink<float_simd> sink;

        fg::graph flow_graph;
        flow_graph.register_node(src);

        std::vector<copy<float_simd>> cpy(10);
        for (std::size_t i = 0LU; i < cpy.size(); i++) {
            flow_graph.register_node(cpy[i]);
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS,
                          flow_graph.connect<"out", "in">(src, cpy[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS,
                          flow_graph.connect<"out", "in">(cpy[i], cpy[i - 1])));
            }
        }

        expect(eq(fg::connection_result_t::SUCCESS,
                  flow_graph.connect<"out", "in">(cpy[cpy.size() - 1], sink)));

        "runtime   src->copy^10->sink SIMD"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

#endif // RUN_SIMD_TESTS

    {
        test_src<float, 1024, 1024> src;
        copy<float, 0, 128> b1;
        copy<float, 1024, 1024> b2;
        copy<float, 32, 128> b3;
        test_sink<float> sink;

        fg::graph flow_graph;
        flow_graph.register_node(src);
        flow_graph.register_node(b1);
        flow_graph.register_node(b2);
        flow_graph.register_node(b3);
        flow_graph.register_node(sink);
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(src, b1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(b1, b2)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(b2, b3)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(b3, sink)));
        "runtime   src(N=1024)->b1(N≤128)->b2(N=1024)->b3(N=32...128)->sink"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

    {
        test_src<float> src;
        multiply<float> mult1(2.0f);
        multiply<float> mult2(0.5f);
        add<float, -1> add1;
        test_sink<float> sink;

        fg::graph flow_graph;
        flow_graph.register_node(src);
        flow_graph.register_node(mult1);
        flow_graph.register_node(mult2);
        flow_graph.register_node(add1);
        flow_graph.register_node(sink);
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(src, mult1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(mult1, mult2)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(mult2, add1)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(add1, sink)));
        "runtime   src->mult(2.0)->mult(0.5)->add(-1)->sink"_benchmark.repeat<N_ITER>() = [&flow_graph]() {
            flow_graph.work();
        };
    }

    {
        test_src<float> src;
        test_sink<float> sink;

        fg::graph flow_graph;
        flow_graph.register_node(src);

        std::vector<multiply<float>> mult1;
        std::vector<multiply<float>> mult2;
        std::vector<add<float, -1>> add1(10);
        for (std::size_t i = 0; i < add1.size(); i++) {
            mult1.emplace_back(2.0f);
            mult2.emplace_back(0.5f);
            flow_graph.register_node(mult1[i]);
            flow_graph.register_node(mult2[i]);
            flow_graph.register_node(add1[i]);
            if (i == 0) {
                expect(eq(fg::connection_result_t::SUCCESS,
                          flow_graph.connect<"out", "in">(src, mult1[i])));
            } else {
                expect(eq(fg::connection_result_t::SUCCESS,
                          flow_graph.connect<"out", "in">(mult1[i], add1[i - 1])));
            }
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(mult1[i], mult2[i])));
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out", "in">(mult2[i], add1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS,
                  flow_graph.connect<"out", "in">(add1[add1.size() - 1], sink)));

        "runtime   src->(mult(2.0)->mult(0.5)->add(-1))^10->sink"_benchmark.repeat<N_ITER>() = [&]() {
            src.work();
            for (auto i = 0; i < add1.size(); i++) {
                mult1[i].work();
                mult2[i].work();
                add1[i].work();
            }
            sink.work();
//            flow_graph.work();
        };
    }
};

int
main() { /* not needed by the UT framework */
}
