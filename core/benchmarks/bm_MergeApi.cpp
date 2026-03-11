#include <benchmark.hpp>

#include <vir/simd.h>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/ImGraph.hpp>

#include <gnuradio-4.0/testing/NullSources.hpp>

#include <gnuradio-4.0/math/Math.hpp>

void printTopology(std::string_view label, gr::Graph& g) { std::println("{}:\n{}", label, gr::graph::draw(g)); }
void printTopology(std::string_view label, gr::Graph&& g) { printTopology(label, g); }

inline constexpr std::size_t N_MAX              = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t N_ITER             = 10;
inline constexpr gr::Size_t  N_SAMPLES          = gr::util::round_up(1'000'000, 1024);
inline constexpr gr::Size_t  N_SAMPLES_FEEDBACK = gr::util::round_up(100'000, 1024);

using namespace gr::blocks::math;

// -- CascadeType: recursively merge N copies of Base via MergeByIndex --
// CascadeType<N, Base> produces N+1 copies: CascadeType<0,B>=B, CascadeType<1,B>=Merge(B,B), etc.

template<std::size_t N, typename Base, typename = Base>
struct CascadeTypeHelper;

template<typename Base, typename Aggregate>
struct CascadeTypeHelper<0, Base, Aggregate> {
    using type = Base;
};

template<std::size_t N, typename Base, typename Aggregate>
struct CascadeTypeHelper {
    using type = gr::MergeByIndex<typename CascadeTypeHelper<N - 1, Base, Aggregate>::type, 0, Base, 0>;
};

template<std::size_t N, typename Base>
using CascadeType = typename CascadeTypeHelper<N, Base>::type;

// -- benchmark-local 2-input adder for FeedbackMerge (Math.hpp's Add<T> uses processBulk/dynamic ports) --

inline constexpr float kAlpha = 0.3f;

template<typename T = float>
struct Adder : gr::Block<Adder<T>> {
    gr::PortIn<T>  in1;
    gr::PortIn<T>  in2;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(Adder, in1, in2, out);

    [[nodiscard]] constexpr T processOne(T a, T b) const noexcept { return a + b; }
};

// IIR low-pass: y[n] = α·x[n] + (1-α)·y[n-1]
using IIRChain = gr::Merge<MultiplyConst<float>, "out", gr::FeedbackMerge<Adder<>, "out", MultiplyConst<float>, "out", "in2">, "in1">;

// IIR low-pass decomposed via SplitMergeCombine: feedback = input + (-α·input) = (1-α)·input
using IIRFeedbackFanOut  = gr::SplitMergeCombine<MultiplyConst<float>, MultiplyConst<float>>;
using IIRChainSplitMerge = gr::Merge<MultiplyConst<float>, "out", gr::FeedbackMerge<Adder<>, "out", IIRFeedbackFanOut, "out", "in2">, "in1">;

// SplitMergeCombine fan-out: src -> {Scale(2), Scale(3)} -> sum = 5x
using SplitMergeChain = gr::SplitMergeCombine<MultiplyConst<float>, MultiplyConst<float>>;

// hand-written scalar IIR reference — baseline for zero-overhead comparison
struct IIRReference {
    float _state{};
    float _alpha = kAlpha;

    [[nodiscard]] constexpr float processOne(float input) noexcept {
        _state = _alpha * input + (1.0f - _alpha) * _state;
        return _state;
    }
};

// -- benchmark-local blocks --

// CountingSink's non-const processOne forces invokeProcessOneNonConst (per-sample atomic state check, no SIMD).
// FastSink uses mutable count + const processOne to enable the SIMD path, isolating merge overhead from sink overhead.
template<typename T>
struct FastSink : gr::Block<FastSink<T>> {
    gr::PortIn<T> in;
    GR_MAKE_REFLECTABLE(FastSink, in);

    mutable gr::Size_t count = 0U;

    void reset() { count = 0U; }

    template<gr::meta::t_or_simd<T> V>
    void processOne(V value) const noexcept {
        if constexpr (gr::meta::any_simd<V>) {
            count += static_cast<gr::Size_t>(V::size());
        } else {
            count++;
        }
        benchmark::force_to_memory(value);
    }
};

#if !DISABLE_SIMD
static_assert(gr::traits::block::can_processOne_simd<FastSink<float>>);
#endif

template<typename T, char op>
struct math_bulk_op : gr::Block<math_bulk_op<T, op>> {
    gr::PortIn<T, gr::RequiredSamples<1, N_MAX>>  in;
    gr::PortOut<T, gr::RequiredSamples<1, N_MAX>> out;
    T                                             value = static_cast<T>(1);

    GR_MAKE_REFLECTABLE(math_bulk_op, in, out, value);

    [[nodiscard]] constexpr gr::work::Status processBulk(std::span<const T> input, std::span<T> output) const noexcept {
        for (std::size_t i = 0; i < input.size(); i++) {
            if constexpr (op == '*') {
                output[i] = input[i] * value;
            } else if constexpr (op == '/') {
                output[i] = input[i] / value;
            } else if constexpr (op == '+') {
                output[i] = input[i] + value;
            } else if constexpr (op == '-') {
                output[i] = input[i] - value;
            } else {
                static_assert(gr::meta::always_false<T>, "unknown op");
            }
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
using multiply_bulk = math_bulk_op<T, '*'>;
template<typename T>
using divide_bulk = math_bulk_op<T, '/'>;
template<typename T>
using add_bulk = math_bulk_op<T, '+'>;

template<typename T, int addend>
struct add : gr::Block<add<T, addend>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(add, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        return a + static_cast<T>(addend);
    }
};
#if !DISABLE_SIMD
static_assert(gr::traits::block::can_processOne_simd<add<float, 1>>);
#endif

template<typename T, std::size_t N_MIN = 1, std::size_t N_MAX = N_MAX, bool use_bulk_operation = false, bool use_memcopy = true>
class copy : public gr::Block<copy<T, N_MIN, N_MAX, use_bulk_operation, use_memcopy>> {
public:
    gr::PortIn<T, gr::RequiredSamples<N_MIN, N_MAX>>  in;
    gr::PortOut<T, gr::RequiredSamples<N_MIN, N_MAX>> out;

    GR_MAKE_REFLECTABLE(copy, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        return a;
    }
};

// -- common type aliases for merged processing chains --

template<typename T>
using MathChain = gr::Merge<MultiplyConst<T>, "out", gr::Merge<DivideConst<T>, "out", add<T, -1>, "in">, "in">;

// -- helpers --

void loop_over_processOne(auto& mergedBlock, gr::Size_t nSamples) {
    using namespace boost::ut;
    auto& sink = mergedBlock._rightBlock;
    sink.count = 0U;

    constexpr std::size_t kWidth = vir::stdx::simd_abi::max_fixed_size<double>;
    if constexpr (requires { mergedBlock.processOne(gr::meta::cw<kWidth>); }) {
        constexpr auto kW = gr::meta::cw<kWidth>;
        std::size_t    i  = 0UZ;
        for (; i + kWidth <= nSamples; i += kWidth) {
            mergedBlock.processOne(kW);
        }
        for (; i < nSamples; i++) {
            mergedBlock.processOne();
        }
    } else {
        for (gr::Size_t i = 0; i < nSamples; i++) {
            mergedBlock.processOne();
        }
    }
    expect(eq(static_cast<gr::Size_t>(sink.count), nSamples)) << "sink count mismatch";
}

// -- graph builders for runtime (a): individual blocks, no merging --

template<typename T>
gr::Graph make_src_sink_graph() {
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_copy_chain_graph(std::size_t depth = 1) {
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});

    using Copy = ::copy<T>;
    std::vector<Copy*> copies(depth);
    for (std::size_t i = 0; i < depth; i++) {
        copies[i] = std::addressof(g.emplaceBlock<Copy>());
        if (i == 0) {
            if (!g.connect<"out", "in">(src, *copies[i]).has_value()) {
                throw std::runtime_error("connection failed");
            }
        } else {
            if (!g.connect<"out", "in">(*copies[i - 1], *copies[i]).has_value()) {
                throw std::runtime_error("connection failed");
            }
        }
    }
    if (!g.connect<"out", "in">(*copies.back(), sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_math_chain_graph(T factor, std::size_t depth = 1) {
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});

    std::vector<MultiplyConst<T>*> mult;
    std::vector<DivideConst<T>*>   div;
    std::vector<add<T, -1>*>       addBlk;
    for (std::size_t i = 0; i < depth; i++) {
        mult.emplace_back(std::addressof(g.emplaceBlock<MultiplyConst<T>>({{"value", factor}})));
        div.emplace_back(std::addressof(g.emplaceBlock<DivideConst<T>>({{"value", factor}})));
        addBlk.emplace_back(std::addressof(g.emplaceBlock<add<T, -1>>()));
    }
    for (std::size_t i = 0; i < depth; i++) {
        if (i == 0) {
            if (!g.connect<"out", "in">(src, *mult[i]).has_value()) {
                throw std::runtime_error("connection failed");
            }
        } else {
            if (!g.connect<"out", "in">(*addBlk[i - 1], *mult[i]).has_value()) {
                throw std::runtime_error("connection failed");
            }
        }
        if (!g.connect<"out", "in">(*mult[i], *div[i]).has_value()) {
            throw std::runtime_error("connection failed");
        }
        if (!g.connect<"out", "in">(*div[i], *addBlk[i]).has_value()) {
            throw std::runtime_error("connection failed");
        }
    }
    if (!g.connect<"out", "in">(*addBlk.back(), sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_bulk_chain_graph(T factor) {
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     mult = g.emplaceBlock<multiply_bulk<T>>({{"value", factor}});
    auto&     div  = g.emplaceBlock<divide_bulk<T>>({{"value", factor}});
    auto&     addB = g.emplaceBlock<add_bulk<T>>({{"value", static_cast<T>(-1)}});
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});

    if (!g.connect<"out", "in">(src, mult).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(mult, div).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(div, addB).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(addB, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_iir_lowpass_graph() {
    gr::Graph g;
    auto&     src   = g.emplaceBlock<gr::testing::ConstantSource<T>>({{"n_samples_max", N_SAMPLES_FEEDBACK}});
    auto&     scale = g.emplaceBlock<MultiplyConst<T>>({{"value", static_cast<T>(kAlpha)}});
    auto&     adder = g.emplaceBlock<Adder<T>>();
    auto&     fb    = g.emplaceBlock<MultiplyConst<T>>({{"value", static_cast<T>(1.0f - kAlpha)}});
    auto&     sink  = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES_FEEDBACK}});

    auto ok = [](std::expected<void, gr::Error> r) {
        if (!r.has_value()) {
            throw std::runtime_error(std::format("connection failed: {}", r.error().message));
        }
    };
    ok(g.connect<"out", "in">(src, scale));
    ok(g.connect<"out", "in1">(scale, adder));
    ok(g.connect<"out", "in">(adder, sink));
    ok(g.connect<"out", "in">(adder, fb));
    ok(g.connect<"out", "in2">(fb, adder));
    return g;
}

// -- graph builders for merge-API (b): merged processing chain + scheduler --

template<typename T>
gr::Graph make_merged_iir_graph() {
    gr::Graph g;
    auto&     src                    = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     proc                   = g.emplaceBlock<IIRChain>();
    proc._leftBlock.value            = kAlpha;
    proc._rightBlock._feedback.value = static_cast<T>(1.0f - kAlpha);
    auto& sink                       = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, proc).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(proc, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_merged_split_merge_graph() {
    gr::Graph g;
    auto&     src        = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     proc       = g.emplaceBlock<SplitMergeChain>();
    proc.path<0>().value = 2.0f;
    proc.path<1>().value = 3.0f;
    auto& sink           = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, proc).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(proc, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_merged_iir_split_merge_graph() {
    gr::Graph g;
    auto&     src                              = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     proc                             = g.emplaceBlock<IIRChainSplitMerge>();
    proc._leftBlock.value                      = kAlpha;
    proc._rightBlock._feedback.path<0>().value = 1.0f;
    proc._rightBlock._feedback.path<1>().value = -kAlpha;
    auto& sink                                 = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, proc).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(proc, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T>
gr::Graph make_merged_src_sink_graph() {
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T, std::size_t depth = 1>
gr::Graph make_merged_copy_graph() {
    using chain = CascadeType<depth - 1, copy<T>>;
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     proc = g.emplaceBlock<chain>();
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, proc).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(proc, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

template<typename T, std::size_t depth = 1>
gr::Graph make_merged_math_graph(T factor) {
    using chain = CascadeType<depth - 1, MathChain<T>>;
    gr::Graph g;
    auto&     src  = g.emplaceBlock<gr::testing::NullSource<T>>();
    auto&     proc = g.emplaceBlock<chain>({{"value", factor}});
    auto&     sink = g.emplaceBlock<gr::testing::CountingSink<T>>({{"n_samples_max", N_SAMPLES}});
    if (!g.connect<"out", "in">(src, proc).has_value()) {
        throw std::runtime_error("connection failed");
    }
    if (!g.connect<"out", "in">(proc, sink).has_value()) {
        throw std::runtime_error("connection failed");
    }
    return g;
}

inline const boost::ut::suite<"runtime benchmarks"> _runtime_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;

    auto run_bm = [](auto makeGraph, const char* name) {
        auto graph = makeGraph();
        printTopology(name, graph);
        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        ::benchmark::benchmark<1LU>{name}.repeat<N_ITER>(N_SAMPLES) = [&sched]() { expect(sched.runAndWait().has_value()); };
    };

    run_bm(make_src_sink_graph<float>, "runtime   src->sink");
    run_bm([] { return make_copy_chain_graph<float>(1); }, "runtime   src->copy->sink");
    run_bm([] { return make_copy_chain_graph<float>(10); }, "runtime   src->copy^10->sink");
    run_bm([] { return make_math_chain_graph<float>(2.0f); }, "runtime   src->mult->div->add->sink - float");
    run_bm([] { return make_math_chain_graph<int>(2); }, "runtime   src->mult->div->add->sink - int");
    run_bm([] { return make_math_chain_graph<float>(2.0f, 10); }, "runtime   src->(mult->div->add)^10->sink - float");
    run_bm([] { return make_math_chain_graph<int>(2, 10); }, "runtime   src->(mult->div->add)^10->sink - int");

    // runtime IIR: discrete feedback cycle at 1 sample/work() — single iteration only
    {
        auto graph = make_iir_lowpass_graph<float>();
        printTopology("runtime   IIR low-pass (feedback) - float", graph);
        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        ::benchmark::benchmark<1LU>{"runtime   IIR low-pass (feedback) - float"}.repeat<1>(N_SAMPLES_FEEDBACK) = [&sched]() { expect(sched.runAndWait().has_value()); };
    }

    ::benchmark::results::add_separator();
    run_bm([] { return make_bulk_chain_graph<float>(2.0f); }, "runtime   src->mult->div->add->sink - float (processBulk)");
    run_bm([] { return make_bulk_chain_graph<int>(2); }, "runtime   src->mult->div->add->sink - int (processBulk)");
};

inline const boost::ut::suite<"merged benchmarks (scheduler)"> _merge_api_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;
    ::benchmark::results::add_separator();

    auto run_bm = [](auto makeGraph, const char* name) {
        auto graph = makeGraph();
        printTopology(name, graph);
        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        ::benchmark::benchmark<1LU>{name}.repeat<N_ITER>(N_SAMPLES) = [&sched]() { expect(sched.runAndWait().has_value()); };
    };

    run_bm(make_merged_src_sink_graph<float>, "merged    src->sink");
    run_bm(make_merged_copy_graph<float, 10>, "merged    src->copy->sink");
    run_bm(make_merged_copy_graph<float, 10>, "merged    src->copy^10->sink");
    run_bm([] { return make_merged_math_graph<float, 1>(2.0f); }, "merged    src->mult->div->add->sink - float");
    run_bm([] { return make_merged_math_graph<int, 1>(2); }, "merged    src->mult->div->add->sink - int");
    run_bm([] { return make_merged_math_graph<float, 10>(2.0f); }, "merged    src->(mult->div->add)^10->sink - float");
    run_bm([] { return make_merged_math_graph<int, 10>(2); }, "merged    src->(mult->div->add)^10->sink - int");
    run_bm(make_merged_iir_graph<float>, "merged    IIR low-pass (FeedbackMerge) - float");

    ::benchmark::results::add_separator();
    run_bm(make_merged_split_merge_graph<float>, "merged    SplitMergeCombine src->{s(2),s(3)}->sink - float");
    run_bm(make_merged_iir_split_merge_graph<float>, "merged    IIR low-pass (SplitMergeCombine feedback) - float");
};

inline const boost::ut::suite<"constexpr benchmarks (direct processOne)"> _constexpr_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;
    using namespace gr;
    using gr::Merge;
    ::benchmark::results::add_separator();

    {
        auto merged = Merge<gr::testing::NullSource<float>, "out", FastSink<float>, "in">();
        printTopology("constexpr src->sink", merged.graph());
        "constexpr src->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&]() { loop_over_processOne(merged, N_SAMPLES); };
    }

    {
        auto merged = Merge<                                                        //
            Merge<gr::testing::NullSource<float>, "out", copy<float>, "in">, "out", //
            FastSink<float>, "in">();
        printTopology("constexpr src->copy->sink", merged.graph());
#if !DISABLE_SIMD
        static_assert(gr::traits::block::can_processOne_simd<copy<float>>);
        static_assert(gr::traits::block::can_processOne_simd<FastSink<float>>);
#endif
        "constexpr src->copy->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&]() { loop_over_processOne(merged, N_SAMPLES); };
    }

    {
        auto merged = Merge<                                                                        //
            Merge<gr::testing::NullSource<float>, "out", CascadeType<9, copy<float>>, "in">, "out", //
            FastSink<float>, "in">();
        printTopology("constexpr src->copy^10->sink", merged.graph());
        "constexpr src->copy^10->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&]() { loop_over_processOne(merged, N_SAMPLES); };
    }

    constexpr auto cascaded_test = []<typename T>(T factor, const char* test_name) {
        auto merged = Merge<                                                     //
            Merge<gr::testing::NullSource<T>, "out", MathChain<T>, "in">, "out", //
            FastSink<T>, "in">({{"value", factor}});
        printTopology(test_name, merged.graph());

        ::benchmark::benchmark<1LU>{test_name}.repeat<N_ITER>(N_SAMPLES) = [&]() { loop_over_processOne(merged, N_SAMPLES); };
    };
    cascaded_test(2.0f, "constexpr src->mult->div->add->sink - float");
    cascaded_test(2, "constexpr src->mult->div->add->sink - int");

    constexpr auto cascaded_test_10 = []<typename T>(T factor, const char* test_name) {
        auto merged = Merge<                                                                     //
            Merge<gr::testing::NullSource<T>, "out", CascadeType<9, MathChain<T>>, "in">, "out", //
            FastSink<T>, "in">({{"value", factor}});
        printTopology(test_name, merged.graph());

        ::benchmark::benchmark<1LU>{test_name}.repeat<N_ITER>(N_SAMPLES) = [&]() { loop_over_processOne(merged, N_SAMPLES); };
    };
    cascaded_test_10(2.0f, "constexpr src->(mult->div->add)^10->sink - float");
    cascaded_test_10(2, "constexpr src->(mult->div->add)^10->sink - int");

    {
        auto  merged                    = gr::Merge<                                                     //
            gr::Merge<gr::testing::NullSource<float>, "out", IIRChain, "in">, "out", //
            FastSink<float>, "in">();
        auto& iir                       = merged._leftBlock._rightBlock;
        iir._leftBlock.value            = kAlpha;
        iir._rightBlock._feedback.value = 1.0f - kAlpha;
        printTopology("constexpr IIR low-pass (FeedbackMerge)", merged.graph());
        "constexpr IIR low-pass (FeedbackMerge) - float"_benchmark.repeat<N_ITER>(N_SAMPLES_FEEDBACK) = [&]() { loop_over_processOne(merged, N_SAMPLES_FEEDBACK); };
    }

    ::benchmark::results::add_separator();

    {
        IIRReference ref;
        "constexpr IIR low-pass (scalar reference) - float"_benchmark.repeat<N_ITER>(N_SAMPLES_FEEDBACK) = [&ref]() {
            ref._state = 0.0f;
            for (gr::Size_t i = 0; i < N_SAMPLES_FEEDBACK; i++) {
                auto v = ref.processOne(0.0f);
                ::benchmark::force_to_memory(v);
            }
        };
    }

    {
        auto merged                                   = gr::Merge<                                                            //
            gr::Merge<gr::testing::NullSource<float>, "out", SplitMergeChain, "in">, "out", //
            FastSink<float>, "in">();
        merged._leftBlock._rightBlock.path<0>().value = 2.0f;
        merged._leftBlock._rightBlock.path<1>().value = 3.0f;
        printTopology("constexpr SplitMergeCombine src->{{scale(2),scale(3)}}->sink", merged.graph());
        "constexpr SplitMergeCombine src->{scale(2),scale(3)}->sink - float"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&]() { loop_over_processOne(merged, N_SAMPLES); };
    }

    {
        auto  merged                              = gr::Merge<                                                               //
            gr::Merge<gr::testing::NullSource<float>, "out", IIRChainSplitMerge, "in">, "out", //
            FastSink<float>, "in">();
        auto& iir                                 = merged._leftBlock._rightBlock;
        iir._leftBlock.value                      = kAlpha;
        iir._rightBlock._feedback.path<0>().value = 1.0f;
        iir._rightBlock._feedback.path<1>().value = -kAlpha;
        printTopology("constexpr IIR low-pass (SplitMergeCombine feedback)", merged.graph());
        "constexpr IIR low-pass (SplitMergeCombine feedback) - float"_benchmark.repeat<N_ITER>(N_SAMPLES_FEEDBACK) = [&]() { loop_over_processOne(merged, N_SAMPLES_FEEDBACK); };
    }
};

inline const boost::ut::suite<"Merge topology display"> _topology = [] {
    using namespace boost::ut;
    using namespace gr;

    auto iir                        = IIRChain();
    iir._leftBlock.value            = kAlpha;
    iir._rightBlock._feedback.value = 1.0f - kAlpha;
    printTopology("IIR chain (FeedbackMerge)", iir.graph());

    auto splitMerge            = SplitMergeChain();
    splitMerge.path<0>().value = 2.0f;
    splitMerge.path<1>().value = 3.0f;
    printTopology("SplitMergeChain", splitMerge.graph());

    auto iirSplitMerge                                  = IIRChainSplitMerge();
    iirSplitMerge._leftBlock.value                      = kAlpha;
    iirSplitMerge._rightBlock._feedback.path<0>().value = 1.0f;
    iirSplitMerge._rightBlock._feedback.path<1>().value = -kAlpha;
    printTopology("IIR SplitMerge chain", iirSplitMerge.graph());

    auto fullChain = gr::Merge<gr::testing::NullSource<float>, "out", IIRChainSplitMerge, "in">();
    printTopology("NullSource -> IIR SplitMerge (nested, flattened)", fullChain.graph());
};

int main() { /* not needed by the UT framework */ }
