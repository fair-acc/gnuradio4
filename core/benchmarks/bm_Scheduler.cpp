#include <benchmark.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Profiler.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/testing/bm_test_helper.hpp>

inline constexpr std::size_t N_ITER    = 10;
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(10'000'000, 1024);
inline constexpr std::size_t N_NODES   = 5;

template<typename T, char op>
class math_op : public gr::Block<math_op<T, op>, gr::PortInNamed<T, "in">, gr::PortOutNamed<T, "out">> {
    T _factor = static_cast<T>(1.0f);

public:
    math_op() = delete;

    explicit math_op(T factor, std::string name_ = gr::this_source_location()) : _factor(factor) { this->name = name_; }

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(const V &a) const noexcept {
        if constexpr (op == '*') {
            return a * _factor;
        } else if constexpr (op == '/') {
            return a / _factor;
        } else if constexpr (op == '+') {
            return a + _factor;
        } else if constexpr (op == '-') {
            return a - _factor;
        } else {
            static_assert(gr::meta::always_false<T>, "unknown op");
        }
    }
};

template<typename T>
using multiply = math_op<T, '*'>;
template<typename T>
using divide = math_op<T, '/'>;

template<typename T, typename Sink, typename Source>
void
create_cascade(gr::Graph &testGraph, Sink &src, Source &sink, std::size_t depth = 1) {
    using namespace boost::ut;
    using namespace benchmark;

    std::vector<multiply<T> *> mult1;
    std::vector<divide<T> *>   mult2;
    for (std::size_t i = 0; i < depth; i++) {
        mult1.emplace_back(std::addressof(testGraph.emplaceBlock<multiply<T>>(T(2), fmt::format("mult.{}", i))));
        mult2.emplace_back(std::addressof(testGraph.emplaceBlock<divide<T>>(T(2), fmt::format("div.{}", i))));
    }

    for (std::size_t i = 0; i < mult1.size(); i++) {
        if (i == 0) {
            expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(src).template to<"in">(*mult1[i])));
        } else {
            expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(*mult2[i - 1]).template to<"in">(*mult1[i])));
        }
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(*mult1[i]).template to<"in">(*mult2[i])));
    }
    expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(*mult2[mult2.size() - 1]).template to<"in">(sink)));
}

template<typename T>
gr::Graph
test_graph_linear(std::size_t depth = 1) {
    gr::Graph testGraph;

    auto     &src  = testGraph.emplaceBlock<test::source<T>>(N_SAMPLES);
    auto     &sink = testGraph.emplaceBlock<test::sink<T>>();

    create_cascade<T>(testGraph, src, sink, depth);

    return testGraph;
}

template<typename T>
gr::Graph
test_graph_bifurcated(std::size_t depth = 1) {
    using namespace boost::ut;
    using namespace benchmark;
    gr::Graph testGraph;

    auto     &src   = testGraph.emplaceBlock<test::source<T>>(N_SAMPLES);
    auto     &sink1 = testGraph.emplaceBlock<test::sink<T>>();
    auto     &sink2 = testGraph.emplaceBlock<test::sink<T>>();

    create_cascade<T>(testGraph, src, sink1, depth);
    create_cascade<T>(testGraph, src, sink2, depth);

    return testGraph;
}

void
exec_bm(auto &scheduler, const std::string &test_case) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
    scheduler.runAndWait();
    expect(eq(test::n_samples_produced, N_SAMPLES)) << fmt::format("did not produce enough output samples for {}", test_case);
    expect(ge(test::n_samples_consumed, N_SAMPLES)) << fmt::format("did not consume enough input samples for {}", test_case);
}

[[maybe_unused]] inline const boost::ut::suite scheduler_tests = [] {
    using namespace gr::profiling;
    using namespace boost::ut;
    using namespace benchmark;
    using thread_pool = gr::thread_pool::BasicThreadPool;
    using gr::scheduler::ExecutionPolicy::multiThreaded;

    auto                  pool = std::make_shared<thread_pool>("custom-pool", gr::thread_pool::CPU_BOUND, 2, 2);

    gr::scheduler::Simple sched1(test_graph_linear<float>(2 * N_NODES), pool);
    "linear graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1]() { exec_bm(sched1, "linear-graph simple-sched"); };

    gr::scheduler::BreadthFirst sched2(test_graph_linear<float>(2 * N_NODES), pool);
    "linear graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2]() { exec_bm(sched2, "linear-graph BFS-sched"); };

    gr::scheduler::Simple sched3(test_graph_bifurcated<float>(N_NODES), pool);
    "bifurcated graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3]() { exec_bm(sched3, "bifurcated-graph simple-sched"); };

    gr::scheduler::BreadthFirst sched4(test_graph_bifurcated<float>(N_NODES), pool);
    "bifurcated graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4]() { exec_bm(sched4, "bifurcated-graph BFS-sched"); };

    gr::scheduler::Simple<multiThreaded> sched1_mt(test_graph_linear<float>(2 * N_NODES), pool);
    "linear graph - simple scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1_mt]() { exec_bm(sched1_mt, "linear-graph simple-sched (multi-threaded)"); };

    gr::scheduler::BreadthFirst<multiThreaded> sched2_mt(test_graph_linear<float>(2 * N_NODES), pool);
    "linear graph - BFS scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2_mt]() { exec_bm(sched2_mt, "linear-graph BFS-sched (multi-threaded)"); };

    gr::scheduler::Simple<multiThreaded> sched3_mt(test_graph_bifurcated<float>(N_NODES), pool);
    "bifurcated graph - simple scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3_mt]() { exec_bm(sched3_mt, "bifurcated-graph simple-sched (multi-threaded)"); };

    gr::scheduler::BreadthFirst<multiThreaded> sched4_mt(test_graph_bifurcated<float>(N_NODES), pool);
    "bifurcated graph - BFS scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4_mt]() { exec_bm(sched4_mt, "bifurcated-graph BFS-sched (multi-threaded)"); };

    gr::scheduler::BreadthFirst<multiThreaded, Profiler> sched4_mt_prof(test_graph_bifurcated<float>(N_NODES), pool);
    "bifurcated graph - BFS scheduler (multi-threaded) with profiling"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4_mt_prof]() {
        exec_bm(sched4_mt_prof, "bifurcated-graph BFS-sched (multi-threaded) with profiling");
    };
};

int
main() { /* not needed by the UT framework */
}
