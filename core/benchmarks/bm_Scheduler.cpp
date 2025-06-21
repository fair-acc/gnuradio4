#include <benchmark.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Profiler.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/math/Math.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

inline constexpr std::size_t N_ITER    = 10;
inline constexpr gr::Size_t  N_SAMPLES = gr::util::round_up(10'000'000, 1024);
inline constexpr std::size_t N_NODES   = 5;

template<typename T, typename Sink, typename Source>
void create_cascade(gr::Graph& testGraph, Sink& src, Source& sink, std::size_t depth = 1) {
    using namespace boost::ut;
    using namespace benchmark;
    using namespace gr::blocks::math;

    std::vector<MultiplyConst<T>*> mult1;
    std::vector<DivideConst<T>*>   mult2;
    for (std::size_t i = 0; i < depth; i++) {
        mult1.emplace_back(std::addressof(testGraph.emplaceBlock<MultiplyConst<T>>({{"value", T(2)}, {"name", std::format("mult.{}", i)}})));
        mult2.emplace_back(std::addressof(testGraph.emplaceBlock<DivideConst<T>>({{"value", T(2)}, {"name", std::format("div.{}", i)}})));
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
gr::Graph test_graph_linear(std::size_t depth = 1) {
    gr::Graph testGraph;

    auto& src  = testGraph.emplaceBlock<gr::testing::ConstantSource<T>>({{"n_samples_max", N_SAMPLES}});
    auto& sink = testGraph.emplaceBlock<gr::testing::NullSink<T>>();

    create_cascade<T>(testGraph, src, sink, depth);

    return testGraph;
}

template<typename T>
gr::Graph test_graph_bifurcated(std::size_t depth = 1) {
    using namespace boost::ut;
    using namespace benchmark;
    gr::Graph testGraph;

    auto& src   = testGraph.emplaceBlock<gr::testing::ConstantSource<T>>({{"n_samples_max", N_SAMPLES}});
    auto& sink1 = testGraph.emplaceBlock<gr::testing::NullSink<T>>();
    auto& sink2 = testGraph.emplaceBlock<gr::testing::NullSink<T>>();

    create_cascade<T>(testGraph, src, sink1, depth);
    create_cascade<T>(testGraph, src, sink2, depth);

    return testGraph;
}

void exec_bm(auto& scheduler, const std::string& test_case) {
    using namespace boost::ut;
    using namespace benchmark;
    expect(scheduler.runAndWait().has_value()) << std::format("scheduler failure for test-case: {}", test_case);
}

[[maybe_unused]] inline const boost::ut::suite scheduler_tests = [] {
    using namespace gr::profiling;
    using namespace boost::ut;
    using namespace benchmark;
    using gr::scheduler::ExecutionPolicy::multiThreaded;

    using namespace gr::thread_pool;
    auto cpu = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>(std::string(kDefaultCpuPoolId), TaskType::CPU_BOUND, 2U, 2U), "CPU");
    gr::thread_pool::Manager::instance().replacePool(std::string(kDefaultCpuPoolId), std::move(cpu));
    const auto minThreads = gr::thread_pool::Manager::defaultCpuPool()->minThreads();
    const auto maxThreads = gr::thread_pool::Manager::defaultCpuPool()->maxThreads();
    std::println("INFO: std::thread::hardware_concurrency() = {} - CPU thread bounds = [{}, {}]", std::thread::hardware_concurrency(), minThreads, maxThreads);

    gr::scheduler::Simple sched1(test_graph_linear<float>(2 * N_NODES));
    "linear graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1]() { exec_bm(sched1, "linear-graph simple-sched"); };

    gr::scheduler::BreadthFirst sched2(test_graph_linear<float>(2 * N_NODES));
    "linear graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2]() { exec_bm(sched2, "linear-graph BFS-sched"); };

    gr::scheduler::Simple sched3(test_graph_bifurcated<float>(N_NODES));
    "bifurcated graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3]() { exec_bm(sched3, "bifurcated-graph simple-sched"); };

    gr::scheduler::BreadthFirst sched4(test_graph_bifurcated<float>(N_NODES));
    "bifurcated graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4]() { exec_bm(sched4, "bifurcated-graph BFS-sched"); };

    gr::scheduler::Simple<multiThreaded> sched1_mt(test_graph_linear<float>(2 * N_NODES));
    "linear graph - simple scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1_mt]() { exec_bm(sched1_mt, "linear-graph simple-sched (multi-threaded)"); };

    gr::scheduler::BreadthFirst<multiThreaded> sched2_mt(test_graph_linear<float>(2 * N_NODES));
    "linear graph - BFS scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2_mt]() { exec_bm(sched2_mt, "linear-graph BFS-sched (multi-threaded)"); };

    gr::scheduler::Simple<multiThreaded> sched3_mt(test_graph_bifurcated<float>(N_NODES));
    "bifurcated graph - simple scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3_mt]() { exec_bm(sched3_mt, "bifurcated-graph simple-sched (multi-threaded)"); };

    gr::scheduler::BreadthFirst<multiThreaded> sched4_mt(test_graph_bifurcated<float>(N_NODES));
    "bifurcated graph - BFS scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4_mt]() { exec_bm(sched4_mt, "bifurcated-graph BFS-sched (multi-threaded)"); };

    gr::scheduler::BreadthFirst<multiThreaded, Profiler> sched4_mt_prof(test_graph_bifurcated<float>(N_NODES));
    "bifurcated graph - BFS scheduler (multi-threaded) with profiling"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4_mt_prof]() { exec_bm(sched4_mt_prof, "bifurcated-graph BFS-sched (multi-threaded) with profiling"); };
};

int main() { /* not needed by the UT framework */ }
