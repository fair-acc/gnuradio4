#include <benchmark.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Profiler.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/math/Math.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using T          = float;
using TestMarker = benchmark::MarkerMap<"first-out", "last-out", "first-in", "last-in">;

inline constexpr std::size_t N_ITER        = 10; // TODO: identify/fix why scheduler `runAndWait()' doesn't re-initialises the connections properly for multi-threaded schedulers
inline constexpr gr::Size_t  N_SAMPLES     = gr::util::round_up(10'000'000, 1024);
inline constexpr std::size_t N_NODES       = 10;                                // the larger, the more pronounced the latency for non-critical-path aware scheduler
inline constexpr gr::Size_t  N_BUFFER_SIZE = gr::util::round_up(200'000, 1024); // the larger, the more pronounced the latency for non-critical-path aware scheduler

inline static TestMarker* _testMarker;

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
            expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(src, N_BUFFER_SIZE).template to<"in">(*mult1[i])));
        } else {
            expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(*mult2[i - 1], N_BUFFER_SIZE).template to<"in">(*mult1[i])));
        }
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(*mult1[i], N_BUFFER_SIZE).template to<"in">(*mult2[i])));
    }
    expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(*mult2[mult2.size() - 1], N_BUFFER_SIZE).template to<"in">(sink)));
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

void exec_bm(auto& scheduler, const std::string& test_case, [[maybe_unused]] TestMarker& testMarker, std::source_location srcLoc = std::source_location::current()) {
    using namespace benchmark;
    _testMarker    = &testMarker;
    const auto res = scheduler.runAndWait();
    boost::ut::expect(res.has_value(), srcLoc) << [&] { return std::format("scheduler failure for test-case: {}\n    - error: {}", test_case, res.error()); } << boost::ut::fatal;
}

[[maybe_unused]] inline const boost::ut::suite<"basic scheduler tests"> scheduler_tests = [] {
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
    TestMarker marker;

    gr::scheduler::Simple sched1(test_graph_linear<T>(2 * N_NODES));
    "linear graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1, &marker]() { exec_bm(sched1, "linear-graph simple-sched", marker); };

    gr::scheduler::BreadthFirst sched2(test_graph_linear<T>(2 * N_NODES));
    "linear graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2, &marker]() { exec_bm(sched2, "linear-graph BFS-sched", marker); };

    gr::scheduler::Simple sched3(test_graph_bifurcated<T>(N_NODES));
    "bifurcated graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3, &marker]() { exec_bm(sched3, "bifurcated-graph simple-sched", marker); };

    gr::scheduler::BreadthFirst sched4(test_graph_bifurcated<T>(N_NODES));
    "bifurcated graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4, &marker]() { exec_bm(sched4, "bifurcated-graph BFS-sched", marker); };

    gr::scheduler::Simple<multiThreaded> sched1_mt(test_graph_linear<T>(2 * N_NODES));
    "linear graph - simple scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1_mt, &marker]() { exec_bm(sched1_mt, "linear-graph simple-sched (multi-threaded)", marker); };

    gr::scheduler::BreadthFirst<multiThreaded> sched2_mt(test_graph_linear<T>(2 * N_NODES));
    "linear graph - BFS scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2_mt, &marker]() { exec_bm(sched2_mt, "linear-graph BFS-sched (multi-threaded)", marker); };

    gr::scheduler::Simple<multiThreaded> sched3_mt(test_graph_bifurcated<T>(N_NODES));
    "bifurcated graph - simple scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3_mt, &marker]() { exec_bm(sched3_mt, "bifurcated-graph simple-sched (multi-threaded)", marker); };

    gr::scheduler::BreadthFirst<multiThreaded> sched4_mt(test_graph_bifurcated<T>(N_NODES));
    "bifurcated graph - BFS scheduler (multi-threaded)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4_mt, &marker]() { exec_bm(sched4_mt, "bifurcated-graph BFS-sched (multi-threaded)", marker); };

    gr::scheduler::BreadthFirst<multiThreaded, Profiler> sched4_mt_prof(test_graph_bifurcated<T>(N_NODES));
    "bifurcated graph - BFS scheduler (multi-threaded) with profiling"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4_mt_prof, &marker]() { exec_bm(sched4_mt_prof, "bifurcated-graph BFS-sched (multi-threaded) with profiling", marker); };

    ::benchmark::results::add_separator();
};

template<typename T>
auto& createSource(gr::Graph& graph) {
    using namespace gr::testing;
    auto& src = graph.emplaceBlock<TagSource<T, ProcessFunction::USE_PROCESS_BULK>>({{"name", "source"}, {"n_samples_max", N_SAMPLES}});
    src._tags = {
        {0UZ, {{gr::tag::TRIGGER_NAME.shortKey(), "first"}, {gr::tag::TRIGGER_TIME.shortKey(), static_cast<uint64_t>(0)}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}}},             //
        {(N_SAMPLES - 1UZ), {{gr::tag::TRIGGER_NAME.shortKey(), "last"}, {gr::tag::TRIGGER_TIME.shortKey(), static_cast<uint64_t>(0)}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}}} //
    };
    src._tagCallback = [](const gr::Tag& tag) {
        std::string triggerName = std::get<std::string>(tag.map.at(gr::tag::TRIGGER_NAME.shortKey()));
        if (triggerName == "first") {
            _testMarker->at<"first-out">().now();
        } else if (triggerName == "last") {
            _testMarker->at<"last-out">().now();
        } else {
            throw gr::exception("unknown trigger name");
        }
    };
    return src;
}

template<typename T>
auto& createSink(gr::Graph& graph, bool instrumentalise = true) {
    using namespace gr::testing;
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>({{"name", "sink"}});
    if (!instrumentalise) {
        return sink;
    }
    sink._tagCallback = [=](const gr::Tag& tag) {
        std::string triggerName = std::get<std::string>(tag.map.at(gr::tag::TRIGGER_NAME.shortKey()));
        if (triggerName == "first") {
            _testMarker->at<"first-in">().now();
        } else if (triggerName == "last") {
            _testMarker->at<"last-in">().now();
        } else {
            throw gr::exception("unknown trigger name");
        }
    };
    return sink;
}

enum class GraphTopology { DEFAULT, LINEAR, FORKED, SPLIT };

void printGraphTopology(const gr::Graph& graph, GraphTopology topology) {
    std::println("Graph Topology: {}:", magic_enum::enum_name(topology));
    for (const auto& block : graph.blocks()) {
        std::println("  - block: {}", block->name());
    }
    for (const auto& edge : graph.edges()) {
        std::println("  - edge: {}", edge);
    }
}

template<typename T>
gr::Graph createInstrumentalisedGraph(GraphTopology topology = GraphTopology::LINEAR) {
    using namespace boost::ut;
    using namespace gr::testing;

    gr::Graph graph;

    switch (topology) {
    case GraphTopology::LINEAR: { // creates a linear sequence of nodes -- deliberately in reverse order
        std::size_t    depth = 4UZ;
        auto&          sink  = createSink<T>(graph);
        SimCompute<T>* lastBlock;
        for (std::size_t i = 0UZ; i < depth; i++) {
            std::string blockName = std::format("sim{}", depth - i);
            auto&       simBlock  = graph.emplaceBlock<SimCompute<T>>({{"name", blockName}});
            if (i == 0UZ) {
                expect(eq(graph.connect(simBlock, "out"s, sink, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
            } else {
                expect(eq(graph.connect(simBlock, "out"s, *lastBlock, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
            }
            lastBlock = &simBlock;
        }
        auto& src = createSource<T>(graph);
        expect(eq(graph.connect(src, "out"s, *lastBlock, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
    } break;
    case GraphTopology::FORKED: { // deliberately 4 sim blocks to mimic the total time of the linear test-case
        auto& src = createSource<T>(graph);
        // branch #1
        auto& simBlock1 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim1"}});
        expect(eq(graph.connect(src, "out"s, simBlock1, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        // branch #2
        auto& simBlock2 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim2"}});
        expect(eq(graph.connect(src, "out"s, simBlock2, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        auto& simBlock3 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim3"}});
        expect(eq(graph.connect(simBlock2, "out"s, simBlock3, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        // merge branch #1 & #2
        auto& adder = graph.emplaceBlock<gr::blocks::math::Add<T>>({{"name", "adder"}, {"n_inputs", 2U}});
        expect(eq(graph.connect(simBlock1, "out"s, adder, "in#0"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        expect(eq(graph.connect(simBlock3, "out"s, adder, "in#1"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));

        auto& simBlock4 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim4"}});
        expect(eq(graph.connect(adder, "out"s, simBlock4, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));

        // final sink
        auto& sink = createSink<T>(graph);
        expect(eq(graph.connect(simBlock4, "out"s, sink, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
    } break;
    case GraphTopology::SPLIT: { // deliberately 4 sim blocks to mimic the total time of the linear test-case
        auto& src       = createSource<T>(graph);
        auto& simBlock1 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim1"}});
        expect(eq(graph.connect(src, "out"s, simBlock1, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        auto& simBlock2 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim2"}});
        expect(eq(graph.connect(simBlock1, "out"s, simBlock2, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        auto& simBlock3 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim3"}});
        expect(eq(graph.connect(simBlock2, "out"s, simBlock3, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        auto& simBlock4 = graph.emplaceBlock<SimCompute<T>>({{"name", "sim4"}});
        expect(eq(graph.connect(simBlock3, "out"s, simBlock4, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));

        auto& sink1 = createSink<T>(graph, false);
        expect(eq(graph.connect(simBlock2, "out"s, sink1, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
        auto& sink2 = createSink<T>(graph, true);
        expect(eq(graph.connect(simBlock4, "out"s, sink2, "in"s, N_BUFFER_SIZE), gr::ConnectionResult::SUCCESS));
    } break;
    default: {
        create_cascade<T>(graph, createSource<T>(graph), createSink<T>(graph), 3UZ);
    }
    }

    return graph;
}

[[maybe_unused]] inline const boost::ut::suite<"scheduler topology tests"> _timed_scheduler_tests = [] {
    using namespace gr::profiling;
    using namespace boost::ut;
    using namespace benchmark;

    "scheduler topology loop"_test = [](GraphTopology topology) {
        const std::string topologyName(magic_enum::enum_name(topology));

        gr::scheduler::Simple simple(createInstrumentalisedGraph<T>(topology));
        printGraphTopology(simple.graph(), topology);
        ::benchmark::benchmark(std::format("Simple scheduler - unlimited work - {}", topologyName)).repeat<N_ITER>(N_SAMPLES) = [&simple](TestMarker& marker) { exec_bm(simple, "test case #1", marker); };

        gr::scheduler::DepthFirst depthFirstSched1(createInstrumentalisedGraph<T>(topology));
        ::benchmark::benchmark(std::format("DFS scheduler - unlimited work - {}", topologyName)).repeat<N_ITER>(N_SAMPLES) = [&depthFirstSched1](TestMarker& marker) { exec_bm(depthFirstSched1, "test case #1", marker); };

        gr::scheduler::BreadthFirst breathFirstSched1(createInstrumentalisedGraph<T>(topology));
        ::benchmark::benchmark(std::format("BFS scheduler - unlimited work - {}", topologyName)).repeat<N_ITER>(N_SAMPLES) = [&breathFirstSched1](TestMarker& marker) { exec_bm(breathFirstSched1, "test case #2", marker); };

        gr::scheduler::DepthFirst depthFirstSched2(createInstrumentalisedGraph<T>(topology), {{"max_work_items", 1024UZ}});
        ::benchmark::benchmark(std::format("DFS scheduler - work limited to 1024 - {}", topologyName)).repeat<N_ITER>(N_SAMPLES) = [&depthFirstSched2](TestMarker& marker) { exec_bm(depthFirstSched2, "test case #2", marker); };

        gr::scheduler::BreadthFirst breathFirstSched2(createInstrumentalisedGraph<T>(topology), {{"max_work_items", 1024UZ}});
        ::benchmark::benchmark(std::format("BFS scheduler - work limited to 1024 - {}", topologyName)).repeat<N_ITER>(N_SAMPLES) = [&breathFirstSched2](TestMarker& marker) { exec_bm(breathFirstSched2, "test case #2", marker); };

        ::benchmark::results::add_separator();
    } | std::vector{GraphTopology::LINEAR, GraphTopology::FORKED, GraphTopology::SPLIT};
};

int main() { /* not needed by the UT framework */ }
