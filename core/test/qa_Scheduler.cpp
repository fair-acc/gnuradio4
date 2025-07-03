#include "message_utils.hpp"
#include <boost/ut.hpp>

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <chrono>

using TraceVectorType = std::vector<std::string>;

class Tracer {
    std::mutex      _traceMutex;
    TraceVectorType _traceVector;

public:
    void trace(std::string_view id) {
        std::scoped_lock lock{_traceMutex};
        if (_traceVector.empty() || _traceVector.back() != id) {
            _traceVector.emplace_back(id);
        }
    }

    TraceVectorType getVector() {
        std::scoped_lock lock{_traceMutex};
        return {_traceVector};
    }
};

// define some example graph nodes
template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T> out;
    gr::Size_t     n_samples_max = 0;

    GR_MAKE_REFLECTABLE(CountSource, out, n_samples_max);

    std::shared_ptr<Tracer> tracer{};
    gr::Size_t              count = 0;

    ~CountSource() {
        if (count != n_samples_max) {
            std::println(stderr, "Error: CountSource did not process expected number of samples: {} vs. {}", count, n_samples_max);
        }
    }

    constexpr T processOne() {
        count++;
        if (count >= n_samples_max) {
            this->requestStop();
        }
        tracer->trace(this->name);
        return static_cast<int>(count);
    }
};

static_assert(gr::BlockLike<CountSource<float>>);

template<typename T>
struct ExpectSink : public gr::Block<ExpectSink<T>> {
    gr::PortIn<T> in;
    gr::Size_t    n_samples_max = 0;

    GR_MAKE_REFLECTABLE(ExpectSink, in, n_samples_max);

    std::shared_ptr<Tracer>                         tracer{};
    gr::Size_t                                      count       = 0;
    gr::Size_t                                      false_count = 0;
    std::function<bool(std::int64_t, std::int64_t)> checker;

    ~ExpectSink() { // TODO: throwing exceptions in destructor is bad -> need to refactor test
        if (count != n_samples_max) {
            std::println(stderr, "Error: ExpectSink did not process expected number of samples: {} vs. {}", count, n_samples_max);
        }
        if (false_count != 0) {
            std::println(stderr, "Error: ExpectSink false count {} is not zero", false_count);
        }
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T>& input) noexcept {
        tracer->trace(this->name);
        for (auto data : input) {
            count++;
            if (!checker(count, data)) {
                false_count++;
            };
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct Scale : public gr::Block<Scale<T>> {
    using R = decltype(std::declval<T>() * std::declval<T>());
    gr::PortIn<T>           original;
    gr::PortOut<R>          scaled;
    std::shared_ptr<Tracer> tracer{};
    T                       scale_factor = T(1.);

    GR_MAKE_REFLECTABLE(Scale, original, scaled, scale_factor);

    [[nodiscard]] constexpr auto processOne(T a) noexcept {
        tracer->trace(this->name);
        return a * scale_factor;
    }
};

template<typename T>
struct Adder : public gr::Block<Adder<T>> {
    using R = decltype(std::declval<T>() + std::declval<T>());
    gr::PortIn<T>  addend0;
    gr::PortIn<T>  addend1;
    gr::PortOut<R> sum;

    GR_MAKE_REFLECTABLE(Adder, addend0, addend1, sum);

    std::shared_ptr<Tracer> tracer;

    [[nodiscard]] constexpr auto processOne(T a, T b) noexcept {
        tracer->trace(this->name);
        return a + b;
    }
};

gr::Graph getGraphLinear(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    gr::Size_t nMaxSamples{100000};

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto& source1      = flow.emplaceBlock<CountSource<int>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer     = tracer;
    auto& scaleBlock1  = flow.emplaceBlock<Scale<int>>({{"name", "mult1"}, {"scale_factor", 2}});
    scaleBlock1.tracer = tracer;
    auto& scaleBlock2  = flow.emplaceBlock<Scale<int>>({{"name", "mult2"}, {"scale_factor", 4}});
    scaleBlock2.tracer = tracer;
    auto& sink         = flow.emplaceBlock<ExpectSink<int>>({{"name", "out"}, {"n_samples_max", nMaxSamples}});
    sink.tracer        = tracer;
    sink.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 8 * count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock2).to<"in">(sink)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock1).to<"original">(scaleBlock2)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source1).to<"original">(scaleBlock1)));

    return flow;
}

gr::Graph getGraphParallel(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    gr::Size_t nMaxSamples{100000};

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto& source1       = flow.emplaceBlock<CountSource<int>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer      = tracer;
    auto& scaleBlock1a  = flow.emplaceBlock<Scale<int>>({{"name", "mult1a"}, {"scale_factor", 2}});
    scaleBlock1a.tracer = tracer;
    auto& scaleBlock2a  = flow.emplaceBlock<Scale<int>>({{"name", "mult2a"}, {"scale_factor", 3}});
    scaleBlock2a.tracer = tracer;
    auto& sinkA         = flow.emplaceBlock<ExpectSink<int>>({{"name", "outa"}, {"n_samples_max", nMaxSamples}});
    sinkA.tracer        = tracer;
    sinkA.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 6 * count; };
    auto& scaleBlock1b  = flow.emplaceBlock<Scale<int>>({{"name", "mult1b"}, {"scale_factor", 3}});
    scaleBlock1b.tracer = tracer;
    auto& scaleBlock2b  = flow.emplaceBlock<Scale<int>>({{"name", "mult2b"}, {"scale_factor", 5}});
    scaleBlock2b.tracer = tracer;
    auto& sinkB         = flow.emplaceBlock<ExpectSink<int>>({{"name", "outb"}, {"n_samples_max", nMaxSamples}});
    sinkB.tracer        = tracer;
    sinkB.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 15 * count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock1a).to<"original">(scaleBlock2a)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock1b).to<"original">(scaleBlock2b)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock2b).to<"in">(sinkB)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source1).to<"original">(scaleBlock1a)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock2a).to<"in">(sinkA)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source1).to<"original">(scaleBlock1b)));

    return flow;
}

/**
 * sets up an example graph
 * ┌───────────┐
 * │           │        ┌───────────┐
 * │ SOURCE    ├───┐    │           │
 * │           │   └────┤   x 2     ├───┐
 * └───────────┘        │           │   │    ┌───────────┐     ┌───────────┐
 *                      └───────────┘   └───►│           │     │           │
 *                                           │  SUM      ├────►│ PRINT     │
 *                           ┌──────────────►│           │     │           │
 * ┌───────────┬             ┤               └───────────┘     └───────────┘
 * │           │             │
 * │  SOURCE   ├─────────────┘
 * │           │
 * └───────────┘
 */
gr::Graph getGraphScaledSum(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location()) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    gr::Size_t nMaxSamples{100000};

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;

    // Generators
    auto& source1     = flow.emplaceBlock<CountSource<int>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer    = tracer;
    auto& source2     = flow.emplaceBlock<CountSource<int>>({{"name", "s2"}, {"n_samples_max", nMaxSamples}});
    source2.tracer    = tracer;
    auto& scaleBlock  = flow.emplaceBlock<Scale<int>>({{"name", "mult"}, {"scale_factor", 2}});
    scaleBlock.tracer = tracer;
    auto& addBlock    = flow.emplaceBlock<Adder<int>>({{"name", "add"}});
    addBlock.tracer   = tracer;
    auto& sink        = flow.emplaceBlock<ExpectSink<int>>({{"name", "out"}, {"n_samples_max", nMaxSamples}});
    sink.tracer       = tracer;
    sink.checker      = [](std::uint64_t count, std::uint64_t data) -> bool { return data == (2 * count) + count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source1).to<"original">(scaleBlock)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock).to<"addend0">(addBlock)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source2).to<"addend1">(addBlock)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"sum">(addBlock).to<"in">(sink)), loc);

    return flow;
}

template<typename TBlock>
void checkBlockNames(const std::vector<TBlock>& joblist, std::set<std::string> set, std::source_location loc = std::source_location()) {
    boost::ut::expect(boost::ut::that % joblist.size() == set.size(), loc);
    for (auto& block : joblist) {
        boost::ut::expect(boost::ut::that % set.contains(std::string(block->name())), loc) << std::format("{} not in {{{}}}\n", block->name(), gr::join(set));
    }
}

template<typename T>
struct LifecycleSource : public gr::Block<LifecycleSource<T>> {
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(LifecycleSource, out);

    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 10;

    [[nodiscard]] constexpr T processOne() noexcept {
        n_samples_produced++;
        if (n_samples_produced >= n_samples_max) {
            this->requestStop();
            return T(n_samples_produced); // this sample will be the last emitted.
        }
        return T(n_samples_produced);
    }
};

template<typename T>
struct LifecycleBlock : public gr::Block<LifecycleBlock<T>> {
    gr::PortIn<T>                in{};
    gr::PortOut<T, gr::Optional> out{};

    GR_MAKE_REFLECTABLE(LifecycleBlock, in, out);

    int process_one_count{};
    int start_count{};
    int stop_count{};
    int reset_count{};
    int pause_count{};
    int resume_count{};

    [[nodiscard]] constexpr T processOne(T a) noexcept {
        process_one_count++;
        return a;
    }

    void start() { start_count++; }

    void stop() { stop_count++; }

    void reset() { reset_count++; }

    void pause() { pause_count++; }

    void resume() { resume_count++; }
};

template<typename T>
struct BusyLoopBlock : public gr::Block<BusyLoopBlock<T>> {
    using enum gr::work::Status;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(BusyLoopBlock, in, out);

    gr::Sequence _produceCount{0};
    gr::Sequence _invokeCount{0};

    [[nodiscard]] constexpr gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) noexcept {
        auto produceCount = _produceCount.value();
        _invokeCount.incrementAndGet();

        if (produceCount == 0) {
            // early return by not explicitly consuming/producing but returning incomplete state
            // normally this should be reserved for "starving" blocks. Here, it's being used to unit-test this alternative behaviour
            return gr::lifecycle::isActive(this->state()) ? INSUFFICIENT_OUTPUT_ITEMS : DONE;
        }

        std::println("##BusyLoopBlock produces data _invokeCount: {}", _invokeCount.value());
        std::ranges::copy(input.begin(), input.end(), output.begin());
        produceCount = _produceCount.subAndGet(1L);
        return OK;
    }
};

const boost::ut::suite<"SchedulerTests"> SchedulerSettingsTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "Direct settings change"_test = [] {
        using scheduler               = gr::scheduler::Simple<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphLinear(trace), gr::thread_pool::kDefaultCpuPoolId};

        auto ret1 = sched.settings().set({{"timeout_ms", gr::Size_t(6)}});
        expect(ret1.empty()) << "setting one known parameter";
        expect(sched.settings().stagedParameters().empty());          // set(...) does not change stagedParameters
        expect(not sched.settings().changed()) << "settings changed"; // set(...) does not change changed()
        std::ignore = sched.settings().activateContext();

        std::println("Staged {}", sched.settings().stagedParameters());
        expect(sched.settings().stagedParameters().contains("timeout_ms"));

        expect(sched.settings().changed()) << "settings changed";
        std::ignore = sched.settings().applyStagedParameters();

        expect(eq(sched.timeout_ms.value, 6U));

        sched.settings().updateActiveParameters();

        auto ret2 = sched.settings().set({{"timeout_ms", gr::Size_t(42)}});
        expect(ret2.empty()) << "setting one known parameter";
        expect(sched.settings().stagedParameters().empty());          // set(...) does not change stagedParameters
        expect(not sched.settings().changed()) << "settings changed"; // set(...) does not change changed()
        std::ignore = sched.settings().activateContext();

        std::println("Staged {}", sched.settings().stagedParameters());
        expect(sched.settings().stagedParameters().contains("timeout_ms"));

        expect(sched.settings().changed()) << "settings changed";
        std::ignore = sched.settings().applyStagedParameters();

        expect(eq(sched.timeout_ms.value, 42U));

        sched.settings().updateActiveParameters();
    };
};

const boost::ut::suite<"SchedulerTests"> SchedulerTests = [] {
    using namespace std::chrono_literals;
    using namespace boost::ut;
    using namespace gr;

    // needs to be exceptionally pinned to [2, 2] min/max thread count of unit-test
    using namespace gr::thread_pool;
    auto cpu = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>(std::string(kDefaultCpuPoolId), TaskType::CPU_BOUND, 2U, 2U), "CPU");
    gr::thread_pool::Manager::instance().replacePool(std::string(kDefaultCpuPoolId), std::move(cpu));
    const auto minThreads = gr::thread_pool::Manager::defaultCpuPool()->minThreads();
    const auto maxThreads = gr::thread_pool::Manager::defaultCpuPool()->maxThreads();
    std::println("INFO: std::thread::hardware_concurrency() = {} - CPU thread bounds = [{}, {}]", std::thread::hardware_concurrency(), minThreads, maxThreads);

    "SimpleScheduler_linear"_test = [] {
        using scheduler               = gr::scheduler::Simple<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphLinear(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out"});
    };

    "BreadthFirstScheduler_linear"_test = [] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphLinear(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out"});
    };

    "SimpleScheduler_parallel"_test = [] {
        using scheduler               = gr::scheduler::Simple<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphParallel(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb", "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb"});
    };

    "BreadthFirstScheduler_parallel"_test = [] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphParallel(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == TraceVectorType{
                                          "s1",
                                          "mult1a",
                                          "mult1b",
                                          "mult2a",
                                          "mult2b",
                                          "outa",
                                          "outb",
                                          "s1",
                                          "mult1a",
                                          "mult1b",
                                          "mult2a",
                                          "mult2b",
                                          "outa",
                                          "outb",
                                      });
    };

    "SimpleScheduler_scaled_sum"_test = [] {
        using scheduler = gr::scheduler::Simple<>;
        // construct an example graph and get an adjacency list for it
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphScaledSum(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out"});
    };

    "BreadthFirstScheduler_scaled_sum"_test = [] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphScaledSum(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out"});
    };

    "SimpleScheduler_linear_multi_threaded"_test = [] {
        using scheduler               = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphLinear(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(that % t.size() >= 8u);
    };

    "BreadthFirstScheduler_linear_multi_threaded"_test = [] {
        using scheduler               = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphLinear(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(sched.jobs()->size() == 2u);
        checkBlockNames(sched.jobs()->at(0), {"s1", "mult2"});
        checkBlockNames(sched.jobs()->at(1), {"mult1", "out"});
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 8u) << std::format("execution order incomplete: {}", gr::join(t, ", "));
    };

    "SimpleScheduler_parallel_multi_threaded"_test = [] {
        using scheduler               = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphParallel(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 14u) << std::format("execution order incomplete: {}", gr::join(t, ", "));
    };

    "BreadthFirstScheduler_parallel_multi_threaded"_test = [] {
        using scheduler               = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphParallel(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(sched.jobs()->size() == 2u);
        checkBlockNames(sched.jobs()->at(0), {"s1", "mult1b", "mult2b", "outb"});
        checkBlockNames(sched.jobs()->at(1), {"mult1a", "mult2a", "outa"});
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "SimpleScheduler_scaled_sum_multi_threaded"_test = [] {
        using scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        // construct an example graph and get an adjacency list for it
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphScaledSum(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "BreadthFirstScheduler_scaled_sum_multi_threaded"_test = [] {
        using scheduler               = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{getGraphScaledSum(trace), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(eq(sched.jobs()->size(), 2u));
        checkBlockNames(sched.jobs()->at(0), {"s1", "mult", "out"});
        checkBlockNames(sched.jobs()->at(1), {"s2", "add"});
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "LifecycleBlock"_test = [] {
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;

        auto& lifecycleSource = flow.emplaceBlock<LifecycleSource<float>>();
        auto& lifecycleBlock  = flow.emplaceBlock<LifecycleBlock<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(lifecycleSource).to<"in">(lifecycleBlock)));

        auto sched = scheduler{std::move(flow), gr::thread_pool::kDefaultCpuPoolId};
        expect(sched.runAndWait().has_value());
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());

        expect(eq(lifecycleSource.n_samples_produced, lifecycleSource.n_samples_max)) << "Source n_samples_produced != n_samples_max";
        expect(eq(lifecycleBlock.process_one_count, lifecycleSource.n_samples_max)) << "process_one_count != n_samples_produced";

        expect(eq(lifecycleBlock.process_one_count, lifecycleSource.n_samples_produced));
        expect(eq(lifecycleBlock.start_count, 1));
        expect(eq(lifecycleBlock.stop_count, 1));
        expect(eq(lifecycleBlock.pause_count, 0));
        expect(eq(lifecycleBlock.resume_count, 0));
        expect(eq(lifecycleBlock.reset_count, 1));
    };

    "propagate DONE check-infinite loop"_test = [] {
        using namespace gr::testing;
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<CountingSource<float>>();
        auto& monitor = flow.emplaceBlock<Copy<float>>();
        auto& sink    = flow.emplaceBlock<NullSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto sched = scheduler{std::move(flow), gr::thread_pool::kDefaultCpuPoolId};

        std::atomic_bool shutDownByWatchdog{false};
        std::thread      watchdogThread([&sched, &shutDownByWatchdog]() {
            while (sched.state() != gr::lifecycle::State::RUNNING) { // wait until scheduler is running
                std::this_thread::sleep_for(40ms);
            }

            if (sched.state() == gr::lifecycle::State::RUNNING) {
                shutDownByWatchdog.store(true, std::memory_order_relaxed);
                sched.requestStop();
            }
        });

        expect(sched.runAndWait().has_value());

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }

        expect(ge(source.count, 0U));
        expect(shutDownByWatchdog.load(std::memory_order_relaxed));
        expect(sched.state() == gr::lifecycle::State::STOPPED);
        std::println("N.B by-design infinite loop correctly stopped after having emitted {} samples", source.count);
    };

    // create and return a watchdog thread and its control flag
    using TDuration     = std::chrono::duration<std::chrono::steady_clock::rep, std::chrono::steady_clock::period>;
    auto createWatchdog = [](auto& sched, TDuration timeOut = 2s, TDuration pollingPeriod = 40ms) {
        using namespace std::chrono_literals;
        auto externalInterventionNeeded = std::make_shared<std::atomic_bool>(false); // unique_ptr because you cannot move atomics

        // Create the watchdog thread
        std::thread watchdogThread([&sched, &externalInterventionNeeded, timeOut, pollingPeriod]() {
            auto timeout = std::chrono::steady_clock::now() + timeOut;
            while (std::chrono::steady_clock::now() < timeout) {
                if (sched.state() == gr::lifecycle::State::STOPPED) {
                    return;
                }
                std::this_thread::sleep_for(pollingPeriod);
            }
            // time-out reached, need to force termination of scheduler
            std::println("watchdog kicked in");
            externalInterventionNeeded->store(true, std::memory_order_relaxed);
            sched.requestStop();
            std::println("requested scheduler to stop");
        });

        return std::make_pair(std::move(watchdogThread), externalInterventionNeeded);
    };

    "propagate source DONE state: down-stream using EOS tag"_test = [&createWatchdog] {
        using namespace gr::testing;
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<ConstantSource<float>>({{"n_samples_max", 1024U}});
        auto& monitor = flow.emplaceBlock<Copy<float>>();
        auto& sink    = flow.emplaceBlock<CountingSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto sched                                        = scheduler{std::move(flow), gr::thread_pool::kDefaultCpuPoolId};
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value());

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(source.count, 1024U));
        expect(eq(sink.count, 1024U));

        std::println("N.B. 'propagate source DONE state: down-stream using EOS tag' test finished");
    };

    "propagate monitor DONE status: down-stream using EOS tag, upstream via disconnecting ports"_test = [&createWatchdog] {
        using namespace gr::testing;
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<NullSource<float>>();
        auto& monitor = flow.emplaceBlock<HeadBlock<float>>({{"n_samples_max", 1024U}});
        auto& sink    = flow.emplaceBlock<CountingSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto sched                                        = scheduler{std::move(flow), gr::thread_pool::kDefaultCpuPoolId};
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value());

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(monitor.count, 1024U));
        expect(eq(sink.count, 1024U));

        std::println("N.B. 'propagate monitor DONE status: down-stream using EOS tag, upstream via disconnecting ports' test finished");
    };

    "propagate sink DONE status: upstream via disconnecting ports"_test = [&createWatchdog] {
        using namespace gr::testing;
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<NullSource<float>>();
        auto& monitor = flow.emplaceBlock<Copy<float>>();
        auto& sink    = flow.emplaceBlock<CountingSink<float>>({{"n_samples_max", 1024U}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto sched                                        = scheduler{std::move(flow), gr::thread_pool::kDefaultCpuPoolId};
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value());

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(sink.count, 1024U));

        std::println("N.B. 'propagate sink DONE status: upstream via disconnecting ports' test finished");
    };

    "blocking scheduler"_test = [] {
        using namespace gr;
        using namespace gr::testing;
        using TScheduler = scheduler::Simple<scheduler::ExecutionPolicy::singleThreadedBlocking>;

        Graph flow;
        auto& source  = flow.emplaceBlock<NullSource<float>>();
        auto& monitor = flow.emplaceBlock<BusyLoopBlock<float>>();
        auto& sink    = flow.emplaceBlock<NullSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto scheduler                     = TScheduler{std::move(flow), gr::thread_pool::kDefaultCpuPoolId};
        scheduler.timeout_ms               = 100U; // also dynamically settable via messages/block interface
        scheduler.timeout_inactivity_count = 10U;  // also dynamically settable via messages/block interface

        expect(eq(0UZ, scheduler.graph().progress().value())) << "initial progress definition (0)";
        std::expected<void, Error> schedulerResult;
        auto                       schedulerThread = std::thread([&scheduler, &schedulerResult] {
            gr::thread_pool::thread::setThreadName("qa_Sched");
            schedulerResult = scheduler.runAndWait();
        });
        expect(awaitCondition(2s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        auto oldProgress = scheduler.graph().progress().value();
        expect(awaitCondition(2s, [&scheduler, &oldProgress] { // wait until there is no more progress (i.e. wait until all initial buffers are filled)
            std::this_thread::sleep_for(200ms);                // wait
            auto newProgress = scheduler.graph().progress().value();
            if (oldProgress == newProgress) {
                return true;
            }
            oldProgress = newProgress;
            return false;
        })) << "BusyLoopBlock sleeping";

        // check that buffers are full
        expect(eq(source.out.streamWriter().available(), 0UZ));
        expect(eq(monitor.in.streamReader().available(), graph::defaultMinBufferSize(true)));

        const auto progressAfterInit = scheduler.graph().progress().value();
        auto       estInvokeCount    = [&monitor] {
            const auto invokeCountInit = monitor._invokeCount.value();
            std::this_thread::sleep_for(20ms);
            return monitor._invokeCount.value() - invokeCountInit;
        };

        const auto invokeCount0 = estInvokeCount();
        expect(eq(scheduler.graph().progress().value(), progressAfterInit)) << "after thread started definition (0) - mark1";

        std::this_thread::sleep_for(200ms); // wait for time-out
        const auto invokeCount1 = estInvokeCount();

        expect(ge(invokeCount0, invokeCount1)) << std::format("info: invoke counts when active: {} sleeping: {}", invokeCount0, invokeCount1);
        std::println("info: invoke counts when active: {} sleeping: {}", invokeCount0, invokeCount1);
        expect(eq(scheduler.graph().progress().value(), progressAfterInit)) << "after thread started definition (0) - mark2";

        monitor._produceCount.setValue(1L);
        const auto invokeCount2 = estInvokeCount();
        expect(ge(invokeCount2, invokeCount1)) << std::format("info: invoke counts when active: {} sleeping: {}", invokeCount2, invokeCount1);
        std::println("info: invoke counts when active: {} sleeping: {}", invokeCount2, invokeCount1);

        expect(ge(scheduler.graph().progress().value(), progressAfterInit)) << "final progress definition (>0)";
        std::println("final progress {}", scheduler.graph().progress().value());

        expect(scheduler.state() == lifecycle::State::RUNNING) << "is running";
        std::println("request to shut-down");
        scheduler.requestStop();

        schedulerThread.join();
        std::string errorMsg = schedulerResult.has_value() ? "" : std::format("nested scheduler execution failed:\n{:f}\n", schedulerResult.error());
        expect(schedulerResult.has_value()) << errorMsg;
    };

    std::println("N.B. test-suite finished");
};

int main() { /* tests are statically executed */ }
