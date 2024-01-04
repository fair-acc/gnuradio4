#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using TraceVectorType = std::vector<std::string>;

class Tracer {
    std::mutex      _traceMutex;
    TraceVectorType _traceVector;

public:
    void
    trace(std::string_view id) {
        std::scoped_lock lock{ _traceMutex };
        if (_traceVector.empty() || _traceVector.back() != id) {
            _traceVector.emplace_back(id);
        }
    }

    TraceVectorType
    getVector() {
        std::scoped_lock lock{ _traceMutex };
        return { _traceVector };
    }
};

// define some example graph nodes
template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T>          out;
    std::size_t             n_samples_max = 0;
    std::shared_ptr<Tracer> tracer{};
    std::size_t             count = 0;

    ~CountSource() { boost::ut::expect(boost::ut::that % count == n_samples_max); }

    constexpr T
    processOne() {
        count++;
        if (count >= n_samples_max) {
            this->requestStop();
        }
        tracer->trace(this->name);
        return static_cast<int>(count);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (CountSource<T>), out, n_samples_max);

static_assert(gr::BlockLike<CountSource<float>>);

template<typename T>
struct ExpectSink : public gr::Block<ExpectSink<T>> {
    gr::PortIn<T>                                   in;
    std::size_t                                     n_samples_max = 0;
    std::shared_ptr<Tracer>                         tracer{};
    std::int64_t                                    count       = 0;
    std::int64_t                                    false_count = 0;
    std::function<bool(std::int64_t, std::int64_t)> checker;

    ~ExpectSink() {
        boost::ut::expect(boost::ut::eq(count, static_cast<std::int64_t>(n_samples_max)))
                << fmt::format("Number of processed samples ({}) must equal to n_samples_max ({}) for ExpectSink ({})", count, n_samples_max, this->name);
        boost::ut::expect(boost::ut::eq(false_count, 0)) << fmt::format("False counter ({}) must equal to 0 for ExpectSink ({})", false_count, this->name);
    }

    [[nodiscard]] gr::work::Status
    processBulk(std::span<const T> input) noexcept {
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

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (ExpectSink<T>), in, n_samples_max);

template<typename T>
struct Scale : public gr::Block<Scale<T>> {
    using R = decltype(std::declval<T>() * std::declval<T>());
    gr::PortIn<T>           original;
    gr::PortOut<R>          scaled;
    std::shared_ptr<Tracer> tracer{};
    T                       scale_factor = T(1.);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) noexcept {
        tracer->trace(this->name);
        return a * scale_factor;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (Scale<T>), original, scaled, scale_factor);

template<typename T>
struct Adder : public gr::Block<Adder<T>> {
    using R = decltype(std::declval<T>() + std::declval<T>());
    gr::PortIn<T>           addend0;
    gr::PortIn<T>           addend1;
    gr::PortOut<R>          sum;
    std::shared_ptr<Tracer> tracer;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a, V b) noexcept {
        tracer->trace(this->name);
        return a + b;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (Adder<T>), addend0, addend1, sum);

gr::Graph
getGraphLinear(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    std::size_t nMaxSamples{ 100000 };

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto &source1      = flow.emplaceBlock<CountSource<int>>({ { "name", "s1" }, { "n_samples_max", nMaxSamples } });
    source1.tracer     = tracer;
    auto &scaleBlock1  = flow.emplaceBlock<Scale<int>>({ { "name", "mult1" }, { "scale_factor", int(2) } });
    scaleBlock1.tracer = tracer;
    auto &scaleBlock2  = flow.emplaceBlock<Scale<int>>({ { "name", "mult2" }, { "scale_factor", int(4) } });
    scaleBlock2.tracer = tracer;
    auto &sink         = flow.emplaceBlock<ExpectSink<int>>({ { "name", "out" }, { "n_samples_max", nMaxSamples } });
    sink.tracer        = tracer;
    sink.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 8 * count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock2).to<"in">(sink)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock1).to<"original">(scaleBlock2)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source1).to<"original">(scaleBlock1)));

    return flow;
}

gr::Graph
getGraphParallel(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    std::size_t nMaxSamples{ 100000 };

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto &source1       = flow.emplaceBlock<CountSource<int>>({ { "name", "s1" }, { "n_samples_max", nMaxSamples } });
    source1.tracer      = tracer;
    auto &scaleBlock1a  = flow.emplaceBlock<Scale<int>>({ { "name", "mult1a" }, { "scale_factor", int(2) } });
    scaleBlock1a.tracer = tracer;
    auto &scaleBlock2a  = flow.emplaceBlock<Scale<int>>({ { "name", "mult2a" }, { "scale_factor", int(3) } });
    scaleBlock2a.tracer = tracer;
    auto &sinkA         = flow.emplaceBlock<ExpectSink<int>>({ { "name", "outa" }, { "n_samples_max", nMaxSamples } });
    sinkA.tracer        = tracer;
    sinkA.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 6 * count; };
    auto &scaleBlock1b  = flow.emplaceBlock<Scale<int>>({ { "name", "mult1b" }, { "scale_factor", int(3) } });
    scaleBlock1b.tracer = tracer;
    auto &scaleBlock2b  = flow.emplaceBlock<Scale<int>>({ { "name", "mult2b" }, { "scale_factor", int(5) } });
    scaleBlock2b.tracer = tracer;
    auto &sinkB         = flow.emplaceBlock<ExpectSink<int>>({ { "name", "outb" }, { "n_samples_max", nMaxSamples } });
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
gr::Graph
getGraphScaledSum(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    std::size_t nMaxSamples{ 100000 };

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;

    // Generators
    auto &source1     = flow.emplaceBlock<CountSource<int>>({ { "name", "s1" }, { "n_samples_max", nMaxSamples } });
    source1.tracer    = tracer;
    auto &source2     = flow.emplaceBlock<CountSource<int>>({ { "name", "s2" }, { "n_samples_max", nMaxSamples } });
    source2.tracer    = tracer;
    auto &scaleBlock  = flow.emplaceBlock<Scale<int>>({ { "name", "mult" }, { "scale_factor", int(2) } });
    scaleBlock.tracer = tracer;
    auto &addBlock    = flow.emplaceBlock<Adder<int>>({ { "name", "add" } });
    addBlock.tracer   = tracer;
    auto &sink        = flow.emplaceBlock<ExpectSink<int>>({ { "name", "out" }, { "n_samples_max", nMaxSamples } });
    sink.tracer       = tracer;
    sink.checker      = [](std::uint64_t count, std::uint64_t data) -> bool { return data == (2 * count) + count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source1).to<"original">(scaleBlock)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"scaled">(scaleBlock).to<"addend0">(addBlock)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source2).to<"addend1">(addBlock)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"sum">(addBlock).to<"in">(sink)));

    return flow;
}

template<typename TBlock>
void
checkBlockNames(const std::vector<TBlock> &joblist, std::set<std::string> set) {
    boost::ut::expect(boost::ut::that % joblist.size() == set.size());
    for (auto &block : joblist) {
        boost::ut::expect(boost::ut::that % set.contains(std::string(block->name()))) << fmt::format("{} not in {}\n", block->name(), set);
    }
}

template<typename T>
struct LifecycleSource : public gr::Block<LifecycleSource<T>> {
    gr::PortOut<T> out;
    std::int32_t   n_samples_produced = 0;
    std::int32_t   n_samples_max      = 10;

    [[nodiscard]] constexpr T
    processOne() noexcept {
        n_samples_produced++;
        if (n_samples_produced >= n_samples_max) {
            this->requestStop();
            return T(n_samples_produced); // this sample will be the last emitted.
        }
        return T(n_samples_produced);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (LifecycleSource<T>), out);

template<typename T>
struct LifecycleBlock : public gr::Block<LifecycleBlock<T>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    int process_one_count{};
    int start_count{};
    int stop_count{};
    int reset_count{};
    int pause_count{};
    int resume_count{};

    [[nodiscard]] constexpr T
    processOne(T a) noexcept {
        process_one_count++;
        return a;
    }

    void
    start() {
        start_count++;
    }

    void
    stop() {
        stop_count++;
    }

    void
    reset() {
        reset_count++;
    }

    void
    pause() {
        pause_count++;
    }

    void
    resume() {
        resume_count++;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (LifecycleBlock<T>), in, out);

const boost::ut::suite SchedulerTests = [] {
    using namespace boost::ut;
    using namespace gr;
    auto threadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2);

    "SimpleScheduler_linear"_test = [&threadPool] {
        using scheduler               = gr::scheduler::Simple<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphLinear(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "SimpleScheduler_linear_iterate"_test = [&threadPool] {
        using scheduler               = gr::scheduler::Simple<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphLinear(trace), threadPool };
        while (!sched.isDone()) {
            sched.iterateAndWait();
        }
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "BreadthFirstScheduler_linear"_test = [&] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphLinear(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "BreadthFirstScheduler_linear_iterate"_test = [&] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphLinear(trace), threadPool };
        while (!sched.isDone()) {
            sched.iterateAndWait();
        }
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "SimpleScheduler_parallel"_test = [&] {
        using scheduler               = gr::scheduler::Simple<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphParallel(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb", "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb" });
    };

    "BreadthFirstScheduler_parallel"_test = [&threadPool] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphParallel(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t
               == TraceVectorType{
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

    "SimpleScheduler_scaled_sum"_test = [&threadPool] {
        using scheduler = gr::scheduler::Simple<>;
        // construct an example graph and get an adjacency list for it
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphScaledSum(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out" });
    };

    "BreadthFirstScheduler_scaled_sum"_test = [&threadPool] {
        using scheduler               = gr::scheduler::BreadthFirst<>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphScaledSum(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == TraceVectorType{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out" });
    };

    "SimpleScheduler_linear_multi_threaded"_test = [&threadPool] {
        using scheduler               = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphLinear(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(that % t.size() >= 8u);
    };

    "BreadthFirstScheduler_linear_multi_threaded"_test = [&threadPool] {
        using scheduler               = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphLinear(trace), threadPool };
        sched.init();
        expect(sched.jobs().size() == 2u);
        checkBlockNames(sched.jobs()[0], { "s1", "mult2" });
        checkBlockNames(sched.jobs()[1], { "mult1", "out" });
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 8u);
    };

    "SimpleScheduler_parallel_multi_threaded"_test = [&threadPool] {
        using scheduler               = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphParallel(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "BreadthFirstScheduler_parallel_multi_threaded"_test = [&threadPool] {
        using scheduler               = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphParallel(trace), threadPool };
        sched.init();
        expect(sched.jobs().size() == 2u);
        checkBlockNames(sched.jobs()[0], { "s1", "mult1b", "mult2b", "outb" });
        checkBlockNames(sched.jobs()[1], { "mult1a", "mult2a", "outa" });
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "SimpleScheduler_scaled_sum_multi_threaded"_test = [&threadPool] {
        using scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        // construct an example graph and get an adjacency list for it
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphScaledSum(trace), threadPool };
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "BreadthFirstScheduler_scaled_sum_multi_threaded"_test = [&threadPool] {
        using scheduler               = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        auto                    sched = scheduler{ getGraphScaledSum(trace), threadPool };
        sched.init();
        expect(sched.jobs().size() == 2u);
        checkBlockNames(sched.jobs()[0], { "s1", "mult", "out" });
        checkBlockNames(sched.jobs()[1], { "s2", "add" });
        sched.runAndWait();
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "LifecycleBlock"_test = [&threadPool] {
        using scheduler = gr::scheduler::Simple<>;
        gr::Graph flow;

        auto &lifecycleSource = flow.emplaceBlock<LifecycleSource<float>>();
        auto &lifecycleBlock  = flow.emplaceBlock<LifecycleBlock<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(lifecycleSource).to<"in">(lifecycleBlock)));

        auto sched = scheduler{ std::move(flow), threadPool };
        sched.runAndWait();
        sched.reset();

        expect(eq(lifecycleSource.n_samples_produced, lifecycleSource.n_samples_max)) << "Source n_samples_produced != n_samples_max";
        expect(eq(lifecycleBlock.process_one_count, lifecycleSource.n_samples_max)) << "process_one_count != n_samples_produced";

        expect(eq(lifecycleBlock.process_one_count, lifecycleSource.n_samples_produced));
        expect(eq(lifecycleBlock.start_count, 1));
        expect(eq(lifecycleBlock.stop_count, 1));
        expect(eq(lifecycleBlock.pause_count, 0));
        expect(eq(lifecycleBlock.resume_count, 0));
        expect(eq(lifecycleBlock.reset_count, 1));
    };
};

int
main() { /* tests are statically executed */
}
