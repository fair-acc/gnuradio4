#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using trace_vector_type = std::vector<std::string>;

class tracer {
    std::mutex        _trace_mutex;
    trace_vector_type _trace_vector;

public:
    void
    trace(std::string_view id) {
        std::scoped_lock lock{ _trace_mutex };
        if (_trace_vector.empty() || _trace_vector.back() != id) {
            _trace_vector.emplace_back(id);
        }
    }

    trace_vector_type
    get_vec() {
        std::scoped_lock lock{ _trace_mutex };
        return { _trace_vector };
    }
};

// define some example graph nodes
template<typename T, std::size_t N>
class count_source : public gr::Block<count_source<T, N>, gr::PortOutNamed<T, "out">> {
    tracer     &_tracer;
    std::size_t _count = 0;

public:
    count_source(tracer &trace, std::string_view name_) : _tracer{ trace } { this->name = name_; }

    constexpr std::make_signed_t<std::size_t>
    available_samples(const count_source & /*d*/) noexcept {
        const auto ret = static_cast<std::make_signed_t<std::size_t>>(N - _count);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    processOne() {
        _tracer.trace(this->name);
        return static_cast<int>(_count++);
    }
};

static_assert(gr::BlockLike<count_source<float, 10U>>);

template<typename T, std::int64_t N>
class expect_sink : public gr::Block<expect_sink<T, N>, gr::PortInNamed<T, "in">> {
    tracer                                         &_tracer;
    std::int64_t                                    _count = 0;
    std::function<void(std::int64_t, std::int64_t)> _checker;

public:
    expect_sink(tracer &trace, std::string_view name_, std::function<void(std::int64_t, std::int64_t)> &&checker) : _tracer{ trace }, _checker(std::move(checker)) { this->name = name_; }

    ~expect_sink() { boost::ut::expect(boost::ut::that % _count == N); }

    [[nodiscard]] gr::WorkReturnStatus
    processBulk(std::span<const T> input) noexcept {
        _tracer.trace(this->name);
        for (auto data : input) {
            _checker(_count, data);
            _count++;
        }
        return gr::WorkReturnStatus::OK;
    }

    constexpr void
    processOne(T /*a*/) noexcept {
        _tracer.trace(this->name());
    }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public gr::Block<scale<T, Scale, R>, gr::PortInNamed<T, "original">, gr::PortOutNamed<R, "scaled">> {
    tracer &_tracer;

public:
    scale(tracer &trace, std::string_view name_) : _tracer{ trace } { this->name = name_; }

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) noexcept {
        _tracer.trace(this->name);
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public gr::Block<adder<T>, gr::PortInNamed<T, "addend0">, gr::PortInNamed<T, "addend1">, gr::PortOutNamed<R, "sum">> {
    tracer &_tracer;

public:
    adder(tracer &trace, std::string_view name_) : _tracer(trace) { this->name = name_; }

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a, V b) noexcept {
        _tracer.trace(this->name);
        return a + b;
    }
};

gr::Graph
get_graph_linear(tracer &trace) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto &source1      = flow.emplaceBlock<count_source<int, 100000>>(trace, "s1");
    auto &scale_block1 = flow.emplaceBlock<scale<int, 2>>(trace, "mult1");
    auto &scale_block2 = flow.emplaceBlock<scale<int, 4>>(trace, "mult2");
    auto &sink         = flow.emplaceBlock<expect_sink<int, 100000>>(trace, "out", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == 8 * count); });

    std::ignore        = flow.connect<"scaled">(scale_block2).to<"in">(sink);
    std::ignore        = flow.connect<"scaled">(scale_block1).to<"original">(scale_block2);
    std::ignore        = flow.connect<"out">(source1).to<"original">(scale_block1);

    return flow;
}

gr::Graph
get_graph_parallel(tracer &trace) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto &source1       = flow.emplaceBlock<count_source<int, 100000>>(trace, "s1");
    auto &scale_block1a = flow.emplaceBlock<scale<int, 2>>(trace, "mult1a");
    auto &scale_block2a = flow.emplaceBlock<scale<int, 3>>(trace, "mult2a");
    auto &sink_a        = flow.emplaceBlock<expect_sink<int, 100000>>(trace, "outa", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == 6 * count); });
    auto &scale_block1b = flow.emplaceBlock<scale<int, 3>>(trace, "mult1b");
    auto &scale_block2b = flow.emplaceBlock<scale<int, 5>>(trace, "mult2b");
    auto &sink_b        = flow.emplaceBlock<expect_sink<int, 100000>>(trace, "outb", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == 15 * count); });

    std::ignore         = flow.connect<"scaled">(scale_block1a).to<"original">(scale_block2a);
    std::ignore         = flow.connect<"scaled">(scale_block1b).to<"original">(scale_block2b);
    std::ignore         = flow.connect<"scaled">(scale_block2b).to<"in">(sink_b);
    std::ignore         = flow.connect<"out">(source1).to<"original">(scale_block1a);
    std::ignore         = flow.connect<"scaled">(scale_block2a).to<"in">(sink_a);
    std::ignore         = flow.connect<"out">(source1).to<"original">(scale_block1b);

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
get_graph_scaled_sum(tracer &trace) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;

    // Generators
    auto &source1     = flow.emplaceBlock<count_source<int, 100000>>(trace, "s1");
    auto &source2     = flow.emplaceBlock<count_source<int, 100000>>(trace, "s2");
    auto &scale_block = flow.emplaceBlock<scale<int, 2>>(trace, "mult");
    auto &add_block   = flow.emplaceBlock<adder<int>>(trace, "add");
    auto &sink        = flow.emplaceBlock<expect_sink<int, 100000>>(trace, "out", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == (2 * count) + count); });

    std::ignore       = flow.connect<"out">(source1).to<"original">(scale_block);
    std::ignore       = flow.connect<"scaled">(scale_block).to<"addend0">(add_block);
    std::ignore       = flow.connect<"out">(source2).to<"addend1">(add_block);
    std::ignore       = flow.connect<"sum">(add_block).to<"in">(sink);

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

const boost::ut::suite SchedulerTests = [] {
    using namespace boost::ut;
    using namespace gr;
    auto thread_pool              = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2);

    "SimpleScheduler_linear"_test = [&thread_pool] {
        using scheduler = gr::scheduler::Simple<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "BreadthFirstScheduler_linear"_test = [&] {
        using scheduler = gr::scheduler::BreadthFirst<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "SimpleScheduler_parallel"_test = [&] {
        using scheduler = gr::scheduler::Simple<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb", "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb" });
    };

    "BreadthFirstScheduler_parallel"_test = [&thread_pool] {
        using scheduler = gr::scheduler::BreadthFirst<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t
               == trace_vector_type{
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

    "SimpleScheduler_scaled_sum"_test = [&thread_pool] {
        using scheduler = gr::scheduler::Simple<>;
        // construct an example graph and get an adjacency list for it
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out" });
    };

    "BreadthFirstScheduler_scaled_sum"_test = [&thread_pool] {
        using scheduler = gr::scheduler::BreadthFirst<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out" });
    };

    "SimpleScheduler_linear_multi_threaded"_test = [&thread_pool] {
        using scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(that % t.size() >= 8u);
    };

    "BreadthFirstScheduler_linear_multi_threaded"_test = [&thread_pool] {
        using scheduler = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.init();
        expect(sched.jobs().size() == 2u);
        checkBlockNames(sched.jobs()[0], { "s1", "mult2" });
        checkBlockNames(sched.jobs()[1], { "mult1", "out" });
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 8u);
    };

    "SimpleScheduler_parallel_multi_threaded"_test = [&thread_pool] {
        using scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "BreadthFirstScheduler_parallel_multi_threaded"_test = [&thread_pool] {
        using scheduler = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.init();
        expect(sched.jobs().size() == 2u);
        checkBlockNames(sched.jobs()[0], { "s1", "mult1b", "mult2b", "outb" });
        checkBlockNames(sched.jobs()[1], { "mult1a", "mult2a", "outa" });
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "SimpleScheduler_scaled_sum_multi_threaded"_test = [&thread_pool] {
        using scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;
        // construct an example graph and get an adjacency list for it
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "BreadthFirstScheduler_scaled_sum_multi_threaded"_test = [&thread_pool] {
        using scheduler = gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.init();
        expect(sched.jobs().size() == 2u);
        checkBlockNames(sched.jobs()[0], { "s1", "mult", "out" });
        checkBlockNames(sched.jobs()[1], { "s2", "add" });
        sched.runAndWait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 10u);
    };
};

int
main() { /* tests are statically executed */
}
