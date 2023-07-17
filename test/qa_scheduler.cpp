#include <boost/ut.hpp>

#include <scheduler.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fg            = fair::graph;

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
class count_source : public fg::node<count_source<T, N>, fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">> {
    tracer     &_tracer;
    std::size_t _count = 0;

public:
    count_source(tracer &trace, std::string_view name_) : _tracer{ trace } { this->name = name_; }

    constexpr std::make_signed_t<std::size_t>
    available_samples(const count_source & /*d*/) const noexcept {
        const auto ret = static_cast<std::make_signed_t<std::size_t>>(N - _count);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    process_one() {
        _tracer.trace(this->name);
        return static_cast<int>(_count++);
    }
};

static_assert(fg::NodeType<count_source<float, 10U>>);

template<typename T, std::int64_t N>
class expect_sink : public fg::node<expect_sink<T, N>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "in">> {
    tracer                                         &_tracer;
    std::int64_t                                    _count = 0;
    std::function<void(std::int64_t, std::int64_t)> _checker;

public:
    expect_sink(tracer &trace, std::string_view name_, std::function<void(std::int64_t, std::int64_t)> &&checker) : _tracer{ trace }, _checker(std::move(checker)) { this->name = name_; }

    ~expect_sink() { boost::ut::expect(boost::ut::that % _count == N); }

    [[nodiscard]] fg::work_return_status_t
    process_bulk(std::span<const T> input) noexcept {
        _tracer.trace(this->name);
        for (auto data : input) {
            _checker(_count, data);
            _count++;
        }
        return fg::work_return_status_t::OK;
    }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fg::node<scale<T, Scale, R>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "original">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "scaled">> {
    tracer &_tracer;

public:
    scale(tracer &trace, std::string_view name_) : _tracer{ trace } { this->name = name_; }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) noexcept {
        _tracer.trace(this->name);
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fg::node<adder<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend0">, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend1">,
                              fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "sum">> {
    tracer &_tracer;

public:
    adder(tracer &trace, std::string_view name_) : _tracer(trace) { this->name = name_; }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) noexcept {
        _tracer.trace(this->name);
        return a + b;
    }
};

fair::graph::graph
get_graph_linear(tracer &trace) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

    // Nodes need to be alive for as long as the flow is
    fg::graph flow;
    // Generators
    auto &source1      = flow.make_node<count_source<int, 100000>>(trace, "s1");
    auto &scale_block1 = flow.make_node<scale<int, 2>>(trace, "mult1");
    auto &scale_block2 = flow.make_node<scale<int, 4>>(trace, "mult2");
    auto &sink         = flow.make_node<expect_sink<int, 100000>>(trace, "out", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == 8 * count); });

    std::ignore        = flow.connect<"scaled">(scale_block2).to<"in">(sink);
    std::ignore        = flow.connect<"scaled">(scale_block1).to<"original">(scale_block2);
    std::ignore        = flow.connect<"out">(source1).to<"original">(scale_block1);

    return flow;
}

fair::graph::graph
get_graph_parallel(tracer &trace) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

    // Nodes need to be alive for as long as the flow is
    fg::graph flow;
    // Generators
    auto &source1       = flow.make_node<count_source<int, 100000>>(trace, "s1");
    auto &scale_block1a = flow.make_node<scale<int, 2>>(trace, "mult1a");
    auto &scale_block2a = flow.make_node<scale<int, 3>>(trace, "mult2a");
    auto &sink_a        = flow.make_node<expect_sink<int, 100000>>(trace, "outa", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == 6 * count); });
    auto &scale_block1b = flow.make_node<scale<int, 3>>(trace, "mult1b");
    auto &scale_block2b = flow.make_node<scale<int, 5>>(trace, "mult2b");
    auto &sink_b        = flow.make_node<expect_sink<int, 100000>>(trace, "outb", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == 15 * count); });

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
fair::graph::graph
get_graph_scaled_sum(tracer &trace) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

    // Nodes need to be alive for as long as the flow is
    fg::graph flow;

    // Generators
    auto &source1     = flow.make_node<count_source<int, 100000>>(trace, "s1");
    auto &source2     = flow.make_node<count_source<int, 100000>>(trace, "s2");
    auto &scale_block = flow.make_node<scale<int, 2>>(trace, "mult");
    auto &add_block   = flow.make_node<adder<int>>(trace, "add");
    auto &sink        = flow.make_node<expect_sink<int, 100000>>(trace, "out", [](std::uint64_t count, std::uint64_t data) { boost::ut::expect(boost::ut::that % data == (2 * count) + count); });

    std::ignore       = flow.connect<"out">(source1).to<"original">(scale_block);
    std::ignore       = flow.connect<"scaled">(scale_block).to<"addend0">(add_block);
    std::ignore       = flow.connect<"out">(source2).to<"addend1">(add_block);
    std::ignore       = flow.connect<"sum">(add_block).to<"in">(sink);

    return flow;
}

template<typename node_type>
void
check_node_names(const std::vector<node_type> &joblist, std::set<std::string> set) {
    boost::ut::expect(boost::ut::that % joblist.size() == set.size());
    for (auto &node : joblist) {
        boost::ut::expect(boost::ut::that % set.contains(std::string(node->name()))) << fmt::format("{} not in {}\n", node->name(), set);
    }
}

const boost::ut::suite SchedulerTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    auto thread_pool              = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2);

    "SimpleScheduler_linear"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::simple<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "BreadthFirstScheduler_linear"_test = [&] {
        using scheduler = fair::graph::scheduler::breadth_first<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "SimpleScheduler_parallel"_test = [&] {
        using scheduler = fair::graph::scheduler::simple<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb", "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb" });
    };

    "BreadthFirstScheduler_parallel"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::breadth_first<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.run_and_wait();
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
        using scheduler = fair::graph::scheduler::simple<>;
        // construct an example graph and get an adjacency list for it
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out" });
    };

    "BreadthFirstScheduler_scaled_sum"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::breadth_first<>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == trace_vector_type{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out" });
    };

    "SimpleScheduler_linear_multi_threaded"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::simple<fg::scheduler::execution_policy::multi_threaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(that % t.size() >= 8u);
    };

    "BreadthFirstScheduler_linear_multi_threaded"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::breadth_first<fg::scheduler::execution_policy::multi_threaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_linear(trace), thread_pool };
        sched.init();
        expect(sched.getJobLists().size() == 2u);
        check_node_names(sched.getJobLists()[0], { "s1", "mult2" });
        check_node_names(sched.getJobLists()[1], { "mult1", "out" });
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 8u);
    };

    "SimpleScheduler_parallel_multi_threaded"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::simple<fg::scheduler::execution_policy::multi_threaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "BreadthFirstScheduler_parallel_multi_threaded"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::breadth_first<fg::scheduler::execution_policy::multi_threaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_parallel(trace), thread_pool };
        sched.init();
        expect(sched.getJobLists().size() == 2u);
        check_node_names(sched.getJobLists()[0], { "s1", "mult1b", "mult2b", "outb" });
        check_node_names(sched.getJobLists()[1], { "mult1a", "mult2a", "outa" });
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "SimpleScheduler_scaled_sum_multi_threaded"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::simple<fg::scheduler::execution_policy::multi_threaded>;
        // construct an example graph and get an adjacency list for it
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "BreadthFirstScheduler_scaled_sum_multi_threaded"_test = [&thread_pool] {
        using scheduler = fair::graph::scheduler::breadth_first<fg::scheduler::execution_policy::multi_threaded>;
        tracer trace{};
        auto   sched = scheduler{ get_graph_scaled_sum(trace), thread_pool };
        sched.init();
        expect(sched.getJobLists().size() == 2u);
        check_node_names(sched.getJobLists()[0], { "s1", "mult", "out" });
        check_node_names(sched.getJobLists()[1], { "s2", "add" });
        sched.run_and_wait();
        auto t = trace.get_vec();
        expect(boost::ut::that % t.size() >= 10u);
    };
};

int
main() { /* tests are statically executed */
}
