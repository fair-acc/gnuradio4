#include <boost/ut.hpp>

#include <scheduler.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template <>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fg = fair::graph;

using trace_vector = std::vector<std::string>;
static void trace(trace_vector &traceVector, std::string_view id) {
    if (traceVector.empty() || traceVector.back() != id) {
        traceVector.emplace_back(id);
    }
}

// define some example graph nodes
template<typename T, std::int64_t N>
class count_source : public fg::node<count_source<T, N>, fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">> {
    trace_vector &tracer;
    std::int64_t count = 0;
public:
    count_source(trace_vector &trace, std::string_view name) : tracer{trace} { this->_name = name;}

    constexpr std::int64_t
    available_samples(const count_source &/*d*/) noexcept {
        const auto ret = static_cast<std::int64_t>(N - count);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    process_one() {
        trace(tracer, this->name());
        return static_cast<int>(count++);
    }
};

template<typename T>
class expect_sink : public fg::node<expect_sink<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "in">> {
    trace_vector &tracer;
    std::int64_t count = 0;
    std::function<void(std::int64_t, std::int64_t)> _checker;
public:
    expect_sink(trace_vector &trace, std::string_view name, std::function<void(std::int64_t, std::int64_t)> &&checker) : tracer{trace}, _checker(std::move(checker)) { this->_name = name;}
    [[nodiscard]] fg::work_return_t
    process_bulk(std::span<const T> input) noexcept {
        trace(tracer, this->name());
        for (auto data: input) {
            _checker(count, data);
            count++;
        }
        return fg::work_return_t::OK;
    }
    constexpr void
    process_one(T /*a*/) const noexcept {
        trace(tracer, this->name());
    }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fg::node<scale<T, Scale, R>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "original">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "scaled">> {
    trace_vector &tracer;
public:
    scale(trace_vector &trace, std::string_view name) : tracer{trace} {this->_name = name;}
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        trace(tracer, this->name());
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fg::node<adder<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend0">, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend1">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "sum">> {
    trace_vector &tracer;
public:
    adder(trace_vector &trace, std::string_view name) : tracer{trace} {this->_name = name;}
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        trace(tracer, this->name());
        return a + b;
    }
};

fair::graph::graph
get_graph_linear(trace_vector &traceVector) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

// Nodes need to be alive for as long as the flow is
    fg::graph flow;
// Generators
    auto& source1 = flow.make_node<count_source<int, 100000>>(traceVector, "s1");
    auto& scale_block1 = flow.make_node<scale<int, 2>>(traceVector, "mult1");
    auto& scale_block2 = flow.make_node<scale<int, 4>>(traceVector, "mult2");
    auto& sink = flow.make_node<expect_sink<int>>(traceVector, "out", [](std::uint64_t count, std::uint64_t data) {
        boost::ut::expect(boost::ut::that % data == 8 * count);
    } );

    std::ignore = flow.connect<"scaled">(scale_block2).to<"in">(sink);
    std::ignore = flow.connect<"scaled">(scale_block1).to<"original">(scale_block2);
    std::ignore = flow.connect<"out">(source1).to<"original">(scale_block1);

    return flow;
}

fair::graph::graph
get_graph_parallel(trace_vector &traceVector) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

// Nodes need to be alive for as long as the flow is
    fg::graph flow;
// Generators
    auto& source1 = flow.make_node<count_source<int, 100000>>(traceVector, "s1");
    auto& scale_block1a = flow.make_node<scale<int, 2>>(traceVector, "mult1a");
    auto& scale_block2a = flow.make_node<scale<int, 3>>(traceVector, "mult2a");
    auto& sink_a = flow.make_node<expect_sink<int>>(traceVector, "outa", [](std::uint64_t count, std::uint64_t data) {
        boost::ut::expect(boost::ut::that % data == 6 * count);
    } );
    auto& scale_block1b = flow.make_node<scale<int, 3>>(traceVector, "mult1b");
    auto& scale_block2b = flow.make_node<scale<int, 5>>(traceVector, "mult2b");
    auto& sink_b = flow.make_node<expect_sink<int>>(traceVector, "outb", [](std::uint64_t count, std::uint64_t data) {
        boost::ut::expect(boost::ut::that % data == 15 * count);
    } );

    std::ignore = flow.connect<"scaled">(scale_block1a).to<"original">(scale_block2a);
    std::ignore = flow.connect<"scaled">(scale_block1b).to<"original">(scale_block2b);
    std::ignore = flow.connect<"scaled">(scale_block2b).to<"in">(sink_b);
    std::ignore = flow.connect<"out">(source1).to<"original">(scale_block1a);
    std::ignore = flow.connect<"scaled">(scale_block2a).to<"in">(sink_a);
    std::ignore = flow.connect<"out">(source1).to<"original">(scale_block1b);

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
get_graph_scaled_sum(trace_vector &traceVector) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

// Nodes need to be alive for as long as the flow is
    fg::graph flow;

// Generators
    auto& source1 = flow.make_node<count_source<int, 100000>>(traceVector, "s1");
    auto& source2 = flow.make_node<count_source<int, 100000>>(traceVector, "s2");
    auto& scale_block = flow.make_node<scale<int, 2>>(traceVector, "mult");
    auto& add_block = flow.make_node<adder<int>>(traceVector, "add");
    auto& sink = flow.make_node<expect_sink<int>>(traceVector, "out", [](std::uint64_t count, std::uint64_t data) {
        boost::ut::expect(boost::ut::that % data == (2 * count) + count);
    } );

    std::ignore = flow.connect<"out">(source1).to<"original">(scale_block);
    std::ignore = flow.connect<"scaled">(scale_block).to<"addend0">(add_block);
    std::ignore = flow.connect<"out">(source2).to<"addend1">(add_block);
    std::ignore = flow.connect<"sum">(add_block).to<"in">(sink);

    return flow;
}

const boost::ut::suite SchedulerTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;

    "SimpleScheduler_linear"_test = [] {
        using scheduler = fair::graph::scheduler::simple;
        trace_vector t{};
        auto sched = scheduler{get_graph_linear(t)};
        sched.work();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == trace_vector{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out" });
    };

    "BreadthFirstScheduler_linear"_test = [] {
        using scheduler = fair::graph::scheduler::breadth_first;
        trace_vector t{};
        auto sched = scheduler{get_graph_linear(t)};
        sched.work();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == trace_vector{ "s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out"});
    };

    "SimpleScheduler_parallel"_test = [] {
        using scheduler = fair::graph::scheduler::simple;
        trace_vector t{};
        auto sched = scheduler{get_graph_parallel(t)};
        sched.work();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == trace_vector{ "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb", "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb"});
    };

    "BreadthFirstScheduler_parallel"_test = [] {
        using scheduler = fair::graph::scheduler::breadth_first;
        trace_vector t{};
        auto sched = scheduler{get_graph_parallel(t)};
        sched.work();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == trace_vector{"s1", "mult1a", "mult1b", "mult2a", "mult2b", "outa", "outb", "s1", "mult1a", "mult1b", "mult2a", "mult2b", "outa", "outb", });
    };

    "SimpleScheduler_scaled_sum"_test = [] {
        using scheduler = fair::graph::scheduler::simple;
        // construct an example graph and get an adjacency list for it
        trace_vector t{};
        auto sched = scheduler{get_graph_scaled_sum(t)};
        sched.work();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == trace_vector{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out"});
    };

    "BreadthFirstScheduler_scaled_sum"_test = [] {
        using scheduler = fair::graph::scheduler::breadth_first;
        trace_vector t{};
        auto sched = scheduler{get_graph_scaled_sum(t)};
        sched.work();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == trace_vector{ "s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out"});
    };
};

int
main() { /* tests are statically executed */
}
