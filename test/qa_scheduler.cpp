#include <boost/ut.hpp>

#include <scheduler.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template <>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fg = fair::graph;

using trace_vector = std::vector<std::pair<std::string, std::size_t>>;
static void trace(trace_vector &traceVector, std::string_view id, std::size_t n = 1) {
    if (!traceVector.empty() && traceVector.back().first == id) {
        traceVector.back().second += n;
    } else {
        traceVector.emplace_back(id, n);
    }
}

template<>
struct fmt::formatter<std::pair<std::string, std::size_t>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::pair<std::string, std::size_t> & fp, FormatContext& ctx) {
    return format_to(ctx.out(), "{} \t{}", fp.first, fp.second);
  }
};

// define some example graph nodes
template<typename T, std::int64_t N>
class random_source : public fg::node<random_source<T, N>, fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">> {
    trace_vector &tracer;
    std::int64_t count = 0;
public:
    random_source(trace_vector &trace, std::string_view name) : tracer{trace} {this->_name = name;}

    constexpr std::int64_t
    available_samples(const random_source &d) noexcept {
        const auto ret = static_cast<std::int64_t>(N - count);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    process_one() {
        trace(tracer, this->name());
        return static_cast<int>(count++);
    }
};

// for some reason instantiating the same class twice in the flowgraph leads to compilation errors
template<typename T, std::int64_t N>
class random_source2 : public fg::node<random_source2<T, N>, fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">> {
    trace_vector &tracer;
    std::int64_t count = 0;
public:
    random_source2(trace_vector &trace, std::string_view name) : tracer{trace} {this->_name = name;}

    constexpr std::int64_t
    available_samples(const random_source2 &d) noexcept {
        const auto ret = std::min(static_cast<std::int64_t>(N - count), std::numeric_limits<std::int64_t>::max());
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    process_one() {
        trace(tracer, this->name());
        return static_cast<int>(1000 + count++);
    }
};

template<typename T>
class cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "in">> {
    trace_vector &tracer;
    std::int64_t count = 0;
public:
    cout_sink(trace_vector &trace, std::string_view name) : tracer{trace} {this->_name = name;}
    [[nodiscard]] fg::work_return_t
    process_bulk(std::span<const T> input) noexcept {
        trace(tracer, this->name(), static_cast<size_t>(input.size()));
        fmt::print("data[{}]: data[0] = {}\n", input.size(), input[0]);
        for (auto data: input) {
            boost::ut::expect(boost::ut::that % data == 2 * count + 1000 + count);
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
        trace(tracer, this->name(), vir::stdx::is_simd_v<V> ? V::size() : 1);
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
        trace(tracer, this->name(), vir::stdx::is_simd_v<V> ? V::size() : 1);
        return a + b;
    }
};

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
get_graph(trace_vector &traceVector) {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

// Nodes need to be alive for as long as the flow is
    fg::graph flow;

// Generators
    auto& source1 = flow.make_node<random_source<int, 100000>>(traceVector, "s1");
    auto& source2 = flow.make_node<random_source2<int, 100000>>(traceVector, "s2");

    auto& scale_block = flow.make_node<scale<int, 2>>(traceVector, "mult");
    auto& add_block = flow.make_node<adder<int>>(traceVector, "add");
    auto& sink = flow.make_node<cout_sink<int>>(traceVector, "out");

    std::ignore = flow.connect<"out">(source1).to<"original">(scale_block);
    std::ignore = flow.connect<"scaled">(scale_block).to<"addend0">(add_block);
    std::ignore = flow.connect<"out">(source2).to<"addend1">(add_block);
    std::ignore = flow.connect<"sum">(add_block).to<"in">(sink);

    return flow;
}

const boost::ut::suite SchedulerTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;

    "SimpleScheduler"_test = [] {
        using scheduler = fair::graph::scheduler::simple;
        // construct an example graph and get an adjacency list for it
        trace_vector t{};
        fair::graph::graph g = get_graph(t);
        auto sched = scheduler{g};
        fmt::print("start running graph:\n");
        sched.work();
        fmt::print("scheduling finished\n");
        fmt::print("Traced block executions:\n\n{}\n\n", fmt::join(t, "\n"));
        expect(t.size() == 10);
        expect(t == trace_vector{
                {"s1", 65536},
                {"s2", 65536},
                {"mult", 65536},
                {"add", 65536},
                {"out", 65536},
                {"s1", 34464},
                {"s2", 34464},
                {"mult", 34464},
                {"add", 34464},
                {"out", 34464}
        });
    };

    "BreadthFirstScheduler"_test = [] {
        using scheduler = fair::graph::scheduler::breadth_first;
        trace_vector t{};
        fair::graph::graph g = get_graph(t);
        auto sched = scheduler{g};
        fmt::print("start running graph:\n");
        sched.work();
        fmt::print("scheduling finished\n");
        fmt::print("Traced block executions:\n{}\n\n", fmt::join(t, "\n"));
        expect(t.size() == 10);
        expect(t == trace_vector{
                {"s2", 65536},
                {"s1", 65536},
                // adder would be executed before scale for breadth first, which leads to non-optimal scheduling here
                {"mult", 65536},
                {"s1", 34464},
                {"add", 65536},
                {"mult", 34464},
                {"out", 65536},
                {"s2", 34464},
                {"add", 34464},
                {"out", 34464}
        });
    };
};

int
main() { /* tests are statically executed */
}