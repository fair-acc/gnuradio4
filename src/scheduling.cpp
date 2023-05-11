#include <fmt/core.h>

#include <scheduler.hpp>

namespace fg = fair::graph;

// define some example graph nodes
template<typename T>
class random_source : public fg::node<random_source<T>, fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "random">> {
public:
constexpr T
process_one() {
    return 42;
}
};

// for some reason instantiating the same class twice in the flowgraph leads to compilation errors
template<typename T>
class random_source2 : public fg::node<random_source2<T>, fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "random2">> {
public:
    constexpr T
    process_one() {
        return 23;
    }
};

template<typename T>
class cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "sink">> {
    int count = 0;
public:
    [[nodiscard]] fg::work_return_t
    process_bulk(std::span<const T> input) noexcept {
        fmt::print("data[{}]: data[0] = {}\n", input.size(), input[0]);
        count += input.size();
        if (count > 300000) {
            fmt::print("sink reached limit\n");
            return fg::work_return_t::DONE;
        } else {
            return fg::work_return_t::OK;
        }
    }

    void
    process_one(T value) {
        std::cout << value << std::endl;
    }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fg::node<scale<T, Scale, R>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "original">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "scaled">> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fg::node<adder<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend0">, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend1">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "sum">> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        return a + b;
    }
};

// sets up an example graph
fair::graph::graph
get_graph() {
    using fg::port_direction_t::INPUT;
    using fg::port_direction_t::OUTPUT;

// Nodes need to be alive for as long as the flow is
    fg::graph flow;

// Generators
    auto& number = flow.make_node<random_source<int>>();
    auto& answer = flow.make_node<random_source2<int>>();

    auto& scaled = flow.make_node<scale<int, 2>>();
    auto& added = flow.make_node<adder<int>>();
    auto& out = flow.make_node<cout_sink<int>>();

    std::ignore = flow.connect<"random">(number).to<"original">(scaled);
    std::ignore = flow.connect<"scaled">(scaled).to<"addend0">(added);
    std::ignore = flow.connect<"random2">(answer).to<"addend1">(added);
    std::ignore = flow.connect<"sum">(added).to<"sink">(out);

    return flow;
}

/**
 * entry point for the code
 * - define how return types are interpreted.
 *  - one block returning DONE should not stop the complete graph
 *  - idea: remove DONE blocks from scheduler (or from the graph?) and remove backwards all blocks leading to it or originating from it
 *    - only after the queues have been cleared?
 *    - queues might have different lengths?
 *    - mark input/output as "stale" and interpret insufficient input/output items as DONE -> how to determine which one is insufficient?
 * - incorporate return value of the work function to optimise for the different chunk sizes scenario
 * - how to incorporate gr4 scheduler (which dynamically loads the scheduler)
 * - use std::graph?
 * - how to incorporate coroutines into the scheduling
 *   - [x] verify emscripten compatibility (conduit example works on nodejs at least)
 * - multiple threads
 *   - emscripten, webWorkers?
**/
int main() {
    using scheduler = fair::graph::scheduler::simple;
    // using scheduler = fair::graph::scheduler::breath_first;
    // construct an example graph and get an adjacency list for it
    fmt::print("instantiating graph:\n");
    fair::graph::graph g = get_graph();
    fmt::print("calculating adjavency list:\n");
    auto sched = scheduler{g};

    // schedule execution of the graph
    fmt::print("start running graph:\n");
    sched.work();
    fmt::print("scheduling finnished\n");
}
