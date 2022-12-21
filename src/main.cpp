#include <array>
#include <fmt/core.h>

#include "graph.hpp"
#include "nodes.hpp"

namespace gr = fair::graph;

int
test01(gr::port_data<int, 1024> &result, const std::vector<int> a, const std::vector<int> b) {
    using graph = gr::merged_node<scale<int>, 0, gr::merged_node<saturate<int>, 0, adder<int>, 1>, 0>;
    constexpr graph node({ -1 }, { { -256, 256 }, {} });
    node.process_batch(result, std::span(a), std::span(b));
    auto reader = result.request_read();
    float  r      = 0;
    for (auto z : reader) r += z;
    return r;
}

int
main() {
    // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
    using graph  = gr::merged_node<scale<int>, 0, gr::merged_node<scale<int>, 0, adder<int>, 0>, 0>;
    graph merged = { { -1 }, { { 2 }, {} } };

    // execute graph
    std::array<int, 4> a = { 1, 2, 3, 4 };
    std::array<int, 4> b = { 10, 10, 10, 10 };

    int                r = 0;
    for (int i = 0; i < 4; ++i) {
        r += merged.process_one(a[i], b[i]);
    }

    fmt::print("Result of graph execution: {}\n", r);
    return r == 20 ? 0 : 1;
}
