#include <array>
#include <fmt/core.h>

#include "graph.hpp"

template<typename T, int Depth>
    requires(Depth > 0)
class delay : public fair::graph::node<delay<T, Depth>, fair::graph::make_input_ports<T>, fair::graph::make_output_ports<T>> {
    std::array<T, Depth> buffer = {};
    int                  pos    = 0;

public:
    [[nodiscard]] constexpr T process_one(T in) noexcept {
        T ret       = buffer[pos];
        buffer[pos] = in;
        if (pos == Depth - 1) {
            pos = 0;
        } else {
            ++pos;
        }
        return ret;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fair::graph::node<adder<T>, fair::graph::make_input_ports<T, T>, fair::graph::make_output_ports<R>> {
public:
    [[nodiscard]] constexpr R process_one(T a, T b) const noexcept { return a + b; }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fair::graph::node<scale<T, Scale, R>, fair::graph::make_input_ports<T>, fair::graph::make_output_ports<R>> {
public:
    [[nodiscard]] constexpr R process_one(T a) const noexcept { return a * Scale; }
};

int main() {
    using fair::graph::merge;
    // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
    auto merged = merge<0, 0>(scale<int, -1>(), merge<0, 0>(scale<int, 2>(), adder<int>()));

    // execute graph
    std::array<int, 4> a = { 1, 2, 3, 4 };
    std::array<int, 4> b = { 10, 10, 10, 10 };

    int                r = 0;
    for (int i = 0; i < 4; ++i) {
        auto [neg, scale, z] = merged.process_one(a[i], b[i]);
        r += z;
    }

    fmt::print("Result of graph execution: {}\n", r);
    return r == 20 ? 0 : 1;
}
