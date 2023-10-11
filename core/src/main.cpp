#include <array>
#include <fmt/core.h>

#include <cassert>

#include <gnuradio-4.0/config.h> // contains the project and compiler flags definitions
#include <gnuradio-4.0/graph.hpp>

namespace grg = gr;

template<typename T>
struct count_source : public grg::Block<count_source<T>> {
    grg::PortOut<T> random;

    constexpr T
    processOne() {
        return 42;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (count_source<T>), random);

template<typename T>
struct expect_sink : public grg::Block<expect_sink<T>> {
    grg::PortIn<T> sink;

    void
    processOne(T value) {
        std::cout << value << std::endl;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (expect_sink<T>), sink);

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public grg::Block<scale<T, Scale, R>> {
    grg::PortIn<T>  original;
    grg::PortOut<R> scaled;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) const noexcept {
        return a * Scale;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, T Scale, typename R), (scale<T, Scale, R>), original, scaled);

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public grg::Block<adder<T>> {
    grg::PortIn<T>  addend0;
    grg::PortIn<T>  addend1;
    grg::PortOut<R> sum;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a, V b) const noexcept {
        return a + b;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename R), (adder<T, R>), addend0, addend1, sum);

using grg::PortType::STREAM, grg::PortDirection::INPUT, grg::PortDirection::OUTPUT;

template<typename T, std::size_t Count = 2>
class duplicate : public grg::Block<duplicate<T, Count>, gr::meta::typelist<grg::PortInNamed<T, "in">>, grg::repeated_ports<Count, T, "out", STREAM, OUTPUT>> {
    using base = grg::Block<duplicate<T, Count>, gr::meta::typelist<grg::PortInNamed<T, "in">>, grg::repeated_ports<Count, T, "out", STREAM, OUTPUT>>;

public:
    using return_type = typename grg::traits::block::return_type<base>;

    [[nodiscard]] constexpr return_type
    processOne(T a) const noexcept {
        return [&a]<std::size_t... Is>(std::index_sequence<Is...>) { return std::make_tuple(((void) Is, a)...); }(std::make_index_sequence<Count>());
    }
};

template<typename T, std::size_t Depth>
    requires(Depth > 0)
struct delay : public grg::Block<delay<T, Depth>> {
    grg::PortIn<T>       in;
    grg::PortOut<T>      out;
    std::array<T, Depth> buffer = {};
    int                  pos    = 0;

    [[nodiscard]] constexpr T
    processOne(T val) noexcept {
        T ret       = buffer[pos];
        buffer[pos] = val;
        if (pos == Depth - 1) {
            pos = 0;
        } else {
            ++pos;
        }
        return ret;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t Depth), (delay<T, Depth>), in, out);

int
main() {
    using grg::merge;
    using grg::mergeByIndex;

    fmt::print("Project compiler: '{}' - version '{}'\n", CXX_COMPILER_ID, CXX_COMPILER_VERSION);
    fmt::print("Project compiler path: '{}' - arg1 '{}'\n", CXX_COMPILER_PATH, CXX_COMPILER_ARG1);
    fmt::print("Project compiler flags: '{}'\n", CXX_COMPILER_FLAGS);

    {
        // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
        auto merged = mergeByIndex<0, 0>(scale<int, -1>(), mergeByIndex<0, 0>(scale<int, 2>(), adder<int>()));

        // execute graph
        std::array<int, 4> a = { 1, 2, 3, 4 };
        std::array<int, 4> b = { 10, 10, 10, 10 };

        int                r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(i, a[i], b[i]);
        }

        fmt::print("Result of graph execution: {}\n", r);

        assert(r == 20);
    }

    {
        auto merged = mergeByIndex<0, 0>(duplicate<int, 2>(), scale<int, 2>());

        // execute graph
        std::array<int, 4> a = { 1, 2, 3, 4 };

        for (std::size_t i = 0; i < 4; ++i) {
            auto tuple    = merged.processOne(i, a[i]);
            auto [r1, r2] = tuple;
            fmt::print("{} {} \n", r1, r2);
        }
    }

    {
        auto merged = merge<"scaled", "addend1">(scale<int, 2>(), adder<int>());

        // execute graph
        std::array<int, 4> a = { 1, 2, 3, 4 };
        std::array<int, 4> b = { 10, 10, 10, 10 };

        int                r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(i, a[i], b[i]);
        }

        fmt::print("Result of graph execution: {}\n", r);

        assert(r == 60);
    }

    {
        auto merged = mergeByIndex<1, 0>(mergeByIndex<0, 0>(duplicate<int, 4>(), scale<int, 2>()), scale<int, 2>());

        // execute graph
        std::array<int, 4> a = { 1, 2, 3, 4 };

        for (std::size_t i = 0; i < 4; ++i) {
            auto tuple            = merged.processOne(i, a[i]);
            auto [r1, r2, r3, r4] = tuple;
            fmt::print("{} {} {} {} \n", r1, r2, r3, r4);
        }
    }

    { auto delayed = delay<int, 2>{}; }

    {
        auto random = count_source<int>{};

        auto merged = mergeByIndex<0, 0>(std::move(random), expect_sink<int>());
        merged.processOne(0);
    }

    {
        auto random = count_source<int>{};

        auto merged = merge<"random", "original">(std::move(random), scale<int, 2>());
        merged.processOne(0);
    }
}
