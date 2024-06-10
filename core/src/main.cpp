#include <array>
#include <fmt/core.h>

#include <cassert>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/config.hpp> // contains the project and compiler flags definitions

template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T> random;

    constexpr T processOne() { return 42; }
};

ENABLE_REFLECTION_FOR_TEMPLATE(CountSource, random);

template<typename T>
struct ExpectSink : public gr::Block<ExpectSink<T>> {
    gr::PortIn<T> sink;

    void processOne(T value) { std::cout << value << std::endl; }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ExpectSink, sink);

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public gr::Block<scale<T, Scale, R>> {
    gr::PortIn<T>  original;
    gr::PortOut<R> scaled;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V a) const noexcept {
        return a * Scale;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, T Scale, typename R), (scale<T, Scale, R>), original, scaled);

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public gr::Block<adder<T>> {
    gr::PortIn<T>  addend0;
    gr::PortIn<T>  addend1;
    gr::PortOut<R> sum;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V a, V b) const noexcept {
        return a + b;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename R), (adder<T, R>), addend0, addend1, sum);
using gr::PortType::STREAM, gr::PortDirection::INPUT, gr::PortDirection::OUTPUT;

template<typename T, std::size_t Count = 2>
class duplicate : public gr::Block<duplicate<T, Count>, gr::meta::typelist<gr::PortInNamed<T, "in">>, gr::repeated_ports<Count, T, "out", STREAM, OUTPUT>> {
    using base = gr::Block<duplicate<T, Count>, gr::meta::typelist<gr::PortInNamed<T, "in">>, gr::repeated_ports<Count, T, "out", STREAM, OUTPUT>>;

public:
    using return_type = typename gr::traits::block::stream_return_type<base>;

    [[nodiscard]] constexpr return_type processOne(T a) const noexcept {
        return [&a]<std::size_t... Is>(std::index_sequence<Is...>) { return std::make_tuple(((void)Is, a)...); }(std::make_index_sequence<Count>());
    }
};

template<typename T, std::size_t Depth>
requires(Depth > 0)
struct delay : public gr::Block<delay<T, Depth>> {
    gr::PortIn<T>        in;
    gr::PortOut<T>       out;
    std::array<T, Depth> buffer = {};
    int                  pos    = 0;

    [[nodiscard]] constexpr T processOne(T val) noexcept {
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

int main() {
    using gr::merge;
    using gr::mergeByIndex;

    fmt::print("Project compiler: '{}' - version '{}'\n", CXX_COMPILER_ID, CXX_COMPILER_VERSION);
    fmt::print("Project compiler path: '{}' - arg1 '{}'\n", CXX_COMPILER_PATH, CXX_COMPILER_ARG1);
    fmt::print("Project compiler flags: '{}'\n", CXX_COMPILER_FLAGS);

    {
        // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
        auto merged = mergeByIndex<0, 0>(scale<int, -1>(), mergeByIndex<0, 0>(scale<int, 2>(), adder<int>()));

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};
        std::array<int, 4> b = {10, 10, 10, 10};

        int r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(i, a[i], b[i]);
        }

        fmt::print("Result of graph execution: {}\n", r);

        assert(r == 20);
    }

    {
        auto merged = mergeByIndex<0, 0>(duplicate<int, 2>(), scale<int, 2>());

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};

        for (std::size_t i = 0; i < 4; ++i) {
            auto tuple    = merged.processOne(i, a[i]);
            auto [r1, r2] = tuple;
            fmt::print("{} {} \n", r1, r2);
        }
    }

    {
        auto merged = merge<"scaled", "addend1">(scale<int, 2>(), adder<int>());

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};
        std::array<int, 4> b = {10, 10, 10, 10};

        int r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(i, a[i], b[i]);
        }

        fmt::print("Result of graph execution: {}\n", r);

        assert(r == 60);
    }

    {
        auto merged = mergeByIndex<1, 0>(mergeByIndex<0, 0>(duplicate<int, 4>(), scale<int, 2>()), scale<int, 2>());

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};

        for (std::size_t i = 0; i < 4; ++i) {
            auto tuple            = merged.processOne(i, a[i]);
            auto [r1, r2, r3, r4] = tuple;
            fmt::print("{} {} {} {} \n", r1, r2, r3, r4);
        }
    }

    { auto delayed = delay<int, 2>{}; }

    {
        auto random = CountSource<int>{};

        auto merged = mergeByIndex<0, 0>(std::move(random), ExpectSink<int>());
        merged.processOne(0);
    }

    {
        auto random = CountSource<int>{};

        auto merged = merge<"random", "original">(std::move(random), scale<int, 2>());
        merged.processOne(0);
    }
}
