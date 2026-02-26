#include <array>
#include <cassert>

#include <format>
#include <print>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/config.hpp> // contains the project and compiler flags definitions

template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T> random;

    GR_MAKE_REFLECTABLE(CountSource, random);

    constexpr T processOne() { return 42; }
};

template<typename T>
struct ExpectSink : public gr::Block<ExpectSink<T>> {
    gr::PortIn<T> sink;

    GR_MAKE_REFLECTABLE(ExpectSink, sink);

    void processOne(T value) { std::println("{}", value); }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public gr::Block<scale<T, Scale, R>> {
    gr::PortIn<T>  original;
    gr::PortOut<R> scaled;

    GR_MAKE_REFLECTABLE(scale, original, scaled);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V a) const noexcept {
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public gr::Block<adder<T>> {
    gr::PortIn<T>  addend0;
    gr::PortIn<T>  addend1;
    gr::PortOut<R> sum;

    GR_MAKE_REFLECTABLE(adder, addend0, addend1, sum);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V a, V b) const noexcept {
        return a + b;
    }
};

template<typename T>
struct duplicate : public gr::Block<duplicate<T>> {
    gr::PortIn<T>                              in;
    std::tuple<gr::PortOut<T>, gr::PortOut<T>> out;

    GR_MAKE_REFLECTABLE(duplicate, in, out);

    [[nodiscard]] constexpr auto processOne(T a) const noexcept {
        return [&a]<std::size_t... Is>(std::index_sequence<Is...>) { return std::tuple{((void)Is, a)...}; }(std::make_index_sequence<std::tuple_size_v<decltype(out)>>());
    }
};

template<typename T, std::size_t Depth>
requires(Depth > 0)
struct delay : public gr::Block<delay<T, Depth>> {
    gr::PortIn<T>        in;
    gr::PortOut<T>       out;
    std::array<T, Depth> buffer = {};
    int                  pos    = 0;

    GR_MAKE_REFLECTABLE(delay, in, out);

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

template<typename TDescr, typename TPort>
void reflectPort(auto idx, const TPort& obj) {
    std::print("  {}.       name: {} / {}\n"
               "           type: {} / {}\n"
               "     descriptor: {}\n",
        idx.value, TDescr::Name.view(), obj.metaInfo.name, gr::refl::type_name<typename TDescr::value_type>.view(), gr::refl::type_name<typename TPort::value_type>.view(), gr::refl::type_name<TDescr>.view());
}

template<gr::refl::reflectable TBlock>
void reflectBlock(const TBlock& obj) {
    using inputs  = gr::traits::block::stream_input_ports<TBlock>;
    using outputs = gr::traits::block::stream_output_ports<TBlock>;
    std::print("reflecting on '{}'\n"
               "# reflectable data members: {}\n"
               "# input streams: {}\n"
               "# output streams: {}\n",
        gr::refl::type_name<TBlock>.view(), gr::refl::data_member_count<TBlock>, inputs::size(), outputs::size());
    std::print("-- input streams:\n");
    inputs::for_each([&]<typename P>(auto idx, P*) { reflectPort<P>(idx, P::getPortObject(obj)); });
    std::print("-- output streams:\n");
    outputs::for_each([&]<typename P>(auto idx, P*) { reflectPort<P>(idx, P::getPortObject(obj)); });
}

int main() {
    using gr::merge;
    using gr::mergeByIndex;

    std::print("Project compiler: '{}' - version '{}'\n", CXX_COMPILER_ID, CXX_COMPILER_VERSION);
    std::print("Project compiler path: '{}' - arg1 '{}'\n", CXX_COMPILER_PATH, CXX_COMPILER_ARG1);
    std::print("Project compiler flags: '{}'\n", CXX_COMPILER_FLAGS);

    {
        // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
        auto merged = mergeByIndex<0, 0>(scale<int, -1>(), mergeByIndex<0, 0>(scale<int, 2>(), adder<int>()));
        reflectBlock(merged);

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};
        std::array<int, 4> b = {10, 10, 10, 10};

        // -2 + 10
        // -4 + 10
        // -6 + 10
        // -8 + 10

        int r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(a[i], b[i]);
        }

        constexpr int expect = 20;

        std::print("Result of graph execution: {}, expected: {}\n", r, expect);

        assert(r == expect);
    }

    {
        auto merged = mergeByIndex<0, 0>(duplicate<int>(), scale<int, 2>());
        static_assert(std::same_as<decltype(merged)::ReturnType, std::tuple<int, int>>);
        reflectBlock(merged);

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};

        for (std::size_t i = 0; i < 4; ++i) {
            auto tuple    = merged.processOne(a[i]);
            auto [r1, r2] = tuple;
            static_assert(std::same_as<std::remove_cvref_t<decltype(r1)>, int>);
            static_assert(std::same_as<std::remove_cvref_t<decltype(r2)>, int>);
            std::print("{} {}, expected: {} {} \n", r1, r2, a[i] * 2, a[i]);
        }
    }

    {
        auto merged = merge<"scaled", "addend1">(scale<int, 2>(), adder<int>());
        reflectBlock(merged);

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};
        std::array<int, 4> b = {10, 10, 10, 10};

        int r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(a[i], b[i]);
        }

        std::print("Result of graph execution: {}\n", r);

        assert(r == 60);
    }

    {
        auto merged = merge<"out1", "original">(merge<"out0", "original">(duplicate<int>(), scale<int, 2>()), scale<int, 2>());
        reflectBlock(merged);

        // execute graph
        std::array<int, 4> a = {1, 2, 3, 4};

        for (std::size_t i = 0; i < 4; ++i) {
            auto tuple    = merged.processOne(a[i]);
            auto [r1, r2] = tuple;
            std::print("{} {} \n", r1, r2);
        }
    }

    { auto delayed = delay<int, 2>{}; }

    {
        auto random = CountSource<int>{};

        auto merged = mergeByIndex<0, 0>(std::move(random), ExpectSink<int>());
        merged.processOne();
    }

    {
        auto random = CountSource<int>{};

        auto merged = merge<"random", "original">(std::move(random), scale<int, 2>());
        std::print("{}\n", merged.processOne());
    }
}
