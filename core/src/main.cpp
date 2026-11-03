#include <array>
#include <cassert>

#include <format>
#include <print>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/config.hpp> // contains the project and compiler flags definitions

#include "EmbeddedDemoBlocks.hpp"

using gr::testing::embedded::adder;
using gr::testing::embedded::CountSource;
using gr::testing::embedded::duplicate;
using gr::testing::embedded::ExpectSink;
using gr::testing::embedded::scale;

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
    inputs::for_each([&obj]<typename P>(auto idx, P*) { reflectPort<P>(idx, P::getPortObject(obj)); });
    std::print("-- output streams:\n");
    outputs::for_each([&obj]<typename P>(auto idx, P*) { reflectPort<P>(idx, P::getPortObject(obj)); });
}

int main() {
    using gr::Merge;
    using gr::MergeByIndex;

    std::print("Project compiler: '{}' - version '{}'\n", CXX_COMPILER_ID, CXX_COMPILER_VERSION);
    std::print("Project compiler path: '{}' - arg1 '{}'\n", CXX_COMPILER_PATH, CXX_COMPILER_ARG1);
    std::print("Project compiler flags: '{}'\n", CXX_COMPILER_FLAGS);

    {
        // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
        auto merged = MergeByIndex<scale<int, -1>, 0, //
            MergeByIndex<scale<int, 2>, 0, adder<int>, 0>, 0>();
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
        auto merged = MergeByIndex<duplicate<int>, 0, scale<int, 2>, 0>();
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
        auto merged = Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();
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
        auto merged = Merge<                                          //
            Merge<duplicate<int>, "out0", scale<int, 2>, "original">, //
            "out1", scale<int, 2>, "original">();
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
        auto merged = MergeByIndex<CountSource<int>, 0, ExpectSink<int>, 0>();
        merged.processOne();
    }

    {
        auto merged = Merge<CountSource<int>, "random", scale<int, 2>, "original">();
        std::print("{}\n", merged.processOne());
    }
}
