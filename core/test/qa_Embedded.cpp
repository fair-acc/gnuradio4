#include <array>

#include <boost/ut.hpp>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/Port.hpp>

#include "EmbeddedDemoBlocks.hpp"

using namespace boost::ut;
using namespace gr::testing::embedded;

const boost::ut::suite<"Embedded freestanding (compiled with -fno-rtti)"> _embedded = [] {
    "merge-API: 2x in -> adder -> scale-by-2 -> scale-by-minus1"_test = [] {
        auto merged = gr::MergeByIndex<scale<int, -1>, 0, //
            gr::MergeByIndex<scale<int, 2>, 0, adder<int>, 0>, 0>();

        std::array<int, 4> a = {1, 2, 3, 4};
        std::array<int, 4> b = {10, 10, 10, 10};

        int r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(a[i], b[i]);
        }
        expect(eq(r, 20)) << "Σ (-2(a + b)) for a=[1,2,3,4], b=[10,10,10,10]";
    };

    "merge-API: duplicate -> scale-by-2 (port-index resolution)"_test = [] {
        auto merged = gr::MergeByIndex<duplicate<int>, 0, scale<int, 2>, 0>();
        static_assert(std::same_as<decltype(merged)::ReturnType, std::tuple<int, int>>);

        std::array<int, 4> a = {1, 2, 3, 4};
        for (std::size_t i = 0; i < 4; ++i) {
            auto [r1, r2] = merged.processOne(a[i]);
            expect(eq(r1, a[i])) << "duplicate's remaining output (tuple element 0)";
            expect(eq(r2, a[i] * 2)) << "scaled output (tuple element 1)";
        }
    };

    "merge-API: scale-by-2 -> adder (named-port resolution)"_test = [] {
        auto merged = gr::Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();

        std::array<int, 4> a = {1, 2, 3, 4};
        std::array<int, 4> b = {10, 10, 10, 10};

        int r = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r += merged.processOne(a[i], b[i]);
        }
        expect(eq(r, 60)) << "Σ (2a + b) for a=[1,2,3,4], b=[10,10,10,10]";
    };

    "merge-API: nested duplicate -> scale + scale (chain reuse)"_test = [] {
        auto merged = gr::Merge<                                          //
            gr::Merge<duplicate<int>, "out0", scale<int, 2>, "original">, //
            "out1", scale<int, 2>, "original">();

        std::array<int, 4> a = {1, 2, 3, 4};
        for (std::size_t i = 0; i < 4; ++i) {
            auto [r1, r2] = merged.processOne(a[i]);
            expect(eq(r1, a[i] * 2)) << "first scaled output";
            expect(eq(r2, a[i] * 2)) << "second scaled output";
        }
    };

    "merge-API: CountSource -> ExpectSink (source-to-sink terminator)"_test = [] {
        auto merged = gr::MergeByIndex<CountSource<int>, 0, ExpectSink<int>, 0>();
        merged.processOne();
        expect(true) << "graph executes without throwing or allocating";
    };

    "merge-API: CountSource -> scale-by-2"_test = [] {
        auto merged = gr::Merge<CountSource<int>, "random", scale<int, 2>, "original">();
        expect(eq(merged.processOne(), 84)) << "42 * 2";
    };
};

int main() { /* tests are statically executed */ }
