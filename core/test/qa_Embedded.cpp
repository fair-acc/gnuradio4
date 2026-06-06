#include <array>
#include <memory_resource>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Port.hpp>

#include "EmbeddedDemoBlocks.hpp"

using namespace boost::ut;
using namespace gr::testing::embedded;

// every demo block must be noexcept on its processing path — the AOT build cannot
// catch user-block throws, so the no-exception promise is locked in at compile time.
static_assert(gr::HasNoexceptProcessFunction<scale<int, 2>>);
static_assert(gr::HasNoexceptProcessFunction<scale<int, -1>>);
static_assert(gr::HasNoexceptProcessFunction<adder<int>>);
static_assert(gr::HasNoexceptProcessFunction<duplicate<int>>);
static_assert(gr::HasNoexceptProcessFunction<CountSource<int>>);
static_assert(gr::HasNoexceptProcessFunction<ExpectSink<int>>);

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

    "PMR-backed run: arena + CountingResource + std::pmr::vector results sink"_test = [] {
        // 4 KiB static arena — embedded SRAM budget. CountingResource wraps it so we can
        // assert the test path's allocations stayed inside the arena and produced no leak.
        gr::pmr::OwnedStaticArenaResource<4096UZ> arena;
        gr::allocator::pmr::CountingResource      counter;
        counter.upstream = &arena;

        auto                  merged = gr::Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();
        std::pmr::vector<int> results{&counter};
        results.reserve(64UZ);

        const std::size_t bytesAfterReserve = arena.used();
        expect(eq(bytesAfterReserve, 64UZ * sizeof(int))) << "reserve(64) consumes exactly int[64]";
        expect(eq(counter.allocCount, 1UZ));

        for (int i = 0; i < 64; ++i) {
            results.push_back(merged.processOne(i, 10));
        }
        expect(eq(results.size(), 64UZ));
        expect(eq(arena.used(), bytesAfterReserve)) << "no reallocation inside reserved capacity";
        expect(eq(counter.allocCount, 1UZ)) << "still one allocation total";
        expect(eq(results[0], 10)) << "(2*0) + 10";
        expect(eq(results[63], 136)) << "(2*63) + 10";
    };
};

int main() { /* tests are statically executed */ }
