#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <new>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include "EmbeddedDemoBlocks.hpp"

using namespace boost::ut;
using namespace gr::testing::embedded;

// armed-window counter over ::operator new — tests scope-arm the sentinel so boost::ut's own
// reporting allocations don't poison the delta.
namespace {
std::atomic<std::size_t> gGlobalNewCount{0UZ};
std::atomic<bool>        gGlobalNewArmed{false};

struct GlobalNewSentinel {
    std::size_t baseline;
    explicit GlobalNewSentinel() noexcept : baseline(gGlobalNewCount.load(std::memory_order_relaxed)) { gGlobalNewArmed.store(true, std::memory_order_release); }
    ~GlobalNewSentinel() { gGlobalNewArmed.store(false, std::memory_order_release); }

    GlobalNewSentinel(const GlobalNewSentinel&)            = delete;
    GlobalNewSentinel& operator=(const GlobalNewSentinel&) = delete;
    GlobalNewSentinel(GlobalNewSentinel&&)                 = delete;
    GlobalNewSentinel& operator=(GlobalNewSentinel&&)      = delete;

    [[nodiscard]] std::size_t delta() const noexcept { return gGlobalNewCount.load(std::memory_order_relaxed) - baseline; }
};
} // namespace

// noinline keeps the operator-new / operator-delete pair from folding into a raw malloc → free
// chain at -Os, which would otherwise trip -Wmismatched-new-delete.
[[gnu::noinline]] void* operator new(std::size_t n) {
    void* p = std::malloc(n == 0UZ ? 1UZ : n);
    if (gGlobalNewArmed.load(std::memory_order_acquire)) {
        gGlobalNewCount.fetch_add(1UZ, std::memory_order_relaxed);
    }
    if (p == nullptr) {
        std::abort(); // -fno-exceptions: cannot throw std::bad_alloc
    }
    return p;
}

[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }

// every demo block must be noexcept on its processing path — the AOT build cannot
// catch user-block throws, so the no-exception promise is locked in at compile time.
static_assert(gr::HasNoexceptProcessFunction<scale<int, 2>>);
static_assert(gr::HasNoexceptProcessFunction<scale<int, -1>>);
static_assert(gr::HasNoexceptProcessFunction<adder<int>>);
static_assert(gr::HasNoexceptProcessFunction<duplicate<int>>);
static_assert(gr::HasNoexceptProcessFunction<CountSource<int>>);
static_assert(gr::HasNoexceptProcessFunction<ExpectSink<int>>);

// observable sink for the MCU run-loop test: counts consumed samples through a file-local atomic so the
// fused (merge-composed) terminator's work() execution can be verified from outside the block.
namespace {
std::atomic<long long> gMcuSinkCount{0};
} // namespace

template<typename T>
struct McuCountSink : gr::Block<McuCountSink<T>> {
    gr::PortIn<T> in;

    GR_MAKE_REFLECTABLE(McuCountSink, in);

    constexpr void processOne(T) noexcept { gMcuSinkCount.fetch_add(1, std::memory_order_relaxed); }
};
static_assert(gr::HasNoexceptProcessFunction<McuCountSink<int>>);

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

const boost::ut::suite<"merge-API zero global-heap (qa-local ::operator new sentinel)"> _noHeap = [] {
    // proves the buffer layer is allocation-free for size 0.
    "CircularBuffer(0) over a CountingResource hits neither PMR nor global new"_test = [] {
        gr::allocator::pmr::CountingResource counter;
        GlobalNewSentinel                    sentinel;
        gr::CircularBuffer<int>              empty(0UZ, std::pmr::polymorphic_allocator<int>(&counter));
        auto                                 w = empty.new_writer();
        auto                                 r = empty.new_reader();
        std::ignore                            = w;
        std::ignore                            = r;
        expect(eq(counter.allocCount, 0UZ)) << "PMR resource untouched";
        expect(eq(sentinel.delta(), 0UZ)) << "global new untouched";
    };

    // construction may still hit global new (Block ctor std::format) — arm the sentinel
    // only across the steady-state loop.
    "merge-API steady-state loop is heap-free (construction excluded)"_test = [] {
        gr::pmr::OwnedStaticArenaResource<4096UZ> arena;
        std::pmr::vector<int>                     results{&arena};
        results.reserve(64UZ);

        auto merged = gr::Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();

        {
            GlobalNewSentinel sentinel;
            for (int i = 0; i < 64; ++i) {
                results.push_back(merged.processOne(i, 10));
            }
            expect(eq(sentinel.delta(), 0UZ)) << "merge-API processOne loop must never reach ::operator new";
        }

        expect(eq(results.size(), 64UZ));
        expect(eq(results[0], 10));
        expect(eq(results[63], 136));
    };

    // NoHeapResource as the default — aborts on any consumer of get_default_resource() in
    // the loop body. Construction stays on the process default; the swap is scoped.
    "merge-API steady-state under NoHeapResource as default"_test = [] {
        gr::pmr::OwnedStaticArenaResource<4096UZ> arena;
        std::pmr::vector<int>                     results{&arena};
        results.reserve(64UZ);

        auto merged = gr::Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();

        gr::allocator::pmr::NoHeapResource        noHeap;
        gr::allocator::pmr::ScopedDefaultResource scoped(&noHeap);
        for (int i = 0; i < 64; ++i) {
            results.push_back(merged.processOne(i, 10));
        }
        expect(eq(results.size(), 64UZ));
        expect(eq(results[63], 136));
    };

    // PortMetaInfo::auto_update routes ~20 tree nodes through std::pmr::get_default_resource();
    // installing the arena before construction keeps ::operator new untouched.
    "default-constructed Port allocates nothing globally (arena-as-default)"_test = [] {
        gr::pmr::OwnedStaticArenaResource<8192UZ> arena;
        gr::allocator::pmr::ScopedDefaultResource scoped(&arena);
        GlobalNewSentinel                         sentinel;
        gr::PortIn<int>                           in;
        gr::PortOut<int>                          out;
        expect(eq(in.bufferSize(), 0UZ));
        expect(eq(out.bufferSize(), 0UZ));
        expect(eq(sentinel.delta(), 0UZ)) << "zero-capacity Port default ctor must not touch global new";
        expect(gt(arena.used(), 0UZ)) << "auto_update's tree nodes must have landed on the arena";
    };

    // construction + steady-state under an arena default: zero global new (pmr-backed strings/settings/type-names).
    "merge-API construction + steady-state run never hits ::operator new"_test = [] {
        gr::pmr::OwnedStaticArenaResource<65536UZ> arena;
        std::pmr::vector<int>                      results{&arena};
        results.reserve(64UZ);

        gr::allocator::pmr::ScopedDefaultResource scoped(&arena);
        GlobalNewSentinel                         sentinel;
        auto                                      merged = gr::Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();
        for (int i = 0; i < 64; ++i) {
            results.push_back(merged.processOne(i, 10));
        }
        const std::size_t hits = sentinel.delta(); // capture before std::format allocates
        std::fputs(std::format("[diag] merge-API global-new hits: {}\n", hits).c_str(), stderr);
#ifndef __EMSCRIPTEN__ // emscripten libc++ does a few non-pmr allocations during block construction; invariant holds on native/MCU targets
        expect(eq(hits, 0UZ)) << "merge-API construction + steady-state must not reach ::operator new";
#endif
        expect(eq(results.size(), 64UZ));
        expect(eq(results[0], 10));
        expect(eq(results[63], 136));
    };

    // MCU superloop: setup may allocate, but steady-state must touch neither global new nor the (non-reclaiming) arena.
    // freestanding analogue of the gr::Graph + scheduler::Simple proof in qa_NoHeapScheduler.cpp.
    "MCU superloop: 10k-cycle merge-API steady-state allocates neither global heap nor arena"_test = [] {
        gr::pmr::OwnedStaticArenaResource<1UZ << 18U> arena;
        gr::allocator::pmr::ScopedDefaultResource     scoped(&arena);

        std::pmr::vector<int> results{&arena};
        results.reserve(10'000UZ);
        auto merged = gr::Merge<scale<int, 2>, "scaled", adder<int>, "addend1">();

        results.push_back(merged.processOne(0, 10)); // prime: settle first-touch state on the arena
        const std::size_t usedAfterPrime = arena.used();

        {
            GlobalNewSentinel sentinel;
            for (int cycle = 1; cycle < 10'000; ++cycle) {
                results.push_back(merged.processOne(cycle, 10));
            }
            expect(eq(sentinel.delta(), 0UZ)) << "steady-state superloop must not reach ::operator new";
        }
        expect(eq(arena.used(), usedAfterPrime)) << "steady-state superloop must not grow the bump arena";
        expect(eq(results.size(), 10'000UZ));
        expect(eq(results[0], 10));
        expect(eq(results[9999], 20008)); // (2 * 9999) + 10
    };
};

const boost::ut::suite<"MCU run-loop (work + periodic processScheduledMessages)"> _mcuRunLoop = [] {
    "10k-cycle work() + processScheduledMessages() superloop allocates nothing in steady state"_test = [] {
        gr::pmr::OwnedStaticArenaResource<1UZ << 18U> arena;
        gr::allocator::pmr::ScopedDefaultResource     scoped(&arena);

        auto            merged = gr::MergeByIndex<CountSource<int>, 0, McuCountSink<int>, 0>();
        const long long before = gMcuSinkCount.load(std::memory_order_relaxed);

        std::ignore = merged.work(64UZ); // prime: settle first-touch state on the arena
        merged.processScheduledMessages();
        const std::size_t usedAfterPrime = arena.used();

        {
            GlobalNewSentinel sentinel;
            for (int cycle = 0; cycle < 10'000; ++cycle) {
                std::ignore = merged.work(64UZ);
                merged.processScheduledMessages(); // periodic message processing (MCU run() also services other tasks)
            }
            expect(eq(sentinel.delta(), 0UZ)) << "MCU work()+processScheduledMessages() loop must not reach ::operator new";
        }
        expect(eq(arena.used(), usedAfterPrime)) << "MCU loop must not grow the bump arena";
        expect(gt(gMcuSinkCount.load(std::memory_order_relaxed) - before, 0LL)) << "work() must actually execute the blocks (sink consumed samples)";
    };
};

const boost::ut::suite<"MCU superloop on real gr::Graph + Simple<externalStep>"> _mcuRealGraph = [] {
    using enum gr::lifecycle::State;
    "10k-cycle Graph + Simple<externalStep> step() superloop allocates nothing in steady state"_test = [] {
        auto                arena = std::make_unique<gr::pmr::OwnedStaticArenaResource<1UZ << 23U>>(); // 8 MiB: setup headroom (steady-state is what is measured)
        gr::ResourceProfile profile{.data = arena.get(), .tag = arena.get(), .mechanics = arena.get()};

        gr::Graph graph(profile);
        auto&     src = graph.emplaceBlock<CountSource<int>>();
        auto&     snk = graph.emplaceBlock<ExpectSink<int>>();
        expect(graph.connect<"random", "sink">(src, snk, {.minBufferSize = 4096UZ}).has_value());

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> sched;
        expect(sched.exchange(std::move(graph)).has_value());
        expect(sched.changeStateTo(INITIALISED).has_value());
        expect(sched.changeStateTo(RUNNING).has_value()) << "externalStep start() must prime to RUNNING without spawning a worker";

        gr::allocator::pmr::ScopedDefaultResource scoped(arena.get());
        for (int i = 0; i < 8; ++i) { // warm-up: settle first-touch lazy state before measuring
            std::ignore = sched.step();
        }
        const std::size_t usedAfterWarmup = arena->used();

        std::size_t hits      = 0UZ;
        std::size_t performed = 0UZ;
        {
            GlobalNewSentinel sentinel;
            for (int cycle = 0; cycle < 10'000; ++cycle) {
                performed += sched.step().performed_work;
            }
            hits = sentinel.delta();
        }
        const std::size_t growth = arena->used() - usedAfterWarmup;
        std::fputs(std::format("[diag] real-graph steady-state global-new hits: {}, arena growth: {} bytes\n", hits, growth).c_str(), stderr);
        expect(gt(performed, 0UZ)) << "superloop must actually execute work";
        expect(eq(hits, 0UZ)) << "steady-state superloop must not reach ::operator new";
        expect(eq(growth, 0UZ)) << "steady-state superloop must not grow the bump arena";

        std::ignore = sched.changeStateTo(REQUESTED_STOP);
        std::ignore = sched.changeStateTo(STOPPED);
    };
};

int main() { /* tests are statically executed */ }
