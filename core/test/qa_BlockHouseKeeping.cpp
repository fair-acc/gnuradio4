#include <atomic>
#include <cstddef>
#include <memory_resource>
#include <string>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

// Test-only block that observes scheduler-driven houseKeeping invocations. Shadows
// Block<Derived>::houseKeeping; calls the base after recording so port iteration still runs.
struct ObservedBlock : public gr::Block<ObservedBlock> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;
    GR_MAKE_REFLECTABLE(ObservedBlock, in, out);

    inline static std::atomic<std::uint64_t> kCalls{0};
    inline static std::atomic<std::uint8_t>  kLastPolicy{0};
    inline static std::atomic<std::uint8_t>  kLastDepth{0};
    inline static std::atomic<std::uint32_t> kPolicyMask{0}; // OR of (1 << policy) over every call
    inline static std::atomic<std::uint32_t> kDepthMask{0};  // OR of (1 << depth) over every call

    static void resetObserved() noexcept {
        kCalls.store(0U, std::memory_order_relaxed);
        kPolicyMask.store(0U, std::memory_order_relaxed);
        kDepthMask.store(0U, std::memory_order_relaxed);
    }

    constexpr void houseKeeping(gr::HouseKeepPolicy policy = gr::HouseKeepPolicy::Normal, gr::HouseKeepDepth depth = gr::HouseKeepDepth::Deep) noexcept {
        kCalls.fetch_add(1U, std::memory_order_relaxed);
        kLastPolicy.store(static_cast<std::uint8_t>(policy), std::memory_order_relaxed);
        kLastDepth.store(static_cast<std::uint8_t>(depth), std::memory_order_relaxed);
        kPolicyMask.fetch_or(1U << static_cast<unsigned>(policy), std::memory_order_relaxed);
        kDepthMask.fetch_or(1U << static_cast<unsigned>(depth), std::memory_order_relaxed);
        gr::Block<ObservedBlock>::houseKeeping(policy, depth);
    }

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

// PMR resource that tracks alloc/dealloc volume and current/peak live bytes. Lets the
// buffer-level tests observe reclamation through the allocator path (no global state).
struct CountingResource : public std::pmr::memory_resource {
    std::pmr::memory_resource* upstream = std::pmr::get_default_resource();
    std::atomic<std::uint64_t> allocCalls{0};
    std::atomic<std::uint64_t> deallocCalls{0};
    std::atomic<std::uint64_t> deallocBytes{0};
    std::atomic<std::uint64_t> liveBytes{0};
    std::atomic<std::uint64_t> peakBytes{0};

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        allocCalls.fetch_add(1U, std::memory_order_relaxed);
        const std::uint64_t now      = liveBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        std::uint64_t       observed = peakBytes.load(std::memory_order_relaxed);
        while (now > observed && !peakBytes.compare_exchange_weak(observed, now, std::memory_order_relaxed)) {
        }
        return upstream->allocate(bytes, alignment);
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        deallocCalls.fetch_add(1U, std::memory_order_relaxed);
        deallocBytes.fetch_add(bytes, std::memory_order_relaxed);
        liveBytes.fetch_sub(bytes, std::memory_order_relaxed);
        upstream->deallocate(p, bytes, alignment);
    }
    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

} // namespace

const boost::ut::suite<"Block::houseKeeping plumbing"> _ = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "Normal policy fires houseKeeping on the message-handling gate"_test = [] {
        ObservedBlock::resetObserved();

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t{16000U}}, {"name", "TagSource"}, {"verbose_console", false}});
        auto& mid  = testGraph.emplaceBlock<ObservedBlock>({{"name", "Observed"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSink"}, {"log_samples", false}, {"log_tags", false}});
        expect(testGraph.connect<"out", "in">(src, mid).has_value()) << boost::ut::fatal;
        expect(testGraph.connect<"out", "in">(mid, sink).has_value()) << boost::ut::fatal;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value()) << boost::ut::fatal;
        expect(sched.runAndWait().has_value()) << boost::ut::fatal;

        expect(ObservedBlock::kCalls.load(std::memory_order_relaxed) > 0U) << "Normal: scheduler should invoke houseKeeping";
        expect((ObservedBlock::kPolicyMask.load(std::memory_order_relaxed) & (1U << static_cast<unsigned>(HouseKeepPolicy::Normal))) != 0U) << "the periodic message-gate pass propagates Normal";
        expect((ObservedBlock::kDepthMask.load(std::memory_order_relaxed) & (1U << static_cast<unsigned>(HouseKeepDepth::Deep))) != 0U);
    };

    "Light policy skips the scheduler-driven houseKeeping call"_test = [] {
        ObservedBlock::resetObserved();

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t{4000U}}, {"name", "TagSource"}, {"verbose_console", false}});
        auto& mid  = testGraph.emplaceBlock<ObservedBlock>({{"name", "Observed"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSink"}, {"log_samples", false}, {"log_tags", false}});
        expect(testGraph.connect<"out", "in">(src, mid).has_value()) << boost::ut::fatal;
        expect(testGraph.connect<"out", "in">(mid, sink).has_value()) << boost::ut::fatal;

        gr::scheduler::Simple<> sched;
        sched.house_keeping_policy = HouseKeepPolicy::Light;
        expect(sched.exchange(std::move(testGraph)).has_value()) << boost::ut::fatal;
        expect(sched.runAndWait().has_value()) << boost::ut::fatal;

        expect(ObservedBlock::kCalls.load(std::memory_order_relaxed) == 0U) << "Light: scheduler must NOT invoke houseKeeping (intrinsic path only)";
    };

    "Shallow depth setting reaches the block"_test = [] {
        ObservedBlock::resetObserved();

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t{4000U}}, {"name", "TagSource"}, {"verbose_console", false}});
        auto& mid  = testGraph.emplaceBlock<ObservedBlock>({{"name", "Observed"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSink"}, {"log_samples", false}, {"log_tags", false}});
        expect(testGraph.connect<"out", "in">(src, mid).has_value()) << boost::ut::fatal;
        expect(testGraph.connect<"out", "in">(mid, sink).has_value()) << boost::ut::fatal;

        gr::scheduler::Simple<> sched;
        sched.house_keeping_depth = HouseKeepDepth::Shallow;
        expect(sched.exchange(std::move(testGraph)).has_value()) << boost::ut::fatal;
        expect(sched.runAndWait().has_value()) << boost::ut::fatal;

        expect(ObservedBlock::kCalls.load(std::memory_order_relaxed) > 0U);
        expect((ObservedBlock::kDepthMask.load(std::memory_order_relaxed) & (1U << static_cast<unsigned>(HouseKeepDepth::Shallow))) != 0U) << "configured Shallow reaches the block via the periodic gate";
    };

    "graph teardown drains buffers with one Aggressive/Deep pass"_test = [] {
        // The periodic message-gate pass uses Normal/Deep; the post-quiescence drain in
        // waitDone() uses Aggressive/Deep and is the last houseKeeping a block observes
        // once runAndWait() returns (workers joined, no work() touching the buffers).
        ObservedBlock::resetObserved();

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t{16000U}}, {"name", "TagSource"}, {"verbose_console", false}});
        auto& mid  = testGraph.emplaceBlock<ObservedBlock>({{"name", "Observed"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSink"}, {"log_samples", false}, {"log_tags", false}});
        expect(testGraph.connect<"out", "in">(src, mid).has_value()) << boost::ut::fatal;
        expect(testGraph.connect<"out", "in">(mid, sink).has_value()) << boost::ut::fatal;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value()) << boost::ut::fatal;
        expect(sched.runAndWait().has_value()) << boost::ut::fatal;

        expect(ObservedBlock::kCalls.load(std::memory_order_relaxed) > 0U) << boost::ut::fatal;
        expect(ObservedBlock::kLastPolicy.load(std::memory_order_relaxed) == static_cast<std::uint8_t>(HouseKeepPolicy::Aggressive)) << "teardown drain must be the last pass";
        expect(ObservedBlock::kLastDepth.load(std::memory_order_relaxed) == static_cast<std::uint8_t>(HouseKeepDepth::Deep));
    };

    "intrinsic writer-pressure path returns memory to the PMR pool"_test = [] {
        // Buffer size 16; pressure threshold = 25% × 16 = 4 slots remaining. Pattern:
        // write 14 (publish=14, reader=0, remaining=2 → pressure), consume 1 (reader=1,
        // remaining=3 → still pressure), then trigger. Each 64-char std::pmr::string
        // forces an allocation on the counting resource; shrink_to_fit() on reclaim must
        // produce at least one matching deallocation.
        CountingResource                                  counter;
        std::pmr::polymorphic_allocator<std::pmr::string> alloc(&counter);
        gr::CircularBuffer<std::pmr::string>              buffer{16UZ, alloc};
        auto                                              writer = buffer.new_writer();
        auto                                              reader = buffer.new_reader();

        for (std::size_t i = 0UZ; i < 14UZ; ++i) {
            auto span = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            expect(!span.empty()) << boost::ut::fatal;
            span[0].assign(64UZ, 'x');
            span.publish(1UZ);
        }
        {
            auto consumeSpan = reader.get(1UZ);
            expect(consumeSpan.consume(1UZ)) << boost::ut::fatal;
        }

        const std::uint64_t deallocBefore = counter.deallocCalls.load(std::memory_order_relaxed);
        auto                span          = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
        expect(!span.empty()) << boost::ut::fatal;
        span[0].clear();
        span.publish(1UZ);

        const std::uint64_t deallocAfter = counter.deallocCalls.load(std::memory_order_relaxed);
        expect(deallocAfter > deallocBefore) << "Deep reclaim under writer pressure should have returned memory to the counting PMR resource";
    };

    "sustained 1 MB chunk traffic stays below buffer-sized peak"_test = [] {
        // Drives 12 rounds × 12 chunks-per-round (= 144 × 1 MB writes) through a 16-slot
        // buffer. The 12 < 16 round size keeps the ring below the pressure threshold so the
        // produce-then-drain cycle never blocks; every round overwrites slot[j % 16] via
        // move-assignment (or via the writer-pressure reclaim hook when it fires). Peak heap
        // must stay bounded by buffer.size() × chunk_bytes regardless — confirms there is
        // no monotonic growth across rounds.
        constexpr std::size_t kChunkBytes  = 1024UZ * 1024UZ;
        constexpr std::size_t kBufferSlots = 16UZ;
        constexpr std::size_t kRoundSize   = 12UZ;
        constexpr std::size_t kRounds      = 12UZ;
        // CircularBuffer doubles its slot array (wrap-free read window), so the natural
        // upper bound is 2 × kBufferSlots × kChunkBytes plus ~1 KiB of slot-array metadata.
        constexpr std::uint64_t kPeakLimit = (2UZ * kBufferSlots * kChunkBytes) + (4UZ * 1024UZ);

        using ChunkT = std::pmr::vector<std::byte>;
        CountingResource                        counter;
        std::pmr::polymorphic_allocator<ChunkT> alloc(&counter);
        gr::CircularBuffer<ChunkT>              buffer{kBufferSlots, alloc};
        auto                                    writer = buffer.new_writer();
        auto                                    reader = buffer.new_reader();

        for (std::size_t r = 0UZ; r < kRounds; ++r) {
            for (std::size_t j = 0UZ; j < kRoundSize; ++j) {
                auto wspan = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
                expect(!wspan.empty()) << boost::ut::fatal;
                wspan[0].assign(kChunkBytes, std::byte{1});
                wspan.publish(1UZ);
            }
            auto rspan = reader.get(kRoundSize);
            expect(rspan.consume(kRoundSize)) << boost::ut::fatal;
        }

        const std::uint64_t peak = counter.peakBytes.load(std::memory_order_relaxed);
        expect(peak <= kPeakLimit) << "peak heap " << peak << " B exceeded buffer-sized limit " << kPeakLimit << " B";
    };

    "pool-backed buffer reaches zero-upstream steady state"_test = [] {
        // Wraps a synchronized_pool_resource over a CountingResource. Once the pool has been
        // primed by the first few rotations, recurring constant-size writes recycle the pool's
        // free-list and never fall through to upstream. Verifies the FIFO pool-recycle property.
        CountingResource                                  upstream;
        std::pmr::synchronized_pool_resource              pool{&upstream};
        std::pmr::polymorphic_allocator<std::pmr::string> alloc{&pool};
        gr::CircularBuffer<std::pmr::string>              buffer{16UZ, alloc};
        auto                                              writer = buffer.new_writer();
        auto                                              reader = buffer.new_reader();

        constexpr std::size_t kStringBytes = 256UZ;
        auto                  push         = [&](std::size_t n) {
            for (std::size_t i = 0UZ; i < n; ++i) {
                auto w = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
                expect(!w.empty()) << boost::ut::fatal;
                w[0].assign(kStringBytes, 'x');
                w.publish(1UZ);
            }
        };
        auto drain = [&](std::size_t n) {
            auto r = reader.get(n);
            expect(r.consume(n)) << boost::ut::fatal;
        };

        for (std::size_t round = 0UZ; round < 4UZ; ++round) { // warm-up — prime the pool's free-list
            push(8UZ);
            drain(8UZ);
        }
        const std::uint64_t upstreamAfterWarmup = upstream.allocCalls.load(std::memory_order_relaxed);
        for (std::size_t round = 0UZ; round < 50UZ; ++round) {
            push(8UZ);
            drain(8UZ);
        }
        const std::uint64_t upstreamAfterSteady = upstream.allocCalls.load(std::memory_order_relaxed);
        expect(upstreamAfterSteady == upstreamAfterWarmup) << "upstream allocations after warm-up: expected 0, got " << (upstreamAfterSteady - upstreamAfterWarmup);
    };

    "static-arena monotonic_buffer_resource hosts the buffer end-to-end"_test = [] {
        // Pins the buffer to a fixed-size byte arena (no upstream / no heap touches after
        // construction). monotonic_buffer_resource doesn't recycle allocations, so the test
        // sizes the arena to cover the buffer's slot array + per-slot string capacity for a
        // single warmup pass. The point: every allocation comes from `arena`, never from the
        // system heap (verified by null_memory_resource as the upstream).
        alignas(64) std::array<std::byte, 64UZ * 1024UZ>  arena{};
        std::pmr::monotonic_buffer_resource               mbr{arena.data(), arena.size(), std::pmr::null_memory_resource()};
        std::pmr::polymorphic_allocator<std::pmr::string> alloc{&mbr};
        gr::CircularBuffer<std::pmr::string>              buffer{16UZ, alloc};
        auto                                              writer = buffer.new_writer();
        auto                                              reader = buffer.new_reader();

        for (std::size_t i = 0UZ; i < 8UZ; ++i) {
            auto w = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            expect(!w.empty()) << boost::ut::fatal;
            w[0].assign(64UZ, 'y');
            w.publish(1UZ);
        }
        auto r = reader.get(8UZ);
        expect(r.consume(8UZ)) << boost::ut::fatal;
        // arrival at this expectation = no bad_alloc from null_memory_resource fallback.
        expect(true);
    };

    "reserve(nSamples, elementReserveHint) reaches steady-state zero allocation"_test = [] {
        // No-pressure pattern (kRoundSize < kBufferSlots ⇒ Deep reclaim never fires ⇒ slot
        // capacity is retained). After warm-up every slot's vector is pre-grown to kChunk via
        // reserve(1, kChunk); subsequent rounds (reserve(1, kChunk) + assign(kChunk)) allocate
        // nothing — this is the RT-safe value of the elementReserveHint overload.
        constexpr std::size_t kChunk       = 4096UZ;
        constexpr std::size_t kBufferSlots = 16UZ;
        constexpr std::size_t kRoundSize   = 12UZ;

        using ChunkT = std::pmr::vector<std::byte>;
        CountingResource                        counter;
        std::pmr::polymorphic_allocator<ChunkT> alloc(&counter);
        gr::CircularBuffer<ChunkT>              buffer{kBufferSlots, alloc};
        auto                                    writer = buffer.new_writer();
        auto                                    reader = buffer.new_reader();

        const auto round = [&] {
            for (std::size_t j = 0UZ; j < kRoundSize; ++j) {
                auto w = writer.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(1UZ, kChunk);
                expect(!w.empty()) << boost::ut::fatal;
                w[0].assign(kChunk, std::byte{1}); // fits the pre-reserved capacity → no realloc
                w.publish(1UZ);
            }
            auto r = reader.get(kRoundSize);
            expect(r.consume(kRoundSize)) << boost::ut::fatal;
        };

        for (std::size_t warm = 0UZ; warm < 4UZ; ++warm) { // 48 writes ≫ 16 slots → every slot grown
            round();
        }
        const std::uint64_t allocsAfterWarmup = counter.allocCalls.load(std::memory_order_relaxed);
        for (std::size_t steady = 0UZ; steady < 4UZ; ++steady) {
            round();
        }
        const std::uint64_t allocsSteady = counter.allocCalls.load(std::memory_order_relaxed);
        expect(eq(allocsSteady, allocsAfterWarmup)) << "reserve(n, hint) + retained capacity must be allocation-free in steady state";
    };
};

int main() { /* boost::ut auto-runs at program exit */ }
