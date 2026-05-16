#include <boost/ut.hpp>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/ByteRingBuffer.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <type_traits>

using namespace boost::ut;

namespace {

// trivially-copyable plain element
using PlainRing = gr::ByteRingBuffer<std::int32_t>;
using MultiRing = gr::ByteRingBuffer<std::int32_t, std::dynamic_extent, gr::ProducerType::Multi>;

// Clearable element to exercise the (reused) CircularBuffer housekeeping passthrough
struct ClearableRec {
    std::int64_t v{0};
    void         clear() noexcept { v = 0; }
    void         shrink_to_fit() noexcept {}
};
using ClearableRing = gr::ByteRingBuffer<ClearableRec>;

// Reservable element: records its reserved capacity so the auto-applied byteChunkSize is observable
struct ReservableRec {
    std::size_t  cap{0UZ};
    std::int64_t v{0};
    void         reserve(std::size_t n) noexcept {
        if (n > cap) {
            cap = n;
        }
    }
    void clear() noexcept { v = 0; }
    void shrink_to_fit() noexcept { cap = 0UZ; }
};
using ReservableRing = gr::ByteRingBuffer<ReservableRec>;

static_assert(gr::BufferLike<PlainRing>);
static_assert(gr::BufferLike<MultiRing>);
static_assert(gr::BufferLike<ClearableRing>);
static_assert(gr::BufferLike<gr::ByteRingBuffer<gr::Tag>>);
static_assert(gr::ReservableElement<ReservableRec>);
static_assert(gr::BufferLike<ReservableRing>);

} // namespace

const boost::ut::suite<"ByteRingBuffer (CircularBuffer composition)"> _byte_ring_buffer_tests = [] {
    "byteChunkSize is carried and auto-applied on reserve(n) for ReservableElement"_test = [] {
        ReservableRing ring{16UZ, /*byteChunkSize=*/128UZ};
        expect(eq(ring.byteChunkSize(), 128UZ));
        auto w = ring.new_writer();
        auto r = ring.new_reader();
        {
            auto s = w.reserve(2UZ); // no explicit hint → auto-applies the buffer's byteChunkSize
            expect(eq(s.size(), 2UZ));
            expect(ge(s[0].cap, 128UZ)) << "slot pre-reserved to the buffer's byteChunkSize";
            expect(ge(s[1].cap, 128UZ));
            s[0].v = 7;
            s.publish(2UZ);
        }
        {
            auto g = r.get();
            expect(eq(g.size(), 2UZ));
            expect(eq(g[0].v, std::int64_t{7}));
            expect(g.consume(2UZ));
        }
        ReservableRing ring2{16UZ, 128UZ}; // explicit hint overrides the default
        auto           w2 = ring2.new_writer();
        {
            auto s = w2.reserve(1UZ, 512UZ);
            expect(ge(s[0].cap, 512UZ)) << "explicit elementReserveHint overrides byteChunkSize";
            s.publish(1UZ);
        }
    };

    "PMR ctor: resource is stored and forwarded to the delegate ring"_test = [] {
        alignas(64) std::array<std::byte, 64UZ * 1024UZ> arena{};
        std::pmr::monotonic_buffer_resource              mbr{arena.data(), arena.size(), std::pmr::null_memory_resource()};
        gr::ByteRingBuffer<std::int32_t>                 ring{16UZ, /*byteChunkSize=*/64UZ, &mbr};
        expect(ring.resource() == &mbr) << "explicit PMR resource is stored";
        expect(eq(ring.byteChunkSize(), 64UZ));
        auto w = ring.new_writer();
        auto r = ring.new_reader();
        {
            auto s = w.reserve(3UZ);
            s[0]   = 1;
            s[1]   = 2;
            s[2]   = 3;
            s.publish(3UZ);
        }
        auto g = r.get();
        expect(eq(g.size(), 3UZ));
        expect(eq(g[2], 3));
        expect(g.consume(3UZ));
        expect(true) << "null upstream ⇒ buffer served entirely from the stack arena (no bad_alloc)";
    };

    "polymorphic_allocator shim: drop-in parity with CircularBuffer(minSize, allocator)"_test = [] {
        alignas(64) std::array<std::byte, 64UZ * 1024UZ> arena{};
        std::pmr::monotonic_buffer_resource              mbr{arena.data(), arena.size(), std::pmr::null_memory_resource()};
        gr::ByteRingBuffer<std::int32_t>                 ring{16UZ, std::pmr::polymorphic_allocator<std::int32_t>(&mbr)};
        expect(ring.resource() == &mbr) << "allocator's resource is stored";
        expect(eq(ring.byteChunkSize(), gr::kDefaultByteChunkSize)) << "byteChunkSize defaults when only the allocator is given";
        auto w = ring.new_writer();
        auto r = ring.new_reader();
        {
            auto s = w.reserve(2UZ);
            s[0]   = 42;
            s[1]   = 43;
            s.publish(2UZ);
        }
        auto g = r.get();
        expect(eq(g.size(), 2UZ));
        expect(eq(g[1], 43));
        expect(g.consume(2UZ));
    };

    "BufferLike drop-in: SISO round-trip via the reused reader/writer"_test = [] {
        PlainRing ring{16UZ};
        expect(ge(ring.size(), 16UZ));
        auto w = ring.new_writer();
        auto r = ring.new_reader();

        {
            auto s = w.reserve(3UZ);
            expect(eq(s.size(), 3UZ));
            s[0] = 10;
            s[1] = 20;
            s[2] = 30;
            s.publish(3UZ);
        }
        expect(eq(r.available(), 3UZ));
        {
            auto g = r.get();
            expect(eq(g.size(), 3UZ));
            expect(eq(g[0], 10));
            expect(eq(g[1], 20));
            expect(eq(g[2], 30));
            expect(g.consume(3UZ));
        }
        expect(eq(r.available(), 0UZ));
    };

    "backpressure + reclaim: full ring blocks tryReserve until consumed"_test = [] {
        PlainRing         ring{8UZ};
        auto              w   = ring.new_writer();
        auto              r   = ring.new_reader();
        const std::size_t cap = ring.size();

        {
            auto s = w.reserve(cap);
            for (std::size_t i = 0UZ; i < cap; ++i) {
                s[i] = static_cast<std::int32_t>(i);
            }
            s.publish(cap);
        }
        {
            auto s = w.tryReserve(1UZ);
            expect(s.empty()) << "no capacity until the reader consumes";
        }
        {
            auto g = r.get();
            expect(eq(g.size(), cap));
            expect(eq(g[cap - 1UZ], static_cast<std::int32_t>(cap - 1UZ)));
            expect(g.consume(cap));
        }
        {
            auto s = w.tryReserve(1UZ);
            expect(eq(s.size(), 1UZ)) << "capacity freed after consume";
            s[0] = 99;
            s.publish(1UZ);
        }
    };

    "SIMO: two readers each see all published; slowest gates reclaim"_test = [] {
        PlainRing ring{16UZ};
        auto      w  = ring.new_writer();
        auto      r1 = ring.new_reader();
        auto      r2 = ring.new_reader();

        {
            auto s = w.reserve(4UZ);
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                s[i] = static_cast<std::int32_t>(100 + i);
            }
            s.publish(4UZ);
        }
        expect(eq(r1.available(), 4UZ));
        expect(eq(r2.available(), 4UZ));
        {
            auto g1 = r1.get();
            expect(eq(g1.size(), 4UZ));
            expect(eq(g1[3], 103));
            expect(g1.consume(4UZ));
        }
        expect(eq(r2.available(), 4UZ)) << "r2 still sees all 4 (not yet consumed)";
        {
            auto g2 = r2.get();
            expect(eq(g2.size(), 4UZ));
            expect(eq(g2[0], 100));
            expect(g2.consume(4UZ));
        }
        expect(eq(r2.available(), 0UZ));
    };

    "MIMO producer type instantiates and round-trips"_test = [] {
        MultiRing ring{16UZ};
        auto      w = ring.new_writer();
        auto      r = ring.new_reader();
        {
            auto s = w.tryReserve(2UZ);
            expect(eq(s.size(), 2UZ));
            s[0] = 7;
            s[1] = 8;
            s.publish(2UZ);
        }
        {
            auto g = r.get();
            expect(eq(g.size(), 2UZ));
            expect(eq(g[0], 7));
            expect(eq(g[1], 8));
            expect(g.consume(2UZ));
        }
    };

    "houseKeeping passthrough compiles and runs for a Clearable element"_test = [] {
        ClearableRing ring{16UZ};
        auto          w = ring.new_writer();
        auto          r = ring.new_reader();
        {
            auto s = w.reserve(2UZ);
            s[0]   = ClearableRec{.v = 1};
            s[1]   = ClearableRec{.v = 2};
            s.publish(2UZ);
        }
        {
            auto g = r.get();
            expect(eq(g.size(), 2UZ));
            expect(eq(g[1].v, std::int64_t{2}));
            expect(g.consume(2UZ));
        }
        ring.houseKeeping(gr::HouseKeepDepth::Deep); // reused CircularBuffer reclaim path
        expect(eq(ring.n_readers(), 1UZ));
        expect(eq(ring.n_writers(), 1UZ));
    };

    "drop-in for the migration target: gr::Tag round-trip"_test = [] {
        gr::ByteRingBuffer<gr::Tag> ring{16UZ};
        auto                        w = ring.new_writer();
        auto                        r = ring.new_reader();
        {
            auto s = w.reserve(1UZ);
            expect(eq(s.size(), 1UZ));
            gr::Tag t{};
            t.index = 5UZ;
            s[0]    = t;
            s.publish(1UZ);
        }
        {
            auto g = r.get();
            expect(eq(g.size(), 1UZ));
            expect(eq(g[0].index, 5UZ));
            expect(g.consume(1UZ));
        }
    };
};

int main() { /* boost::ut auto-runs */ }
