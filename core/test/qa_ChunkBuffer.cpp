#include <boost/ut.hpp>

#include <gnuradio-4.0/ChunkBuffer.hpp>
#include <gnuradio-4.0/ChunkPool.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/ValueMap.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

using namespace boost::ut;

namespace {
using gr::pmt::ValueMap;
using gr::pmt::ValueMapView;

struct TestRecord {
    std::size_t                sequence{};
    std::span<const std::byte> payload{};
};

static_assert(std::is_trivially_copyable_v<TestRecord>);
static_assert(gr::BufferLike<gr::ChunkBuffer<TestRecord, gr::ProducerType::Single>>);

// distinct small packed map per tag (each blob ~300 B with the default 8-entry reservation)
[[nodiscard]] ValueMap makeMap(int i) {
    ValueMap m;
    m.insert_or_assign("idx", std::int32_t{i});
    m.insert_or_assign("rate", 48'000.0f + static_cast<float>(i));
    return m;
}

[[nodiscard]] bool publishTag(gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single>::Writer& w, std::size_t index, const ValueMap& m) {
    auto span = w.tryReserve(1UZ);
    if (span.size() == 0UZ) {
        return false;
    }
    const std::span<const std::byte> blob   = gr::tagPayloadBlob(m);
    const std::span<std::byte>       stored = span.storeBlob(0UZ, blob);
    if (stored.size() != blob.size()) {
        return false;
    }
    span[0UZ] = gr::makeStoredTag(index, stored);
    span.publish(1UZ);
    return true;
}

// publish n tags (index = k * 10) and return the originals for later comparison
[[nodiscard]] std::vector<ValueMap> publishTags(gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single>::Writer& w, int n) {
    std::vector<ValueMap> originals;
    originals.reserve(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        ValueMap   m  = makeMap(k);
        const bool ok = publishTag(w, static_cast<std::size_t>(k) * 10UZ, m);
        expect(ok) << "publishTag must not backpressure in a sized buffer";
        originals.push_back(std::move(m));
    }
    return originals;
}

[[nodiscard]] bool sameMap(const ValueMapView& view, const ValueMap& owning) { return view == static_cast<const ValueMapView&>(owning); }
} // namespace

const boost::ut::suite<"ChunkBuffer"> _chunkBuffer = [] {
    "direct descriptor assignment works without blobs"_test = [] {
        gr::ChunkBuffer<std::uint32_t, gr::ProducerType::Single> buf(8UZ);

        auto w = buf.new_writer();
        auto r = buf.new_reader();
        {
            auto span = w.reserve(3UZ);
            expect(eq(span.size(), 3UZ));
            span[0UZ] = 11U;
            span[1UZ] = 22U;
            span[2UZ] = 33U;
            span.publish(3UZ);
        }

        auto read = r.get();
        expect(eq(read.size(), 3UZ));
        expect(eq(read[0UZ], 11U));
        expect(eq(read[1UZ], 22U));
        expect(eq(read[2UZ], 33U));
        expect(read.consume(3UZ));
    };

    "generic blob-backed descriptors round-trip without Tag or ValueMap"_test = [] {
        gr::allocator::pmr::CountingResource                  upstream;
        gr::ChunkPool                                         pool(&upstream, 256UZ);
        gr::ChunkBuffer<TestRecord, gr::ProducerType::Single> buf(8UZ, pool);
        constexpr std::array<std::byte, 5>                    payload{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}, std::byte{0x05}};

        auto w = buf.new_writer();
        auto r = buf.new_reader();
        {
            auto span = w.reserve(2UZ);
            expect(eq(span.size(), 2UZ));
            span[0UZ]                         = TestRecord{.sequence = 41UZ, .payload = {}};
            const std::span<std::byte> stored = span.storeBlob(1UZ, payload);
            expect(eq(stored.size(), payload.size()));
            span[1UZ] = TestRecord{.sequence = 42UZ, .payload = stored};
            span.publish(2UZ);
        }

        {
            auto read = r.get();
            expect(eq(read.size(), 2UZ));
            expect(eq(read[0UZ].sequence, 41UZ));
            expect(read[0UZ].payload.empty());
            expect(eq(read[1UZ].sequence, 42UZ));
            expect(std::ranges::equal(read[1UZ].payload, payload));
            expect(read.consume(2UZ));
        }
        buf.houseKeeping(gr::HouseKeepDepth::Deep);
        expect(eq(pool.residentChunks(), 0UZ));
    };

    "content round-trips and lower_bound-by-index works on the descriptor span (R7 reader side)"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(64UZ, pool);

        auto       w         = buf.new_writer();
        auto       r         = buf.new_reader();
        const auto originals = publishTags(w, 6);

        auto span = r.get();
        expect(eq(span.size(), 6UZ));

        for (std::size_t k = 0; k < span.size(); ++k) {
            expect(eq(span[k].index, static_cast<std::size_t>(k) * 10UZ));
            expect(sameMap(span[k].map, originals[k])) << "Tag view aliases the chunk bytes and matches the published map";
        }

        // the central reader-side claim, exercised at runtime: random-access + &Tag::index projection
        auto it = std::ranges::lower_bound(span, std::size_t{30U}, std::ranges::less{}, &gr::Tag::index);
        expect((it != span.end()) and eq(it->index, std::size_t{30U}));
        expect(span.consume(6UZ));
    };

    "blobs span multiple chunks intact (bip-packed, cross-boundary)"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ); // small ⇒ a couple dozen tags need several chunks
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(128UZ, pool);

        auto       w         = buf.new_writer();
        auto       r         = buf.new_reader();
        const auto originals = publishTags(w, 24);

        expect(pool.residentChunks() >= 2UZ) << "24 tags must straddle more than one chunk";

        auto span = r.get();
        expect(eq(span.size(), 24UZ));
        bool allEqual = true;
        for (std::size_t k = 0; k < span.size(); ++k) {
            allEqual = allEqual and sameMap(span[k].map, originals[k]);
        }
        expect(allEqual) << "every blob intact across chunk boundaries";
        expect(span.consume(24UZ));
    };

    "UAF guard: a got-but-not-consumed chunk is NOT reclaimed; consume then release"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(64UZ, pool);

        auto       w         = buf.new_writer();
        auto       r         = buf.new_reader();
        const auto originals = publishTags(w, 3); // one chunk

        {
            auto span = r.get(); // got, NOT consumed
            expect(eq(span.size(), 3UZ));

            buf.houseKeeping(gr::HouseKeepDepth::Deep); // gating cursor unmoved ⇒ chunk pinned
            expect(eq(pool.freeChunks(), 0UZ)) << "chunk must NOT be released while a reader holds views";
            expect(ge(pool.residentChunks(), 1UZ));
            expect(sameMap(span[0].map, originals[0])) << "view still valid after housekeeping";

            expect(span.consume(3UZ)); // now consumed
        }
        buf.houseKeeping(gr::HouseKeepDepth::Deep); // drained ⇒ released + returned upstream
        expect(eq(pool.residentChunks(), 0UZ)) << "drained chunk returns to upstream under Deep";
    };

    "SIMO: chunks reclaim only behind the slowest reader"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(128UZ, pool);

        auto w      = buf.new_writer();
        auto r1     = buf.new_reader();
        auto r2     = buf.new_reader(); // both registered before publishing
        std::ignore = publishTags(w, 24);
        expect(ge(pool.residentChunks(), 2UZ));

        {
            auto s1 = r1.get();
            expect(s1.consume(s1.size())); // r1 fully consumed
        }
        buf.houseKeeping(gr::HouseKeepDepth::Deep);
        expect(ge(pool.residentChunks(), 2UZ)) << "r2 still lagging ⇒ nothing released";

        {
            auto s2 = r2.get();
            expect(s2.consume(s2.size())); // r2 catches up
        }
        buf.houseKeeping(gr::HouseKeepDepth::Deep);
        expect(eq(pool.residentChunks(), 0UZ)) << "both consumed ⇒ all chunks reclaimed";
    };

    "return-policy knob rides HouseKeepDepth: Shallow keeps resident, Deep returns upstream"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(128UZ, pool);

        auto w      = buf.new_writer();
        auto r      = buf.new_reader();
        std::ignore = publishTags(w, 24);
        {
            auto s = r.get();
            expect(s.consume(s.size()));
        }

        buf.houseKeeping(gr::HouseKeepDepth::Shallow); // dead chunks → pool free-list, NOT upstream
        expect(gt(pool.freeChunks(), 0UZ));
        expect(gt(pool.residentChunks(), 0UZ));
        expect(eq(upstream.deallocCount, 0UZ)) << "Shallow keeps RAM resident for fast re-acquire";

        buf.houseKeeping(gr::HouseKeepDepth::Deep); // now hand it back
        expect(eq(pool.residentChunks(), 0UZ));
        expect(gt(upstream.deallocCount, 0UZ)) << "Deep returns idle RAM to the shared upstream";
    };

    "steady state is allocation-free: warm-up grows, then chunks recycle from the free-list"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(128UZ, pool);

        auto w = buf.new_writer();
        auto r = buf.new_reader();

        auto roundTrip = [&] {
            std::ignore = publishTags(w, 16);
            {
                auto s = r.get();
                expect(s.consume(s.size()));
            } // release the span before housekeeping (mirrors the scheduler: housekeeping runs between work() calls)
            buf.houseKeeping(gr::HouseKeepDepth::Shallow); // keep chunks resident on the free-list
        };

        roundTrip(); // warm-up: grows the pool to the working set
        const std::size_t allocsAfterWarmup = upstream.allocCount;
        expect(gt(allocsAfterWarmup, 0UZ));

        for (int round = 0; round < 5; ++round) {
            roundTrip();
        }
        expect(eq(upstream.allocCount, allocsAfterWarmup)) << "steady state touches no upstream — chunks reused from the free-list";
    };

    "multiplexing (Shallow): two edges share one pool — B reuses A's returned chunks (peak-concurrent, not Σ)"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ); // one pool, shared graph-wide
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> edgeA(128UZ, pool);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> edgeB(128UZ, pool);

        auto wA                 = edgeA.new_writer();
        auto rA                 = edgeA.new_reader();
        std::ignore             = publishTags(wA, 24); // edge A bursts
        const std::size_t peakA = pool.residentChunks();
        expect(ge(peakA, 2UZ));
        {
            auto s = rA.get();
            expect(s.consume(s.size()));
        }
        edgeA.houseKeeping(gr::HouseKeepDepth::Shallow); // A drains → its chunks go to the shared free-list
        expect(eq(pool.freeChunks(), peakA));
        const std::size_t allocsAfterA = upstream.allocCount;

        auto wB     = edgeB.new_writer();
        auto rB     = edgeB.new_reader();
        std::ignore = publishTags(wB, 24); // edge B bursts the same size, temporally disjoint
        expect(eq(upstream.allocCount, allocsAfterA)) << "B reused A's freed chunks — no new upstream allocation";
        expect(eq(pool.residentChunks(), peakA)) << "resident ≈ one edge's worth (peak-concurrent), not 2N";
        {
            auto s = rB.get();
            expect(s.consume(s.size()));
        }
    };

    "owned pool: blobs draw from the injected (descriptor-allocator) resource, not a global pool"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        // no explicit ChunkPool: the buffer owns one whose upstream is the descriptor allocator's resource
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(64UZ, gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single>::DescAllocator{&upstream}, 0UZ, 1024UZ);

        auto       w         = buf.new_writer();
        auto       r         = buf.new_reader();
        const auto originals = publishTags(w, 6);
        expect(gt(upstream.allocCount, 0UZ)) << "tag blobs (and the descriptor ring) allocate from the injected resource";

        {
            auto span = r.get();
            expect(eq(span.size(), 6UZ));
            bool allEqual = true;
            for (std::size_t k = 0; k < span.size(); ++k) {
                allEqual = allEqual and sameMap(span[k].map, originals[k]);
            }
            expect(allEqual) << "content round-trips through the owned pool";
            expect(span.consume(6UZ));
        } // release the span first: the gating cursor advances on release, not consume()

        const std::size_t deallocBefore = upstream.deallocCount;
        buf.houseKeeping(gr::HouseKeepDepth::Deep);
        expect(gt(upstream.deallocCount, deallocBefore)) << "Deep returns the owned pool's chunks to the injected resource";
    };

    "owned pool: steady state is allocation-free (Shallow recycles from the free-list)"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(128UZ, gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single>::DescAllocator{&upstream}, 0UZ, 1024UZ);

        auto w = buf.new_writer();
        auto r = buf.new_reader();

        auto roundTrip = [&] {
            std::ignore = publishTags(w, 16);
            {
                auto s = r.get();
                expect(s.consume(s.size()));
            }
            buf.houseKeeping(gr::HouseKeepDepth::Shallow); // active-work cadence: keep chunks resident
        };

        roundTrip(); // warm-up grows the owned pool to the working set
        const std::size_t allocsAfterWarmup = upstream.allocCount;
        expect(gt(allocsAfterWarmup, 0UZ));

        for (int round = 0; round < 5; ++round) {
            roundTrip();
        }
        expect(eq(upstream.allocCount, allocsAfterWarmup)) << "owned-pool steady state touches no upstream — the MCU no-heap path";
    };

    "jumbo tag (blob > chunk size) round-trips via the oversized size-class — no silent drop"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 256UZ); // small chunk so an ordinary map is jumbo
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> buf(64UZ, pool);

        ValueMap big;
        big.insert_or_assign("payload", std::string(400, 'x')); // single large value ⇒ blob exceeds a standard chunk
        expect(gt(big.blob().size(), 256UZ)) << "precondition: the blob must exceed the standard chunk size";

        auto w = buf.new_writer();
        auto r = buf.new_reader();
        expect(publishTag(w, 7UZ, big)) << "jumbo publish must succeed (oversized size-class), not backpressure or drop";

        {
            auto span = r.get();
            expect(eq(span.size(), 1UZ));
            expect(eq(span[0].index, std::size_t{7U}));
            expect(sameMap(span[0].map, big)) << "jumbo blob content intact end-to-end (the qa_Tags forward-loss case)";
            expect(span.consume(1UZ));
        }
        buf.houseKeeping(gr::HouseKeepDepth::Deep);
        expect(eq(pool.residentChunks(), 0UZ)) << "oversized chunk returned straight to upstream, not the fixed free-list";
        expect(eq(pool.freeChunks(), 0UZ)) << "oversized chunks never land on the recyclable free-list";
    };

    "multiplexing (Deep): A's idle RAM returns upstream (available elsewhere); B re-acquires it"_test = [] {
        gr::allocator::pmr::CountingResource               upstream;
        gr::ChunkPool                                      pool(&upstream, 1024UZ);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> edgeA(128UZ, pool);
        gr::ChunkBuffer<gr::Tag, gr::ProducerType::Single> edgeB(128UZ, pool);

        auto wA     = edgeA.new_writer();
        auto rA     = edgeA.new_reader();
        std::ignore = publishTags(wA, 24);
        {
            auto s = rA.get();
            expect(s.consume(s.size()));
        }
        edgeA.houseKeeping(gr::HouseKeepDepth::Deep); // eager return: A's RAM goes back to the shared upstream
        expect(eq(pool.residentChunks(), 0UZ)) << "A's chunks fully returned — RAM available elsewhere";
        expect(gt(upstream.deallocCount, 0UZ));
        const std::size_t allocsAfterADeep = upstream.allocCount;

        auto wB     = edgeB.new_writer();
        auto rB     = edgeB.new_reader();
        std::ignore = publishTags(wB, 24); // B must re-acquire from upstream — the cost of eager return
        expect(gt(upstream.allocCount, allocsAfterADeep)) << "B re-allocates — the re-acquire cost of returning RAM elsewhere";
        {
            auto s = rB.get();
            expect(s.consume(s.size()));
        }
    };
};

int main() { return 0; }
