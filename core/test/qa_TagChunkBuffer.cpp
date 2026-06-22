#include <boost/ut.hpp>

#include <gnuradio-4.0/ChunkPool.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TagChunkBuffer.hpp>
#include <gnuradio-4.0/ValueMap.hpp>

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <vector>

using namespace boost::ut;

namespace {
using gr::pmt::ValueMap;
using gr::pmt::ValueMapView;

// distinct small packed map per tag (each blob ~300 B with the default 8-entry reservation)
[[nodiscard]] ValueMap makeMap(int i) {
    ValueMap m;
    m.insert_or_assign("idx", std::int32_t{i});
    m.insert_or_assign("rate", 48'000.0f + static_cast<float>(i));
    return m;
}

// publish n tags (index = k * 10) and return the originals for later comparison
[[nodiscard]] std::vector<ValueMap> publishTags(gr::TagChunkBuffer<>::Writer& w, int n) {
    std::vector<ValueMap> originals;
    originals.reserve(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        ValueMap   m  = makeMap(k);
        const bool ok = w.publishTag(static_cast<std::uint64_t>(k) * 10U, m); // serialises m.blob() straight into a chunk
        expect(ok) << "publishTag must not backpressure in a sized buffer";
        originals.push_back(std::move(m));
    }
    return originals;
}

[[nodiscard]] bool sameMap(const ValueMapView& view, const ValueMap& owning) { return view == static_cast<const ValueMapView&>(owning); }
} // namespace

const boost::ut::suite<"TagChunkBuffer"> _tagChunkBuffer = [] {
    "content round-trips and lower_bound-by-index works on the descriptor span (R7 reader side)"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ);
        gr::TagChunkBuffer<>                 buf(64UZ, pool);

        auto       w         = buf.new_writer();
        auto       r         = buf.new_reader();
        const auto originals = publishTags(w, 6);

        auto span = r.get();
        expect(eq(span.size(), 6UZ));

        for (std::size_t k = 0; k < span.size(); ++k) {
            expect(eq(span[k].index, static_cast<std::uint64_t>(k) * 10U));
            expect(sameMap(span[k].map, originals[k])) << "BasicTag<false> view aliases the chunk bytes and matches the published map";
        }

        // the central reader-side claim, exercised at runtime: random-access + &BasicTag<false>::index projection
        auto it = std::ranges::lower_bound(span, std::uint64_t{30U}, std::ranges::less{}, &gr::BasicTag<false>::index);
        expect((it != span.end()) and eq(it->index, std::uint64_t{30U}));
        expect(span.consume(6UZ));
    };

    "blobs span multiple chunks intact (bip-packed, cross-boundary)"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ); // small ⇒ a couple dozen tags need several chunks
        gr::TagChunkBuffer<>                 buf(128UZ, pool);

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
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ);
        gr::TagChunkBuffer<>                 buf(64UZ, pool);

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
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ);
        gr::TagChunkBuffer<>                 buf(128UZ, pool);

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
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ);
        gr::TagChunkBuffer<>                 buf(128UZ, pool);

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
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ);
        gr::TagChunkBuffer<>                 buf(128UZ, pool);

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
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ); // one pool, shared graph-wide
        gr::TagChunkBuffer<>                 edgeA(128UZ, pool);
        gr::TagChunkBuffer<>                 edgeB(128UZ, pool);

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

    "multiplexing (Deep): A's idle RAM returns upstream (available elsewhere); B re-acquires it"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 1024UZ);
        gr::TagChunkBuffer<>                 edgeA(128UZ, pool);
        gr::TagChunkBuffer<>                 edgeB(128UZ, pool);

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
