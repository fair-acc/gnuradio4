#ifndef GNURADIO_TAGCHUNKBUFFER_HPP
#define GNURADIO_TAGCHUNKBUFFER_HPP

#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <memory_resource>
#include <span>
#include <utility>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/ChunkPool.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/ValueMap.hpp>

namespace gr {

/// Process-global default chunk pool for tag blobs. Prototype/benchmark stand-in for the
/// (deferred) per-graph / per-compute-domain pool that production multi-edge multiplexing needs.
[[nodiscard]] inline ChunkPool& defaultTagChunkPool() {
    static ChunkPool pool(std::pmr::new_delete_resource(), kDefaultChunkBytes);
    return pool;
}

/**
 * @brief Drop-in tag buffer: fixed-stride `BasicTag<false>` index ring + variable blobs in a shared `ChunkPool`.
 *
 * Satisfies the `Port` tag-buffer contract (`new_reader`/`new_writer`, `reserve`/`publish`/`get`/`consume`,
 * the connect `buffer()` round-trip, `houseKeeping`). The descriptor ring stores non-owning
 * `BasicTag<false>` slots (index + `ValueMapView` over chunk bytes) — directly the reader element the
 * `Port` read path already projects. The writer serialises each tag's packed `ValueMap` blob straight
 * into a pool chunk (the `assignTag` customization point), so no owning `Tag` is staged. Chunks are
 * reclaimed behind the slowest reader by `houseKeeping(depth)` — `Shallow` returns them to the pool
 * free-list (resident), `Deep` hands them back upstream (RAM elsewhere). Single-producer only.
 *
 * Shared state (`descriptors` + chunk chain + pool pointer) lives behind a `shared_ptr`, so a copy — and
 * the `buffer()` round-trip used by `Graph::connect` — shares one underlying buffer.
 */
template<ProducerType producerType = ProducerType::Single>
class TagChunkBuffer {
public:
    using value_type      = BasicTag<false>;
    using DescriptorRing  = CircularBuffer<BasicTag<false>, std::dynamic_extent, producerType>;
    using DescAllocator   = std::pmr::polymorphic_allocator<BasicTag<false>>;
    using InnerWriterType = decltype(std::declval<DescriptorRing&>().new_writer());
    using InnerReaderType = decltype(std::declval<DescriptorRing&>().new_reader());

private:
    struct Sealed {
        std::span<std::byte> bytes;
        std::size_t          releaseThreshold; // chunk dead once descriptors.min_reader_position() >= this
    };

    struct State {
        DescriptorRing       descriptors;
        ChunkPool*           pool;
        std::size_t          keepResident;
        std::span<std::byte> head{}; // current write chunk (single producer)
        std::size_t          headOffset{0UZ};
        std::deque<Sealed>   sealed{};

        State(std::size_t cap, DescAllocator alloc, ChunkPool* p, std::size_t keep) : descriptors(cap, alloc), pool(p), keepResident(keep) {}
    };

    std::shared_ptr<State> _state;

    explicit TagChunkBuffer(std::shared_ptr<State> state) noexcept : _state(std::move(state)) {} // round-trip / buffer() adoption

    // serialise a packed wire blob into the chain's head chunk; return the chunk pointer + length for a descriptor.
    // descAbsPos is the absolute descriptor position this tag will occupy (for the seal threshold).
    [[nodiscard]] static std::span<std::byte> serialiseBlob(State& s, std::span<const std::byte> blob, std::size_t descAbsPos) noexcept {
        const std::size_t need = blob.size();
        if (s.head.empty() || s.headOffset + need > s.head.size()) {
            if (!s.head.empty()) {
                s.sealed.push_back(Sealed{s.head, descAbsPos}); // prior chunk's last tag is at descAbsPos-1 ⇒ dead at min_reader ≥ descAbsPos
            }
            s.head       = s.pool->acquire();
            s.headOffset = 0UZ;
            if (s.head.empty() || need > s.head.size()) {
                return {}; // pool exhausted or jumbo tag — degrade to an empty view (rare; uncapped pool never hits this)
            }
        }
        std::byte* dst = s.head.data() + s.headOffset;
        std::memcpy(dst, blob.data(), need);
        s.headOffset += need;
        return {dst, need};
    }

public:
    explicit TagChunkBuffer(std::size_t descriptorCapacity, ChunkPool& pool = defaultTagChunkPool(), std::size_t keepResidentChunks = 0UZ) //
        : _state(std::make_shared<State>(descriptorCapacity, DescAllocator{}, &pool, keepResidentChunks)) {}

    // drop-in parity with the (minSize, polymorphic_allocator) construction site in Port; the allocator backs the
    // descriptor ring's slots, the (shared) ChunkPool backs the variable blobs.
    TagChunkBuffer(std::size_t descriptorCapacity, DescAllocator alloc, ChunkPool& pool = defaultTagChunkPool(), std::size_t keepResidentChunks = 0UZ) //
        : _state(std::make_shared<State>(descriptorCapacity, alloc, &pool, keepResidentChunks)) {}

    TagChunkBuffer(const TagChunkBuffer&)                = default; // shares state (the connect round-trip relies on this)
    TagChunkBuffer(TagChunkBuffer&&) noexcept            = default;
    TagChunkBuffer& operator=(const TagChunkBuffer&)     = default;
    TagChunkBuffer& operator=(TagChunkBuffer&&) noexcept = default;
    ~TagChunkBuffer()                                    = default;

    template<SpanReleasePolicy policy>
    class WriterSpan {
        using InnerSpan = decltype(std::declval<InnerWriterType&>().template reserve<policy>(0UZ));
        InnerSpan   _descSpan;
        State*      _state;
        std::size_t _base; // absolute descriptor position at reserve time

    public:
        using value_type = BasicTag<false>;

        WriterSpan(InnerSpan descSpan, State* state, std::size_t base) noexcept : _descSpan(std::move(descSpan)), _state(state), _base(base) {}

        [[nodiscard]] std::size_t    size() const noexcept { return _descSpan.size(); }
        [[nodiscard]] decltype(auto) operator[](std::size_t i) noexcept { return _descSpan[i]; } // BasicTag<false>& slot (e.g. .index read-back)
        constexpr void               publish(std::size_t nTags) noexcept { _descSpan.publish(nTags); }

        // contiguous-range / SpanLike surface over the descriptor slots, so this satisfies WriterSpanLike (the
        // Port OutputSpanLike concept requires it). The actual tag write goes through assignTag, not slot assignment.
        [[nodiscard]] std::span<BasicTag<false>> asSpan() noexcept { return std::span<BasicTag<false>>(_descSpan); }
        [[nodiscard]] auto                       begin() noexcept { return asSpan().begin(); }
        [[nodiscard]] auto                       end() noexcept { return asSpan().end(); }
        [[nodiscard]] BasicTag<false>*           data() noexcept { return asSpan().data(); }
        operator std::span<BasicTag<false>>() noexcept { return asSpan(); }

        // the Port writer customization point: serialise the tag blob into a chunk + write the descriptor slot.
        template<WireMapLike TPropertyMap>
        void assignTag(std::size_t i, std::size_t index, TPropertyMap&& tagData, std::pmr::memory_resource* /*unused: blobs live in the pool*/) noexcept {
            const std::span<std::byte> stored = serialiseBlob(*_state, tagData.blob(), _base + i);
            _descSpan[i]                      = BasicTag<false>{index, static_cast<const ValueMapView&>(ValueMap::makeView(std::span<const std::byte>{stored.data(), stored.size()}))};
        }
    };

    class Writer {
        InnerWriterType        _w;
        std::shared_ptr<State> _state;

    public:
        Writer(InnerWriterType w, std::shared_ptr<State> state) noexcept : _w(std::move(w)), _state(std::move(state)) {}
        Writer(const Writer&)            = delete;
        Writer& operator=(const Writer&) = delete;
        Writer(Writer&&)                 = default;
        Writer& operator=(Writer&&)      = default;
        ~Writer()                        = default;

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] WriterSpan<policy> reserve(std::size_t nTags) noexcept {
            return WriterSpan<policy>(_w.template reserve<policy>(nTags), _state.get(), _w.position());
        }
        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] WriterSpan<policy> tryReserve(std::size_t nTags) noexcept {
            return WriterSpan<policy>(_w.template tryReserve<policy>(nTags), _state.get(), _w.position());
        }

        // convenience for tests / direct callers: reserve-1, serialise, publish-1. false ⇒ backpressure.
        template<WireMapLike TPropertyMap>
        [[nodiscard]] bool publishTag(std::size_t index, TPropertyMap&& tagData) noexcept {
            WriterSpan<SpanReleasePolicy::ProcessNone> span = tryReserve<SpanReleasePolicy::ProcessNone>(1UZ);
            if (span.size() == 0UZ) {
                return false;
            }
            span.assignTag(0UZ, index, std::forward<TPropertyMap>(tagData), resource());
            span.publish(1UZ);
            return true;
        }

        [[nodiscard]] std::size_t                position() const noexcept { return _w.position(); }
        [[nodiscard]] std::size_t                available() const noexcept { return _w.available(); }
        [[nodiscard]] std::size_t                nRequestedSamplesToPublish() const noexcept { return _w.nRequestedSamplesToPublish(); }
        [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _w.resource(); }
        [[nodiscard]] TagChunkBuffer             buffer() const noexcept { return TagChunkBuffer(_state); }
    };

    class Reader {
        InnerReaderType        _r;
        std::shared_ptr<State> _state;

    public:
        Reader(InnerReaderType r, std::shared_ptr<State> state) noexcept : _r(std::move(r)), _state(std::move(state)) {}

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] auto get(std::size_t nTags = std::numeric_limits<std::size_t>::max()) {
            return _r.template get<policy>(nTags); // ReaderSpan<BasicTag<false>> — random-access, .index/.map, consume/tryConsume
        }
        [[nodiscard]] std::size_t    position() const noexcept { return _r.position(); }
        [[nodiscard]] std::size_t    available() const noexcept { return _r.available(); }
        [[nodiscard]] std::size_t    nSamplesConsumed() const noexcept { return _r.nSamplesConsumed(); }
        [[nodiscard]] bool           isConsumeRequested() const noexcept { return _r.isConsumeRequested(); }
        [[nodiscard]] TagChunkBuffer buffer() const noexcept { return TagChunkBuffer(_state); }
    };

    [[nodiscard]] Writer      new_writer() { return Writer(_state->descriptors.new_writer(), _state); }
    [[nodiscard]] Reader      new_reader() { return Reader(_state->descriptors.new_reader(), _state); }
    [[nodiscard]] std::size_t size() const noexcept { return _state->descriptors.size(); }
    [[nodiscard]] std::size_t n_writers() const { return _state->descriptors.n_writers(); }
    [[nodiscard]] std::size_t n_readers() const { return _state->descriptors.n_readers(); }
    [[nodiscard]] ChunkPool&  pool() const noexcept { return *_state->pool; }

    // reclaim driven by the scheduler housekeeping pass (Block.hpp invokes tagBuffer.houseKeeping(depth)).
    void houseKeeping(HouseKeepDepth depth) noexcept {
        State&            s      = *_state;
        const std::size_t minPos = s.descriptors.min_reader_position(); // gating cursor: advances on consume(), not get()
        while (!s.sealed.empty() && minPos >= s.sealed.front().releaseThreshold) {
            s.pool->release(s.sealed.front().bytes);
            s.sealed.pop_front();
        }
        if (s.sealed.empty() && !s.head.empty() && minPos >= s.descriptors.cursor_sequence().value()) { // 0-chunk floor: head fully consumed
            s.pool->release(s.head);
            s.head       = {};
            s.headOffset = 0UZ;
        }
        if (depth == HouseKeepDepth::Deep) {
            s.pool->reclaimToUpstream(s.keepResident);
        }
    }
};

static_assert(BufferLike<TagChunkBuffer<>>);

} // namespace gr

#endif // GNURADIO_TAGCHUNKBUFFER_HPP
