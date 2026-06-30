#ifndef GNURADIO_CHUNKBUFFER_HPP
#define GNURADIO_CHUNKBUFFER_HPP

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/ChunkPool.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr {

/**
 * @brief Chunked descriptor+blob buffer satisfying the BufferLike concept/contract:
 * a fixed-stride `T` descriptor ring + variable-size blobs in a `ChunkPool`.
 *
 * By default each buffer **owns** a `ChunkPool` whose upstream is the descriptor allocator's
 * `memory_resource` — so blobs are drawn from the injected resource, an arena on MCU,
 * not a process-global heap. Quiet edges hand chunks back to that resource via housekeeping, so they
 * multiplex through the shared upstream rather than each holding a fixed reservation. An explicit
 * `ChunkPool&` may instead be passed to share one pool graph-wide (the cross-edge multiplexing path).
 *
 * Reclaim rides the scheduler's `houseKeeping(depth)` cadence: `Shallow` (the active-work default)
 * returns drained chunks to the pool free-list (resident → steady-state publish is allocation-free);
 * `Deep` (the quiescence pass) additionally hands the free-list back upstream (RAM goes elsewhere). The
 * sealed-chunk bookkeeping is an intrusive FIFO threaded through the chunks themselves (a small node at
 * each chunk's head), so housekeeping touches no heap. Single-producer only.
 *
 * Shared state (descriptors + chunk chain + pool) lives behind a `shared_ptr`, so a copy — and the
 * `buffer()` round-trip used by `Graph::connect` — shares one underlying buffer.
 */
template<gr::meta::trivially_copyable T, ProducerType producerType = ProducerType::Single>
struct ChunkBuffer {
    using value_type      = T;
    using DescriptorRing  = CircularBuffer<T, std::dynamic_extent, producerType>;
    using DescAllocator   = std::pmr::polymorphic_allocator<T>;
    using InnerWriterType = decltype(std::declval<DescriptorRing&>().new_writer());
    using InnerReaderType = decltype(std::declval<DescriptorRing&>().new_reader());

private:
    struct SealedHeader {
        std::byte*  next;             // next sealed chunk's data ptr, or nullptr (intrusive FIFO node, lives in the chunk)
        std::size_t releaseThreshold; // chunk dead once descriptors.min_reader_position() >= this
        std::size_t chunkSize;        // this chunk's byte size (chunkBytes for a standard chunk, larger for an oversized one)
    };
    static constexpr std::size_t kChunkHeaderBytes = sizeof(SealedHeader); // reserved at each chunk's head for the node

    [[nodiscard]] static SealedHeader readHeader(std::byte* chunk) noexcept {
        SealedHeader h;
        std::memcpy(&h, chunk, sizeof(h));
        return h;
    }
    static void writeHeader(std::byte* chunk, SealedHeader h) noexcept { std::memcpy(chunk, &h, sizeof(h)); }

    struct State {
        DescriptorRing           descriptors;
        std::optional<ChunkPool> ownedPool;    // engaged ⇒ this buffer owns its pool (upstream = ring resource)
        ChunkPool*               externalPool; // non-null ⇒ explicit shared pool (graph-wide multiplexing)
        std::size_t              keepResident; // free-list floor retained on a Deep reclaim
        std::size_t              chunkBytes;
        std::span<std::byte>     head{}; // current write chunk (single producer)
        std::size_t              headOffset{0UZ};
        std::size_t              headSize{0UZ};       // byte size of the current head chunk (standard or oversized)
        std::byte*               sealedHead{nullptr}; // intrusive FIFO of full chunks awaiting reader drain
        std::byte*               sealedTail{nullptr};
        std::size_t              headLastWritten{0UZ}; // highest descriptor position written into `head`; head is reclaimable once min_reader passes it
        std::atomic_flag         _guard{};             // mutual-exclusion of producer serialiseBlob() vs scheduler houseKeeping() on the (non-ring) chunk bookkeeping

        State(std::size_t cap, DescAllocator alloc, std::size_t keep, std::size_t chunk) : descriptors(cap, alloc), externalPool(nullptr), keepResident(keep), chunkBytes(chunk) {
            assert(chunkBytes >= kChunkHeaderBytes && "chunk must hold the sealed-chunk header");
            ownedPool.emplace(alloc.resource(), chunk);
        }
        State(std::size_t cap, DescAllocator alloc, ChunkPool& ext, std::size_t keep) : descriptors(cap, alloc), externalPool(&ext), keepResident(keep), chunkBytes(ext.chunkBytes()) { assert(chunkBytes >= kChunkHeaderBytes && "chunk must hold the sealed-chunk header"); }

        ~State() {
            while (sealedHead != nullptr) {
                const SealedHeader h     = readHeader(sealedHead);
                std::byte*         chunk = sealedHead;
                sealedHead               = h.next;
                releaseChunk(std::span<std::byte>{chunk, h.chunkSize});
            }
            if (!head.empty()) {
                releaseChunk(head);
            }
        }

        [[nodiscard]] ChunkPool& pool() noexcept { return ownedPool ? *ownedPool : *externalPool; }

        // route a chunk back to its origin: oversized one-offs (larger than a standard chunk) go straight to upstream,
        // standard chunks return to the recyclable free-list.
        void releaseChunk(std::span<std::byte> chunk) noexcept {
            if (chunk.size() > chunkBytes) {
                pool().releaseOversized(chunk);
            } else {
                pool().release(chunk);
            }
        }
    };

    std::shared_ptr<State> _state;

    explicit ChunkBuffer(std::shared_ptr<State> state) noexcept : _state(std::move(state)) {} // round-trip / buffer() adoption

    // RAII release of State::_guard (acquired explicitly at the call site): the producer (serialiseBlob)
    // spin-acquires — it must publish; houseKeeping is low-priority and try-acquires, deferring on contention.
    struct StateGuard {
        std::atomic_flag& _flag;
        explicit StateGuard(std::atomic_flag& flag) noexcept : _flag(flag) {}
        StateGuard(const StateGuard&)            = delete;
        StateGuard& operator=(const StateGuard&) = delete;
        ~StateGuard() { _flag.clear(std::memory_order_release); }
    };

    static void sealHead(State& s, std::size_t releaseThreshold) noexcept {
        writeHeader(s.head.data(), SealedHeader{nullptr, releaseThreshold, s.headSize});
        if (s.sealedTail != nullptr) {
            SealedHeader tail = readHeader(s.sealedTail);
            tail.next         = s.head.data();
            writeHeader(s.sealedTail, tail);
        } else {
            s.sealedHead = s.head.data();
        }
        s.sealedTail = s.head.data();
    }

    // serialise a blob into the chain's head chunk; return the chunk pointer + length for a descriptor.
    // descAbsPos is the absolute descriptor position this entry will occupy (for the seal threshold).
    [[nodiscard]] static std::span<std::byte> serialiseBlob(State& s, std::span<const std::byte> blob, std::size_t descAbsPos) noexcept {
        while (s._guard.test_and_set(std::memory_order_acquire)) { /* brief: wait out a best-effort houseKeeping pass */
        }
        StateGuard        guard(s._guard); // RAII release; serialises chunk bookkeeping vs houseKeeping (off-thread async producers)
        const std::size_t need        = blob.size();
        const std::size_t standardFit = s.chunkBytes - kChunkHeaderBytes;
        if (s.head.empty() || s.headOffset + need > s.head.size()) {
            if (!s.head.empty()) {
                sealHead(s, descAbsPos); // prior chunk's last descriptor is at descAbsPos-1 ⇒ dead at min_reader ≥ descAbsPos
            }
            // jumbo blob (> a standard chunk's usable bytes) → a dedicated oversized chunk holding the header + this blob,
            // so an over-reserved blob is never silently dropped; standard blobs take a recyclable pool chunk.
            s.head       = need > standardFit ? s.pool().acquireOversized(kChunkHeaderBytes + need) : s.pool().acquire();
            s.headSize   = s.head.size();
            s.headOffset = kChunkHeaderBytes; // reserve the intrusive node header at the chunk head
            if (s.head.empty()) {
                return {}; // genuine pool-cap exhaustion (capped pools only) — caller backpressures; NOT a size-class drop
            }
        }
        std::byte* dst = s.head.data() + s.headOffset;
        std::memcpy(dst, blob.data(), need);
        s.headOffset += need;
        s.headLastWritten = descAbsPos; // head now carries an entry at descAbsPos ⇒ pin it until min_reader passes (closes the write-before-publish window)
        return {dst, need};
    }

    // allocate the shared State (control block + State) from the descriptor allocator's resource — so even the
    // wrapper lands on the injected arena, not global new; the MCU/arena-as-default construction path.
    template<typename... StateArgs>
    [[nodiscard]] static std::shared_ptr<State> makeState(std::pmr::memory_resource* res, StateArgs&&... args) {
        return std::allocate_shared<State>(std::pmr::polymorphic_allocator<State>(res), std::forward<StateArgs>(args)...);
    }

public:
    explicit ChunkBuffer(std::size_t descriptorCapacity, std::size_t keepResidentChunks = 0UZ, std::size_t chunkBytes = kDefaultChunkBytes) //
        : _state(makeState(DescAllocator{}.resource(), descriptorCapacity, DescAllocator{}, keepResidentChunks, chunkBytes)) {}

    // drop-in parity with the (minSize, polymorphic_allocator) construction site in Port; the allocator backs the
    // descriptor ring slots AND the owned blob pool's upstream — so blobs draw from the injected resource.
    ChunkBuffer(std::size_t descriptorCapacity, DescAllocator alloc, std::size_t keepResidentChunks = 0UZ, std::size_t chunkBytes = kDefaultChunkBytes) //
        : _state(makeState(alloc.resource(), descriptorCapacity, alloc, keepResidentChunks, chunkBytes)) {}

    // explicit shared pool — one ChunkPool multiplexed across many edges (graph-global / per-domain).
    ChunkBuffer(std::size_t descriptorCapacity, ChunkPool& sharedPool, std::size_t keepResidentChunks = 0UZ) //
        : _state(makeState(DescAllocator{}.resource(), descriptorCapacity, DescAllocator{}, sharedPool, keepResidentChunks)) {}

    ChunkBuffer(std::size_t descriptorCapacity, DescAllocator alloc, ChunkPool& sharedPool, std::size_t keepResidentChunks = 0UZ) //
        : _state(makeState(alloc.resource(), descriptorCapacity, alloc, sharedPool, keepResidentChunks)) {}

    ChunkBuffer(const ChunkBuffer&)                = default; // shares state (the connect round-trip relies on this)
    ChunkBuffer(ChunkBuffer&&) noexcept            = default;
    ChunkBuffer& operator=(const ChunkBuffer&)     = default;
    ChunkBuffer& operator=(ChunkBuffer&&) noexcept = default;
    ~ChunkBuffer()                                 = default;

    template<SpanReleasePolicy policy>
    class WriterSpan {
        using InnerSpan = decltype(std::declval<InnerWriterType&>().template reserve<policy>(0UZ));
        InnerSpan   _descSpan;
        State*      _state;
        std::size_t _base; // absolute descriptor position at reserve time

    public:
        using value_type = T;

        WriterSpan(InnerSpan descSpan, State* state, std::size_t base) noexcept : _descSpan(std::move(descSpan)), _state(state), _base(base) {}

        [[nodiscard]] std::size_t    size() const noexcept { return _descSpan.size(); }
        [[nodiscard]] decltype(auto) operator[](std::size_t i) noexcept { return _descSpan[i]; }
        constexpr void               publish(std::size_t nItems) noexcept { _descSpan.publish(nItems); }

        // contiguous-range / SpanLike surface over the descriptor slots, so this satisfies WriterSpanLike.
        [[nodiscard]] std::span<T> asSpan() noexcept { return std::span<T>(_descSpan); }
        [[nodiscard]] auto         begin() noexcept { return asSpan().begin(); }
        [[nodiscard]] auto         end() noexcept { return asSpan().end(); }
        [[nodiscard]] T*           data() noexcept { return asSpan().data(); }
                                   operator std::span<T>() noexcept { return asSpan(); }

        [[nodiscard]] std::span<std::byte> storeBlob(std::size_t i, std::span<const std::byte> blob) noexcept { return serialiseBlob(*_state, blob, _base + i); }
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
        [[nodiscard]] WriterSpan<policy> reserve(std::size_t nItems) noexcept {
            return WriterSpan<policy>(_w.template reserve<policy>(nItems), _state.get(), _w.position());
        }
        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] WriterSpan<policy> tryReserve(std::size_t nItems) noexcept {
            return WriterSpan<policy>(_w.template tryReserve<policy>(nItems), _state.get(), _w.position());
        }

        [[nodiscard]] std::size_t                position() const noexcept { return _w.position(); }
        [[nodiscard]] std::size_t                available() const noexcept { return _w.available(); }
        [[nodiscard]] std::size_t                nRequestedSamplesToPublish() const noexcept { return _w.nRequestedSamplesToPublish(); }
        [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _w.resource(); }
        [[nodiscard]] ChunkBuffer                buffer() const noexcept { return ChunkBuffer(_state); }
    };

    class Reader {
        InnerReaderType        _r;
        std::shared_ptr<State> _state;

    public:
        Reader(InnerReaderType r, std::shared_ptr<State> state) noexcept : _r(std::move(r)), _state(std::move(state)) {}

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] auto get(std::size_t nItems = std::numeric_limits<std::size_t>::max()) {
            return _r.template get<policy>(nItems); // descriptor span: random-access, consume/tryConsume
        }
        [[nodiscard]] std::size_t position() const noexcept { return _r.position(); }
        [[nodiscard]] std::size_t available() const noexcept { return _r.available(); }
        [[nodiscard]] std::size_t nSamplesConsumed() const noexcept { return _r.nSamplesConsumed(); }
        [[nodiscard]] bool        isConsumeRequested() const noexcept { return _r.isConsumeRequested(); }
        [[nodiscard]] ChunkBuffer buffer() const noexcept { return ChunkBuffer(_state); }
    };

    [[nodiscard]] Writer      new_writer() { return Writer(_state->descriptors.new_writer(), _state); }
    [[nodiscard]] Reader      new_reader() { return Reader(_state->descriptors.new_reader(), _state); }
    [[nodiscard]] std::size_t size() const noexcept { return _state->descriptors.size(); }
    [[nodiscard]] std::size_t n_writers() const { return _state->descriptors.n_writers(); }
    [[nodiscard]] std::size_t n_readers() const { return _state->descriptors.n_readers(); }
    [[nodiscard]] ChunkPool&  pool() const noexcept { return _state->pool(); }

    // reclaim driven by the scheduler housekeeping pass.
    void houseKeeping(HouseKeepDepth depth) noexcept {
        State& s = *_state;
        if (s._guard.test_and_set(std::memory_order_acquire)) {
            return; // producer mid-serialiseBlob ⇒ defer reclaim to the next pass (best-effort, low-priority)
        }
        StateGuard        guard(s._guard);                              // RAII release
        const std::size_t minPos = s.descriptors.min_reader_position(); // gating cursor: advances on ReaderSpan release
        while (s.sealedHead != nullptr) {
            const SealedHeader h = readHeader(s.sealedHead);
            if (minPos < h.releaseThreshold) {
                break; // oldest sealed chunk still pinned by a reader
            }
            std::byte* chunk = s.sealedHead;
            s.sealedHead     = h.next;
            if (s.sealedHead == nullptr) {
                s.sealedTail = nullptr;
            }
            s.releaseChunk(std::span<std::byte>{chunk, h.chunkSize}); // h.chunkSize: standard → free-list, oversized → upstream
        }
        if (s.sealedHead == nullptr && !s.head.empty() && minPos > s.headLastWritten) { // 0-chunk floor: head fully consumed (all blob-backed descriptors drained)
            s.releaseChunk(s.head);
            s.head       = {};
            s.headOffset = 0UZ;
            s.headSize   = 0UZ;
        }
        if (depth == HouseKeepDepth::Deep) {
            s.pool().reclaimToUpstream(s.keepResident);
        }
    }
};

static_assert(BufferLike<ChunkBuffer<std::uint32_t, ProducerType::Single>>);

} // namespace gr

#endif // GNURADIO_CHUNKBUFFER_HPP
