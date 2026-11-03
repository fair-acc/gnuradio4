#ifndef GNURADIO_CHUNKBUFFER_HPP
#define GNURADIO_CHUNKBUFFER_HPP

#include <atomic>
#include <bit>
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
#include <vector>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/ChunkPool.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr {

/** a device-only ring's payload slot per descriptor */
inline constexpr std::size_t kDeviceEventSlotBytes = 1024UZ;

/**
 * Chunked descriptor+blob buffer (BufferLike): a fixed-stride `T` descriptor ring plus variable-size blobs in a `ChunkPool`, by default owned and drawn from the descriptor allocator's resource (an arena on MCU), or an explicit shared `ChunkPool&`.
 * Reclaim rides `houseKeeping(depth)`: `Shallow` returns drained chunks to the pool free-list (steady-state publish allocates nothing), `Deep` also hands the free-list upstream.
 * Over device-only memory (`isDeviceOnly`) only the blobs live there; the state and descriptor ring take the resource's host-addressable sibling.
 * Several producers may share one buffer: a blob is filed against the position its writer claimed, never the publish cursor, which lags out-of-order claims.
 * The state sits behind a `shared_ptr`, so a copy and the `buffer()` round-trip of `Graph::connect` share one buffer.
 */
template<gr::meta::trivially_copyable T, ProducerType producerType = ProducerType::Single>
struct ChunkBuffer {
    using value_type      = T;
    using DescriptorRing  = CircularBuffer<T, std::dynamic_extent, producerType>;
    using DescAllocator   = std::pmr::polymorphic_allocator<T>;
    using InnerWriterType = decltype(std::declval<DescriptorRing&>().new_writer());
    using InnerReaderType = decltype(std::declval<DescriptorRing&>().new_reader());

private:
    struct SealedChunk {
        std::span<std::byte> chunk;
        std::size_t          releaseThreshold; /**< chunk is dead once descriptors.min_reader_position() >= this */
    };

    [[nodiscard]] static std::pmr::memory_resource* hostAddressable(std::pmr::memory_resource* resource) noexcept {
        const MemoryResourceCapabilities capabilities = memoryResourceCapabilities(resource);
        if (!capabilities.deviceOnly) {
            return resource;
        }
        return capabilities.hostAddressable != nullptr ? capabilities.hostAddressable : std::pmr::get_default_resource();
    }

    static constexpr bool kIsEventRing = requires(const T& event) { event.blob(); };

    struct State {
        DescriptorRing                descriptors;
        std::optional<ChunkPool>      ownedPool;
        ChunkPool*                    externalPool = nullptr;
        bool                          blobsHostAddressable;
        std::size_t                   keepResident; /**< free-list floor retained on a Deep reclaim */
        std::size_t                   chunkBytes;
        std::span<std::byte>          head{};
        std::size_t                   headOffset{0UZ};
        std::pmr::vector<SealedChunk> sealed;
        std::size_t                   sealedFront{0UZ};
        std::size_t                   headLastWritten{0UZ}; /**< highest descriptor position written into `head`; head is reclaimable once min_reader passes it */
        std::atomic_flag              _guard{};             /**< serialises producer reserveBytes() against houseKeeping() on the chunk bookkeeping */
        std::pmr::memory_resource*    slotResource{nullptr};
        std::span<std::byte>          slots{}; /**< device-only ring: descriptor position p keeps its payload at slot p % capacity, written by a kernel only */

        State(std::size_t cap, std::pmr::memory_resource* blobs, std::pmr::memory_resource* host, std::size_t keep, std::size_t chunk) //
            : descriptors(cap, DescAllocator(host)), blobsHostAddressable(host == blobs), keepResident(keep), chunkBytes(chunk), sealed(host) {
            ownedPool.emplace(blobs, chunk, kChunkAlignment, std::numeric_limits<std::size_t>::max(), host);
            if constexpr (kIsEventRing) {
                if (!blobsHostAddressable) {
                    slotResource = blobs;
                    slots        = {static_cast<std::byte*>(slotResource->allocate(descriptors.size() * kDeviceEventSlotBytes, kChunkAlignment)), descriptors.size() * kDeviceEventSlotBytes};
                }
            }
        }
        State(std::size_t cap, std::pmr::memory_resource* blobs, std::pmr::memory_resource* host, ChunkPool& ext, std::size_t keep) //
            : descriptors(cap, DescAllocator(host)), externalPool(&ext), blobsHostAddressable(host == blobs), keepResident(keep), chunkBytes(ext.chunkBytes()), sealed(host) {}

        ~State() {
            for (std::size_t i = sealedFront; i < sealed.size(); ++i) {
                releaseChunk(sealed[i].chunk);
            }
            if (!head.empty()) {
                releaseChunk(head);
            }
            if (!slots.empty()) {
                slotResource->deallocate(slots.data(), slots.size(), kChunkAlignment);
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

    /** releases State::_guard, acquired at the call site: the producer (reserveBytes) spin-acquires, houseKeeping try-acquires and defers on contention */
    struct StateGuard {
        std::atomic_flag& _flag;
        explicit StateGuard(std::atomic_flag& flag) noexcept : _flag(flag) {}
        StateGuard(const StateGuard&)            = delete;
        StateGuard& operator=(const StateGuard&) = delete;
        ~StateGuard() { _flag.clear(std::memory_order_release); }
    };

    /** reserves `bytes` of uninitialised storage in the head chunk, its address a multiple of `alignment` (<= the pool's chunk alignment); `descAbsPos` is the absolute position of the descriptor that will own it, for the seal threshold */
    [[nodiscard]] static std::span<std::byte> reserveBytes(State& s, std::size_t bytes, std::size_t alignment, std::size_t descAbsPos) noexcept {
        while (s._guard.test_and_set(std::memory_order_acquire)) { /* brief: wait out a best-effort houseKeeping pass */
        }
        StateGuard guard(s._guard);
        assert(alignment <= kChunkAlignment && std::has_single_bit(alignment) && "a power of two, no larger than a fresh chunk's alignment");
        const auto alignedOffset = [&s, alignment](std::size_t offset) {
            const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(s.head.data()) + offset;
            return offset + ((alignment - (address & (alignment - 1UZ))) & (alignment - 1UZ));
        };
        if (s.head.empty() || alignedOffset(s.headOffset) + bytes > s.head.size()) {
            if (!s.head.empty()) {
                s.sealed.push_back(SealedChunk{s.head, s.headLastWritten + 1UZ});
            }
            s.head       = bytes > s.chunkBytes ? s.pool().acquireOversized(bytes) : s.pool().acquire();
            s.headOffset = 0UZ;
            if (s.head.empty()) {
                return {}; // genuine pool-cap exhaustion (capped pools only) — caller backpressures; NOT a size-class drop
            }
        }
        const std::size_t at = alignedOffset(s.headOffset);
        s.headOffset         = at + bytes;
        s.headLastWritten    = std::max(s.headLastWritten, descAbsPos);
        return {s.head.data() + at, bytes};
    }

    /** allocates the shared State from the injected resource (or its host-addressable sibling), so even the wrapper lands on the arena, not global new */
    template<typename... StateArgs>
    [[nodiscard]] static std::shared_ptr<State> makeState(std::pmr::memory_resource* res, std::size_t descriptorCapacity, StateArgs&&... args) {
        std::pmr::memory_resource* const host = hostAddressable(res);
        return std::allocate_shared<State>(std::pmr::polymorphic_allocator<State>(host), descriptorCapacity, res, host, std::forward<StateArgs>(args)...);
    }

public:
    explicit ChunkBuffer(std::size_t descriptorCapacity, std::size_t keepResidentChunks = 0UZ, std::size_t chunkBytes = kDefaultChunkBytes) //
        : _state(makeState(DescAllocator{}.resource(), descriptorCapacity, keepResidentChunks, chunkBytes)) {}

    // drop-in parity with the (minSize, polymorphic_allocator) construction site in Port; the allocator backs the
    // descriptor ring slots AND the owned blob pool's upstream — so blobs draw from the injected resource.
    ChunkBuffer(std::size_t descriptorCapacity, DescAllocator alloc, std::size_t keepResidentChunks = 0UZ, std::size_t chunkBytes = kDefaultChunkBytes) //
        : _state(makeState(alloc.resource(), descriptorCapacity, keepResidentChunks, chunkBytes)) {}

    // explicit shared pool — one ChunkPool multiplexed across many edges (graph-global / per-domain).
    ChunkBuffer(std::size_t descriptorCapacity, ChunkPool& sharedPool, std::size_t keepResidentChunks = 0UZ) //
        : _state(makeState(DescAllocator{}.resource(), descriptorCapacity, sharedPool, keepResidentChunks)) {}

    ChunkBuffer(std::size_t descriptorCapacity, DescAllocator alloc, ChunkPool& sharedPool, std::size_t keepResidentChunks = 0UZ) //
        : _state(makeState(alloc.resource(), descriptorCapacity, sharedPool, keepResidentChunks)) {}

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

        [[nodiscard]] std::span<std::byte> reserveBlob(std::size_t i, std::size_t bytes, std::size_t alignment) noexcept { return reserveBytes(*_state, bytes, alignment, _base + i); }
        [[nodiscard]] bool                 blobsHostAddressable() const noexcept { return _state->blobsHostAddressable; }
        [[nodiscard]] std::span<std::byte> storeBlob(std::size_t i, std::span<const std::byte> blob, std::size_t alignment) noexcept {
            if (!_state->blobsHostAddressable) {
                return {};
            }
            const std::span<std::byte> stored = reserveBlob(i, blob.size(), alignment);
            if (stored.size() == blob.size()) {
                std::memcpy(stored.data(), blob.data(), blob.size());
            }
            return stored;
        }

        /** a device-only ring's payload slots, for a kernel to store into in place: entry i of this span owns slot (first + i) % count */
        struct SlotArea {
            std::byte*  slots     = nullptr;
            std::size_t slotBytes = 0UZ;
            std::size_t first     = 0UZ;
            std::size_t count     = 0UZ;
        };
        [[nodiscard]] SlotArea slotArea() const noexcept {
            const std::size_t capacity = _state->descriptors.size();
            return _state->slots.empty() ? SlotArea{} : SlotArea{_state->slots.data(), kDeviceEventSlotBytes, _base % capacity, capacity};
        }

        [[nodiscard]] constexpr static SpanReleasePolicy spanReleasePolicy() noexcept { return policy; }
        [[nodiscard]] constexpr static bool              isMultiProducerStrategy() noexcept { return InnerSpan::isMultiProducerStrategy(); }
        [[nodiscard]] std::size_t                        claimedPosition() const noexcept { return _base; }
        [[nodiscard]] std::size_t                        nRequestedSamplesToPublish() const noexcept { return _descSpan.nRequestedSamplesToPublish(); }
        [[nodiscard]] bool                               isPublishRequested() const noexcept { return _descSpan.isPublishRequested(); }
        [[nodiscard]] bool                               isFullyPublished() const noexcept { return _descSpan.isFullyPublished(); }
        [[nodiscard]] std::size_t                        instanceCount() const noexcept { return _descSpan.instanceCount(); }
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
            auto              inner   = _w.template reserve<policy>(nItems);
            const std::size_t claimed = inner.claimedPosition();
            return WriterSpan<policy>(std::move(inner), _state.get(), claimed);
        }
        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] WriterSpan<policy> tryReserve(std::size_t nItems) noexcept {
            auto              inner   = _w.template tryReserve<policy>(nItems);
            const std::size_t claimed = inner.claimedPosition();
            return WriterSpan<policy>(std::move(inner), _state.get(), claimed);
        }

        [[nodiscard]] std::size_t                position() const noexcept { return _w.position(); }
        [[nodiscard]] std::size_t                available() const noexcept { return _w.available(); }
        [[nodiscard]] std::size_t                nRequestedSamplesToPublish() const noexcept { return _w.nRequestedSamplesToPublish(); }
        [[nodiscard]] bool                       isPublishRequested() const noexcept { return _w.isPublishRequested(); }
        [[nodiscard]] std::size_t                nWriters() const noexcept { return _w.nWriters(); }
        [[nodiscard]] std::size_t                nReaders() const noexcept { return _w.nReaders(); }
        [[nodiscard]] std::size_t                bufferCapacity() const noexcept { return _w.bufferCapacity(); }
        [[nodiscard]] const void*                bufferIdentity() const noexcept { return static_cast<const void*>(_state.get()); }
        [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _w.resource(); }
        [[nodiscard]] bool                       blobsHostAddressable() const noexcept { return _state->blobsHostAddressable; }
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
        [[nodiscard]] std::size_t nWriters() const noexcept { return _r.nWriters(); }
        [[nodiscard]] std::size_t nReaders() const noexcept { return _r.nReaders(); }
        [[nodiscard]] std::size_t bufferCapacity() const noexcept { return _r.bufferCapacity(); }
        [[nodiscard]] const void* bufferIdentity() const noexcept { return static_cast<const void*>(_state.get()); }
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
            return;
        }
        StateGuard        guard(s._guard);
        const std::size_t minPos = s.descriptors.min_reader_position();
        while (s.sealedFront < s.sealed.size() && minPos >= s.sealed[s.sealedFront].releaseThreshold) {
            s.releaseChunk(s.sealed[s.sealedFront].chunk);
            ++s.sealedFront;
        }
        if (s.sealedFront == s.sealed.size()) {
            s.sealed.clear();
            s.sealedFront = 0UZ;
        } else if (2UZ * s.sealedFront > s.sealed.size()) {
            s.sealed.erase(s.sealed.begin(), s.sealed.begin() + static_cast<std::ptrdiff_t>(s.sealedFront));
            s.sealedFront = 0UZ;
        }
        if (s.sealed.empty() && !s.head.empty() && minPos > s.headLastWritten) {
            s.releaseChunk(s.head);
            s.head       = {};
            s.headOffset = 0UZ;
        }
        if (depth == HouseKeepDepth::Deep) {
            s.pool().reclaimToUpstream(s.keepResident);
        }
    }
};

static_assert(BufferLike<ChunkBuffer<std::uint32_t, ProducerType::Single>>);

} // namespace gr

#endif // GNURADIO_CHUNKBUFFER_HPP
