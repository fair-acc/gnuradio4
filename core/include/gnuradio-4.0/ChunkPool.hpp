#ifndef GNURADIO_CHUNKPOOL_HPP
#define GNURADIO_CHUNKPOOL_HPP

#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <mutex>
#include <span>

namespace gr {

inline constexpr std::size_t kDefaultChunkBytes = 4096UZ;
inline constexpr std::size_t kChunkAlignment    = 64UZ; // ≥ cache line; raise to device granularity for USM

/**
 * @brief Graph-global free-list of fixed-size byte chunks over a shared upstream resource.
 *
 * The unit of multiplexing — and the future USM-mapping unit — for the chunked tag transport. Per-edge
 * tag buffers draw blob chunks here and return them when drained, so resident RAM tracks the
 * *peak-concurrent* working set across all edges instead of Σ(per-edge worst case). `reclaimToUpstream`
 * hands idle chunks back to the upstream resource so the RAM becomes available to other subsystems —
 * this is the return-policy knob, driven by housekeeping depth (Shallow keeps chunks resident for fast
 * re-acquire; Deep returns them).
 *
 * The free-list is intrusive (each free chunk stores the next-free pointer in its own first bytes), so
 * acquire/release touch no heap. Prototype concurrency is a single mutex (acquire/release are
 * per-chunk, not per-tag); production wants a lock-free Treiber stack or per-compute-domain shards.
 */
class ChunkPool {
    std::pmr::memory_resource* _upstream;
    std::size_t                _chunkBytes;
    std::size_t                _alignment;
    std::size_t                _maxChunks; // hard cap → exception-free backpressure (no caught bad_alloc)
    mutable std::mutex         _mutex;
    std::byte*                 _freeHead{nullptr};   // intrusive free-list head
    std::size_t                _freeCount{0UZ};      // chunks on the free-list
    std::size_t                _residentChunks{0UZ}; // allocated from upstream (in-use + free)

    [[nodiscard]] static std::byte* nextOf(std::byte* chunk) noexcept {
        std::byte* next = nullptr;
        std::memcpy(&next, chunk, sizeof(next));
        return next;
    }
    static void setNext(std::byte* chunk, std::byte* next) noexcept { std::memcpy(chunk, &next, sizeof(next)); }

public:
    explicit ChunkPool(std::pmr::memory_resource* upstream, std::size_t chunkBytes = kDefaultChunkBytes, std::size_t alignment = kChunkAlignment, std::size_t maxChunks = std::numeric_limits<std::size_t>::max()) noexcept : _upstream(upstream ? upstream : std::pmr::get_default_resource()), _chunkBytes(chunkBytes), _alignment(alignment), _maxChunks(maxChunks) { assert(_chunkBytes >= sizeof(std::byte*) && "chunk must hold an intrusive next-free pointer"); }

    ChunkPool(const ChunkPool&)            = delete;
    ChunkPool& operator=(const ChunkPool&) = delete;
    ChunkPool(ChunkPool&&)                 = delete;
    ChunkPool& operator=(ChunkPool&&)      = delete;

    // only free-list chunks are owned here; chunks still in use must be release()d before destruction.
    ~ChunkPool() {
        while (_freeHead != nullptr) {
            std::byte* p = _freeHead;
            _freeHead    = nextOf(p);
            _upstream->deallocate(p, _chunkBytes, _alignment);
        }
    }

    // O(1) free-list pop; on miss grows from upstream up to maxChunks. Empty span ⇒ cap hit ⇒ caller backpressures.
    [[nodiscard]] std::span<std::byte> acquire() noexcept {
        std::lock_guard lock(_mutex);
        if (_freeHead != nullptr) {
            std::byte* p = _freeHead;
            _freeHead    = nextOf(p);
            --_freeCount;
            return {p, _chunkBytes};
        }
        if (_residentChunks >= _maxChunks) {
            return {}; // exception-free backpressure
        }
        std::byte* p = static_cast<std::byte*>(_upstream->allocate(_chunkBytes, _alignment));
        ++_residentChunks;
        return {p, _chunkBytes};
    }

    void release(std::span<std::byte> chunk) noexcept {
        if (chunk.empty()) {
            return;
        }
        std::lock_guard lock(_mutex);
        setNext(chunk.data(), _freeHead);
        _freeHead = chunk.data();
        ++_freeCount;
    }

    // return-policy knob: hand free-list chunks beyond keepResident back to upstream (RAM goes elsewhere).
    void reclaimToUpstream(std::size_t keepResident = 0UZ) noexcept {
        std::lock_guard lock(_mutex);
        while (_freeCount > keepResident) {
            std::byte* p = _freeHead;
            _freeHead    = nextOf(p);
            --_freeCount;
            --_residentChunks;
            _upstream->deallocate(p, _chunkBytes, _alignment);
        }
    }

    [[nodiscard]] std::size_t chunkBytes() const noexcept { return _chunkBytes; }
    [[nodiscard]] std::size_t residentChunks() const noexcept {
        std::lock_guard lock(_mutex);
        return _residentChunks;
    }
    [[nodiscard]] std::size_t freeChunks() const noexcept {
        std::lock_guard lock(_mutex);
        return _freeCount;
    }
};

} // namespace gr

#endif // GNURADIO_CHUNKPOOL_HPP
