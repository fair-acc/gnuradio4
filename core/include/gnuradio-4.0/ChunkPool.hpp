#ifndef GNURADIO_CHUNKPOOL_HPP
#define GNURADIO_CHUNKPOOL_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory_resource>
#include <mutex>
#include <span>
#include <vector>

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
 * The free-list is held beside the chunks, never in them, so the chunks may be memory the host cannot
 * touch (device-only USM); it is sized as each chunk is allocated, so acquire/release touch no heap.
 * Concurrency is one mutex: acquire/release are per chunk, not per tag.
 */
class ChunkPool {
    std::pmr::memory_resource*             _upstream;
    std::size_t                            _chunkBytes;
    std::size_t                            _alignment;
    std::size_t                            _maxChunks;
    mutable std::mutex                     _mutex;
    std::pmr::vector<std::byte*>           _freeStandard; /**< capacity >= resident standard chunks, so release() never allocates */
    std::pmr::vector<std::span<std::byte>> _freeOversized;
    std::size_t                            _residentChunks{0UZ}; /**< allocated from upstream (in-use + free), oversized included */
    std::size_t                            _residentOversized{0UZ};

    /** geometric: an exact reserve per new chunk would copy the whole list each time */
    static void reserveFor(auto& list, std::size_t nChunks) {
        if (list.capacity() < nChunks) {
            list.reserve(std::max(nChunks, 2UZ * list.capacity()));
        }
    }

public:
    /** `bookkeeping` holds the free-list; it must be host-addressable, `upstream` need not be */
    explicit ChunkPool(std::pmr::memory_resource* upstream, std::size_t chunkBytes = kDefaultChunkBytes, std::size_t alignment = kChunkAlignment, std::size_t maxChunks = std::numeric_limits<std::size_t>::max(), std::pmr::memory_resource* bookkeeping = std::pmr::get_default_resource()) noexcept : _upstream(upstream ? upstream : std::pmr::get_default_resource()), _chunkBytes(chunkBytes), _alignment(alignment), _maxChunks(maxChunks), _freeStandard(bookkeeping), _freeOversized(bookkeeping) {}

    ChunkPool(const ChunkPool&)            = delete;
    ChunkPool& operator=(const ChunkPool&) = delete;
    ChunkPool(ChunkPool&&)                 = delete;
    ChunkPool& operator=(ChunkPool&&)      = delete;

    // only free-list chunks are owned here; chunks still in use must be release()d before destruction.
    ~ChunkPool() {
        for (std::byte* p : _freeStandard) {
            _upstream->deallocate(p, _chunkBytes, _alignment);
        }
        for (std::span<std::byte> chunk : _freeOversized) {
            _upstream->deallocate(chunk.data(), chunk.size(), _alignment);
        }
    }

    // O(1) free-list pop; on miss grows from upstream up to maxChunks. Empty span ⇒ cap hit ⇒ caller backpressures.
    [[nodiscard]] std::span<std::byte> acquire() noexcept {
        std::lock_guard lock(_mutex);
        if (!_freeStandard.empty()) {
            std::byte* p = _freeStandard.back();
            _freeStandard.pop_back();
            return {p, _chunkBytes};
        }
        if (_residentChunks >= _maxChunks) {
            return {}; // exception-free backpressure
        }
        reserveFor(_freeStandard, _residentChunks + 1UZ);
        std::byte* p = static_cast<std::byte*>(_upstream->allocate(_chunkBytes, _alignment));
        ++_residentChunks;
        return {p, _chunkBytes};
    }

    void release(std::span<std::byte> chunk) noexcept {
        if (chunk.empty()) {
            return;
        }
        std::lock_guard lock(_mutex);
        _freeStandard.push_back(chunk.data());
    }

    /** a blob larger than a standard chunk (a jumbo tag, a large tensor): the most recently released oversized chunk that holds it, else one from upstream at the requested size; the returned span may be larger than `bytes`; honours the same hard cap */
    [[nodiscard]] std::span<std::byte> acquireOversized(std::size_t bytes) noexcept {
        std::lock_guard lock(_mutex);
        const auto      fits = [bytes](std::span<std::byte> chunk) { return chunk.size() >= bytes; };
        if (auto latest = std::find_if(_freeOversized.rbegin(), _freeOversized.rend(), fits); latest != _freeOversized.rend()) {
            const std::span<std::byte> chunk = *latest;
            *latest                          = _freeOversized.back();
            _freeOversized.pop_back();
            return chunk;
        }
        if (_residentChunks >= _maxChunks) {
            return {}; // exception-free backpressure (same cap as standard chunks)
        }
        reserveFor(_freeOversized, ++_residentOversized);
        std::byte* p = static_cast<std::byte*>(_upstream->allocate(bytes, _alignment));
        ++_residentChunks;
        return {p, bytes};
    }

    void releaseOversized(std::span<std::byte> chunk) noexcept {
        if (chunk.empty()) {
            return;
        }
        std::lock_guard lock(_mutex);
        _freeOversized.push_back(chunk);
    }

    // return-policy knob: hand free-list chunks beyond keepResident back to upstream (RAM goes elsewhere).
    void reclaimToUpstream(std::size_t keepResident = 0UZ) noexcept {
        std::lock_guard lock(_mutex);
        while (_freeStandard.size() > keepResident) {
            --_residentChunks;
            _upstream->deallocate(_freeStandard.back(), _chunkBytes, _alignment);
            _freeStandard.pop_back();
        }
        for (std::span<std::byte> chunk : _freeOversized) {
            --_residentChunks;
            --_residentOversized;
            _upstream->deallocate(chunk.data(), chunk.size(), _alignment);
        }
        _freeOversized.clear();
    }

    [[nodiscard]] std::size_t chunkBytes() const noexcept { return _chunkBytes; }
    [[nodiscard]] std::size_t residentChunks() const noexcept {
        std::lock_guard lock(_mutex);
        return _residentChunks;
    }
    [[nodiscard]] std::size_t freeChunks() const noexcept {
        std::lock_guard lock(_mutex);
        return _freeStandard.size();
    }
};

} // namespace gr

#endif // GNURADIO_CHUNKPOOL_HPP
