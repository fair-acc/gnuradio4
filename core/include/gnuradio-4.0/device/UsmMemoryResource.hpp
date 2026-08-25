#ifndef GNURADIO_USM_MEMORY_RESOURCE_HPP
#define GNURADIO_USM_MEMORY_RESOURCE_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <memory_resource>
#include <new>

#include <gnuradio-4.0/ComputeDomain.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>

namespace gr::device {

/**
 * @brief PMR memory resource backed by SYCL Unified Shared Memory; aligned `operator new` without SYCL.
 *
 * `shared` for a buffer a kernel writes, `hostPinned` for one that crosses back to the host.
 * `registerUsmProvider()` makes device edges allocate through it.
 */
enum class UsmKind : std::uint8_t { shared, hostPinned, deviceOnly };

class UsmMemoryResource : public gr::MemoryResource {
#if GR_DEVICE_HAS_SYCL
    sycl::queue* _queue = nullptr;
    UsmKind      _kind  = UsmKind::shared; // only ever read by the allocating paths, which are themselves SYCL-only
#endif

public:
    UsmMemoryResource() = default;

#if GR_DEVICE_HAS_SYCL
    explicit UsmMemoryResource(sycl::queue& q, UsmKind kind = UsmKind::shared) : _queue(&q), _kind(kind) {}

    // USM pointers are bound to their queue's context
    [[nodiscard]] sycl::queue* queue() const noexcept { return _queue; }
#endif

    [[nodiscard]] MemoryResourceCapabilities capabilities() const noexcept override {
#if GR_DEVICE_HAS_SYCL
        if (_kind != UsmKind::deviceOnly) {
            return {}; // shared and host USM are addressable from the host, so they copy as ordinary memory
        }
        // the copy is enqueued, not awaited: the queue is in-order, only device work reads this memory, and a
        // teardown free drains the queue first
        return {.deviceOnly = true, //
            .copyWithin     = [](void* destination, const void* source, std::size_t bytes, void* context) { std::ignore = static_cast<sycl::queue*>(context)->memcpy(destination, source, bytes); },
            .copyContext    = _queue};
#else
        return {};
#endif
    }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes == 0) {
            bytes = 1;
        }
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL
            if (_queue) {
                void* p = _kind == UsmKind::hostPinned ? sycl::aligned_alloc_host(alignment, bytes, *_queue) : _kind == UsmKind::deviceOnly ? sycl::aligned_alloc_device(alignment, bytes, *_queue) : sycl::aligned_alloc_shared(alignment, bytes, *_queue);
                if (p) {
                    return p;
                }
                throw std::bad_alloc();
            }
#endif
        }
        return ::operator new(bytes, std::align_val_t{alignment});
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL
            if (_queue) {
                if (_kind == UsmKind::deviceOnly) {
                    _queue->wait(); // a kernel still reading this allocation would fault: nothing else orders the free
                }
                sycl::free(p, *_queue);
                return;
            }
#endif
        }
        ::operator delete(p, std::align_val_t{alignment});
    }

    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        const auto* o = dynamic_cast<const UsmMemoryResource*>(&other);
        if (!o) {
            return false;
        }
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL
            return _queue == o->_queue;
#endif
        }
        return true;
    }
};

#if GR_DEVICE_HAS_SYCL
// pinned host memory for a boundary edge: filled by one bulk copy, then read by the host. One per queue.
inline UsmMemoryResource& pinnedHostResourceFor(sycl::queue& queue) {
    static auto& byQueue   = *new std::map<sycl::queue*, std::unique_ptr<UsmMemoryResource>>(); // never destroyed, as `syclQueues()`: holds USM bound to those queues
    auto [entry, inserted] = byQueue.try_emplace(&queue);
    if (inserted) {
        entry->second = std::make_unique<UsmMemoryResource>(queue, UsmKind::hostPinned);
    }
    return *entry->second;
}

// device-only memory for an edge interior to one device: the host never touches it, so the ring mirrors its own
// wrap through `copyWithin` rather than needing the memory to be double-mapped. One per queue.
inline UsmMemoryResource& deviceOnlyResourceFor(sycl::queue& queue) {
    static auto& byQueue   = *new std::map<sycl::queue*, std::unique_ptr<UsmMemoryResource>>(); // never destroyed, as `syclQueues()`: holds USM bound to those queues
    auto [entry, inserted] = byQueue.try_emplace(&queue);
    if (inserted) {
        entry->second = std::make_unique<UsmMemoryResource>(queue, UsmKind::deviceOnly);
    }
    return *entry->second;
}

#endif

namespace detail {

inline UsmMemoryResource& defaultUsmResource() {
    static UsmMemoryResource instance;
    return instance;
}

inline std::pmr::memory_resource* usmProvider(const ComputeDomain& /*dom*/, void* ctx) {
    if (ctx) {
        return static_cast<std::pmr::memory_resource*>(ctx);
    }
    return &defaultUsmResource();
}

} // namespace detail

inline void registerUsmProvider() { ComputeRegistry::instance().registerProvider("sycl", &detail::usmProvider); }

} // namespace gr::device

#endif // GNURADIO_USM_MEMORY_RESOURCE_HPP
